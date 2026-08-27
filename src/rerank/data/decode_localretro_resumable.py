#!/usr/bin/env python
"""Resumable, output-equivalent LocalRetro Top-50 decoder.

The pinned upstream decoder accidentally calls ``decode_localtemplate`` twice
for every edit proposal and retains all results only in RAM until every product
has finished.  This compute-only implementation prepares each product once,
calls the same pinned decoder once per proposal, caches compiled reaction
SMARTS inside each worker, and commits completed products to SQLite.  Candidate
ordering, duplicate handling, scores, and the Top-50 stopping rule are kept
identical to upstream.
"""

from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from rdkit import Chem

from rerank.data.prepare_localretro_current_dataset import atomic_json, fingerprint, sha256_file
from rerank.experiments.run_localretro_revision import (
    DATASET_NAME,
    LOCALRETRO_COMMIT,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    compile_decoded_predictions,
    validate_workspace,
)


DECODER_PROTOCOL_ID = "localretro-decode-resumable-v1"
DEFAULT_WORKERS = 8
TOP_K = 50

_RAW: dict[int, tuple[str, list[str]]] = {}
_ATOM_TEMPLATES: dict[int, str] = {}
_BOND_TEMPLATES: dict[int, str] = {}
_TEMPLATE_INFOS: dict[str, dict[str, Any]] = {}
_TD: Any = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _install_workspace_imports(workspace: Path) -> None:
    for candidate in (workspace / "scripts", workspace):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)


def _worker_init() -> None:
    global _TD
    from LocalTemplate import template_decoder as template_decoder

    original = template_decoder.rdChemReactions.ReactionFromSmarts
    if not hasattr(original, "cache_info"):
        template_decoder.rdChemReactions.ReactionFromSmarts = functools.lru_cache(
            maxsize=None
        )(original)
    _TD = template_decoder


def _parse_prediction(encoded: str) -> tuple[str, int, int, float]:
    """Parse LocalRetro's ``(a|b, site, class, score)`` without ``eval``.

    The pinned writer deliberately emits the edit type as the bare names
    ``a`` and ``b``.  Upstream resolves those names through module globals;
    ``ast.literal_eval`` therefore cannot read the official file format.
    """

    try:
        expression = ast.parse(encoded, mode="eval").body
    except SyntaxError as error:
        raise ValueError(f"Malformed LocalRetro prediction: {encoded!r}") from error
    if not isinstance(expression, ast.Tuple) or len(expression.elts) != 4:
        raise ValueError(f"Malformed LocalRetro prediction tuple: {encoded!r}")
    edit_node, site_node, class_node, score_node = expression.elts
    if isinstance(edit_node, ast.Name) and edit_node.id in {"a", "b"}:
        edit_type = edit_node.id
    elif (
        isinstance(edit_node, ast.Constant)
        and isinstance(edit_node.value, str)
        and edit_node.value in {"a", "b"}
    ):
        edit_type = edit_node.value
    else:
        raise ValueError(f"Invalid LocalRetro edit type: {encoded!r}")
    try:
        site_index = ast.literal_eval(site_node)
        template_class = ast.literal_eval(class_node)
        score = ast.literal_eval(score_node)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid LocalRetro prediction values: {encoded!r}") from error
    if (
        not isinstance(site_index, int)
        or isinstance(site_index, bool)
        or not isinstance(template_class, int)
        or isinstance(template_class, bool)
        or not isinstance(score, (int, float))
        or isinstance(score, bool)
    ):
        raise ValueError(f"Invalid LocalRetro prediction value types: {encoded!r}")
    return edit_type, site_index, template_class, float(score)


def _decode_fast(test_id: int) -> tuple[int, list[str]]:
    """Decode one product with upstream ordering and one decoder call/proposal."""

    if _TD is None:
        _worker_init()
    product_smiles, predictions = _RAW[int(test_id)]
    molecule = Chem.MolFromSmiles(product_smiles)
    if molecule is None:
        raise ValueError(f"Invalid product SMILES for test_id={test_id}.")
    atom_sites, bond_sites = _TD.get_edit_site(molecule)
    index_map = _TD.get_idx_map(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    decoded_predictions: list[str] = []
    for encoded_prediction in predictions:
        if len(encoded_prediction) == 1:
            continue
        edit_type, site_index, template_class, score = _parse_prediction(
            encoded_prediction
        )
        template_class = int(template_class)
        if edit_type == "a":
            prediction_site: int | tuple[int, int] = atom_sites[int(site_index)]
            template = _ATOM_TEMPLATES[template_class]
            if len(template.split(">>")[0].split(".")) > 1:
                prediction_site = index_map[prediction_site]
        else:
            prediction_site = bond_sites[int(site_index)]
            template = _BOND_TEMPLATES[template_class]
            if len(template.split(">>")[0].split(".")) > 1:
                prediction_site = (
                    index_map[prediction_site[0]],
                    index_map[prediction_site[1]],
                )
        local_template = ">>".join(
            f"({smarts})" for smarts in template.split("_")[0].split(">>")
        )
        try:
            decoded = _TD.decode_localtemplate(
                molecule,
                prediction_site,
                local_template,
                _TEMPLATE_INFOS[template],
            )
        except Exception:
            continue
        if decoded is None:
            continue
        candidate = str((decoded, score))
        if candidate in decoded_predictions:
            continue
        decoded_predictions.append(candidate)
        if len(decoded_predictions) >= TOP_K:
            break
    return int(test_id), decoded_predictions


def _load_context(workspace: Path) -> dict[str, Any]:
    global _RAW, _ATOM_TEMPLATES, _BOND_TEMPLATES, _TEMPLATE_INFOS

    validate_workspace(workspace)
    _install_workspace_imports(workspace)
    dataset = workspace / "data" / DATASET_NAME
    raw = workspace / "outputs" / "raw_prediction" / f"LocalRetro_{DATASET_NAME}.txt"
    decoded = (
        workspace / "outputs" / "decoded_prediction" / f"LocalRetro_{DATASET_NAME}.txt"
    )
    inventory = dataset / "inference_products.jsonl"
    paths = {
        "raw": raw,
        "decoded": decoded,
        "inventory": inventory,
        "atom_templates": dataset / "atom_templates.csv",
        "bond_templates": dataset / "bond_templates.csv",
        "template_infos": dataset / "template_infos.csv",
    }
    for name, path in paths.items():
        if name != "decoded" and not path.is_file():
            raise FileNotFoundError(f"Required decoder input is missing: {path}")

    raw_predictions: dict[int, tuple[str, list[str]]] = {}
    with raw.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        if not header.startswith("Test_id\tProduct\tPrediction 1"):
            raise RuntimeError("Raw LocalRetro prediction header differs.")
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise RuntimeError(f"Malformed raw prediction line {line_number}.")
            test_id = int(fields[0])
            if test_id in raw_predictions:
                raise RuntimeError(f"Duplicate raw prediction test_id={test_id}.")
            raw_predictions[test_id] = (fields[1], fields[2:])
    expected_ids = set(range(len(raw_predictions)))
    if set(raw_predictions) != expected_ids:
        raise RuntimeError("Raw LocalRetro prediction IDs are not contiguous from zero.")
    inventory_count = sum(
        bool(line.strip()) for line in inventory.read_text(encoding="utf-8").splitlines()
    )
    if inventory_count != len(raw_predictions):
        raise RuntimeError(
            f"Raw/inventory product counts differ: {len(raw_predictions)} vs {inventory_count}."
        )

    atom_frame = pd.read_csv(paths["atom_templates"])
    bond_frame = pd.read_csv(paths["bond_templates"])
    info_frame = pd.read_csv(paths["template_infos"])
    _RAW = raw_predictions
    _ATOM_TEMPLATES = {
        int(row.Class): str(row.Template) for row in atom_frame.itertuples(index=False)
    }
    _BOND_TEMPLATES = {
        int(row.Class): str(row.Template) for row in bond_frame.itertuples(index=False)
    }
    _TEMPLATE_INFOS = {
        str(row.Template): {
            "edit_site": ast.literal_eval(str(row.edit_site)),
            "change_H": ast.literal_eval(str(row.change_H)),
            "change_C": ast.literal_eval(str(row.change_C)),
            "change_S": ast.literal_eval(str(row.change_S)),
        }
        for row in info_frame.itertuples(index=False)
    }
    return {
        "workspace": workspace,
        "paths": paths,
        "product_count": len(raw_predictions),
        "fingerprints": {
            name: fingerprint(path)
            for name, path in paths.items()
            if name != "decoded"
        },
    }


def _audit_equivalence(
    context: Mapping[str, Any], output: Path, workers: int, sample_size: int
) -> dict[str, Any]:
    if output.is_file():
        audit = json.loads(output.read_text(encoding="utf-8"))
        if audit.get("decoder_protocol_id") != DECODER_PROTOCOL_ID:
            raise RuntimeError("Existing decoder audit uses another protocol.")
        if audit.get("input_raw_sha256") != context["fingerprints"]["raw"]["sha256"]:
            raise RuntimeError("Existing decoder audit belongs to another raw input.")
        if audit.get("exact_equal") is not True:
            raise RuntimeError("Existing decoder equivalence audit did not pass.")
        return audit

    count = int(context["product_count"])
    anchors = [0, 1, 2, 3, count // 4, count // 2, 3 * count // 4, count - 1]
    sample_ids = sorted(set(anchors))[:sample_size]
    from Decode_predictions import get_k_predictions

    official_args = {
        "raw_predictions": {
            test_id: [_RAW[test_id][0], *_RAW[test_id][1]] for test_id in sample_ids
        },
        "atom_templates": _ATOM_TEMPLATES,
        "bond_templates": _BOND_TEMPLATES,
        "template_infos": _TEMPLATE_INFOS,
        "rxn_class_given": False,
        "top_k": TOP_K,
    }
    started = time.perf_counter()
    fork = mp.get_context("fork")
    official_function = functools.partial(get_k_predictions, args=official_args)
    with fork.Pool(processes=min(workers, len(sample_ids))) as pool:
        official_rows = dict(pool.imap_unordered(official_function, sample_ids))
    official = {
        test_id: official_rows[test_id][0] for test_id in sorted(official_rows)
    }
    with fork.Pool(
        processes=min(workers, len(sample_ids)), initializer=_worker_init
    ) as pool:
        fast = dict(pool.imap_unordered(_decode_fast, sample_ids))
    exact = official == fast
    audit = {
        "schema_version": 1,
        "decoder_protocol_id": DECODER_PROTOCOL_ID,
        "comparator": "pinned upstream Decode_predictions.get_k_predictions",
        "single_intended_change": (
            "remove duplicate decoder call, cache invariant preparation/SMARTS, "
            "and persist products incrementally"
        ),
        "sample_ids": sample_ids,
        "sample_products": len(sample_ids),
        "official_output_sha256": _json_sha256(official),
        "accelerated_output_sha256": _json_sha256(fast),
        "exact_equal": exact,
        "input_raw_sha256": context["fingerprints"]["raw"]["sha256"],
        "runtime_seconds": time.perf_counter() - started,
        "created_at_utc": utc_now(),
    }
    atomic_json(output, audit)
    if not exact:
        mismatches = [test_id for test_id in sample_ids if official[test_id] != fast[test_id]]
        raise RuntimeError(f"Accelerated decoder differs on audit IDs: {mismatches}")
    return audit


def _metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    return {
        key: json.loads(value)
        for key, value in connection.execute("SELECT key,value FROM metadata")
    }


def _set_metadata(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
        (key, json.dumps(value, sort_keys=True)),
    )


def _write_complete_progress(state: Path, product_count: int) -> None:
    atomic_json(
        state / "progress.json",
        {
            "status": "complete",
            "decoder_protocol_id": DECODER_PROTOCOL_ID,
            "completed_products": product_count,
            "product_count": product_count,
            "percent": 100.0,
            "eta_seconds": 0.0,
            "updated_at_utc": utc_now(),
        },
    )


def run_decode(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.workers <= 8:
        raise ValueError("This 9-core instance permits 1--8 decoder workers.")
    workspace = Path(args.workspace).resolve()
    context = _load_context(workspace)
    decoded = context["paths"]["decoded"]
    state = Path(args.state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True)
    audit_path = state / "equivalence_audit.json"
    audit = _audit_equivalence(context, audit_path, args.workers, args.audit_products)
    manifest_path = state / "decoder_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["output"]["sha256"] != sha256_file(decoded):
            raise RuntimeError("Completed decoder output differs from its manifest.")
        _write_complete_progress(state, int(manifest["product_count"]))
        print(json.dumps(manifest, indent=2), flush=True)
        return manifest
    if decoded.exists():
        raise FileExistsError("Decoded output exists without a completed decoder manifest.")

    database = state / "decoded_products.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS decoded ("
        "test_id INTEGER PRIMARY KEY, predictions TEXT NOT NULL, completed_at_utc TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    expected_metadata = {
        "decoder_protocol_id": DECODER_PROTOCOL_ID,
        "raw_sha256": context["fingerprints"]["raw"]["sha256"],
        "product_count": context["product_count"],
        "top_k": TOP_K,
    }
    existing = _metadata(connection)
    for key, expected in expected_metadata.items():
        if key in existing and existing[key] != expected:
            raise RuntimeError(f"Decoder resume database has incompatible {key}.")
        _set_metadata(connection, key, expected)
    connection.commit()

    completed = {
        int(row[0]) for row in connection.execute("SELECT test_id FROM decoded")
    }
    pending = [test_id for test_id in sorted(_RAW) if test_id not in completed]
    print(
        f"FAST DECODE: {len(completed):,} cached, {len(pending):,} pending, "
        f"workers={args.workers}.",
        flush=True,
    )
    started = time.perf_counter()
    fork = mp.get_context("fork")
    with fork.Pool(processes=args.workers, initializer=_worker_init) as pool:
        for offset, (test_id, predictions) in enumerate(
            pool.imap_unordered(_decode_fast, pending, chunksize=1), start=1
        ):
            connection.execute(
                "INSERT INTO decoded(test_id,predictions,completed_at_utc) VALUES(?,?,?)",
                (test_id, json.dumps(predictions, separators=(",", ":")), utc_now()),
            )
            if offset % 8 == 0:
                connection.commit()
            total = len(completed) + offset
            if total % 100 == 0 or total == context["product_count"]:
                connection.commit()
                elapsed = max(time.perf_counter() - started, 1e-9)
                rate = offset / elapsed
                eta = (context["product_count"] - total) / rate
                progress = {
                    "status": "running",
                    "decoder_protocol_id": DECODER_PROTOCOL_ID,
                    "completed_products": total,
                    "product_count": context["product_count"],
                    "percent": 100.0 * total / context["product_count"],
                    "workers": args.workers,
                    "products_per_second": rate,
                    "eta_seconds": eta,
                    "updated_at_utc": utc_now(),
                }
                atomic_json(state / "progress.json", progress)
                width = int(progress["percent"] * 30 / 100)
                bar = "#" * width + "-" * (30 - width)
                print(
                    f"FAST DECODE [{bar}] {progress['percent']:.1f}% | "
                    f"{total:,}/{context['product_count']:,} | "
                    f"{rate:.2f} product/s | ETA {eta/3600:.2f} h",
                    flush=True,
                )
    connection.commit()
    row_count = int(connection.execute("SELECT COUNT(*) FROM decoded").fetchone()[0])
    if row_count != context["product_count"]:
        raise RuntimeError(f"Decoded product count differs: {row_count}.")
    lines: list[str] = []
    for test_id, encoded in connection.execute(
        "SELECT test_id,predictions FROM decoded ORDER BY test_id"
    ):
        predictions = json.loads(encoded)
        lines.append("\t".join([str(test_id), *predictions]) + "\n")
    _atomic_text(decoded, "".join(lines))
    _set_metadata(connection, "complete", True)
    _set_metadata(connection, "completed_at_utc", utc_now())
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()

    manifest = {
        "schema_version": 1,
        "decoder_protocol_id": DECODER_PROTOCOL_ID,
        "workspace_protocol_id": PROTOCOL_ID,
        "comparator": "pinned upstream LocalRetro decoder",
        "single_intended_change": "compute-only resumable output-equivalent decoding",
        "settings": {
            "workers": args.workers,
            "top_k": TOP_K,
            "duplicate_decoder_call_removed": True,
            "product_preparation_cached": True,
            "reaction_smarts_compilation_cached_per_worker": True,
            "result_order": "ascending test_id",
        },
        "inputs": context["fingerprints"],
        "equivalence_audit": fingerprint(audit_path),
        "equivalence_passed": audit["exact_equal"],
        "state_database": fingerprint(database),
        "product_count": row_count,
        "output": fingerprint(decoded),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "rdkit": importlib.metadata.version("rdkit"),
        },
        "created_at_utc": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    _write_complete_progress(state, row_count)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(args.workspace).resolve()
    validate_workspace(workspace)
    dataset = workspace / "data" / DATASET_NAME
    decoder_manifest_path = Path(args.decoder_manifest).resolve()
    decoder = json.loads(decoder_manifest_path.read_text(encoding="utf-8"))
    if decoder.get("decoder_protocol_id") != DECODER_PROTOCOL_ID:
        raise RuntimeError("Decoder manifest protocol differs.")
    if decoder.get("equivalence_passed") is not True:
        raise RuntimeError("Decoder equivalence gate did not pass.")
    decoded = Path(decoder["output"]["path"])
    if decoder["output"]["sha256"] != sha256_file(decoded):
        raise RuntimeError("Decoded output differs from its frozen manifest.")

    freeze_path = workspace / "outputs" / "revision_freeze" / "checkpoint_freeze.json"
    checkpoint = workspace / "models" / f"LocalRetro_{DATASET_NAME}.pth"
    inference_manifest = dataset / "inference_input_manifest.json"
    inventory = dataset / "inference_products.jsonl"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("protocol_id") != PROTOCOL_ID or freeze.get("test_partition_loaded") is not False:
        raise RuntimeError("Invalid pre-test LocalRetro freeze.")
    if freeze["checkpoint"]["sha256"] != sha256_file(checkpoint):
        raise RuntimeError("Checkpoint differs from the immutable freeze.")

    final_dir = workspace / "outputs" / "revision_localretro_top50"
    final_predictions = final_dir / "localretro_top50.jsonl"
    final_manifest = final_dir / "manifest.json"
    if final_manifest.is_file():
        result = json.loads(final_manifest.read_text(encoding="utf-8"))
        if result["decoder_manifest"]["sha256"] != sha256_file(decoder_manifest_path):
            raise RuntimeError("Existing final result belongs to another decoder manifest.")
        print(json.dumps(result, indent=2), flush=True)
        return result
    if final_dir.exists():
        raise FileExistsError("Partial final LocalRetro result directory exists.")
    final_dir.mkdir(parents=True)
    counts = compile_decoded_predictions(
        decoded_path=decoded,
        inventory_path=inventory,
        output_path=final_predictions,
        cap=TOP_K,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "comparator": "AiZynthFinder cap50-legacy-anchored-v1 Top-50 pool",
        "single_intended_change": "generator changed to pinned retrained LocalRetro",
        "localretro_commit": LOCALRETRO_COMMIT,
        "checkpoint_freeze": fingerprint(freeze_path),
        "checkpoint": fingerprint(checkpoint),
        "inference_input": fingerprint(inference_manifest),
        "decoder_manifest": fingerprint(decoder_manifest_path),
        "settings": {
            "top_edit_predictions": 100,
            "top_decoded_unique_candidates": TOP_K,
            "reaction_class_given": False,
            "prior": "1-(rank-1)/max(n-1,1)",
            "decoder_protocol_id": DECODER_PROTOCOL_ID,
        },
        "counts": counts,
        "outputs": {
            "raw_edits": decoder["inputs"]["raw"],
            "decoded_upstream_equivalent": fingerprint(decoded),
            "candidate_jsonl": fingerprint(final_predictions),
        },
        "test_partition_loaded_only_after_freeze": True,
        "created_at_utc": utc_now(),
    }
    atomic_json(final_manifest, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    decode = commands.add_parser("decode")
    decode.add_argument("--workspace", required=True)
    decode.add_argument("--state-dir", required=True)
    decode.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    decode.add_argument("--audit-products", type=int, default=8)
    finish = commands.add_parser("finalize")
    finish.add_argument("--workspace", required=True)
    finish.add_argument("--decoder-manifest", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "decode":
        run_decode(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
