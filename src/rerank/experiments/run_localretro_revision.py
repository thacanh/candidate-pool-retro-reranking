#!/usr/bin/env python
"""Fail-closed LocalRetro training and Top-50 inference for WS-E.

The pinned upstream checkout remains immutable.  A disposable run workspace is
staged from it, receives only mapped train/validation reactions, and is trained
with the official architecture/default optimization settings plus seed 42.
The official test/inference loader is unavailable until the selected checkpoint
has been fingerprinted in an immutable pre-test freeze.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
from rdkit import Chem

from rerank.data.prepare_localretro_current_dataset import (
    atomic_json,
    atomic_jsonl,
    canonical_fragments,
    fingerprint,
    sha256_file,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "localretro-top50-current-split-filtered-v2"
LOCALRETRO_COMMIT = "eba83e72efabeb854fec86c865e8743c295a8a1e"
DATASET_NAME = "CHEMFORMER_50K_RXNMAPPER"
TRAINING_SEED = 42
OFFICIAL_SETTINGS = {
    "batch_size": 16,
    "num_epochs": 50,
    "patience": 5,
    "max_clip": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-6,
    "schedule_step": 10,
    "num_workers": 0,
    "print_every": 20,
    "config": "default_config.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_state(repository: Path) -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip()
    return commit, dirty


def validate_official_checkout(repository: str | Path) -> dict[str, Any]:
    root = Path(repository).resolve()
    if (root / ".git").is_dir():
        commit, dirty = _git_state(root)
        if dirty:
            raise RuntimeError("LocalRetro checkout has uncommitted changes.")
        verification = "clean git checkout"
    else:
        marker = root / ".pinned_revision.json"
        if not marker.is_file():
            raise FileNotFoundError("LocalRetro source lacks git metadata or a pinned marker.")
        marker_body = json.loads(marker.read_text(encoding="utf-8"))
        commit = str(marker_body.get("commit", ""))
        verification = "bundle source hashes plus pinned revision marker"
    if commit != LOCALRETRO_COMMIT:
        raise RuntimeError("LocalRetro source is not the pinned commit.")
    license_path = root / "LICENSE"
    checkpoint = root / "models" / "LocalRetro_USPTO_50K.pth"
    return {
        "repository": str(root),
        "commit": commit,
        "verification": verification,
        "license": fingerprint(license_path),
        "upstream_pretrained_checkpoint_audit_only": (
            fingerprint(checkpoint) if checkpoint.is_file() else None
        ),
    }


def validate_workspace(workspace: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(workspace).resolve()
    manifest_path = root / "PINNED_SOURCE.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run workspace lacks PINNED_SOURCE.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Run workspace protocol differs.")
    if manifest.get("localretro_commit") != LOCALRETRO_COMMIT:
        raise RuntimeError("Run workspace LocalRetro commit differs.")
    return root, manifest


def stage_workspace(
    *, official_root: str | Path, mapped_dataset_dir: str | Path, workspace: str | Path
) -> dict[str, Any]:
    source_record = validate_official_checkout(official_root)
    dataset_source = Path(mapped_dataset_dir).resolve()
    mapping_manifest = dataset_source / "mapping_manifest.json"
    raw_train = dataset_source / "raw_train.csv"
    raw_val = dataset_source / "raw_val.csv"
    for path in (mapping_manifest, raw_train, raw_val):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (dataset_source / "raw_test.csv").exists():
        raise RuntimeError("Mapped training source unexpectedly contains raw_test.csv.")
    mapping = json.loads(mapping_manifest.read_text(encoding="utf-8"))
    if mapping.get("counts", {}).get("test_rows_loaded") != 0:
        raise RuntimeError("Mapped dataset manifest is not test-closed.")

    final = Path(workspace).resolve()
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite LocalRetro workspace: {final}")
    staging = final.with_name(f".{final.name}.{os.getpid()}.tmp")

    def ignore(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).resolve().relative_to(Path(official_root).resolve())
        ignored: set[str] = set()
        if relative == Path("."):
            ignored.update({".git", "outputs"})
        if relative == Path("models"):
            ignored.update(name for name in names if name.endswith(".pth"))
        if relative == Path("data"):
            ignored.update(name for name in names if name.startswith("USPTO_"))
        return ignored

    try:
        shutil.copytree(Path(official_root).resolve(), staging, ignore=ignore)
        target_dataset = staging / "data" / DATASET_NAME
        target_dataset.mkdir(parents=True)
        shutil.copy2(raw_train, target_dataset / "raw_train.csv")
        shutil.copy2(raw_val, target_dataset / "raw_val.csv")
        shutil.copy2(mapping_manifest, target_dataset / "mapping_manifest.json")
        source_manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "comparator": "AiZynthFinder cap50-legacy-anchored-v1 generator",
            "single_intended_change": "candidate generator changed to LocalRetro",
            "localretro_commit": LOCALRETRO_COMMIT,
            "upstream": source_record,
            "dataset": {
                "name": DATASET_NAME,
                "mapping_manifest": fingerprint(mapping_manifest),
                "raw_train": fingerprint(target_dataset / "raw_train.csv"),
                "raw_val": fingerprint(target_dataset / "raw_val.csv"),
                "raw_test_present": False,
            },
            "created_at_utc": utc_now(),
        }
        atomic_json(staging / "PINNED_SOURCE.json", source_manifest)
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return source_manifest


@contextlib.contextmanager
def _workspace_imports(workspace: Path, subdirectory: str) -> Iterator[None]:
    previous_cwd = Path.cwd()
    added = [str(workspace / subdirectory), str(workspace)]
    for path in reversed(added):
        sys.path.insert(0, path)
    os.chdir(workspace / subdirectory)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        for path in added:
            with contextlib.suppress(ValueError):
                sys.path.remove(path)


def preprocess_train_valid(*, workspace: str | Path) -> dict[str, Any]:
    root, source_manifest = validate_workspace(workspace)
    dataset = root / "data" / DATASET_NAME
    if (dataset / "raw_test.csv").exists():
        raise RuntimeError("raw_test.csv must not exist before checkpoint freeze.")
    output_manifest = dataset / "preprocess_train_valid_manifest.json"
    if output_manifest.exists() or (dataset / "labeled_data.csv").exists():
        raise FileExistsError("Refusing to overwrite LocalRetro preprocessing output.")
    start = time.perf_counter()
    command = [
        sys.executable,
        "Extract_from_train_data.py",
        "-d",
        DATASET_NAME,
    ]
    subprocess.run(command, cwd=root / "preprocessing", check=True)

    with _workspace_imports(root, "preprocessing"):
        from Run_preprocessing import (
            build_template_extractor,
            labeling_dataset,
            load_templates,
        )

        args = {
            "dataset": DATASET_NAME,
            "force": False,
            "retro": True,
            "verbose": False,
            "max_edit_n": 8,
            "use_stereo": True,
            "min_template_n": 1,
            "output_dir": f"../data/{DATASET_NAME}",
        }
        template_dicts, template_infos = load_templates(args)
        extractor = build_template_extractor(args)
        validation = labeling_dataset(
            args, "val", template_dicts, template_infos, extractor
        ).copy()
        training = labeling_dataset(
            args, "train", template_dicts, template_infos, extractor
        ).copy()

    training["Split"] = "train"
    validation["Split"] = "val"
    combined = pd.concat([training, validation], ignore_index=True)
    combined["Mask"] = [
        int(float(frequency) >= 1) for frequency in combined["Frequency"]
    ]
    combined.to_csv(dataset / "labeled_data.csv", index=False, lineterminator="\n")
    if set(combined["Split"]) != {"train", "val"}:
        raise RuntimeError("Preprocessed data contains an unexpected split.")
    if (dataset / "preprocessed_test.csv").exists():
        raise RuntimeError("Official test was unexpectedly preprocessed.")
    template_outputs = {
        name: fingerprint(dataset / name)
        for name in ("atom_templates.csv", "bond_templates.csv", "template_infos.csv")
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "comparator": "unmapped current train/official-validation reactions",
        "single_intended_change": "LocalRetro template extraction and labels",
        "localretro_source": fingerprint(root / "PINNED_SOURCE.json"),
        "counts": {
            "train": len(training),
            "valid": len(validation),
            "test": 0,
            "masked_train_valid": int((combined["Mask"] == 0).sum()),
            "atom_templates": len(pd.read_csv(dataset / "atom_templates.csv")),
            "bond_templates": len(pd.read_csv(dataset / "bond_templates.csv")),
        },
        "outputs": {
            **template_outputs,
            "preprocessed_train": fingerprint(dataset / "preprocessed_train.csv"),
            "preprocessed_valid": fingerprint(dataset / "preprocessed_val.csv"),
            "labeled_train_valid": fingerprint(dataset / "labeled_data.csv"),
        },
        "test_partition_loaded": False,
        "runtime_seconds": time.perf_counter() - start,
        "created_at_utc": utc_now(),
    }
    atomic_json(output_manifest, result)
    return result


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_and_freeze(
    *, workspace: str | Path, device: str = "cuda:0"
) -> dict[str, Any]:
    root, source_manifest = validate_workspace(workspace)
    dataset = root / "data" / DATASET_NAME
    preprocess_manifest = dataset / "preprocess_train_valid_manifest.json"
    if not preprocess_manifest.is_file():
        raise FileNotFoundError("Train/valid preprocessing has not completed.")
    if (dataset / "raw_test.csv").exists():
        raise RuntimeError("raw_test.csv must not exist during checkpoint selection.")
    checkpoint = root / "models" / f"LocalRetro_{DATASET_NAME}.pth"
    freeze_path = root / "outputs" / "revision_freeze" / "checkpoint_freeze.json"
    history_path = root / "outputs" / "revision_freeze" / "validation_history.json"
    if checkpoint.exists() or freeze_path.exists() or history_path.exists():
        raise FileExistsError("Refusing to overwrite LocalRetro training/freeze output.")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    _seed_everything(TRAINING_SEED)
    start = time.perf_counter()

    import torch
    from torch.utils.data import DataLoader

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    with _workspace_imports(root, "scripts"):
        from dgl.data.utils import Subset
        from dataset import USPTODataset
        from Train import run_a_train_epoch, run_an_eval_epoch
        from utils import collate_molgraphs, init_featurizer, load_model

        args = {
            **OFFICIAL_SETTINGS,
            "dataset": DATASET_NAME,
            "gpu": device,
            "mode": "train",
            "device": resolved_device,
            "model_path": f"../models/LocalRetro_{DATASET_NAME}.pth",
            "config_path": "../data/configs/default_config.json",
            "data_dir": f"../data/{DATASET_NAME}",
        }
        args = init_featurizer(args)
        model, loss_criterion, optimizer, scheduler, stopper = load_model(args)
        dataset_object = USPTODataset(
            args,
            smiles_to_graph=__import__("functools").partial(
                __import__("dgllife.utils", fromlist=["smiles_to_bigraph"]).smiles_to_bigraph,
                add_self_loop=True,
            ),
            node_featurizer=args["node_featurizer"],
            edge_featurizer=args["edge_featurizer"],
        )
        if len(dataset_object.test_ids) != 0:
            raise RuntimeError("Training dataset unexpectedly exposes test IDs.")
        generator = torch.Generator()
        generator.manual_seed(TRAINING_SEED)
        train_loader = DataLoader(
            Subset(dataset_object, dataset_object.train_ids),
            batch_size=args["batch_size"],
            shuffle=True,
            generator=generator,
            collate_fn=collate_molgraphs,
            num_workers=0,
        )
        valid_loader = DataLoader(
            Subset(dataset_object, dataset_object.val_ids),
            batch_size=args["batch_size"],
            shuffle=False,
            collate_fn=collate_molgraphs,
            num_workers=0,
        )
        history: list[dict[str, Any]] = []
        best_score: float | None = None
        best_epoch: int | None = None
        for epoch in range(args["num_epochs"]):
            run_a_train_epoch(
                args, epoch, model, train_loader, loss_criterion, optimizer
            )
            validation_loss = float(
                run_an_eval_epoch(args, model, valid_loader, loss_criterion)
            )
            early_stop = bool(stopper.step(validation_loss, model))
            scheduler.step()
            if best_score is None or validation_loss < best_score:
                best_score = validation_loss
                best_epoch = epoch + 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "validation_loss": validation_loss,
                    "best_validation_loss": best_score,
                    "early_stop": early_stop,
                }
            )
            print(
                f"epoch {epoch+1}/{args['num_epochs']} validation_loss={validation_loss:.6f} "
                f"best={best_score:.6f}",
                flush=True,
            )
            if early_stop:
                break

    if not checkpoint.is_file() or best_epoch is None or best_score is None:
        raise RuntimeError("LocalRetro training did not produce a checkpoint.")
    atomic_json(
        history_path,
        {
            "protocol_id": PROTOCOL_ID,
            "selection_split": "official valid",
            "selection_metric": "validation cross-entropy loss",
            "history": history,
        },
    )
    freeze = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "comparator": "AiZynthFinder cap50-legacy-anchored-v1 generator",
        "single_intended_change": "candidate generator changed to LocalRetro",
        "localretro_commit": LOCALRETRO_COMMIT,
        "training_seed": TRAINING_SEED,
        "settings": OFFICIAL_SETTINGS,
        "selection": {
            "split": "official valid",
            "metric": "validation loss",
            "best_epoch": best_epoch,
            "best_validation_loss": best_score,
            "epochs_completed": len(history),
        },
        "input_fingerprints": {
            "workspace_source": fingerprint(root / "PINNED_SOURCE.json"),
            "preprocess_manifest": fingerprint(preprocess_manifest),
            "config": fingerprint(root / "data" / "configs" / "default_config.json"),
        },
        "checkpoint": fingerprint(checkpoint),
        "validation_history": fingerprint(history_path),
        "test_partition_loaded": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "dgl": importlib.metadata.version("dgl"),
            "dgllife": importlib.metadata.version("dgllife"),
            "rdkit": importlib.metadata.version("rdkit"),
            "device": str(resolved_device),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "runtime_seconds": time.perf_counter() - start,
        "created_at_utc": utc_now(),
    }
    atomic_json(freeze_path, freeze)
    return freeze


def _candidate_identity(smiles: str) -> tuple[str, ...]:
    return canonical_fragments(smiles)


def compile_decoded_predictions(
    *, decoded_path: str | Path,
    inventory_path: str | Path,
    output_path: str | Path,
    cap: int = 50,
) -> dict[str, Any]:
    inventory_records = [
        json.loads(line)
        for line in Path(inventory_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    products = {int(record["test_id"]): str(record["product"]) for record in inventory_records}
    parsed: dict[int, list[tuple[str, float]]] = {}
    invalid = 0
    duplicates = 0
    with Path(decoded_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if not fields or not fields[0]:
                continue
            test_id = int(fields[0])
            candidates: dict[tuple[str, ...], tuple[str, float, int]] = {}
            for first_seen, encoded in enumerate(fields[1:]):
                if not encoded:
                    continue
                try:
                    candidate, score = ast.literal_eval(encoded)
                    candidate = str(candidate)
                    score = float(score)
                    identity = _candidate_identity(candidate)
                    if not identity:
                        raise ValueError("empty candidate")
                except Exception:
                    invalid += 1
                    continue
                current = candidates.get(identity)
                if current is not None:
                    duplicates += 1
                    if score <= current[1]:
                        continue
                candidates[identity] = (candidate, score, first_seen)
            ordered = sorted(candidates.values(), key=lambda value: (-value[1], value[2]))[:cap]
            parsed[test_id] = [(candidate, score) for candidate, score, _ in ordered]
    if set(parsed) != set(products):
        raise RuntimeError(
            f"Decoded product IDs differ: missing={len(set(products)-set(parsed))}, "
            f"extra={len(set(parsed)-set(products))}."
        )
    records: list[dict[str, Any]] = []
    products_without_candidates = 0
    for test_id in sorted(products):
        candidates = parsed[test_id]
        if not candidates:
            products_without_candidates += 1
        denominator = max(len(candidates) - 1, 1)
        for rank, (candidate, score) in enumerate(candidates, start=1):
            records.append(
                {
                    "test_id": test_id,
                    "product": products[test_id],
                    "reactant": candidate,
                    "raw_score": score,
                    "generator_rank": rank,
                    "prior": 1.0 - (rank - 1) / denominator,
                    "source_localretro": 1,
                    "source_aizynthfinder": 0,
                    "protocol_id": PROTOCOL_ID,
                }
            )
    atomic_jsonl(Path(output_path).resolve(), records)
    return {
        "products": len(products),
        "products_without_candidates": products_without_candidates,
        "candidates": len(records),
        "invalid_decoded_candidates": invalid,
        "canonical_duplicates_removed": duplicates,
        "maximum_candidates_per_product": max((len(value) for value in parsed.values()), default=0),
    }


def infer_and_decode(*, workspace: str | Path, device: str = "cuda:0") -> dict[str, Any]:
    root, _source_manifest = validate_workspace(workspace)
    dataset = root / "data" / DATASET_NAME
    freeze_path = root / "outputs" / "revision_freeze" / "checkpoint_freeze.json"
    inference_manifest = dataset / "inference_input_manifest.json"
    inventory = dataset / "inference_products.jsonl"
    if not freeze_path.is_file() or not inference_manifest.is_file() or not inventory.is_file():
        raise FileNotFoundError("Checkpoint freeze or product-only inference inputs are missing.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    checkpoint = root / "models" / f"LocalRetro_{DATASET_NAME}.pth"
    if freeze.get("test_partition_loaded") is not False:
        raise RuntimeError("Invalid pre-test checkpoint freeze.")
    if freeze["checkpoint"]["sha256"] != sha256_file(checkpoint):
        raise RuntimeError("Checkpoint differs from the immutable freeze.")
    if (dataset / "class_test.csv").exists():
        raise RuntimeError("Reaction-class-given inference is not permitted.")
    decoded = root / "outputs" / "decoded_prediction" / f"LocalRetro_{DATASET_NAME}.txt"
    raw = root / "outputs" / "raw_prediction" / f"LocalRetro_{DATASET_NAME}.txt"
    final_dir = root / "outputs" / "revision_localretro_top50"
    final_predictions = final_dir / "localretro_top50.jsonl"
    final_manifest = final_dir / "manifest.json"
    if raw.exists() or decoded.exists() or final_dir.exists():
        raise FileExistsError("Refusing to overwrite LocalRetro inference outputs.")
    start = time.perf_counter()
    subprocess.run(
        [sys.executable, "Test.py", "-d", DATASET_NAME, "-g", device],
        cwd=root / "scripts",
        check=True,
    )
    subprocess.run(
        [sys.executable, "Decode_predictions.py", "-d", DATASET_NAME],
        cwd=root / "scripts",
        check=True,
    )
    final_dir.mkdir(parents=True)
    counts = compile_decoded_predictions(
        decoded_path=decoded,
        inventory_path=inventory,
        output_path=final_predictions,
        cap=50,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "comparator": "AiZynthFinder cap50-legacy-anchored-v1 Top-50 pool",
        "single_intended_change": "generator changed to pinned retrained LocalRetro",
        "checkpoint_freeze": fingerprint(freeze_path),
        "checkpoint": fingerprint(checkpoint),
        "inference_input": fingerprint(inference_manifest),
        "settings": {
            "top_edit_predictions": 100,
            "top_decoded_unique_candidates": 50,
            "reaction_class_given": False,
            "prior": "1-(rank-1)/max(n-1,1)",
        },
        "counts": counts,
        "outputs": {
            "raw_edits": fingerprint(raw),
            "decoded_upstream": fingerprint(decoded),
            "candidate_jsonl": fingerprint(final_predictions),
        },
        "test_partition_loaded_only_after_freeze": True,
        "runtime_seconds": time.perf_counter() - start,
        "created_at_utc": utc_now(),
    }
    atomic_json(final_manifest, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--official-root", required=True)
    stage.add_argument("--mapped-dataset-dir", required=True)
    stage.add_argument("--workspace", required=True)
    preprocess = commands.add_parser("preprocess-train-valid")
    preprocess.add_argument("--workspace", required=True)
    train = commands.add_parser("train-freeze")
    train.add_argument("--workspace", required=True)
    train.add_argument("--device", default="cuda:0")
    infer = commands.add_parser("infer-top50")
    infer.add_argument("--workspace", required=True)
    infer.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "stage":
        result = stage_workspace(
            official_root=args.official_root,
            mapped_dataset_dir=args.mapped_dataset_dir,
            workspace=args.workspace,
        )
    elif args.command == "preprocess-train-valid":
        result = preprocess_train_valid(workspace=args.workspace)
    elif args.command == "train-freeze":
        result = train_and_freeze(workspace=args.workspace, device=args.device)
    else:
        result = infer_and_decode(workspace=args.workspace, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
