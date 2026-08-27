"""Prepare the current Chemformer USPTO-50K split for LocalRetro safely.

The Chemformer artifact contains RDKit molecules without atom-map numbers,
while LocalRetro template extraction requires mapped reactions.  This module
therefore maps only the current training and official-validation partitions
with one pinned RXNMapper model.  Official-test reactants are never mapped or
copied into the training workspace.  Product-only inference input is released
only after an immutable LocalRetro checkpoint freeze exists.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem


SCHEMA_VERSION = 1
PROTOCOL_ID = "localretro-chemformer50k-rxnmapper-filtered-v2"
DOWNSTREAM_PROTOCOL_ID = "localretro-top50-current-split-filtered-v2"
RXNMAPPER_COMMIT = "a01ecdcd5ac944850e9691739c1df858e005fd39"
RXNMAPPER_MODEL_RELATIVE = (
    "rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k"
)
RXNMAPPER_MODEL_FILES = {
    "config.json": "50bf1efa8c9710b38f017f332bb974ccc39df3bef028ef3d9d3b1f019a195e3f",
    "pytorch_model.bin": "8541f3f500dae71abe678d546bd035ca946e2d1c819f6b2cf41a97faedd7e6a2",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer_config.json": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "training_args.bin": "440fdf4459c2f51c3f042ccad777342b06bf3f491c251b712ab2aada5ff7a2eb",
    "vocab.txt": "99e9ad4949844c56205ed51dc62edaecb69787d63c6c74d607116c93eb2a5738",
}
ALLOWED_MAPPING_SPLITS = ("train", "valid")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def split_reaction(reaction: str) -> tuple[str, str, str]:
    parts = str(reaction).split(">")
    if len(parts) != 3:
        raise ValueError(f"Reaction must contain exactly two '>' separators: {reaction}")
    return parts[0], parts[1], parts[2]


def canonical_fragments(
    smiles: str, *, isomeric_smiles: bool = True
) -> tuple[str, ...]:
    if not str(smiles):
        return ()
    canonical: list[str] = []
    for fragment in str(smiles).split("."):
        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None:
            raise ValueError(f"Invalid molecule SMILES: {fragment}")
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        # Atom-map numbers participate in RDKit canonical ranking.  Removing
        # them without refreshing computed properties can therefore leave a
        # stale atom order and produce a different (but graph-equivalent) ring
        # closure such as ``c23`` versus ``c32``.  Recompute the sanitized
        # stereochemical graph before using canonical SMILES as an identity.
        molecule.ClearComputedProps()
        Chem.SanitizeMol(molecule)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
        canonical.append(
            Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=isomeric_smiles
            )
        )
    return tuple(sorted(canonical))


def reaction_identity(
    reaction: str, *, isomeric_smiles: bool = True
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    reactants, _agents, product = split_reaction(reaction)
    return (
        canonical_fragments(product, isomeric_smiles=isomeric_smiles),
        canonical_fragments(reactants, isomeric_smiles=isomeric_smiles),
    )


def _unmapped_molecule_key(molecule: Chem.Mol) -> str:
    copy = Chem.Mol(molecule)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=False)


def _transfer_side_atom_maps(original_side: str, mapped_side: str) -> str:
    """Transfer mapper indices onto the exact original stereochemical graphs."""

    originals = [Chem.MolFromSmiles(value) for value in original_side.split(".") if value]
    mapped = [Chem.MolFromSmiles(value) for value in mapped_side.split(".") if value]
    if any(molecule is None for molecule in originals + mapped):
        raise ValueError("Cannot transfer atom maps from an invalid molecule.")
    mapped_by_key: dict[str, list[Chem.Mol]] = collections.defaultdict(list)
    for molecule in mapped:
        mapped_by_key[_unmapped_molecule_key(molecule)].append(molecule)
    restored: list[str] = []
    for original in originals:
        key = _unmapped_molecule_key(original)
        candidates = mapped_by_key.get(key, [])
        if not candidates:
            raise ValueError(
                "RXNMapper output is not non-stereochemically identical to its input."
            )
        mapped_molecule = candidates.pop(0)
        query = Chem.Mol(mapped_molecule)
        mapper_numbers = [atom.GetAtomMapNum() for atom in query.GetAtoms()]
        for atom in query.GetAtoms():
            atom.SetAtomMapNum(0)
        matches = original.GetSubstructMatches(query, useChirality=False, uniquify=True)
        full_matches = [match for match in matches if len(match) == original.GetNumAtoms()]
        if not full_matches:
            raise ValueError("Could not transfer RXNMapper indices to the original graph.")
        match = min(full_matches)
        for query_index, original_index in enumerate(match):
            original.GetAtomWithIdx(original_index).SetAtomMapNum(
                int(mapper_numbers[query_index])
            )
        restored.append(
            Chem.MolToSmiles(original, canonical=True, isomericSmiles=True)
        )
    if any(values for values in mapped_by_key.values()):
        raise ValueError("RXNMapper output has unmatched molecular fragments.")
    return ".".join(restored)


def transfer_atom_maps_to_original(unmapped: str, mapped: str) -> tuple[str, dict[str, Any]]:
    """Keep original chemistry/stereo and copy only RXNMapper atom indices."""

    if reaction_identity(unmapped, isomeric_smiles=False) != reaction_identity(
        mapped, isomeric_smiles=False
    ):
        raise ValueError(
            "RXNMapper output is not non-stereochemically identical to its input."
        )
    original_reactants, original_agents, original_product = split_reaction(unmapped)
    mapped_reactants, _mapped_agents, mapped_product = split_reaction(mapped)
    restored = ">".join(
        (
            _transfer_side_atom_maps(original_reactants, mapped_reactants),
            original_agents,
            _transfer_side_atom_maps(original_product, mapped_product),
        )
    )
    if reaction_identity(restored) != reaction_identity(unmapped):
        raise ValueError("Atom-map transfer did not preserve the exact input chemistry.")
    stereo_restored = reaction_identity(mapped) != reaction_identity(unmapped)
    return restored, {
        "raw_mapper_exact_stereo_match": not stereo_restored,
        "original_stereochemistry_restored": stereo_restored,
    }


def validate_mapped_reaction(
    unmapped: str, mapped: str, *, allow_unbalanced_product: bool = False
) -> dict[str, Any]:
    if reaction_identity(unmapped) != reaction_identity(mapped):
        raise ValueError("Mapped reaction is not chemically identical to its input.")
    reactants, _agents, product = split_reaction(mapped)
    product_molecule = Chem.MolFromSmiles(product)
    if product_molecule is None:
        raise ValueError("Mapped product is invalid.")
    product_maps = [atom.GetAtomMapNum() for atom in product_molecule.GetAtoms()]
    if not product_maps or any(value <= 0 for value in product_maps):
        raise ValueError("Every product atom must have a positive atom-map number.")
    if len(product_maps) != len(set(product_maps)):
        raise ValueError("Product atom-map numbers are not unique.")

    reactant_maps: list[int] = []
    reactant_atom_count = 0
    for fragment in reactants.split("."):
        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None:
            raise ValueError("Mapped reactant fragment is invalid.")
        reactant_atom_count += molecule.GetNumAtoms()
        reactant_maps.extend(
            atom.GetAtomMapNum()
            for atom in molecule.GetAtoms()
            if atom.GetAtomMapNum() > 0
        )
    if len(reactant_maps) != len(set(reactant_maps)):
        raise ValueError("Positive reactant atom-map numbers are not unique.")
    missing_product_maps = sorted(set(product_maps) - set(reactant_maps))
    if missing_product_maps and not allow_unbalanced_product:
        raise ValueError("A product atom-map number is absent from the reactants.")
    return {
        "product_atom_count": len(product_maps),
        "reactant_atom_count": reactant_atom_count,
        "reactant_mapped_atom_count": len(reactant_maps),
        "reactant_unmapped_atom_count": reactant_atom_count - len(reactant_maps),
        "product_atom_maps_absent_from_reactants": len(missing_product_maps),
    }


def _load_current_rows(
    source_csv: str | Path, metadata_csv: str | Path
) -> list[dict[str, Any]]:
    source = pd.read_csv(source_csv)
    metadata = pd.read_csv(metadata_csv)
    if len(source) != len(metadata):
        raise ValueError("Source and metadata row counts differ.")
    required_source = {"reactants_smiles", "products_smiles", "set"}
    required_metadata = {"reaction_id", "source_split", "reaction_class"}
    if not required_source.issubset(source.columns):
        raise ValueError("Source CSV lacks the required columns.")
    if not required_metadata.issubset(metadata.columns):
        raise ValueError("Metadata CSV lacks the required columns.")
    records: list[dict[str, Any]] = []
    for index, (source_row, metadata_row) in enumerate(
        zip(source.itertuples(index=False), metadata.itertuples(index=False))
    ):
        reaction_id = int(metadata_row.reaction_id)
        source_split = str(metadata_row.source_split)
        if reaction_id != index:
            raise ValueError(f"Reaction ID mismatch at row {index}.")
        if source_split != str(source_row.set):
            raise ValueError(f"Split mismatch at row {index}.")
        records.append(
            {
                "reaction_id": reaction_id,
                "split": source_split,
                "reaction_class": int(metadata_row.reaction_class),
                "unmapped_reaction": (
                    f"{source_row.reactants_smiles}>>{source_row.products_smiles}"
                ),
            }
        )
    return records


def contiguous_bounds(item_count: int, shard_index: int, shard_count: int) -> tuple[int, int]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count.")
    base, remainder = divmod(item_count, shard_count)
    start = shard_index * base + min(shard_index, remainder)
    stop = start + base + int(shard_index < remainder)
    return start, stop


def verify_rxnmapper_assets(rxnmapper_root: str | Path) -> dict[str, Any]:
    root = Path(rxnmapper_root).resolve()
    if (root / ".git").is_dir():
        import subprocess

        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
        if dirty:
            raise RuntimeError("RXNMapper checkout has uncommitted changes.")
        verification = "clean git checkout"
    else:
        marker = root / ".pinned_revision.json"
        if not marker.is_file():
            raise FileNotFoundError("RXNMapper source lacks git metadata or a pinned marker.")
        marker_body = json.loads(marker.read_text(encoding="utf-8"))
        commit = str(marker_body.get("commit", ""))
        verification = "bundle source hashes plus pinned revision marker"
    if commit != RXNMAPPER_COMMIT:
        raise RuntimeError("RXNMapper source is not the pinned commit.")
    model_root = root / RXNMAPPER_MODEL_RELATIVE
    files: dict[str, Any] = {}
    for name, expected in RXNMAPPER_MODEL_FILES.items():
        path = model_root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Pinned RXNMapper model asset differs: {name}")
        files[name] = fingerprint(path)
    return {
        "repository_root": str(root),
        "commit": commit,
        "verification": verification,
        "model_files": files,
    }


def _mapped_paths(output_root: Path, shard_index: int, shard_count: int) -> tuple[Path, Path]:
    stem = f"mapped_shard_{shard_index:03d}_of_{shard_count:03d}"
    return output_root / "shards" / f"{stem}.jsonl", output_root / "shards" / f"{stem}.manifest.json"


def map_shard(
    *,
    source_csv: str | Path,
    metadata_csv: str | Path,
    rxnmapper_root: str | Path,
    output_root: str | Path,
    shard_index: int,
    shard_count: int,
    batch_size: int = 16,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    output = Path(output_root).resolve()
    data_path, manifest_path = _mapped_paths(output, shard_index, shard_count)
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable mapping shard {shard_index}/{shard_count}."
        )
    source_fp = fingerprint(source_csv)
    metadata_fp = fingerprint(metadata_csv)
    asset_record = verify_rxnmapper_assets(rxnmapper_root)
    rows = [
        row
        for row in _load_current_rows(source_csv, metadata_csv)
        if row["split"] in ALLOWED_MAPPING_SPLITS
    ]
    start, stop = contiguous_bounds(len(rows), shard_index, shard_count)
    selected = rows[start:stop]

    root = Path(rxnmapper_root).resolve()
    sys.path.insert(0, str(root))
    try:
        import rxnmapper as rxnmapper_package
        from rxnmapper import BatchedMapper
    finally:
        if sys.path[0] == str(root):
            sys.path.pop(0)

    random.seed(2026)
    np.random.seed(2026)
    import torch

    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    mapper = BatchedMapper(
        batch_size=batch_size,
        model_path=root / RXNMAPPER_MODEL_RELATIVE,
        canonicalize=True,
    )
    mapper.mapper.model.eval()
    start_time = time.perf_counter()
    results = list(
        mapper.map_reactions_with_info(
            [row["unmapped_reaction"] for row in selected], detailed=False
        )
    )
    if len(results) != len(selected):
        raise RuntimeError("RXNMapper returned the wrong result count.")

    output_records: list[dict[str, Any]] = []
    unmapped_reactant_atoms = 0
    stereo_restored_reactions = 0
    exclusions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row, result in zip(selected, results):
        try:
            if not result or "mapped_rxn" not in result:
                raise ValueError("RXNMapper returned an empty result.")
            mapped_reaction, transfer = transfer_atom_maps_to_original(
                row["unmapped_reaction"], str(result["mapped_rxn"])
            )
            validation = validate_mapped_reaction(
                row["unmapped_reaction"], mapped_reaction,
                allow_unbalanced_product=True,
            )
            missing_product_atoms = int(
                validation["product_atom_maps_absent_from_reactants"]
            )
            if missing_product_atoms:
                exclusions.append(
                    {
                        "reaction_id": row["reaction_id"],
                        "split": row["split"],
                        "reason": "product_atoms_absent_from_reactants",
                        "missing_product_atom_count": missing_product_atoms,
                    }
                )
                continue
            unmapped_reactant_atoms += validation["reactant_unmapped_atom_count"]
            stereo_restored_reactions += int(
                transfer["original_stereochemistry_restored"]
            )
            output_records.append(
                {
                    **row,
                    "mapped_reaction": mapped_reaction,
                    "mapping_confidence": float(result["confidence"]),
                    "mapping_validation": validation,
                    "map_transfer": transfer,
                    "protocol_id": PROTOCOL_ID,
                }
            )
        except Exception as error:
            failures.append(
                {"reaction_id": row["reaction_id"], "error": str(error)}
            )
    if failures:
        failure_path = output / "shards" / f"mapping_failures_{shard_index:03d}_of_{shard_count:03d}.json"
        atomic_json(failure_path, {"protocol_id": PROTOCOL_ID, "failures": failures})
        raise RuntimeError(
            f"Mapping shard failed validation for {len(failures)} reactions; see {failure_path}."
        )

    exclusion_path = (
        output / "shards" / f"mapping_exclusions_{shard_index:03d}_of_{shard_count:03d}.json"
    )
    atomic_json(
        exclusion_path,
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "rule": (
                "exclude only source reactions whose mapped product contains atom-map "
                "indices absent from all mapped reactants"
            ),
            "exclusions": exclusions,
        },
    )
    atomic_jsonl(data_path, output_records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "comparator": "unmapped Chemformer USPTO-50K train+valid reactions",
        "single_intended_change": "add deterministic RXNMapper atom-map numbers",
        "shard": {
            "index": shard_index,
            "count": shard_count,
            "start_in_train_valid_sequence": start,
            "stop_in_train_valid_sequence": stop,
            "source_reaction_count": len(selected),
            "retained_reaction_count": len(output_records),
            "excluded_reaction_count": len(exclusions),
        },
        "input_fingerprints": {"source_csv": source_fp, "metadata_csv": metadata_fp},
        "rxnmapper": asset_record,
        "settings": {
            "mapped_splits": list(ALLOWED_MAPPING_SPLITS),
            "batch_size": batch_size,
            "canonicalize": True,
            "atom_map_transfer": "RXNMapper indices onto exact original isomeric graphs",
            "unbalanced_reaction_policy": "exclude with immutable per-shard audit",
            "random_seed": 2026,
            "model_eval_mode": True,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "rdkit": importlib.metadata.version("rdkit"),
            "rxnmapper": getattr(rxnmapper_package, "__version__", "source-checkout"),
        },
        "validation": {
            "chemically_identical": len(output_records),
            "failures": 0,
            "excluded_unbalanced_reactions": len(exclusions),
            "stereochemistry_restored_reactions": stereo_restored_reactions,
            "unmapped_reactant_atoms": unmapped_reactant_atoms,
        },
        "exclusions": fingerprint(exclusion_path),
        "output": fingerprint(data_path),
        "runtime_seconds": time.perf_counter() - start_time,
        "created_at_utc": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def compile_mapping(
    *,
    source_csv: str | Path,
    metadata_csv: str | Path,
    mapping_root: str | Path,
    shard_count: int,
    dataset_dir: str | Path,
) -> dict[str, Any]:
    final_dir = Path(dataset_dir).resolve()
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite LocalRetro dataset: {final_dir}")
    rows = [
        row
        for row in _load_current_rows(source_csv, metadata_csv)
        if row["split"] in ALLOWED_MAPPING_SPLITS
    ]
    mapped_by_id: dict[int, dict[str, Any]] = {}
    excluded_by_id: dict[int, dict[str, Any]] = {}
    shard_manifests: list[dict[str, Any]] = []
    root = Path(mapping_root).resolve()
    for shard_index in range(shard_count):
        data_path, manifest_path = _mapped_paths(root, shard_index, shard_count)
        if not data_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing mapping shard {shard_index}/{shard_count}.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError("Mapping shard protocol differs.")
        if manifest["output"]["sha256"] != sha256_file(data_path):
            raise RuntimeError("Mapping shard checksum differs.")
        exclusion_record = manifest.get("exclusions")
        if not exclusion_record:
            raise RuntimeError("Mapping shard lacks the required exclusion audit.")
        exclusion_path = Path(exclusion_record["path"])
        if not exclusion_path.is_file() or exclusion_record["sha256"] != sha256_file(
            exclusion_path
        ):
            raise RuntimeError("Mapping shard exclusion audit checksum differs.")
        exclusion_payload = json.loads(exclusion_path.read_text(encoding="utf-8"))
        if exclusion_payload.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError("Mapping exclusion protocol differs.")
        for exclusion in exclusion_payload.get("exclusions", []):
            reaction_id = int(exclusion["reaction_id"])
            if reaction_id in excluded_by_id or reaction_id in mapped_by_id:
                raise RuntimeError(f"Duplicate excluded reaction ID {reaction_id}.")
            if exclusion.get("reason") != "product_atoms_absent_from_reactants":
                raise RuntimeError("Mapping shard has an unapproved exclusion reason.")
            if int(exclusion.get("missing_product_atom_count", 0)) < 1:
                raise RuntimeError("Mapping exclusion lacks a positive missing-atom count.")
            excluded_by_id[reaction_id] = exclusion
        shard_manifests.append(fingerprint(manifest_path))
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                reaction_id = int(record["reaction_id"])
                if reaction_id in mapped_by_id:
                    raise RuntimeError(f"Duplicate mapped reaction ID {reaction_id}.")
                validate_mapped_reaction(
                    record["unmapped_reaction"], record["mapped_reaction"]
                )
                mapped_by_id[reaction_id] = record
    expected_ids = {int(row["reaction_id"]) for row in rows}
    observed_ids = set(mapped_by_id) | set(excluded_by_id)
    if set(mapped_by_id) & set(excluded_by_id):
        raise RuntimeError("Mapped and excluded reaction IDs overlap.")
    if observed_ids != expected_ids:
        raise RuntimeError(
            f"Mapped/excluded ID coverage differs: missing={len(expected_ids-observed_ids)}, "
            f"extra={len(observed_ids-expected_ids)}."
        )

    staging = final_dir.with_name(f".{final_dir.name}.{os.getpid()}.tmp")
    staging.mkdir(parents=True)
    try:
        counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        excluded_counts: dict[str, int] = {}
        for split in ALLOWED_MAPPING_SPLITS:
            split_rows = [
                mapped_by_id[int(row["reaction_id"])]
                for row in rows
                if row["split"] == split
                and int(row["reaction_id"]) in mapped_by_id
            ]
            frame = pd.DataFrame(
                {
                    "id": [str(row["reaction_id"]) for row in split_rows],
                    "class": ["UNK"] * len(split_rows),
                    "reactants>reagents>production": [
                        row["mapped_reaction"] for row in split_rows
                    ],
                }
            )
            target_name = "raw_val.csv" if split == "valid" else "raw_train.csv"
            frame.to_csv(staging / target_name, index=False, lineterminator="\n")
            counts[split] = len(frame)
            source_counts[split] = sum(row["split"] == split for row in rows)
            excluded_counts[split] = sum(
                exclusion["split"] == split for exclusion in excluded_by_id.values()
            )
        combined_exclusions = sorted(
            excluded_by_id.values(), key=lambda value: int(value["reaction_id"])
        )
        atomic_json(
            staging / "mapping_exclusions.json",
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "exclusion_rule": "product atom-map index absent from all reactants",
                "counts_by_split": excluded_counts,
                "total": len(combined_exclusions),
                "exclusions": combined_exclusions,
            },
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "comparator": "current Chemformer USPTO-50K train/valid split",
            "single_intended_change": "add one consistent RXNMapper mapping for LocalRetro",
            "counts": {
                **counts,
                "total": sum(counts.values()),
                "source_train_valid_total": sum(source_counts.values()),
                "excluded_train": excluded_counts["train"],
                "excluded_valid": excluded_counts["valid"],
                "excluded_total": len(combined_exclusions),
                "test_rows_loaded": 0,
            },
            "source_counts": source_counts,
            "exclusion_policy": (
                "preprocessing-only removal of chemically unbalanced source reactions; "
                "all IDs and missing-product-atom counts are retained"
            ),
            "test_partition_policy": (
                "no raw_test.csv is created; product-only inference is released after checkpoint freeze"
            ),
            "inputs": {
                "source_csv": fingerprint(source_csv),
                "metadata_csv": fingerprint(metadata_csv),
                "mapping_shard_manifests": shard_manifests,
            },
            "outputs": {
                "raw_train": fingerprint(staging / "raw_train.csv"),
                "raw_val": fingerprint(staging / "raw_val.csv"),
                "mapping_exclusions": fingerprint(staging / "mapping_exclusions.json"),
            },
            "created_at_utc": utc_now(),
        }
        atomic_json(staging / "mapping_manifest.json", manifest)
        os.replace(staging, final_dir)
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def prepare_product_only_inference(
    *,
    anchored_pool: str | Path,
    checkpoint_freeze: str | Path,
    dataset_dir: str | Path,
    inventory_output: str | Path,
) -> dict[str, Any]:
    freeze_path = Path(checkpoint_freeze).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("protocol_id") != DOWNSTREAM_PROTOCOL_ID:
        raise RuntimeError("LocalRetro checkpoint freeze has the wrong protocol.")
    if freeze.get("test_partition_loaded") is not False:
        raise RuntimeError("Checkpoint freeze is not a pre-test freeze.")
    dataset = Path(dataset_dir).resolve()
    raw_test = dataset / "raw_test.csv"
    inventory = Path(inventory_output).resolve()
    if raw_test.exists() or inventory.exists():
        raise FileExistsError("Refusing to overwrite product-only inference input.")

    products: list[str] = []
    seen: set[tuple[str, ...]] = set()
    with Path(anchored_pool).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            product = str(record["product"])
            key = canonical_fragments(product)
            if key in seen:
                continue
            seen.add(key)
            products.append(product)
    if not products:
        raise RuntimeError("Anchored pool contains no products.")
    frame = pd.DataFrame(
        {
            "id": [str(index) for index in range(len(products))],
            "class": ["UNK"] * len(products),
            "reactants>reagents>production": [
                f"{product}>>{product}" for product in products
            ],
        }
    )
    raw_test.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_test.with_name(f".{raw_test.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, raw_test)
    atomic_jsonl(
        inventory,
        (
            {
                "test_id": index,
                "product": product,
                "canonical_product": ".".join(canonical_fragments(product)),
            }
            for index, product in enumerate(products)
        ),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": DOWNSTREAM_PROTOCOL_ID,
        "checkpoint_freeze": fingerprint(freeze_path),
        "anchored_pool": fingerprint(anchored_pool),
        "product_count": len(products),
        "ground_truth_reactants_in_inference_file": False,
        "outputs": {"raw_test": fingerprint(raw_test), "inventory": fingerprint(inventory)},
        "created_at_utc": utc_now(),
    }
    atomic_json(dataset / "inference_input_manifest.json", result)
    return result


def crosswalk_audit(
    *, source_csv: str | Path, gln_root: str | Path, output: str | Path
) -> dict[str, Any]:
    current = pd.read_csv(source_csv)
    current_index: dict[Any, list[tuple[int, str]]] = collections.defaultdict(list)
    current_rows: list[tuple[int, str, Any]] = []
    for index, row in current.iterrows():
        key = (
            canonical_fragments(str(row["products_smiles"])),
            canonical_fragments(str(row["reactants_smiles"])),
        )
        split = str(row["set"])
        current_index[key].append((index, split))
        current_rows.append((index, split, key))
    mapped_index: dict[Any, list[tuple[str, int]]] = collections.defaultdict(list)
    gln_files: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        path = Path(gln_root) / f"schneider50k_{split}.csv"
        gln_files[split] = fingerprint(path)
        frame = pd.read_csv(path)
        for row_index, reaction in enumerate(
            frame["reactants>reagents>production"].astype(str)
        ):
            mapped_index[reaction_identity(reaction)].append((split, row_index))
    status: collections.Counter[str] = collections.Counter()
    by_split: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for _index, split, key in current_rows:
        matches = mapped_index.get(key, [])
        if not matches:
            label = "missing"
        elif len(matches) > 1:
            label = "ambiguous_mapped"
        elif len(current_index[key]) > 1:
            label = "ambiguous_current"
        elif matches[0][0] == split:
            label = "unique_same_split"
        else:
            label = "unique_cross_split"
        status[label] += 1
        by_split[split][label] += 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "fail-closed audit; GLN mappings are not used to train the current benchmark",
        "decision": (
            "use one pinned RXNMapper over all current train+valid reactions because the crosswalk is incomplete"
        ),
        "current": {
            "fingerprint": fingerprint(source_csv),
            "rows": len(current_rows),
            "identities": len(current_index),
            "duplicate_identities": sum(len(value) > 1 for value in current_index.values()),
        },
        "gln": {
            "files": gln_files,
            "rows": sum(len(value) for value in mapped_index.values()),
            "identities": len(mapped_index),
            "duplicate_identities": sum(len(value) > 1 for value in mapped_index.values()),
        },
        "crosswalk": {
            "status_counts": dict(status),
            "by_current_split": {key: dict(value) for key, value in by_split.items()},
            "safe_for_direct_mapping_transfer": False,
        },
        "created_at_utc": utc_now(),
    }
    atomic_json(Path(output).resolve(), report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("crosswalk-audit")
    audit.add_argument("--source-csv", required=True)
    audit.add_argument("--gln-root", required=True)
    audit.add_argument("--output", required=True)

    mapping = subparsers.add_parser("map-shard")
    mapping.add_argument("--source-csv", required=True)
    mapping.add_argument("--metadata-csv", required=True)
    mapping.add_argument("--rxnmapper-root", required=True)
    mapping.add_argument("--output-root", required=True)
    mapping.add_argument("--shard-index", type=int, required=True)
    mapping.add_argument("--shard-count", type=int, required=True)
    mapping.add_argument("--batch-size", type=int, default=16)

    compile_parser = subparsers.add_parser("compile-mapping")
    compile_parser.add_argument("--source-csv", required=True)
    compile_parser.add_argument("--metadata-csv", required=True)
    compile_parser.add_argument("--mapping-root", required=True)
    compile_parser.add_argument("--shard-count", type=int, required=True)
    compile_parser.add_argument("--dataset-dir", required=True)

    inference = subparsers.add_parser("prepare-inference")
    inference.add_argument("--anchored-pool", required=True)
    inference.add_argument("--checkpoint-freeze", required=True)
    inference.add_argument("--dataset-dir", required=True)
    inference.add_argument("--inventory-output", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "crosswalk-audit":
        result = crosswalk_audit(
            source_csv=args.source_csv, gln_root=args.gln_root, output=args.output
        )
    elif args.command == "map-shard":
        result = map_shard(
            source_csv=args.source_csv,
            metadata_csv=args.metadata_csv,
            rxnmapper_root=args.rxnmapper_root,
            output_root=args.output_root,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            batch_size=args.batch_size,
        )
    elif args.command == "compile-mapping":
        result = compile_mapping(
            source_csv=args.source_csv,
            metadata_csv=args.metadata_csv,
            mapping_root=args.mapping_root,
            shard_count=args.shard_count,
            dataset_dir=args.dataset_dir,
        )
    else:
        result = prepare_product_only_inference(
            anchored_pool=args.anchored_pool,
            checkpoint_freeze=args.checkpoint_freeze,
            dataset_dir=args.dataset_dir,
            inventory_output=args.inventory_output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
