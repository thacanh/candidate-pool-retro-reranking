#!/usr/bin/env python
"""Apply the one official Chemformer checkpoint migration, fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_ID = "f1-chemformer-checkpoint-migration-v1"
EXPECTED_SOURCE_SIZE = 537_441_893
EXPECTED_SOURCE_SHA256 = (
    "44203e603e0ed9919213fdd822cb0bff844bd9fbae6f5f5882e1771046f0b287"
)
EXPECTED_VOCABULARY_SIZE = 523


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict) -> str:
    """Hash tensor names, shapes, dtypes and exact CPU bytes deterministically."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_existing(source: Path, output: Path, manifest_path: Path) -> dict:
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if record.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Existing migrated checkpoint has the wrong protocol.")
    if record.get("source", {}).get("sha256") != sha256(source):
        raise RuntimeError("Existing migration manifest belongs to another source.")
    if record.get("output", {}).get("sha256") != sha256(output):
        raise RuntimeError("Existing migrated checkpoint was modified.")
    return record


def migrate(source: Path, output: Path, manifest_path: Path) -> dict:
    if output.exists() or manifest_path.exists():
        if output.is_file() and manifest_path.is_file():
            return validate_existing(source, output, manifest_path)
        raise FileExistsError("Partial checkpoint migration exists; preserve it for audit.")
    if source.stat().st_size != EXPECTED_SOURCE_SIZE or sha256(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Source checkpoint differs from the pinned Figshare asset.")

    # The pinned file is an official trusted PyTorch-Lightning checkpoint.  Its
    # hyperparameters contain legacy molbart classes, hence weights_only=False
    # and the pinned Chemformer-1.0 source on PYTHONPATH are both required.
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    hyperparameters = checkpoint.get("hyper_parameters")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(hyperparameters, dict) or not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint lacks Lightning hyperparameters/state_dict.")
    if hyperparameters.get("vocab_size") != EXPECTED_VOCABULARY_SIZE:
        raise RuntimeError("Legacy checkpoint vocab_size is not the pinned 523 tokens.")
    if "vocabulary_size" in hyperparameters:
        raise RuntimeError("Source checkpoint is already migrated or ambiguous.")

    tensor_hash_before = state_dict_sha256(state_dict)
    hyperparameters["vocabulary_size"] = hyperparameters.pop("vocab_size")
    tensor_hash_after = state_dict_sha256(state_dict)
    if tensor_hash_before != tensor_hash_after:
        raise RuntimeError("State tensors changed during metadata-only migration.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)
    output_hash = sha256(output)
    record = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "single_intended_change": (
            "rename hyper_parameters.vocab_size to vocabulary_size exactly as "
            "specified by the official Chemformer README"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source.resolve()),
            "size_bytes": source.stat().st_size,
            "sha256": EXPECTED_SOURCE_SHA256,
        },
        "output": {
            "path": str(output.resolve()),
            "size_bytes": output.stat().st_size,
            "sha256": output_hash,
        },
        "checkpoint_metadata": {
            "pytorch_lightning_version": checkpoint.get("pytorch-lightning_version"),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "state_tensor_count": len(state_dict),
            "state_dict_logical_sha256_before": tensor_hash_before,
            "state_dict_logical_sha256_after": tensor_hash_after,
            "vocabulary_size": hyperparameters["vocabulary_size"],
            "removed_hyperparameter": "vocab_size",
            "added_hyperparameter": "vocabulary_size",
        },
    }
    atomic_json(manifest_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--legacy-source", required=True)
    args = parser.parse_args()
    legacy_source = str(Path(args.legacy_source).resolve())
    if legacy_source not in sys.path:
        sys.path.insert(0, legacy_source)

    global torch
    import torch

    record = migrate(
        Path(args.source).resolve(),
        Path(args.output).resolve(),
        Path(args.manifest).resolve(),
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
