#!/usr/bin/env python
"""Fail-closed F1 environment, source, input and asset preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_ID = "f1-chemformer-forward-roundtrip-v1"
EXPECTED_REPOSITORIES = {
    "assets/chemformer_official": "53a2819076bd16f36131839c7fb88157cfc2ce92",
    "assets/chemformer_checkpoint_source": "0ca3c1b5f810a0ff106bbb846f511629eac3b4a5",
    "assets/aizynthmodels_official": "f9262c2c44720a194ae67a37c35b411b20aec2c8",
}
EXPECTED_FILES = {
    "assets/chemformer_forward/last.ckpt": (
        537_441_893,
        "44203e603e0ed9919213fdd822cb0bff844bd9fbae6f5f5882e1771046f0b287",
    ),
    "assets/chemformer_official/bart_vocab_downstream.json": (
        16_588,
        "39bd8a0343cd88aeeb8b23e74189bc14b994f1461ad2979d8fcb41ef05068cd3",
    ),
    "assets/chemformer_checkpoint_source/bart_vocab_downstream.txt": (
        5_949,
        "faf1d7cec83558a4aa4065b1b4e89847b0d5a83724f2218e52f5bee4442969fb",
    ),
}
EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "pandas": "1.4.2",
    "pytorch-lightning": "2.2.1",
    "torchmetrics": "1.3.1",
    "rdkit": "2022.9.3",
    "reaction-utils": "1.5.0",
    "scikit-learn": "1.4.2",
    "scipy": "1.12.0",
    "setuptools": "80.9.0",
    "hydra-core": "1.3.2",
    "omegaconf": "2.3.0",
    "psutil": "7.0.0",
    "tqdm": "4.66.5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_repository(root: Path, relative: str, expected_commit: str) -> dict:
    repository = root / relative
    marker = repository / ".pinned_revision.json"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        commit = payload.get("commit")
        if commit != expected_commit:
            raise RuntimeError(f"Pinned source marker differs: {relative}")
        return {"path": relative, "commit": commit, "verification": "bundle SHA-256 manifest"}
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-c", "core.autocrlf=true", "-c", "core.filemode=false", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit or dirty:
        raise RuntimeError(f"Official source is not the clean pinned revision: {relative}")
    return {"path": relative, "commit": commit, "verification": "clean git checkout"}


def verify_bundle_manifest(root: Path) -> int | None:
    path = root / "CHEMFORMER_BUNDLE_MANIFEST.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    verified = 0
    for row in manifest.get("files", []):
        candidate = (root / row["path"]).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise RuntimeError(f"Unsafe/missing bundle path: {row['path']}")
        if candidate.stat().st_size != int(row["size_bytes"]) or sha256(candidate) != row["sha256"]:
            raise RuntimeError(f"Bundle file differs: {row['path']}")
        verified += 1
    if verified == 0:
        raise RuntimeError("Chemformer bundle manifest has no files.")
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    root = Path(args.bundle_root).resolve()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(f"F1 requires Python 3.10, found {sys.version}.")

    files = {}
    for relative, (size, digest) in EXPECTED_FILES.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
            raise RuntimeError(f"Pinned F1 file differs or is missing: {relative}")
        files[relative] = {"size_bytes": size, "sha256": digest}
    repositories = {
        relative: verify_repository(root, relative, commit)
        for relative, commit in EXPECTED_REPOSITORIES.items()
    }

    prepared_manifest_path = root / "outputs/jcheminform_revision/f1_roundtrip/prepared/prepare_manifest.json"
    prepared = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    if (
        prepared.get("protocol_id") != PROTOCOL_ID
        or prepared.get("unique_forward_inputs") != 4_762
        or prepared.get("covered_reactions") != 3_985
        or prepared.get("test_partition_used_for_training_or_selection") is not False
    ):
        raise RuntimeError("Prepared F1 manifest is not the frozen approved input.")
    prepared_files = {}
    for key, fingerprint in prepared["outputs"].items():
        name = {"usage": "frozen_top1_usages.csv", "inference_map": "inference_map.csv", "chemformer_input": "chemformer_input.tsv"}[key]
        path = prepared_manifest_path.parent / name
        actual = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        if (
            actual["size_bytes"] != int(fingerprint["size_bytes"])
            or actual["sha256"] != str(fingerprint["sha256"])
        ):
            raise RuntimeError(f"Prepared F1 artifact differs: {key}")
        prepared_files[key] = actual

    installed = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if installed != EXPECTED_VERSIONS:
        raise RuntimeError(f"Chemformer environment version mismatch: {installed!r}")
    import psutil
    import torch
    if torch.__version__ != "2.2.2+cu121":
        raise RuntimeError(f"Expected torch 2.2.2+cu121, found {torch.__version__}.")
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the F1 retained run.")
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        if args.require_cuda and properties.total_memory < 8 * 1024**3:
            raise RuntimeError("F1 retained GPU run requires at least 8 GiB VRAM.")
        gpu = {
            "name": properties.name,
            "vram_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "torch_cuda": torch.version.cuda,
        }

    legacy_tokens = [line.strip() for line in (root / "assets/chemformer_checkpoint_source/bart_vocab_downstream.txt").read_text().splitlines() if line.strip()]
    modern_tokens = json.loads((root / "assets/chemformer_official/bart_vocab_downstream.json").read_text())["vocabulary"]
    if legacy_tokens != modern_tokens or len(modern_tokens) != 523:
        raise RuntimeError("Legacy and AiZynthModels vocabularies are not index-equivalent.")

    payload = {
        "status": "pass",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "bundle_manifest_files_verified": verify_bundle_manifest(root),
        "repositories": repositories,
        "files": files,
        "prepared": prepared_files,
        "environment": {"python": sys.version, "torch": torch.__version__, "packages": installed},
        "environment_lock": {
            "path": "chemformer_jobs/linux/requirements-chemformer-linux-py310.lock",
            "sha256": sha256(root / "chemformer_jobs/linux/requirements-chemformer-linux-py310.lock"),
            "artifact_hashes_required": True,
        },
        "gpu": gpu,
        "host_ram_bytes": psutil.virtual_memory().total,
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "vocabulary_index_equivalence": "523/523",
        "test_partition_used_for_training_or_selection": False,
    }
    atomic_json(root / "logs/chemformer/MACHINE_CHECK.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
