"""Fail-closed machine and bundle validation for the AiZynth one-pass job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


EXPECTED_PACKAGES = {
    "aizynthfinder": "4.4.1",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "rdkit": "2023.9.6",
    "onnxruntime": "1.23.2",
    "reaction-utils": "1.9.3",
    "networkx": "2.8.8",
    "dask": "2024.12.1",
    "distributed": "2024.12.1",
    "tables": "3.10.1",
    "tqdm": "4.67.1",
    "psutil": "7.0.0",
}

EXPECTED_INPUTS = {
    "modelchem/config.yml": "8d04a70d140725bfc40841d2b34a69418c5099ffa2dc286e1721504259f2e2e3",
    "modelchem/uspto_model.onnx": "bd0a3cb74cd7068de474c8fb789a00a66bc42c75636d66510ccac585ebe928f8",
    "modelchem/uspto_templates.csv.gz": "a4f1945e90cfa195538320833d68aed38f14e2fcc2f8afb5d958bc920edcafbe",
    "modelchem/uspto_ringbreaker_model.onnx": "1bf0690352d9e9212d7dbe8b35649caf74f73ef0b30edefdfdac37fce38085be",
    "modelchem/uspto_ringbreaker_templates.csv.gz": "5616a056454b10a2f044e69e027422128986856ebd958541a3bf9f837e3a0d14",
    "modelchem/uspto_filter_model.onnx": "ad29aa32bdfcbe37065045546493806cf04899c55386c438905d83fb14bb6320",
    "data/uspto_smiles.csv": "688c5b8ea7c3269b53ae15ffca9ec98f51fd29ea3fc25edc8fa66cabe9042d6a",
    "outputs/rerank_dataset.jsonl": "9ec1cf192c49eeb7d74a320dd721287fabdef9863cc06f95d0f13baab8c3ff85",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_bundle_manifest(root: Path) -> int:
    manifest_path = root / "BUNDLE_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("BUNDLE_MANIFEST.json is missing; do not run an unpackaged workspace.")
    # Windows PowerShell 5 writes ``-Encoding utf8`` with a BOM. Accepting
    # ``utf-8-sig`` keeps the JSON payload identical while remaining strict
    # about every bundled file's recorded size and SHA-256 below.
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Bundle manifest has no file list.")
    for entry in files:
        relative = entry["path"]
        target = root / relative
        if not target.is_file():
            raise RuntimeError(f"Bundle file missing: {relative}")
        if target.stat().st_size != int(entry["size_bytes"]):
            raise RuntimeError(f"Bundle file size differs: {relative}")
        if sha256_file(target) != entry["sha256"]:
            raise RuntimeError(f"Bundle file hash differs: {relative}")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.bundle_root).resolve()

    if sys.version_info[:3] != (3, 10, 14):
        raise RuntimeError(f"Expected Python 3.10.14, found {sys.version.split()[0]}.")

    resolved_packages: dict[str, str] = {}
    for package, expected in EXPECTED_PACKAGES.items():
        actual = importlib.metadata.version(package)
        resolved_packages[package] = actual
        if actual != expected:
            raise RuntimeError(f"Package mismatch for {package}: {actual} != {expected}")

    for relative, expected_sha in EXPECTED_INPUTS.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"Pinned input differs or is missing: {relative}")

    manifest_count = verify_bundle_manifest(root)
    ram_bytes = int(psutil.virtual_memory().total)
    free_bytes = int(shutil.disk_usage(root).free)
    if ram_bytes < 12 * 1024**3:
        raise RuntimeError("At least 12 GiB host RAM is required for the conservative worker plan.")
    if free_bytes < 15 * 1024**3:
        raise RuntimeError("At least 15 GiB free disk is required before candidate generation.")

    # Model/config initialization only: no product is submitted to the policy.
    from aizynthfinder.context.config import Configuration

    config = Configuration.from_file(str(root / "modelchem/config.yml"))
    config.expansion_policy.select_all()

    from recommend_workers import available_cpu_count, recommend_workers

    result = {
        "status": "pass",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": "A-CAP50-ONEPASS-v1",
        "bundle_manifest_files_verified": manifest_count,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": resolved_packages,
        "linux_lock": {
            "path": "aizynth_jobs/requirements-aizynth-linux-py310.lock",
            "sha256": sha256_file(root / "aizynth_jobs/requirements-aizynth-linux-py310.lock"),
        },
        "host_ram_bytes": ram_bytes,
        "free_disk_bytes": free_bytes,
        "available_logical_cpus": available_cpu_count(),
        "recommended_workers": recommend_workers(),
        "model_initialization": "pass; no scientific product submitted",
        "filter_policy_called": False,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
