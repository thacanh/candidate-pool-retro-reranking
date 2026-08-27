#!/usr/bin/env python
"""Run once: the shared trained 2D comparator for all conformer replicates."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (42, 43, 44, 45, 46)
PROTOCOL_ID = "legacy-cap10-fixed50-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_existing(root: Path) -> bool:
    completed_path = root / "COMPLETED.json"
    if not completed_path.is_file():
        return False
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    if completed.get("status") != "complete":
        raise RuntimeError("Existing shared 2D folder is not marked complete.")
    if completed.get("manifest_sha256") != sha256(root / "manifest.json"):
        raise RuntimeError("Existing shared 2D manifest checksum failed.")
    for line in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = root / relative
        if not target.is_file() or sha256(target) != digest:
            raise RuntimeError(f"Existing shared 2D checksum failed: {relative}")
    return True


def validate_ranking(ranking: Path) -> dict:
    inner = json.loads((ranking / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((ranking / "per_seed_metrics.json").read_text(encoding="utf-8"))
    if inner.get("feature_mode") != "2d+prior" or inner.get("seeds") != list(SEEDS):
        raise RuntimeError("2D ranking manifest has the wrong arm or seeds.")
    if set(metrics) != {str(seed) for seed in SEEDS}:
        raise RuntimeError("2D metrics are not exactly seeds 42--46.")
    for seed in SEEDS:
        cell = metrics[str(seed)]
        if int(cell.get("n_eval_reactions", -1)) != 3985:
            raise RuntimeError(f"Seed {seed} does not contain 3,985 covered reactions.")
        prediction = ranking / f"eval_seed{seed}.csv"
        with open(prediction, encoding="utf-8", newline="") as handle:
            rows = sum(1 for _ in csv.DictReader(handle))
        if rows != 3985:
            raise RuntimeError(f"Seed {seed} prediction CSV has {rows} rows, expected 3985.")
    return inner


def main() -> None:
    root = Path("outputs/jcheminform_revision/shared_2d_legacy_fixed50").resolve()
    if root.exists() and verify_existing(root):
        print(f"Already complete and checksummed: {root}")
        return
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"Partial shared 2D folder preserved at {root}; inspect or move it before rerunning."
        )
    root.mkdir(parents=True, exist_ok=True)
    ranking = root / "ranking"
    features = root / "features"
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    command = [
        sys.executable,
        "-m",
        "rerank.experiments.run_controlled_study",
        "--candidate-jsonl", "outputs/rerank_dataset.jsonl",
        "--source-csv", "data/uspto_smiles.csv",
        "--metadata-csv", "data/uspto_reaction_metadata.csv",
        "--feature-mode", "2d+prior",
        "--seeds", ",".join(map(str, SEEDS)),
        "--device", "cpu",
        "--output-dir", str(ranking),
        "--feature-cache-dir", str(features),
    ]
    subprocess.run(command, check=True)
    inner = validate_ranking(ranking)
    manifest = {
        "schema_version": 1,
        "workstream": "WS-B shared 2D comparator",
        "protocol_id": PROTOCOL_ID,
        "comparator": "prior-sorted AiZynthFinder candidate order",
        "single_intended_change": "train the four-input 2D-only RankerMLP",
        "feature_mode": "2d+prior",
        "training_seeds": list(SEEDS),
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - start_clock,
        "device": "cpu",
        "input_fingerprints": inner.get("input_fingerprints"),
        "frozen_config": inner.get("frozen_config"),
        "ranking_manifest_sha256": sha256(ranking / "manifest.json"),
    }
    atomic_json(root / "manifest.json", manifest)
    retained = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"COMPLETED.json", "checksums.sha256"}
    )
    with open(root / "checksums.sha256", "w", encoding="utf-8", newline="\n") as handle:
        for path in retained:
            handle.write(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")
    atomic_json(root / "COMPLETED.json", {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": sha256(root / "manifest.json"),
    })
    print(f"Shared trained 2D comparator complete: {root}")


if __name__ == "__main__":
    main()
