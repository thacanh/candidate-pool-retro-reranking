#!/usr/bin/env python
"""Validate collected seed folders and build one compact run index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rerank.data.build_conformer_sqlite import ALLOWED_CONFORMER_SEEDS


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(seed_root: Path) -> dict:
    checksum_path = seed_root / "checksums.sha256"
    if not checksum_path.is_file():
        raise RuntimeError(f"Missing checksum manifest: {checksum_path}")
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = (seed_root / Path(relative)).resolve()
        try:
            path.relative_to(seed_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Checksum path escaped seed folder: {relative}") from exc
        if not path.is_file():
            raise RuntimeError(f"Missing retained artifact: {path}")
        observed = _sha256(path)
        if observed != digest:
            raise RuntimeError(
                f"Checksum mismatch for {path}: {observed} != {digest}."
            )
        checked += 1
    return {"checked_files": checked, "passed": True}


def collect_runs(root: str | Path, seeds=ALLOWED_CONFORMER_SEEDS) -> tuple[list[dict], dict]:
    root = Path(root).resolve()
    rows = []
    missing = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        completed_path = seed_root / "COMPLETED.json"
        manifest_path = seed_root / "manifest.json"
        summary_path = seed_root / "result_summary.json"
        if not all(path.is_file() for path in (completed_path, manifest_path, summary_path)):
            missing.append(seed)
            continue
        checksum_validation = verify_checksums(seed_root)
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(completed.get("seed", -1)) != seed:
            raise RuntimeError(f"Seed identity mismatch in {completed_path}.")
        if int(manifest.get("conformer_seed", -1)) != seed:
            raise RuntimeError(f"Seed identity mismatch in {manifest_path}.")
        rows.append(
            {
                "conformer_seed": seed,
                "conformer_label": summary["conformer_label"],
                "confirmatory_role": summary["confirmatory_role"],
                "top1_mean": summary["top1_mean"],
                "top1_std": summary["top1_std"],
                "mrr_mean": summary["mrr_mean"],
                "mrr_std": summary["mrr_std"],
                "runtime_seconds": completed["runtime_seconds"],
                "large_atom_cache_retained": completed[
                    "large_atom_cache_retained"
                ],
                "retained_file_count": completed["retained_file_count"],
                "checksum_files_verified": checksum_validation["checked_files"],
                "seed_folder": str(seed_root),
            }
        )
    audit = {
        "requested_seeds": list(seeds),
        "complete_seeds": [row["conformer_seed"] for row in rows],
        "missing_seeds": missing,
        "all_complete": not missing,
    }
    if missing:
        raise RuntimeError(
            "Conformer collection is incomplete; missing validated folders for seeds "
            + ", ".join(map(str, missing))
        )
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="outputs/jcheminform_revision/conformers"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/jcheminform_revision/conformer_aggregate",
    )
    args = parser.parse_args()
    rows, audit = collect_runs(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with open(
        output / "conformer_run_index.csv", "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit": audit,
        "runs": rows,
        "next_stage": (
            "B1 crossed feature stability, B2 5x5 variance decomposition, "
            "and B3 ten-conformer scalar averaging"
        ),
    }
    (output / "conformer_run_index.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
