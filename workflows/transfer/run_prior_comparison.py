"""N3 stage-boundary restart, preserving and verifying every completed archive."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from archive_results import archive_stage, verify_stage, _sha256
from package_inputs import PRIMARY, FEATURES, K1, POOLS, INDEX, COMMON_COHORT

PROTOCOL = "dd-n3-pool-specific-selection-v1"


def run_stage(stage, result, archives, approval, log, required, command):
    result, archives, log = Path(result), Path(archives), Path(log)
    receipt_path = archives / f"{stage}.receipt.json"
    if receipt_path.exists():
        verify_stage(SimpleNamespace(receipt=str(receipt_path)))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["stage"] != stage or receipt["protocol_id"] != PROTOCOL:
            raise ValueError("Wrong completed-stage receipt")
        prefix = f"{stage}/results/"
        expected = {name[len(prefix):]: record for name, record in receipt["members"].items()
                    if name.startswith(prefix)}
        actual = {p.relative_to(result).as_posix() for p in result.rglob("*") if p.is_file()}
        if actual != set(expected):
            raise ValueError(f"Local completed stage membership changed: {stage}")
        for name, record in expected.items():
            path = result / name
            if path.is_symlink() or _sha256(path) != record["sha256"]:
                raise ValueError(f"Completed stage changed: {path}")
        print(f"{stage}: existing archive AND local results verified; skipping", flush=True)
        return
    manifest = result / "manifest.json"
    if not manifest.exists():
        if result.exists() and any(result.iterdir()):
            raise RuntimeError(f"{stage} is partial, not complete. Preserved for audit; no automatic overwrite.")
        log.parent.mkdir(parents=True, exist_ok=True)
        # Preserve failed logs even when no scientific output was written.
        with log.open("a", encoding="utf-8") as handle:
            handle.write("\n--- STAGE INVOCATION ---\n")
            handle.flush()
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    archive_stage(SimpleNamespace(stage=stage, protocol_id=PROTOCOL,
        result_dir=str(result), archive_dir=str(archives), approval_record=str(approval),
        required=[required], log_file=str(log) if log.exists() else None))
    verify_stage(SimpleNamespace(receipt=str(receipt_path)))
    print(f"{stage}: complete, archived and verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--archives", default="/workspace/dd_n3_archives")
    parser.add_argument("--approval", default="docs/dd_remote_approval.json")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    py, g1 = sys.executable, f"{PRIMARY}/g1_20seed"
    s1, s2, s3 = (f"outputs/dd_final_round/N3_S{i}" for i in (1, 2, 3))
    common = ["--approval-record", args.approval, "--k1-root", K1,
              "--feature-root", FEATURES, "--products-jsonl", f"{POOLS}/products.jsonl",
              "--source-csv", "data/uspto_smiles.csv", "--metadata-csv", "data/uspto_reaction_metadata.csv",
              "--device", "cpu"]
    diagnostic = [py, "-m", "rerank.experiments.run_prior_diagnostics", *common,
        "--historical-freeze", f"{g1}/freeze.json", "--historical-model-root", g1,
        "--historical-cache", "outputs/jcheminform_revision/conformers/seed_42/features/official_3d_prior_schema1.pkl",
        "--historical-selection-bundle", f"{PRIMARY}/selection/train_valid_only.pkl",
        "--historical-prediction-root", f"{g1}/test_results/predictions", "--output-dir", s1]
    fitting = [*common, "--pool-root", POOLS, "--index-root", INDEX,
        "--primary-freeze", f"{PRIMARY}/freeze/model_selection.json", "--s2-root", s2,
        "--workers", str(args.workers)]
    commands = [diagnostic, [py, "workflows/transfer/run_prior_selection.py", "s2", *fitting],
        [py, "workflows/transfer/run_prior_selection.py", "s3", *fitting, "--s3-root", s3,
         "--historical-prediction-root", f"{g1}/test_results/predictions",
         "--common-reactions-csv", COMMON_COHORT]]
    for i, (result, required, command) in enumerate(zip((s1, s2, s3),
            ("manifest.json", "choice_freeze.json", "transfer_effects.csv"), commands), 1):
        stage = f"N3_S{i}"
        print(f"BEGIN {stage}", flush=True)
        run_stage(stage, result, args.archives, args.approval,
                  f"logs/dd_final_round/{stage}.log", required, command)
    print(f"N3 COMPLETE: {args.archives}/N3_S1.zip, N3_S2.zip, N3_S3.zip", flush=True)


if __name__ == "__main__":
    main()
