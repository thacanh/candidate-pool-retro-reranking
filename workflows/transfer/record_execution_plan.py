"""Print a hash-bound N2/N3/N4 sign-off request; never approves it."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


RUNNERS = {
    "dd-n2-cross-inference-v1": ["src/rerank/experiments/run_cross_inference.py"],
    "dd-n3-pool-specific-selection-v1": [
        "src/rerank/experiments/run_prior_diagnostics.py",
        "src/rerank/experiments/run_prior_sensitivity.py",
        "src/rerank/experiments/select_prior_transform.py",
        "workflows/transfer/run_prior_selection.py",
    ],
    "dd-n4-morgan-atom-transfer-v1": [
        "src/rerank/experiments/build_pool_morgan_features.py",
        "src/rerank/experiments/run_morgan_comparison.py",
        "workflows/transfer/run_morgan_comparison.py",
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    plan = root / "docs" / "analysis_plan.md"
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "docs/analysis_plan.md"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    record = {
        "schema_version": 1,
        "status": "pending_supervisor_signoff",
        "approved_protocol_ids": [],
        "blocked_protocol_ids": {
            protocol: "Separate supervisor sign-off required before execution"
            for protocol in RUNNERS
        },
        "supervisor": None,
        "approval_date": None,
        "analysis_plan_commit": commit,
        "analysis_plan_sha256": sha256(plan),
        "runner_sha256": {
            protocol: {Path(name).name: sha256(root / name) for name in paths}
            for protocol, paths in RUNNERS.items()
        },
        "note": "Supervisor must independently review the corrected plan and each runner before changing status/approved_protocol_ids. Do not auto-approve.",
    }
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
