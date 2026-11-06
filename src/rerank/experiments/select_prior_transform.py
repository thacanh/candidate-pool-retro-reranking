"""Freeze per-pool N3 baseline prior choice from official validation only."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from rerank.experiments.protocol_gate import fingerprint, require_approval, resolve_frozen
from rerank.pool_streaming import atomic_json


PROTOCOL_ID = "dd-n3-pool-specific-selection-v1"
SEEDS = tuple(range(42, 47))
TRANSFORMS = ("raw", "log", "rank")
POOLS = ("aizynthfinder_only", "localretro_only", "merged")
TIE_EPS = 1e-12


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def choose_transform(means: dict[str, float]) -> str:
    """Maximize mean validation MRR; ties prefer raw, log, then midrank."""

    if set(means) != set(TRANSFORMS) or not all(np.isfinite(value) for value in means.values()):
        raise ValueError("All three finite validation means are required.")
    best = max(means.values())
    return next(name for name in TRANSFORMS if best - means[name] <= TIE_EPS)


def select(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    root = Path(args.output_root).resolve()
    if (root / "manifest.json").exists() or (root / "choice_freeze.json").exists():
        raise FileExistsError("N3 S2 choice is immutable and already exists.")
    choices = {}
    inputs = {}
    for pool in POOLS:
        original_root = Path(args.k1_root) / pool
        original_freeze_path = original_root / "models" / "freeze.json"
        original_freeze = _load(original_freeze_path)
        if (
            not original_freeze.get("complete")
            or original_freeze.get("pool_name") != pool
            or original_freeze.get("selected_prior_transform") != "raw"
        ):
            raise ValueError(f"K1 raw baseline is not frozen for {pool}.")
        inputs[f"{pool}_raw_freeze"] = fingerprint(original_freeze_path)
        per_transform = {}
        trial_records = {}
        for transform in TRANSFORMS:
            values = []
            trial_records[transform] = []
            for seed in SEEDS:
                if transform == "raw":
                    trial_path = resolve_frozen(
                        original_freeze["trials"]["baseline"][str(seed)],
                        original_root / "models" / "validation" / "baseline" / f"seed_{seed}",
                    )
                else:
                    trial_path = (
                        Path(args.n3_root)
                        / pool
                        / transform
                        / "models"
                        / "validation"
                        / "baseline"
                        / f"seed_{seed}"
                        / "trial.json"
                    )
                trial = _load(trial_path)
                if (
                    trial.get("pool_name") != pool
                    or trial.get("arm") != "baseline"
                    or int(trial.get("seed", -1)) != seed
                    or trial.get("test_partition_loaded") is not False
                ):
                    raise ValueError("N3 validation trial identity/gate differs.")
                for artifact in ("checkpoint", "normalizer"):
                    resolve_frozen(trial[artifact], trial_path.parent)
                values.append(float(trial["best_validation_mrr"]))
                trial_records[transform].append(fingerprint(trial_path))
            per_transform[transform] = {
                "mean_best_validation_mrr": float(np.mean(values)),
                "per_seed": dict(zip(map(str, SEEDS), values)),
            }
            if transform != "raw":
                feature = Path(args.n3_root) / pool / transform / "features" / "manifest.json"
                selection = Path(args.n3_root) / pool / transform / "selection" / "manifest.json"
                feature_record, selection_record = _load(feature), _load(selection)
                if (
                    feature_record.get("prior_transform") != transform
                    or selection_record.get("prior_transform") != transform
                    or selection_record.get("pool_name") != pool
                    or selection_record.get("test_ground_truth_loaded") is not False
                ):
                    raise ValueError("N3 transform/selection manifest differs.")
                inputs[f"{pool}_{transform}_features"] = fingerprint(feature)
                inputs[f"{pool}_{transform}_selection"] = fingerprint(selection)
        means = {name: per_transform[name]["mean_best_validation_mrr"] for name in TRANSFORMS}
        choices[pool] = {
            "selected": choose_transform(means),
            "validation": per_transform,
            "trial_records": trial_records,
        }
    root.mkdir(parents=True, exist_ok=True)
    freeze = {
        "schema_version": 1,
        "record_kind": "dd_n3_prior_choice_freeze",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "comparator": "raw, log(p+1e-12), descending within-list midrank; baseline seeds 42-46",
        "single_intended_change": "select prior transform independently per expanded pool on official validation",
        "seeds": list(SEEDS),
        "tie_rule": "raw, log, rank within 1e-12 of maximum",
        "choices": choices,
        "inputs": inputs,
        "approval": fingerprint(args.approval_record),
        "test_partition_loaded": False,
        "test_references_loaded": False,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(root / "choice_freeze.json", freeze)
    atomic_json(root / "manifest.json", freeze)
    return freeze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--analysis-plan", default="docs/analysis_plan.md")
    parser.add_argument("--k1-root", required=True)
    parser.add_argument("--n3-root", required=True)
    parser.add_argument("--output-root", required=True)
    print(json.dumps(select(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
