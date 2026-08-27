#!/usr/bin/env python
"""Fail-closed stage orchestration for the sharded cap10-tuned-v1 runner.

This file contains no training logic.  It gives borrowed machines stable stage
commands while preserving the validation-only search and post-freeze test lock
implemented by :mod:`run_tuned_revision`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rerank.analysis.analyze_conformer_aggregate import (
    FEATURE_CACHE_NAME,
    VALIDATION_FEATURE_CACHE_NAME,
    verify_seed_run,
)
from rerank.revision_tuning import PRIOR_TRANSFORMS


@dataclass(frozen=True)
class WorkflowPaths:
    root: Path
    selection_bundle: Path
    search_root: Path
    prior_freeze: Path
    selection_freeze: Path
    test_results: Path
    capacity_plan: Path
    capacity_freeze: Path
    capacity_results: Path


def workflow_paths(root: str | Path) -> WorkflowPaths:
    root = Path(root).resolve()
    return WorkflowPaths(
        root=root,
        selection_bundle=root / "selection" / "train_valid_only.pkl",
        search_root=root / "search",
        prior_freeze=root / "freeze" / "prior_transform.json",
        selection_freeze=root / "freeze" / "model_selection.json",
        test_results=root / "test_results",
        capacity_plan=root / "freeze" / "capacity_plan.json",
        capacity_freeze=root / "freeze" / "capacity_selection.json",
        capacity_results=root / "capacity_test_results",
    )


def _base_command() -> list[str]:
    return [sys.executable, "-m", "rerank.experiments.run_tuned_revision"]


def stage_commands(args: argparse.Namespace, paths: WorkflowPaths) -> list[list[str]]:
    seed_root = Path(args.conformer_root).resolve() / f"seed_{args.conformer_seed}"
    feature_root = seed_root / "features"
    base = _base_command()
    common_search = [
        "--selection-bundle", str(paths.selection_bundle),
        "--output-root", str(paths.search_root),
        "--shard-index", str(args.shard_index),
        "--shard-count", str(args.shard_count),
        "--device", args.device,
        "--compact-progress",
    ]
    stop_after_epoch = getattr(args, "stop_after_epoch", None)
    if stop_after_epoch is not None:
        common_search.extend([
            "--stop-after-epoch", str(stop_after_epoch),
            "--stop-margin-seconds", str(args.stop_margin_seconds),
        ])
    if args.stage == "prepare":
        return [[
            *base, "prepare",
            "--train-test-cache", str(feature_root / FEATURE_CACHE_NAME),
            "--validation-cache", str(feature_root / VALIDATION_FEATURE_CACHE_NAME),
            "--conformer-seed", str(args.conformer_seed),
            "--output", str(paths.selection_bundle),
        ]]
    if args.stage == "baseline-shard":
        return [
            [*base, "search", *common_search, "--arm", "baseline", "--prior-transform", transform]
            for transform in PRIOR_TRANSFORMS
        ]
    if args.stage == "freeze-prior":
        return [[
            *base, "select-prior",
            "--selection-bundle", str(paths.selection_bundle),
            "--output-root", str(paths.search_root),
            "--output", str(paths.prior_freeze),
        ]]
    if args.stage == "augmented-shard":
        if not paths.prior_freeze.is_file():
            raise FileNotFoundError("Augmented shards require the frozen prior-transform record.")
        prior = json.loads(paths.prior_freeze.read_text(encoding="utf-8"))
        selected_transform = prior.get("selected_prior_transform")
        if selected_transform not in PRIOR_TRANSFORMS:
            raise ValueError("Prior-freeze record has no supported selected transform.")
        return [[
            *base, "search", *common_search,
            "--arm", "augmented", "--prior-transform", selected_transform,
            "--prior-freeze", str(paths.prior_freeze),
        ]]
    if args.stage == "freeze-selection":
        return [[
            *base, "freeze-selection",
            "--selection-bundle", str(paths.selection_bundle),
            "--prior-freeze", str(paths.prior_freeze),
            "--output-root", str(paths.search_root),
            "--output", str(paths.selection_freeze),
        ]]
    if args.stage == "evaluate-test":
        return [[
            *base, "evaluate-test",
            "--selection-freeze", str(paths.selection_freeze),
            "--train-test-cache", str(feature_root / FEATURE_CACHE_NAME),
            "--output-root", str(paths.search_root),
            "--result-dir", str(paths.test_results),
            "--device", args.device,
        ]]
    if args.stage == "prepare-capacity":
        required = (
            args.capacity_dropout, args.capacity_learning_rate, args.capacity_margin,
        )
        if any(value is None for value in required) or not str(args.capacity_decision_note or "").strip():
            raise ValueError("prepare-capacity requires all three settings and a decision note.")
        return [[
            *base, "prepare-capacity",
            "--selection-bundle", str(paths.selection_bundle),
            "--prior-freeze", str(paths.prior_freeze),
            "--dropout", str(args.capacity_dropout),
            "--learning-rate", str(args.capacity_learning_rate),
            "--margin", str(args.capacity_margin),
            "--decision-note", str(args.capacity_decision_note),
            "--output", str(paths.capacity_plan),
        ]]
    if args.stage == "capacity-fit":
        return [
            [
                *base, "run-capacity",
                "--selection-bundle", str(paths.selection_bundle),
                "--capacity-plan", str(paths.capacity_plan),
                "--output-root", str(paths.search_root),
                "--arm", arm, "--device", args.device,
            ]
            for arm in ("baseline", "augmented")
        ]
    if args.stage == "freeze-capacity":
        return [[
            *base, "freeze-capacity",
            "--selection-bundle", str(paths.selection_bundle),
            "--capacity-plan", str(paths.capacity_plan),
            "--output-root", str(paths.search_root),
            "--output", str(paths.capacity_freeze),
        ]]
    if args.stage == "evaluate-capacity-test":
        return [[
            *base, "evaluate-capacity-test",
            "--capacity-freeze", str(paths.capacity_freeze),
            "--train-test-cache", str(feature_root / FEATURE_CACHE_NAME),
            "--output-root", str(paths.search_root),
            "--result-dir", str(paths.capacity_results),
            "--device", args.device,
        ]]
    raise ValueError(f"Unsupported executable stage: {args.stage}")


def status(paths: WorkflowPaths, shard_count: int) -> dict:
    def count(pattern: str) -> int:
        return sum(1 for _ in paths.search_root.glob(pattern)) if paths.search_root.exists() else 0

    return {
        "selection_bundle": paths.selection_bundle.is_file(),
        "baseline_trial_json": count("search/baseline/*/config_*/seed_*/trial.json"),
        "baseline_expected": 3 * 81 * 5,
        "prior_freeze": paths.prior_freeze.is_file(),
        "augmented_trial_json": count("search/augmented/*/config_*/seed_*/trial.json"),
        "augmented_expected": 81 * 5,
        "selection_freeze": paths.selection_freeze.is_file(),
        "test_manifest": (paths.test_results / "manifest.json").is_file(),
        "capacity_plan": paths.capacity_plan.is_file(),
        "capacity_freeze": paths.capacity_freeze.is_file(),
        "capacity_test_manifest": (paths.capacity_results / "manifest.json").is_file(),
        "configured_shard_count": shard_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare", "baseline-shard", "freeze-prior", "augmented-shard",
            "freeze-selection", "evaluate-test", "status",
            "prepare-capacity", "capacity-fit", "freeze-capacity",
            "evaluate-capacity-test",
        ),
    )
    parser.add_argument("--conformer-seed", type=int, required=True, choices=range(42, 47))
    parser.add_argument("--conformer-root", default="outputs/jcheminform_revision/conformers")
    parser.add_argument("--output-root")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=9)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--capacity-dropout", type=float)
    parser.add_argument("--capacity-learning-rate", type=float)
    parser.add_argument("--capacity-margin", type=float)
    parser.add_argument("--capacity-decision-note")
    parser.add_argument(
        "--stop-after-epoch",
        type=float,
        help="Budget deadline forwarded to search stages for trial-safe pause.",
    )
    parser.add_argument("--stop-margin-seconds", type=float, default=600.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count).")
    root = args.output_root or (
        f"outputs/jcheminform_revision/tuned_primary/conformer_seed_{args.conformer_seed}"
    )
    paths = workflow_paths(root)
    if args.stage == "status":
        print(json.dumps(status(paths, args.shard_count), indent=2))
        return
    # This hashes every retained seed artifact before any stage can consume it.
    verify_seed_run(
        Path(args.conformer_root) / f"seed_{args.conformer_seed}",
        args.conformer_seed,
    )
    commands = stage_commands(args, paths)
    if args.print_only:
        print(json.dumps(commands, indent=2))
        return
    for command in commands:
        result = subprocess.run(command)
        if result.returncode == 75:
            print("Search paused safely at the rental-budget boundary.")
            raise SystemExit(75)
        result.check_returncode()


if __name__ == "__main__":
    main()
