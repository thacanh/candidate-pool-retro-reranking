"""Isolated Digital Discovery J/K truncation runner.

This module deliberately reuses frozen WS-E feature shards while writing only
under ``outputs/digital_discovery_round_jk``.  It never generates candidates,
embeddings, or tuning searches.  Scientific commands fail closed until a
committed, approved amendment record is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from rerank.experiments import run_ws_e_ranking as ws_e
from rerank.ws_e_streaming import (
    AUGMENTED_COLUMNS,
    BASELINE_COLUMNS,
    atomic_json,
    fingerprint,
)


PROTOCOL_ID = "dd-expanded-pool-cap10-d1-v1"
SEEDS = tuple(range(42, 62))
POOLS = ("aizynthfinder_only", "localretro_only", "merged")
OUTPUT_NAMESPACE = Path("outputs/digital_discovery_round_jk")
PROTECTED_PREFIXES = (
    Path("outputs/revision_analysis"),
    Path("outputs/jcheminform_revision/numerical_freeze_v2"),
    Path("paper/overleaf"),
)
EXPECTED_CONFIGS = {
    "baseline": {
        "hidden_width": 128,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "margin": 0.3,
    },
    "augmented": {
        "hidden_width": 128,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "margin": 0.1,
    },
}


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _config_without_index(record: Mapping) -> dict:
    return {
        key: record[key]
        for key in ("hidden_width", "dropout", "learning_rate", "margin")
    }


def _assert_primary_freeze(path: str | Path) -> dict:
    freeze = _load_json(path)
    if freeze.get("protocol_id") != "cap10-tuned-v1":
        raise PermissionError("K1 requires the frozen cap10-tuned-v1 D1 freeze.")
    if freeze.get("selected_prior_transform") != "raw":
        raise PermissionError("K1 requires the frozen raw prior transform.")
    for arm, key in (("baseline", "selected_baseline"), ("augmented", "selected_augmented")):
        actual = _config_without_index(freeze[key]["config"])
        if actual != EXPECTED_CONFIGS[arm]:
            raise PermissionError(
                f"Frozen {arm} configuration differs from the approved width-128 D1 config: {actual}"
            )
    return freeze


def _resolved(path: str | Path) -> Path:
    return Path(path).resolve()


def _assert_isolated_output(path: str | Path) -> Path:
    root = Path.cwd().resolve()
    output = _resolved(path)
    allowed = (root / OUTPUT_NAMESPACE).resolve()
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise PermissionError(f"J/K output must stay under {allowed}: {output}") from exc
    for protected in PROTECTED_PREFIXES:
        protected_path = (root / protected).resolve()
        try:
            output.relative_to(protected_path)
        except ValueError:
            continue
        raise PermissionError(f"J/K command cannot write protected JoC path: {output}")
    return output


def _assert_approval(path: str | Path, plan_path: str | Path) -> dict:
    record = _load_json(path)
    required = {
        "protocol_id": PROTOCOL_ID,
        "status": "approved",
        "width128_d1_resolution": "approved",
        "j4_bin_assignment": "median_across_seeds_worse_bin_on_half_tie",
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise PermissionError(f"J/K approval gate failed: {key} must equal {expected!r}.")
    if not record.get("supervisor") or not record.get("approval_date"):
        raise PermissionError("J/K approval record lacks supervisor/date.")
    plan = fingerprint(plan_path)
    if record.get("analysis_plan_sha256") != plan["sha256"]:
        raise PermissionError("Approved J/K record belongs to a different analysis-plan revision.")
    if not record.get("analysis_plan_commit"):
        raise PermissionError("Approved J/K record lacks the committed analysis-plan hash.")
    if not isinstance(record.get("platform_lock"), Mapping) or not record["platform_lock"]:
        raise PermissionError("Approved J/K record lacks the platform lock.")
    return record


def _approval_provenance(path: str | Path, plan_path: str | Path) -> dict:
    record = _assert_approval(path, plan_path)
    return {
        "approval_record": fingerprint(path),
        "analysis_plan": fingerprint(plan_path),
        "analysis_plan_commit": record["analysis_plan_commit"],
        "approval_date": record["approval_date"],
        "supervisor": record["supervisor"],
        "j4_bin_assignment": record["j4_bin_assignment"],
        "width128_d1_resolution": record["width128_d1_resolution"],
        "platform_lock": record["platform_lock"],
    }


def _parameter_count(input_dim: int, hidden_width: int = 128) -> int:
    return input_dim * hidden_width + hidden_width + hidden_width + 1


def preflight(args: argparse.Namespace) -> dict:
    primary = _assert_primary_freeze(args.primary_freeze)
    pools = {}
    for pool, feature_freeze in zip(POOLS, args.feature_freeze, strict=True):
        record = _load_json(feature_freeze)
        if record.get("pool_name") != pool or not record.get("complete"):
            raise ValueError(f"Incomplete or mismatched feature freeze for {pool}.")
        pools[pool] = {
            "feature_freeze": fingerprint(feature_freeze),
            "product_count": int(record["product_count"]),
            "candidate_rows": int(record["candidate_rows"]),
            "feature_names": list(record["feature_names"]),
        }
    approval = None
    if args.approval_record:
        approval = _approval_provenance(args.approval_record, args.analysis_plan)
    return {
        "status": "ready" if approval else "preparation_only_approval_pending",
        "protocol_id": PROTOCOL_ID,
        "source_primary_freeze": fingerprint(args.primary_freeze),
        "source_primary_protocol": primary["protocol_id"],
        "candidate_cap": 10,
        "seeds": list(SEEDS),
        "candidate_generation": False,
        "embedding_generation": False,
        "hyperparameter_search": False,
        "configs": EXPECTED_CONFIGS,
        "actual_parameter_counts": {
            "baseline": _parameter_count(len(BASELINE_COLUMNS)),
            "augmented": _parameter_count(len(AUGMENTED_COLUMNS)),
        },
        "checklist_289_parameter_sentence_rejected": True,
        "pools": pools,
        "approval": approval,
        "output_namespace": str((Path.cwd() / OUTPUT_NAMESPACE).resolve()),
    }


def prepare_selection(args: argparse.Namespace) -> dict:
    approval = _approval_provenance(args.approval_record, args.analysis_plan)
    _assert_primary_freeze(args.primary_freeze)
    output = _assert_isolated_output(args.output_root)
    delegated = argparse.Namespace(
        feature_freeze=args.feature_freeze,
        pool_jsonl=args.pool_jsonl,
        products_jsonl=args.products_jsonl,
        pool_index_npz=args.pool_index_npz,
        pool_index_manifest=args.pool_index_manifest,
        source_csv=args.source_csv,
        metadata_csv=args.metadata_csv,
        output_root=str(output),
        candidate_cap=10,
        seeds=SEEDS,
        output_protocol_id=PROTOCOL_ID,
    )
    result = ws_e.prepare_selection(delegated)
    if result["pool_name"] not in POOLS:
        raise ValueError("K1 received an unapproved expanded pool.")
    result["round_jk_approval"] = approval
    atomic_json(output / "manifest.json", result)
    return result


def fit_validation(args: argparse.Namespace) -> dict:
    approval = _approval_provenance(args.approval_record, args.analysis_plan)
    _assert_primary_freeze(args.primary_freeze)
    _assert_isolated_output(args.selection_root)
    output = _assert_isolated_output(args.output_root)
    selection = _load_json(Path(args.selection_root) / "manifest.json")
    if (
        selection.get("protocol_id") != PROTOCOL_ID
        or int(selection.get("candidate_cap", 0)) != 10
        or tuple(selection.get("seeds", ())) != SEEDS
    ):
        raise PermissionError("K1 fit requires a matching cap-10/20-seed selection freeze.")
    delegated = argparse.Namespace(
        selection_root=args.selection_root,
        primary_freeze=args.primary_freeze,
        output_root=str(output),
        arm=args.arm,
        seed=args.seed,
        device=args.device,
    )
    result = ws_e.fit_validation(delegated)
    expected_parameters = _parameter_count(
        len(BASELINE_COLUMNS if args.arm == "baseline" else AUGMENTED_COLUMNS)
    )
    if int(result["model_parameters"]) != expected_parameters:
        raise RuntimeError("K1 model parameter assertion failed.")
    result["round_jk_approval"] = approval
    atomic_json(
        output / "validation" / args.arm / f"seed_{args.seed}" / "trial.json",
        result,
    )
    return result


def freeze_models(args: argparse.Namespace) -> dict:
    approval = _approval_provenance(args.approval_record, args.analysis_plan)
    _assert_primary_freeze(args.primary_freeze)
    _assert_isolated_output(args.selection_root)
    _assert_isolated_output(args.output_root)
    output = _assert_isolated_output(args.output)
    delegated = argparse.Namespace(
        selection_root=args.selection_root,
        primary_freeze=args.primary_freeze,
        output_root=args.output_root,
        output=str(output),
    )
    result = ws_e.freeze(delegated)
    result["round_jk_approval"] = approval
    atomic_json(output, result)
    return result


def evaluate_test(args: argparse.Namespace) -> dict:
    approval = _approval_provenance(args.approval_record, args.analysis_plan)
    _assert_primary_freeze(args.primary_freeze)
    for path in (args.selection_root, args.output_root, args.result_dir):
        _assert_isolated_output(path)
    delegated = argparse.Namespace(**vars(args))
    result = ws_e.evaluate_test(delegated)
    result["round_jk_approval"] = approval
    atomic_json(Path(args.result_dir) / "manifest.json", result)
    return result


def status(args: argparse.Namespace) -> dict:
    root = _assert_isolated_output(args.pool_root)
    selection_manifest = root / "selection" / "manifest.json"
    selection = _load_json(selection_manifest) if selection_manifest.is_file() else None
    trials = {}
    for arm in ("baseline", "augmented"):
        trials[arm] = sum(
            (root / "models" / "validation" / arm / f"seed_{seed}" / "trial.json").is_file()
            for seed in SEEDS
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "selection_ready": bool(selection),
        "selection_cap": None if selection is None else selection.get("candidate_cap"),
        "validation_trials": trials,
        "validation_expected_per_arm": len(SEEDS),
        "model_freeze": (root / "models" / "freeze.json").is_file(),
        "official_test": (root / "test_results" / "manifest.json").is_file(),
    }


def _approval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--analysis-plan", default="docs/analysis_plan.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("preflight")
    check.add_argument("--primary-freeze", required=True)
    check.add_argument("--feature-freeze", action="append", required=True)
    check.add_argument("--approval-record")
    check.add_argument("--analysis-plan", default="docs/analysis_plan.md")

    prepare = sub.add_parser("prepare-selection")
    _approval_arguments(prepare)
    prepare.add_argument("--primary-freeze", required=True)
    prepare.add_argument("--feature-freeze", required=True)
    prepare.add_argument("--pool-jsonl", required=True)
    prepare.add_argument("--products-jsonl", required=True)
    prepare.add_argument("--pool-index-npz", required=True)
    prepare.add_argument("--pool-index-manifest", required=True)
    prepare.add_argument("--source-csv", default="data/uspto_smiles.csv")
    prepare.add_argument("--metadata-csv", default="data/uspto_reaction_metadata.csv")
    prepare.add_argument("--output-root", required=True)

    fit = sub.add_parser("fit-validation")
    _approval_arguments(fit)
    fit.add_argument("--primary-freeze", required=True)
    fit.add_argument("--selection-root", required=True)
    fit.add_argument("--output-root", required=True)
    fit.add_argument("--arm", choices=("baseline", "augmented"), required=True)
    fit.add_argument("--seed", type=int, choices=SEEDS, required=True)
    fit.add_argument("--device", default="cpu")

    freeze = sub.add_parser("freeze")
    _approval_arguments(freeze)
    freeze.add_argument("--primary-freeze", required=True)
    freeze.add_argument("--selection-root", required=True)
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument("--output", required=True)

    evaluate = sub.add_parser("evaluate-test")
    _approval_arguments(evaluate)
    evaluate.add_argument("--model-freeze", required=True)
    evaluate.add_argument("--feature-freeze", required=True)
    evaluate.add_argument("--selection-root", required=True)
    evaluate.add_argument("--primary-freeze", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--pool-jsonl", required=True)
    evaluate.add_argument("--products-jsonl", required=True)
    evaluate.add_argument("--pool-index-npz", required=True)
    evaluate.add_argument("--pool-index-manifest", required=True)
    evaluate.add_argument("--source-csv", default="data/uspto_smiles.csv")
    evaluate.add_argument("--metadata-csv", default="data/uspto_reaction_metadata.csv")
    evaluate.add_argument("--result-dir", required=True)
    evaluate.add_argument("--device", default="cpu")

    show = sub.add_parser("status")
    show.add_argument("--pool-root", required=True)
    args = parser.parse_args()
    if args.command == "preflight" and len(args.feature_freeze) != len(POOLS):
        parser.error("preflight requires three --feature-freeze values in AiZ, LocalRetro, merged order")
    return args


def main() -> None:
    args = parse_args()
    commands = {
        "preflight": preflight,
        "prepare-selection": prepare_selection,
        "fit-validation": fit_validation,
        "freeze": freeze_models,
        "evaluate-test": evaluate_test,
        "status": status,
    }
    print(json.dumps(commands[args.command](args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
