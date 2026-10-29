"""Extend frozen historical Morgan control from five to twenty paired seeds.

Fit 47--61 on official train/validation only; freeze all 20 model references
before evaluating official test. The first five existing fits are reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np

from rerank.experiments.protocol_gate import fingerprint, require_approval, sha256
from rerank.experiments.run_cross_inference import _score_lists
from rerank.features import FeatureNormalizer
from rerank.model import RankerMLP
from rerank.tuning import (
    GridConfig,
    load_selection_bundle,
    train_validation_trial,
    transform_selection_cache,
)
from rerank.experiments.run_tuned import validate_selection_freeze
from rerank.study_data import canonicalize_reactant_set
from rerank.pool_streaming import atomic_json, atomic_npz


PROTOCOL_ID = "dd-n4-morgan-atom-transfer-v1"
SEEDS = tuple(range(42, 62))
EXISTING = tuple(range(42, 47))
REUSE_AUDIT_SHA256 = '3ee8816ea0962c42d2121580d7175c55e2829cb606d3b1e781ccf21ab236199d'


def checked_frozen_reuse(ids, computed, old_ranks, audit):
    """Accept only the exact owner-reviewed replay discrepancy set, not new ties."""
    if set(old_ranks) != set(ids):
        raise ValueError('Frozen Morgan prediction coverage differs')
    actual = [
        (rid, int(rank), int(old_ranks[rid]))
        for rid, rank in zip(ids, computed, strict=True)
        if rank != old_ranks[rid]
    ]
    expected = [(r['reaction_id'], r['computed'], r['frozen']) for r in audit['mismatches']]
    if actual != expected:
        raise ValueError('Morgan replay differs from the explicitly approved audit')
    return np.asarray([old_ranks[rid] for rid in ids], dtype=np.uint8)


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _artifact(root: Path, trial: dict, kind: str) -> Path:
    source = root / trial[f"{kind}_relpath"]
    if not source.resolve().is_relative_to(root.resolve()):
        raise ValueError("Historical Morgan trial path escapes its model root.")
    if "sha256:" + sha256(source) != trial[f"{kind}_sha256"]:
        raise ValueError(f"Historical Morgan {kind} checksum differs.")
    return source


def fit_one(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    if args.seed in EXISTING or args.seed not in SEEDS:
        raise ValueError("Historical Morgan N4 fits only missing seeds 47--61.")
    primary = validate_selection_freeze(args.primary_freeze)
    control = _load(args.existing_control_freeze)
    if (
        control.get("protocol_id") != "cap10-tuned-encoder-control-v1"
        or control.get("primary_selection_fingerprint") != primary["selection_fingerprint"]
        or control.get("prior_transform") != "raw"
    ):
        raise PermissionError("Existing Morgan/primary freezes differ.")
    source = load_selection_bundle(args.control_selection_bundle)
    if sha256(args.control_selection_bundle) != str(
        control["control_selection_bundle"]["sha256"]
    ).removeprefix("sha256:"):
        raise ValueError("Morgan control selection bundle checksum differs.")
    if source["representation_provenance"].get("source_protocol_id") != "C-MORGAN-ATOM":
        raise ValueError("Selection bundle does not contain the atom-centred Morgan control.")
    output_root = Path(args.output_root).resolve()
    trial_root = output_root / "validation" / f"seed_{args.seed}"
    trial_path = trial_root / "trial.json"
    checkpoint = trial_root / "best_checkpoint.pt"
    normalizer = trial_root / "normalizer.npz"
    if any(path.exists() for path in (trial_path, checkpoint, normalizer)):
        raise FileExistsError("Partial or complete N4 historical seed already exists.")
    config = GridConfig(**control["primary_selected_config"])
    view = transform_selection_cache(source, "augmented", "raw")
    trained = train_validation_trial(
        view, config, args.seed, args.device, checkpoint, normalizer, allowed_seeds=SEEDS
    )
    result = {
        "schema_version": 1,
        "record_kind": "dd_n4_historical_morgan_trial",
        "protocol_id": PROTOCOL_ID,
        "seed": args.seed,
        "arm": "augmented",
        "config": control["primary_selected_config"],
        "prior_transform": "raw",
        "selection_bundle": fingerprint(args.control_selection_bundle),
        "checkpoint_relpath": checkpoint.relative_to(output_root).as_posix(),
        "checkpoint_sha256": "sha256:" + sha256(checkpoint),
        "normalizer_relpath": normalizer.relative_to(output_root).as_posix(),
        "normalizer_sha256": "sha256:" + sha256(normalizer),
        "test_partition_loaded": False,
        "runtime_seconds": time.perf_counter() - started,
        **trained,
    }
    atomic_json(trial_path, result)
    return result


def freeze(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    primary = validate_selection_freeze(args.primary_freeze)
    old = _load(args.existing_control_freeze)
    g1 = _load(args.g1_freeze)
    if (
        old["primary_selection_fingerprint"] != primary["selection_fingerprint"]
        or tuple(g1.get("seeds", ())) != SEEDS
    ):
        raise ValueError("Historical baseline/Morgan comparator freeze mismatch.")
    root = Path(args.output_root).resolve()
    manifest_path = root / "freeze.json"
    if manifest_path.exists():
        raise FileExistsError("N4 historical model freeze already exists.")
    trials = {}
    for seed in SEEDS:
        if seed in EXISTING:
            trial = old["selected_control"]["trials"][str(seed)]
            model_root = Path(args.existing_control_root).resolve()
            trial_path = None
        else:
            model_root = root
            trial_path = root / "validation" / f"seed_{seed}" / "trial.json"
            trial = _load(trial_path)
        if (
            int(trial["seed"]) != seed
            or trial["config"] != old["primary_selected_config"]
            or trial.get("test_partition_loaded") is not False
        ):
            raise ValueError("N4 historical validation trial differs.")
        for kind in ("checkpoint", "normalizer"):
            _artifact(model_root, trial, kind)
        trials[str(seed)] = {
            "model_root": str(model_root),
            "trial": trial,
            "trial_file": fingerprint(trial_path) if trial_path else None,
        }
    result = {
        "schema_version": 1,
        "record_kind": "dd_n4_historical_20seed_freeze",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "comparator": "frozen historical G1 2D baseline versus atom-centred Morgan augmented",
        "single_intended_change": "extend existing five-seed Morgan control to seeds 42--61",
        "seeds": list(SEEDS),
        "trials": trials,
        "existing_control_freeze": fingerprint(args.existing_control_freeze),
        "g1_freeze": fingerprint(args.g1_freeze),
        "control_selection_bundle": fingerprint(args.control_selection_bundle),
        "approval": fingerprint(args.approval_record),
        "test_partition_loaded": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(manifest_path, result)
    return result


def evaluate(args: argparse.Namespace) -> dict:
    import torch

    started = time.perf_counter()
    approval = require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    audit_path = getattr(args, 'reproduction_audit', None)
    audited = {}
    if audit_path:
        if (
            sha256(audit_path) != REUSE_AUDIT_SHA256
            or approval.get('n4_replay_reuse', {}).get('audit_sha256') != REUSE_AUDIT_SHA256
        ):
            raise PermissionError('Historical reuse needs the exact approved audit')
        audited = {
            int(row['seed']): row
            for row in (
                json.loads(line)
                for line in Path(audit_path).read_text().splitlines()
                if line.strip()
            )
        }
        if set(audited) != set(EXISTING):
            raise ValueError('Historical audit seed set differs')
    freeze_path = Path(args.output_root) / "freeze.json"
    frozen = _load(freeze_path)
    if frozen.get("protocol_id") != PROTOCOL_ID or tuple(frozen["seeds"]) != SEEDS:
        raise PermissionError("N4 historical twenty-seed freeze is incomplete.")
    old = _load(args.existing_control_freeze)
    if frozen["existing_control_freeze"]["sha256"] != sha256(args.existing_control_freeze) or old[
        "retained_control_train_test_cache_sha256"
    ] != sha256(args.control_cache):
        raise ValueError("Morgan test cache/control freeze checksum differs.")
    output = Path(args.result_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("N4 historical test result already exists.")
    with Path(args.control_cache).open("rb") as handle:
        payload = pickle.load(handle)["payload"]
    ids = [int(meta["reaction_id"]) for meta in payload["eval_metadata"]]
    if len(ids) != 3985 or len(set(ids)) != len(ids):
        raise ValueError("N4 historical official-test coverage differs from frozen 3,985.")
    matrices = [np.asarray(item, dtype=np.float32) for item in payload["eval_features"]]
    masks = []
    for (_, candidates), truth in zip(
        payload["eval_pwc"], payload["eval_ground_truths"], strict=True
    ):
        target = canonicalize_reactant_set(truth)
        mask = np.asarray(
            [canonicalize_reactant_set(item["smiles"]) == target for item in candidates], dtype=bool
        )
        if not mask.any():
            raise ValueError("Morgan historical test reference is absent from candidate list.")
        masks.append(mask)
    augmented = np.empty((len(SEEDS), len(ids)), dtype=np.uint8)
    baseline = np.empty_like(augmented)
    recomputed = np.empty_like(augmented)
    for seed_index, seed in enumerate(SEEDS):
        record = frozen["trials"][str(seed)]
        model_root = Path(record["model_root"])
        trial = record["trial"]
        checkpoint = _artifact(model_root, trial, "checkpoint")
        normalizer = FeatureNormalizer.load(str(_artifact(model_root, trial, "normalizer")))
        config = trial["config"]
        model = RankerMLP(
            input_dim=7,
            hidden_dims=[int(config["hidden_width"])],
            dropout=float(config["dropout"]),
            use_batch_norm=False,
        ).to(args.device)
        model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
        model.eval()
        scores = _score_lists(model, normalizer, matrices, args.device)
        for index, (mask, score) in enumerate(zip(masks, scores, strict=True)):
            augmented[seed_index, index] = (
                int(np.flatnonzero(mask[np.argsort(-score, kind="stable")])[0]) + 1
            )
        recomputed[seed_index] = augmented[seed_index]
        with (Path(args.historical_baseline_predictions) / f"baseline_seed_{seed}.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            reference = {
                int(row["reaction_id"]): int(row["reranked_rank"]) for row in csv.DictReader(handle)
            }
        if set(reference) != set(ids):
            raise ValueError("G1 baseline prediction coverage differs.")
        baseline[seed_index] = [reference[item] for item in ids]
        if seed in EXISTING:
            with (Path(args.existing_control_predictions) / f"augmented_seed_{seed}.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                old_ranks = {
                    int(row["reaction_id"]): int(row["reranked_rank"])
                    for row in csv.DictReader(handle)
                }
            if audited:
                audit = audited[seed]
                for key, path in [
                    ('checkpoint', checkpoint),
                    ('normalizer', _artifact(model_root, trial, 'normalizer')),
                    (
                        'predictions',
                        Path(args.existing_control_predictions) / f'augmented_seed_{seed}.csv',
                    ),
                ]:
                    if sha256(path) != audit[f'{key}_sha256']:
                        raise ValueError('Approved Morgan audit artifact changed')
                augmented[seed_index] = checked_frozen_reuse(
                    ids, augmented[seed_index], old_ranks, audit
                )
            elif not np.array_equal(augmented[seed_index], [old_ranks[item] for item in ids]):
                raise ValueError("N4 first-five historical Morgan test ranks do not reproduce.")
        print(f'N4 historical evaluation: seed {seed} complete', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    arrays = output / "reaction_ranks.npz"
    atomic_npz(
        arrays,
        reaction_ids=np.asarray(ids, dtype=np.int32),
        seeds=np.asarray(SEEDS, dtype=np.int32),
        baseline_ranks=baseline,
        augmented_ranks=augmented,
        recomputed_augmented_ranks=recomputed,
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "comparator": frozen["comparator"],
        "single_intended_change": frozen["single_intended_change"],
        "test_reactions_covered": len(ids),
        "first_five_exact_reproduction": bool(np.array_equal(recomputed[:5], augmented[:5])),
        "historical_prediction_policy": 'reuse frozen first-five with owner-approved replay audit'
        if audited
        else 'exact replay',
        "replay_audit": fingerprint(audit_path) if audit_path else None,
        "replay_rank_disagreements": int(np.count_nonzero(recomputed != augmented)),
        "model_freeze": fingerprint(freeze_path),
        "control_cache": fingerprint(args.control_cache),
        "reaction_ranks": fingerprint(arrays),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": args.device,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit-one", "freeze", "evaluate"))
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--analysis-plan", default="docs/analysis_plan.md")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--primary-freeze")
    parser.add_argument("--existing-control-freeze", required=True)
    parser.add_argument("--existing-control-root")
    parser.add_argument("--control-selection-bundle")
    parser.add_argument("--g1-freeze")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--control-cache")
    parser.add_argument("--historical-baseline-predictions")
    parser.add_argument("--existing-control-predictions")
    parser.add_argument("--result-dir")
    args = parser.parse_args()
    required = {
        "fit-one": ("primary_freeze", "control_selection_bundle", "seed"),
        "freeze": (
            "primary_freeze",
            "control_selection_bundle",
            "g1_freeze",
            "existing_control_root",
        ),
        "evaluate": (
            "control_cache",
            "historical_baseline_predictions",
            "existing_control_predictions",
            "result_dir",
        ),
    }
    for name in required[args.command]:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for {args.command}")
    result = {"fit-one": fit_one, "freeze": freeze, "evaluate": evaluate}[args.command](args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
