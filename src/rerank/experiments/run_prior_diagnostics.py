"""N3-S1 frozen-model prior-lock diagnostic for four candidate pools.

The step computes ranking agreement, prior skew/gap and the actual frozen
training-pair normalizer clipping. No weights or normalizers are fitted.
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
from scipy.stats import kendalltau, skew

from rerank.experiments.cross_inference_features import clipping_audit
from rerank.experiments.protocol_gate import fingerprint, require_approval, resolve_frozen, sha256
from rerank.experiments.run_cross_inference import (
    _checked_diagonal_rank,
    _expanded_features,
    _historical_reference_scores,
    _score_lists,
)
from rerank.features import FeatureNormalizer
from rerank.model import RankerMLP
from rerank.tuning import load_selection_bundle, transform_selection_cache
from rerank.study_data import (
    canonicalize_reactant_set,
    canonicalize_smiles,
    load_reactions,
    make_pairwise_dataset,
)
from rerank.pool_streaming import arm_view, atomic_json, digest_key, identity_digest, load_products


PROTOCOL_ID = "dd-n3-pool-specific-selection-v1"
POOLS = ("historical_cap10", "aizynthfinder_only", "localretro_only", "merged")
SEEDS = tuple(range(42, 62))
ARMS = ("baseline", "augmented")


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prior_summary(lists: list[np.ndarray]) -> dict:
    skew_values = []
    gaps = []
    ties = 0
    for features in lists:
        prior = np.asarray(features[:, 0], dtype=np.float64)
        if len(prior) > 1:
            top = np.sort(prior)[::-1]
            gaps.append(float(top[0] - top[1]))
            if np.ptp(prior) > 0 and len(prior) > 2:
                skew_values.append(float(skew(prior, bias=False)))
            ties += int(len(np.unique(prior)) < len(prior))
    return {
        "mean_within_list_skew": float(np.mean(skew_values)) if skew_values else None,
        "median_top1_top2_gap": float(np.median(gaps)) if gaps else None,
        "prior_tie_list_fraction": ties / max(len(lists), 1),
        "skew_defined_lists": len(skew_values),
        "gap_defined_lists": len(gaps),
    }


def _model(trial: dict, checkpoint: Path, input_dim: int, device: str):
    import torch

    config = trial["config"]
    model = RankerMLP(
        input_dim=input_dim,
        hidden_dims=[int(config["hidden_width"])],
        dropout=float(config["dropout"]),
        use_batch_norm=False,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def _historical_lists(args: argparse.Namespace):
    with Path(args.historical_cache).open("rb") as handle:
        payload = pickle.load(handle)["payload"]
    rows = {}
    for meta, (_, candidates), truth, features in zip(
        payload["eval_metadata"],
        payload["eval_pwc"],
        payload["eval_ground_truths"],
        payload["eval_features"],
        strict=True,
    ):
        if meta.get("source_split") != "test":
            raise ValueError("Historical diagnostic cache is not official-test only.")
        target = canonicalize_reactant_set(truth)
        mask = np.asarray(
            [canonicalize_reactant_set(item["smiles"]) == target for item in candidates], dtype=bool
        )
        if not mask.any():
            raise ValueError("Historical covered test row lost its reference.")
        rows[int(meta["reaction_id"])] = (np.asarray(features, dtype=np.float32), mask)
    return rows


def _expanded_lists(args: argparse.Namespace, pool: str):
    k1 = Path(args.k1_root) / pool
    test_manifest = _load(k1 / "test_results" / "manifest.json")
    feature_path = Path(args.feature_root) / pool / "manifest.json"
    freeze_path = k1 / "models" / "freeze.json"
    if test_manifest["feature_freeze"]["sha256"] != sha256(feature_path) or test_manifest[
        "model_freeze"
    ]["sha256"] != sha256(freeze_path):
        raise ValueError("K1 test result does not bind frozen model/features.")
    feature_lookup = _expanded_features(feature_path, pool)
    product_path = Path(args.products_jsonl)
    rank_by_product = {
        str(row["canonical_product"]): int(row["product_rank"])
        for row in load_products(product_path)
    }
    frozen_rows = {}
    with (k1 / "test_results" / "reaction_ranks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("covered"):
                frozen_rows[int(row["reaction_id"])] = row
    reactions = {
        r.reaction_id: r
        for r in load_reactions(args.source_csv, args.metadata_csv)
        if r.source_split == "test" and r.reaction_id in frozen_rows
    }
    if set(reactions) != set(frozen_rows):
        raise ValueError("K1 covered IDs differ from official-test source.")
    rows = {}
    for reaction_id, reaction in reactions.items():
        rank = rank_by_product[reaction.product_key]
        digests, features = feature_lookup[rank]
        digests, features = digests[:10], features[:10]
        target = identity_digest(reaction.ground_truth_key)
        mask = np.asarray([digest_key(item) == target for item in digests], dtype=bool)
        if not mask.any() or len(features) != int(frozen_rows[reaction_id]["candidate_count"]):
            raise ValueError("K1 frozen candidate alignment differs.")
        rows[reaction_id] = (features, mask)
    return rows, frozen_rows, _load(freeze_path)


def _historical_training_pairs(args: argparse.Namespace, arm: str, seed: int) -> np.ndarray:
    bundle = load_selection_bundle(args.historical_selection_bundle)
    view = transform_selection_cache(bundle, arm, "raw")
    dataset = make_pairwise_dataset(view, seed=seed, max_neg_per_pos=5, negative_mining="random")
    return np.concatenate(
        [
            np.stack([row[0] for row in dataset._pairs]),
            np.stack([row[1] for row in dataset._pairs]),
        ],
        axis=0,
    )


def diagnostic(args: argparse.Namespace) -> dict:
    import torch

    started = time.perf_counter()
    require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("N3-S1 result already exists.")
    hist_freeze = _load(args.historical_freeze)
    if (
        hist_freeze.get("protocol_id") != "cap10-tuned-20seed-v1"
        or tuple(hist_freeze["seeds"]) != SEEDS
    ):
        raise PermissionError("Wrong historical G1 freeze.")
    if hist_freeze["retained_train_test_cache_sha256"] != sha256(args.historical_cache):
        raise ValueError("Historical test feature cache differs from G1 freeze.")
    report = {}
    numerical_tie_audit = []
    for pool in POOLS:
        if pool == "historical_cap10":
            rows = _historical_lists(args)
            freeze = hist_freeze
            frozen_ranks = {}
            for arm in ARMS:
                for seed in SEEDS:
                    with (Path(args.historical_prediction_root) / f"{arm}_seed_{seed}.csv").open(
                        encoding="utf-8", newline=""
                    ) as handle:
                        frozen_ranks[(arm, seed)] = {
                            int(row["reaction_id"]): int(row["reranked_rank"])
                            for row in csv.DictReader(handle)
                        }
        else:
            rows, frozen_ranks, freeze = _expanded_lists(args, pool)
        ids = sorted(rows)
        matrices = [rows[item][0] for item in ids]
        masks = [rows[item][1] for item in ids]
        pool_report = {"covered_reactions": len(ids), "prior": _prior_summary(matrices), "arms": {}}
        for arm in ARMS:
            arm_report = {}
            for seed in SEEDS:
                if pool == "historical_cap10":
                    trial = freeze[f"selected_{arm}"]["trials"][str(seed)]
                    model_root = Path(args.historical_model_root)
                    checkpoint = model_root / trial["checkpoint_relpath"]
                    normalizer_path = model_root / trial["normalizer_relpath"]
                    if "sha256:" + sha256(checkpoint) != trial["checkpoint_sha256"]:
                        raise ValueError("Historical checkpoint checksum differs.")
                    if "sha256:" + sha256(normalizer_path) != trial["normalizer_sha256"]:
                        raise ValueError("Historical normalizer checksum differs.")
                    arm_matrices = [
                        item[:, (0, 1, 5, 6) if arm == "baseline" else tuple(range(7))]
                        for item in matrices
                    ]
                    pair_values = _historical_training_pairs(args, arm, seed)
                else:
                    model_root = Path(args.k1_root) / pool / "models"
                    trial_path = resolve_frozen(
                        freeze["trials"][arm][str(seed)],
                        model_root / "validation" / arm / f"seed_{seed}",
                    )
                    trial = _load(trial_path)
                    checkpoint = resolve_frozen(trial["checkpoint"], trial_path.parent)
                    normalizer_path = resolve_frozen(trial["normalizer"], trial_path.parent)
                    arm_matrices = [arm_view(item, arm) for item in matrices]
                    selection_root = Path(args.k1_root) / pool / "selection"
                    selection = _load(selection_root / "manifest.json")
                    pair_path = resolve_frozen(selection["train_pairs"][str(seed)], selection_root)
                    with np.load(pair_path, allow_pickle=False) as pairs:
                        pair_values = np.concatenate(
                            (arm_view(pairs["positive"], arm), arm_view(pairs["negative"], arm)),
                            axis=0,
                        )
                normalizer = FeatureNormalizer.load(str(normalizer_path))
                clipping = clipping_audit(
                    pair_values, normalizer.mean_, normalizer.std_, normalizer.clip_sigma
                )
                model = _model(trial, checkpoint, arm_matrices[0].shape[1], args.device)
                scores = _score_lists(model, normalizer, arm_matrices, args.device)
                agreements = []
                taus = []
                seed_ties = []
                for reaction_id, features, arm_features, mask, score in zip(
                    ids, matrices, arm_matrices, masks, scores, strict=True
                ):
                    prior = features[:, 0]
                    agreements.append(int(np.argmax(score) == np.argmax(prior)))
                    if len(prior) > 1:
                        tau = float(kendalltau(prior, score).statistic)
                        if np.isfinite(tau):
                            taus.append(tau)
                    actual_rank = (
                        int(np.flatnonzero(mask[np.argsort(-score, kind="stable")])[0]) + 1
                    )
                    frozen_rank = (
                        frozen_ranks[(arm, seed)][reaction_id]
                        if pool == "historical_cap10"
                        else int(frozen_ranks[reaction_id][f"{arm}_rank_seed_{seed}"])
                    )
                    if actual_rank != frozen_rank:
                        reference_scores = (
                            _historical_reference_scores(args, arm, seed, reaction_id)
                            if pool == "historical_cap10"
                            else None
                        )
                        try:
                            _, audit = _checked_diagonal_rank(
                                actual_rank,
                                frozen_rank,
                                arm_features,
                                mask,
                                score,
                                reference_scores,
                            )
                        except ValueError as exc:
                            raise ValueError(
                                f"N3-S1 frozen rank reproduction failed: {pool}/{arm}/{seed}/{reaction_id}"
                            ) from exc
                        item = {
                            "pool": pool,
                            "arm": arm,
                            "seed": seed,
                            "reaction_id": reaction_id,
                            "recomputed_rank": actual_rank,
                            "frozen_rank": frozen_rank,
                            **audit,
                        }
                        seed_ties.append(item)
                        numerical_tie_audit.append(item)
                arm_report[str(seed)] = {
                    "top1_equals_prior_fraction": float(np.mean(agreements)),
                    "mean_kendall_tau_defined": float(np.mean(taus)) if taus else None,
                    "kendall_tau_defined_lists": len(taus),
                    "kendall_tau_undefined_lists": len(ids) - len(taus),
                    "training_pair_prior_clipping": clipping,
                    "frozen_rank_reproduction": (
                        "exact_except_audited_identical_input_ties" if seed_ties else "exact"
                    ),
                    "numerical_tie_count": len(seed_ties),
                    "diagnostic_score_source": "recomputed frozen model; stable candidate order for exact ties",
                }
                print(f"N3-S1 {pool}/{arm}/{seed}: {len(ids)} test reactions", flush=True)
            pool_report["arms"][arm] = arm_report
        report[pool] = pool_report
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "status": "complete",
        "stage": "N3_S1",
        "protocol_id": PROTOCOL_ID,
        "comparator": "frozen learned ranking versus frozen raw prior on four candidate pools",
        "single_intended_change": "diagnose prior lock and training-pair clipping without refitting",
        "seeds": list(SEEDS),
        "pools": report,
        "numerical_tie_audit": numerical_tie_audit,
        "numerical_tie_policy": (
            "N2 identical-input gate: contiguous indistinguishable arm-feature group; "
            "local and available historical score span <= 1e-6. Frozen ranks are not changed. "
            "Prior-agreement and Kendall diagnostics use recomputed scores and may vary "
            "at numerical ties across platforms. All other rank mismatches fail closed."
        ),
        "approval": fingerprint(args.approval_record),
        "inputs": {
            name: fingerprint(getattr(args, name))
            for name in (
                "historical_freeze",
                "historical_cache",
                "historical_selection_bundle",
                "source_csv",
                "metadata_csv",
                "products_jsonl",
            )
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": args.device,
        },
        "training_performed": False,
        "normalizer_refit_performed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--analysis-plan", default="docs/analysis_plan.md")
    parser.add_argument("--historical-freeze", required=True)
    parser.add_argument("--historical-model-root", required=True)
    parser.add_argument("--historical-cache", required=True)
    parser.add_argument("--historical-selection-bundle", required=True)
    parser.add_argument("--historical-prediction-root", required=True)
    parser.add_argument("--k1-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--products-jsonl", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(diagnostic(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
