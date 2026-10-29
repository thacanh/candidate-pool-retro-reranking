"""Approved-only N2 2x2 frozen-model cross-inference (no training/refitting).

The two historical/K1 diagonal cells are independently reproduced on the
3,814-reaction common set before either off-diagonal cell is accepted.
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

from rerank.analysis.analyze_predictions import (
    crossed_product_seed_bootstrap,
    product_cluster_bootstrap,
)
from rerank.experiments.cross_inference_features import clipping_audit, training_arm_features
from rerank.experiments.protocol_gate import fingerprint, require_approval, resolve_frozen, sha256
from rerank.features import FeatureNormalizer
from rerank.model import RankerMLP
from rerank.study_data import canonicalize_reactant_set, canonicalize_smiles, load_reactions
from rerank.pool_streaming import (
    atomic_json,
    digest_key,
    identity_digest,
    load_products,
)


PROTOCOL_ID = "dd-n2-cross-inference-v1"
POOLS = ("historical_cap10", "aizynthfinder_only")
ARMS = ("baseline", "augmented")
SEEDS = tuple(range(42, 62))
METRICS = ("top1", "mrr")


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _common_reactions(path: Path) -> tuple[list[int], dict[int, str]]:
    products: dict[int, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["pool"] != "aizynthfinder_only":
                continue
            reaction_id = int(row["reaction_id"])
            product = canonicalize_smiles(row["canonical_product"])
            if reaction_id in products:
                raise ValueError("Duplicate AiZ common reaction in frozen M2 table.")
            products[reaction_id] = product
    if len(products) != 3814 or len(set(products.values())) != 3814:
        raise ValueError("N2 requires exactly 3,814 distinct common products/reactions.")
    return sorted(products), products


def _check_freezes(historical: Path, expanded: Path) -> tuple[dict, dict]:
    h = _read_json(historical)
    k = _read_json(expanded)
    if h.get("protocol_id") != "cap10-tuned-20seed-v1" or tuple(h.get("seeds", ())) != SEEDS:
        raise PermissionError("Wrong frozen historical 20-seed model family.")
    if h.get("prior_transform") != "raw" or h.get("test_partition_loaded") is not False:
        raise PermissionError("Historical model freeze is not validation-only raw-prior.")
    if (k.get("protocol_id") != "dd-expanded-pool-cap10-d1-v1"
            or k.get("pool_name") != "aizynthfinder_only" or not k.get("complete")
            or tuple(k.get("seeds", ())) != SEEDS or k.get("candidate_cap") != 10
            or k.get("selected_prior_transform") != "raw"
            or k.get("test_partition_loaded") is not False):
        raise PermissionError("Wrong frozen AiZ-only 20-seed model family.")
    return h, k


def _expanded_features(feature_freeze_path: Path, pool_name: str = "aizynthfinder_only") -> dict[int, tuple[np.ndarray, np.ndarray]]:
    freeze = _read_json(feature_freeze_path)
    if not freeze.get("complete") or freeze.get("pool_name") != pool_name:
        raise ValueError("Incomplete/wrong expanded-pool feature freeze.")
    shard_dir = feature_freeze_path.parent / "shards"
    lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    row_count = 0
    if len(freeze["shard_manifests"]) != int(freeze["shard_count"]):
        raise ValueError("Incomplete AiZ-only feature shard set.")
    for record in freeze["shard_manifests"]:
        shard_manifest_path = resolve_frozen(record, shard_dir)
        shard_manifest = _read_json(shard_manifest_path)
        data_path = resolve_frozen(shard_manifest["output"], shard_dir)
        with np.load(data_path, allow_pickle=False) as arrays:
            ranks = arrays["product_rank"]
            digests = arrays["candidate_sha256"]
            features = arrays["features"]
            if features.ndim != 2 or features.shape[1] != 9 or len(ranks) != len(features):
                raise ValueError("Wrong expanded-pool frozen feature shape.")
            row_count += len(features)
            boundaries = np.flatnonzero(np.r_[True, ranks[1:] != ranks[:-1], True])
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                rank = int(ranks[start])
                if rank in lookup:
                    raise ValueError("Product rank appears in multiple feature shards.")
                lookup[rank] = (digests[start:stop].copy(), features[start:stop].copy())
    if (row_count != int(freeze["candidate_rows"])
            or any(rank < 0 or rank >= int(freeze["product_count"]) for rank in lookup)):
        raise ValueError("Feature lookup has malformed product/candidate coverage.")
    return lookup


def _load_lists(args: argparse.Namespace, reaction_ids: list[int], products: dict[int, str]):
    wanted = set(reaction_ids)
    with Path(args.historical_cache).open("rb") as handle:
        historical = pickle.load(handle)["payload"]
    if any(str(item.get("source_split")) != "test" for item in historical["eval_metadata"]):
        raise PermissionError("Historical evaluation cache is not official-test-only.")
    historical_lists = {}
    for metadata, (product, candidates), truth, features in zip(
        historical["eval_metadata"], historical["eval_pwc"],
        historical["eval_ground_truths"], historical["eval_features"], strict=True
    ):
        reaction_id = int(metadata["reaction_id"])
        if reaction_id not in wanted:
            continue
        if canonicalize_smiles(product) != products[reaction_id]:
            raise ValueError("Historical product identity differs from frozen M2.")
        if len(candidates) != len(features) or not 1 <= len(candidates) <= 10:
            raise ValueError("Historical candidate/feature order or cap differs.")
        target = canonicalize_reactant_set(truth)
        mask = np.asarray([canonicalize_reactant_set(item["smiles"]) == target
                           for item in candidates], dtype=bool)
        if not mask.any():
            raise ValueError("Historical common reaction lacks its reference candidate.")
        historical_lists[reaction_id] = (np.asarray(features, dtype=np.float32), mask)
    if set(historical_lists) != wanted:
        raise ValueError("Historical cache does not cover the full N2 common set.")

    reactions = {r.reaction_id: r for r in load_reactions(args.source_csv, args.metadata_csv)
                 if r.source_split == "test" and r.reaction_id in wanted}
    if set(reactions) != wanted:
        raise ValueError("Official test reaction IDs differ from frozen M2 common set.")
    by_product = {str(item["canonical_product"]): int(item["product_rank"])
                  for item in load_products(args.products_jsonl)}
    features_by_rank = _expanded_features(Path(args.aiz_feature_freeze))
    aiz_lists = {}
    for reaction_id in reaction_ids:
        reaction = reactions[reaction_id]
        if reaction.product_key != products[reaction_id]:
            raise ValueError("AiZ product key differs from frozen M2 common set.")
        product_rank = by_product[reaction.product_key]
        digests, features = features_by_rank[product_rank]
        digests, features = digests[:10], features[:10]
        target = identity_digest(reaction.ground_truth_key)
        mask = np.asarray([digest_key(item) == target for item in digests], dtype=bool)
        if not mask.any() or not 1 <= len(features) <= 10:
            raise ValueError("AiZ common reaction lacks its candidate/reference within cap 10.")
        aiz_lists[reaction_id] = (features, mask)
    return {"historical_cap10": historical_lists, "aizynthfinder_only": aiz_lists}


def _frozen_trial(args: argparse.Namespace, training_pool: str, arm: str, seed: int,
                  historical_freeze: dict, aiz_freeze: dict) -> tuple[dict, Path, Path]:
    if training_pool == "historical_cap10":
        root = Path(args.historical_model_root)
        trial = historical_freeze[f"selected_{arm}"]["trials"][str(seed)]
        checkpoint = root / trial["checkpoint_relpath"]
        normalizer = root / trial["normalizer_relpath"]
        if "sha256:" + sha256(checkpoint) != trial["checkpoint_sha256"]:
            raise ValueError("Historical checkpoint fingerprint differs.")
        if "sha256:" + sha256(normalizer) != trial["normalizer_sha256"]:
            raise ValueError("Historical normalizer fingerprint differs.")
    else:
        root = Path(args.aiz_model_root)
        record = aiz_freeze["trials"][arm][str(seed)]
        trial_path = resolve_frozen(record, root / "validation" / arm / f"seed_{seed}")
        trial = _read_json(trial_path)
        if trial.get("seed") != seed or trial.get("arm") != arm:
            raise ValueError("AiZ frozen trial identity differs.")
        checkpoint = resolve_frozen(trial["checkpoint"], trial_path.parent)
        normalizer = resolve_frozen(trial["normalizer"], trial_path.parent)
    if trial.get("prior_transform") not in (None, "raw"):
        raise ValueError("N2 accepts only raw-prior frozen checkpoints.")
    return trial, checkpoint, normalizer


def _score_lists(model, normalizer, matrices: list[np.ndarray], device: str) -> list[np.ndarray]:
    """Match the frozen test runners' one-reaction-at-a-time CPU scoring."""
    import torch

    scores = []
    with torch.no_grad():
        for matrix in matrices:
            values = torch.as_tensor(normalizer.transform(matrix),
                                     dtype=torch.float32, device=device)
            scores.append(model.score(values).detach().cpu().numpy().reshape(-1))
    return scores


def _historical_reference_scores(args: argparse.Namespace, arm: str, seed: int,
                                 reaction_id: int) -> np.ndarray:
    path = Path(args.historical_prediction_root) / f"{arm}_seed_{seed}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["reaction_id"]) == reaction_id:
                scores = np.asarray(json.loads(row["reranked_scores_json"]), dtype=np.float64)
                if not len(scores) or not np.isfinite(scores).all():
                    raise ValueError("Frozen historical scores are missing or non-finite.")
                return scores
    raise ValueError("Frozen historical reference reaction is absent.")


def _checked_diagonal_rank(computed: int, frozen: int, matrix: np.ndarray,
                           mask: np.ndarray, scores: np.ndarray,
                           frozen_scores: np.ndarray | None) -> tuple[int, dict | None]:
    """Keep a frozen rank only within an indistinguishable feature-row group."""
    if computed == frozen:
        return frozen, None
    positive = np.flatnonzero(mask)
    if len(positive) != 1 or not 1 <= computed <= len(matrix) or not 1 <= frozen <= len(matrix):
        raise ValueError("N2 diagonal reproduction failed outside an audited numerical tie.")
    indistinguishable = np.flatnonzero(np.all(matrix == matrix[positive[0]], axis=1))
    order = np.argsort(-scores, kind="stable")
    group_positions = np.flatnonzero(np.isin(order, indistinguishable)) + 1
    low, high = min(computed, frozen), max(computed, frozen)
    local_span = float(np.ptp(scores[indistinguishable]))
    reference_span = (float(np.ptp(frozen_scores[low - 1:high]))
                      if frozen_scores is not None and len(frozen_scores) == len(matrix) else None)
    if (len(indistinguishable) < 2 or computed not in group_positions
            or frozen not in group_positions
            or len(group_positions) != int(group_positions[-1] - group_positions[0] + 1)
            or local_span > 1e-6
            or (frozen_scores is not None and (reference_span is None or reference_span > 1e-6))):
        raise ValueError("N2 diagonal reproduction failed outside an audited numerical tie.")
    return frozen, {"indistinguishable_candidate_count": int(len(indistinguishable)),
                    "local_score_span": local_span,
                    "reference_tied_rank_score_span": reference_span}


def _reference_ranks(args: argparse.Namespace, reaction_ids: list[int]) -> dict:
    wanted = set(reaction_ids)
    references = {pool: {arm: {} for arm in ARMS} for pool in POOLS}
    for arm in ARMS:
        for seed in SEEDS:
            path = Path(args.historical_prediction_root) / f"{arm}_seed_{seed}.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                values = {int(row["reaction_id"]): int(row["reranked_rank"])
                          for row in csv.DictReader(handle) if int(row["reaction_id"]) in wanted}
            if set(values) != wanted:
                raise ValueError("Historical diagonal reference predictions are incomplete.")
            references["historical_cap10"][arm][seed] = values
    rows = {}
    with Path(args.aiz_prediction_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            reaction_id = int(row["reaction_id"])
            if reaction_id in wanted:
                rows[reaction_id] = row
    if set(rows) != wanted:
        raise ValueError("AiZ diagonal reference predictions are incomplete.")
    for arm in ARMS:
        for seed in SEEDS:
            references["aizynthfinder_only"][arm][seed] = {
                reaction_id: int(rows[reaction_id][f"{arm}_rank_seed_{seed}"])
                for reaction_id in reaction_ids
            }
    return references, rows


def _rank_scores(ranks: np.ndarray, metric: str) -> np.ndarray:
    if metric == "top1":
        return (ranks == 1).astype(np.float64)
    if metric == "mrr":
        return 1.0 / ranks.astype(np.float64)
    raise ValueError(metric)


def run(args: argparse.Namespace) -> dict:
    import torch

    started = time.perf_counter()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("N2 output is immutable; refusing an existing nonempty directory.")
    approval = require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    h_freeze, a_freeze = _check_freezes(Path(args.historical_freeze), Path(args.aiz_freeze))
    if sha256(args.historical_cache) != h_freeze["retained_train_test_cache_sha256"]:
        raise ValueError("Historical official-test feature cache differs from the model freeze.")
    historical_test_manifest = _read_json(args.historical_test_manifest)
    aiz_test_manifest = _read_json(args.aiz_test_manifest)
    aiz_selection_manifest = _read_json(args.aiz_selection_manifest)
    if historical_test_manifest.get("g1_freeze", {}).get("sha256") != sha256(args.historical_freeze):
        raise ValueError("Historical test manifest does not bind the supplied G1 model freeze.")
    if aiz_test_manifest.get("model_freeze", {}).get("sha256") != sha256(args.aiz_freeze):
        raise ValueError("AiZ test manifest does not bind the supplied K1 model freeze.")
    if aiz_test_manifest.get("feature_freeze", {}).get("sha256") != sha256(args.aiz_feature_freeze):
        raise ValueError("AiZ test manifest does not bind the supplied feature freeze.")
    if aiz_test_manifest.get("predictions", {}).get("sha256") != sha256(args.aiz_prediction_jsonl):
        raise ValueError("AiZ frozen prediction archive checksum differs.")
    if aiz_selection_manifest.get("inputs", {}).get("feature_freeze", {}).get("sha256") != sha256(args.aiz_feature_freeze):
        raise ValueError("AiZ validation selection does not bind the supplied feature freeze.")
    # Validation-only model and feature freezes are checked before reading any test label.
    ids, products = _common_reactions(Path(args.common_reactions_csv))
    lists = _load_lists(args, ids, products)
    reference, aiz_reference_rows = _reference_ranks(args, ids)
    for index, reaction_id in enumerate(ids):
        expected_count = int(aiz_reference_rows[reaction_id]["candidate_count"])
        if len(lists["aizynthfinder_only"][reaction_id][0]) != expected_count:
            raise ValueError("AiZ candidate count differs from frozen K1 predictions.")
        prior_position = int(np.flatnonzero(lists["aizynthfinder_only"][reaction_id][1])[0]) + 1
        if prior_position != int(aiz_reference_rows[reaction_id]["prior_rank"]):
            raise ValueError("AiZ ground-truth order differs from frozen K1 predictions.")

    ranks = np.empty((2, 2, 2, len(SEEDS), len(ids)), dtype=np.uint8)
    clipping_rows = []
    diagonal_tie_audit = []
    diagonal_violations = []
    for train_index, training_pool in enumerate(POOLS):
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                trial, checkpoint, normalizer_path = _frozen_trial(
                    args, training_pool, arm, seed, h_freeze, a_freeze
                )
                normalizer = FeatureNormalizer.load(str(normalizer_path))
                config = trial["config"]
                input_dim = 4 if training_pool == "historical_cap10" and arm == "baseline" else (
                    7 if training_pool == "historical_cap10" else (6 if arm == "baseline" else 9)
                )
                model = RankerMLP(input_dim=input_dim,
                                  hidden_dims=[int(config["hidden_width"])],
                                  dropout=float(config["dropout"]),
                                  use_batch_norm=False).to(args.device)
                model.load_state_dict(torch.load(checkpoint, map_location=args.device,
                                                 weights_only=True))
                model.eval()
                for eval_index, evaluation_pool in enumerate(POOLS):
                    matrices = [training_arm_features(lists[evaluation_pool][reaction_id][0],
                                training_pool=training_pool,
                                evaluation_pool=evaluation_pool, arm=arm) for reaction_id in ids]
                    flat = np.concatenate(matrices, axis=0)
                    audit = clipping_audit(flat, normalizer.mean_, normalizer.std_,
                                           normalizer.clip_sigma)
                    clipping_rows.append({"training_pool": training_pool,
                                          "evaluation_pool": evaluation_pool,
                                          "arm": arm, "seed": seed, **audit})
                    scores = _score_lists(model, normalizer, matrices, args.device)
                    for reaction_index, (reaction_id, score) in enumerate(zip(ids, scores, strict=True)):
                        mask = lists[evaluation_pool][reaction_id][1]
                        position = np.flatnonzero(mask[np.argsort(-score, kind="stable")])
                        computed_rank = int(position[0]) + 1
                        if training_pool == evaluation_pool:
                            frozen = reference[training_pool][arm][seed][reaction_id]
                            frozen_scores = (_historical_reference_scores(args, arm, seed, reaction_id)
                                             if computed_rank != frozen and training_pool == "historical_cap10"
                                             else None)
                            tie_audit = None
                            try:
                                computed_rank, tie_audit = _checked_diagonal_rank(
                                    computed_rank, frozen, matrices[reaction_index], mask,
                                    score, frozen_scores
                                )
                            except ValueError:
                                diagonal_violations.append({"training_pool": training_pool,
                                                            "arm": arm, "seed": seed,
                                                            "reaction_id": reaction_id,
                                                            "frozen_rank": frozen,
                                                            "computed_rank": computed_rank})
                            if tie_audit is not None:
                                diagonal_tie_audit.append({"training_pool": training_pool,
                                                           "arm": arm, "seed": seed,
                                                           "reaction_id": reaction_id,
                                                           "frozen_rank": frozen,
                                                           "reason": "indistinguishable arm feature rows; frozen Linux CPU rank retained",
                                                           **tie_audit})
                        ranks[train_index, eval_index, arm_index, seed_index,
                              reaction_index] = computed_rank

    if diagonal_violations:
        raise ValueError(f"N2 diagonal reproduction failed for {len(diagonal_violations)} "
                         f"reaction/arm/seed cells; first five: {diagonal_violations[:5]}")

    clusters = np.asarray([products[item] for item in ids], dtype=object)
    effects = []
    for train_index, training_pool in enumerate(POOLS):
        for eval_index, evaluation_pool in enumerate(POOLS):
            cell_clipping = [row for row in clipping_rows if row["training_pool"] == training_pool
                             and row["evaluation_pool"] == evaluation_pool]
            uninterpretable = any(row["uninterpretable_prior_clipping_gt_10pct"]
                                  for row in cell_clipping)
            for metric_index, metric in enumerate(METRICS):
                difference = (_rank_scores(ranks[train_index, eval_index, 1], metric)
                              - _rank_scores(ranks[train_index, eval_index, 0], metric))
                product_ci = product_cluster_bootstrap(difference, clusters,
                                                       args.bootstrap_samples,
                                                       args.rng_seed + 100 * train_index
                                                       + 10 * eval_index + metric_index)
                crossed_ci = crossed_product_seed_bootstrap(difference, clusters,
                                                             args.bootstrap_samples,
                                                             args.rng_seed + 1000 + 100 * train_index
                                                             + 10 * eval_index + metric_index)
                effects.append({"training_pool": training_pool,
                                "evaluation_pool": evaluation_pool, "metric": metric,
                                "effect": float(difference.mean()),
                                "product_ci95_low": product_ci["ci95_low"],
                                "product_ci95_high": product_ci["ci95_high"],
                                "crossed_ci95_low": crossed_ci["ci95_low"],
                                "crossed_ci95_high": crossed_ci["ci95_high"],
                                "prior_clipping_gt_10pct": uninterpretable,
                                "interpretation": "uninterpretable_clipping" if uninterpretable else (
                                    "descriptive_noncommensurate_prior" if training_pool != evaluation_pool
                                    else "diagonal_reproduction")})

    output.mkdir(parents=True, exist_ok=True)
    rank_path = output / "cell_ranks.npz"
    np.savez_compressed(rank_path, ranks=ranks, reaction_ids=np.asarray(ids, dtype=np.int32),
                        seeds=np.asarray(SEEDS, dtype=np.int32))
    effect_path = output / "cross_inference_effects.csv"
    with effect_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    clipping_path = output / "clipping.csv"
    with clipping_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clipping_rows[0]))
        writer.writeheader()
        writer.writerows(clipping_rows)
    result = {
        "schema_version": 1, "status": "complete", "protocol_id": PROTOCOL_ID,
        "comparator": "frozen historical/AiZ 20-seed diagonal cells on 3,814 common reactions",
        "single_intended_change": "apply each training pool's frozen checkpoint/normalizer to the other pool's frozen feature lists",
        "scope": {"common_reactions": len(ids), "seeds": list(SEEDS),
                  "training_pools": list(POOLS), "evaluation_pools": list(POOLS)},
        "diagonal_rank_reproduction": (
            "exact_except_audited_identical_input_ties" if diagonal_tie_audit
            else "exact_for_all_arms_seeds_reactions"
        ),
        "diagonal_tie_audit": diagonal_tie_audit,
        "off_diagonal_interpretation": "descriptive; stored prior scales are not commensurate",
        "approval": fingerprint(args.approval_record),
        "inputs": {name: fingerprint(getattr(args, name)) for name in (
            "historical_freeze", "aiz_freeze", "historical_cache", "aiz_feature_freeze",
            "historical_test_manifest", "aiz_test_manifest", "aiz_selection_manifest",
            "aiz_prediction_jsonl", "common_reactions_csv",
            "source_csv", "metadata_csv", "products_jsonl") if Path(getattr(args, name)).is_file()},
        "historical_prediction_csvs": {
            f"{arm}_{seed}": fingerprint(Path(args.historical_prediction_root) /
                                         f"{arm}_seed_{seed}.csv")
            for arm in ARMS for seed in SEEDS
        },
        "outputs": {name: fingerprint(path) for name, path in (
            ("cell_ranks", rank_path), ("effects", effect_path), ("clipping", clipping_path))},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "device": args.device},
        "training_performed": False, "normalizer_refit_performed": False,
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
    parser.add_argument("--historical-prediction-root", required=True)
    parser.add_argument("--historical-test-manifest", required=True)
    parser.add_argument("--aiz-freeze", required=True)
    parser.add_argument("--aiz-model-root", required=True)
    parser.add_argument("--aiz-feature-freeze", required=True)
    parser.add_argument("--aiz-prediction-jsonl", required=True)
    parser.add_argument("--aiz-test-manifest", required=True)
    parser.add_argument("--aiz-selection-manifest", required=True)
    parser.add_argument("--common-reactions-csv", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--products-jsonl", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=2090)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
