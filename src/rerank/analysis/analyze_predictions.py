#!/usr/bin/env python
"""Final paired G1--G3/H1--H2 analysis from frozen prediction CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from rerank.study_data import canonicalize_smiles


TOP1_FAMILY_SEED = 2027
BOOTSTRAP_SEED = 2026
LOGIT_BOOTSTRAP_SEED = 2028
N_BOOTSTRAP = 10_000
N_SIGN_FLIPS = 100_000
CONTINUOUS_PREDICTORS = (
    "delta_fragment_count",
    "delta_heavy_atom_ratio",
    "delta_tanimoto_to_product",
    "delta_ring_count_vs_product",
    "delta_rotatable_bonds",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def benjamini_hochberg(p_values) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("BH requires a non-empty vector of probabilities.")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def clustered_sign_flip(
    differences: np.ndarray,
    clusters: np.ndarray,
    n_draws: int = N_SIGN_FLIPS,
    seed: int = TOP1_FAMILY_SEED,
) -> float:
    reaction_values = np.asarray(differences, dtype=np.float64).mean(axis=0)
    clusters = np.asarray(clusters, dtype=object)
    unique = list(dict.fromkeys(clusters.tolist()))
    totals = np.asarray([reaction_values[clusters == key].sum() for key in unique])
    totals = totals[np.abs(totals) > 0]
    if len(totals) == 0:
        return 1.0
    observed = abs(float(reaction_values.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    denominator = len(reaction_values)
    while completed < n_draws:
        size = min(2000, n_draws - completed)
        signs = rng.integers(0, 2, size=(size, len(totals)), dtype=np.int8)
        estimates = np.abs(((signs * 2.0 - 1.0) @ totals) / denominator)
        extreme += int(np.sum(estimates >= observed - 1e-15))
        completed += size
    return float((extreme + 1) / (n_draws + 1))


def product_cluster_bootstrap(
    differences: np.ndarray,
    clusters: np.ndarray,
    n_draws: int,
    seed: int,
) -> dict:
    values = np.asarray(differences, dtype=np.float64).mean(axis=0)
    clusters = np.asarray(clusters, dtype=object)
    unique = list(dict.fromkeys(clusters.tolist()))
    indices = [np.flatnonzero(clusters == key) for key in unique]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=np.float64)
    for index in range(n_draws):
        sampled = rng.integers(0, len(indices), size=len(indices))
        rows = np.concatenate([indices[item] for item in sampled])
        draws[index] = float(values[rows].mean())
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"effect": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def crossed_product_seed_bootstrap(
    differences: np.ndarray,
    clusters: np.ndarray,
    n_draws: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    values = np.asarray(differences, dtype=np.float64)
    clusters = np.asarray(clusters, dtype=object)
    unique = list(dict.fromkeys(clusters.tolist()))
    indices = [np.flatnonzero(clusters == key) for key in unique]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=np.float64)
    for index in range(n_draws):
        sampled_clusters = rng.integers(0, len(indices), size=len(indices))
        rows = np.concatenate([indices[item] for item in sampled_clusters])
        sampled_seeds = rng.integers(0, values.shape[0], size=values.shape[0])
        draws[index] = float(values[np.ix_(sampled_seeds, rows)].mean())
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "effect": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_product_clusters": len(unique),
        "n_seeds": int(values.shape[0]),
        "n_bootstrap": n_draws,
        "rng_seed": seed,
        "method": "independent canonical-product-cluster and paired-seed resampling",
    }


def _load_pair(prediction_dir: Path, seed: int) -> pd.DataFrame:
    baseline_path = prediction_dir / f"baseline_seed_{seed}.csv"
    augmented_path = prediction_dir / f"augmented_seed_{seed}.csv"
    left = pd.read_csv(baseline_path)
    right = pd.read_csv(augmented_path)
    required = {
        "reaction_id", "reaction_class", "product_smiles", "ground_truth",
        "reranked_hit@1", "reranked_rr", "reranked_rank", "reranked_candidates_json",
        "baseline_candidates_json",
    }
    for path, frame in ((baseline_path, left), (augmented_path, right)):
        if not required.issubset(frame):
            raise RuntimeError(f"Prediction schema is incomplete: {path}")
        if frame["reaction_id"].duplicated().any() or len(frame) == 0:
            raise RuntimeError(f"Prediction reactions are empty or duplicated: {path}")
    identity = [
        "reaction_id", "reaction_class", "product_smiles", "ground_truth",
        "baseline_candidates_json",
    ]
    if len(left) != len(right) or not left[identity].equals(right[identity]):
        raise RuntimeError(f"Baseline/augmented prediction pairing failed for seed {seed}.")
    result = left[identity].copy()
    for label, frame in (("baseline", left), ("augmented", right)):
        for column in ("reranked_hit@1", "reranked_rr", "reranked_rank", "reranked_candidates_json"):
            result[f"{label}_{column}"] = frame[column].to_numpy()
    return result


def load_prediction_matrices(prediction_dir: str | Path, seeds) -> tuple[dict, dict[int, pd.DataFrame]]:
    prediction_dir = Path(prediction_dir).resolve()
    aligned = {int(seed): _load_pair(prediction_dir, int(seed)) for seed in seeds}
    reference = aligned[int(seeds[0])]
    identity = ["reaction_id", "reaction_class", "product_smiles", "ground_truth", "baseline_candidates_json"]
    for seed, frame in aligned.items():
        if not reference[identity].equals(frame[identity]):
            raise RuntimeError(f"Cross-seed reaction pairing failed for seed {seed}.")
    matrices = {}
    for metric in ("reranked_hit@1", "reranked_rr", "reranked_rank"):
        matrices[metric] = {
            arm: np.stack([aligned[int(seed)][f"{arm}_{metric}"].to_numpy(float) for seed in seeds])
            for arm in ("baseline", "augmented")
        }
    return matrices, aligned


def build_class_tests(matrices: dict, reference: pd.DataFrame, n_bootstrap: int, n_flips: int) -> pd.DataFrame:
    classes = reference["reaction_class"].to_numpy(int)
    clusters_all = np.asarray([canonicalize_smiles(value) or str(value) for value in reference["product_smiles"]], dtype=object)
    top1 = matrices["reranked_hit@1"]["augmented"] - matrices["reranked_hit@1"]["baseline"]
    mrr = matrices["reranked_rr"]["augmented"] - matrices["reranked_rr"]["baseline"]
    rows = []
    for reaction_class in range(1, 11):
        mask = classes == reaction_class
        if not mask.any():
            raise RuntimeError(f"Prespecified reaction class {reaction_class} is absent.")
        top_ci = product_cluster_bootstrap(top1[:, mask], clusters_all[mask], n_bootstrap, BOOTSTRAP_SEED + reaction_class)
        mrr_ci = product_cluster_bootstrap(mrr[:, mask], clusters_all[mask], n_bootstrap, BOOTSTRAP_SEED + 100 + reaction_class)
        rows.append({
            "reaction_class": reaction_class,
            "n_reactions": int(mask.sum()),
            "top1_effect": top_ci["effect"],
            "top1_ci95_low": top_ci["ci95_low"],
            "top1_ci95_high": top_ci["ci95_high"],
            "top1_raw_p": clustered_sign_flip(top1[:, mask], clusters_all[mask], n_flips, TOP1_FAMILY_SEED),
            "mrr_effect": mrr_ci["effect"],
            "mrr_ci95_low": mrr_ci["ci95_low"],
            "mrr_ci95_high": mrr_ci["ci95_high"],
        })
    frame = pd.DataFrame(rows)
    frame["top1_bh_q"] = benjamini_hochberg(frame["top1_raw_p"])
    frame["top1_bh_significant_q05"] = frame["top1_bh_q"] <= 0.05
    return frame


def build_net_flips(matrices: dict, seeds) -> pd.DataFrame:
    baseline = matrices["reranked_hit@1"]["baseline"]
    augmented = matrices["reranked_hit@1"]["augmented"]
    rows = []
    for index, seed in enumerate(seeds):
        promoted = int(np.sum((baseline[index] == 0) & (augmented[index] == 1)))
        degraded = int(np.sum((baseline[index] == 1) & (augmented[index] == 0)))
        rows.append({
            "seed": int(seed), "improved": promoted, "degraded": degraded,
            "unchanged": int(len(baseline[index]) - promoted - degraded),
            "net_flips": promoted - degraded,
        })
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def candidate_descriptors(candidate: str, product: str) -> tuple[float, ...]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, rdMolDescriptors

    candidate_mol = Chem.MolFromSmiles(str(candidate))
    product_mol = Chem.MolFromSmiles(str(product))
    if candidate_mol is None or product_mol is None:
        raise RuntimeError(f"Cannot compute H descriptors for {candidate!r} / {product!r}.")
    fragments = len(Chem.GetMolFrags(candidate_mol))
    product_heavy = product_mol.GetNumHeavyAtoms()
    if product_heavy < 1:
        raise RuntimeError("Product has no heavy atoms.")
    ratio = candidate_mol.GetNumHeavyAtoms() / product_heavy
    candidate_fp = AllChem.GetMorganFingerprintAsBitVect(candidate_mol, 2, nBits=2048, useChirality=False)
    product_fp = AllChem.GetMorganFingerprintAsBitVect(product_mol, 2, nBits=2048, useChirality=False)
    tanimoto = DataStructs.TanimotoSimilarity(candidate_fp, product_fp)
    ring_change = rdMolDescriptors.CalcNumRings(candidate_mol) - rdMolDescriptors.CalcNumRings(product_mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(candidate_mol, rdMolDescriptors.NumRotatableBondsOptions.Strict)
    return float(fragments), float(ratio), float(tanimoto), float(ring_change), float(rotatable)


def _top1(value: str) -> str:
    candidates = json.loads(value)
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Prediction row has no reranked top-1 candidate.")
    return str(candidates[0])


def build_h2_observations(matrices: dict, aligned: dict[int, pd.DataFrame], seeds) -> pd.DataFrame:
    reference = aligned[int(seeds[0])]
    rank_delta = matrices["reranked_rank"]["baseline"] - matrices["reranked_rank"]["augmented"]
    mean_rank_delta = rank_delta.mean(axis=0)
    rows = []
    for row_index in np.flatnonzero(np.abs(mean_rank_delta) > 1e-12):
        descriptor_deltas = []
        product = str(reference.iloc[row_index]["product_smiles"])
        for seed in seeds:
            frame = aligned[int(seed)]
            baseline_candidate = _top1(frame.iloc[row_index]["baseline_reranked_candidates_json"])
            augmented_candidate = _top1(frame.iloc[row_index]["augmented_reranked_candidates_json"])
            baseline_desc = np.asarray(candidate_descriptors(baseline_candidate, product))
            augmented_desc = np.asarray(candidate_descriptors(augmented_candidate, product))
            descriptor_deltas.append(augmented_desc - baseline_desc)
        values = np.mean(descriptor_deltas, axis=0)
        rows.append({
            "reaction_id": int(reference.iloc[row_index]["reaction_id"]),
            "canonical_product": canonicalize_smiles(product) or product,
            "reaction_class": int(reference.iloc[row_index]["reaction_class"]),
            "mean_reference_rank_delta": float(mean_rank_delta[row_index]),
            "promoted": int(mean_rank_delta[row_index] > 0),
            **{name: float(value) for name, value in zip(CONTINUOUS_PREDICTORS, values)},
        })
    return pd.DataFrame(rows)


def descriptor_distributions(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor in CONTINUOUS_PREDICTORS:
        for outcome, group in observations.groupby("promoted"):
            values = group[predictor].to_numpy(float)
            rows.append({
                "predictor": predictor,
                "outcome": "promoted" if int(outcome) else "degraded",
                "n": len(values), "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            })
    return pd.DataFrame(rows)


def plot_descriptor_figure(observations: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "delta_fragment_count": "Fragment count",
        "delta_heavy_atom_ratio": "Heavy-atom ratio",
        "delta_tanimoto_to_product": "Tanimoto to product",
        "delta_ring_count_vs_product": "Ring count vs product",
        "delta_rotatable_bonds": "Rotatable bonds",
    }
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    for axis, predictor in zip(axes.flat, CONTINUOUS_PREDICTORS):
        degraded = observations.loc[observations["promoted"] == 0, predictor].to_numpy(float)
        promoted = observations.loc[observations["promoted"] == 1, predictor].to_numpy(float)
        axis.boxplot([degraded, promoted], labels=["Degraded", "Promoted"], showfliers=False)
        axis.axhline(0, color="0.75", linewidth=0.8)
        axis.set_title(labels[predictor])
        axis.set_ylabel("Augmented minus 2D top-1")
    axes.flat[-1].axis("off")
    figure.suptitle("Post-hoc descriptor differences for reactions with changed mean rank")
    figure.tight_layout()
    temporary = path.with_suffix(path.suffix + ".tmp.png")
    figure.savefig(temporary, dpi=300, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def _fit_logit(frame: pd.DataFrame, predictors: list[str]):
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

    x = sm.add_constant(frame[predictors].astype(float), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerfectSeparationWarning)
        fitted = sm.GLM(frame["promoted"].astype(float), x, family=sm.families.Binomial()).fit()
    if not fitted.converged or not np.isfinite(fitted.params).all():
        raise RuntimeError("Logistic fit failed to converge.")
    return fitted


def logistic_bootstrap(
    observations: pd.DataFrame,
    class3_only: bool,
    n_bootstrap: int,
    seed: int = LOGIT_BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, dict]:
    frame = observations.copy()
    continuous = list(CONTINUOUS_PREDICTORS)
    if class3_only:
        frame = frame[frame["reaction_class"] == 3].copy()
        counts = frame["promoted"].value_counts().to_dict()
        if counts.get(0, 0) < 20 or counts.get(1, 0) < 20:
            return pd.DataFrame(), {
                "status": "not_fitted", "reason": "class 3 has fewer than 20 observations in an outcome",
                "n_degraded": int(counts.get(0, 0)), "n_promoted": int(counts.get(1, 0)),
            }
        predictors = ["delta_rotatable_bonds"]
    else:
        class_dummies = pd.get_dummies(frame["reaction_class"], prefix="class", dtype=float)
        reference_class_column = sorted(class_dummies.columns)[0]
        class_dummies = class_dummies.drop(columns=[reference_class_column])
        frame = pd.concat([frame, class_dummies], axis=1)
        predictors = continuous + list(class_dummies.columns)

    scaling = {}
    for name in continuous:
        mean = float(frame[name].mean())
        std = float(frame[name].std(ddof=0))
        if not math.isfinite(std) or std <= 0:
            if name in predictors:
                predictors.remove(name)
            continue
        frame[name] = (frame[name] - mean) / std
        scaling[name] = {"mean": mean, "population_std": std}
    fitted = _fit_logit(frame, predictors)
    coefficient_names = list(fitted.params.index)
    clusters = frame["canonical_product"].to_numpy(object)
    unique = list(dict.fromkeys(clusters.tolist()))
    indices = [np.flatnonzero(clusters == key) for key in unique]
    rng = np.random.default_rng(seed)
    draws = []
    failed = 0
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(indices), size=len(indices))
        rows = np.concatenate([indices[index] for index in sampled])
        bootstrap_frame = frame.iloc[rows]
        try:
            estimate = _fit_logit(bootstrap_frame, predictors).params.reindex(coefficient_names).to_numpy(float)
            if np.isfinite(estimate).all():
                draws.append(estimate)
            else:
                failed += 1
        except Exception:
            failed += 1
    if not draws:
        raise RuntimeError("Every H2 clustered bootstrap refit failed.")
    draws_array = np.stack(draws)
    low, high = np.percentile(draws_array, [2.5, 97.5], axis=0)
    result_rows = []
    for index, name in enumerate(coefficient_names):
        coefficient = float(fitted.params[name])
        result_rows.append({
            "model": "class3_flexibility" if class3_only else "all_class",
            "term": name, "coefficient": coefficient,
            "coefficient_ci95_low": float(low[index]), "coefficient_ci95_high": float(high[index]),
            "odds_ratio": float(np.exp(np.clip(coefficient, -700, 700))),
            "odds_ratio_ci95_low": float(np.exp(np.clip(low[index], -700, 700))),
            "odds_ratio_ci95_high": float(np.exp(np.clip(high[index], -700, 700))),
        })
    audit = {
        "status": "fitted", "n_observations": len(frame),
        "n_product_clusters": len(unique), "predictors": predictors,
        "standardization": scaling, "bootstrap_success": len(draws),
        "bootstrap_failed_or_separated": failed, "bootstrap_seed": seed,
        "n_bootstrap_requested": n_bootstrap,
        "interpretation": "descriptive and post hoc; conditional on reactions with changed mean reference rank",
    }
    return pd.DataFrame(result_rows), audit


def run_analysis(args) -> dict:
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Analysis requires at least two unique paired seeds.")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty final-analysis folder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    matrices, aligned = load_prediction_matrices(args.prediction_dir, seeds)
    reference = aligned[seeds[0]]
    clusters = np.asarray([canonicalize_smiles(value) or str(value) for value in reference["product_smiles"]], dtype=object)
    top1_delta = matrices["reranked_hit@1"]["augmented"] - matrices["reranked_hit@1"]["baseline"]
    mrr_delta = matrices["reranked_rr"]["augmented"] - matrices["reranked_rr"]["baseline"]
    g1 = {
        "top1": {
            "product_only": product_cluster_bootstrap(
                top1_delta, clusters, args.bootstrap_samples, BOOTSTRAP_SEED
            ),
            "seed_marginal": crossed_product_seed_bootstrap(
                top1_delta, clusters, args.bootstrap_samples, BOOTSTRAP_SEED
            ),
        },
        "mrr": {
            "product_only": product_cluster_bootstrap(
                mrr_delta, clusters, args.bootstrap_samples, BOOTSTRAP_SEED
            ),
            "seed_marginal": crossed_product_seed_bootstrap(
                mrr_delta, clusters, args.bootstrap_samples, BOOTSTRAP_SEED
            ),
        },
    }
    classes = build_class_tests(matrices, reference, args.bootstrap_samples, args.sign_flips)
    flips = build_net_flips(matrices, seeds)
    observations = build_h2_observations(matrices, aligned, seeds)
    distributions = descriptor_distributions(observations)
    all_model, all_audit = logistic_bootstrap(observations, False, args.logit_bootstrap)
    class3_model, class3_audit = logistic_bootstrap(observations, True, args.logit_bootstrap)
    models = pd.concat([all_model, class3_model], ignore_index=True)
    atomic_json(output / "g1_seed_marginal.json", g1)
    atomic_csv(output / "g2_class_tests.csv", classes)
    atomic_csv(output / "g3_net_flips.csv", flips)
    atomic_csv(output / "h2_observations.csv", observations)
    atomic_csv(output / "h1_descriptor_distributions.csv", distributions)
    atomic_csv(output / "h2_logistic_models.csv", models)
    plot_descriptor_figure(observations, output / "h1_descriptor_distributions.png")
    manifest = {
        "protocol_id": "cap10-final-statistics-v1",
        "paired_seeds": list(seeds),
        "prediction_dir": str(Path(args.prediction_dir).resolve()),
        "prediction_sha256": {
            path.name: file_sha256(path)
            for path in sorted(Path(args.prediction_dir).glob("*.csv"))
            if any(f"_{seed}.csv" in path.name for seed in seeds)
        },
        "g2_family": "exactly classes 1--10 conditional Top-1; BH q=0.05",
        "class3_survives_bh_q05": bool(classes.loc[classes["reaction_class"] == 3, "top1_bh_significant_q05"].iloc[0]),
        "net_flip_totals": {
            "improved": int(flips["improved"].sum()), "degraded": int(flips["degraded"].sum()),
            "unchanged": int(flips["unchanged"].sum()), "net": int(flips["net_flips"].sum()),
        },
        "h2_all_class": all_audit,
        "h2_class3": class3_audit,
        "generated_files": [
            "g1_seed_marginal.json", "g2_class_tests.csv", "g3_net_flips.csv",
            "h2_observations.csv", "h1_descriptor_distributions.csv", "h2_logistic_models.csv",
            "h1_descriptor_distributions.png",
        ],
    }
    manifest["generated_sha256"] = {
        name: file_sha256(output / name) for name in manifest["generated_files"]
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--sign-flips", type=int, default=N_SIGN_FLIPS)
    parser.add_argument("--logit-bootstrap", type=int, default=N_BOOTSTRAP)
    return parser


if __name__ == "__main__":
    print(json.dumps(run_analysis(build_parser().parse_args()), indent=2))
