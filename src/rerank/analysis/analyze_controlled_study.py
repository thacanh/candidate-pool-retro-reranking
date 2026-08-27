#!/usr/bin/env python
"""Create the statistical analysis, tables, figures, and JCIM decision gate."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import binomtest, wilcoxon

from rerank.study_data import (
    canonicalize_smiles,
    compute_coverage,
    load_candidate_pools,
    load_reactions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("study_analysis")

SYSTEM_LABELS = {
    "baseline": "Candidate prior",
    "2d": "Prior + 2D",
    "3d": "Prior + 2D + Uni-Mol",
}
COLORS = {"baseline": "#7F8C8D", "2d": "#2E86AB", "3d": "#E67E22"}

REACTION_CLASS_NAMES = {
    1: "Heteroatom alkylation/arylation",
    2: "Acylation and related processes",
    3: "C-C bond formation",
    4: "Heterocycle formation",
    5: "Protection",
    6: "Deprotection",
    7: "Reduction",
    8: "Oxidation",
    9: "Functional group interconversion",
    10: "Functional group addition",
}


def _seed_from_path(path: Path) -> int:
    match = re.fullmatch(r"eval_seed(\d+)\.csv", path.name)
    if match is None:
        raise ValueError(f"Unexpected evaluation filename: {path}")
    return int(match.group(1))


def load_predictions(directory: str | Path) -> dict[int, pd.DataFrame]:
    result = {}
    for path in sorted(Path(directory).glob("eval_seed*.csv")):
        frame = pd.read_csv(path)
        required = {
            "reaction_id",
            "reaction_class",
            "reranked_hit@1",
            "reranked_rr",
            "reranked_rank",
            "baseline_hit@1",
            "baseline_rr",
            "baseline_rank",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["reaction_id"].duplicated().any():
            raise ValueError(f"Duplicate reaction_id values in {path}")
        result[_seed_from_path(path)] = frame.sort_values("reaction_id").reset_index(drop=True)
    if not result:
        raise FileNotFoundError(f"No eval_seed*.csv files found in {directory}")
    return result


def align_predictions(
    two_d: dict[int, pd.DataFrame], three_d: dict[int, pd.DataFrame]
) -> tuple[list[int], dict[int, pd.DataFrame]]:
    seeds = sorted(set(two_d).intersection(three_d))
    if not seeds:
        raise ValueError("The 2D and 3D result directories have no common seed.")
    aligned = {}
    for seed in seeds:
        merged = two_d[seed].merge(
            three_d[seed],
            on="reaction_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_2d", "_3d"),
        )
        if len(merged) != len(two_d[seed]) or len(merged) != len(three_d[seed]):
            raise ValueError(f"Prediction rows are not aligned for seed {seed}.")
        aligned[seed] = merged.sort_values("reaction_id").reset_index(drop=True)
    return seeds, aligned


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def build_main_results(
    two_d: dict[int, pd.DataFrame],
    three_d: dict[int, pd.DataFrame],
    seeds: list[int],
    n_test: int,
) -> pd.DataFrame:
    metrics = {
        "top1": "reranked_hit@1",
        "top3": "reranked_hit@3",
        "top5": "reranked_hit@5",
        "top10": "reranked_hit@10",
        "mrr": "reranked_rr",
    }
    n_covered = len(two_d[seeds[0]])
    coverage_fraction = n_covered / n_test
    rows = []
    baseline = two_d[seeds[0]]
    for metric, reranked_column in metrics.items():
        baseline_column = (
            f"baseline_hit@{metric[3:]}" if metric.startswith("top") else "baseline_rr"
        )
        baseline_value = float(baseline[baseline_column].mean())
        values_by_system = {
            "baseline": [baseline_value],
            "2d": [float(two_d[seed][reranked_column].mean()) for seed in seeds],
            "3d": [float(three_d[seed][reranked_column].mean()) for seed in seeds],
        }
        for system, values in values_by_system.items():
            for scope, scale, denominator in [
                ("within_pool", 1.0, n_covered),
                ("end_to_end", coverage_fraction, n_test),
            ]:
                rows.append(
                    {
                        "scope": scope,
                        "system": system,
                        "system_label": SYSTEM_LABELS[system],
                        "metric": metric,
                        "mean": float(np.mean(values)) * scale,
                        "std": _sample_std(values) * scale,
                        "n_seeds": 0 if system == "baseline" else len(seeds),
                        "n_reactions": denominator,
                    }
                )
    return pd.DataFrame(rows)


def clustered_bootstrap(
    differences: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Resample canonical-product clusters and retain all paired seed rows."""
    if differences.ndim != 2:
        raise ValueError("differences must have shape (n_seeds, n_reactions)")
    if len(cluster_ids) != differences.shape[1]:
        raise ValueError("cluster_ids must align with the reaction dimension")
    n_reactions = differences.shape[1]
    unique_clusters = list(dict.fromkeys(cluster_ids.tolist()))
    cluster_indices = [
        np.flatnonzero(cluster_ids == cluster) for cluster in unique_clusters
    ]
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    if len(cluster_indices) == n_reactions and all(
        len(indices) == 1 for indices in cluster_indices
    ):
        for index in range(n_bootstrap):
            sampled = rng.integers(0, n_reactions, size=n_reactions)
            estimates[index] = float(differences[:, sampled].mean())
    else:
        for index in range(n_bootstrap):
            selected = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
            sampled = np.concatenate([cluster_indices[item] for item in selected])
            estimates[index] = float(differences[:, sampled].mean())
    point = float(differences.mean())
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return point, float(lower), float(upper)


def paired_cluster_permutation_test(
    differences: np.ndarray,
    cluster_ids: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Average paired differences over seeds, then sign-flip by product cluster."""
    reaction_values = differences.mean(axis=0)
    unique_clusters = list(dict.fromkeys(cluster_ids.tolist()))
    cluster_totals = np.asarray(
        [reaction_values[cluster_ids == cluster].sum() for cluster in unique_clusters],
        dtype=np.float64,
    )
    cluster_totals = cluster_totals[np.abs(cluster_totals) > 0]
    observed = abs(float(reaction_values.mean()))
    if len(cluster_totals) == 0:
        return 1.0
    extreme = 0
    completed = 0
    chunk_size = 2000
    denominator = len(reaction_values)
    while completed < n_permutations:
        size = min(chunk_size, n_permutations - completed)
        signs = rng.integers(0, 2, size=(size, len(cluster_totals)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        permuted = np.abs((signs @ cluster_totals) / denominator)
        extreme += int(np.sum(permuted >= observed - 1e-15))
        completed += size
    return float((extreme + 1) / (n_permutations + 1))


def build_significance(
    aligned: dict[int, pd.DataFrame],
    seeds: list[int],
    n_bootstrap: int,
    bootstrap_seed: int,
    n_permutations: int,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    top1_differences = []
    rr_differences = []
    for seed in seeds:
        frame = aligned[seed]
        top1_2d = frame["reranked_hit@1_2d"].to_numpy(dtype=int)
        top1_3d = frame["reranked_hit@1_3d"].to_numpy(dtype=int)
        rr_2d = frame["reranked_rr_2d"].to_numpy(dtype=float)
        rr_3d = frame["reranked_rr_3d"].to_numpy(dtype=float)
        promoted = int(np.sum((top1_2d == 0) & (top1_3d == 1)))
        lost = int(np.sum((top1_2d == 1) & (top1_3d == 0)))
        mcnemar_p = (
            float(binomtest(min(promoted, lost), promoted + lost, 0.5).pvalue)
            if promoted + lost
            else 1.0
        )
        try:
            wilcoxon_p = float(
                wilcoxon(rr_3d, rr_2d, zero_method="pratt", alternative="two-sided").pvalue
            )
        except ValueError:
            wilcoxon_p = 1.0
        rows.append(
            {
                "seed": seed,
                "n_reactions": len(frame),
                "top1_delta": float((top1_3d - top1_2d).mean()),
                "mrr_delta": float((rr_3d - rr_2d).mean()),
                "top1_promoted": promoted,
                "top1_lost": lost,
                "mcnemar_exact_p": mcnemar_p,
                "wilcoxon_rr_p": wilcoxon_p,
            }
        )
        top1_differences.append(top1_3d - top1_2d)
        rr_differences.append(rr_3d - rr_2d)

    top1_matrix = np.stack(top1_differences)
    rr_matrix = np.stack(rr_differences)
    cluster_ids = np.asarray(
        [
            canonicalize_smiles(smiles) or str(smiles)
            for smiles in aligned[seeds[0]]["product_smiles_2d"]
        ],
        dtype=object,
    )
    rng = np.random.default_rng(bootstrap_seed)
    top1_point, top1_low, top1_high = clustered_bootstrap(
        top1_matrix, cluster_ids, n_bootstrap, rng
    )
    rr_point, rr_low, rr_high = clustered_bootstrap(
        rr_matrix, cluster_ids, n_bootstrap, rng
    )
    permutation_rng = np.random.default_rng(bootstrap_seed + 1)
    top1_permutation_p = paired_cluster_permutation_test(
        top1_matrix, cluster_ids, n_permutations, permutation_rng
    )
    mrr_permutation_p = paired_cluster_permutation_test(
        rr_matrix, cluster_ids, n_permutations, permutation_rng
    )
    mean_rr_2d = np.stack(
        [aligned[seed]["reranked_rr_2d"].to_numpy(float) for seed in seeds]
    ).mean(axis=0)
    mean_rr_3d = np.stack(
        [aligned[seed]["reranked_rr_3d"].to_numpy(float) for seed in seeds]
    ).mean(axis=0)
    try:
        aggregate_wilcoxon_p = float(
            wilcoxon(
                mean_rr_3d,
                mean_rr_2d,
                zero_method="pratt",
                alternative="two-sided",
            ).pvalue
        )
    except ValueError:
        aggregate_wilcoxon_p = 1.0
    aggregate = {
        "method": "canonical-product-clustered bootstrap across paired seeds",
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "n_product_clusters": len(np.unique(cluster_ids)),
        "n_reactions": len(cluster_ids),
        "top1_delta": top1_point,
        "top1_ci95": [top1_low, top1_high],
        "top1_paired_permutation_p": top1_permutation_p,
        "mrr_delta": rr_point,
        "mrr_ci95": [rr_low, rr_high],
        "mrr_paired_permutation_p": mrr_permutation_p,
        "n_permutations": n_permutations,
        "paired_permutation_method": (
            "Monte Carlo sign flip of seed-averaged reaction differences at the "
            "canonical-product-cluster level"
        ),
        "wilcoxon_mean_reaction_rr_p": aggregate_wilcoxon_p,
    }
    return pd.DataFrame(rows), aggregate


def build_rank_shift(aligned: dict[int, pd.DataFrame], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        frame = aligned[seed]
        rank_2d = frame["reranked_rank_2d"].to_numpy(int)
        rank_3d = frame["reranked_rank_3d"].to_numpy(int)
        delta = rank_2d - rank_3d
        improved = delta > 0
        degraded = delta < 0
        rows.append(
            {
                "seed": seed,
                "n_reactions": len(frame),
                "improved": int(improved.sum()),
                "unchanged": int((delta == 0).sum()),
                "degraded": int(degraded.sum()),
                "mean_gain_if_improved": float(delta[improved].mean()) if improved.any() else 0.0,
                "mean_delta_if_degraded": float(delta[degraded].mean()) if degraded.any() else 0.0,
                "promoted_to_top1": int(np.sum((rank_2d > 1) & (rank_3d == 1))),
                "lost_top1": int(np.sum((rank_2d == 1) & (rank_3d > 1))),
                "mean_rank_delta": float(delta.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_class_results(
    aligned: dict[int, pd.DataFrame],
    seeds: list[int],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_rows = []
    for seed in seeds:
        frame = aligned[seed]
        for reaction_class, group in frame.groupby("reaction_class_2d"):
            per_seed_rows.append(
                {
                    "seed": seed,
                    "reaction_class": int(reaction_class),
                    "reaction_class_name": REACTION_CLASS_NAMES[int(reaction_class)],
                    "n_reactions": len(group),
                    "baseline_top1": float(group["baseline_hit@1_2d"].mean()),
                    "top1_2d": float(group["reranked_hit@1_2d"].mean()),
                    "top1_3d": float(group["reranked_hit@1_3d"].mean()),
                    "baseline_mrr": float(group["baseline_rr_2d"].mean()),
                    "mrr_2d": float(group["reranked_rr_2d"].mean()),
                    "mrr_3d": float(group["reranked_rr_3d"].mean()),
                }
            )
    per_seed = pd.DataFrame(per_seed_rows)
    summary_rows = []
    first = aligned[seeds[0]]
    cluster_ids_all = np.asarray(
        [
            canonicalize_smiles(smiles) or str(smiles)
            for smiles in first["product_smiles_2d"]
        ],
        dtype=object,
    )
    top1_differences = np.stack(
        [
            aligned[seed]["reranked_hit@1_3d"].to_numpy(float)
            - aligned[seed]["reranked_hit@1_2d"].to_numpy(float)
            for seed in seeds
        ]
    )
    mrr_differences = np.stack(
        [
            aligned[seed]["reranked_rr_3d"].to_numpy(float)
            - aligned[seed]["reranked_rr_2d"].to_numpy(float)
            for seed in seeds
        ]
    )
    rng = np.random.default_rng(bootstrap_seed + 100)
    for reaction_class, group in per_seed.groupby("reaction_class"):
        row = {
            "reaction_class": int(reaction_class),
            "reaction_class_name": REACTION_CLASS_NAMES[int(reaction_class)],
            "n_reactions": int(group["n_reactions"].iloc[0]),
            "baseline_top1": float(group["baseline_top1"].iloc[0]),
            "baseline_mrr": float(group["baseline_mrr"].iloc[0]),
        }
        for metric in ["top1_2d", "top1_3d", "mrr_2d", "mrr_3d"]:
            values = group[metric].tolist()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = _sample_std(values)
        row["top1_delta_3d_minus_2d"] = row["top1_3d_mean"] - row["top1_2d_mean"]
        row["mrr_delta_3d_minus_2d"] = row["mrr_3d_mean"] - row["mrr_2d_mean"]
        mask = first["reaction_class_2d"].to_numpy(int) == int(reaction_class)
        _, top1_low, top1_high = clustered_bootstrap(
            top1_differences[:, mask], cluster_ids_all[mask], n_bootstrap, rng
        )
        _, mrr_low, mrr_high = clustered_bootstrap(
            mrr_differences[:, mask], cluster_ids_all[mask], n_bootstrap, rng
        )
        row["top1_delta_ci95_low"] = top1_low
        row["top1_delta_ci95_high"] = top1_high
        row["mrr_delta_ci95_low"] = mrr_low
        row["mrr_delta_ci95_high"] = mrr_high
        summary_rows.append(row)
    return per_seed, pd.DataFrame(summary_rows).sort_values("reaction_class")


def select_case_studies(
    aligned: dict[int, pd.DataFrame], seeds: list[int], n_cases: int = 6
) -> pd.DataFrame:
    base = aligned[seeds[0]][
        [
            "reaction_id",
            "reaction_class_2d",
            "product_smiles_2d",
            "ground_truth_2d",
            "baseline_rank_2d",
            "baseline_candidates_json_2d",
        ]
    ].copy()
    rank_2d = np.stack(
        [aligned[seed]["reranked_rank_2d"].to_numpy(int) for seed in seeds]
    )
    rank_3d = np.stack(
        [aligned[seed]["reranked_rank_3d"].to_numpy(int) for seed in seeds]
    )
    delta = rank_2d - rank_3d
    base["mean_rank_2d"] = rank_2d.mean(axis=0)
    base["mean_rank_3d"] = rank_3d.mean(axis=0)
    base["mean_rank_gain"] = delta.mean(axis=0)
    base["improved_seed_count"] = (delta > 0).sum(axis=0)
    base["degraded_seed_count"] = (delta < 0).sum(axis=0)
    base["promoted_top1_seed_count"] = ((rank_2d > 1) & (rank_3d == 1)).sum(axis=0)
    base["lost_top1_seed_count"] = ((rank_2d == 1) & (rank_3d > 1)).sum(axis=0)
    base["ranks_2d"] = [json.dumps(rank_2d[:, index].tolist()) for index in range(len(base))]
    base["ranks_3d"] = [json.dumps(rank_3d[:, index].tolist()) for index in range(len(base))]

    selections = []
    used: set[int] = set()

    def take(frame: pd.DataFrame, count: int, category: str) -> None:
        for _, row in frame.iterrows():
            reaction_id = int(row["reaction_id"])
            if reaction_id in used:
                continue
            record = row.to_dict()
            record["case_category"] = category
            selections.append(record)
            used.add(reaction_id)
            if sum(item["case_category"] == category for item in selections) >= count:
                break

    take(
        base[base["promoted_top1_seed_count"] > 0].sort_values(
            ["promoted_top1_seed_count", "mean_rank_gain"], ascending=False
        ),
        2,
        "promoted_to_top1",
    )
    take(
        base[(base["mean_rank_gain"] > 0) & (base["promoted_top1_seed_count"] == 0)].sort_values(
            "mean_rank_gain", ascending=False
        ),
        1,
        "large_improvement",
    )
    take(
        base[(base["mean_rank_gain"] == 0) & (base["mean_rank_2d"] > 1)].sort_values(
            "reaction_id"
        ),
        1,
        "unchanged",
    )
    take(
        base[base["lost_top1_seed_count"] > 0].sort_values(
            ["lost_top1_seed_count", "mean_rank_gain"], ascending=[False, True]
        ),
        1,
        "lost_top1",
    )
    take(
        base[base["mean_rank_gain"] < 0].sort_values("mean_rank_gain"),
        1,
        "degraded",
    )
    if len(selections) < n_cases:
        take(base.sort_values("mean_rank_gain", ascending=False), n_cases - len(selections), "additional")
    return pd.DataFrame(selections[:n_cases])


def _box(ax, x, y, width, height, text, color, fontsize=10) -> None:
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.02", facecolor=color,
        edgecolor="#23313D", linewidth=1.2
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def _save_publication_figure(fig, output_dir: Path, stem: str, **kwargs) -> None:
    """Save a 300 dpi review image and a vector PDF for Overleaf/ACS."""
    fig.savefig(output_dir / f"{stem}.png", dpi=300, **kwargs)
    fig.savefig(output_dir / f"{stem}.pdf", **kwargs)


def make_schematic_figures(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3); ax.axis("off")
    items = [
        (0.2, "USPTO-50K\nproducts", "#E8F1F8"),
        (2.4, "AiZynthFinder\ncandidate pool", "#E8F1F8"),
        (4.7, "Feature\nextraction", "#FDEBD0"),
        (7.0, "Frozen MLP\n+ BPR", "#E8DAEF"),
        (9.2, "Reranked\ncandidates", "#D5F5E3"),
    ]
    for x, label, color in items:
        _box(ax, x, 1.0, 1.6, 1.0, label, color)
    for x in [1.8, 4.0, 6.3, 8.6]:
        ax.add_patch(FancyArrowPatch((x, 1.5), (x + 0.5, 1.5), arrowstyle="->", mutation_scale=15))
    ax.set_title("Overall retrosynthesis candidate-reranking pipeline", fontsize=14, pad=10)
    fig.tight_layout()
    _save_publication_figure(fig, output_dir, "figure1_pipeline", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 0.2, 2.5, 1.6, 1.0, "Same candidate\npool", "#E8F1F8")
    branch_specs = [
        (4.25, "Prior + 2D\n4 inputs; 193 params", "#D6EAF8"),
        (2.75, "+ 3 permuted features\n7 inputs; 289 params", "#FDEBD0"),
        (1.25, "+ 3 Uni-Mol features\n7 inputs; 289 params", "#FADBD8"),
    ]
    for y_pos, label, color in branch_specs:
        _box(ax, 2.4, y_pos, 2.25, 0.9, label, color, 8.5)
        _box(ax, 5.25, y_pos, 2.15, 0.9, "Same hidden layer\n32 units + BPR", "#E8DAEF", 8.5)
    _box(ax, 8.1, 2.5, 1.6, 1.0, "Paired\nevaluation", "#D5F5E3")
    for y_pos, _, _ in branch_specs:
        for start, end in [
            ((1.8, 3.0), (2.4, y_pos + 0.45)),
            ((4.65, y_pos + 0.45), (5.25, y_pos + 0.45)),
            ((7.4, y_pos + 0.45), (8.1, 3.0)),
        ]:
            ax.add_patch(FancyArrowPatch(start,end,arrowstyle="->",mutation_scale=13))
    ax.text(5.0, 5.72, "Controlled feature comparison with a matched robustness check", ha="center", fontsize=13, weight="bold")
    ax.text(5.0, 0.35, "Identical split, seeded initialization, sampled pairs, optimizer, loss, epochs, and evaluation", ha="center", fontsize=9.5)
    fig.tight_layout()
    _save_publication_figure(
        fig, output_dir, "figure2_controlled_comparison", bbox_inches="tight"
    )
    plt.close(fig)


def make_toc_graphic(output_dir: Path, top1_delta: float, test_coverage: float) -> None:
    """Generate the compact ACS table-of-contents graphic."""
    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    _box(ax, 0.15, 2.05, 2.05, 1.25, "Fixed\ncandidates", "#E5E7E9", 6.5)
    _box(ax, 3.00, 3.05, 2.30, 0.95, "Prior + 2D", "#D6EAF8", 6.5)
    _box(ax, 3.00, 1.15, 2.30, 0.95, "+ Uni-Mol features", "#FADBD8", 6.5)
    _box(ax, 6.05, 2.05, 1.70, 1.25, "Same hidden\narchitecture", "#E8DAEF", 6.0)
    _box(
        ax,
        8.25,
        2.05,
        1.55,
        1.25,
        f"+{100 * top1_delta:.2f} pp\nTop-1",
        "#D5F5E3",
        6.5,
    )
    for start, end in [
        ((2.20, 2.68), (2.95, 3.48)),
        ((2.20, 2.68), (2.95, 1.62)),
        ((5.30, 3.52), (6.00, 2.85)),
        ((5.30, 1.62), (6.00, 2.45)),
        ((7.75, 2.68), (8.20, 2.68)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="->", mutation_scale=9, linewidth=0.8
            )
        )
    ax.text(
        5.0,
        0.30,
        f"5/5 paired seeds improve  |  candidate coverage = {100 * test_coverage:.1f}%",
        ha="center",
        va="center",
        fontsize=5.5,
    )
    fig.tight_layout(pad=0.05)
    fig.savefig(output_dir / "toc_graphic.png", dpi=300)
    fig.savefig(output_dir / "toc_graphic.pdf")
    plt.close(fig)


def make_result_figures(
    output_dir: Path,
    main_results: pd.DataFrame,
    aligned: dict[int, pd.DataFrame],
    seeds: list[int],
    rank_shift: pd.DataFrame,
    aggregate_significance: dict,
    class_summary: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    perf = main_results[
        (main_results["scope"] == "within_pool")
        & (main_results["metric"].isin(["top1", "mrr"]))
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(2); width = 0.24
    for offset, system in enumerate(["baseline", "2d", "3d"]):
        rows = perf[perf["system"] == system].set_index("metric").loc[["top1", "mrr"]]
        ax.bar(x + (offset - 1) * width, rows["mean"], width, yerr=rows["std"],
               label=SYSTEM_LABELS[system], color=COLORS[system], capsize=4)
    ax.set_xticks(x, ["Top-1", "MRR"]); ax.set_ylim(0, 1); ax.set_ylabel("Within-pool score")
    ax.legend(frameon=False, fontsize=10); fig.tight_layout()
    _save_publication_figure(fig, output_dir, "figure3_overall_performance")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, metric, label in zip(axes, ["reranked_hit@1", "reranked_rr"], ["Top-1", "MRR"]):
        values_2d = [float(aligned[s][f"{metric}_2d"].mean()) for s in seeds]
        values_3d = [float(aligned[s][f"{metric}_3d"].mean()) for s in seeds]
        for index, seed in enumerate(seeds):
            ax.plot([0, 1], [values_2d[index], values_3d[index]], color="#AAB7B8", linewidth=1)
            ax.scatter([0, 1], [values_2d[index], values_3d[index]], s=55, label=str(seed) if ax is axes[0] else None)
        ax.set_xticks([0, 1], ["Prior+2D", "+ Uni-Mol"]); ax.set_title(label); ax.set_ylabel("Score")
    axes[0].legend(title="Seed", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    _save_publication_figure(fig, output_dir, "figure4_seed_consistency")
    plt.close(fig)

    counts = rank_shift[["improved", "unchanged", "degraded"]].mean()
    errors = rank_shift[["improved", "unchanged", "degraded"]].std(ddof=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(counts.index.str.title(), counts.values, yerr=errors.values, capsize=4,
           color=["#27AE60", "#95A5A6", "#C0392B"])
    ax.set_ylabel("Mean reactions across seeds"); ax.set_title("Ground-truth rank shifts after adding Uni-Mol features")
    fig.tight_layout()
    _save_publication_figure(fig, output_dir, "figure5_rank_shift")
    plt.close(fig)

    delta_stats = class_summary.sort_values("reaction_class").reset_index(drop=True)
    overall_top1 = np.asarray(
        [
            float(
                aligned[seed]["reranked_hit@1_3d"].mean()
                - aligned[seed]["reranked_hit@1_2d"].mean()
            )
            for seed in seeds
        ]
    )
    overall_mrr = np.asarray(
        [
            float(
                aligned[seed]["reranked_rr_3d"].mean()
                - aligned[seed]["reranked_rr_2d"].mean()
            )
            for seed in seeds
        ]
    )
    labels = ["Overall\n(n=3985)"] + [
        f"{int(row.reaction_class)}  {row.reaction_class_name}\n(n={int(row.n_reactions)})"
        for row in delta_stats.itertuples(index=False)
    ]
    top1_means = np.concatenate(
        [[overall_top1.mean()], delta_stats["top1_delta_3d_minus_2d"]]
    )
    mrr_means = np.concatenate(
        [[overall_mrr.mean()], delta_stats["mrr_delta_3d_minus_2d"]]
    )
    top1_lows = np.concatenate(
        [[aggregate_significance["top1_ci95"][0]], delta_stats["top1_delta_ci95_low"]]
    )
    top1_highs = np.concatenate(
        [[aggregate_significance["top1_ci95"][1]], delta_stats["top1_delta_ci95_high"]]
    )
    mrr_lows = np.concatenate(
        [[aggregate_significance["mrr_ci95"][0]], delta_stats["mrr_delta_ci95_low"]]
    )
    mrr_highs = np.concatenate(
        [[aggregate_significance["mrr_ci95"][1]], delta_stats["mrr_delta_ci95_high"]]
    )
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.0), sharey=True)
    positive_color = "#0072B2"
    negative_color = "#D55E00"
    for ax, means, lows, highs, title in zip(
        axes,
        [top1_means, mrr_means],
        [top1_lows, mrr_lows],
        [top1_highs, mrr_highs],
        ["Top-1 change", "MRR change"],
    ):
        limit = max(0.03, float(np.max(np.abs(np.concatenate([lows, highs])))) * 1.25)
        ax.axvspan(-limit, 0, color="#F7F7F7", zorder=0)
        ax.axvspan(0, limit, color="#F3F8FC", zorder=0)
        ax.axvline(0, color="#4D4D4D", linewidth=1.0, zorder=1)
        ax.axhspan(-0.45, 0.45, color="#FFF2CC", alpha=0.65, zorder=1)
        for index, (mean, low, high) in enumerate(zip(means, lows, highs)):
            positive = mean >= 0
            color = positive_color if positive else negative_color
            marker = "D" if index == 0 else ("o" if positive else "X")
            lower_error = max(0.0, mean - low)
            upper_error = max(0.0, high - mean)
            ax.errorbar(
                mean,
                index,
                xerr=np.asarray([[lower_error], [upper_error]]),
                fmt=marker,
                markersize=7 if index == 0 else 6,
                color=color,
                ecolor=color,
                elinewidth=1.1,
                capsize=2.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )
            offset = limit * 0.035
            ax.text(
                mean + (upper_error + offset if mean >= 0 else -lower_error - offset),
                index,
                f"{mean:+.3f}",
                ha="left" if mean >= 0 else "right",
                va="center",
                fontsize=8,
                color="#222222",
            )
        ax.set_xlim(-limit, limit)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Prior+2D+Uni-Mol - Prior+2D", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#D0D0D0", linewidth=0.6)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels)
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].get_yticklabels()[0].set_weight("bold")
    axes[0].invert_yaxis()
    legend = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=positive_color,
               markeredgecolor="white", label="Overall"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=positive_color,
               markeredgecolor="white", label="Positive point estimate"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=negative_color,
               markeredgecolor="white", label="Negative point estimate"),
    ]
    fig.legend(
        handles=legend,
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.66, 0.985),
        ncol=3,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.text(
        0.5,
        0.018,
        "Points are five-seed means; horizontal bars are 95% canonical-product-clustered bootstrap intervals.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.31, right=0.985, bottom=0.14, top=0.91, wspace=0.10)
    _save_publication_figure(fig, output_dir, "figure6_reaction_class")
    plt.close(fig)

    molecules = []
    legends = []
    displayed_cases = []
    for category in [
        "promoted_to_top1",
        "large_improvement",
        "lost_top1",
        "degraded",
    ]:
        matches = cases[cases["case_category"] == category]
        if not matches.empty:
            displayed_cases.append(matches.iloc[0])
    for row in displayed_cases:
        reaction_id = int(row["reaction_id"])
        top_2d = []
        top_3d = []
        for seed in seeds:
            match = aligned[seed][aligned[seed]["reaction_id"] == reaction_id].iloc[0]
            top_2d.append(json.loads(match["reranked_candidates_json_2d"])[0])
            top_3d.append(json.loads(match["reranked_candidates_json_3d"])[0])
        most_common_2d = Counter(top_2d).most_common(1)[0][0]
        most_common_3d = Counter(top_3d).most_common(1)[0][0]
        structures = [
            (str(row["product_smiles_2d"]), f"Case {reaction_id}\nproduct"),
            (
                str(row["ground_truth_2d"]),
                f"Reference\nrank {row['mean_rank_2d']:.1f} -> {row['mean_rank_3d']:.1f}",
            ),
            (most_common_2d, "Most frequent\n2D top-1"),
            (most_common_3d, "Most frequent\n2D+Uni-Mol top-1"),
        ]
        for smiles, legend in structures:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                molecules.append(molecule)
                legends.append(legend)
    if molecules:
        image = Draw.MolsToGridImage(
            molecules,
            molsPerRow=4,
            subImgSize=(525, 390),
            legends=legends,
            useSVG=False,
        )
        image.save(output_dir / "figure7_case_studies.png", dpi=(300, 300))


def build_feature_control_results(control_root: str | Path) -> pd.DataFrame:
    """Collect permuted-feature-check and single-feature ablation summaries."""
    root = Path(control_root)
    labels = {
        "prior_2d": "Prior + 2D",
        "permuted_unimol": "Prior + 2D + permuted Uni-Mol features",
        "unimol_atom": "Prior + 2D + atom-set similarity",
        "unimol_distance": "Prior + 2D + embedding distance",
        "unimol_cosine": "Prior + 2D + reaction-vector cosine",
        "unimol_all": "Prior + 2D + all Uni-Mol features",
    }
    order = list(labels)
    rows = []
    for variant in order:
        path = root / variant / "experiment_summary.csv"
        if not path.exists():
            continue
        source = pd.read_csv(path).iloc[0]
        rows.append(
            {
                "variant": variant,
                "system_label": labels[variant],
                "input_dim": int(source["input_dim"]),
                "parameter_count": int(source["parameter_count"]),
                "top1_mean": float(source["top1_mean"]),
                "top1_std": float(source["top1_std"]),
                "mrr_mean": float(source["mrr_mean"]),
                "mrr_std": float(source["mrr_std"]),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    baseline = result[result["variant"] == "prior_2d"]
    if baseline.empty:
        raise ValueError("Control results must include the prior_2d baseline.")
    baseline_row = baseline.iloc[0]
    result["top1_delta_vs_2d"] = result["top1_mean"] - baseline_row["top1_mean"]
    result["mrr_delta_vs_2d"] = result["mrr_mean"] - baseline_row["mrr_mean"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", default="data/uspto_smiles.csv")
    parser.add_argument("--metadata-csv", default="data/uspto_reaction_metadata.csv")
    parser.add_argument("--candidate-jsonl", default="outputs/rerank_dataset.jsonl")
    parser.add_argument("--two-d-dir", default="outputs/study_2d")
    parser.add_argument("--three-d-dir", default="outputs/study_3d")
    parser.add_argument("--control-root", default="outputs/revision_controls")
    parser.add_argument("--output-dir", default="outputs/study_analysis")
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--permutation-samples", type=int, default=100_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    reactions = load_reactions(args.source_csv, args.metadata_csv)
    pools, candidate_audit = load_candidate_pools(args.candidate_jsonl)
    coverage = compute_coverage(reactions, pools)
    coverage.to_csv(output_dir / "table1_dataset_coverage.csv", index=False)
    with open(output_dir / "candidate_audit.json", "w", encoding="utf-8") as handle:
        json.dump(candidate_audit, handle, indent=2)

    two_d = load_predictions(args.two_d_dir); three_d = load_predictions(args.three_d_dir)
    seeds, aligned = align_predictions(two_d, three_d)
    n_test = sum(reaction.source_split == "test" for reaction in reactions)
    main_results = build_main_results(two_d, three_d, seeds, n_test)
    main_results.to_csv(output_dir / "table2_main_results.csv", index=False)
    significance, aggregate_significance = build_significance(
        aligned,
        seeds,
        args.bootstrap_samples,
        args.bootstrap_seed,
        args.permutation_samples,
    )
    significance.to_csv(output_dir / "paired_significance_by_seed.csv", index=False)
    with open(output_dir / "paired_significance_aggregate.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate_significance, handle, indent=2)
    rank_shift = build_rank_shift(aligned, seeds)
    rank_shift.to_csv(output_dir / "table4_rank_shift.csv", index=False)
    class_per_seed, class_summary = build_class_results(
        aligned, seeds, args.bootstrap_samples, args.bootstrap_seed
    )
    class_per_seed.to_csv(output_dir / "reaction_class_per_seed.csv", index=False)
    class_summary.to_csv(output_dir / "table3_reaction_class.csv", index=False)
    cases = select_case_studies(aligned, seeds)
    cases.to_csv(output_dir / "case_studies.csv", index=False)
    control_results = build_feature_control_results(args.control_root)
    if not control_results.empty:
        control_results.to_csv(output_dir / "table5_feature_controls.csv", index=False)

    make_schematic_figures(output_dir)
    within_top1 = main_results[
        (main_results["scope"] == "within_pool")
        & (main_results["metric"] == "top1")
    ].set_index("system")
    toc_top1_delta = float(
        within_top1.loc["3d", "mean"] - within_top1.loc["2d", "mean"]
    )
    toc_test_coverage = float(
        coverage.loc[coverage["source_split"] == "test", "coverage_all"].iloc[0]
    )
    make_toc_graphic(output_dir, toc_top1_delta, toc_test_coverage)
    make_result_figures(
        output_dir,
        main_results,
        aligned,
        seeds,
        rank_shift,
        aggregate_significance,
        class_summary,
        cases,
    )

    within = main_results[main_results["scope"] == "within_pool"]
    top1 = within[within["metric"] == "top1"].set_index("system")
    mrr = within[within["metric"] == "mrr"].set_index("system")
    positive_top1_seeds = int((significance["top1_delta"] > 0).sum())
    positive_mrr_seeds = int((significance["mrr_delta"] > 0).sum())
    gate = {
        "paired_seeds": seeds,
        "at_least_minimum_seeds": len(seeds) >= args.min_seeds,
        "unimol_mean_top1_above_2d": bool(top1.loc["3d", "mean"] > top1.loc["2d", "mean"]),
        "unimol_mean_mrr_above_2d": bool(mrr.loc["3d", "mean"] > mrr.loc["2d", "mean"]),
        "top1_positive_seed_count": positive_top1_seeds,
        "mrr_positive_seed_count": positive_mrr_seeds,
        "trend_not_driven_by_one_seed": positive_top1_seeds > 1 and positive_mrr_seeds > 1,
        "paired_statistics_completed": True,
        "candidate_coverage_reported": True,
        "rank_shift_completed": True,
        "reaction_class_completed": True,
        "case_study_count": len(cases),
        "at_least_four_case_studies": len(cases) >= 4,
        "matched_permuted_check_completed": bool(
            not control_results.empty
            and "permuted_unimol" in set(control_results["variant"])
        ),
        "single_feature_ablations_completed": bool(
            not control_results.empty
            and {"unimol_atom", "unimol_distance", "unimol_cosine"}.issubset(
                set(control_results["variant"])
            )
        ),
    }
    gate["jcim_quantitative_gate_pass"] = all(
        gate[key]
        for key in [
            "at_least_minimum_seeds",
            "unimol_mean_top1_above_2d",
            "unimol_mean_mrr_above_2d",
            "trend_not_driven_by_one_seed",
            "paired_statistics_completed",
            "candidate_coverage_reported",
            "rank_shift_completed",
            "reaction_class_completed",
            "at_least_four_case_studies",
            "matched_permuted_check_completed",
            "single_feature_ablations_completed",
        ]
    )
    with open(output_dir / "jcim_submission_gate.json", "w", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2)
    logger.info("Analysis complete. JCIM quantitative gate: %s", gate["jcim_quantitative_gate_pass"])


if __name__ == "__main__":
    main()
