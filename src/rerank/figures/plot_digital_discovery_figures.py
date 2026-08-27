"""Generate the Digital Discovery figures from frozen numerical artifacts.

The script is read-only with respect to all scientific outputs.  Every plotted
quantity is selected by a stable protocol field and the run fails closed when
the expected row is missing or duplicated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


NAVY = "#16324F"
BLUE = "#2F6690"
TEAL = "#2A9D8F"
ORANGE = "#E9A23B"
RED = "#C94C4C"
PURPLE = "#735DA5"
GREY = "#667085"
LIGHT = "#E8EDF2"
INK = "#1D2939"
WHITE = "#FFFFFF"
FULL_WIDTH_IN = 174.0 / 25.4
RASTER_DPI = 600
TOC_WIDTH_IN = 8.0 / 2.54
TOC_HEIGHT_IN = 4.0 / 2.54


def artifact_path(repo: Path, public_path: str, local_path: str) -> Path:
    """Prefer the neutral release path and retain a local provenance fallback."""
    public = repo / public_path
    if public.is_file():
        return public
    local = repo / local_path
    if local.is_file():
        return local
    raise FileNotFoundError(
        "Required frozen artifact is absent from both the public release and "
        f"the research worktree: {public_path!r}, {local_path!r}"
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "axes.edgecolor": "#98A2B3",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "figure.dpi": 150,
            "savefig.dpi": RASTER_DPI,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def one(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}; found {len(selected)}")
    return selected.iloc[0]


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Creator": "rerank.figures.plot_digital_discovery_figures",
        "Subject": "Frozen Digital Discovery candidate-pool analysis",
    }
    fig.savefig(output_dir / f"{stem}.pdf", facecolor=WHITE, metadata=metadata)
    fig.savefig(output_dir / f"{stem}.png", facecolor=WHITE, dpi=RASTER_DPI)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def figure_design(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (0.03, 0.57, 0.18, 0.28, "Frozen benchmark", "USPTO-50K\nofficial test: 5,004\n20 paired seeds"),
        (0.27, 0.57, 0.18, 0.28, "Candidate pools", "Historical cap-10\nAiZynthFinder-only\nLocalRetro-only\nmerged union"),
        (0.51, 0.57, 0.18, 0.28, "Matched reranking", "Prior + 2D\nversus\nPrior + 2D + Uni-Mol"),
        (0.75, 0.57, 0.22, 0.28, "Separated outcomes", "Coverage\nWithin-pool Top-1 / MRR\nHeadroom capture"),
    ]
    colors = [NAVY, BLUE, TEAL, PURPLE]
    for (x, y, w, h, title, body), color in zip(stages, colors):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.0,
            edgecolor=color,
            facecolor=WHITE,
        )
        ax.add_patch(box)
        ax.add_patch(
            FancyBboxPatch(
                (x, y + h - 0.07),
                w,
                0.07,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=0,
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="center", color=WHITE, fontsize=8.0, fontweight="bold")
        ax.text(x + w / 2, y + 0.095, body, ha="center", va="center", color=INK, fontsize=7.0, linespacing=1.25)

    for left, right in zip(stages[:-1], stages[1:]):
        start = (left[0] + left[2] + 0.008, left[1] + left[3] / 2)
        end = (right[0] - 0.008, right[1] + right[3] / 2)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=GREY))

    lower = FancyBboxPatch(
        (0.12, 0.12),
        0.76,
        0.24,
        boxstyle="round,pad=0.018,rounding_size=0.022",
        linewidth=1.2,
        edgecolor=RED,
        facecolor="#FFF7F4",
    )
    ax.add_patch(lower)
    ax.text(0.50, 0.295, "Matched transfer test on 3,814 commonly covered reactions", ha="center", va="center", fontsize=8.8, fontweight="bold", color=RED)
    ax.text(
        0.50,
        0.205,
        "Expanded-pool effect minus historical-pool effect\n"
        "paired product-cluster bootstrap + sign-flip inference",
        ha="center",
        va="center",
        fontsize=7.2,
        linespacing=1.25,
        color=INK,
    )
    for x in (0.36, 0.60):
        ax.add_patch(FancyArrowPatch((x, 0.57), (x, 0.37), arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color=RED))

    save(fig, output_dir, "figure1_dd_design")


def figure_coverage_effects(repo: Path, output_dir: Path) -> None:
    j2 = pd.read_csv(
        artifact_path(
            repo,
            "outputs/transfer_analysis/j1_j2_filtered_v2/j2_headroom_capture.csv",
            "outputs/digital_discovery_round_jk/reanalysis/j1_j2_filtered_v2/j2_headroom_capture.csv",
        )
    )
    k1 = pd.read_csv(
        artifact_path(
            repo,
            "outputs/transfer_analysis/k1/k1_truncated_pool_effects.csv",
            "outputs/digital_discovery_round_jk/reanalysis/k1/k1_truncated_pool_effects.csv",
        )
    )
    ledger = pd.read_csv(
        artifact_path(
            repo,
            "outputs/historical_anchor/numerical_freeze_v2/paired_effects.csv",
            "outputs/jcheminform_revision/numerical_freeze_v2/paired_effects.csv",
        )
    )

    expected = ["historical_cap10", "aizynthfinder_only", "localretro_only", "merged"]
    if list(j2["pool"]) != expected:
        raise ValueError("J2 pool order or membership differs from the frozen four-pool protocol")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.75), gridspec_kw={"width_ratios": [0.92, 1.25]})
    labels = ["Historical\nanchor", "AiZ-only", "LocalRetro-\nonly", "Merged\nunion"]
    colors = [NAVY, BLUE, ORANGE, PURPLE]
    x = np.arange(4)
    coverage = j2["coverage"].to_numpy(float) * 100
    bars = ax1.bar(x, coverage, color=colors, width=0.62)
    for bar, value in zip(bars, coverage):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.65, f"{value:.1f}%", ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    ax1.set_xticks(x, labels)
    ax1.set_ylim(74, 95)
    ax1.set_ylabel("Official-test coverage (%)")
    ax1.set_title("Candidate-pool coverage", pad=10)
    ax1.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax1.spines[["top", "right"]].set_visible(False)
    panel_label(ax1, "A")

    hist = one(ledger, analysis_id="G1", scope="seed_marginal", metric="top1")
    rows = [("Historical cap-10", hist)] + [
        (name, one(k1, pool=pool, metric="top1"))
        for name, pool in zip(["AiZynthFinder-only", "LocalRetro-only", "Merged union"], expected[1:])
    ]
    y = np.arange(4)[::-1]
    for yi, (label, row), color in zip(y, rows, colors):
        estimate = float(row["effect"]) * 100
        if label == "Historical cap-10":
            low = float(row["ci95_low"]) * 100
            high = float(row["ci95_high"]) * 100
        else:
            low = float(row["seed_marginal_ci95_low"]) * 100
            high = float(row["seed_marginal_ci95_high"]) * 100
        ax2.errorbar(estimate, yi, xerr=[[estimate - low], [high - estimate]], fmt="o", color=color, ecolor=color, ms=5.5, capsize=3, lw=1.3)
        ax2.text(high + 0.05, yi, f"{estimate:+.2f} [{low:+.2f}, {high:+.2f}]", va="center", fontsize=7.0, color=INK)
    ax2.axvline(0, color="#98A2B3", lw=0.9, ls="--")
    ax2.set_yticks(y, [r[0] for r in rows])
    ax2.set_xlim(-1.25, 1.40)
    ax2.set_ylim(-0.65, 3.65)
    ax2.set_xlabel("Uni-Mol-augmented minus 2D Top-1 (percentage points)")
    ax2.set_title("Matched 20-seed reranking effect", pad=10)
    ax2.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    panel_label(ax2, "B")

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.86, wspace=0.38)
    save(fig, output_dir, "figure2_dd_coverage_effects")


def figure_transfer_shift(repo: Path, output_dir: Path) -> None:
    transfer = pd.read_csv(
        artifact_path(
            repo,
            "outputs/transfer_analysis/m2b_transfer_inference/m2b_transfer_loss_inference.csv",
            "outputs/digital_discovery_round_jk/reanalysis/m2b_transfer_inference/m2b_transfer_loss_inference.csv",
        )
    )
    summary = pd.read_csv(
        artifact_path(
            repo,
            "outputs/transfer_analysis/m2_pool_shift/m2_pool_summary.csv",
            "outputs/digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_pool_summary.csv",
        )
    )
    pools = ["aizynthfinder_only", "localretro_only", "merged"]
    labels = ["AiZynthFinder-only", "LocalRetro-only", "Merged union"]
    colors = [BLUE, ORANGE, PURPLE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 4.0), gridspec_kw={"width_ratios": [1.22, 0.95]})
    positions = np.array([2.55, 1.55, 0.55])
    offsets = {"top1": 0.16, "mrr": -0.16}
    markers = {"top1": "o", "mrr": "s"}
    for metric in ("top1", "mrr"):
        for pos, pool, color in zip(positions, pools, colors):
            row = one(transfer, pool=pool, metric=metric)
            est = float(row["mean_transfer_loss"]) * 100
            low = float(row["product_cluster_ci95_low"]) * 100
            high = float(row["product_cluster_ci95_high"]) * 100
            ax1.errorbar(est, pos + offsets[metric], xerr=[[est - low], [high - est]], fmt=markers[metric], color=color, ecolor=color, ms=5.0, capsize=2.5, lw=1.2)
    ax1.axvline(0, color="#98A2B3", lw=0.9, ls="--")
    ax1.set_yticks(positions, labels)
    ax1.set_xlim(-2.75, 0.30)
    ax1.set_xlabel("Expanded-pool minus historical effect (percentage points)")
    ax1.set_title("Direct paired loss of transfer", pad=10)
    ax1.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(axis="y", length=0)
    ax1.scatter([], [], marker="o", color=GREY, label="Top-1")
    ax1.scatter([], [], marker="s", color=GREY, label="MRR")
    ax1.legend(loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(0.0, -0.23))
    panel_label(ax1, "A")

    shifted = summary[summary["pool"].isin(pools)].set_index("pool").loc[pools]
    x = np.arange(3)
    width = 0.34
    ax2.bar(x - width / 2, shifted["jaccard_vs_historical_mean"], width, label="Candidate-set Jaccard", color=colors, edgecolor=WHITE)
    ax2.bar(x + width / 2, shifted["kendall_order_vs_historical_mean"], width, label="Shared-order concordance", color=colors, alpha=0.45, hatch="//", edgecolor=WHITE)
    ax2.set_xticks(x, ["AiZ", "LocalRetro", "Merged"])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Mean similarity to historical pool")
    ax2.set_title("Candidate-list shift", pad=10)
    ax2.grid(axis="y", color=LIGHT, linewidth=0.7)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=1, frameon=False)
    panel_label(ax2, "B")

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.26, top=0.86, wspace=0.42)
    save(fig, output_dir, "figure3_dd_transfer_shift")


def figure_descriptor_diagnostics(repo: Path, output_dir: Path) -> None:
    associations = pd.read_csv(
        artifact_path(
            repo,
            "outputs/transfer_analysis/m2_pool_shift/m2_transfer_loss_associations.csv",
            "outputs/digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_transfer_loss_associations.csv",
        )
    )
    pools = ["aizynthfinder_only", "localretro_only", "merged"]
    pool_labels = ["AiZ", "LocalRetro", "Merged"]
    metrics = ["transfer_delta_top1", "transfer_delta_mrr"]
    metric_labels = ["Top-1", "MRR"]
    features = [
        "candidate_count_shift_vs_historical",
        "jaccard_vs_historical",
        "shared_candidate_count",
        "kendall_order_vs_historical",
        "mean_pairwise_morgan_distance_shift_vs_historical",
        "max_nonreference_similarity_shift_vs_historical",
    ]
    feature_labels = [
        "Candidate-count shift",
        "Set Jaccard",
        "Shared count",
        "Shared-order Kendall",
        "Morgan-diversity shift",
        "Max non-reference similarity shift",
    ]
    values = np.empty((len(features), len(pools) * len(metrics)), dtype=float)
    columns: list[str] = []
    for pool_index, (pool, pool_label) in enumerate(zip(pools, pool_labels)):
        for metric_index, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            column_index = pool_index * len(metrics) + metric_index
            columns.append(f"{pool_label}\n{metric_label}")
            for feature_index, feature in enumerate(features):
                row = one(associations, pool=pool, feature=feature, metric=metric)
                values[feature_index, column_index] = float(row["spearman_rho"])

    max_abs = float(np.nanmax(np.abs(values)))
    if max_abs > 0.031:
        raise ValueError(f"Frozen M2 association exceeds reported bound: {max_abs}")

    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.75))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-0.04, vmax=0.04, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), columns)
    ax.set_yticks(np.arange(len(feature_labels)), feature_labels)
    ax.tick_params(axis="both", length=0)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=7.1,
                color=INK,
            )
    ax.set_title(
        "Reaction-level associations between candidate-list shift and transfer loss",
        pad=12,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.035)
    colorbar.set_label("Spearman $\\rho$")
    ax.text(
        0.0,
        -0.18,
        f"All 36 absolute correlations are at most {max_abs:.3f}; values are descriptive, not causal.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=INK,
    )
    fig.subplots_adjust(left=0.30, right=0.94, bottom=0.24, top=0.87)
    save(fig, output_dir, "figure4_dd_descriptor_diagnostics")


def _mean_seed_metric(manifest: dict, arm: str, scope: str, metric: str) -> float:
    per_seed = manifest["per_seed_metrics"][arm]
    expected_seeds = {str(seed) for seed in range(42, 62)}
    if set(per_seed) != expected_seeds:
        raise ValueError(f"Expected paired seeds 42--61 for {arm}; found {sorted(per_seed)}")
    return float(np.mean([float(per_seed[seed][scope][metric]) for seed in sorted(per_seed)]))


def figure_operational_performance(repo: Path, output_dir: Path) -> None:
    compact_path = repo / "outputs/transfer_analysis/figure_inputs/operational_performance.csv"
    if compact_path.is_file():
        compact = pd.read_csv(compact_path)
        pool_order = ["historical_cap10", "aizynthfinder_only", "localretro_only", "merged"]
        system_order = ["candidate_prior", "prior_2d", "prior_2d_unimol"]
        if set(compact["pool"]) != set(pool_order) or set(compact["system"]) != set(system_order):
            raise ValueError("Compact operational-performance table has unexpected pools or systems")
        labels = ["Historical", "AiZ-only", "LocalRetro-only", "Merged union"]
        within = []
        end_to_end = []
        for pool in pool_order:
            rows = compact.set_index(["pool", "system"])
            within.append([float(rows.loc[(pool, system), "within_pool_top1"]) for system in system_order])
            end_to_end.append([float(rows.loc[(pool, system), "end_to_end_top1"]) for system in system_order])
        _plot_operational_performance(output_dir, labels, within, end_to_end)
        return

    historical_path = (
        repo
        / "outputs/jcheminform_revision/tuned_primary/conformer_seed_42/"
        "g1_20seed/test_results/manifest.json"
    )
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_per_seed = historical["per_seed_metrics"]
    expected_seeds = {str(seed) for seed in range(42, 62)}
    if set(historical_per_seed["baseline"]) != expected_seeds:
        raise ValueError("Historical G1 manifest does not contain paired seeds 42--61")

    hist_coverage = 3985 / 5004
    hist_prior = float(next(iter(historical_per_seed["baseline"].values()))["baseline_top1"])
    hist_2d = float(np.mean([float(row["top1"]) for row in historical_per_seed["baseline"].values()]))
    hist_aug = float(np.mean([float(row["top1"]) for row in historical_per_seed["augmented"].values()]))

    labels = ["Historical", "AiZ-only", "LocalRetro-only", "Merged union"]
    within = [[hist_prior, hist_2d, hist_aug]]
    end_to_end = [[hist_prior * hist_coverage, hist_2d * hist_coverage, hist_aug * hist_coverage]]
    for pool in ("aizynthfinder_only", "localretro_only", "merged"):
        path = repo / f"outputs/digital_discovery_round_jk/k1/{pool}/test_results/manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        within.append(
            [
                float(manifest["prior_metrics"]["within_pool"]["top1"]),
                _mean_seed_metric(manifest, "baseline", "within_pool", "top1"),
                _mean_seed_metric(manifest, "augmented", "within_pool", "top1"),
            ]
        )
        end_to_end.append(
            [
                float(manifest["prior_metrics"]["end_to_end"]["top1"]),
                _mean_seed_metric(manifest, "baseline", "end_to_end", "top1"),
                _mean_seed_metric(manifest, "augmented", "end_to_end", "top1"),
            ]
        )

    _plot_operational_performance(output_dir, labels, within, end_to_end)


def _plot_operational_performance(
    output_dir: Path,
    labels: list[str],
    within: list[list[float]],
    end_to_end: list[list[float]],
) -> None:
    within_values = np.asarray(within, dtype=float) * 100
    end_values = np.asarray(end_to_end, dtype=float) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.85), sharex=True)
    x = np.arange(len(labels))
    width = 0.24
    systems = ["Generator prior", "Prior + 2D", "Prior + 2D + Uni-Mol"]
    colors = ["#98A2B3", BLUE, TEAL]
    for system_index, (system, color) in enumerate(zip(systems, colors)):
        offset = (system_index - 1) * width
        ax1.bar(x + offset, within_values[:, system_index], width, color=color, label=system)
        ax2.bar(x + offset, end_values[:, system_index], width, color=color, label=system)

    for ax, title, ylabel, limits in (
        (ax1, "Conditional ranking on covered reactions", "Within-pool Top-1 (%)", (52, 78)),
        (ax2, "End-to-end benchmark score", "End-to-end Top-1 (%)", (44, 66)),
    ):
        ax.set_xticks(x, labels, rotation=16, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*limits)
        ax.set_title(title, pad=10)
        ax.grid(axis="y", color=LIGHT, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax1, "A")
    panel_label(ax2, "B")
    handles, legend_labels = ax2.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.27, top=0.86, wspace=0.28)
    save(fig, output_dir, "figure5_dd_operational_performance")


def toc_graphical_abstract(output_dir: Path) -> None:
    """Write the journal-sized graphical abstract without reusing a paper figure."""
    fig, ax = plt.subplots(figsize=(TOC_WIDTH_IN, TOC_HEIGHT_IN))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.04),
            0.97,
            0.92,
            boxstyle="round,pad=0.008,rounding_size=0.025",
            facecolor="#F8FAFC",
            edgecolor=LIGHT,
            linewidth=1.0,
        )
    )
    ax.text(
        0.50,
        0.86,
        "The same reranker does not imply the same gain",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=NAVY,
    )

    panels = [
        (0.055, "Historical pool", NAVY, "Representation gain", "+0.67 pp"),
        (0.625, "Shifted pool", PURPLE, "Gain attenuates", "~0 pp"),
    ]
    for x0, title, color, message, effect in panels:
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.24),
                0.31,
                0.50,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                facecolor=WHITE,
                edgecolor=color,
                linewidth=1.1,
            )
        )
        ax.text(x0 + 0.155, 0.65, title, ha="center", va="center", fontsize=6.8, fontweight="bold", color=color)
        candidate_y = [0.54, 0.465, 0.39]
        candidate_widths = [0.20, 0.15, 0.10] if x0 < 0.5 else [0.14, 0.21, 0.17]
        for index, (y, width) in enumerate(zip(candidate_y, candidate_widths)):
            face = TEAL if index == 0 else "#D7DEE8"
            ax.add_patch(
                FancyBboxPatch(
                    (x0 + 0.055, y - 0.027),
                    width,
                    0.045,
                    boxstyle="round,pad=0.004,rounding_size=0.012",
                    facecolor=face,
                    edgecolor="none",
                )
            )
        ax.text(x0 + 0.155, 0.335, message, ha="center", va="center", fontsize=5.8, color=INK)
        ax.text(x0 + 0.155, 0.275, effect, ha="center", va="center", fontsize=7.6, fontweight="bold", color=color)

    ax.add_patch(
        FancyArrowPatch(
            (0.39, 0.49),
            (0.60, 0.49),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.3,
            color=RED,
        )
    )
    ax.text(0.495, 0.57, "candidate-pool shift", ha="center", va="center", fontsize=5.8, color=RED)
    ax.text(
        0.50,
        0.095,
        "Evaluate coverage and conditional ranking separately",
        ha="center",
        va="center",
        fontsize=6.1,
        fontweight="bold",
        color=INK,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(output_dir / "toc_graphical_abstract.tiff", dpi=RASTER_DPI, facecolor=WHITE)
        fig.savefig(output_dir / "toc_graphical_abstract.png", dpi=RASTER_DPI, facecolor=WHITE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output_dir = args.output_dir or repo / "paper/digital_discovery/figures"
    configure_style()
    figure_design(output_dir)
    figure_coverage_effects(repo, output_dir)
    figure_transfer_shift(repo, output_dir)
    figure_descriptor_diagnostics(repo, output_dir)
    figure_operational_performance(repo, output_dir)
    toc_graphical_abstract(output_dir)
    print(f"Wrote Digital Discovery figures to {output_dir}")


if __name__ == "__main__":
    main()
