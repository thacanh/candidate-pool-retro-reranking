"""Generate the revised main-text figures from numerical freeze v2.

This module is deliberately read-only with respect to the numerical ledger.  It
selects rows by their stable workstream/analysis/comparison identifiers and
writes publication figures only.  Missing or duplicated rows fail closed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#173F5F"
BLUE = "#20639B"
TEAL = "#2A9D8F"
ORANGE = "#F4A261"
RED = "#C44536"
PURPLE = "#6C5B7B"
GREY = "#667085"
LIGHT_GREY = "#E8EDF2"
INK = "#1D2939"
FULL_WIDTH_IN = 174.0 / 25.4
RASTER_DPI = 600


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "axes.edgecolor": "#98A2B3",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
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
        raise ValueError(f"Expected exactly one ledger row for {filters}; found {len(selected)}")
    return selected.iloc[0]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, *, with_png: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Creator": "rerank.figures.plot_revision_main_figures",
        "Subject": "Journal of Cheminformatics revision; numerical freeze v2",
    }
    fig.savefig(output_dir / f"{stem}.pdf", facecolor="white", metadata=metadata)
    if with_png:
        fig.savefig(output_dir / f"{stem}.png", facecolor="white", dpi=RASTER_DPI)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.15,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=INK,
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none"},
    )


def figure_design(output_dir: Path, background_path: Path, *, with_png: bool) -> None:
    """Overlay exact protocol text on the generated, non-data workflow artwork."""
    if not background_path.is_file():
        raise FileNotFoundError(f"Figure 1 background not found: {background_path}")
    background = plt.imread(background_path)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.86))
    ax.imshow(background, extent=(0, 1, 0, 1), aspect="auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    centers = [0.111, 0.315, 0.508, 0.696, 0.889]
    top = [
        ("Data", "USPTO-50K\n40,029 train\n5,004 validation / test\n233 overlaps removed"),
        ("Candidate pool", "Historical cap-10\nanchor\n3,985 test reactions\ncovered"),
        ("Paired arms", "Prior + 2D\nversus\n2D + Uni-Mol"),
        ("Model selection", "Validation only\n81 configurations\nper arm\nPrior transform frozen"),
        ("Frozen test", "20 paired seeds\nTop-1 and MRR\nClustered intervals"),
    ]
    bottom = [
        ("Conformers", "5 × 5 gate\n10-conformer mean"),
        ("Encoders", "Morgan; GROVER\nProjected probe"),
        ("Rankers", "Equal capacity\nLightGBM; RXN-EBM"),
        ("Candidate pools", "AiZynthFinder\nLocalRetro\nMerged cap-50"),
        ("Outcomes", "Round trip; stereo\nSalts; classes; flips"),
    ]
    for x, (title, body) in zip(centers, top):
        ax.text(x, 0.725, title, ha="center", va="center", fontsize=6.7, fontweight="bold", color=INK)
        ax.text(x, 0.605, body, ha="center", va="center", fontsize=5.15, linespacing=1.18, color=INK)
    for x, (title, body) in zip(centers, bottom):
        ax.text(x, 0.265, title, ha="center", va="center", fontsize=6.25, fontweight="bold", color=INK)
        ax.text(x, 0.145, body, ha="center", va="center", fontsize=4.95, linespacing=1.18, color=INK)

    save_figure(fig, output_dir, "figure1_revision_design", with_png=with_png)


def figure_primary(
    system: pd.DataFrame, effects: pd.DataFrame, output_dir: Path, *, with_png: bool
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(FULL_WIDTH_IN, 3.85),
        gridspec_kw={"width_ratios": [1.04, 1.0]},
    )

    systems = ["candidate_prior", "prior_2d", "prior_2d_unimol"]
    labels = ["Candidate\nprior", "Prior + 2D", "Prior + 2D\n+ Uni-Mol"]
    colors = [GREY, BLUE, TEAL]
    x = np.arange(len(systems))
    width = 0.34
    for offset, metric, label, hatch in [(-width / 2, "top1", "Top-1", ""), (width / 2, "mrr", "MRR", "//")]:
        values = [
            float(one(system, analysis_id="D5", system="candidate_prior", metric=metric, scope="within_pool").estimate),
            float(one(system, analysis_id="D1", system="prior_2d", metric=metric, scope="within_pool").estimate),
            float(one(system, analysis_id="D1", system="prior_2d_unimol", metric=metric, scope="within_pool").estimate),
        ]
        bars = ax1.bar(x + offset, np.asarray(values) * 100, width, label=label, color=colors, edgecolor="white", linewidth=0.8, hatch=hatch)
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, value * 100 + 0.35, f"{value*100:.2f}", ha="center", va="bottom", fontsize=8.0, rotation=90)
    ax1.set_xticks(x, labels)
    ax1.set_ylim(72, 87.5)
    ax1.set_ylabel("Within-pool metric (%)")
    ax1.set_title("Validation-selected models (5-seed estimates)", pad=12)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
        handlelength=2.6,
    )
    ax1.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)
    ax1.spines[["top", "right"]].set_visible(False)
    panel_label(ax1, "A")

    rows: list[tuple[str, pd.Series, str, str]] = []
    for metric, display in [("top1", "Top-1"), ("mrr", "MRR")]:
        rows.append((display, one(effects, analysis_id="G1", scope="product_only", metric=metric), "Product-cluster CI", BLUE))
        rows.append((display, one(effects, analysis_id="G1", scope="seed_marginal", metric=metric), "Seed-marginal CI", TEAL))
    y = np.asarray([3.15, 2.75, 1.65, 1.25])
    for yi, (metric_label, row, ci_label, color) in zip(y, rows):
        estimate = float(row.effect) * 100
        low = float(row.ci95_low) * 100
        high = float(row.ci95_high) * 100
        ax2.errorbar(estimate, yi, xerr=[[estimate - low], [high - estimate]], fmt="o", ms=5.5, color=color, ecolor=color, capsize=3, lw=1.4)
        ax2.text(high + 0.08, yi, f"{estimate:+.2f} [{low:+.2f}, {high:+.2f}]", va="center", fontsize=8.0, color=INK)
    ax2.axvline(0, color="#98A2B3", lw=1, ls="--")
    ax2.set_yticks([2.95, 1.45], ["Top-1", "MRR"])
    ax2.set_ylim(0.75, 3.65)
    ax2.set_xlim(-0.2, 1.55)
    ax2.set_xlabel("Uni-Mol-augmented minus 2D (percentage points)")
    ax2.set_title("Paired primary effect and 95% intervals", pad=12)
    ax2.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.scatter([], [], color=BLUE, label="Product-cluster CI")
    ax2.scatter([], [], color=TEAL, label="Seed-marginal CI")
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
    )
    panel_label(ax2, "B")

    fig.subplots_adjust(left=0.10, right=0.985, top=0.86, bottom=0.25, wspace=0.23)
    save_figure(fig, output_dir, "figure2_primary_effects", with_png=with_png)


def forest(
    ax: plt.Axes,
    rows: list[tuple[str, pd.Series]],
    title: str,
    xlim: tuple[float, float],
    colors: list[str] | None = None,
) -> None:
    if colors is None:
        colors = [BLUE] * len(rows)
    y = np.arange(len(rows))[::-1]
    for yi, (label, row), color in zip(y, rows, colors):
        est = float(row.effect) * 100
        low = float(row.ci95_low) * 100
        high = float(row.ci95_high) * 100
        ax.errorbar(est, yi, xerr=[[est - low], [high - est]], fmt="o", color=color, ecolor=color, ms=4.7, capsize=2.5, lw=1.2)
    ax.axvline(0, color="#98A2B3", lw=0.9, ls="--")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(*xlim)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Top-1 difference (percentage points)")
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def figure_robustness(
    system: pd.DataFrame, effects: pd.DataFrame, output_dir: Path, *, with_png: bool
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH_IN, 6.45))
    ax = axes[0, 0]
    b_rows = [
        ("5×5 single-conformer", one(effects, analysis_id="B2", metric="top1")),
        ("10-conformer average vs 2D", one(effects, analysis_id="B3", comparison="b3_minus_2d", metric="top1")),
        ("Average vs single conformer", one(effects, analysis_id="B3", comparison="b3_minus_single_conformer", metric="top1")),
    ]
    forest(ax, b_rows, "Conformer robustness", (-0.55, 1.45), [TEAL, TEAL, GREY])
    panel_label(ax, "A")

    ax = axes[0, 1]
    c_rows = [
        ("Morgan – 2D", one(effects, analysis_id="C1", comparison="morgan_minus_2d", metric="top1")),
        ("GROVER – 2D", one(effects, analysis_id="C1", comparison="grover_minus_2d", metric="top1")),
        ("Uni-Mol – 2D", one(effects, analysis_id="C1", comparison="unimol_minus_2d", metric="top1")),
        ("Uni-Mol – GROVER", one(effects, analysis_id="C1", comparison="unimol_minus_grover", metric="top1")),
        ("Uni-Mol – Morgan", one(effects, analysis_id="C1", comparison="unimol_minus_morgan", metric="top1")),
    ]
    forest(ax, c_rows, "Encoder attribution", (-0.65, 1.45), [ORANGE, PURPLE, TEAL, GREY, GREY])
    panel_label(ax, "B")

    ax = axes[1, 0]
    pool_names = ["AiZynthFinder", "LocalRetro", "Merged union"]
    pool_prefixes = ["aizynthfinder_only", "localretro_only", "merged"]
    pool_colors = [BLUE, ORANGE, PURPLE]
    coverage = [float(one(system, analysis_id="E-POOLS", system=f"{p}_baseline", metric="top1", scope="within_pool").coverage) * 100 for p in pool_prefixes]
    x = np.arange(3)
    bars = ax.bar(x, coverage, color=pool_colors, width=0.58)
    for bar, value in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.55, f"{value:.1f}%", ha="center", va="bottom", fontsize=8.0)
    ax.set_ylim(75, 100)
    ax.set_ylabel("Test-product coverage (%)")
    ax.set_xticks(x, ["AiZynth", "LocalRetro", "Merged"])
    ax.set_title("Expanded-pool coverage", pad=12)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = ax.twinx()
    for xi, p, color in zip(x, pool_prefixes, pool_colors):
        row = one(effects, analysis_id="E-POOLS", comparison=f"{p}_augmented_minus_baseline", metric="top1", scope="within_pool")
        est, low, high = [float(row[c]) * 100 for c in ["effect", "ci95_low", "ci95_high"]]
        ax2.errorbar(xi, est, yerr=[[est - low], [high - est]], fmt="D", color=INK, ecolor=INK, ms=4.3, capsize=2.5, lw=1.1)
    ax2.axhline(0, color="#98A2B3", lw=0.8, ls="--")
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_ylabel("Within-pool Top-1 effect (pp)", color=INK)
    ax2.spines["top"].set_visible(False)
    panel_label(ax, "C")

    ax = axes[1, 1]
    outcome_rows = [
        ("Exact-match Top-1", one(effects, analysis_id="F1", metric="exact_match_top1")),
        ("Round-trip beam-1", one(effects, analysis_id="F1", metric="roundtrip_beam1")),
        ("Round-trip beam-5", one(effects, analysis_id="F1", metric="roundtrip_beam5")),
        ("Chirality sensitivity", one(effects, analysis_id="F2", comparison="augmented_true_minus_false", metric="top1")),
        ("Salt sensitivity", one(effects, analysis_id="F3", comparison="augmented_salt_minus_current", metric="top1")),
    ]
    forest(ax, outcome_rows, "Outcome and chemistry sensitivities", (-0.45, 1.4), [TEAL, TEAL, GREY, ORANGE, PURPLE])
    ax.set_xlabel("Effect (percentage points)")
    panel_label(ax, "D")

    fig.subplots_adjust(left=0.23, right=0.95, top=0.91, bottom=0.12, hspace=0.42, wspace=1.12)
    save_figure(fig, output_dir, "figure3_robustness_attribution", with_png=with_png)


def figure_classes(
    effects: pd.DataFrame, counts: pd.DataFrame, output_dir: Path, *, with_png: bool
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(FULL_WIDTH_IN, 4.35),
        gridspec_kw={"width_ratios": [1.32, 0.9]},
    )
    rows = []
    colors = []
    for cls in range(1, 11):
        row = one(effects, analysis_id="G2", comparison=f"class_{cls}_augmented_minus_2d", metric="top1")
        n = int(row.n_reactions)
        label = f"Class {cls}{'*' if cls == 3 else ''}  (n={n})"
        rows.append((label, row))
        colors.append(RED if cls == 3 else BLUE)
    forest(ax1, rows, "Reaction-class effects (BH family: 10 Top-1 tests)", (-7.5, 7.5), colors)
    ax1.set_xlabel("Augmented minus 2D Top-1 (percentage points)")
    class3 = one(effects, analysis_id="G2", comparison="class_3_augmented_minus_2d", metric="top1")
    panel_label(ax1, "A")

    values = [
        int(one(counts, analysis_id="G3", count_name="improved")["count"]),
        int(one(counts, analysis_id="G3", count_name="degraded")["count"]),
        int(one(counts, analysis_id="G3", count_name="unchanged")["count"]),
    ]
    labels = ["Improved", "Degraded", "Unchanged"]
    colors = [TEAL, RED, LIGHT_GREY]
    total = sum(values)
    wedges, _ = ax2.pie(
        values,
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.36, "edgecolor": "white"},
    )
    ax2.text(0, 0.09, f"{total:,}", ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
    ax2.text(0, -0.12, "seed × reaction\npairs", ha="center", va="center", fontsize=8.0, color=GREY)
    legend_labels = [f"{label}: {value:,} ({value/total*100:.1f}%)" for label, value in zip(labels, values)]
    ax2.legend(
        wedges,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        borderaxespad=0,
    )
    net = int(one(counts, analysis_id="G3", count_name="net")["count"])
    ax2.set_title(f"Top-1 promotion/degradation\nNet promotions: {net:+,}", pad=14)
    panel_label(ax2, "B")

    fig.text(
        0.16,
        0.01,
        f"* Class 3: {float(class3.effect)*100:+.2f} percentage points; BH q={float(class3.q_value):.4f}.",
        color=RED,
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.20, wspace=0.22)
    save_figure(fig, output_dir, "figure4_class_and_flips", with_png=with_png)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger-dir", type=Path, default=Path("outputs/jcheminform_revision/numerical_freeze_v2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/overleaf"),
        help="Write separate submission figure files beside main.tex, as required by the template.",
    )
    parser.add_argument(
        "--figure1-background",
        type=Path,
        default=Path("paper/overleaf/source_assets/figure1_workflow_background.png"),
        help="Text-free generated artwork; exact protocol labels are overlaid by this module.",
    )
    parser.add_argument(
        "--with-png",
        action="store_true",
        help=f"Also write {RASTER_DPI}-dpi PNG previews; vector PDF is always written.",
    )
    args = parser.parse_args()
    configure_style()

    effects = pd.read_csv(args.ledger_dir / "paired_effects.csv")
    system = pd.read_csv(args.ledger_dir / "system_performance.csv")
    counts = pd.read_csv(args.ledger_dir / "promotion_degradation_counts.csv")
    figure_design(args.output_dir, args.figure1_background, with_png=args.with_png)
    figure_primary(system, effects, args.output_dir, with_png=args.with_png)
    figure_robustness(system, effects, args.output_dir, with_png=args.with_png)
    figure_classes(effects, counts, args.output_dir, with_png=args.with_png)
    print(f"Wrote four revised figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
