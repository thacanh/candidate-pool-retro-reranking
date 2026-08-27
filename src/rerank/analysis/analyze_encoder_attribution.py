"""Paired encoder-attribution analysis for frozen cap-10 predictions.

This analysis implements the prespecified WS-C equivalence rule:

* endpoint: encoder-minus-frozen-2D gain;
* pairing: reaction, product cluster and training seed;
* uncertainty: canonical-product-clustered percentile bootstrap;
* equivalence: 90% CI wholly inside +/-0.0025 Top-1 and +/-0.0020 MRR.

No model is fitted and no candidate/test prediction is regenerated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROTOCOL_ID = "cap10-tuned-encoder-attribution-v1"
SEEDS = (42, 43, 44, 45, 46)
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 2026
EXPECTED_REACTIONS = 3_985
EQUIVALENCE_MARGINS = {"top1": 0.0025, "mrr": 0.0020}
PAIRING_COLUMNS = (
    "reaction_id",
    "source_split",
    "reaction_class",
    "candidate_count",
    "coverage_rank",
    "product_smiles",
    "ground_truth",
    "baseline_hit@1",
    "baseline_hit@3",
    "baseline_hit@5",
    "baseline_hit@10",
    "baseline_rr",
    "baseline_rank",
    "baseline_top1",
    "baseline_candidates_json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": file_sha256(resolved),
    }


@dataclass(frozen=True)
class PredictionSet:
    name: str
    values: dict[str, np.ndarray]
    pairing: pd.DataFrame
    clusters: np.ndarray
    files: list[dict]


def _require_columns(frame: pd.DataFrame, path: Path) -> None:
    required = set(PAIRING_COLUMNS) | {"reranked_hit@1", "reranked_rr"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns in {path}: {missing}")


def _assert_exact_pairing(reference: pd.DataFrame, observed: pd.DataFrame, label: str) -> None:
    if len(reference) != len(observed):
        raise ValueError(
            f"Pairing row-count mismatch for {label}: {len(observed)} != {len(reference)}"
        )
    for column in PAIRING_COLUMNS:
        left = reference[column].astype(str).to_numpy()
        right = observed[column].astype(str).to_numpy()
        mismatches = np.flatnonzero(left != right)
        if len(mismatches):
            row = int(mismatches[0])
            raise ValueError(
                f"Pairing mismatch for {label}, column {column}, row {row}: "
                f"{right[row]!r} != {left[row]!r}"
            )


def _manifest_metric(manifest: dict, manifest_layout: str, seed: int, metric: str) -> float:
    if manifest_layout == "primary_baseline":
        return float(manifest["per_seed_metrics"]["baseline"][str(seed)][metric])
    if manifest_layout == "primary_augmented":
        return float(manifest["per_seed_metrics"]["augmented"][str(seed)][metric])
    if manifest_layout == "encoder_control":
        return float(manifest["per_seed_metrics"][str(seed)][metric])
    raise ValueError(f"Unknown manifest layout: {manifest_layout}")


def load_prediction_set(
    name: str,
    result_root: Path,
    filename_prefix: str,
    manifest_layout: str,
    seeds: Iterable[int] = SEEDS,
    expected_reactions: int = EXPECTED_REACTIONS,
    reference_pairing: pd.DataFrame | None = None,
) -> PredictionSet:
    result_root = result_root.resolve()
    manifest_path = result_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing frozen test manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_layout == "encoder_control":
        if not manifest.get("test_partition_loaded_only_after_both_freezes", False):
            raise PermissionError(f"Encoder-control test gate is not frozen: {manifest_path}")
        if not manifest.get("primary_2d_predictions_are_the_frozen_comparator", False):
            raise PermissionError(f"Frozen primary comparator is not declared: {manifest_path}")
    else:
        if not manifest.get("test_partition_loaded_only_after_selection_freeze", False):
            raise PermissionError(f"Primary test gate is not frozen: {manifest_path}")

    metric_rows = {"top1": [], "mrr": []}
    pairing: pd.DataFrame | None = None
    files = [file_record(manifest_path)]
    for seed in seeds:
        path = result_root / "predictions" / f"{filename_prefix}_seed_{seed}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen prediction file: {path}")
        frame = pd.read_csv(path)
        _require_columns(frame, path)
        if len(frame) != expected_reactions:
            raise ValueError(
                f"Expected {expected_reactions} covered reactions in {path}; got {len(frame)}"
            )
        if pairing is None:
            pairing = frame.loc[:, PAIRING_COLUMNS].copy()
        else:
            _assert_exact_pairing(pairing, frame, f"{name}/seed_{seed}")
        if reference_pairing is not None:
            _assert_exact_pairing(reference_pairing, frame, f"{name}/seed_{seed}")

        top1 = frame["reranked_hit@1"].to_numpy(dtype=np.float64)
        mrr = frame["reranked_rr"].to_numpy(dtype=np.float64)
        for metric, values in (("top1", top1), ("mrr", mrr)):
            expected = _manifest_metric(manifest, manifest_layout, seed, metric)
            observed = float(values.mean())
            if not np.isclose(observed, expected, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"Prediction/manifest {metric} mismatch for {name}/seed_{seed}: "
                    f"{observed} != {expected}"
                )
            metric_rows[metric].append(values)
        files.append(file_record(path))

    assert pairing is not None
    if not pairing["reaction_id"].is_unique:
        raise ValueError(f"Reaction IDs are not unique for {name}")
    return PredictionSet(
        name=name,
        values={key: np.stack(rows, axis=0) for key, rows in metric_rows.items()},
        pairing=pairing,
        clusters=pairing["product_smiles"].astype(str).to_numpy(dtype=object),
        files=files,
    )


def clustered_intervals(
    differences: np.ndarray,
    clusters: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Average paired seeds, then resample canonical-product clusters."""
    differences = np.asarray(differences, dtype=np.float64)
    clusters = np.asarray(clusters, dtype=object)
    if differences.ndim != 2 or differences.shape[1] != len(clusters):
        raise ValueError("differences must be seeds x reactions and align with clusters")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    reaction_values = differences.mean(axis=0)
    unique = list(dict.fromkeys(clusters.tolist()))
    cluster_rows = [np.flatnonzero(clusters == key) for key in unique]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    if len(cluster_rows) == len(clusters) and all(len(rows) == 1 for rows in cluster_rows):
        # USPTO-50K's covered test set has one reaction per canonical product.
        # Batch the identical bootstrap operation to avoid 10,000 Python loops.
        batch_size = 256
        for start in range(0, n_bootstrap, batch_size):
            stop = min(start + batch_size, n_bootstrap)
            sampled = rng.integers(
                0,
                len(cluster_rows),
                size=(stop - start, len(cluster_rows)),
                dtype=np.int32,
            )
            draws[start:stop] = reaction_values[sampled].mean(axis=1)
    else:
        for index in range(n_bootstrap):
            sampled = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
            rows = np.concatenate([cluster_rows[item] for item in sampled])
            draws[index] = float(reaction_values[rows].mean())
    ci90_low, ci90_high = np.percentile(draws, [5.0, 95.0])
    ci95_low, ci95_high = np.percentile(draws, [2.5, 97.5])
    return {
        "effect": float(reaction_values.mean()),
        "ci90_low": float(ci90_low),
        "ci90_high": float(ci90_high),
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
        "n_product_clusters": len(unique),
        "n_reactions": len(clusters),
        "n_seeds": int(differences.shape[0]),
        "bootstrap_samples": int(n_bootstrap),
        "rng_seed": int(seed),
    }


def analyze(
    primary_root: Path,
    morgan_root: Path,
    grover_root: Path,
    output_dir: Path,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    expected_reactions: int = EXPECTED_REACTIONS,
) -> dict:
    primary_manifest = json.loads((primary_root / "manifest.json").read_text(encoding="utf-8"))
    selection_fingerprint = primary_manifest["selection_fingerprint"]
    for control_root in (morgan_root, grover_root):
        control = json.loads((control_root / "manifest.json").read_text(encoding="utf-8"))
        if control.get("primary_selection_fingerprint") != selection_fingerprint:
            raise ValueError(f"Primary selection fingerprint mismatch: {control_root}")

    baseline = load_prediction_set(
        "2d", primary_root, "baseline", "primary_baseline",
        expected_reactions=expected_reactions,
    )
    unimol = load_prediction_set(
        "unimol", primary_root, "augmented", "primary_augmented",
        expected_reactions=expected_reactions, reference_pairing=baseline.pairing,
    )
    morgan = load_prediction_set(
        "morgan", morgan_root, "augmented", "encoder_control",
        expected_reactions=expected_reactions, reference_pairing=baseline.pairing,
    )
    grover = load_prediction_set(
        "grover", grover_root, "augmented", "encoder_control",
        expected_reactions=expected_reactions, reference_pairing=baseline.pairing,
    )
    systems = {item.name: item for item in (baseline, morgan, grover, unimol)}
    comparisons = (
        ("morgan_minus_2d", "morgan", "2d"),
        ("grover_minus_2d", "grover", "2d"),
        ("unimol_minus_2d", "unimol", "2d"),
        ("grover_minus_morgan", "grover", "morgan"),
        ("unimol_minus_grover", "unimol", "grover"),
        ("unimol_minus_morgan", "unimol", "morgan"),
    )

    rows: list[dict] = []
    for comparison, left, right in comparisons:
        for metric in ("top1", "mrr"):
            intervals = clustered_intervals(
                systems[left].values[metric] - systems[right].values[metric],
                baseline.clusters,
                n_bootstrap=n_bootstrap,
                seed=bootstrap_seed,
            )
            margin = EQUIVALENCE_MARGINS[metric]
            rows.append(
                {
                    "comparison": comparison,
                    "left_system": left,
                    "right_system": right,
                    "metric": metric,
                    **intervals,
                    "equivalence_margin": margin,
                    "metric_equivalent": bool(
                        intervals["ci90_low"] > -margin
                        and intervals["ci90_high"] < margin
                    ),
                    "superior_95": bool(intervals["ci95_low"] > 0.0),
                    "inferior_95": bool(intervals["ci95_high"] < 0.0),
                }
            )

    frame = pd.DataFrame(rows)
    pair_decisions: dict[str, dict] = {}
    for comparison, _, _ in comparisons:
        subset = frame[frame["comparison"] == comparison]
        pair_decisions[comparison] = {
            "equivalent_both_endpoints": bool(subset["metric_equivalent"].all()),
            "superior_both_endpoints_95": bool(subset["superior_95"].all()),
            "inferior_both_endpoints_95": bool(subset["inferior_95"].all()),
        }

    all_three_equivalent = all(
        pair_decisions[name]["equivalent_both_endpoints"]
        for name in (
            "grover_minus_morgan",
            "unimol_minus_grover",
            "unimol_minus_morgan",
        )
    )
    if all_three_equivalent:
        interpretation = "representation_agnostic_atom_level_comparison"
    elif (
        pair_decisions["unimol_minus_grover"]["equivalent_both_endpoints"]
        and pair_decisions["unimol_minus_morgan"]["superior_both_endpoints_95"]
    ):
        interpretation = "learned_atom_embedding_claim_only"
    elif (
        pair_decisions["unimol_minus_grover"]["superior_both_endpoints_95"]
        and pair_decisions["unimol_minus_morgan"]["superior_both_endpoints_95"]
    ):
        interpretation = "unimol_specific_representation_effect_not_3d_causality"
    else:
        interpretation = "mixed_or_inconclusive_encoder_attribution"

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite encoder-attribution output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "encoder_attribution_summary.csv"
    frame.to_csv(summary_path, index=False)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "comparator": "frozen cap10-tuned-v1 four-input 2D baseline",
        "single_intended_change": "atom representation source only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_fingerprint": selection_fingerprint,
        "seeds": list(SEEDS),
        "pairing_gate": {
            "status": "exact",
            "columns": list(PAIRING_COLUMNS),
            "reactions": expected_reactions,
            "canonical_product_clusters": int(len(set(baseline.clusters.tolist()))),
        },
        "method": {
            "endpoint": "encoder-minus-2D gain; pairwise encoder contrasts cancel the common 2D comparator",
            "bootstrap": "paired canonical-product-clustered percentile bootstrap retaining all five seed rows",
            "bootstrap_samples": n_bootstrap,
            "rng_seed": bootstrap_seed,
            "equivalence_test": "TOST alpha 0.05 via 90% paired clustered interval",
            "equivalence_margins": EQUIVALENCE_MARGINS,
            "superiority_reporting": "95% paired clustered interval excludes zero",
        },
        "input_files": {
            name: prediction_set.files for name, prediction_set in systems.items()
        },
        "pair_decisions": pair_decisions,
        "all_three_equivalent": all_three_equivalent,
        "prespecified_interpretation": interpretation,
        "summary_csv": file_record(summary_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--morgan-root", type=Path, required=True)
    parser.add_argument("--grover-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--expected-reactions", type=int, default=EXPECTED_REACTIONS)
    args = parser.parse_args()
    result = analyze(
        args.primary_root,
        args.morgan_root,
        args.grover_root,
        args.output_dir,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        expected_reactions=args.expected_reactions,
    )
    print(json.dumps({
        "protocol_id": result["protocol_id"],
        "pairing_gate": result["pairing_gate"],
        "pair_decisions": result["pair_decisions"],
        "prespecified_interpretation": result["prespecified_interpretation"],
    }, indent=2))


if __name__ == "__main__":
    main()
