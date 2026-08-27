#!/usr/bin/env python
"""Prepare and analyze the frozen Chemformer forward round-trip control.

The module deliberately separates preparation from model execution.  ``prepare``
reads only already-frozen official-test predictions and writes a deduplicated
Chemformer input table.  ``evaluate`` accepts the untouched beam output from the
pinned forward checkpoint and produces the F1 tables and paired statistics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem

from rerank.analysis.analyze_controlled_study import (
    clustered_bootstrap,
    paired_cluster_permutation_test,
)
from rerank.revision_tuning import file_fingerprint
from rerank.experiments.run_tuned_revision import immutable_json_dump


F1_PROTOCOL_ID = "f1-chemformer-forward-roundtrip-v1"
G1_SEEDS = tuple(range(42, 62))
EXPECTED_COVERED_REACTIONS = 3_985
CHEMFORMER_BEAMS = 5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026
PERMUTATION_REPLICATES = 100_000
PERMUTATION_SEED = 2027

SYSTEM_PRIOR = "candidate_prior"
SYSTEM_2D = "prior_2d"
SYSTEM_AUGMENTED = "prior_2d_unimol"
SYSTEMS = (SYSTEM_PRIOR, SYSTEM_2D, SYSTEM_AUGMENTED)

REQUIRED_PREDICTION_COLUMNS = {
    "reaction_id",
    "source_split",
    "product_smiles",
    "ground_truth",
    "baseline_top1",
    "baseline_hit@1",
    "reranked_top1",
    "reranked_hit@1",
}


@dataclass(frozen=True)
class AssetPaths:
    checkpoint: Path
    vocabulary: Path
    repository: Path


def canonical_product(smiles: Any) -> str | None:
    """Return an isomeric canonical product identity, or ``None`` if invalid."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def canonical_fragment_set(smiles: Any) -> tuple[str, ...] | None:
    """Return the frozen fragment-order-invariant precursor identity."""
    fragments: list[str] = []
    for fragment in str(smiles).split("."):
        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None:
            return None
        fragments.append(
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        )
    return tuple(sorted(fragments))


def _require_empty_directory(path: Path, label: str) -> Path:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {label}: {path.resolve()}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _matches_frozen_content(path: Path, fingerprint: dict[str, Any]) -> bool:
    """Compare immutable file content while permitting bundle relocation."""
    actual = file_fingerprint(path)
    return (
        int(actual["size_bytes"]) == int(fingerprint["size_bytes"])
        and str(actual["sha256"]) == str(fingerprint["sha256"])
    )


def _expected_asset_record(ledger_path: Path) -> dict[str, Any]:
    ledger = _load_json(ledger_path)
    try:
        record = ledger["assets"]["chemformer_uspto50k_forward"]
    except KeyError as exc:
        raise ValueError("Chemformer forward asset is absent from the ledger.") from exc
    required = {
        "repository_commit",
        "checkpoint_size_bytes",
        "checkpoint_sha256",
        "vocabulary_size_bytes",
        "vocabulary_sha256",
        "task",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Chemformer asset ledger is incomplete: {sorted(missing)}")
    if record["task"] != "forward_prediction":
        raise ValueError("Pinned Chemformer checkpoint is not a forward model.")
    return record


def verify_assets(ledger_path: Path, paths: AssetPaths) -> dict[str, Any]:
    """Fail closed unless checkpoint, vocabulary and source revision are exact."""
    record = _expected_asset_record(ledger_path)
    checkpoint = file_fingerprint(paths.checkpoint)
    vocabulary = file_fingerprint(paths.vocabulary)
    if checkpoint["size_bytes"] != int(record["checkpoint_size_bytes"]):
        raise ValueError("Chemformer checkpoint size differs from the ledger.")
    if checkpoint["sha256"] != str(record["checkpoint_sha256"]):
        raise ValueError("Chemformer checkpoint SHA-256 differs from the ledger.")
    if vocabulary["size_bytes"] != int(record["vocabulary_size_bytes"]):
        raise ValueError("Chemformer vocabulary size differs from the ledger.")
    if vocabulary["sha256"] != str(record["vocabulary_sha256"]):
        raise ValueError("Chemformer vocabulary SHA-256 differs from the ledger.")

    try:
        commit = subprocess.run(
            ["git", "-C", str(paths.repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(paths.repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot verify the pinned Chemformer repository.") from exc
    if commit != str(record["repository_commit"]):
        raise ValueError("Chemformer source commit differs from the ledger.")
    if dirty:
        raise ValueError("Chemformer source checkout is not clean.")
    return {
        "ledger": file_fingerprint(ledger_path),
        "checkpoint": checkpoint,
        "vocabulary": vocabulary,
        "repository": {
            "path": str(paths.repository.resolve()),
            "commit": commit,
            "clean": True,
        },
        "task": "forward_prediction",
    }


def _load_prediction(path: Path, expected_reactions: int) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("reaction_id").reset_index(drop=True)
    missing = REQUIRED_PREDICTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if len(frame) != expected_reactions:
        raise ValueError(
            f"{path} has {len(frame)} rows; expected {expected_reactions}."
        )
    if frame["reaction_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate reaction IDs.")
    if set(frame["source_split"].astype(str)) != {"test"}:
        raise ValueError(f"{path} is not exclusively official-test data.")
    return frame


def _assert_same(left: pd.DataFrame, right: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if not left[column].equals(right[column]):
            raise ValueError(f"Frozen prediction files disagree on {column}.")


def _validate_exact_hit(frame: pd.DataFrame, prediction: str, hit: str) -> None:
    computed = np.asarray(
        [
            canonical_fragment_set(value) == canonical_fragment_set(reference)
            for value, reference in zip(
                frame[prediction], frame["ground_truth"], strict=True
            )
        ],
        dtype=np.int8,
    )
    recorded = frame[hit].to_numpy(dtype=np.int8)
    if not np.array_equal(computed, recorded):
        raise ValueError(f"Recorded {hit} disagrees with canonical exact matching.")


def _append_usage(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    system: str,
    seed: int,
    prediction_column: str,
    hit_column: str,
) -> None:
    for reaction_id, product, reference, prediction, hit in zip(
        frame["reaction_id"],
        frame["product_smiles"],
        frame["ground_truth"],
        frame[prediction_column],
        frame[hit_column],
        strict=True,
    ):
        rows.append(
            {
                "system": system,
                "seed": seed,
                "reaction_id": int(reaction_id),
                "product_smiles": str(product),
                "ground_truth": str(reference),
                "precursor_smiles": str(prediction),
                "exact_match_top1": int(hit),
            }
        )


def prepare_inputs(
    *,
    prediction_dir: Path,
    g1_manifest_path: Path,
    output_dir: Path,
    asset_ledger_path: Path,
    assets: AssetPaths,
    seeds: tuple[int, ...] = G1_SEEDS,
    expected_reactions: int = EXPECTED_COVERED_REACTIONS,
) -> dict[str, Any]:
    output_dir = _require_empty_directory(output_dir, "F1 preparation")
    asset_record = verify_assets(asset_ledger_path, assets)
    g1_manifest = _load_json(g1_manifest_path)
    if g1_manifest.get("manifest_kind") != "g1_20seed_post_freeze_test_evaluation":
        raise PermissionError("F1 requires the frozen post-G1 test manifest.")
    if not g1_manifest.get("test_partition_loaded_only_after_20seed_freeze"):
        raise PermissionError("G1 manifest does not prove the pre-test freeze gate.")

    usage_rows: list[dict[str, Any]] = []
    prediction_inputs: dict[str, dict[str, Any]] = {}
    anchor: pd.DataFrame | None = None
    for seed in seeds:
        baseline_path = prediction_dir / f"baseline_seed_{seed}.csv"
        augmented_path = prediction_dir / f"augmented_seed_{seed}.csv"
        baseline = _load_prediction(baseline_path, expected_reactions)
        augmented = _load_prediction(augmented_path, expected_reactions)
        _assert_same(
            baseline,
            augmented,
            ("reaction_id", "product_smiles", "ground_truth", "baseline_top1"),
        )
        if anchor is None:
            anchor = baseline
            _validate_exact_hit(anchor, "baseline_top1", "baseline_hit@1")
            _append_usage(
                usage_rows,
                anchor,
                system=SYSTEM_PRIOR,
                seed=-1,
                prediction_column="baseline_top1",
                hit_column="baseline_hit@1",
            )
        else:
            _assert_same(
                anchor,
                baseline,
                (
                    "reaction_id",
                    "product_smiles",
                    "ground_truth",
                    "baseline_top1",
                    "baseline_hit@1",
                ),
            )
        _validate_exact_hit(baseline, "reranked_top1", "reranked_hit@1")
        _validate_exact_hit(augmented, "reranked_top1", "reranked_hit@1")
        _append_usage(
            usage_rows,
            baseline,
            system=SYSTEM_2D,
            seed=seed,
            prediction_column="reranked_top1",
            hit_column="reranked_hit@1",
        )
        _append_usage(
            usage_rows,
            augmented,
            system=SYSTEM_AUGMENTED,
            seed=seed,
            prediction_column="reranked_top1",
            hit_column="reranked_hit@1",
        )
        prediction_inputs[str(seed)] = {
            "baseline": file_fingerprint(baseline_path),
            "augmented": file_fingerprint(augmented_path),
        }

    usage = pd.DataFrame(usage_rows)
    usage.insert(0, "usage_id", np.arange(len(usage), dtype=np.int64))
    inference = usage[["precursor_smiles", "product_smiles"]].drop_duplicates(
        keep="first"
    )
    inference = inference.reset_index(drop=True)
    inference.insert(0, "inference_id", np.arange(len(inference), dtype=np.int64))
    usage = usage.merge(
        inference,
        on=["precursor_smiles", "product_smiles"],
        how="left",
        validate="many_to_one",
    )
    if usage["inference_id"].isna().any():
        raise RuntimeError("Failed to assign an inference ID to every F1 usage row.")
    usage["inference_id"] = usage["inference_id"].astype(np.int64)
    usage = usage.sort_values("usage_id").reset_index(drop=True)

    chemformer_input = inference.rename(
        columns={"precursor_smiles": "reactants", "product_smiles": "products"}
    )
    chemformer_input["set"] = "test"
    chemformer_input = chemformer_input[["reactants", "products", "set"]]

    usage_path = output_dir / "frozen_top1_usages.csv"
    inference_map_path = output_dir / "inference_map.csv"
    model_input_path = output_dir / "chemformer_input.tsv"
    usage.to_csv(usage_path, index=False)
    inference.to_csv(inference_map_path, index=False)
    chemformer_input.to_csv(model_input_path, sep="\t", index=False)

    manifest = {
        "schema_version": 1,
        "protocol_id": F1_PROTOCOL_ID,
        "comparator": "frozen exact-match Top-1 evaluation",
        "single_intended_change": "score frozen Top-1 precursor sets with one pinned forward model",
        "source_g1_manifest": file_fingerprint(g1_manifest_path),
        "source_predictions": prediction_inputs,
        "assets": asset_record,
        "systems": list(SYSTEMS),
        "seeds": list(seeds),
        "covered_reactions": expected_reactions,
        "usage_rows": int(len(usage)),
        "unique_forward_inputs": int(len(inference)),
        "beam_settings": {"primary": 1, "sensitivity": CHEMFORMER_BEAMS},
        "canonical_output_identity": "RDKit canonical isomeric product SMILES",
        "invalid_outputs": "counted as failures",
        "test_partition_loaded": True,
        "test_partition_used_for_training_or_selection": False,
        "outputs": {
            "usage": file_fingerprint(usage_path),
            "inference_map": file_fingerprint(inference_map_path),
            "chemformer_input": file_fingerprint(model_input_path),
        },
    }
    immutable_json_dump(manifest, output_dir / "prepare_manifest.json", "F1 preparation manifest")
    return manifest


def _read_chemformer_output(path: Path, expected_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    target_columns = [
        column for column in ("target_smiles", "ground_truth") if column in frame
    ]
    if len(target_columns) != 1:
        raise ValueError(
            "Chemformer output must contain exactly one target column: "
            "target_smiles (legacy Chemformer) or ground_truth (AiZynthModels)."
        )
    required = {
        f"sampled_smiles_{beam}" for beam in range(1, CHEMFORMER_BEAMS + 1)
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Chemformer output is missing columns: {sorted(missing)}")
    if len(frame) != expected_rows:
        raise ValueError(
            f"Chemformer output has {len(frame)} rows; expected {expected_rows}."
        )
    frame = frame.rename(columns={target_columns[0]: "target_smiles"})
    return frame


def _metric_rows(scored: pd.DataFrame, seeds: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        system_frame = scored[scored["system"] == system]
        system_seeds = (-1,) if system == SYSTEM_PRIOR else seeds
        for seed in system_seeds:
            frame = system_frame[system_frame["seed"] == seed]
            rows.append(
                {
                    "system": system,
                    "seed": "fixed" if seed == -1 else str(seed),
                    "n_reactions": int(len(frame)),
                    "exact_match_top1": float(frame["exact_match_top1"].mean()),
                    "roundtrip_beam1": float(frame["roundtrip_beam1"].mean()),
                    "roundtrip_beam5": float(frame["roundtrip_beam5"].mean()),
                    "invalid_beam1": int(frame["beam1_canonical"].isna().sum()),
                    "invalid_any_beam": int(frame["invalid_beam_count"].gt(0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _paired_roundtrip_statistics(
    scored: pd.DataFrame,
    seeds: tuple[int, ...],
    *,
    bootstrap_replicates: int,
    permutation_replicates: int,
) -> dict[str, Any]:
    differences: dict[str, list[np.ndarray]] = {
        "exact_match_top1": [],
        "roundtrip_beam1": [],
        "roundtrip_beam5": [],
    }
    anchor_products: np.ndarray | None = None
    for seed in seeds:
        baseline = scored[
            (scored["system"] == SYSTEM_2D) & (scored["seed"] == seed)
        ].sort_values("reaction_id")
        augmented = scored[
            (scored["system"] == SYSTEM_AUGMENTED) & (scored["seed"] == seed)
        ].sort_values("reaction_id")
        if len(baseline) != len(augmented) or not baseline["reaction_id"].reset_index(
            drop=True
        ).equals(augmented["reaction_id"].reset_index(drop=True)):
            raise ValueError("F1 paired systems are not reaction aligned.")
        if anchor_products is None:
            anchor_products = baseline["product_smiles"].map(canonical_product).to_numpy(
                dtype=object
            )
        for metric in differences:
            differences[metric].append(
                augmented[metric].to_numpy(dtype=float)
                - baseline[metric].to_numpy(dtype=float)
            )
    if anchor_products is None or any(value is None for value in anchor_products):
        raise ValueError("F1 contains an invalid frozen target product.")

    output: dict[str, Any] = {}
    for offset, (metric, rows) in enumerate(differences.items()):
        matrix = np.stack(rows)
        bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED + offset)
        point, lower, upper = clustered_bootstrap(
            matrix, anchor_products, bootstrap_replicates, bootstrap_rng
        )
        permutation_rng = np.random.default_rng(PERMUTATION_SEED + offset)
        output[metric] = {
            "augmented_minus_2d": point,
            "ci95": [lower, upper],
            "paired_cluster_sign_flip_p": paired_cluster_permutation_test(
                matrix,
                anchor_products,
                permutation_replicates,
                permutation_rng,
            ),
            "positive_seed_count": int(np.sum(matrix.mean(axis=1) > 0)),
            "zero_seed_count": int(np.sum(matrix.mean(axis=1) == 0)),
            "negative_seed_count": int(np.sum(matrix.mean(axis=1) < 0)),
        }
    output["settings"] = {
        "bootstrap": {
            "method": "canonical-product-clustered paired bootstrap",
            "replicates": bootstrap_replicates,
            "base_rng_seed": BOOTSTRAP_SEED,
        },
        "permutation": {
            "method": "two-sided sign flip of seed-averaged reaction differences by canonical product cluster",
            "replicates": permutation_replicates,
            "base_rng_seed": PERMUTATION_SEED,
        },
    }
    return output


def evaluate_output(
    *,
    prepare_dir: Path,
    chemformer_output_path: Path,
    result_dir: Path,
    seeds: tuple[int, ...] = G1_SEEDS,
    expected_reactions: int = EXPECTED_COVERED_REACTIONS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    permutation_replicates: int = PERMUTATION_REPLICATES,
) -> dict[str, Any]:
    result_dir = _require_empty_directory(result_dir, "F1 evaluation")
    manifest_path = prepare_dir / "prepare_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("protocol_id") != F1_PROTOCOL_ID:
        raise PermissionError("F1 preparation manifest has the wrong protocol.")
    for name, fingerprint in manifest["outputs"].items():
        path = {
            "usage": prepare_dir / "frozen_top1_usages.csv",
            "inference_map": prepare_dir / "inference_map.csv",
            "chemformer_input": prepare_dir / "chemformer_input.tsv",
        }[name]
        if not _matches_frozen_content(path, fingerprint):
            raise ValueError(f"Prepared F1 artifact was modified: {name}")

    usage = pd.read_csv(prepare_dir / "frozen_top1_usages.csv")
    inference = pd.read_csv(prepare_dir / "inference_map.csv")
    output = _read_chemformer_output(chemformer_output_path, len(inference))
    target_expected = inference["product_smiles"].map(canonical_product).tolist()
    target_observed = output["target_smiles"].map(canonical_product).tolist()
    if target_expected != target_observed:
        raise ValueError("Chemformer output order/targets do not match F1 input.")

    scored_inference = inference.copy()
    canonical_columns: list[str] = []
    for beam in range(1, CHEMFORMER_BEAMS + 1):
        column = f"beam{beam}_canonical"
        canonical_columns.append(column)
        scored_inference[column] = output[f"sampled_smiles_{beam}"].map(
            canonical_product
        )
    targets = scored_inference["product_smiles"].map(canonical_product)
    if targets.isna().any():
        raise ValueError("Prepared F1 target includes invalid product SMILES.")
    scored_inference["roundtrip_beam1"] = (
        scored_inference["beam1_canonical"] == targets
    ).astype(np.int8)
    scored_inference["roundtrip_beam5"] = np.asarray(
        [
            any(row[column] == target for column in canonical_columns)
            for (_, row), target in zip(
                scored_inference.iterrows(), targets, strict=True
            )
        ],
        dtype=np.int8,
    )
    scored_inference["invalid_beam_count"] = scored_inference[
        canonical_columns
    ].isna().sum(axis=1)
    scored = usage.merge(
        scored_inference[
            [
                "inference_id",
                *canonical_columns,
                "roundtrip_beam1",
                "roundtrip_beam5",
                "invalid_beam_count",
            ]
        ],
        on="inference_id",
        how="left",
        validate="many_to_one",
    )
    expected_usage = expected_reactions * (1 + 2 * len(seeds))
    if len(scored) != expected_usage or scored["roundtrip_beam1"].isna().any():
        raise ValueError("F1 scored usage table is incomplete.")

    metric_table = _metric_rows(scored, seeds)
    aggregate_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        rows = metric_table[metric_table["system"] == system]
        aggregate_rows.append(
            {
                "system": system,
                "n_seeds": 0 if system == SYSTEM_PRIOR else len(seeds),
                "n_reactions": expected_reactions,
                "exact_match_top1_mean": float(rows["exact_match_top1"].mean()),
                "exact_match_top1_sd": float(rows["exact_match_top1"].std(ddof=1))
                if len(rows) > 1
                else 0.0,
                "roundtrip_beam1_mean": float(rows["roundtrip_beam1"].mean()),
                "roundtrip_beam1_sd": float(rows["roundtrip_beam1"].std(ddof=1))
                if len(rows) > 1
                else 0.0,
                "roundtrip_beam5_mean": float(rows["roundtrip_beam5"].mean()),
                "roundtrip_beam5_sd": float(rows["roundtrip_beam5"].std(ddof=1))
                if len(rows) > 1
                else 0.0,
            }
        )
    aggregate_table = pd.DataFrame(aggregate_rows)
    paired = _paired_roundtrip_statistics(
        scored,
        seeds,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
    )

    scored_path = result_dir / "reaction_level_roundtrip.csv"
    per_seed_path = result_dir / "per_seed_metrics.csv"
    aggregate_path = result_dir / "system_metrics.csv"
    paired_path = result_dir / "paired_statistics.json"
    scored.to_csv(scored_path, index=False)
    metric_table.to_csv(per_seed_path, index=False)
    aggregate_table.to_csv(aggregate_path, index=False)
    immutable_json_dump(paired, paired_path, "F1 paired statistics")
    result_manifest = {
        "schema_version": 1,
        "protocol_id": F1_PROTOCOL_ID,
        "comparator": "exact-match evaluation of the same frozen Top-1 precursor sets",
        "single_intended_change": "add pinned Chemformer forward round-trip scoring",
        "prepare_manifest": file_fingerprint(manifest_path),
        "chemformer_output": file_fingerprint(chemformer_output_path),
        "systems": list(SYSTEMS),
        "seeds": list(seeds),
        "covered_reactions": expected_reactions,
        "test_partition_loaded": True,
        "test_partition_used_for_training_or_selection": False,
        "beam1_primary": True,
        "beam5_sensitivity": True,
        "invalid_outputs_counted_as_failures": True,
        "outputs": {
            "reaction_level": file_fingerprint(scored_path),
            "per_seed_metrics": file_fingerprint(per_seed_path),
            "system_metrics": file_fingerprint(aggregate_path),
            "paired_statistics": file_fingerprint(paired_path),
        },
    }
    immutable_json_dump(
        result_manifest, result_dir / "manifest.json", "F1 result manifest"
    )
    return result_manifest


def _default_assets(root: Path) -> AssetPaths:
    return AssetPaths(
        checkpoint=root / "assets" / "chemformer_forward" / "last.ckpt",
        vocabulary=root
        / "assets"
        / "chemformer_official"
        / "bart_vocab_downstream.json",
        repository=root / "assets" / "chemformer_official",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-assets")
    prepare = subparsers.add_parser("prepare")
    evaluate = subparsers.add_parser("evaluate")
    for item in (verify, prepare):
        item.add_argument("--asset-ledger", default="data/revision_external_assets.json")
        item.add_argument("--checkpoint", default="assets/chemformer_forward/last.ckpt")
        item.add_argument(
            "--vocabulary",
            default="assets/chemformer_official/bart_vocab_downstream.json",
        )
        item.add_argument("--repository", default="assets/chemformer_official")
    prepare.add_argument(
        "--prediction-dir",
        default=(
            "outputs/jcheminform_revision/tuned_primary/conformer_seed_42/"
            "g1_20seed/test_results/predictions"
        ),
    )
    prepare.add_argument(
        "--g1-manifest",
        default=(
            "outputs/jcheminform_revision/tuned_primary/conformer_seed_42/"
            "g1_20seed/test_results/manifest.json"
        ),
    )
    prepare.add_argument(
        "--output-dir",
        default="outputs/jcheminform_revision/f1_roundtrip/prepared",
    )
    evaluate.add_argument(
        "--prepare-dir",
        default="outputs/jcheminform_revision/f1_roundtrip/prepared",
    )
    evaluate.add_argument("--chemformer-output", required=True)
    evaluate.add_argument(
        "--result-dir",
        default="outputs/jcheminform_revision/f1_roundtrip/results",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"verify-assets", "prepare"}:
        assets = AssetPaths(
            checkpoint=Path(args.checkpoint).resolve(),
            vocabulary=Path(args.vocabulary).resolve(),
            repository=Path(args.repository).resolve(),
        )
        if args.command == "verify-assets":
            result = verify_assets(Path(args.asset_ledger).resolve(), assets)
        else:
            result = prepare_inputs(
                prediction_dir=Path(args.prediction_dir).resolve(),
                g1_manifest_path=Path(args.g1_manifest).resolve(),
                output_dir=Path(args.output_dir).resolve(),
                asset_ledger_path=Path(args.asset_ledger).resolve(),
                assets=assets,
            )
    else:
        result = evaluate_output(
            prepare_dir=Path(args.prepare_dir).resolve(),
            chemformer_output_path=Path(args.chemformer_output).resolve(),
            result_dir=Path(args.result_dir).resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
