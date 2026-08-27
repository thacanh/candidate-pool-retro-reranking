import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import rerank.analysis.analyze_conformer_aggregate as aggregate


def test_aggregate_uses_current_seed_runner_cache_schema():
    assert aggregate.STUDY_CACHE_SCHEMA == 1
    assert aggregate.FEATURE_CACHE_NAME == "official_3d_prior_schema1.pkl"
    assert (
        aggregate.VALIDATION_FEATURE_CACHE_NAME
        == "official_valid_3d_prior_schema1.pkl"
    )


def _feature_array(seed: int) -> np.ndarray:
    value = float(seed - 42)
    return np.asarray(
        [
            [0.8, 0.7, 0.10 + value, 1.00 + value, -0.20 + value, 1.0, 0.5],
            [0.2, 0.3, 0.20 + value, 2.00 + value, -0.10 + value, 1.0, 1.5],
        ],
        dtype=np.float32,
    )


def _feature_blobs(seed: int) -> tuple[dict, dict]:
    candidates = [
        {"smiles": "CC", "canonical_smiles": "CC", "prior": 0.8},
        {"smiles": "CCCCCCCC", "canonical_smiles": "CCCCCCCC", "prior": 0.2},
    ]
    array = _feature_array(seed)
    main = {
        "schema_version": aggregate.STUDY_CACHE_SCHEMA,
        "feature_mode": "3d+prior",
        "payload": {
            "train_products": [
                {
                    "product_key": "CCO",
                    "product_smiles": "CCO",
                    "candidates": candidates,
                    "positive_indices": [0],
                    "negative_indices": [1],
                    "features": array.copy(),
                }
            ],
            "eval_pwc": [("CCO", candidates)],
            "eval_ground_truths": ["CC"],
            "eval_metadata": [
                {"reaction_id": 10, "source_split": "test", "reaction_class": 1}
            ],
            "eval_features": [array.copy()],
            "audit": {},
        },
    }
    valid = {
        "schema_version": aggregate.STUDY_CACHE_SCHEMA,
        "artifact_kind": "official_validation_conformer_features",
        "protocol_id": "cap10-conformer-features-v1",
        "conformer_seed": seed,
        "feature_mode": "3d+prior",
        "payload": {
            "eval_pwc": [("CCN", candidates)],
            "eval_ground_truths": ["CC"],
            "eval_metadata": [
                {"reaction_id": 11, "source_split": "valid", "reaction_class": 1}
            ],
            "eval_features": [array.copy()],
        },
        "audit": {},
    }
    return main, valid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_seed(root: Path, conformer_seed: int) -> None:
    seed_root = root / f"seed_{conformer_seed}"
    features = seed_root / "features"
    ranking = seed_root / "ranking_legacy_fixed50"
    features.mkdir(parents=True)
    ranking.mkdir()
    main, valid = _feature_blobs(conformer_seed)
    with open(features / aggregate.FEATURE_CACHE_NAME, "wb") as handle:
        pickle.dump(main, handle)
    with open(features / aggregate.VALIDATION_FEATURE_CACHE_NAME, "wb") as handle:
        pickle.dump(valid, handle)

    metrics = {}
    products = ["C", "CC", "CCC", "CCCC"]
    for training_seed in aggregate.TRAINING_SEEDS:
        conformer_index = conformer_seed - 42
        training_index = training_seed - 42
        n_hits = 1 + ((conformer_index + training_index) % 4)
        baseline_rr = np.full(4, 0.25, dtype=float)
        reranked_rr = baseline_rr + 0.02 * (1 + conformer_index + training_index)
        reranked_hit = np.asarray([1.0] * n_hits + [0.0] * (4 - n_hits))
        metrics[str(training_seed)] = {
            "top1": float(reranked_hit.mean()),
            "top3": 1.0,
            "top5": 1.0,
            "top10": 1.0,
            "mrr": float(reranked_rr.mean()),
            "baseline_top1": 0.0,
            "baseline_top3": 1.0,
            "baseline_top5": 1.0,
            "baseline_top10": 1.0,
            "baseline_mrr": 0.25,
        }
        with open(
            ranking / f"eval_seed{training_seed}.csv",
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            fields = [
                "reaction_id",
                "source_split",
                "product_smiles",
                "ground_truth",
                "baseline_hit@1",
                "reranked_hit@1",
                "baseline_rr",
                "reranked_rr",
                "baseline_rank",
                "baseline_candidates_json",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, product in enumerate(products):
                writer.writerow(
                    {
                        "reaction_id": 100 + index,
                        "source_split": "test",
                        "product_smiles": product,
                        "ground_truth": "CC",
                        "baseline_hit@1": 0,
                        "reranked_hit@1": reranked_hit[index],
                        "baseline_rr": baseline_rr[index],
                        "reranked_rr": reranked_rr[index],
                        "baseline_rank": 4,
                        "baseline_candidates_json": '["C","CC"]',
                    }
                )
    (ranking / "per_seed_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (ranking / "manifest.json").write_text(
        json.dumps(
            {
                "feature_mode": "3d+prior",
                "seeds": list(aggregate.TRAINING_SEEDS),
            }
        ),
        encoding="utf-8",
    )
    (seed_root / "embedding_summary.json").write_text(
        json.dumps(
            {
                "conformer_seed": conformer_seed,
                "required_items": 2,
                "required_keys_sha256": "c" * 64,
                "stored_items": 2,
                "null_embedding_items": 0,
                "status_counts": {"ok": 2},
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    (seed_root / "embedding_non_ok.csv").write_text(
        "smiles,status,error\n", encoding="utf-8"
    )
    result_summary = {
        "conformer_seed": conformer_seed,
        "conformer_label": f"C{conformer_seed - 41}",
    }
    (seed_root / "result_summary.json").write_text(
        json.dumps(result_summary), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "protocol_id": "legacy-cap10-fixed50-v1",
        "conformer_seed": conformer_seed,
        "training_seeds": list(aggregate.TRAINING_SEEDS),
        "input_fingerprints": {
            "source_csv": {"size_bytes": 1, "sha256": "a" * 64},
            "candidate_jsonl": {"size_bytes": 2, "sha256": "b" * 64},
        },
    }
    (seed_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    retained = [
        path
        for path in seed_root.rglob("*")
        if path.is_file() and path.name not in {"COMPLETED.json", "checksums.sha256"}
    ]
    (seed_root / "checksums.sha256").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(seed_root).as_posix()}\n"
            for path in sorted(retained)
        ),
        encoding="utf-8",
    )
    completed = {
        "seed": conformer_seed,
        "status": "complete",
        "manifest_sha256": _sha256(seed_root / "manifest.json"),
        "result_summary_sha256": _sha256(seed_root / "result_summary.json"),
    }
    (seed_root / "COMPLETED.json").write_text(
        json.dumps(completed), encoding="utf-8"
    )


def _write_shared_2d(root: Path, conformer_root: Path) -> None:
    ranking = root / "ranking"
    ranking.mkdir(parents=True)
    metrics = {}
    for training_seed in aggregate.TRAINING_SEEDS:
        source = conformer_root / "seed_42" / "ranking_legacy_fixed50" / f"eval_seed{training_seed}.csv"
        with open(source, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0])
        for index, row in enumerate(rows):
            row["reranked_hit@1"] = "1" if index == 0 else "0"
            row["reranked_rr"] = "0.3"
        with open(ranking / source.name, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        metrics[str(training_seed)] = {"top1": 0.25, "mrr": 0.3}
    (ranking / "per_seed_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (ranking / "manifest.json").write_text(
        json.dumps({"feature_mode": "2d+prior", "seeds": list(aggregate.TRAINING_SEEDS)}),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"protocol_id": "legacy-cap10-fixed50-v1"}), encoding="utf-8"
    )
    retained = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "checksums.sha256").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in retained
        ),
        encoding="utf-8",
    )
    (root / "COMPLETED.json").write_text(
        json.dumps({"status": "complete", "manifest_sha256": _sha256(root / "manifest.json")}),
        encoding="utf-8",
    )


def test_b1_reports_pairwise_correlations_cv_and_strata():
    features = {}
    for seed in aggregate.CONFIRMATORY_CONFORMER_SEEDS:
        shift = seed - 42
        features[seed] = np.asarray(
            [
                [1.0 + shift, 2.0 + shift, 3.0 + shift],
                [2.0 + shift, 4.0 + shift, 6.0 + shift],
                [0.0, 0.0, 0.0],
            ]
        )
    result = aggregate.analyze_b1(
        features, np.asarray(["rigid_zero", "intermediate_1_4", "flexible_ge5"])
    )
    all_rows = [
        row
        for row in result["stability"]
        if row["stratum"] == "all" and row["scalar"] == aggregate.SCALAR_NAMES[0]
    ]
    assert len(all_rows) == 10
    assert all_rows[0]["abs_diff_mean"] == pytest.approx(2.0 / 3.0)
    assert all_rows[0]["pearson"] > 0.95
    cv_all = next(
        row
        for row in result["cv"]
        if row["stratum"] == "all" and row["scalar"] == aggregate.SCALAR_NAMES[0]
    )
    assert cv_all["n_cv_undefined_abs_mean_le_1e-8"] == 1
    assert {row["stratum"] for row in result["strata"]} == {
        "all",
        "rigid_0_4",
        "rigid_zero",
        "intermediate_1_4",
        "flexible_ge5",
    }
    counts = {row["stratum"]: row["n_unique_pairs"] for row in result["strata"]}
    assert counts["rigid_0_4"] == (
        counts["rigid_zero"] + counts["intermediate_1_4"]
    )


def test_b3_averages_only_pair_level_scalar_columns():
    first_main, first_valid = _feature_blobs(42)
    remaining = []
    for seed in range(43, 52):
        main, valid = _feature_blobs(seed)
        remaining.append((seed, main, valid))
    averaged_main, averaged_valid = aggregate.average_feature_blobs(
        first_main, first_valid, remaining
    )
    observed = averaged_main["payload"]["train_products"][0]["features"]
    expected = _feature_array(42)
    expected[:, aggregate.SCALAR_COLUMNS] += 4.5
    assert np.array_equal(
        observed[:, aggregate.NON_SCALAR_COLUMNS],
        _feature_array(42)[:, aggregate.NON_SCALAR_COLUMNS],
    )
    assert np.allclose(observed[:, aggregate.SCALAR_COLUMNS], expected[:, aggregate.SCALAR_COLUMNS])
    assert averaged_main["representation_aggregation"]["atom_embeddings_averaged"] is False
    assert averaged_valid["conformer_seeds"] == list(range(42, 52))


def test_crossed_bootstrap_is_deterministic_and_reml_reports_components():
    differences = np.arange(25 * 6, dtype=float).reshape(25, 6) / 1000.0
    clusters = np.asarray(["A", "A", "B", "C", "D", "D"], dtype=object)
    first, first_draws = aggregate.crossed_bootstrap(
        differences, clusters, n_samples=100, seed=2026, batch_size=20
    )
    second, second_draws = aggregate.crossed_bootstrap(
        differences, clusters, n_samples=100, seed=2026, batch_size=20
    )
    assert first == second
    assert np.array_equal(first_draws, second_draws)
    assert first["point_estimate"] == pytest.approx(float(differences.mean()))
    fitted = aggregate.fit_crossed_reml(np.arange(25, dtype=float).reshape(5, 5) / 100.0)
    assert set(fitted["variance_components"]) == {
        "training_seed",
        "conformer",
        "residual_seed_by_conformer",
    }
    assert all(value >= 0 for value in fitted["variance_components"].values())


def test_full_synthetic_aggregate_and_checksum_failure(tmp_path):
    conformer_root = tmp_path / "conformers"
    for seed in aggregate.ALL_CONFORMER_SEEDS:
        _write_seed(conformer_root, seed)
    shared_2d = tmp_path / "shared_2d"
    _write_shared_2d(shared_2d, conformer_root)
    output = tmp_path / "aggregate"
    result = aggregate.run_aggregate(
        conformer_root,
        output,
        shared_2d_root=shared_2d,
        n_bootstrap=100,
        bootstrap_seed=2026,
    )
    assert result["pairing_gate"]["ranking_reaction_and_2d_alignment"] == "passed"
    with open(output / "b2_5x5_gain_grid.csv", encoding="utf-8", newline="") as handle:
        first_cell = next(csv.DictReader(handle))
    assert float(first_cell["baseline_top1"]) == pytest.approx(0.25)
    assert float(first_cell["top1_gain"]) == pytest.approx(
        float(first_cell["augmented_top1"]) - 0.25
    )
    assert "top1_robustness_gate" in result["analyses"]["B2"]["results"]
    with open(output / "b3_avg10_train_test_features.pkl", "rb") as handle:
        averaged = pickle.load(handle)
    observed = averaged["payload"]["train_products"][0]["features"]
    assert observed[0, 2] == pytest.approx(0.10 + 4.5)
    assert (output / "b2_crossed_bootstrap_draws.csv").is_file()
    assert len((output / "checksums.sha256").read_text(encoding="utf-8").splitlines()) == 12

    tampered = conformer_root / "seed_51" / "result_summary.json"
    tampered.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        aggregate.verify_seed_run(conformer_root / "seed_51", 51)
