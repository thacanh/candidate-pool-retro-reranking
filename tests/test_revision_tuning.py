import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import rerank.experiments.run_tuned_revision as tuned_runner

from rerank.revision_tuning import (
    AUGMENTED_COLUMNS,
    BASELINE_COLUMNS,
    CAPACITY_CONTROL_ID,
    CapacitySettings,
    GRID_TIE_EPSILON,
    SEEDS,
    assert_d3_capacity_match,
    capacity_arm_config,
    capacity_settings_fingerprint,
    descending_midrank_score,
    enumerate_d1_grid,
    load_selection_bundle,
    prepare_selection_bundle,
    select_prior_transform,
    select_shared_config,
    shard_config_indices,
    train_validation_trial,
    trainable_parameter_count,
    transform_feature_matrix,
)
from rerank.experiments.run_tuned_revision import (
    _score_test_arm,
    build_parser,
    canonical_fingerprint,
    immutable_json_dump,
    load_capacity_test_cache_after_freeze,
    load_test_cache_after_freeze,
    require_clean_evaluation_result_dir,
    require_no_partial_trial_artifacts,
)


def _trial(config_index, seed, score):
    return {
        "status": "completed",
        "config_index": config_index,
        "seed": seed,
        "best_validation_mrr": score,
        "best_epoch": 3,
    }


def _complete_grid(score_fn):
    return [
        _trial(config.index, seed, score_fn(config.index, seed))
        for config in enumerate_d1_grid()
        for seed in SEEDS
    ]


def test_d1_grid_exact_order_and_shards_cover_once():
    grid = enumerate_d1_grid()
    assert len(grid) == 81
    assert grid[0].hidden_width == 32
    assert grid[0].dropout == 0.0
    assert grid[0].learning_rate == 1e-4
    assert grid[0].margin == 0.0
    assert grid[1].margin == 0.1
    assert grid[3].learning_rate == 3e-4
    assert grid[-1].hidden_width == 128
    assert grid[-1].dropout == 0.3
    assert grid[-1].learning_rate == 1e-3
    assert grid[-1].margin == 0.3

    shards = [shard_config_indices(index, 7) for index in range(7)]
    flattened = [item for shard in shards for item in shard]
    assert sorted(flattened) == list(range(81))
    assert len(flattened) == len(set(flattened))


def test_descending_midrank_exact_ties_and_singleton():
    actual = descending_midrank_score([0.9, 0.5, 0.5, 0.1])
    np.testing.assert_allclose(actual, [1.0, 0.5, 0.5, 0.0])
    np.testing.assert_array_equal(descending_midrank_score([0.2]), [1.0])


def test_prior_transform_and_feature_views_do_not_mutate_source():
    source = np.asarray(
        [
            [0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.2, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    original = source.copy()
    baseline = transform_feature_matrix(source, "log", BASELINE_COLUMNS)
    augmented = transform_feature_matrix(source, "rank", AUGMENTED_COLUMNS)
    np.testing.assert_array_equal(source, original)
    assert baseline.shape == (2, 4)
    assert augmented.shape == (2, 7)
    np.testing.assert_allclose(baseline[:, 0], np.log([0.8 + 1e-12, 0.2 + 1e-12]))
    np.testing.assert_array_equal(augmented[:, 0], [1.0, 0.0])


def test_shared_config_requires_all_81_by_5_and_uses_epsilon_tie_rule():
    trials = _complete_grid(
        lambda index, seed: 0.5 + (GRID_TIE_EPSILON / 2 if index == 1 else 0.0)
    )
    selected = select_shared_config(trials)
    assert selected["config"]["index"] == 0

    trials = _complete_grid(lambda index, seed: 0.6 if index == 17 else 0.5)
    selected = select_shared_config(trials)
    assert selected["config"]["index"] == 17
    assert set(selected["per_seed_best_epoch"]) == {str(seed) for seed in SEEDS}

    with pytest.raises(RuntimeError, match="all 81 x 5"):
        select_shared_config(trials[:-1])


def test_prior_outer_selection_is_baseline_only_and_tie_order_is_frozen():
    tied = _complete_grid(lambda index, seed: 0.5)
    selection = select_prior_transform({"raw": tied, "log": tied, "rank": tied})
    assert selection["selected_prior_transform"] == "raw"

    better_log = _complete_grid(lambda index, seed: 0.6 if index == 4 else 0.5)
    selection = select_prior_transform(
        {"raw": tied, "log": better_log, "rank": tied}
    )
    assert selection["selected_prior_transform"] == "log"
    assert selection["selected_baseline"]["config"]["index"] == 4


def test_d3_models_are_exactly_capacity_matched():
    assertion = assert_d3_capacity_match()
    assert trainable_parameter_count(4, 48) == 289
    assert trainable_parameter_count(7, 32) == 289
    assert assertion["baseline"]["parameters"] == 289
    assert assertion["augmented"]["parameters"] == 289


def test_trial_uses_official_validation_mrr_and_writes_selected_artifacts(tmp_path):
    selection_cache = {
        "train_products": [
            {
                "features": np.asarray(
                    [[0.9, 0.8, 1.0, 1.0], [0.1, 0.2, 2.0, 0.8]],
                    dtype=np.float32,
                ),
                "positive_indices": [0],
                "negative_indices": [1],
            }
        ],
        "eval_pwc": [
            (
                "CCO",
                [
                    {"smiles": "CC.O", "prior": 0.9},
                    {"smiles": "C.CO", "prior": 0.1},
                ],
            )
        ],
        "eval_ground_truths": ["CC.O"],
        "eval_metadata": [{"source_split": "valid"}],
        "eval_features": [
            np.asarray(
                [[0.9, 0.8, 1.0, 1.0], [0.1, 0.2, 2.0, 0.8]],
                dtype=np.float32,
            )
        ],
    }
    checkpoint = tmp_path / "checkpoint.pt"
    normalizer = tmp_path / "normalizer.npz"
    result = train_validation_trial(
        selection_cache,
        enumerate_d1_grid()[0],
        42,
        "cpu",
        checkpoint,
        normalizer,
        max_epochs=2,
        patience=1,
    )
    assert checkpoint.is_file()
    assert normalizer.is_file()
    assert 1 <= result["best_epoch"] <= 2
    assert result["epochs_completed"] <= 2
    assert result["model_parameters"] == 193


def test_prepare_bundle_strips_test_and_keeps_only_official_validation(tmp_path):
    matrix = np.ones((2, 7), dtype=np.float32)
    legacy = {
        "payload": {
            "feature_mode": "3d+prior",
            "train_products": [
                {
                    "features": matrix,
                    "positive_indices": [0],
                    "negative_indices": [1],
                }
            ],
            "eval_pwc": [("test-product", [])],
            "eval_features": [matrix],
            "eval_ground_truths": ["test-ground-truth"],
            "eval_metadata": [{"source_split": "test"}],
            "audit": {"legacy": True},
        }
    }
    validation = {
        "feature_mode": "3d+prior",
        "conformer_seed": 42,
        "payload": {
            "eval_pwc": [
                (
                    "valid-product",
                    [
                        {"smiles": "CC.O", "prior": 0.8},
                        {"smiles": "C.CO", "prior": 0.2},
                    ],
                )
            ],
            "eval_features": [matrix],
            "eval_ground_truths": ["valid-ground-truth"],
            "eval_metadata": [{"source_split": "valid"}],
        },
        "audit": {"validation": True},
    }
    legacy_path = tmp_path / "legacy.pkl"
    validation_path = tmp_path / "valid.pkl"
    output = tmp_path / "selection.pkl"
    with open(legacy_path, "wb") as handle:
        pickle.dump(legacy, handle)
    with open(validation_path, "wb") as handle:
        pickle.dump(validation, handle)

    prepare_selection_bundle(legacy_path, validation_path, output, 42)
    bundle = load_selection_bundle(output)
    assert "eval_pwc" not in bundle
    assert "eval_features" not in bundle
    assert "post_selection_test" not in bundle
    assert bundle["validation_payload"]["eval_pwc"][0][0] == "valid-product"
    assert bundle["audit"]["test_fields_discarded"] == [
        "eval_features",
        "eval_ground_truths",
        "eval_metadata",
        "eval_pwc",
    ]


def test_test_artifact_is_not_opened_before_valid_selection_freeze(tmp_path):
    invalid = tmp_path / "invalid_freeze.json"
    invalid.write_text(json.dumps({"record_kind": "draft"}), encoding="utf-8")
    missing_test = tmp_path / "must_not_be_opened.pkl"
    with pytest.raises(PermissionError, match="selection freeze"):
        load_test_cache_after_freeze(invalid, missing_test)

    valid_record = {
        "record_kind": "model_selection_freeze",
        "protocol_id": "cap10-tuned-v1",
        "seeds": list(SEEDS),
        "test_partition_loaded": False,
        "retained_train_test_cache_sha256": "0" * 64,
    }
    valid_record["selection_fingerprint"] = canonical_fingerprint(valid_record)
    valid = tmp_path / "valid_freeze.json"
    valid.write_text(json.dumps(valid_record), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_test_cache_after_freeze(valid, missing_test)


def _synthetic_seven_column_bundle():
    matrix = np.asarray(
        [
            [0.9, 0.8, 0.7, 0.4, 0.3, 1.0, 1.0],
            [0.1, 0.2, 0.1, 0.9, -0.2, 2.0, 0.8],
        ],
        dtype=np.float32,
    )
    candidates = [
        {"smiles": "CC.O", "prior": 0.9},
        {"smiles": "C.CO", "prior": 0.1},
    ]
    return {
        "selection_bundle_schema": 1,
        "protocol_id": "cap10-tuned-v1",
        "seeds": list(SEEDS),
        "feature_mode": "3d+prior",
        "representation_provenance": {
            "kind": "indexed_conformer",
            "encoder_control_has_conformer": True,
            "conformer_seed": 42,
            "prepare_conformer_seed_argument_is_scientific_provenance": True,
        },
        "train_products": [
            {
                "features": matrix.copy(),
                "positive_indices": [0],
                "negative_indices": [1],
            }
        ],
        "validation_payload": {
            "eval_pwc": [("CCO", candidates)],
            "eval_features": [matrix.copy()],
            "eval_ground_truths": ["CC.O"],
            "eval_metadata": [{"source_split": "valid"}],
        },
    }


def test_capacity_settings_are_explicit_and_width_is_the_only_arm_difference():
    settings = CapacitySettings(dropout=0.1, learning_rate=3e-4, margin=0.1)
    baseline = capacity_arm_config(settings, "baseline")
    augmented = capacity_arm_config(settings, "augmented")
    assert baseline.hidden_width == 48
    assert augmented.hidden_width == 32
    assert baseline.dropout == augmented.dropout == settings.dropout
    assert baseline.learning_rate == augmented.learning_rate == settings.learning_rate
    assert baseline.margin == augmented.margin == settings.margin
    assert capacity_settings_fingerprint(settings).startswith("sha256:")

    with pytest.raises(ValueError, match="dropout"):
        capacity_arm_config(
            CapacitySettings(dropout=0.2, learning_rate=3e-4, margin=0.1),
            "baseline",
        )

    # The CLI cannot silently choose under-specified non-width settings.
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "prepare-capacity",
                "--selection-bundle",
                "selection.pkl",
                "--prior-freeze",
                "prior.json",
                "--output",
                "capacity.json",
            ]
        )


def test_capacity_both_arms_train_and_post_freeze_scoring_paths(tmp_path):
    bundle = _synthetic_seven_column_bundle()
    output_root = tmp_path / "capacity-output"
    train_test_cache = tmp_path / "train_test.pkl"
    with open(train_test_cache, "wb") as handle:
        pickle.dump({"payload": {"placeholder": True}}, handle)
    bundle["input_fingerprints"] = {
        "retained_train_test_cache": {
            "sha256": tuned_runner.file_sha256(train_test_cache)
        }
    }
    selection_bundle = tmp_path / "selection.pkl"
    with open(selection_bundle, "wb") as handle:
        pickle.dump(bundle, handle)
    bundle_sha256 = "sha256:" + tuned_runner.file_sha256(selection_bundle)
    prior = {
        "record_kind": "prior_transform_freeze",
        "protocol_id": "cap10-tuned-v1",
        "selection_bundle_sha256": bundle_sha256,
        "selected_prior_transform": "raw",
    }
    prior["freeze_fingerprint"] = canonical_fingerprint(prior)
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    capacity_plan = tmp_path / "capacity_plan.json"
    tuned_runner.run_prepare_capacity(
        SimpleNamespace(
            selection_bundle=str(selection_bundle),
            prior_freeze=str(prior_path),
            dropout=0.0,
            learning_rate=1e-4,
            margin=0.0,
            decision_note="Synthetic pre-result settings lock",
            output=str(capacity_plan),
        )
    )
    original_max_epochs = tuned_runner.MAX_EPOCHS
    original_patience = tuned_runner.PATIENCE
    tuned_runner.MAX_EPOCHS = 1
    tuned_runner.PATIENCE = 1
    try:
        for arm in ("baseline", "augmented"):
            tuned_runner.run_capacity(
                SimpleNamespace(
                    selection_bundle=str(selection_bundle),
                    capacity_plan=str(capacity_plan),
                    output_root=str(output_root),
                    arm=arm,
                    seeds=(42,),
                    device="cpu",
                )
            )
    finally:
        tuned_runner.MAX_EPOCHS = original_max_epochs
        tuned_runner.PATIENCE = original_patience

    payload = {
        **bundle["validation_payload"],
        "eval_metadata": [{"source_split": "test"}],
    }
    selected = {}
    for arm in ("baseline", "augmented"):
        trial_path = tuned_runner.capacity_trial_path(output_root, arm, 42)
        trial = json.loads(trial_path.read_text(encoding="utf-8"))
        assert trial["model_parameters"] == 289
        selected[arm] = {"trials": {str(seed): trial for seed in SEEDS}}

    baseline_metrics = _score_test_arm(
        payload,
        selected["baseline"],
        "raw",
        "baseline",
        output_root,
        tmp_path / "predictions",
        "cpu",
    )
    augmented_metrics = _score_test_arm(
        payload,
        selected["augmented"],
        "raw",
        "augmented",
        output_root,
        tmp_path / "predictions",
        "cpu",
    )
    assert set(baseline_metrics) == set(augmented_metrics) == {
        str(seed) for seed in SEEDS
    }
    assert baseline_metrics["42"]["n_test_reactions"] == 1
    assert augmented_metrics["42"]["n_test_reactions"] == 1


def test_capacity_test_artifact_is_locked_until_dedicated_freeze(tmp_path):
    invalid = tmp_path / "invalid_capacity_freeze.json"
    invalid.write_text(
        json.dumps({"record_kind": "capacity_execution_plan"}), encoding="utf-8"
    )
    missing_test = tmp_path / "must_not_be_opened.pkl"
    with pytest.raises(PermissionError, match="D-CAPACITY"):
        load_capacity_test_cache_after_freeze(invalid, missing_test)

    valid_record = {
        "record_kind": "capacity_selection_freeze",
        "protocol_id": "cap10-tuned-v1",
        "control_id": CAPACITY_CONTROL_ID,
        "paired_seeds": list(SEEDS),
        "test_partition_loaded": False,
        "capacity_assertion": assert_d3_capacity_match(),
        "retained_train_test_cache_sha256": "0" * 64,
    }
    valid_record["capacity_freeze_fingerprint"] = canonical_fingerprint(valid_record)
    valid = tmp_path / "valid_capacity_freeze.json"
    valid.write_text(json.dumps(valid_record), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_capacity_test_cache_after_freeze(valid, missing_test)


def test_encoder_control_prepare_omits_conformer_provenance_but_real_cache_is_strict(
    tmp_path,
):
    matrix = np.ones((2, 7), dtype=np.float32)
    compatibility = {
        "cache_layout": "seven-column 3d+prior",
        "conformer_seed_field_omitted": True,
        "prepare_conformer_seed_argument_is_not_encoder_provenance": True,
    }
    legacy = {
        "feature_mode": "3d+prior",
        "encoder_control_has_conformer": False,
        "tuned_runner_compatibility": compatibility,
        "protocol_id": "encoder-control-test",
        "payload": {
            "feature_mode": "3d+prior",
            "train_products": [
                {
                    "features": matrix,
                    "positive_indices": [0],
                    "negative_indices": [1],
                }
            ],
            "eval_pwc": [("CCO", [{"smiles": "CC.O"}, {"smiles": "C.CO"}])],
            "eval_features": [matrix],
            "eval_ground_truths": ["CC.O"],
            "eval_metadata": [{"source_split": "test"}],
        },
    }
    validation = {
        "feature_mode": "3d+prior",
        "encoder_control_has_conformer": False,
        "tuned_runner_compatibility": compatibility,
        "protocol_id": "encoder-control-test",
        "payload": {
            "eval_pwc": [("CCO", [{"smiles": "CC.O"}, {"smiles": "C.CO"}])],
            "eval_features": [matrix],
            "eval_ground_truths": ["CC.O"],
            "eval_metadata": [{"source_split": "valid"}],
        },
    }
    legacy_path = tmp_path / "encoder_train_test.pkl"
    validation_path = tmp_path / "encoder_valid.pkl"
    output = tmp_path / "encoder_selection.pkl"
    with open(legacy_path, "wb") as handle:
        pickle.dump(legacy, handle)
    with open(validation_path, "wb") as handle:
        pickle.dump(validation, handle)
    prepare_selection_bundle(legacy_path, validation_path, output, 999)
    bundle = load_selection_bundle(output)
    assert "conformer_seed" not in bundle
    provenance = bundle["representation_provenance"]
    assert provenance["kind"] == "encoder_control_without_conformer"
    assert provenance["conformer_seed"] is None
    assert provenance["prepare_conformer_seed_argument"] == 999
    assert provenance["prepare_conformer_seed_argument_is_scientific_provenance"] is False

    real_validation = dict(validation)
    real_validation.pop("encoder_control_has_conformer")
    real_validation.pop("tuned_runner_compatibility")
    with open(validation_path, "wb") as handle:
        pickle.dump(real_validation, handle)
    real_legacy = dict(legacy)
    real_legacy.pop("encoder_control_has_conformer")
    real_legacy.pop("tuned_runner_compatibility")
    with open(legacy_path, "wb") as handle:
        pickle.dump(real_legacy, handle)
    with pytest.raises(ValueError, match="must declare conformer_seed"):
        prepare_selection_bundle(legacy_path, validation_path, output, 42)


def test_immutable_decision_records_never_overwrite_existing_content(tmp_path):
    decision = tmp_path / "prior_freeze.json"
    decision.write_text("user-owned", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        immutable_json_dump({"replacement": True}, decision, "prior-transform freeze")
    assert decision.read_text(encoding="utf-8") == "user-owned"

    # The command-level preflight happens before it tries to load any inputs.
    with pytest.raises(FileExistsError, match="prior-transform freeze"):
        tuned_runner.run_select_prior(
            SimpleNamespace(
                output=str(decision),
                output_root=str(tmp_path / "missing-output-root"),
                selection_bundle=str(tmp_path / "missing-selection.pkl"),
            )
        )


def test_missing_trial_record_with_partial_artifacts_fails_closed(tmp_path):
    run_dir = tmp_path / "search" / "seed_42"
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best_checkpoint.pt"
    checkpoint.write_bytes(b"partial-user-checkpoint")
    result = run_dir / "trial.json"
    normalizer = run_dir / "normalizer.npz"
    with pytest.raises(FileExistsError, match="Trial result is absent"):
        require_no_partial_trial_artifacts(result, checkpoint, normalizer)
    assert checkpoint.read_bytes() == b"partial-user-checkpoint"


def test_evaluation_outputs_and_predictions_are_append_never(tmp_path):
    result_dir = tmp_path / "evaluation"
    result_dir.mkdir()
    existing = result_dir / "manifest.json"
    existing.write_text("user-owned", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty evaluation"):
        require_clean_evaluation_result_dir(result_dir)
    assert existing.read_text(encoding="utf-8") == "user-owned"

    predictions = tmp_path / "predictions"
    predictions.mkdir()
    existing_prediction = predictions / "baseline_seed_42.csv"
    existing_prediction.write_text("user-owned-prediction", encoding="utf-8")
    with pytest.raises(FileExistsError, match="reaction-level predictions"):
        _score_test_arm(
            payload={},
            selected={},
            transform="raw",
            arm="baseline",
            output_root=tmp_path,
            prediction_dir=predictions,
            device="cpu",
        )
    assert existing_prediction.read_text(encoding="utf-8") == "user-owned-prediction"
