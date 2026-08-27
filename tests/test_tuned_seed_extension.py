import argparse

import pytest

from rerank.revision_tuning import enumerate_d1_grid, train_validation_trial
from rerank.experiments.run_tuned_seed_extension import G1_SEEDS, parse_seeds


def test_g1_seed_family_is_exactly_42_through_61():
    assert G1_SEEDS == tuple(range(42, 62))
    assert parse_seeds("42,61") == (42, 61)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_seeds("41,42")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_seeds("42,42")


def test_trial_seed_gate_stays_d1_by_default_and_requires_explicit_g1_allowlist(
    monkeypatch, tmp_path
):
    calls = []

    def fake_dataset(*args, **kwargs):
        calls.append(kwargs["seed"])
        raise RuntimeError("past seed gate")

    monkeypatch.setattr("rerank.study_data.make_pairwise_dataset", fake_dataset)
    common = (
        {}, enumerate_d1_grid()[0], 47, "cpu",
        tmp_path / "checkpoint.pt", tmp_path / "normalizer.npz",
    )

    with pytest.raises(ValueError, match="explicit trial seed allowlist"):
        train_validation_trial(*common)
    assert calls == []

    with pytest.raises(RuntimeError, match="past seed gate"):
        train_validation_trial(*common, allowed_seeds=G1_SEEDS)
    assert calls == [47]
