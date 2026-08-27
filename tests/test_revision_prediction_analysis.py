import numpy as np
import pandas as pd
import pytest

from rerank.analysis.analyze_revision_predictions import (
    CONTINUOUS_PREDICTORS,
    benjamini_hochberg,
    build_net_flips,
    clustered_sign_flip,
    crossed_product_seed_bootstrap,
    plot_descriptor_figure,
)


def test_bh_known_family_and_fixed_order():
    observed = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
    np.testing.assert_allclose(observed, [0.02, 0.04, 0.04, 0.008])


def test_crossed_seed_bootstrap_is_deterministic():
    differences = np.asarray([[1, 0, -1, 1], [1, 1, -1, 0]], dtype=float)
    clusters = np.asarray(["A", "A", "B", "C"], dtype=object)
    first = crossed_product_seed_bootstrap(differences, clusters, 100, 2026)
    second = crossed_product_seed_bootstrap(differences, clusters, 100, 2026)
    assert first == second
    assert first["effect"] == pytest.approx(differences.mean())


def test_sign_flip_null_and_net_flip_counts():
    zeros = np.zeros((2, 4))
    assert clustered_sign_flip(zeros, np.asarray(list("ABCD")), 100, 2027) == 1.0
    matrices = {
        "reranked_hit@1": {
            "baseline": np.asarray([[0, 1, 0], [1, 0, 0]]),
            "augmented": np.asarray([[1, 0, 0], [1, 1, 0]]),
        }
    }
    frame = build_net_flips(matrices, [42, 43])
    assert frame["improved"].tolist() == [1, 1]
    assert frame["degraded"].tolist() == [1, 0]
    assert frame["unchanged"].tolist() == [1, 2]


def test_descriptor_figure_is_written(tmp_path):
    frame = pd.DataFrame(
        {
            "promoted": [0, 0, 1, 1],
            **{name: [0.0, 1.0, 0.5, 1.5] for name in CONTINUOUS_PREDICTORS},
        }
    )
    target = tmp_path / "descriptors.png"
    plot_descriptor_figure(frame, target)
    assert target.is_file() and target.stat().st_size > 0
