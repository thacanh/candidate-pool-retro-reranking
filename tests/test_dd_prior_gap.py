import numpy as np
import pytest

from rerank.analysis.analyze_dd_prior_gap import (
    bootstrap_groups, bootstrap_pools, gap_groups, interval_row, weights,
)


def test_gap_singletons_ties_and_boundary():
    gaps, cuts, groups = gap_groups([[1], [1, 1], [1, 0.75], [1, 0.5], [1, 0.25], [1, 0]])
    assert np.isnan(gaps[0])
    np.testing.assert_array_equal(cuts, [.25, .5, .75])
    np.testing.assert_array_equal(groups, [-1, 0, 0, 1, 2, 3])


def test_no_arbitrary_split_of_equal_gaps():
    _, cuts, groups = gap_groups([[1, 1]]*8)
    np.testing.assert_array_equal(cuts, [0, 0, 0])
    np.testing.assert_array_equal(groups, np.zeros(8))


@pytest.mark.parametrize("priors", [[], [[1]], [[np.nan, 1]], [[]]])
def test_invalid_gaps_fail(priors):
    with pytest.raises(ValueError):
        gap_groups(priors)


def test_weights_normalize():
    w = weights(np.random.default_rng(1), 100, 7)
    np.testing.assert_allclose(w.sum(axis=1), 1)
    np.testing.assert_allclose(w*7, np.rint(w*7))


def test_pool_difference_shared_products():
    # With identical seed-invariant inputs and shared products, difference is zero.
    v = np.broadcast_to(np.arange(5)/10, (2, 20, 5))
    a = bootstrap_pools(v, 200, 5)
    np.testing.assert_allclose(a[:, 2], 0, atol=1e-15)
    np.testing.assert_array_equal(a, bootstrap_pools(v, 200, 5))


def test_pool_constant_shift():
    v = np.ones((2, 20, 4))
    v[0] *= .1
    v[1] *= .3
    b = bootstrap_pools(v, 101, 6)
    np.testing.assert_allclose(b, np.tile([.1, .3, .2], (101, 1)))


def test_group_seed_pairing_and_empty_bins():
    groups = np.array([0, 0, 3, 3])
    v = np.broadcast_to(np.arange(20)[None, :, None]/20, (2, 20, 4))
    b = bootstrap_groups(v, groups, 201, 7)
    assert np.isnan(b[:, 1:3]).all()
    np.testing.assert_allclose(b[:, 3]-b[:, 0], 0, atol=1e-15)


def test_direct_group_contrast_known_difference():
    v = np.zeros((2, 20, 8))
    v[:, :, 6:] = .25
    b = bootstrap_groups(v, np.repeat(np.arange(4), 2), 100, 8)
    np.testing.assert_allclose(b[:, 3]-b[:, 0], .25)
    row = interval_row("contrast", .25, b[:, 3, 0]-b[:, 0, 0])
    assert row["effect_pp"] == 25
    assert np.isclose(row["ci95_low_pp"], 25)
