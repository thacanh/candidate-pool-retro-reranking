import json
import pickle

import numpy as np
import pandas as pd

from rerank.evaluate import evaluate_reranking
from rerank.cached_encoder import SqliteCachedUniMolEncoder
from rerank.features import FeatureExtractor
from rerank.study_data import (
    build_official_feature_cache,
    canonicalize_reactant_set,
    compute_coverage,
    load_candidate_pools,
    load_reactions,
    make_pairwise_dataset,
)
from rerank.data.convert_embedding_cache import convert
from rerank.experiments.run_feature_controls import transform_feature_cache
from rerank.analysis.analyze_controlled_study import clustered_bootstrap, paired_cluster_permutation_test


class BombEncoder:
    def __getattr__(self, name):
        raise AssertionError(f"2D extraction called encoder.{name}")


class FakeExtractor:
    def extract_features_batch(self, product_smiles, candidates, priors, ranks):
        return np.asarray(
            [[prior, rank, len(candidate)] for candidate, prior, rank in zip(candidates, priors, ranks)],
            dtype=np.float32,
        )


def test_fragment_order_is_ignored():
    assert canonicalize_reactant_set("CCO.O") == canonicalize_reactant_set("O.CCO")


def test_2d_features_never_touch_unimol_encoder():
    extractor = FeatureExtractor(BombEncoder(), feature_mode="2d+prior")
    matrix = extractor.extract_features_batch(
        product_smiles="CCO",
        candidates=["CC.O", "CCN"],
        priors=[0.8, 0.2],
        ranks=[0, 1],
    )
    assert matrix.shape == (2, 4)
    assert np.isfinite(matrix).all()


def test_official_split_cache_and_coverage(tmp_path):
    source = pd.DataFrame(
        [
            {"reactants_smiles": "CC.O", "products_smiles": "CCO", "set": "train"},
            {"reactants_smiles": "CN.O", "products_smiles": "CNO", "set": "train"},
            {"reactants_smiles": "CO.O", "products_smiles": "COO", "set": "valid"},
            {"reactants_smiles": "CN.O", "products_smiles": "CNO", "set": "test"},
            {"reactants_smiles": "C.Cl", "products_smiles": "CCl", "set": "test"},
        ]
    )
    source_path = tmp_path / "source.csv"
    source.to_csv(source_path, index=False)
    metadata = pd.DataFrame(
        {
            "reaction_id": range(5),
            "source_split": source["set"],
            "reaction_class": [1, 2, 3, 2, 4],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    records = [
        ("CCO", "CC.O", 0.7),
        ("CCO", "C.CO", 0.3),
        ("CNO", "CN.O", 0.6),
        ("CNO", "C.NO", 0.4),
        ("CCl", "C.Cl", 0.9),
        ("CCl", "C.Cl", 0.8),  # duplicate identity must be removed
        ("CCl", "ClC", 0.1),
    ]
    candidate_path = tmp_path / "candidates.jsonl"
    with open(candidate_path, "w", encoding="utf-8") as handle:
        for product, reactant, prior in records:
            handle.write(
                json.dumps(
                    {"product": product, "reactant": reactant, "prior": prior, "label": 0}
                )
                + "\n"
            )

    reactions = load_reactions(source_path, metadata_path)
    pools, audit = load_candidate_pools(candidate_path)
    assert audit["deduplicated_candidate_lines"] == 1
    coverage = compute_coverage(reactions, pools)
    test_row = coverage[coverage["source_split"] == "test"].iloc[0]
    assert test_row["n_reactions"] == 2
    assert test_row["coverage_all"] == 1.0

    cache = build_official_feature_cache(
        reactions,
        pools,
        FakeExtractor(),
        feature_mode="2d+prior",
        train_split="train",
        eval_split="test",
        exclude_cross_split_train_products=True,
    )
    # CNO appears in both train and test, so only CCO remains in training.
    assert cache["audit"]["train_products"] == 1
    assert cache["audit"]["train_overlap_reactions_excluded"] == 1
    assert cache["audit"]["eval_reactions_covered"] == 2
    dataset = make_pairwise_dataset(cache, seed=42, max_neg_per_pos=5)
    assert len(dataset) == 1
    assert dataset.feature_dim == 3


def test_evaluation_exports_reaction_metadata_and_ranks(tmp_path):
    products = [
        (
            "CCO",
            [
                {"smiles": "CC.O", "prior": 0.7},
                {"smiles": "C.CO", "prior": 0.3},
            ],
        )
    ]
    precomputed = [
        (
            [products[0][1][1], products[0][1][0]],
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
    ]
    output = tmp_path / "eval.csv"
    evaluate_reranking(
        products_with_candidates=products,
        ground_truths=["CC.O"],
        reranker=None,
        output_csv=str(output),
        precomputed_reranked_results=precomputed,
        reaction_metadata=[{"reaction_id": 7, "reaction_class": 1, "source_split": "test"}],
    )
    row = pd.read_csv(output).iloc[0]
    assert row["reaction_id"] == 7
    assert row["baseline_rank"] == 1
    assert row["reranked_rank"] == 2
    assert json.loads(row["reranked_candidates_json"])[0] == "C.CO"


def test_streaming_pickle_conversion_and_disk_encoder(tmp_path):
    expected = np.arange(512, dtype=np.float32).reshape(1, 512)
    source_data = {"CCO": expected}
    for index in range(1_200):
        source_data[f"unused-{index}"] = np.full((2, 512), index, dtype=np.float32)
    source = tmp_path / "embeddings.pkl"
    with open(source, "wb") as handle:
        pickle.dump(source_data, handle, protocol=5)
    output = tmp_path / "embeddings.sqlite"
    metadata = convert(source, output, {"CCO"})
    assert metadata["source_items"] == 1_201
    assert metadata["stored_items"] == 1

    encoder = SqliteCachedUniMolEncoder(str(output), log_misses=False, strict=True)
    actual = encoder.encode_atoms("CCO")
    np.testing.assert_array_equal(actual, expected)


def test_feature_control_views_preserve_base_and_match_capacity():
    base_matrix = np.asarray(
        [
            [0.9, 0.8, 0.7, 1.1, 0.2, 2.0, 1.0],
            [0.3, 0.4, 0.5, 2.2, -0.1, 1.0, 0.8],
            [0.1, 0.2, 0.3, 3.3, 0.6, 3.0, 1.2],
        ],
        dtype=np.float32,
    )
    payload = {
        "feature_mode": "3d+prior",
        "train_products": [{"features": base_matrix.copy()}],
        "eval_features": [base_matrix.copy()],
        "audit": {"feature_mode": "3d+prior"},
    }
    baseline = transform_feature_cache(payload, "prior_2d")
    full = transform_feature_cache(payload, "unimol_all")
    control = transform_feature_cache(payload, "permuted_unimol", control_seed=7)

    assert baseline["train_products"][0]["features"].shape == (3, 4)
    assert full["train_products"][0]["features"].shape == (3, 7)
    assert control["train_products"][0]["features"].shape == (3, 7)
    np.testing.assert_array_equal(payload["train_products"][0]["features"], base_matrix)
    np.testing.assert_array_equal(control["train_products"][0]["features"][:, [0, 1, 5, 6]], base_matrix[:, [0, 1, 5, 6]])
    for column in (2, 3, 4):
        np.testing.assert_array_equal(
            np.sort(control["train_products"][0]["features"][:, column]),
            np.sort(base_matrix[:, column]),
        )


def test_product_clustered_inference_is_paired_and_deterministic():
    differences = np.asarray(
        [[1.0, 0.0, -0.5], [0.5, 0.0, -0.5]], dtype=np.float64
    )
    clusters = np.asarray(["A", "A", "B"], dtype=object)
    first = clustered_bootstrap(
        differences, clusters, 200, np.random.default_rng(11)
    )
    second = clustered_bootstrap(
        differences, clusters, 200, np.random.default_rng(11)
    )
    assert first == second
    assert first[0] == differences.mean()
    p_value = paired_cluster_permutation_test(
        differences, clusters, 500, np.random.default_rng(12)
    )
    assert 0.0 < p_value <= 1.0
