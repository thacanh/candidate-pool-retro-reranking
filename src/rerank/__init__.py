"""
Learning-to-rank reranking pipeline for retrosynthesis.
Candidates are produced by AiZynthFinder (NOT Chemformer).

Modules:
    encoder          - Module 2: UniMol SMILES encoder with LRU cache
    features         - Module 3: Multi-modal 11-dim feature extraction
    dataset          - Module 4: Pairwise dataset builder (JSONL-based)
    model            - Module 5: MLP reranker
    loss             - Module 6: Pairwise ranking loss
    trainer          - Module 7: Training loop
    inference        - Module 8: Reranking inference
    evaluate         - Module 9: Evaluation (baseline vs. reranked)
    embedding_store  - DATA SPEC: UniMol atom-level embedding store
"""

from rerank.encoder import UniMolEncoder
from rerank.cached_encoder import CachedUniMolEncoder
from rerank.features import FeatureExtractor, FeatureNormalizer, FEATURE_NAMES, fit_normalizer_from_dataset
from rerank.dataset import PairwiseRankingDataset, build_pairwise_dataset
from rerank.model import RankerMLP
from rerank.loss import pairwise_ranking_loss
from rerank.trainer import RankerTrainer
from rerank.inference import Reranker
from rerank.evaluate import evaluate_reranking
from rerank.embedding_store import (
    EmbeddingStore,
    MockEmbeddingStore,
    SampleEmbedding,
    atom_set_similarity,
    reaction_distance,
)

__all__ = [
    # Encoding
    "UniMolEncoder",
    "CachedUniMolEncoder",
    # Features
    "FeatureExtractor",
    "FeatureNormalizer",
    "FEATURE_NAMES",
    "fit_normalizer_from_dataset",
    # Dataset
    "PairwiseRankingDataset",
    "build_pairwise_dataset",
    # Model / training
    "RankerMLP",
    "pairwise_ranking_loss",
    "RankerTrainer",
    # Inference / evaluation
    "Reranker",
    "evaluate_reranking",
    # DATA SPEC
    "EmbeddingStore",
    "MockEmbeddingStore",
    "SampleEmbedding",
    "atom_set_similarity",
    "reaction_distance",
]
