"""Feature extraction, training and evaluation for reaction-candidate reranking."""
from rerank.encoder import UniMolEncoder
from rerank.cached_encoder import CachedUniMolEncoder
from rerank.features import (
    FeatureExtractor,
    FeatureNormalizer,
    FEATURE_NAMES,
    fit_normalizer_from_dataset,
)
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
    # Embedding storage
    "EmbeddingStore",
    "MockEmbeddingStore",
    "SampleEmbedding",
    "atom_set_similarity",
    "reaction_distance",
]
