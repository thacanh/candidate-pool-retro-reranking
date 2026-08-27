"""
MODULE 4 — Dataset Builder (AiZynthFinder edition)
====================================================
Builds pairwise training samples from a pre-labeled JSONL file produced
by AiZynthFinder — NO Chemformer, NO candidate generation.

Input file: outputs/rerank_dataset.jsonl
    Each line: {"product": str, "reactant": str, "label": 0|1, "prior": float}
    Multiple rows per product (one per candidate).

Process:
    1. Stream JSONL; group all candidates per product.
    2. Sort candidates by prior DESC → assign rank (0-based).
    3. Split into positives (label=1) and negatives (label=0).
    4. For each positive: sample up to max_neg_per_pos negatives.
    5. Batch-encode all candidates via UniMolEncoder (one pass per product).
    6. Extract 11-dim feature vectors via FeatureExtractor.
    7. Return PairwiseRankingDataset of (x_pos, x_neg) float32 tensor pairs.

DATA SPEC compliance:
    - product_atom_emb  : encode_atoms(product_smiles)   — computed ONCE per product.
    - candidate embs    : encode_atoms_fragments_batch() — ALL candidates in one pass.
    - prior             : used in place of log_prob for f1 and rank assignment.
    - FEATURE_DIM stays at 11 (unchanged from features.py).
    - EmbeddingStore    : optional; used for product embeddings when available.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Suppress RDKit C++ valence / atom warnings
try:
    from rdkit import RDLogger as _RDLogger
    _RDLogger.DisableLog("rdApp.*")
except Exception:
    pass



# Internal data structure


@dataclass
class _Candidate:
    """One AiZynthFinder candidate for a given product."""
    reactant: str
    label:    int    # 0 or 1
    prior:    float
    rank:     int = field(default=0, init=False)  # assigned after sorting



# SMILES canonicalisation helper


def _canonicalize(smiles: str) -> str:
    """Return canonical SMILES using RDKit, or original string on failure."""
    try:
        from rdkit import Chem  # type: ignore
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return canonical if canonical else smiles
    except Exception:
        return smiles



# JSONL streaming + grouping


def _load_and_group(jsonl_path: str) -> Dict[str, List[_Candidate]]:
    """
    Stream the JSONL file line-by-line and group candidates by product.

    After grouping, candidates within each product are sorted by prior
    descending and assigned a 0-based rank.

    Returns:
        dict: canonical_product_smiles → [_Candidate, …]
    """
    logger.info("Streaming JSONL from %s …", jsonl_path)
    groups: Dict[str, List[_Candidate]] = defaultdict(list)
    n_skipped = 0

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for raw in tqdm(fh, desc="Loading JSONL", unit="line"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            product  = rec.get("product",  "")
            reactant = rec.get("reactant", "")
            label    = int(rec.get("label", 0))
            prior    = float(rec.get("prior", 0.0))

            if not product or not reactant:
                n_skipped += 1
                continue

            groups[product].append(_Candidate(reactant=reactant, label=label, prior=prior))

    logger.info(
        "Grouped %d unique products (%d lines skipped).", len(groups), n_skipped
    )

    # Assign rank within each product (prior DESC → rank 0 = best)
    for cands in groups.values():
        cands.sort(key=lambda c: c.prior, reverse=True)
        for rank, cand in enumerate(cands):
            cand.rank = rank

    return dict(groups)



# Core builder — extracts features and assembles pairs


def build_pairwise_dataset(
    jsonl_path:          str,
    feature_extractor,
    max_neg_per_pos:     Optional[int] = None,
    seed:                int = 42,
    embedding_store=None,
    negative_mining: Literal["random", "mixed"] = "random",
) -> "PairwiseRankingDataset":
    """
    Build a ``PairwiseRankingDataset`` from ``rerank_dataset.jsonl``.

    No Chemformer, no candidate generation.  Candidates come from the JSONL.

    Args:
        jsonl_path:
            Path to ``outputs/rerank_dataset.jsonl``.
        feature_extractor:
            A ``FeatureExtractor`` instance (from features.py).
            Must expose ``extract_features_batch(product_smiles, candidates,
            log_probs, dataset_idx)``.
        max_neg_per_pos:
            Maximum negatives to pair with each positive per product.
            ``None`` = use ALL negatives (default).
        seed:
            Random seed for reproducible negative sampling.
        embedding_store:
            Optional ``EmbeddingStore`` / ``MockEmbeddingStore``.
            When provided, product atom embeddings are looked up from the
            store instead of being re-encoded by UniMolEncoder.

    Returns:
        A ``PairwiseRankingDataset`` ready for the training loop.
    """
    rng = random.Random(seed)
    groups = _load_and_group(jsonl_path)

    # Cache coverage check
    # Warn early if the encoder cache is mostly empty — this causes live
    # UniMol calls per SMILES and makes the build extremely slow.
    encoder = getattr(feature_extractor, "encoder", None)
    if hasattr(encoder, "_cache") and len(encoder._cache) > 0:
        # Sample unique SMILES from the dataset
        sample_smiles: set = set()
        for product_smi, cands in groups.items():
            sample_smiles.add(product_smi)
            for c in cands:
                for frag in c.reactant.split("."):
                    sample_smiles.add(frag.strip())
            if len(sample_smiles) > 500:
                break
        hits = sum(1 for s in sample_smiles if s in encoder._cache)
        hit_pct = 100.0 * hits / max(len(sample_smiles), 1)
        if hit_pct < 50:
            logger.warning(
                "LOW CACHE COVERAGE: only %.1f%% of sampled SMILES are cached."
                " Consider running build_embedding.py first to avoid slow live encoding.",
                hit_pct,
            )
        else:
            logger.info("Cache coverage check: %.1f%% hit rate on sampled SMILES.", hit_pct)

    all_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    n_products          = len(groups)
    n_with_positives    = 0
    n_without_positives = 0
    n_without_negatives = 0

    for product_smi, candidates in tqdm(groups.items(), desc="Building pairs", unit="product"):

        positives = [c for c in candidates if c.label == 1]
        negatives = [c for c in candidates if c.label == 0]

        if not positives:
            logger.debug("No positive candidate for '%s' — skipping.", product_smi)
            n_without_positives += 1
            continue

        if not negatives:
            logger.debug("No negative candidates for '%s' — skipping.", product_smi)
            n_without_negatives += 1
            continue

        n_with_positives += 1

        # Batch-encode ALL candidates in one UniMol pass
        # Sort by rank so the feature extractor receives them in priority order
        # (positives and negatives interleaved as they appear in the ranked list).
        all_cands_sorted = sorted(candidates, key=lambda c: c.rank)
        cand_smiles      = [c.reactant for c in all_cands_sorted]

        # prior replaces log_prob: passed directly as the f1 scalar signal
        cand_priors = [c.prior for c in all_cands_sorted]

        # ranks are pre-assigned (0-based, prior DESC) — passed explicitly
        # so FeatureExtractor does not fall back to loop-index as rank
        ranks = [c.rank for c in all_cands_sorted]

        # Resolve dataset_idx for EmbeddingStore (not available from JSONL —
        # pass None; FeatureExtractor will live-encode the product).
        dataset_idx: Optional[int] = None
        if embedding_store is not None:
            # Future: if a product→idx mapping is ever available, plug in here.
            pass

        # extract_features_batch encodes the product ONCE and all candidates
        # in a single encode_atoms_fragments_batch() call.
        feature_matrix = feature_extractor.extract_features_batch(
            product_smiles=product_smi,
            candidates=cand_smiles,
            log_probs=cand_priors,   # prior ← replaces log_prob
            ranks=ranks,             # f2: correct prior-sorted rank
            dataset_idx=dataset_idx,
        )
        # feature_matrix shape: (len(all_cands_sorted), FEATURE_DIM)

        # Build a lookup: reactant_smiles → feature_vector
        feat_by_reactant: Dict[str, np.ndarray] = {
            cand.reactant: feature_matrix[i]
            for i, cand in enumerate(all_cands_sorted)
        }

        # Pairwise (pos, neg) assembly
        for pos in positives:
            x_pos = feat_by_reactant.get(pos.reactant)
            if x_pos is None:
                continue

            neg_pool = negatives
            if max_neg_per_pos is not None:
                if negative_mining == "mixed" and len(negatives) > max_neg_per_pos:
                    # Mixed hard negative sampling to prevent label/prior leakage
                    sorted_negs = sorted(negatives, key=lambda c: c.prior, reverse=True)
                    k1 = max_neg_per_pos // 2
                    k2 = max_neg_per_pos - k1
                    hard = sorted_negs[:k1]
                    remaining = sorted_negs[k1:]
                    rand = rng.sample(remaining, min(k2, len(remaining)))
                    neg_pool = hard + rand
                else:
                    neg_pool = rng.sample(negatives, min(max_neg_per_pos, len(negatives)))

            for neg in neg_pool:
                x_neg = feat_by_reactant.get(neg.reactant)
                if x_neg is None:
                    continue
                all_pairs.append((x_pos.copy(), x_neg.copy()))

    # Report runtime cache coverage metrics if available
    if hasattr(encoder, "get_coverage_metrics"):
        cov = encoder.get_coverage_metrics()
        ratio = cov["coverage_ratio"]
        logger.info(
            "Runtime Cache Coverage Summary: %.1f%% (%d/%d fragments hit, %d skipped)",
            ratio * 100, cov["hit_fragments"], cov["total_fragments"], cov["skipped_fragments"]
        )
        if ratio < 0.95:
            logger.warning(
                "WARNING - LOW RUNTIME CACHE COVERAGE: %.1f%%. This means 3D features are "
                "being zeroed out for missing fragments, potentially invalidating results.",
                ratio * 100
            )

    logger.info(
        "Dataset built: %d pairwise samples from %d/%d products with positives "
        "(%d had no positive, %d had no negative).",
        len(all_pairs),
        n_with_positives,
        n_products,
        n_without_positives,
        n_without_negatives,
    )

    return PairwiseRankingDataset(all_pairs, seed=seed)



# PyTorch Dataset (interface unchanged)


class PairwiseRankingDataset(Dataset):
    """
    PyTorch Dataset of pairwise (positive, negative) feature samples.

    Each item:
        (x_pos, x_neg) — both float32 tensors of shape (FEATURE_DIM,) = (11,)

    Training objective example (margin ranking loss):
        loss = max(0, margin - (score(x_pos) - score(x_neg)))

    Args:
        pairs: Pre-built list of (x_positive, x_negative) numpy array pairs.
        seed:  Random seed (stored for reproducibility metadata).
    """

    def __init__(
        self,
        pairs: List[Tuple[np.ndarray, np.ndarray]],
        seed: int = 42,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for PairwiseRankingDataset.")

        self._pairs = pairs
        self._seed  = seed
        self._tensor_pos = None
        self._tensor_neg = None

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if self._tensor_pos is not None and self._tensor_neg is not None:
            return self._tensor_pos[idx], self._tensor_neg[idx]
        x_pos, x_neg = self._pairs[idx]
        return (
            torch.tensor(x_pos, dtype=torch.float32),
            torch.tensor(x_neg, dtype=torch.float32),
        )

    @property
    def feature_dim(self) -> int:
        """Dimensionality of each feature vector (should be 11)."""
        if not self._pairs:
            return 0
        return len(self._pairs[0][0])

    def apply_normalizer(self, normalizer, *, show_progress: bool = True) -> None:
        """
        Re-normalize all stored (x_pos, x_neg) pairs in-place.

        MUST be called after fit_normalizer_from_dataset() to ensure
        training features match inference features (which go through
        FeatureExtractor.normalizer at runtime).

        Usage:
            dataset = build_pairwise_dataset(...)
            normalizer = fit_normalizer_from_dataset(dataset)
            dataset.apply_normalizer(normalizer)   # ← call this!
            extractor.normalizer = normalizer
        """
        new_pairs = []
        for x_pos, x_neg in tqdm(
            self._pairs,
            desc="Normalizing pairs",
            unit="pair",
            leave=False,
            disable=not show_progress,
        ):
            new_pairs.append((
                normalizer.transform(np.asarray(x_pos, dtype=np.float32)),
                normalizer.transform(np.asarray(x_neg, dtype=np.float32)),
            ))
        self._pairs = new_pairs
        # Materialize contiguous tensors once.  The previous implementation
        # rebuilt two tensors for every pair in every epoch, even though the
        # normalized feature vectors are immutable during training.  Caching
        # them changes neither values nor DataLoader shuffle order.
        self._tensor_pos = torch.from_numpy(
            np.stack([pair[0] for pair in self._pairs]).astype(np.float32, copy=False)
        )
        self._tensor_neg = torch.from_numpy(
            np.stack([pair[1] for pair in self._pairs]).astype(np.float32, copy=False)
        )
        logger.info(
            "apply_normalizer: re-normalized %d pairs in-place.", len(self._pairs)
        )



# High-level helper: load 1 file → auto-split train / eval by product


def build_train_eval_datasets(
    jsonl_path:      str,
    feature_extractor,
    train_ratio:     float = 0.8,
    max_neg_per_pos: Optional[int] = None,
    split_seed:      int = 42,
    train_seed:      int = 42,
    negative_mining: Literal["random", "mixed"] = "random",
) -> Tuple[
    "PairwiseRankingDataset",
    List[Tuple[str, List[dict]]],
    List[str],
]:
    """
    Load 1 JSONL → split at **product level** → build train dataset + eval groups.

    Split at product level (not pair level) to prevent leakage:
    - Each product only appears in train OR eval, never both.

    Args:
        jsonl_path:        Path to the single JSONL file.
        feature_extractor: FeatureExtractor instance (mode='2d' or '3d').
        train_ratio:       Ratio of products used for training (default 0.8).
        max_neg_per_pos:   Maximum negatives per positive during training.
        split_seed:        Random seed for splitting products (train/eval split).
        train_seed:        Random seed for pair sampling & training.
        negative_mining:   Negative mining strategy: 'random' or 'mixed'.

    Returns:
        train_dataset:               PairwiseRankingDataset (train only).
        eval_products_with_cands:    List of (product_smiles, candidates) for eval.
        eval_ground_truths:          List of ground truth reactant strings.
    """
    rng_split = random.Random(split_seed)

    # 1. Load and group all products
    groups = _load_and_group(jsonl_path)
    all_products = sorted(groups.keys())  # sorted for reproducibility

    # Only keep products that have at least 1 positive AND 1 negative
    valid_products = [
        p for p in all_products
        if any(c.label == 1 for c in groups[p])
        and any(c.label == 0 for c in groups[p])
    ]
    logger.info(
        "Total products: %d | valid (has pos+neg): %d | discarded: %d",
        len(all_products), len(valid_products), len(all_products) - len(valid_products),
    )

    # 2. Split at product level
    rng_split.shuffle(valid_products)
    n_train = max(1, int(len(valid_products) * train_ratio))
    train_products = set(valid_products[:n_train])
    eval_products  = set(valid_products[n_train:])

    logger.info(
        "Split: %d train products | %d eval products  (ratio=%.0f%%)",
        len(train_products), len(eval_products), train_ratio * 100,
    )

    # 3. Build training JSONL groups
    train_groups = {p: groups[p] for p in train_products}

    # 4. Build train PairwiseRankingDataset
    all_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    rng_pairs = random.Random(train_seed)

    for product_smi, candidates in tqdm(
        train_groups.items(), desc="Building train pairs", unit="product"
    ):
        positives = [c for c in candidates if c.label == 1]
        negatives = [c for c in candidates if c.label == 0]

        all_cands_sorted = sorted(candidates, key=lambda c: c.rank)
        cand_smiles = [c.reactant for c in all_cands_sorted]
        cand_priors = [c.prior    for c in all_cands_sorted]
        ranks       = [c.rank     for c in all_cands_sorted]

        feature_matrix = feature_extractor.extract_features_batch(
            product_smiles=product_smi,
            candidates=cand_smiles,
            log_probs=cand_priors,
            ranks=ranks,
            dataset_idx=None,
        )

        feat_by_idx = {cand.reactant: feature_matrix[i] for i, cand in enumerate(all_cands_sorted)}

        for pos in positives:
            x_pos = feat_by_idx.get(pos.reactant)
            if x_pos is None:
                continue
            neg_pool = negatives
            if max_neg_per_pos is not None:
                if negative_mining == "mixed" and len(negatives) > max_neg_per_pos:
                    # Mixed hard negative sampling to prevent label/prior leakage
                    sorted_negs = sorted(negatives, key=lambda c: c.prior, reverse=True)
                    k1 = max_neg_per_pos // 2
                    k2 = max_neg_per_pos - k1
                    hard = sorted_negs[:k1]
                    remaining = sorted_negs[k1:]
                    rand = rng_pairs.sample(remaining, min(k2, len(remaining)))
                    neg_pool = hard + rand
                else:
                    neg_pool = rng_pairs.sample(negatives, min(max_neg_per_pos, len(negatives)))
            for neg in neg_pool:
                x_neg = feat_by_idx.get(neg.reactant)
                if x_neg is None:
                    continue
                all_pairs.append((x_pos.copy(), x_neg.copy()))

    train_dataset = PairwiseRankingDataset(all_pairs, seed=train_seed)
    logger.info("Train dataset: %d pairwise samples.", len(train_dataset))

    # Report runtime cache coverage metrics if available
    encoder = getattr(feature_extractor, "encoder", None)
    if hasattr(encoder, "get_coverage_metrics"):
        cov = encoder.get_coverage_metrics()
        ratio = cov["coverage_ratio"]
        logger.info(
            "Runtime Cache Coverage Summary: %.1f%% (%d/%d fragments hit, %d skipped)",
            ratio * 100, cov["hit_fragments"], cov["total_fragments"], cov["skipped_fragments"]
        )
        if ratio < 0.95:
            logger.warning(
                "WARNING - LOW RUNTIME CACHE COVERAGE: %.1f%%. This means 3D features are "
                "being zeroed out for missing fragments, potentially invalidating results.",
                ratio * 100
            )

    # 5. Build eval groups (format cho TripleComparison)
    eval_products_with_cands: List[Tuple[str, List[dict]]] = []
    eval_ground_truths: List[str] = []

    for prod in tqdm(sorted(eval_products), desc="Preparing eval", unit="product"):
        candidates = groups[prod]
        gt = next((c.reactant for c in candidates if c.label == 1), None)
        if gt is None:
            continue
        cands_for_eval = sorted(
            [{"smiles": c.reactant, "prior": c.prior} for c in candidates],
            key=lambda c: c["prior"], reverse=True,
        )
        eval_products_with_cands.append((prod, cands_for_eval))
        eval_ground_truths.append(gt)

    logger.info(
        "Eval: %d products, %d ground truths.",
        len(eval_products_with_cands), len(eval_ground_truths),
    )

    return train_dataset, eval_products_with_cands, eval_ground_truths

