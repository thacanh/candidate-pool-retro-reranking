"""
MODULE 3 — Feature Extraction (v3 — fixed ablation)
======================================================
Separate feature vectors into two completely independent modes:

  MODE '2d+prior'  (FEATURE_DIM_2D = 4):
      f1  = prior                         (AiZynth score)
      f3  = morgan_fingerprint_similarity  (2D Tanimoto)
      f9  = number_of_fragments            (structural)
      f10 = heavy_atom_ratio               (structural)

  MODE '3d+prior'  (FEATURE_DIM_3D = 7):
      f1  = prior
      f3  = morgan_fingerprint_similarity
      f5  = atom_set_similarity  (UniMol 3D — soft atom-set Chamfer)
      f6  = reaction_distance    (UniMol 3D — L2 norm of mean-emb diff)
      f8  = cosine_reaction_vec  (UniMol 3D — angle between prod_mean & reaction_vec)
      f9  = number_of_fragments
      f10 = heavy_atom_ratio

FIXES vs v2:
  - REMOVED reaction_vec_norm  (= reaction_distance exactly, r=1.000 — pure duplicate)
  - REMOVED cosine_mean        (r=-0.956 with reaction_distance — near-collinear via
                                 d²=2(1−cos) identity when embeddings are normalized)
  - REMOVED atom_count_penalty because it is algebraically redundant with
            heavy_atom_ratio in the frozen feature set

f2 (rank) is completely removed — rank correlates directly with prior, leading to leakage.

Fair comparison between 2D and 3D:
  - Train model_2d on feature_mode='2d+prior' -> does NOT use UniMol
  - Train model_3d on feature_mode='3d+prior' -> uses UniMol embeddings
  - Δ(3D - 2D) isolates the three features derived from UniMol atom embeddings
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Feature set modes and dimensions (v3 — explicit ablation abstraction)
FEATURE_NAMES_MAP = {
    "prior_only": [
        "prior_or_log_prob",   # f1
    ],
    "2d": [
        "morgan_similarity",   # f3
        "n_fragments",         # f9
        "heavy_atom_ratio",    # f10
    ],
    "3d": [
        "morgan_similarity",   # f3
        "atom_set_similarity", # f5 (UniMol) — soft Chamfer atom similarity
        "reaction_distance",   # f6 (UniMol) — L2 norm of mean-emb diff
        "cosine_reaction_vec", # f8 (UniMol) — cos(prod_mean, reaction_vec)
        "n_fragments",         # f9
        "heavy_atom_ratio",    # f10
        # REMOVED: cosine_mean (r=-0.956 with reaction_distance, near-collinear)
        # REMOVED: reaction_vec_norm (= reaction_distance exactly, r=1.000)
        # REMOVED: atom_count_penalty (= |1-heavy_atom_ratio|, r=1.000 with heavy_atom_ratio)
    ],
    "2d+prior": [
        "prior_or_log_prob",   # f1
        "morgan_similarity",   # f3
        "n_fragments",         # f9
        "heavy_atom_ratio",    # f10
    ],
    "3d+prior": [
        "prior_or_log_prob",   # f1
        "morgan_similarity",   # f3
        "atom_set_similarity", # f5 (UniMol) — soft Chamfer atom similarity
        "reaction_distance",   # f6 (UniMol) — L2 norm of mean-emb diff
        "cosine_reaction_vec", # f8 (UniMol) — cos(prod_mean, reaction_vec)
        "n_fragments",         # f9
        "heavy_atom_ratio",    # f10
        # REMOVED: cosine_mean (r=-0.956 with reaction_distance, near-collinear)
        # REMOVED: reaction_vec_norm (= reaction_distance exactly, r=1.000)
        # REMOVED: atom_count_penalty (= |1-heavy_atom_ratio|, r=1.000 with heavy_atom_ratio)
    ]
}

FEATURE_NAMES_2D = FEATURE_NAMES_MAP["2d+prior"]   # Keep old names for backward compat
FEATURE_NAMES_3D = FEATURE_NAMES_MAP["3d+prior"]
FEATURE_NAMES = FEATURE_NAMES_3D
FEATURE_DIM_2D = len(FEATURE_NAMES_2D)
FEATURE_DIM_3D = len(FEATURE_NAMES_3D)
FEATURE_DIM = FEATURE_DIM_3D

DISABLE_RANK_FEATURE = True  # rank feature REMOVED — was causing leakage



from functools import lru_cache

_MORGAN_GEN = None

@lru_cache(maxsize=100000)
def _get_heavy_atom_count(smiles: str) -> int:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() if mol is not None else 0
    except Exception:
        return 0

@lru_cache(maxsize=100000)
def _get_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        global _MORGAN_GEN
        if _MORGAN_GEN is None:
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
            _MORGAN_GEN = GetMorganGenerator(radius=radius, fpSize=n_bits)
        return _MORGAN_GEN.GetFingerprint(mol)
    except Exception:
        return None

def _morgan_similarity(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> float:
    try:
        fp_a = _get_morgan_fingerprint(smiles_a, radius, n_bits)
        fp_b = _get_morgan_fingerprint(smiles_b, radius, n_bits)
        if fp_a is None or fp_b is None:
            return 0.0
        from rdkit import DataStructs
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception as exc:
        logger.debug("Morgan similarity failed: %s", exc)
        return 0.0


def _atom_count_penalty(product_smiles: str, candidate_smiles: str) -> float:
    try:
        n_product = _get_heavy_atom_count(product_smiles)
        n_candidate = sum(
            _get_heavy_atom_count(f.strip())
            for f in candidate_smiles.split(".")
            if f.strip()
        )
        return abs(n_product - n_candidate) / max(n_product, 1)
    except Exception:
        return 1.0


def _heavy_atom_ratio(product_smiles: str, candidate_smiles: str) -> float:
    try:
        n_product = _get_heavy_atom_count(product_smiles)
        if n_product == 0:
            return 1.0
        n_candidate = sum(
            _get_heavy_atom_count(f.strip())
            for f in candidate_smiles.split(".")
            if f.strip()
        )
        return n_candidate / n_product
    except Exception:
        return 1.0



# Atom-level 3D features (AGENT.md DATA SPEC formulas)


def _atom_set_similarity(prod_emb: np.ndarray, react_emb: np.ndarray) -> float:
    """
    Symmetric soft atom-set similarity: average of product-to-reactant
    and reactant-to-product mean max cosine similarities.
    """
    if prod_emb.size == 0 or react_emb.size == 0:
        return 0.0
    # L2-normalize each atom embedding
    p_norm = prod_emb / (np.linalg.norm(prod_emb, axis=1, keepdims=True) + 1e-10)
    r_norm = react_emb / (np.linalg.norm(react_emb, axis=1, keepdims=True) + 1e-10)
    sim = p_norm @ r_norm.T
    prod_to_react = sim.max(axis=1).mean()
    react_to_prod = sim.max(axis=0).mean()
    return float(0.5 * (prod_to_react + react_to_prod))



def _reaction_distance(prod_emb: np.ndarray, react_emb: np.ndarray) -> float:
    """norm(prod.mean(0) - react.mean(0))"""
    if prod_emb.shape[0] == 0 or react_emb.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(prod_emb.mean(axis=0) - react_emb.mean(axis=0)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0



# FIX #6 — embedding quality check


def _check_embedding(name: str, emb: np.ndarray) -> None:
    """Log a warning when an embedding is all-zero (cache miss indicator)."""
    if emb.std() < 1e-9:
        logger.debug(
            "Zero embedding detected for %s — likely a cache miss. "
            "Features f5,f6,f7,f8,f11 will be 0 for this sample.",
            name,
        )



# Feature Normalizer (FIX #4)


class FeatureNormalizer:
    """
    StandardScaler for the 11-dimensional feature vector.

    FIX #4: Without normalization, large-scale features (f6 ≈ [0,50],
    f11 ≈ [0,100+]) dominate gradients and mask small-scale informative
    features (f3, f7 ≈ [0,1]).

    Usage:
        normalizer = FeatureNormalizer()
        normalizer.fit(feature_matrix_2d)   # from training data
        x_norm = normalizer.transform(x)    # at train AND inference time
        normalizer.save("normalizer.npz")
        normalizer = FeatureNormalizer.load("normalizer.npz")

    Clips extreme values to ±clip_sigma to handle outliers.
    """

    def __init__(self, clip_sigma: float = 5.0) -> None:
        self.clip_sigma = clip_sigma
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        if X.ndim == 1:
            X = X[np.newaxis, :]
        self.mean_ = X.mean(axis=0).astype(np.float32)
        self.std_ = np.maximum(X.std(axis=0), 1e-6).astype(np.float32)
        self._fitted = True
        logger.info(
            "FeatureNormalizer fitted on %d samples. Mean: %s, Std: %s",
            len(X), self.mean_.round(4), self.std_.round(4),
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = np.asarray(X, dtype=np.float32)
        was_1d = X.ndim == 1
        if was_1d:
            X = X[np.newaxis, :]
        X_norm = (X - self.mean_) / self.std_
        X_norm = np.clip(X_norm, -self.clip_sigma, self.clip_sigma)
        return X_norm[0] if was_1d else X_norm

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted.")
        np.savez(path, mean=self.mean_, std=self.std_, clip_sigma=[self.clip_sigma])
        logger.info("FeatureNormalizer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        data = np.load(path)
        norm = cls(clip_sigma=float(data["clip_sigma"][0]))
        norm.mean_ = data["mean"].astype(np.float32)
        norm.std_ = data["std"].astype(np.float32)
        norm._fitted = True
        logger.info("FeatureNormalizer loaded from %s", path)
        return norm

    @property
    def is_fitted(self) -> bool:
        return self._fitted



# Main class


class FeatureExtractor:
    """
    Multi-modal feature extractor for (product, candidate) pairs.

    FIXES:
      FIX #3: f2 = rank / max(n_candidates - 1, 1) → [0, 1].
              Set DISABLE_RANK_FEATURE=True in this module to zero it.
      FIX #4: normalizer applied in extract_features_batch() (batch path).
      FIX #6: embedding quality check logs zero-vector warnings.

    Args:
        encoder:         A UniMolEncoder or CachedUniMolEncoder instance.
        embedding_store: Optional EmbeddingStore for product embeddings.
        normalizer:      Optional fitted FeatureNormalizer. If provided,
                         features are normalized before being returned.
    """

    def __init__(
        self,
        encoder,
        embedding_store=None,
        normalizer: Optional[FeatureNormalizer] = None,
        feature_mode: Literal["prior_only", "2d", "3d", "2d+prior", "3d+prior"] = "3d+prior",
    ) -> None:
        self.encoder = encoder
        self.store = embedding_store
        self.normalizer = normalizer
        if feature_mode not in FEATURE_NAMES_MAP:
            raise ValueError(
                f"feature_mode must be one of {list(FEATURE_NAMES_MAP.keys())}, "
                f"got {feature_mode!r}"
            )
        self._feature_mode = feature_mode
        logger.info(
            "FeatureExtractor initialized in mode='%s' (dim=%d)",
            self._feature_mode,
            self.feature_dim,
        )

    @property
    def feature_dim(self) -> int:
        return len(FEATURE_NAMES_MAP[self._feature_mode])



    # Public API — single candidate


    def extract_features(
        self,
        product_smiles: str,
        candidate_smiles: str,
        log_prob: float,
        rank: int,
        n_candidates: int = 10,
        product_atom_emb: Optional[np.ndarray] = None,
        dataset_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the 11-dim feature vector for one candidate.

        For batch extraction (all candidates for one product), prefer
        extract_features_batch() which calls the batched encoder.

        Args:
            n_candidates: Total number of candidates for this product.
                          Used to scale f2 to [0,1].
        """
        if "3d" in self._feature_mode:
            prod_emb = self._resolve_product_emb(
                product_smiles, product_atom_emb, dataset_idx
            )
            react_emb = self.encoder.encode_atoms_fragments(candidate_smiles)
            _check_embedding(f"product:{product_smiles[:20]}", prod_emb)
            _check_embedding(f"reactant:{candidate_smiles[:20]}", react_emb)
        else:
            # The controlled 2D arm must not touch the UniMol cache.  Apart
            # from being unnecessary, loading the atom cache here consumed
            # ~18 GB and made a nominally 2D experiment depend on 3D assets.
            prod_emb = np.empty((0, 0), dtype=np.float32)
            react_emb = np.empty((0, 0), dtype=np.float32)
        return self._compute(
            product_smiles, candidate_smiles, log_prob, rank,
            n_candidates, prod_emb, react_emb,
        )


    # Public API — batch (all candidates for one product)


    def extract_features_batch(
        self,
        product_smiles: str,
        candidates: List[str],
        priors: Optional[List[float]] = None,
        log_probs: Optional[List[float]] = None,
        ranks: Optional[List[int]] = None,
        dataset_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract features for all candidates of one product.

        FIX #3: f2 is now scaled by (len(candidates) - 1) so it lives in [0,1]
                regardless of beam size. This reduces rank leakage magnitude.

        Args:
            product_smiles: Target product SMILES.
            candidates:     List of candidate reactant SMILES strings.
            priors:         AiZynthFinder prior probabilities (preferred).
            log_probs:      Backward-compat alias for priors.
            ranks:          Pre-assigned 0-based ranks. Falls back to loop index.
            dataset_idx:    Optional dataset row index for store-mode.

        Returns:
            np.ndarray of shape (len(candidates), FEATURE_DIM).
        """
        if not candidates:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        values: List[float] = priors if priors is not None else (log_probs or [])
        if len(values) != len(candidates):
            raise ValueError(
                f"Length mismatch: {len(candidates)} candidates but "
                f"{len(values)} prior/log_prob values."
            )

        if "3d" in self._feature_mode:
            prod_emb = self._resolve_product_emb(product_smiles, None, dataset_idx)
            _check_embedding(f"product:{product_smiles[:30]}", prod_emb)
            react_embs = self.encoder.encode_atoms_fragments_batch(candidates)
        else:
            prod_emb = np.empty((0, 0), dtype=np.float32)
            react_embs = [
                np.empty((0, 0), dtype=np.float32) for _ in candidates
            ]

        n_cands = len(candidates)
        features = []
        for i, (cand, prior, react_emb) in enumerate(zip(candidates, values, react_embs)):
            rank = ranks[i] if ranks is not None else i
            fv = self._compute(
                product_smiles, cand, prior, rank, n_cands, prod_emb, react_emb,
            )
            features.append(fv)

        result = np.stack(features, axis=0)

        # FIX #4: apply normalizer on the full batch
        if self.normalizer is not None and self.normalizer.is_fitted:
            result = self.normalizer.transform(result)

        return result


    # Internal


    def _compute(
        self,
        product_smiles: str,
        candidate_smiles: str,
        log_prob: float,
        rank: int,
        n_candidates: int,
        prod_emb: np.ndarray,
        react_emb: np.ndarray,
    ) -> np.ndarray:
        """
        Build feature vector. Mode is controlled by self._feature_mode.
        Supports: prior_only, 2d, 3d, 2d+prior, 3d+prior
        """
        feats = {}
        
        # prior (f1)
        feats["prior_or_log_prob"] = float(log_prob)
        
        # 2D features
        fragments = [f.strip() for f in candidate_smiles.split(".") if f.strip()]
        feats["morgan_similarity"] = _morgan_similarity(product_smiles, candidate_smiles)
        feats["atom_count_penalty"] = _atom_count_penalty(product_smiles, candidate_smiles)
        feats["n_fragments"] = float(len(fragments))
        feats["heavy_atom_ratio"] = _heavy_atom_ratio(product_smiles, candidate_smiles)
        
        # 3D features (UniMol)
        if "3d" in self._feature_mode:
            # Safeguard in case embeddings are empty
            if prod_emb.size > 0 and react_emb.size > 0:
                prod_mean = prod_emb.mean(axis=0)
                react_mean = react_emb.mean(axis=0)
                reaction_vec = prod_mean - react_mean

                feats["atom_set_similarity"] = _atom_set_similarity(prod_emb, react_emb)
                feats["reaction_distance"] = _reaction_distance(prod_emb, react_emb)
                feats["cosine_reaction_vec"] = _cosine(prod_mean, reaction_vec)
                # NOTE: cosine_mean REMOVED — r=-0.956 with reaction_distance
                # (d²=2(1−cos) identity for normalized embeddings → collinear)
                # NOTE: reaction_vec_norm REMOVED — identical to reaction_distance (r=1.000)
            else:
                feats["atom_set_similarity"] = 0.0
                feats["reaction_distance"] = 0.0
                feats["cosine_reaction_vec"] = 0.0
                
        ordered_names = FEATURE_NAMES_MAP[self._feature_mode]
        return np.array([feats[name] for name in ordered_names], dtype=np.float32)


    def _resolve_product_emb(
        self,
        product_smiles: str,
        explicit_emb: Optional[np.ndarray],
        dataset_idx: Optional[int],
    ) -> np.ndarray:
        if explicit_emb is not None:
            emb = np.asarray(explicit_emb, dtype=np.float32)
            return emb[np.newaxis, :] if emb.ndim == 1 else emb

        if self.store is not None and dataset_idx is not None:
            sample = self.store.get_embedding(dataset_idx)
            if sample is not None:
                return sample.product_emb
            logger.debug("dataset_idx=%d not in store; falling back to live encode.", dataset_idx)

        return self.encoder.encode_atoms(product_smiles)



# Utility: fit FeatureNormalizer from a PairwiseRankingDataset


def fit_normalizer_from_dataset(dataset) -> FeatureNormalizer:
    """
    Collect all feature vectors from a PairwiseRankingDataset and fit
    a FeatureNormalizer on them.

    Both positive and negative feature vectors are included so the normalizer
    statistics reflect the full feature distribution.

    FIX #4: This must be called BEFORE building the FeatureExtractor for
    inference. Save the result and reload it at inference time.

    Usage:
        dataset = build_pairwise_dataset(...)
        normalizer = fit_normalizer_from_dataset(dataset)
        normalizer.save("outputs/normalizer.npz")

        # Attach to extractor for inference
        extractor = FeatureExtractor(encoder, normalizer=normalizer)
    """
    all_feats = []
    for x_pos, x_neg in dataset:
        all_feats.append(x_pos.numpy())
        all_feats.append(x_neg.numpy())

    X = np.stack(all_feats, axis=0)   # (2 * N_pairs, FEATURE_DIM)
    norm = FeatureNormalizer()
    norm.fit(X)
    logger.info(
        "FeatureNormalizer fitted on %d feature vectors (%d pairs).",
        len(all_feats), len(dataset),
    )
    return norm
