"""
DATA SPEC — UniMol Embedding Store
====================================
Loads precomputed 3D atom-level embeddings from the standard directory layout
defined in AGENT.md and provides correctly index-mapped access to them.

Required directory structure:
    DATA_ROOT/
        product_emb.npy        # object array, each entry shape (Np, 512)
        reactant_emb.npy       # object array, each entry shape (Nr, 512)
        product_coords.npy     # optional, each entry shape (Np, 3)
        reactant_coords.npy    # optional, each entry shape (Nr, 3)
        valid_indices.json     # list of original dataset indices that have embeddings
        metadata.json          # optional
        reactions.csv          # original dataset (SMILES)

CRITICAL Rules (from AGENT.md):
    - NEVER index .npy arrays directly with dataset_idx.
    - ALWAYS use idx_map: dataset_idx → embedding_idx.
    - Samples absent from valid_indices.json must be skipped (return None).
    - Embeddings are atom-level: shape (n_atoms, 512) — NOT a single vector.
    - reactant_emb.npy holds ground-truth embeddings ONLY.
      Candidate embeddings MUST be computed separately via UniMolEncoder.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBED_DIM = 512



# Data container


@dataclass
class SampleEmbedding:
    """Atom-level embeddings and optional coordinates for one reaction sample."""

    dataset_idx: int                      # original index in reactions.csv
    product_emb: np.ndarray               # shape (Np, 512)
    reactant_emb: np.ndarray              # shape (Nr, 512) — ground-truth only
    product_coords: Optional[np.ndarray]  # shape (Np, 3) or None
    reactant_coords: Optional[np.ndarray] # shape (Nr, 3) or None

    def __post_init__(self):
        # Validate dimensionality at construction time.
        self._validate_emb("product_emb", self.product_emb)
        self._validate_emb("reactant_emb", self.reactant_emb)

    @staticmethod
    def _validate_emb(name: str, arr: np.ndarray) -> None:
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be 2D (n_atoms, {EMBED_DIM}), got shape {arr.shape}"
            )
        if arr.shape[1] != EMBED_DIM:
            raise ValueError(
                f"{name} embedding dim must be {EMBED_DIM}, got {arr.shape[1]}"
            )



# Embedding Store


class EmbeddingStore:
    """
    Load and serve precomputed UniMol atom-level embeddings via a safe
    index mapping enforced by valid_indices.json.

    Args:
        data_root: Path to the DATA_ROOT directory containing the .npy
                   and .json files.
        load_coords: If True, also load coordinate arrays (optional files).
    """

    def __init__(
        self,
        data_root: str | Path,
        load_coords: bool = True,
    ) -> None:
        self._root = Path(data_root)
        self._load_coords = load_coords

        # Mapping: original_dataset_idx → position in .npy arrays
        self._idx_map: Dict[int, int] = {}
        self._valid_indices: List[int] = []

        # Raw numpy arrays (object dtype — variable-length atom embeddings)
        self._product_embs: Optional[np.ndarray] = None
        self._reactant_embs: Optional[np.ndarray] = None
        self._product_coords: Optional[np.ndarray] = None
        self._reactant_coords: Optional[np.ndarray] = None

        self._loaded = False


    # Loading


    def load(self) -> "EmbeddingStore":
        """
        Load all embedding data from DATA_ROOT.
        Returns self for chaining.
        """
        self._load_index_map()
        self._load_embeddings()
        self._loaded = True
        logger.info(
            "EmbeddingStore loaded: %d valid samples from %s",
            len(self._valid_indices),
            self._root,
        )
        return self

    def _load_index_map(self) -> None:
        """
        Step 1 (AGENT.md): Load valid_indices.json and build idx_map.

            idx_map = { orig_idx: i for i, orig_idx in enumerate(valid_indices) }
        """
        index_path = self._root / "valid_indices.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"valid_indices.json not found at {index_path}. "
                "This file is required for correct index mapping."
            )

        with open(index_path, "r", encoding="utf-8") as f:
            self._valid_indices = json.load(f)

        if not isinstance(self._valid_indices, list):
            raise ValueError("valid_indices.json must contain a JSON list of integers.")

        # Build forward mapping: original_idx → position in embedding array
        self._idx_map = {
            int(orig_idx): i
            for i, orig_idx in enumerate(self._valid_indices)
        }
        logger.debug("Index map built: %d entries.", len(self._idx_map))

    def _load_embeddings(self) -> None:
        """Step 2 (AGENT.md): Load .npy arrays with allow_pickle=True."""
        self._product_embs = self._load_npy("product_emb.npy", required=True)
        self._reactant_embs = self._load_npy("reactant_emb.npy", required=True)

        if self._load_coords:
            self._product_coords = self._load_npy("product_coords.npy", required=False)
            self._reactant_coords = self._load_npy("reactant_coords.npy", required=False)

        # Sanity check: length must match valid_indices
        n_valid = len(self._valid_indices)
        for name, arr in [
            ("product_emb.npy", self._product_embs),
            ("reactant_emb.npy", self._reactant_embs),
        ]:
            if arr is not None and len(arr) != n_valid:
                raise ValueError(
                    f"{name} has {len(arr)} entries but valid_indices.json "
                    f"has {n_valid} entries. They must match."
                )

    def _load_npy(self, filename: str, required: bool = True) -> Optional[np.ndarray]:
        path = self._root / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Required file not found: {path}")
            logger.debug("Optional file not found, skipping: %s", path)
            return None
        arr = np.load(path, allow_pickle=True)
        logger.debug("Loaded %s: %d entries.", filename, len(arr))
        return arr


    # Access


    def get_embedding(self, dataset_idx: int) -> Optional[SampleEmbedding]:
        """
        Step 3 (AGENT.md): Retrieve atom-level embeddings for a sample.

        ❌ NEVER uses dataset_idx to index .npy arrays directly.
        ✅ ALWAYS translates via idx_map first.

        Args:
            dataset_idx: The row index of the sample in the original dataset
                         (reactions.csv row number, 0-based).

        Returns:
            SampleEmbedding if dataset_idx is in valid_indices, else None.
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before accessing embeddings.")

        # Rule 2: handle missing samples gracefully
        if dataset_idx not in self._idx_map:
            logger.debug(
                "dataset_idx=%d not in valid_indices — sample skipped.", dataset_idx
            )
            return None

        # Rule 1: ALWAYS use idx_map, NEVER raw dataset_idx
        i = self._idx_map[dataset_idx]

        prod_emb = np.asarray(self._product_embs[i], dtype=np.float32)
        react_emb = np.asarray(self._reactant_embs[i], dtype=np.float32)

        # Ensure 2D: some storages save (512,) for single-atom molecules
        if prod_emb.ndim == 1:
            prod_emb = prod_emb[np.newaxis, :]   # (1, 512)
        if react_emb.ndim == 1:
            react_emb = react_emb[np.newaxis, :] # (1, 512)

        prod_coords = None
        react_coords = None
        if self._product_coords is not None:
            prod_coords = np.asarray(self._product_coords[i], dtype=np.float32)
        if self._reactant_coords is not None:
            react_coords = np.asarray(self._reactant_coords[i], dtype=np.float32)

        return SampleEmbedding(
            dataset_idx=dataset_idx,
            product_emb=prod_emb,
            reactant_emb=react_emb,
            product_coords=prod_coords,
            reactant_coords=react_coords,
        )

    def has_embedding(self, dataset_idx: int) -> bool:
        """Return True if dataset_idx has a precomputed embedding."""
        return dataset_idx in self._idx_map

    @property
    def valid_indices(self) -> List[int]:
        """All dataset indices that have precomputed embeddings."""
        return list(self._valid_indices)

    @property
    def n_samples(self) -> int:
        return len(self._valid_indices)

    def __len__(self) -> int:
        return self.n_samples



# Atom-level 3D feature functions (MANDATORY per AGENT.md)


def atom_set_similarity(prod_emb: np.ndarray, react_emb: np.ndarray) -> float:
    """
    Atom-set similarity between product and reactant embeddings.

    Per AGENT.md:
        sim = prod_emb @ react_emb.T      # (Np, Nr)
        return sim.max(axis=1).mean()

    Each product atom is matched to its most similar reactant atom,
    then averaged — a soft Hausdorff-like measure.

    Args:
        prod_emb:  shape (Np, 512)
        react_emb: shape (Nr, 512)

    Returns:
        Scalar similarity score.
    """
    if prod_emb.shape[0] == 0 or react_emb.shape[0] == 0:
        return 0.0

    # (Np, Nr) — raw dot-product similarity
    sim = prod_emb @ react_emb.T
    return float(sim.max(axis=1).mean())


def reaction_distance(prod_emb: np.ndarray, react_emb: np.ndarray) -> float:
    """
    L2 distance between mean product embedding and mean reactant embedding.

    Per AGENT.md:
        return norm(prod_emb.mean(axis=0) - react_emb.mean(axis=0))

    Args:
        prod_emb:  shape (Np, 512)
        react_emb: shape (Nr, 512)

    Returns:
        Scalar distance.
    """
    if prod_emb.shape[0] == 0 or react_emb.shape[0] == 0:
        return 0.0

    prod_mean = prod_emb.mean(axis=0)   # (512,)
    react_mean = react_emb.mean(axis=0) # (512,)
    return float(np.linalg.norm(prod_mean - react_mean))



# Helpers for building a mock store (testing / development)


class MockEmbeddingStore:
    """
    In-memory EmbeddingStore substitute for testing without real .npy files.
    Uses random atom-level embeddings of configurable shape.

    Args:
        valid_indices: List of dataset indices to treat as valid.
        n_atoms_product: Number of atoms in mock product embeddings.
        n_atoms_reactant: Number of atoms in mock reactant embeddings.
        embed_dim: Embedding dimensionality.
        seed: Random seed.
    """

    def __init__(
        self,
        valid_indices: List[int],
        n_atoms_product: int = 5,
        n_atoms_reactant: int = 4,
        embed_dim: int = EMBED_DIM,
        seed: int = 0,
    ) -> None:
        self._valid_indices = list(valid_indices)
        self._idx_map: Dict[int, int] = {
            int(orig): i for i, orig in enumerate(valid_indices)
        }
        rng = np.random.RandomState(seed)
        n = len(valid_indices)
        # Pre-generate random atom embeddings
        self._product_embs = [
            rng.randn(n_atoms_product, embed_dim).astype(np.float32) for _ in range(n)
        ]
        self._reactant_embs = [
            rng.randn(n_atoms_reactant, embed_dim).astype(np.float32) for _ in range(n)
        ]
        self._loaded = True

    def get_embedding(self, dataset_idx: int) -> Optional[SampleEmbedding]:
        if dataset_idx not in self._idx_map:
            return None
        i = self._idx_map[dataset_idx]
        return SampleEmbedding(
            dataset_idx=dataset_idx,
            product_emb=self._product_embs[i],
            reactant_emb=self._reactant_embs[i],
            product_coords=None,
            reactant_coords=None,
        )

    def has_embedding(self, dataset_idx: int) -> bool:
        return dataset_idx in self._idx_map

    @property
    def valid_indices(self) -> List[int]:
        return list(self._valid_indices)

    @property
    def n_samples(self) -> int:
        return len(self._valid_indices)

    def __len__(self) -> int:
        return self.n_samples
