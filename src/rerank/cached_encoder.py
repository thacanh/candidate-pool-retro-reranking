"""
cached_encoder.py
==================
CachedUniMolEncoder: serves atom-level embeddings from a precomputed
pickle cache (built by build_embedding.py).

FIXES:
  FIX #7: log_misses defaults to True so cache misses are visible.
  FIX #7: strict=True raises on miss (use during dataset build to catch gaps).
  FIX #6: coverage_report() helper to check cache hit rate before training.
"""

import os
import pickle
import sqlite3
import numpy as np
from typing import Dict, List, Optional
from rdkit import Chem
from tqdm import tqdm

EMBED_DIM = 512


from functools import lru_cache

@lru_cache(maxsize=100000)
def canon(s: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


class CachedUniMolEncoder:
    def __init__(
        self,
        cache_path: str,
        fallback_device: str = "cpu",
        log_misses: bool = True,    # FIX #7: default True — misses are now visible
        strict: bool = False,       # FIX #7: raise on miss during debug
    ):
        self._cache: Dict[str, Optional[np.ndarray]] = {}
        self._fallback_device = fallback_device
        self._live_encoder = None

        self.log_misses = log_misses
        self.strict = strict

        self.total_fragments = 0
        self.hit_fragments = 0
        self.skipped_fragments = 0

        self._load_cache(cache_path)


    # LOAD CACHE

    def _load_cache(self, path: str) -> None:
        with open(path, "rb") as f:
            raw = pickle.load(f)

        for smi, emb in raw.items():
            self._cache[smi] = (
                None if emb is None else np.asarray(emb, dtype=np.float32)
            )

        print(
            f"[CachedUniMolEncoder] Cache loaded: {self.cache_size} valid "
            f"/ {self.cache_failed} failed / {self.cache_total} total"
        )


    # FIX #6 — coverage report

    def coverage_report(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Check what fraction of SMILES (and their fragments) are in cache.

        Call this before build_pairwise_dataset() to detect low coverage.

        Returns dict with keys: hit_rate, n_hits, n_total, n_missing.
        """
        frags: List[str] = []
        for smi in smiles_list:
            for f in smi.split("."):
                f = f.strip()
                if f:
                    frags.append(f)

        n_total = len(frags)
        hits = 0
        missing = []
        for f in tqdm(frags, desc="Coverage check", unit="frag", leave=False):
            c = canon(f)
            if c is not None and c in self._cache and self._cache[c] is not None:
                hits += 1
            else:
                missing.append(f)

        hit_rate = hits / max(n_total, 1)
        if hit_rate < 0.8:
            import logging
            logging.getLogger(__name__).warning(
                "WARNING - LOW CACHE COVERAGE: %.1f%% (%d/%d fragments hit).\n"
                "   Features f5,f6,f7,f8,f11 will be ZERO for missing entries.\n"
                "   Run: python -m rerank.data.build_embedding --jsonl outputs/rerank_dataset.jsonl",
                hit_rate * 100, hits, n_total,
            )

        return {
            "hit_rate": hit_rate,
            "n_hits": hits,
            "n_total": n_total,
            "n_missing": len(missing),
        }



    # CORE ENCODING (fragment-level)

    def encode_atoms(self, smiles: str) -> np.ndarray:
        """
        Encode a single SMILES (possibly multi-fragment) to atom embeddings.
        Each fragment is looked up independently; results are concatenated.

        Returns shape (n_atoms, EMBED_DIM). Falls back to zero matrix on miss.
        """
        if not smiles:
            return np.zeros((1, EMBED_DIM), dtype=np.float32)

        fragments = [f.strip() for f in smiles.split(".") if f.strip()]
        atom_embs = []

        for frag in fragments:
            c = canon(frag)
            if c is None:
                continue

            self.total_fragments += 1

            if c not in self._cache:
                self.skipped_fragments += 1
                if self.log_misses:
                    import logging
                    logging.getLogger(__name__).warning(
                        "[CACHE MISS] '%s' (canonical: '%s') — skipping fragment. "
                        "Re-run build_embedding.py to populate cache.",
                        frag, c,
                    )
                if self.strict:
                    raise RuntimeError(
                        f"[CACHE MISS] '{c}' — strict mode enabled. "
                        "Run build_embedding.py first."
                    )
                continue

            emb = self._cache[c]
            if emb is None:
                self.skipped_fragments += 1
                if self.log_misses:
                    import logging
                    logging.getLogger(__name__).warning(
                        "[FAILED EMBEDDING] '%s' — encoding failed during build. "
                        "Skipping fragment.",
                        c,
                    )
                if self.strict:
                    raise RuntimeError(f"[FAILED EMBEDDING] '{c}'")
                continue

            self.hit_fragments += 1
            atom_embs.append(emb)

        if not atom_embs:
            return np.zeros((1, EMBED_DIM), dtype=np.float32)

        return np.concatenate(atom_embs, axis=0)

    def get_coverage_metrics(self) -> dict:
        """Return runtime fragment-level cache coverage statistics."""
        coverage_ratio = self.hit_fragments / max(self.total_fragments, 1)
        return {
            "total_fragments": self.total_fragments,
            "hit_fragments": self.hit_fragments,
            "skipped_fragments": self.skipped_fragments,
            "coverage_ratio": coverage_ratio,
        }


    # Alias used by FeatureExtractor
    encode_atoms_fragments = encode_atoms


    # REQUIRED BY FEATURE EXTRACTOR

    def encode_atoms_fragments_batch(self, smiles_list: List[str]) -> List[np.ndarray]:
        """Batch-encode a list of SMILES. Returns list of (n_atoms, EMBED_DIM) arrays."""
        return [self.encode_atoms(s) for s in smiles_list]


    # HIGH LEVEL API

    def encode_smiles(self, smiles_list: List[str]) -> np.ndarray:
        """Encode list of SMILES to molecular-level CLS embeddings (N, EMBED_DIM)."""
        return np.stack(
            [self.encode_atoms(s).mean(axis=0) for s in smiles_list],
            axis=0,
        ).astype(np.float32)

    def encode_single(self, smiles: str) -> np.ndarray:
        """Encode one SMILES to a CLS-level vector (EMBED_DIM,)."""
        return self.encode_smiles([smiles])[0]


    # DEBUG / INFO

    @property
    def cache_size(self) -> int:
        """Number of successfully encoded SMILES in cache."""
        return sum(1 for v in self._cache.values() if v is not None)

    @property
    def cache_failed(self) -> int:
        """Number of SMILES that failed encoding (stored as None)."""
        return sum(1 for v in self._cache.values() if v is None)

    @property
    def cache_total(self) -> int:
        """Total number of SMILES entries in cache."""
        return len(self._cache)


class SqliteCachedUniMolEncoder:
    """Disk-backed atom embedding cache for memory-constrained machines.

    The interface mirrors :class:`CachedUniMolEncoder`, but each molecular
    array is fetched from SQLite on demand.  This keeps the controlled 3D
    study representation identical while avoiding an 18 GB in-memory dict.
    """

    def __init__(
        self,
        cache_path: str,
        fallback_device: str = "cpu",
        log_misses: bool = True,
        strict: bool = False,
    ):
        del fallback_device  # retained for interface compatibility
        uri = f"file:{cache_path}?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True)
        repair_path = f"{cache_path}.repair.sqlite"
        self._repair_connection = None
        if os.path.exists(repair_path):
            repair_uri = f"file:{repair_path}?mode=ro"
            self._repair_connection = sqlite3.connect(repair_uri, uri=True)
        self.log_misses = log_misses
        self.strict = strict
        self.total_fragments = 0
        self.hit_fragments = 0
        self.skipped_fragments = 0

        complete = self._connection.execute(
            "SELECT value FROM metadata WHERE key='complete'"
        ).fetchone()
        if complete is None or complete[0] != "1":
            raise RuntimeError(f"Incomplete SQLite embedding cache: {cache_path}")
        counts = self._connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN data IS NULL THEN 1 ELSE 0 END), 0) "
            "FROM embeddings"
        ).fetchone()
        self._cache_total = int(counts[0])
        self._cache_failed = int(counts[1])
        print(
            f"[SqliteCachedUniMolEncoder] Cache opened: {self.cache_size} valid "
            f"/ {self.cache_failed} failed / {self.cache_total} total"
        )

    @lru_cache(maxsize=2048)
    def _fetch(self, canonical_smiles: str) -> Optional[np.ndarray]:
        row = self._connection.execute(
            "SELECT n_rows, n_cols, data FROM embeddings WHERE smiles=?",
            (canonical_smiles,),
        ).fetchone()
        if row is None and self._repair_connection is not None:
            row = self._repair_connection.execute(
                "SELECT n_rows, n_cols, data FROM embeddings WHERE smiles=?",
                (canonical_smiles,),
            ).fetchone()
        if row is None or row[2] is None:
            return None
        array = np.frombuffer(row[2], dtype=np.float32).reshape(
            int(row[0]), int(row[1])
        )
        # Detach from SQLite's transient bytes object before caching.
        return array.copy()

    def encode_atoms(self, smiles: str) -> np.ndarray:
        if not smiles:
            return np.zeros((1, EMBED_DIM), dtype=np.float32)
        atom_embeddings = []
        for fragment in [part.strip() for part in smiles.split(".") if part.strip()]:
            canonical = canon(fragment)
            if canonical is None:
                continue
            self.total_fragments += 1
            embedding = self._fetch(canonical)
            if embedding is None:
                self.skipped_fragments += 1
                if self.log_misses:
                    import logging
                    logging.getLogger(__name__).warning(
                        "[SQLITE CACHE MISS] '%s' (canonical: '%s')", fragment, canonical
                    )
                if self.strict:
                    raise RuntimeError(f"[SQLITE CACHE MISS] '{canonical}'")
                continue
            self.hit_fragments += 1
            atom_embeddings.append(embedding)
        if not atom_embeddings:
            return np.zeros((1, EMBED_DIM), dtype=np.float32)
        return np.concatenate(atom_embeddings, axis=0)

    encode_atoms_fragments = encode_atoms

    def encode_atoms_fragments_batch(self, smiles_list: List[str]) -> List[np.ndarray]:
        return [self.encode_atoms(smiles) for smiles in smiles_list]

    def encode_smiles(self, smiles_list: List[str]) -> np.ndarray:
        return np.stack(
            [self.encode_atoms(smiles).mean(axis=0) for smiles in smiles_list], axis=0
        ).astype(np.float32)

    def encode_single(self, smiles: str) -> np.ndarray:
        return self.encode_smiles([smiles])[0]

    def get_coverage_metrics(self) -> dict:
        return {
            "total_fragments": self.total_fragments,
            "hit_fragments": self.hit_fragments,
            "skipped_fragments": self.skipped_fragments,
            "coverage_ratio": self.hit_fragments / max(self.total_fragments, 1),
        }

    @property
    def cache_size(self) -> int:
        return self._cache_total - self._cache_failed

    @property
    def cache_failed(self) -> int:
        return self._cache_failed

    @property
    def cache_total(self) -> int:
        return self._cache_total
