"""
rerank.data.build_embedding
===========================
Batch-encode all unique SMILES from rerank_dataset.jsonl using UniMol
and save atom-level embeddings (n_atoms, 512) to a pickle file.

Two encoding paths:
  FAST  — SMILES has a pre-computed .xyz file  -> feed (atoms, coords)
           directly to UniMol, skipping 3D conformer generation.
  SLOW  — No .xyz available -> normal SMILES path (RDKit ETKDG + UniMol).

Features:
  - RDKit canonicalization + deduplication before encoding
  - True batch encoding via model.get_repr()
  - Resume: skips already-encoded SMILES if output file exists
  - Per-SMILES failure logging to failed_smiles.txt

Usage (Google Colab / local):
    # Full run — SMILES only:
    python -m rerank.data.build_embedding \\
        --jsonl   outputs/rerank_dataset.jsonl \\
        --output  outputs/atom_embeddings.pkl  \\
        --batch   64                           \\
        --device  cuda

    # With .xyz acceleration (products):
    python -m rerank.data.build_embedding \\
        --jsonl      outputs/rerank_dataset.jsonl \\
        --output     outputs/atom_embeddings.pkl  \\
        --batch      64                            \\
        --device     cuda                          \\
        --xyz_dir    data/xyz                      \\
        --xyz_csv    data/uspto_smiles.csv         \\
        --xyz_valid  data/valid_indices.json

Output:
    outputs/atom_embeddings.pkl  -- dict[canonical_smiles -> np.ndarray (n_atoms, 512)]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Aggressively silence ALL noisy third-party loggers
import warnings
warnings.filterwarnings("ignore")   # suppress Python-level warnings

# RDKit logs directly to stderr from C++ — must use its own API
try:
    from rdkit import RDLogger as _RDLogger
    _RDLogger.DisableLog("rdApp.*")   # silences valence errors, atom warnings, etc.
except Exception:
    pass

_SILENCE = [
    "unimol_tools",
    "unimol_tools.data.dictionary",
    "unimol_tools.data.conformer",
    "unimol_tools.tasks.trainer",
    "unimol_tools.models.unimol",
    "unimol_tools.weights.weighthub",
    "Uni-Mol Tools",
    "unicore",
    "httpcore", "httpx",
    "huggingface_hub", "filelock", "urllib3",
    "numexpr",
]
for _name in _SILENCE:
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.ERROR)
    _lg.propagate = False


EMBED_DIM = 512



# Step 1 — RDKit canonicalization helper


def _canonicalize(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canon = Chem.MolToSmiles(mol, canonical=True)
        return canon if canon else None
    except Exception:
        return None



# Step 2 — Collect unique canonical SMILES from JSONL


def collect_unique_smiles(jsonl_path: str) -> List[str]:
    """
    Stream JSONL, canonicalize every product + every reactant fragment,
    return sorted list of unique canonical SMILES.
    """
    seen: set[str] = set()
    n_raw   = 0
    n_frags = 0

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for raw in tqdm(fh, desc="Reading JSONL", unit="line"):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            n_raw += 1

            for field in ("product", "reactant"):
                value = rec.get(field, "").strip()
                if not value:
                    continue
                for frag in value.split("."):
                    frag = frag.strip()
                    if not frag:
                        continue
                    n_frags += 1
                    canon = _canonicalize(frag)
                    if canon:
                        seen.add(canon)

    unique = sorted(seen)
    log.info(
        "JSONL: %d records | %d raw fragments -> %d unique  (dedup %.1fx)",
        n_raw, n_frags, len(unique), n_frags / max(len(unique), 1),
    )
    return unique



# Step 3 — XYZ file loading (optional acceleration path)


def _parse_xyz(path: str) -> Optional[Tuple[List[str], np.ndarray]]:
    """
    Parse a standard .xyz file.
    Returns (atom_symbols, coords) where coords is (n_atoms, 3) float32.
    Returns None on any parse error.
    """
    try:
        with open(path, "r") as f:
            lines = f.read().splitlines()
        n_atoms = int(lines[0].strip())
        # line 1 is comment/SMILES — skip
        atoms, coords = [], []
        for line in lines[2: 2 + n_atoms]:
            parts = line.split()
            if len(parts) < 4:
                return None
            atoms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(atoms) != n_atoms:
            return None
        return atoms, np.array(coords, dtype=np.float32)
    except Exception:
        return None


def build_xyz_lookup(
    xyz_dir:       str,
    csv_path:      str,
    valid_json:    str,
) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """
    Build  canonical_smiles -> (atom_symbols, coords_ndarray)  from xyz files.

    Mapping:
        valid_indices.json[i]  ->  csv row index  ->  product SMILES
        xyz_dir/prod_{valid[i]}.xyz               ->  3D coordinates
    """
    import pandas as pd

    log.info("Loading xyz lookup from %s ...", xyz_dir)

    with open(valid_json) as f:
        valid_indices = json.load(f)

    df = pd.read_csv(csv_path)

    lookup: Dict[str, Tuple[List[str], np.ndarray]] = {}
    n_missing, n_parse_err, n_canon_err = 0, 0, 0

    for idx in tqdm(valid_indices, desc="Loading .xyz", unit="file"):
        xyz_path = os.path.join(xyz_dir, f"prod_{idx}.xyz")
        if not os.path.exists(xyz_path):
            n_missing += 1
            continue

        parsed = _parse_xyz(xyz_path)
        if parsed is None:
            n_parse_err += 1
            continue

        atoms, coords = parsed
        raw_smi = df.iloc[idx]["products_smiles"]
        canon   = _canonicalize(str(raw_smi))
        if canon is None:
            n_canon_err += 1
            continue

        lookup[canon] = (atoms, coords)

    log.info(
        "XYZ lookup: %d entries  (missing=%d  parse_err=%d  canon_err=%d)",
        len(lookup), n_missing, n_parse_err, n_canon_err,
    )
    return lookup



# Step 4 — Parse atomic representations from model output


def _parse_atomic_reprs(result, batch_len: int) -> Optional[List[np.ndarray]]:
    if isinstance(result, dict):
        raw = result.get("atomic_reprs") or result.get("atom_repr")
        if raw is None:
            return None
    else:
        try:
            arr = np.asarray(result, dtype=np.float32)
            if arr.ndim == 3:
                return [arr[i] for i in range(arr.shape[0])]
        except Exception:
            pass
        return None

    if not isinstance(raw, (list, tuple)) or len(raw) != batch_len:
        return None

    embs = []
    for item in raw:
        arr = np.asarray(item, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        embs.append(arr if (arr.ndim == 2 and arr.shape[1] == EMBED_DIM) else None)
    return embs



# Step 5 — Encoding functions


def encode_batch_smiles(
    model,
    batch: List[str],
) -> Dict[str, Optional[np.ndarray]]:
    """Standard SMILES -> conformer gen -> UniMol -> atom embeddings."""
    results: Dict[str, Optional[np.ndarray]] = {s: None for s in batch}

    # Attempt 1: true batch
    try:
        raw  = model.get_repr(batch, return_atomic_reprs=True)
        embs = _parse_atomic_reprs(raw, len(batch))
        if embs is not None:
            for smi, emb in zip(batch, embs):
                if emb is not None and np.any(emb != 0):
                    results[smi] = emb
            return results
    except Exception as exc:
        log.debug("Batch SMILES call failed (%s) — per-SMILES fallback.", exc)

    # Attempt 2: per-SMILES
    for smi in batch:
        try:
            raw  = model.get_repr([smi], return_atomic_reprs=True)
            embs = _parse_atomic_reprs(raw, 1)
            if embs and embs[0] is not None and np.any(embs[0] != 0):
                results[smi] = embs[0]
        except Exception as exc:
            log.debug("Single SMILES failed '%s': %s", smi, exc)

    return results


def encode_batch_xyz(
    model,
    batch_smiles:  List[str],
    batch_atoms:   List[List[str]],
    batch_coords:  List[np.ndarray],
) -> Dict[str, Optional[np.ndarray]]:
    """
    Feed pre-computed 3D coordinates directly to UniMol, bypassing
    conformer generation.  Falls back to SMILES path on API mismatch.
    """
    results: Dict[str, Optional[np.ndarray]] = {s: None for s in batch_smiles}

    # unimol_tools accepts {'atoms': ..., 'coordinates': ...} in recent versions
    try:
        data = {
            "atoms":       batch_atoms,
            "coordinates": batch_coords,
        }
        raw  = model.get_repr(data, return_atomic_reprs=True)
        embs = _parse_atomic_reprs(raw, len(batch_smiles))
        if embs is not None:
            for smi, emb in zip(batch_smiles, embs):
                if emb is not None and np.any(emb != 0):
                    results[smi] = emb
            return results
    except Exception as exc:
        log.debug("XYZ batch call failed (%s) — falling back to SMILES path.", exc)

    # Fallback: use SMILES (slower, regenerates conformer)
    return encode_batch_smiles(model, batch_smiles)



# Step 6 — Cache helpers


def load_cache(output_path: str) -> Dict[str, np.ndarray]:
    if os.path.exists(output_path):
        log.info("Resuming: loading existing cache from %s ...", output_path)
        with open(output_path, "rb") as f:
            cache = pickle.load(f)
        log.info("Resumed with %d already-encoded SMILES.", len(cache))
        return cache
    return {}


def _save(cache: Dict[str, np.ndarray], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)



# Step 7 — Main pipeline


def build_embeddings(
    jsonl_path:  str,
    output_path: str,
    batch_size:  int  = 64,
    device:      str  = "cpu",
    fail_log:    str  = "outputs/failed_smiles.txt",
    limit:       Optional[int] = None,
    xyz_dir:     Optional[str] = None,
    xyz_csv:     Optional[str] = None,
    xyz_valid:   Optional[str] = None,
) -> None:

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(fail_log)    or ".", exist_ok=True)

    # Collect unique SMILES
    unique_smiles = collect_unique_smiles(jsonl_path)
    total = len(unique_smiles)

    # Resume
    cache: Dict[str, np.ndarray] = load_cache(output_path)
    todo  = [s for s in unique_smiles if s not in cache]

    if limit is not None:
        todo = todo[:limit]
        log.info("--limit %d: processing only first %d SMILES.", limit, len(todo))

    log.info("To encode: %d / %d  (already done: %d)", len(todo), total, len(cache))

    if not todo:
        log.info("All SMILES already encoded. Nothing to do.")
        return

    # Optional: load xyz lookup
    xyz_lookup: Dict[str, Tuple[List[str], np.ndarray]] = {}
    if xyz_dir and xyz_csv and xyz_valid:
        xyz_lookup = build_xyz_lookup(xyz_dir, xyz_csv, xyz_valid)

    if xyz_lookup:
        todo_xyz  = [s for s in todo if s     in xyz_lookup]
        todo_smi  = [s for s in todo if s not in xyz_lookup]
        log.info(
            "Encoding plan: %d via xyz (fast)  +  %d via SMILES (conformer gen)",
            len(todo_xyz), len(todo_smi),
        )
    else:
        todo_xyz = []
        todo_smi = todo
        log.info("No xyz lookup — encoding all %d via SMILES path.", len(todo_smi))

    # Load UniMol model
    from rerank.encoder import UniMolEncoder
    encoder = UniMolEncoder(device=device)
    encoder._load_model()
    model = encoder._model

    if model is None:
        log.error("UniMol model failed to load. Aborting.")
        sys.exit(1)

    log.info("UniMol model ready on device='%s'.", device)

    # Encode
    failed:   List[str] = []
    n_done    = 0
    t_start   = time.time()
    save_every = 500

    def _process_results(res: Dict[str, Optional[np.ndarray]]) -> None:
        nonlocal n_done
        for smi, emb in res.items():
            if emb is not None:
                canon = _canonicalize(smi)
                if canon:
                    cache[canon] = emb
                    n_done += 1
            else:
                canon = _canonicalize(smi)
                if canon:
                    cache[canon] = None
                    failed.append(canon)
        total_todo = len(todo_xyz) + len(todo_smi)
        # stdout progress (1 line overwrite) — visible via Popen pipe
        print(
            f"\rEncoding: {n_done}/{total_todo} done | failed: {len(failed)}",
            end="", flush=True, file=sys.stdout
        )
        if n_done > 0 and n_done % save_every < batch_size:
            _save(cache, output_path)
            elapsed = time.time() - t_start
            speed   = n_done / elapsed
            eta = (total_todo - n_done) / speed / 60 if speed > 0 else 0
            # print newline at checkpoint to avoid overwriting the progress bar
            print(
                f"\nCheckpoint: {n_done}/{total_todo} | {speed:.1f} smi/s | ETA ~{eta:.0f} min",
                flush=True, file=sys.stdout
            )
            log.info(
                "Checkpoint: %d/%d encoded  |  %.1f smi/s  |  ETA ~%.0f min",
                n_done, total_todo, speed, eta,
            )

    # Path A: xyz-accelerated
    if todo_xyz:
        log.info("--- XYZ path: %d SMILES ---", len(todo_xyz))
        batches = [todo_xyz[i: i + batch_size] for i in range(0, len(todo_xyz), batch_size)]
        for batch in tqdm(batches, desc="Encoding (xyz)", unit="batch"):
            batch_atoms  = [xyz_lookup[s][0] for s in batch]
            batch_coords = [xyz_lookup[s][1] for s in batch]
            res = encode_batch_xyz(model, batch, batch_atoms, batch_coords)
            _process_results(res)

    # Path B: SMILES path
    if todo_smi:
        log.info("--- SMILES path: %d SMILES ---", len(todo_smi))
        batches = [todo_smi[i: i + batch_size] for i in range(0, len(todo_smi), batch_size)]
        for batch in tqdm(batches, desc="Encoding (smiles)", unit="batch"):
            res = encode_batch_smiles(model, batch)
            _process_results(res)

    # Final save
    _save(cache, output_path)
    print()  # newline after the final \r

    elapsed = time.time() - t_start
    speed   = max(n_done, 1) / elapsed

    # Print summary to stdout so it shows in Popen pipe
    print(
        f"✅ Done: {n_done} encoded | {len(failed)} failed | "
        f"{elapsed/60:.1f} min | {speed:.1f} smi/s | saved → {output_path}",
        flush=True, file=sys.stdout
    )
    log.info("=" * 60)
    log.info("Done.  Encoded: %d  |  Failed: %d  |  Total cache: %d",
             n_done, len(failed), len(cache))
    log.info("Elapsed: %.1f min  |  Speed: %.1f smi/s", elapsed / 60, speed)
    log.info("Output: %s", output_path)

    if failed:
        with open(fail_log, "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        log.warning("%d SMILES failed -- see %s", len(failed), fail_log)



# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute UniMol atom-level embeddings for retrosynthesis reranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--jsonl",    default="outputs/rerank_dataset.jsonl",
                   help="Path to rerank_dataset.jsonl")
    p.add_argument("--output",   default="outputs/atom_embeddings.pkl",
                   help="Output pickle path")
    p.add_argument("--batch",    type=int, default=64,
                   help="Batch size for UniMol model.get_repr()")
    p.add_argument("--device",   default="cpu",
                   help="'cuda' or 'cpu'")
    p.add_argument("--fail_log", default="outputs/failed_smiles.txt",
                   help="File to log failed SMILES")
    p.add_argument("--limit",    type=int, default=None,
                   help="Process only first N unique SMILES (testing)")

    # XYZ acceleration (optional)
    g = p.add_argument_group("XYZ acceleration (optional)")
    g.add_argument("--xyz_dir",   default=None,
                   help="Directory containing prod_0.xyz, prod_1.xyz, ...")
    g.add_argument("--xyz_csv",   default=None,
                   help="Path to uspto_smiles.csv (index -> product SMILES mapping)")
    g.add_argument("--xyz_valid", default=None,
                   help="Path to valid_indices.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_embeddings(
        jsonl_path  = args.jsonl,
        output_path = args.output,
        batch_size  = args.batch,
        device      = args.device,
        fail_log    = args.fail_log,
        limit       = args.limit,
        xyz_dir     = args.xyz_dir,
        xyz_csv     = args.xyz_csv,
        xyz_valid   = args.xyz_valid,
    )
