"""Pinned rxn-ebm FF-EBM primitives for the D5 external comparison.

Only the public model and fingerprint implementation from the pinned upstream
repository are imported.  Candidate generation, split construction, test
locking, checkpoint selection, and result manifests remain controlled by this
repository.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset

from rerank.study_data import canonicalize_reactant_set, canonicalize_smiles


PROTOCOL_ID = "D-EXTERNAL-RXN-EBM-FF-CAP10-v1"
COMPARATOR = "frozen cap-10 candidate-prior ordering"
SINGLE_INTENDED_CHANGE = "published rxn-ebm feedforward energy reranker"

RXN_EBM_COMMIT = "1919eeccdd31e16ec7a44478b756bcd974c35a3c"
RETRORANKER_COMMIT = "22b5765743f3325b0e3b51b0cb9cf2d908081399"
PUBLISHED_EBM_SEEDS = (0, 20210423, 77777777)

CANDIDATE_WIDTH = 10
FP_RADIUS = 3
FP_SIZE = 16_384
FP_COMPONENTS = 3
REACTION_FP_DIM = FP_SIZE * FP_COMPONENTS
RXN_TYPE = "hybrid_all"

PUBLISHED_FF_SETTINGS = {
    "model_name": "FeedforwardEBM",
    "encoder_hidden_size": [1024, 128],
    "encoder_dropout": 0.2,
    "encoder_activation": "PReLU",
    "out_hidden_sizes": [128],
    "out_activation": "PReLU",
    "out_dropout": 0.2,
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 96,
    "batch_size_eval": 96,
    "epochs": 40,
    "lr_scheduler": "ReduceLROnPlateau",
    "lr_scheduler_criteria": "top1_validation_accuracy",
    "lr_scheduler_factor": 0.2,
    "lr_scheduler_patience": 0,
    "lr_floor": 8e-7,
    "early_stop_criteria": "top1_validation_accuracy",
    "early_stop_patience": 2,
    "gradient_clip": 0.0,
    "rxn_type": RXN_TYPE,
    "fingerprint_radius": FP_RADIUS,
    "fingerprint_size_per_component": FP_SIZE,
    "candidate_width": CANDIDATE_WIDTH,
}

UPSTREAM_CORE_FILES = (
    "rxnebm/model/FF.py",
    "rxnebm/model/model_utils.py",
    "rxnebm/data/fp_utils.py",
    "scripts/retrosim/FeedforwardEBM.sh",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def canonical_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, target)


def immutable_target(path: str | Path, label: str) -> Path:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    existing = [item for item in (target, temporary) if item.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing or partial {label}: "
            + ", ".join(str(item.resolve()) for item in existing)
        )
    return target


def verify_pinned_repository(
    repository: str | Path,
    expected_commit: str,
    core_files: Sequence[str] = (),
) -> dict:
    root = Path(repository).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != expected_commit:
        raise RuntimeError(
            f"Pinned repository mismatch at {root}: expected {expected_commit}, got {head}."
        )
    fingerprints = {}
    for relative in core_files:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Pinned upstream file is absent: {path}")
        changed = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", expected_commit, "--", relative],
            check=False,
        ).returncode
        if changed != 0:
            raise RuntimeError(f"Pinned upstream core file is modified: {relative}")
        fingerprints[relative] = file_fingerprint(path)
    return {
        "path": str(root),
        "commit": head,
        "core_files": fingerprints,
    }


def import_pinned_rxn_ebm(repository: str | Path):
    root = str(Path(repository).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    # The pinned fp_utils module imports nmslib but never references it.  NMSLIB
    # is only needed by unrelated nearest-neighbour proposal code and has no
    # maintained wheel for the revision runtime.  A deliberately empty module
    # keeps that unused import from broadening the scientific environment.
    if "nmslib" not in sys.modules and importlib.util.find_spec("nmslib") is None:
        shim = types.ModuleType("nmslib")
        shim.__spec__ = importlib.machinery.ModuleSpec("nmslib", loader=None)
        sys.modules["nmslib"] = shim
    fp_utils = importlib.import_module("rxnebm.data.fp_utils")
    # RDKit renamed this keyword from useCountSimulation to countSimulation.
    # Keep the upstream call site byte-identical and translate only the API
    # spelling; the boolean value and fingerprint algorithm are unchanged.
    if not getattr(fp_utils, "_revision_rdkit_keyword_compat", False):
        original_generator = fp_utils.GetMorganGenerator

        def compatible_morgan_generator(*args, **kwargs):
            if "useCountSimulation" in kwargs:
                if "countSimulation" in kwargs:
                    raise TypeError("Both count-simulation keyword spellings were supplied.")
                kwargs["countSimulation"] = kwargs.pop("useCountSimulation")
            return original_generator(*args, **kwargs)

        fp_utils.GetMorganGenerator = compatible_morgan_generator
        fp_utils._revision_rdkit_keyword_compat = True
    ff_module = importlib.import_module("rxnebm.model.FF")
    model_utils = importlib.import_module("rxnebm.model.model_utils")
    return fp_utils, ff_module, model_utils


def model_args() -> SimpleNamespace:
    return SimpleNamespace(
        rxn_type=RXN_TYPE,
        rctfp_size=FP_SIZE,
        prodfp_size=FP_SIZE,
        difffp_size=FP_SIZE,
        encoder_hidden_size=list(PUBLISHED_FF_SETTINGS["encoder_hidden_size"]),
        encoder_dropout=PUBLISHED_FF_SETTINGS["encoder_dropout"],
        encoder_activation=PUBLISHED_FF_SETTINGS["encoder_activation"],
        out_hidden_sizes=list(PUBLISHED_FF_SETTINGS["out_hidden_sizes"]),
        out_dropout=PUBLISHED_FF_SETTINGS["out_dropout"],
        out_activation=PUBLISHED_FF_SETTINGS["out_activation"],
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class QueryRow:
    product_smiles: str
    candidate_smiles: tuple[str, ...]
    candidate_keys: tuple[str, ...]
    true_index: int
    reaction_id: int | None
    product_key: str

    def identity(self) -> dict:
        return {
            "product_key": self.product_key,
            "candidate_keys": list(self.candidate_keys),
            "true_index": self.true_index,
            "reaction_id": self.reaction_id,
        }


def training_query_rows(train_products: Sequence[Mapping]) -> list[QueryRow]:
    """Expand the 45 multi-positive products without creating false negatives."""

    result: list[QueryRow] = []
    for product in train_products:
        candidates = product["candidates"]
        positives = tuple(int(index) for index in product["positive_indices"])
        positive_set = set(positives)
        if not positives or not product.get("negative_indices"):
            raise ValueError("Prepared training product lacks a positive or negative candidate.")
        negatives = [index for index in range(len(candidates)) if index not in positive_set]
        if negatives != [int(index) for index in product["negative_indices"]]:
            raise ValueError("Training candidate labels do not match the frozen pool order.")
        product_key = str(product["product_key"])
        if canonicalize_smiles(str(product["product_smiles"])) != product_key:
            raise ValueError("Training product identity differs from its frozen canonical key.")
        for positive in positives:
            order = [positive, *negatives]
            smiles = tuple(str(candidates[index]["smiles"]) for index in order)
            keys = tuple(
                str(candidates[index].get("canonical_smiles") or "") for index in order
            )
            if any(canonicalize_reactant_set(smi) != key for smi, key in zip(smiles, keys)):
                raise ValueError("Training candidate identity changed during D5 adaptation.")
            result.append(
                QueryRow(
                    product_smiles=str(product["product_smiles"]),
                    candidate_smiles=smiles,
                    candidate_keys=keys,
                    true_index=0,
                    reaction_id=None,
                    product_key=product_key,
                )
            )
    return result


def evaluation_query_rows(payload: Mapping) -> list[QueryRow]:
    required = {"eval_pwc", "eval_ground_truths", "eval_metadata"}
    if not required.issubset(payload):
        raise ValueError("Evaluation payload is incomplete.")
    lengths = {len(payload[key]) for key in required}
    if len(lengths) != 1:
        raise ValueError("Evaluation payload arrays are misaligned.")
    rows: list[QueryRow] = []
    for (product, candidates), truth, metadata in zip(
        payload["eval_pwc"], payload["eval_ground_truths"], payload["eval_metadata"]
    ):
        product_key = canonicalize_smiles(str(product))
        truth_key = canonicalize_reactant_set(str(truth))
        candidate_smiles = tuple(str(item["smiles"]) for item in candidates)
        candidate_keys = tuple(
            canonicalize_reactant_set(smiles) or "" for smiles in candidate_smiles
        )
        matches = [index for index, key in enumerate(candidate_keys) if key == truth_key]
        if product_key is None or truth_key is None or len(matches) != 1:
            raise ValueError("Evaluation row lacks exactly one canonical ground-truth match.")
        if not 1 <= len(candidate_smiles) <= CANDIDATE_WIDTH:
            raise ValueError("D5 requires one to ten frozen candidates per evaluation row.")
        if int(metadata.get("coverage_rank", matches[0] + 1)) != matches[0] + 1:
            raise ValueError("Evaluation coverage rank differs from canonical candidate order.")
        rows.append(
            QueryRow(
                product_smiles=str(product),
                candidate_smiles=candidate_smiles,
                candidate_keys=candidate_keys,
                true_index=matches[0],
                reaction_id=int(metadata["reaction_id"]),
                product_key=product_key,
            )
        )
    return rows


def adapter_audit(train_rows: Sequence[QueryRow], valid_rows: Sequence[QueryRow]) -> dict:
    return {
        "train_rows": len(train_rows),
        "train_unique_products": len({row.product_key for row in train_rows}),
        "train_multi_positive_extra_rows": len(train_rows)
        - len({row.product_key for row in train_rows}),
        "validation_rows": len(valid_rows),
        "validation_unique_products": len({row.product_key for row in valid_rows}),
        "max_candidate_count_train": max(len(row.candidate_smiles) for row in train_rows),
        "max_candidate_count_validation": max(
            len(row.candidate_smiles) for row in valid_rows
        ),
        "train_identity_fingerprint": canonical_fingerprint(
            [row.identity() for row in train_rows]
        ),
        "validation_identity_fingerprint": canonical_fingerprint(
            [row.identity() for row in valid_rows]
        ),
        "multi_positive_policy": (
            "one positive-first EBM row per official reference; all other official "
            "references for that product are excluded from negatives"
        ),
        "evaluation_order_policy": "frozen candidate-prior order, unchanged",
    }


_WORKER_FP_UTILS = None


def _fingerprint_worker_init(repository: str) -> None:
    global _WORKER_FP_UTILS
    _WORKER_FP_UTILS, _, _ = import_pinned_rxn_ebm(repository)


def _fingerprint_query_worker(task: tuple[str, tuple[str, ...]]) -> sparse.csr_matrix:
    if _WORKER_FP_UTILS is None:
        raise RuntimeError("Fingerprint worker was not initialized.")
    product, candidates = task
    reaction_fps = []
    for candidate in candidates:
        reactant_fp, product_fp = _WORKER_FP_UTILS.rcts_prod_fps_from_rxn_smi_dist(
            f"{candidate}>>{product}", FP_RADIUS, FP_SIZE
        )
        reaction_fps.append(
            _WORKER_FP_UTILS.make_rxn_fp(
                reactant_fp, product_fp, RXN_TYPE
            ).tocsr()
        )
    missing = CANDIDATE_WIDTH - len(reaction_fps)
    if missing < 0:
        raise ValueError("Candidate count exceeds the frozen cap-10 width.")
    if missing:
        reaction_fps.extend(
            sparse.csr_matrix((1, REACTION_FP_DIM), dtype=np.int32)
            for _ in range(missing)
        )
    return sparse.hstack(reaction_fps, format="csr", dtype=np.int32)


def build_fingerprint_matrix(
    rows: Sequence[QueryRow],
    repository: str | Path,
    workers: int = 1,
    progress=None,
) -> sparse.csr_matrix:
    tasks = [(row.product_smiles, row.candidate_smiles) for row in rows]
    if workers < 1:
        raise ValueError("workers must be positive.")
    if workers == 1:
        _fingerprint_worker_init(str(Path(repository).resolve()))
        iterator: Iterable[sparse.csr_matrix] = map(_fingerprint_query_worker, tasks)
        matrices = list(progress(iterator, total=len(tasks)) if progress else iterator)
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_fingerprint_worker_init,
            initargs=(str(Path(repository).resolve()),),
        ) as pool:
            iterator = pool.map(_fingerprint_query_worker, tasks, chunksize=8)
            matrices = list(progress(iterator, total=len(tasks)) if progress else iterator)
    matrix = sparse.vstack(matrices, format="csr", dtype=np.int32)
    expected_shape = (len(rows), CANDIDATE_WIDTH * REACTION_FP_DIM)
    if matrix.shape != expected_shape:
        raise AssertionError(f"Fingerprint matrix shape {matrix.shape} != {expected_shape}.")
    return matrix


def atomic_sparse_save(matrix: sparse.csr_matrix, path: str | Path) -> None:
    target = immutable_target(path, "fingerprint matrix")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.npz")
    if temporary.exists():
        raise FileExistsError(f"Partial fingerprint matrix already exists: {temporary}")
    sparse.save_npz(temporary, matrix, compressed=True)
    os.replace(temporary, target)


class SparseQueryDataset(Dataset):
    def __init__(self, matrix: sparse.csr_matrix):
        if matrix.shape[1] != CANDIDATE_WIDTH * REACTION_FP_DIM:
            raise ValueError("D5 sparse matrix has the wrong fingerprint width.")
        self.matrix = matrix.tocsr()

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def __getitem__(self, index: int):
        dense = self.matrix[index].toarray().reshape(
            CANDIDATE_WIDTH, REACTION_FP_DIM
        )
        mask = np.any(dense != 0, axis=1)
        return torch.from_numpy(dense.astype(np.float32)), torch.from_numpy(mask)


def stable_ranks(energies: np.ndarray, candidate_counts: Sequence[int]) -> np.ndarray:
    values = np.asarray(energies, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != CANDIDATE_WIDTH:
        raise ValueError("Energy matrix must have shape N x 10.")
    ranks = np.full_like(values, fill_value=-1, dtype=np.int32)
    for row_index, count in enumerate(candidate_counts):
        count = int(count)
        if not 1 <= count <= CANDIDATE_WIDTH:
            raise ValueError("Candidate count must be between one and ten.")
        order = np.argsort(values[row_index, :count], kind="stable")
        ranks[row_index, order] = np.arange(1, count + 1, dtype=np.int32)
    return ranks


def ranking_metrics(
    energies: np.ndarray, true_indices: Sequence[int], candidate_counts: Sequence[int]
) -> dict:
    ranks = stable_ranks(energies, candidate_counts)
    true_ranks = np.asarray(
        [ranks[index, int(true)] for index, true in enumerate(true_indices)],
        dtype=np.int32,
    )
    if np.any(true_ranks < 1):
        raise AssertionError("A ground-truth index lies outside its candidate pool.")
    return {
        "top1": float(np.mean(true_ranks <= 1)),
        "top3": float(np.mean(true_ranks <= 3)),
        "top5": float(np.mean(true_ranks <= 5)),
        "mrr": float(np.mean(1.0 / true_ranks)),
        "true_ranks": true_ranks,
        "rank_matrix": ranks,
    }


def prior_baseline_energies(rows: Sequence[QueryRow]) -> np.ndarray:
    values = np.full((len(rows), CANDIDATE_WIDTH), np.inf, dtype=np.float64)
    for index, row in enumerate(rows):
        values[index, : len(row.candidate_smiles)] = np.arange(
            len(row.candidate_smiles), dtype=np.float64
        )
    return values
