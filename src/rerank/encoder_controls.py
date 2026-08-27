"""Fail-closed atom encoders for the prespecified WS-C controls.

The adapters expose only query-scoped atom states.  They intentionally do not
own a global atom cache: a caller encodes one product and batches of candidates,
computes the three frozen pair scalars, and discards the atom matrices.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import numpy as np


CONTROL_FEATURE_NAMES = (
    "atom_set_similarity",
    "reaction_distance",
    "cosine_reaction_vec",
)
FULL_FEATURE_NAMES = (
    "prior_or_log_prob",
    "morgan_similarity",
    "atom_set_similarity",
    "reaction_distance",
    "cosine_reaction_vec",
    "n_fragments",
    "heavy_atom_ratio",
)

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
MORGAN_USE_CHIRALITY = False

GROVER_REPOSITORY_COMMIT = "40b6d97098e4508687912f3c05eca369fc2c6213"
GROVER_CHECKPOINT_BASENAME = "grover_base.pt"
GROVER_CHECKPOINT_SIZE = 193_589_423
GROVER_CHECKPOINT_SHA256 = (
    "47e095880d71baf29ea6f6253473cd56d5406213fa82959c6e14ea469e06b1de"
)


class EncoderControlError(RuntimeError):
    """Raised when a control cannot preserve its prespecified identity."""


class AtomEncoder(Protocol):
    def initialize(self) -> None: ...

    def encode_fragments_batch(self, smiles: Sequence[str]) -> list[np.ndarray]: ...

    def metadata(self) -> dict: ...


class GroverAtomBackend(Protocol):
    """External backend contract required from the pinned GROVER checkout."""

    def encode_atom_states(
        self, smiles_batch: Sequence[str], device: str
    ) -> Sequence[Mapping[str, Any]]: ...


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": file_sha256(resolved),
    }


def _split_fragments(smiles: str) -> list[str]:
    fragments = [fragment.strip() for fragment in str(smiles).split(".")]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        raise EncoderControlError("SMILES contains no molecular fragment.")
    return fragments


def _atom_set_similarity(product: np.ndarray, reactant: np.ndarray) -> float:
    if product.size == 0 or reactant.size == 0:
        return 0.0
    product_norm = product / (
        np.linalg.norm(product, axis=1, keepdims=True) + 1e-10
    )
    reactant_norm = reactant / (
        np.linalg.norm(reactant, axis=1, keepdims=True) + 1e-10
    )
    similarities = product_norm @ reactant_norm.T
    product_to_reactant = similarities.max(axis=1).mean()
    reactant_to_product = similarities.max(axis=0).mean()
    return float(0.5 * (product_to_reactant + reactant_to_product))


def _reaction_distance(product: np.ndarray, reactant: np.ndarray) -> float:
    if product.shape[0] == 0 or reactant.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(product.mean(axis=0) - reactant.mean(axis=0)))


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-10 or second_norm <= 1e-10:
        return 0.0
    return float(np.dot(first, second) / (first_norm * second_norm))


def compute_control_scalars(
    product_atom_states: np.ndarray,
    reactant_atom_states: np.ndarray,
) -> np.ndarray:
    """Compute the frozen f5/f6/f8 formulas without retaining atom states."""
    product = np.asarray(product_atom_states, dtype=np.float32)
    reactant = np.asarray(reactant_atom_states, dtype=np.float32)
    if product.ndim != 2 or reactant.ndim != 2:
        raise EncoderControlError("Atom representations must be two-dimensional.")
    if product.shape[1] != reactant.shape[1]:
        raise EncoderControlError("Product/reactant representation widths differ.")
    if product.shape[0] == 0 or reactant.shape[0] == 0:
        raise EncoderControlError("Empty atom representations are not permitted.")
    product_mean = product.mean(axis=0)
    reactant_mean = reactant.mean(axis=0)
    reaction_vector = product_mean - reactant_mean
    return np.asarray(
        [
            _atom_set_similarity(product, reactant),
            _reaction_distance(product, reactant),
            _cosine(product_mean, reaction_vector),
        ],
        dtype=np.float32,
    )


def compute_query_control_features(
    encoder: AtomEncoder,
    product_smiles: str,
    candidate_smiles: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    """Stream one query through an encoder and retain only three scalars/pair."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    candidates = list(candidate_smiles)
    product_states = encoder.encode_fragments_batch([product_smiles])[0]
    rows: list[np.ndarray] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        candidate_states = encoder.encode_fragments_batch(batch)
        if len(candidate_states) != len(batch):
            raise EncoderControlError("Encoder output is not batch-aligned.")
        rows.extend(
            compute_control_scalars(product_states, states)
            for states in candidate_states
        )
        del candidate_states
    del product_states
    if not rows:
        return np.empty((0, len(CONTROL_FEATURE_NAMES)), dtype=np.float32)
    return np.stack(rows).astype(np.float32, copy=False)


def compute_full_query_features(
    encoder: AtomEncoder,
    product_smiles: str,
    candidate_smiles: Sequence[str],
    priors: Sequence[float],
    batch_size: int,
) -> np.ndarray:
    """Return the exact frozen seven-column ``3d+prior`` feature matrix."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    candidates = list(candidate_smiles)
    prior_values = np.asarray(priors, dtype=np.float32)
    if prior_values.ndim != 1 or len(prior_values) != len(candidates):
        raise EncoderControlError("Priors are not aligned to candidate rows.")
    if not np.isfinite(prior_values).all():
        raise EncoderControlError("Candidate priors must be finite.")
    controls = compute_query_control_features(
        encoder, product_smiles, candidates, batch_size
    )
    product_molecule = Chem.MolFromSmiles(product_smiles)
    if product_molecule is None or product_molecule.GetNumHeavyAtoms() == 0:
        raise EncoderControlError("Product is not a valid non-empty molecule.")
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    product_fingerprint = generator.GetFingerprint(product_molecule)
    product_heavy_atoms = product_molecule.GetNumHeavyAtoms()
    rows = np.empty((len(candidates), len(FULL_FEATURE_NAMES)), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        candidate_molecule = Chem.MolFromSmiles(candidate)
        if candidate_molecule is None:
            raise EncoderControlError(f"Invalid candidate molecule: {candidate!r}")
        candidate_fingerprint = generator.GetFingerprint(candidate_molecule)
        fragments = _split_fragments(candidate)
        candidate_heavy_atoms = 0
        for fragment in fragments:
            fragment_molecule = Chem.MolFromSmiles(fragment)
            if fragment_molecule is None:
                raise EncoderControlError(f"Invalid candidate fragment: {fragment!r}")
            candidate_heavy_atoms += fragment_molecule.GetNumHeavyAtoms()
        rows[index] = np.asarray(
            [
                prior_values[index],
                float(
                    DataStructs.TanimotoSimilarity(
                        product_fingerprint, candidate_fingerprint
                    )
                ),
                controls[index, 0],
                controls[index, 1],
                controls[index, 2],
                float(len(fragments)),
                float(candidate_heavy_atoms / product_heavy_atoms),
            ],
            dtype=np.float32,
        )
    if not np.isfinite(rows).all():
        raise EncoderControlError("Seven-column feature computation produced non-finite data.")
    return rows


class MorganAtomEncoder:
    """Deterministic radius-2, 2,048-bit atom-centred Morgan control."""

    def __init__(self) -> None:
        self._initialized = False
        self._metadata: dict = {}

    def initialize(self) -> None:
        if self._initialized:
            return
        from rdkit import rdBase

        try:
            distribution_version = importlib.metadata.version("rdkit")
        except importlib.metadata.PackageNotFoundError:
            distribution_version = None
        self._metadata = {
            "name": "MorganAtomEncoder",
            "device": "cpu",
            "rdkit_distribution_version": distribution_version,
            "rdkit_runtime_version": rdBase.rdkitVersion,
            "atom_vector": {
                "algorithm": "Morgan bit vector",
                "radius": MORGAN_RADIUS,
                "n_bits": MORGAN_N_BITS,
                "from_atoms": "[atom_index]",
                "use_chirality": MORGAN_USE_CHIRALITY,
                "dtype": "float32",
            },
            "fragment_handling": "encode_each_dot_fragment_then_concatenate",
            "global_atom_cache": False,
        }
        self._initialized = True

    @staticmethod
    def _encode_fragment(fragment: str) -> np.ndarray:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise EncoderControlError(f"Invalid molecular fragment: {fragment!r}")
        rows: list[np.ndarray] = []
        for atom_index in range(molecule.GetNumAtoms()):
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule,
                MORGAN_RADIUS,
                nBits=MORGAN_N_BITS,
                fromAtoms=[atom_index],
                useChirality=MORGAN_USE_CHIRALITY,
            )
            row = np.zeros(MORGAN_N_BITS, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fingerprint, row)
            rows.append(row)
        return np.stack(rows).astype(np.float32, copy=False)

    def encode_fragments_batch(self, smiles: Sequence[str]) -> list[np.ndarray]:
        if not self._initialized:
            raise EncoderControlError("MorganAtomEncoder must be initialized first.")
        encoded: list[np.ndarray] = []
        for value in smiles:
            fragments = _split_fragments(value)
            encoded.append(
                np.concatenate(
                    [self._encode_fragment(fragment) for fragment in fragments], axis=0
                ).astype(np.float32, copy=False)
            )
        return encoded

    def metadata(self) -> dict:
        if not self._initialized:
            raise EncoderControlError("Morgan metadata is unavailable before initialize().")
        return dict(self._metadata)


def _default_git_head(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_factory(specification: str, repo_path: Path) -> Callable[..., Any]:
    if ":" not in specification:
        raise EncoderControlError(
            "GROVER backend factory must use the form 'module:function'."
        )
    module_name, function_name = specification.split(":", 1)
    sys.path.insert(0, str(repo_path))
    try:
        module = importlib.import_module(module_name)
    finally:
        if sys.path and sys.path[0] == str(repo_path):
            sys.path.pop(0)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise EncoderControlError(f"GROVER backend factory is not callable: {specification}")
    return factory


@dataclass(frozen=True)
class GroverAssetExpectations:
    repository_commit: str = GROVER_REPOSITORY_COMMIT
    checkpoint_basename: str = GROVER_CHECKPOINT_BASENAME
    checkpoint_size: int = GROVER_CHECKPOINT_SIZE
    checkpoint_sha256: str = GROVER_CHECKPOINT_SHA256


class GroverAtomEncoder:
    """Pinned GROVER adapter with an explicit RDKit-atom alignment contract.

    The external backend must return, for every input molecule, a mapping with
    ``atom_representations`` and ``rdkit_atom_indices``.  The latter maps each
    returned row to the corresponding atom index of RDKit's input-molecule
    order.  Missing, duplicate, or out-of-range mappings fail closed.
    """

    def __init__(
        self,
        repo_path: Optional[str | Path] = None,
        checkpoint_path: Optional[str | Path] = None,
        device: str = "auto",
        batch_size: int = 32,
        atom_state_choice: Optional[str] = None,
        backend_factory: Optional[Callable[..., GroverAtomBackend]] = None,
        backend_factory_spec: Optional[str] = None,
        expectations: GroverAssetExpectations = GroverAssetExpectations(),
        git_head_reader: Callable[[Path], str] = _default_git_head,
    ) -> None:
        self.repo_path = Path(
            repo_path or os.environ.get("GROVER_REPO", "")
        ).expanduser()
        self.checkpoint_path = Path(
            checkpoint_path or os.environ.get("GROVER_CHECKPOINT", "")
        ).expanduser()
        self.requested_device = device
        self.batch_size = batch_size
        self.atom_state_choice = atom_state_choice
        self.backend_factory = backend_factory
        self.backend_factory_spec = backend_factory_spec or os.environ.get(
            "GROVER_ATOM_BACKEND_FACTORY"
        )
        self.expectations = expectations
        self.git_head_reader = git_head_reader
        self.device: Optional[str] = None
        self._backend: Optional[GroverAtomBackend] = None
        self._metadata: dict = {}

    def _resolve_device(self) -> str:
        if self.requested_device in {"cpu", "cuda"}:
            selected = self.requested_device
        elif self.requested_device == "auto":
            try:
                import torch

                selected = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                selected = "cpu"
        else:
            raise EncoderControlError("device must be auto, cpu, or cuda")
        if selected == "cuda":
            try:
                import torch
            except ImportError as exc:
                raise EncoderControlError("CUDA requested but PyTorch is unavailable.") from exc
            if not torch.cuda.is_available():
                raise EncoderControlError("CUDA requested but no CUDA device is available.")
        return selected

    def initialize(self) -> None:
        if self._backend is not None:
            return
        if not str(self.repo_path) or str(self.repo_path) == ".":
            raise EncoderControlError("GROVER_REPO/repo_path is required.")
        if not str(self.checkpoint_path) or str(self.checkpoint_path) == ".":
            raise EncoderControlError("GROVER_CHECKPOINT/checkpoint_path is required.")
        repo = self.repo_path.resolve()
        checkpoint = self.checkpoint_path.resolve()
        if not repo.is_dir():
            raise EncoderControlError(f"Pinned GROVER repository is missing: {repo}")
        if not checkpoint.is_file():
            raise EncoderControlError(f"Pinned GROVER checkpoint is missing: {checkpoint}")
        if checkpoint.name != self.expectations.checkpoint_basename:
            raise EncoderControlError(
                f"GROVER checkpoint must be named {self.expectations.checkpoint_basename}."
            )
        checkpoint_fingerprint = file_fingerprint(checkpoint)
        if checkpoint_fingerprint["size_bytes"] != self.expectations.checkpoint_size:
            raise EncoderControlError("GROVER checkpoint size does not match the pin.")
        if checkpoint_fingerprint["sha256"] != self.expectations.checkpoint_sha256:
            raise EncoderControlError("GROVER checkpoint SHA-256 does not match the pin.")
        try:
            repository_commit = self.git_head_reader(repo)
        except Exception as exc:
            raise EncoderControlError("Could not inspect the GROVER repository commit.") from exc
        if repository_commit != self.expectations.repository_commit:
            raise EncoderControlError(
                "GROVER repository HEAD does not match the prespecified commit."
            )
        from rerank.grover_official_backend import ATOM_STATE_CHOICES

        if self.atom_state_choice not in ATOM_STATE_CHOICES:
            raise EncoderControlError(
                "GROVER requires an explicit atom_state_choice from "
                f"{ATOM_STATE_CHOICES}; the signed plan does not approve a default."
            )
        factory = self.backend_factory
        if factory is None:
            if self.backend_factory_spec:
                factory = _load_factory(self.backend_factory_spec, repo)
            else:
                from rerank.grover_official_backend import OfficialGroverAtomBackend

                factory = OfficialGroverAtomBackend
        self.device = self._resolve_device()
        self._backend = factory(
            repo_path=repo,
            checkpoint_path=checkpoint,
            device=self.device,
            atom_state_choice=self.atom_state_choice,
        )
        if not hasattr(self._backend, "encode_atom_states"):
            raise EncoderControlError("GROVER backend lacks encode_atom_states().")
        self._metadata = {
            "name": "GroverAtomEncoder",
            "device": self.device,
            "repository": {
                "path": str(repo),
                "commit": repository_commit,
            },
            "checkpoint": checkpoint_fingerprint,
            "atom_state_choice": self.atom_state_choice,
            "atom_state_choice_required_by_plan": True,
            "official_fingerprint_source_note": (
                "fingerprint_source=atom concatenates atom_from_atom and "
                "atom_from_bond after pooling; rationale only, not approval"
            ),
            "atom_alignment": (
                "backend row -> explicit rdkit_atom_indices; reordered to RDKit "
                "input-molecule atom order; exact permutation required"
            ),
            "fragment_handling": "encode_each_dot_fragment_then_concatenate",
            "global_atom_cache": False,
        }
        backend_metadata = getattr(self._backend, "metadata", None)
        if callable(backend_metadata):
            self._metadata["official_backend"] = backend_metadata()

    @staticmethod
    def _align(fragment: str, output: Mapping[str, Any]) -> np.ndarray:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise EncoderControlError(f"Invalid molecular fragment: {fragment!r}")
        if not isinstance(output, Mapping):
            raise EncoderControlError("GROVER backend output must be a mapping.")
        if "atom_representations" not in output or "rdkit_atom_indices" not in output:
            raise EncoderControlError(
                "GROVER output lacks explicit atom representations/indices."
            )
        states = np.asarray(output["atom_representations"], dtype=np.float32)
        indices = [int(value) for value in output["rdkit_atom_indices"]]
        expected_count = molecule.GetNumAtoms()
        if states.ndim != 2 or states.shape[0] != len(indices):
            raise EncoderControlError("GROVER states and atom-index map are misaligned.")
        if sorted(indices) != list(range(expected_count)):
            raise EncoderControlError(
                "GROVER atom indices are not an exact RDKit-atom permutation."
            )
        aligned = np.empty((expected_count, states.shape[1]), dtype=np.float32)
        for source_row, rdkit_atom_index in enumerate(indices):
            aligned[rdkit_atom_index] = states[source_row]
        if not np.isfinite(aligned).all():
            raise EncoderControlError("GROVER returned non-finite atom states.")
        return aligned

    def encode_fragments_batch(self, smiles: Sequence[str]) -> list[np.ndarray]:
        if self._backend is None or self.device is None:
            raise EncoderControlError("GroverAtomEncoder must be initialized first.")
        parsed = [_split_fragments(value) for value in smiles]
        unique: list[str] = []
        seen: set[str] = set()
        for fragments in parsed:
            for fragment in fragments:
                if fragment not in seen:
                    seen.add(fragment)
                    unique.append(fragment)
        fragment_states: dict[str, np.ndarray] = {}
        for start in range(0, len(unique), self.batch_size):
            batch = unique[start : start + self.batch_size]
            outputs = self._backend.encode_atom_states(batch, device=self.device)
            if len(outputs) != len(batch):
                raise EncoderControlError("GROVER backend output is not batch-aligned.")
            for fragment, output in zip(batch, outputs):
                fragment_states[fragment] = self._align(fragment, output)
        results = [
            np.concatenate([fragment_states[fragment] for fragment in fragments], axis=0)
            .astype(np.float32, copy=False)
            for fragments in parsed
        ]
        del fragment_states
        return results

    def metadata(self) -> dict:
        if self._backend is None:
            raise EncoderControlError("GROVER metadata is unavailable before initialize().")
        return dict(self._metadata)
