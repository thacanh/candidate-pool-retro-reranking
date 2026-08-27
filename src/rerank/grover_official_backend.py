"""Concrete atom-state backend for the pinned official GROVER checkout.

The official fingerprint wrapper uses ``fingerprint_source='atom'`` to pool a
concatenation of ``atom_from_atom`` and ``atom_from_bond``.  That implementation
detail motivates exposing ``concatenation`` but does not approve it as the
scientific WS-C atom state.  Callers must explicitly select one named source.
"""

from __future__ import annotations

import inspect
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ATOM_STATE_CHOICES = (
    "atom_from_atom",
    "atom_from_bond",
    "concatenation",
)


class OfficialGroverBackendError(RuntimeError):
    """Raised when the official GROVER atom-state contract is not met."""


class OfficialGroverAtomBackend:
    """Load ``GroverFpGeneration`` and expose exact ``a_scope`` atom slices."""

    def __init__(
        self,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        device: str,
        atom_state_choice: str,
    ) -> None:
        if atom_state_choice not in ATOM_STATE_CHOICES:
            raise OfficialGroverBackendError(
                "atom_state_choice must explicitly be one of "
                f"{ATOM_STATE_CHOICES}; no default is scientifically approved."
            )
        self.repo_path = Path(repo_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = str(device)
        self.atom_state_choice = atom_state_choice
        self._load_official_model()

    def _load_official_model(self) -> None:
        import torch
        from grover.data.molgraph import mol2graph
        from grover.model.models import GroverFpGeneration
        from grover.util.utils import load_checkpoint

        # The current Namespace is completed from checkpoint args by the pinned
        # official load_checkpoint implementation. fingerprint_source='atom'
        # constructs GroverFpGeneration's official two-view atom fingerprint;
        # extraction below still obeys the separately recorded scientific choice.
        current_args = Namespace(
            parser_name="fingerprint",
            fingerprint_source="atom",
            cuda=self.device == "cuda",
            device=self.device,
            no_cache=True,
            # The official grover_base.pt pretraining checkpoint omits this
            # parser default even though GROVEREmbedding requires it while
            # reconstructing the model. Official add_pretrain_args pins 0.0.
            dropout=0.0,
        )
        parameters = inspect.signature(load_checkpoint).parameters
        kwargs: dict[str, Any] = {}
        if "current_args" in parameters:
            kwargs["current_args"] = current_args
        else:
            raise OfficialGroverBackendError(
                "Pinned GROVER load_checkpoint lacks current_args; cannot "
                "construct a verified GroverFpGeneration wrapper."
            )
        if "cuda" in parameters:
            kwargs["cuda"] = self.device == "cuda"
        if "logger" in parameters:
            # Passing None makes the official loader print one line for every
            # checkpoint tensor. Preserve a quiet, inspectable production log.
            logger = logging.getLogger("rerank.grover_checkpoint")
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
            logger.setLevel(logging.CRITICAL)
            kwargs["logger"] = logger
        model = load_checkpoint(str(self.checkpoint_path), **kwargs)
        if not isinstance(model, GroverFpGeneration):
            raise OfficialGroverBackendError(
                "load_checkpoint did not construct GroverFpGeneration."
            )
        grover_encoder = getattr(model, "grover", None)
        if grover_encoder is None or not callable(grover_encoder):
            raise OfficialGroverBackendError(
                "GroverFpGeneration does not expose its grover encoder."
            )
        model_args = getattr(model, "args", current_args)
        model = model.to(self.device)
        model.eval()
        self._torch = torch
        self._model = model
        self._grover_encoder = model.grover
        self._model_args = model_args
        self._mol2graph = mol2graph

    def _selected_states(self, outputs: Mapping[str, Any]):
        if not isinstance(outputs, Mapping):
            raise OfficialGroverBackendError("GROVER encoder output is not a mapping.")
        required = {"atom_from_atom", "atom_from_bond"}
        if not required.issubset(outputs):
            raise OfficialGroverBackendError(
                "GROVER encoder output lacks atom_from_atom/atom_from_bond."
            )
        atom_from_atom = outputs["atom_from_atom"]
        atom_from_bond = outputs["atom_from_bond"]
        if atom_from_atom.ndim != 2 or atom_from_bond.ndim != 2:
            raise OfficialGroverBackendError("GROVER atom states must be matrices.")
        if atom_from_atom.shape[0] != atom_from_bond.shape[0]:
            raise OfficialGroverBackendError("GROVER atom views have different row counts.")
        if self.atom_state_choice == "atom_from_atom":
            return atom_from_atom
        if self.atom_state_choice == "atom_from_bond":
            return atom_from_bond
        return self._torch.cat((atom_from_atom, atom_from_bond), dim=1)

    def encode_atom_states(
        self, smiles_batch: Sequence[str], device: str
    ) -> list[dict]:
        from rdkit import Chem

        if str(device) != self.device:
            raise OfficialGroverBackendError(
                "Backend device changed after checkpoint initialization."
            )
        smiles = [str(value) for value in smiles_batch]
        if not smiles:
            return []
        molecules = [Chem.MolFromSmiles(value) for value in smiles]
        if any(molecule is None or molecule.GetNumAtoms() == 0 for molecule in molecules):
            raise OfficialGroverBackendError("Invalid molecule reached GROVER batching.")

        # mol2graph constructs one MolGraph per input in list order. Its
        # get_components contract is:
        # f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a.
        batch_graph = self._mol2graph(smiles, {}, self._model_args)
        components = batch_graph.get_components()
        if len(components) != 8:
            raise OfficialGroverBackendError(
                "Pinned BatchMolGraph must expose exactly eight components."
            )
        a_scope = components[5]
        if len(a_scope) != len(smiles):
            raise OfficialGroverBackendError("a_scope is not molecule-aligned.")
        # Mirror pinned GTransEncoder.forward exactly. It moves f_atoms,
        # f_bonds, a2b, b2a, b2revb and a2a internally while deliberately
        # leaving a_scope/b_scope as CPU-side Python scope data.
        with self._torch.no_grad():
            outputs = self._grover_encoder(components)
        states = self._selected_states(outputs)

        results: list[dict] = []
        for smiles_index, (scope, molecule) in enumerate(zip(a_scope, molecules)):
            if len(scope) != 2:
                raise OfficialGroverBackendError("Each a_scope entry must be (start, size).")
            atom_start, atom_count = (int(value) for value in scope)
            expected_count = int(molecule.GetNumAtoms())
            if atom_count != expected_count:
                raise OfficialGroverBackendError(
                    f"a_scope atom count differs from RDKit for input {smiles_index}."
                )
            atom_stop = atom_start + atom_count
            if atom_start < 0 or atom_stop > int(states.shape[0]):
                raise OfficialGroverBackendError("a_scope points outside atom-state rows.")
            matrix = (
                states[atom_start:atom_stop]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            if matrix.shape[0] != expected_count or not np.isfinite(matrix).all():
                raise OfficialGroverBackendError(
                    "Sliced GROVER atom states are incomplete or non-finite."
                )
            # Official MolGraph builds f_atoms by iterating mol.GetAtoms() and
            # BatchMolGraph concatenates without a within-molecule permutation.
            results.append(
                {
                    "atom_representations": matrix,
                    "rdkit_atom_indices": list(range(expected_count)),
                }
            )
        return results

    def metadata(self) -> dict:
        return {
            "backend": "official_grover_molgraph",
            "model_wrapper": "GroverFpGeneration",
            "batching": "mol2graph -> BatchMolGraph.get_components",
            "atom_scope_component_index": 5,
            "atom_state_choice": self.atom_state_choice,
            "official_fingerprint_source": "atom",
            "checkpoint_missing_parser_defaults": {"dropout": 0.0},
            "official_fingerprint_source_behavior": (
                "concatenates pooled atom_from_atom and atom_from_bond views; "
                "documented as rationale, not scientific approval"
            ),
            "rdkit_atom_order": (
                "MolGraph mol.GetAtoms order; exact a_scope slice; identity map"
            ),
        }
