import hashlib
import json
import pickle
import sys
import types
from argparse import Namespace

import numpy as np
import pytest

from rerank.data.build_encoder_control_features import (
    build_feature_shards,
    iter_candidate_queries,
    main as build_main,
)
from rerank.data.compile_encoder_control_cache import compile_encoder_control_caches
from rerank.data.merge_encoder_feature_partitions import merge_feature_partitions
from rerank.encoder_controls import (
    CONTROL_FEATURE_NAMES,
    FULL_FEATURE_NAMES,
    GROVER_CHECKPOINT_BASENAME,
    GROVER_REPOSITORY_COMMIT,
    EncoderControlError,
    GroverAssetExpectations,
    GroverAtomEncoder,
    MorganAtomEncoder,
    compute_control_scalars,
    compute_full_query_features,
    compute_query_control_features,
)
from rerank.features import (
    _atom_set_similarity,
    _cosine,
    _heavy_atom_ratio,
    _morgan_similarity,
    _reaction_distance,
)
from rerank.grover_official_backend import OfficialGroverAtomBackend
from rerank.revision_tuning import load_selection_bundle, prepare_selection_bundle


class FakeAtomEncoder:
    def __init__(self):
        self.calls = []
        self.initialized = False

    def initialize(self):
        self.initialized = True

    def encode_fragments_batch(self, smiles):
        assert self.initialized
        self.calls.append(list(smiles))
        return [
            np.asarray(
                [[float(len(value)), 1.0], [1.0, float(len(value) + 1)]],
                dtype=np.float32,
            )
            for value in smiles
        ]

    def metadata(self):
        assert self.initialized
        return {"name": "fake-atom-encoder", "global_atom_cache": False}


def test_control_scalars_are_exactly_the_current_three_formulas():
    product = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    reactant = np.asarray([[1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    product_mean = product.mean(axis=0)
    reaction_vector = product_mean - reactant.mean(axis=0)
    expected = np.asarray(
        [
            _atom_set_similarity(product, reactant),
            _reaction_distance(product, reactant),
            _cosine(product_mean, reaction_vector),
        ],
        dtype=np.float32,
    )

    observed = compute_control_scalars(product, reactant)

    np.testing.assert_allclose(observed, expected, rtol=0, atol=0)
    assert CONTROL_FEATURE_NAMES == (
        "atom_set_similarity",
        "reaction_distance",
        "cosine_reaction_vec",
    )


def test_query_features_stream_product_once_and_candidates_in_batches():
    encoder = FakeAtomEncoder()
    encoder.initialize()

    features = compute_query_control_features(
        encoder, "product", ["a", "bb", "ccc", "dddd", "eeeee"], batch_size=2
    )

    assert features.shape == (5, 3)
    assert encoder.calls == [
        ["product"],
        ["a", "bb"],
        ["ccc", "dddd"],
        ["eeeee"],
    ]


def test_full_rows_preserve_frozen_seven_column_order_and_formulas():
    encoder = FakeAtomEncoder()
    encoder.initialize()
    candidates = ["C.O", "CC"]
    rows = compute_full_query_features(
        encoder, "CCO", candidates, [0.8, 0.2], batch_size=2
    )

    assert FULL_FEATURE_NAMES == (
        "prior_or_log_prob",
        "morgan_similarity",
        "atom_set_similarity",
        "reaction_distance",
        "cosine_reaction_vec",
        "n_fragments",
        "heavy_atom_ratio",
    )
    assert rows.shape == (2, 7)
    np.testing.assert_array_equal(rows[:, 0], np.asarray([0.8, 0.2], dtype=np.float32))
    np.testing.assert_allclose(
        rows[:, 1], [_morgan_similarity("CCO", value) for value in candidates]
    )
    np.testing.assert_array_equal(rows[:, 5], [2.0, 1.0])
    np.testing.assert_allclose(
        rows[:, 6], [_heavy_atom_ratio("CCO", value) for value in candidates]
    )


def test_morgan_atom_vectors_use_exact_prespecified_options(monkeypatch):
    from rdkit.Chem import AllChem

    calls = []
    original = AllChem.GetMorganFingerprintAsBitVect

    def spy(molecule, radius, *args, **kwargs):
        calls.append((molecule.GetNumAtoms(), radius, dict(kwargs)))
        return original(molecule, radius, *args, **kwargs)

    monkeypatch.setattr(AllChem, "GetMorganFingerprintAsBitVect", spy)
    encoder = MorganAtomEncoder()
    encoder.initialize()

    states = encoder.encode_fragments_batch(["CC.O"])[0]

    assert states.shape == (3, 2048)
    assert states.dtype == np.float32
    assert len(calls) == 3
    assert all(radius == 2 for _, radius, _ in calls)
    assert all(call[2]["nBits"] == 2048 for call in calls)
    assert all(call[2]["useChirality"] is False for call in calls)
    assert [call[2]["fromAtoms"] for call in calls] == [[0], [1], [0]]
    assert encoder.metadata()["global_atom_cache"] is False


class FakeGroverBackend:
    def __init__(self, invalid_mapping=False):
        self.invalid_mapping = invalid_mapping
        self.calls = []

    def encode_atom_states(self, smiles_batch, device):
        from rdkit import Chem

        self.calls.append((list(smiles_batch), device))
        outputs = []
        for smiles in smiles_batch:
            count = Chem.MolFromSmiles(smiles).GetNumAtoms()
            indices = list(reversed(range(count)))
            if self.invalid_mapping and count > 1:
                indices = [0] * count
            states = np.asarray(
                [[float(index + 1), float(count)] for index in indices],
                dtype=np.float32,
            )
            outputs.append(
                {
                    "atom_representations": states,
                    "rdkit_atom_indices": indices,
                }
            )
        return outputs


def _grover_encoder(tmp_path, backend, atom_state_choice="atom_from_atom"):
    repo = tmp_path / "grover"
    repo.mkdir()
    checkpoint = tmp_path / GROVER_CHECKPOINT_BASENAME
    checkpoint.write_bytes(b"synthetic-grover")
    expected = GroverAssetExpectations(
        repository_commit=GROVER_REPOSITORY_COMMIT,
        checkpoint_basename=GROVER_CHECKPOINT_BASENAME,
        checkpoint_size=checkpoint.stat().st_size,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    captures = {}

    def factory(**kwargs):
        captures.update(kwargs)
        return backend

    encoder = GroverAtomEncoder(
        repo_path=repo,
        checkpoint_path=checkpoint,
        device="cpu",
        batch_size=2,
        atom_state_choice=atom_state_choice,
        backend_factory=factory,
        expectations=expected,
        git_head_reader=lambda path: GROVER_REPOSITORY_COMMIT,
    )
    return encoder, captures


def test_grover_adapter_pins_assets_and_reorders_explicit_rdkit_atom_map(tmp_path):
    backend = FakeGroverBackend()
    encoder, captures = _grover_encoder(tmp_path, backend)
    encoder.initialize()

    states = encoder.encode_fragments_batch(["CO.N"])[0]

    np.testing.assert_array_equal(
        states,
        np.asarray([[1.0, 2.0], [2.0, 2.0], [1.0, 1.0]], dtype=np.float32),
    )
    assert captures["device"] == "cpu"
    assert captures["atom_state_choice"] == "atom_from_atom"
    metadata = encoder.metadata()
    assert metadata["repository"]["commit"] == GROVER_REPOSITORY_COMMIT
    assert metadata["checkpoint"]["path"].endswith(GROVER_CHECKPOINT_BASENAME)
    assert metadata["atom_state_choice"] == "atom_from_atom"
    assert metadata["global_atom_cache"] is False


def test_grover_adapter_fails_closed_on_ambiguous_atom_alignment(tmp_path):
    encoder, _ = _grover_encoder(tmp_path, FakeGroverBackend(invalid_mapping=True))
    encoder.initialize()

    with pytest.raises(EncoderControlError, match="exact RDKit-atom permutation"):
        encoder.encode_fragments_batch(["CO"])


def test_grover_adapter_fails_closed_without_external_paths(monkeypatch):
    monkeypatch.delenv("GROVER_REPO", raising=False)
    monkeypatch.delenv("GROVER_CHECKPOINT", raising=False)
    encoder = GroverAtomEncoder(device="cpu", atom_state_choice="atom_from_atom")

    with pytest.raises(EncoderControlError, match="GROVER_REPO"):
        encoder.initialize()


def _install_fake_official_grover(monkeypatch):
    import torch

    captures = {}

    class FakeGroverFpGeneration:
        def __init__(self, args):
            self.args = args

        def to(self, device):
            captures["model_device"] = device
            return self

        def eval(self):
            captures["model_eval"] = True
            return self

        def grover(self, components):
            captures["grover_components"] = components
            return {
                "atom_from_atom": torch.tensor(
                    [[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
                ),
                "atom_from_bond": torch.tensor(
                    [
                        [0.0, 0.0],
                        [100.0, 1000.0],
                        [200.0, 2000.0],
                        [300.0, 3000.0],
                    ]
                ),
            }

    class FakeBatchMolGraph:
        def get_components(self):
            tensor = torch.zeros((1, 1), dtype=torch.float32)
            return (
                tensor,
                tensor,
                tensor,
                tensor,
                tensor,
                [(1, 2), (3, 1)],
                [(1, 1), (2, 1)],
                tensor,
            )

    def mol2graph(smiles_batch, shared_dict, args):
        captures["mol2graph"] = {
            "smiles_batch": list(smiles_batch),
            "shared_dict": shared_dict,
            "args": args,
        }
        return FakeBatchMolGraph()

    def load_checkpoint(path, current_args=None, cuda=None, logger=None):
        captures["load_checkpoint"] = {
            "path": path,
            "current_args": current_args,
            "cuda": cuda,
            "logger": logger,
        }
        return FakeGroverFpGeneration(current_args)

    modules = {
        "grover": types.ModuleType("grover"),
        "grover.data": types.ModuleType("grover.data"),
        "grover.data.molgraph": types.ModuleType("grover.data.molgraph"),
        "grover.model": types.ModuleType("grover.model"),
        "grover.model.models": types.ModuleType("grover.model.models"),
        "grover.util": types.ModuleType("grover.util"),
        "grover.util.utils": types.ModuleType("grover.util.utils"),
    }
    modules["grover"].__path__ = []
    modules["grover.data"].__path__ = []
    modules["grover.model"].__path__ = []
    modules["grover.util"].__path__ = []
    modules["grover.data.molgraph"].mol2graph = mol2graph
    modules["grover.model.models"].GroverFpGeneration = FakeGroverFpGeneration
    modules["grover.util.utils"].load_checkpoint = load_checkpoint
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return captures


@pytest.mark.parametrize(
    "choice,expected_first",
    [
        ("atom_from_atom", [[1.0, 10.0], [2.0, 20.0]]),
        ("atom_from_bond", [[100.0, 1000.0], [200.0, 2000.0]]),
        (
            "concatenation",
            [[1.0, 10.0, 100.0, 1000.0], [2.0, 20.0, 200.0, 2000.0]],
        ),
    ],
)
def test_official_grover_backend_uses_molgraph_a_scope_and_named_view(
    monkeypatch, tmp_path, choice, expected_first
):
    captures = _install_fake_official_grover(monkeypatch)
    backend = OfficialGroverAtomBackend(
        repo_path=tmp_path,
        checkpoint_path=tmp_path / "grover_base.pt",
        device="cpu",
        atom_state_choice=choice,
    )

    outputs = backend.encode_atom_states(["CO", "N"], device="cpu")

    np.testing.assert_array_equal(outputs[0]["atom_representations"], expected_first)
    assert outputs[0]["rdkit_atom_indices"] == [0, 1]
    assert outputs[1]["rdkit_atom_indices"] == [0]
    assert captures["mol2graph"]["smiles_batch"] == ["CO", "N"]
    args = captures["load_checkpoint"]["current_args"]
    assert isinstance(args, Namespace)
    assert args.parser_name == "fingerprint"
    assert args.fingerprint_source == "atom"
    assert args.no_cache is True
    assert args.dropout == 0.0
    assert captures["load_checkpoint"]["logger"] is not None
    assert backend.metadata()["atom_scope_component_index"] == 5
    assert backend.metadata()["checkpoint_missing_parser_defaults"] == {
        "dropout": 0.0
    }


def test_official_grover_backend_preserves_raw_cpu_scopes_for_cuda_contract(
    monkeypatch, tmp_path
):
    captures = _install_fake_official_grover(monkeypatch)
    backend = OfficialGroverAtomBackend(
        repo_path=tmp_path,
        checkpoint_path=tmp_path / "grover_base.pt",
        device="cuda",
        atom_state_choice="atom_from_atom",
    )

    backend.encode_atom_states(["CO", "N"], device="cuda")

    components = captures["grover_components"]
    assert isinstance(components, tuple)
    assert components[0].device.type == "cpu"
    assert components[5] == [(1, 2), (3, 1)]
    assert components[6] == [(1, 1), (2, 1)]
    assert captures["model_device"] == "cuda"


def test_grover_cli_requires_explicit_atom_state_choice(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_encoder_control_features.py",
            "--input-jsonl",
            str(tmp_path / "input.jsonl"),
            "--output-dir",
            str(tmp_path / "output"),
            "--encoder",
            "grover",
        ],
    )

    with pytest.raises(SystemExit, match="atom-state-choice is required"):
        build_main()


def test_grover_cli_rejects_unapproved_atom_state_choice(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_encoder_control_features.py",
            "--input-jsonl",
            str(tmp_path / "input.jsonl"),
            "--output-dir",
            str(tmp_path / "output"),
            "--encoder",
            "grover",
            "--grover-atom-state-choice",
            "atom_from_atom",
        ],
    )

    with pytest.raises(SystemExit, match="frozen to GROVER concatenation"):
        build_main()


def test_candidate_queries_group_noncontiguous_canonical_products(tmp_path):
    input_path = tmp_path / "noncontiguous.jsonl"
    records = [
        {"product": "C(C)O", "reactant": "C", "prior": 0.5},
        {"product": "CCN", "reactant": "N", "prior": 0.6},
        {"product": "CCO", "reactant": "CO", "prior": 0.7},
        {"product": "CCO", "reactant": "C", "prior": 0.9},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    queries = list(iter_candidate_queries(input_path))

    assert len(queries) == 2
    assert queries[0].canonical_product_identity == "CCO"
    assert queries[0].candidate_smiles == ["C", "CO"]
    assert queries[0].priors == [0.9, 0.7]
    assert queries[1].canonical_product_identity == "CCN"


def test_scalar_shards_resume_without_writing_atom_representations(tmp_path):
    input_path = tmp_path / "queries.jsonl"
    records = [
        {"product": "CCO", "reactant": "C", "prior": 0.9},
        {"product": "CCO", "reactant": "CC", "prior": 0.8},
        {"product": "CCO", "reactant": "C", "prior": 0.1},
        {"product": "CCN", "reactant": "N", "prior": 0.7},
        {"product": "CCC", "reactant": "O", "prior": 0.6},
        {"product": "CCC", "reactant": "CO", "prior": 0.5},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    output_dir = tmp_path / "features"
    first_encoder = FakeAtomEncoder()

    partial = build_feature_shards(
        input_path,
        output_dir,
        first_encoder,
        protocol_id="C-FAKE",
        encoder_batch_size=2,
        queries_per_shard=1,
        max_queries=1,
    )

    assert partial["status"] == "partial"
    assert partial["completed_query_count"] == 1
    second_encoder = FakeAtomEncoder()
    complete = build_feature_shards(
        input_path,
        output_dir,
        second_encoder,
        protocol_id="C-FAKE",
        encoder_batch_size=2,
        queries_per_shard=1,
        resume=True,
    )

    assert complete["status"] == "complete"
    assert complete["completed_query_count"] == 3
    assert complete["completed_pair_count"] == 5
    assert len(complete["shards"]) == 3
    assert all("CCO" not in batch for batch in second_encoder.calls)
    assert complete["atom_embeddings_written"] is False
    for shard in complete["shards"]:
        with np.load(output_dir / shard["path"], allow_pickle=False) as payload:
            assert set(payload.files) == {
                "schema_version",
                "feature_names",
                "query_ids",
                "product_smiles",
                "canonical_product_identities",
                "query_offsets",
                "candidate_smiles",
                "canonical_candidate_identities",
                "priors",
                "ranks",
                "features",
            }
            assert payload["features"].shape[1] == 7
            assert tuple(payload["feature_names"].tolist()) == FULL_FEATURE_NAMES


    source = tmp_path / "source.csv"
    source.write_text(
        "reactants_smiles,products_smiles,set\n"
        "C,CCO,train\n"
        "N,CCN,valid\n"
        "O,CCC,test\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "reaction_id,reaction_class,source_split\n"
        "0,1,train\n1,2,valid\n2,3,test\n",
        encoding="utf-8",
    )
    train_test_path = tmp_path / "train_test.pkl"
    validation_path = tmp_path / "validation.pkl"
    assert train_test_path != validation_path
    compiled = compile_encoder_control_caches(
        output_dir / "manifest.json",
        source,
        metadata,
        train_test_path,
        validation_path,
    )

    assert compiled["atom_embeddings_written"] is False
    with train_test_path.open("rb") as handle:
        train_test = pickle.load(handle)
    payload = train_test["payload"]
    assert payload["feature_mode"] == "3d+prior"
    assert payload["train_products"][0]["features"].shape == (2, 7)
    np.testing.assert_array_equal(payload["train_products"][0]["labels"], [1, 0])
    assert payload["train_products"][0]["source_reaction_ids"] == [0]
    assert payload["eval_metadata"][0]["reaction_id"] == 2
    assert payload["eval_metadata"][0]["source_split"] == "test"
    assert train_test["encoder_control_has_conformer"] is False
    assert "conformer_seed" not in train_test
    with validation_path.open("rb") as handle:
        validation = pickle.load(handle)
    assert validation["payload"]["eval_metadata"][0]["reaction_id"] == 1
    assert validation["payload"]["eval_metadata"][0]["source_split"] == "valid"
    assert validation["encoder_control_has_conformer"] is False
    assert "conformer_seed" not in validation
    assert validation["tuned_runner_compatibility"] == {
        "cache_layout": "seven-column 3d+prior",
        "conformer_seed_field_omitted": True,
        "prepare_conformer_seed_argument_is_not_encoder_provenance": True,
    }

    selection_path = tmp_path / "selection.pkl"
    prepare_selection_bundle(train_test_path, validation_path, selection_path, 42)
    selection = load_selection_bundle(selection_path)
    assert selection["train_products"][0]["features"].shape[1] == 7
    assert selection["validation_payload"]["eval_features"][0].shape[1] == 7
    assert "eval_metadata" not in selection


def test_independent_query_partitions_merge_to_exact_serial_features(tmp_path):
    input_path = tmp_path / "queries.jsonl"
    records = [
        {"product": "CCO", "reactant": "C", "prior": 0.9},
        {"product": "CCO", "reactant": "CC", "prior": 0.4},
        {"product": "CCN", "reactant": "N", "prior": 0.8},
        {"product": "CCC", "reactant": "O", "prior": 0.7},
        {"product": "CCC", "reactant": "CO", "prior": 0.3},
        {"product": "CCCl", "reactant": "Cl", "prior": 0.6},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    serial_dir = tmp_path / "serial"
    serial = build_feature_shards(
        input_path,
        serial_dir,
        FakeAtomEncoder(),
        protocol_id="C-FAKE-PARTITIONS",
        encoder_batch_size=2,
        queries_per_shard=2,
    )
    feature_root = tmp_path / "parallel"
    partition_paths = []
    encoders = []
    for partition_index, (start, stop) in enumerate(((0, 2), (2, 4))):
        encoder = FakeAtomEncoder()
        partition_dir = feature_root / "partitions" / f"part_{partition_index:02d}"
        partition = build_feature_shards(
            input_path,
            partition_dir,
            encoder,
            protocol_id="C-FAKE-PARTITIONS",
            encoder_batch_size=2,
            queries_per_shard=2,
            query_start_index=start,
            query_stop_index=stop,
        )
        assert partition["status"] == "complete"
        assert partition["identity"]["parallelization"] == (
            "independent_query_scoped_worker_partition"
        )
        assert all(len(call) <= 2 for call in encoder.calls)
        encoders.append(encoder)
        partition_paths.append(partition_dir / "manifest.json")

    merged = merge_feature_partitions(
        list(reversed(partition_paths)),
        feature_root / "manifest.json",
        expected_query_count=4,
        expected_pair_count=6,
    )

    assert merged["schema_version"] == 3
    assert merged["completed_query_count"] == serial["completed_query_count"] == 4
    assert merged["completed_pair_count"] == serial["completed_pair_count"] == 6
    assert merged["identity"]["partition_count"] == 2
    assert merged["identity"]["global_atom_cache"] is False
    assert merged["atom_embeddings_written"] is False

    def load_rows(manifest, root):
        rows = []
        query_ids = []
        for shard in manifest["shards"]:
            with np.load(root / shard["path"], allow_pickle=False) as payload:
                rows.append(payload["features"])
                query_ids.extend(payload["query_ids"].tolist())
        return query_ids, np.concatenate(rows, axis=0)

    serial_ids, serial_rows = load_rows(serial, serial_dir)
    merged_ids, merged_rows = load_rows(merged, feature_root)
    assert merged_ids == serial_ids
    np.testing.assert_array_equal(merged_rows, serial_rows)

    source = tmp_path / "partition-source.csv"
    source.write_text(
        "reactants_smiles,products_smiles,set\n"
        "C,CCO,train\n"
        "N,CCN,valid\n"
        "O,CCC,test\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "partition-metadata.csv"
    metadata.write_text(
        "reaction_id,reaction_class,source_split\n"
        "0,1,train\n1,2,valid\n2,3,test\n",
        encoding="utf-8",
    )
    compiled = compile_encoder_control_caches(
        feature_root / "manifest.json",
        source,
        metadata,
        tmp_path / "partition-train-test.pkl",
        tmp_path / "partition-validation.pkl",
    )
    assert compiled["atom_embeddings_written"] is False
    assert compiled["train_audit"]["train_products"] == 1

    with pytest.raises(ValueError, match="identity does not match"):
        build_feature_shards(
            input_path,
            partition_paths[0].parent,
            FakeAtomEncoder(),
            protocol_id="C-FAKE-PARTITIONS",
            encoder_batch_size=2,
            queries_per_shard=2,
            resume=True,
            query_start_index=0,
            query_stop_index=3,
        )


@pytest.mark.parametrize("existing_index", [0, 1])
def test_cache_compiler_preflight_refuses_either_existing_target(
    tmp_path, existing_index
):
    targets = [tmp_path / "train_test.pkl", tmp_path / "validation.pkl"]
    targets[existing_index].write_bytes(b"preserve-me")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        compile_encoder_control_caches(
            tmp_path / "manifest-does-not-need-to-be-opened.json",
            tmp_path / "source-does-not-need-to-be-opened.csv",
            tmp_path / "metadata-does-not-need-to-be-opened.csv",
            targets[0],
            targets[1],
        )

    assert targets[existing_index].read_bytes() == b"preserve-me"
    assert not targets[1 - existing_index].exists()


def test_cache_compiler_requires_distinct_output_paths_before_input_read(tmp_path):
    target = tmp_path / "same.pkl"

    with pytest.raises(ValueError, match="must be separate"):
        compile_encoder_control_caches(
            tmp_path / "missing-manifest.json",
            tmp_path / "missing-source.csv",
            tmp_path / "missing-metadata.csv",
            target,
            target,
        )
    assert not target.exists()
