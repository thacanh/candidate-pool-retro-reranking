import hashlib
import json
import os
import sys
import types

import numpy as np
import pytest

from rerank.benchmarks.benchmark_conformer_timing import (
    C1_CONFORMER_SEED,
    REQUIRED_CHECKPOINT_BASENAME,
    REQUIRED_DICTIONARY_BASENAME,
    InferenceBatch,
    PreparedBatch,
    UniMolCpuTimingBackend,
    collect_official_feature_workload,
    collect_required_smiles,
    run_timing_pilot,
    write_reports,
)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeBackend:
    def __init__(self, clock):
        self.clock = clock
        self.initialized = False
        self.preprocessed = []

    def initialize(self):
        self.clock.advance(5.0)
        self.initialized = True

    def preprocess(self, smiles, conformer_seed):
        assert self.initialized
        assert conformer_seed == C1_CONFORMER_SEED
        self.clock.advance(len(smiles) * 1.0)
        self.preprocessed.extend(smiles)
        statuses = [
            "fallback_2d" if value == "s41-a" else
            "failure" if value == "s31-a" else
            "ok"
            for value in smiles
        ]
        payloads = [None if status == "failure" else value for value, status in zip(smiles, statuses)]
        return PreparedBatch(payloads=payloads, statuses=statuses)

    def infer(self, prepared):
        self.clock.advance(len(prepared.payloads) * 2.0)
        representations = [
            None if payload is None else np.ones((2, 4), dtype=np.float32)
            for payload in prepared.payloads
        ]
        return InferenceBatch(representations)

    def metadata(self):
        return {
            "name": "fake",
            "package_fingerprint": "sha256:fake-package",
            "checkpoint_fingerprint": "sha256:fake-checkpoint",
        }


def _fake_serializer(clock):
    def serialize(representations):
        clock.advance(len(representations) * 0.5)
        arrays = sum(item is not None for item in representations)
        return {
            "arrays": arrays,
            "serialized_bytes": arrays * 32,
            "nonfinite_values": 0,
        }

    return serialize


class FakeRss:
    def __init__(self, initial=10_000):
        self.value = initial

    def __call__(self):
        self.value += 100
        return self.value


def test_collect_required_smiles_reports_exact_key_count(tmp_path):
    candidate_path = tmp_path / "candidates.jsonl"
    candidate_path.write_text(
        "\n".join(
            [
                json.dumps({"product": "CCO", "reactant": "CC.O"}),
                json.dumps({"product": "OCC", "reactant": "O.CC"}),
                "not-json",
                json.dumps({"product": "bad", "reactant": "C"}),
            ]
        ),
        encoding="utf-8",
    )
    inventory = collect_required_smiles(candidate_path)
    assert inventory.smiles == ["C", "CC", "CCO", "O"]
    assert inventory.audit["required_key_count"] == 4
    assert inventory.audit["malformed_json_lines"] == 1
    assert inventory.audit["invalid_molecular_smiles"] == 1


def test_official_workload_uses_feature_cache_eligibility_rules(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "reactants_smiles,products_smiles,set\n"
        "CC,CCO,train\n"
        "CC,CCC,train\n"
        "C,CO,train\n"
        "N,CN,train\n"
        "CC,CCC,valid\n"
        "CC,CCCC,valid\n"
        "N,CCN,test\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "reaction_id,reaction_class,source_split\n"
        "0,1,train\n1,1,train\n2,1,train\n3,1,train\n"
        "4,1,valid\n5,1,valid\n6,1,test\n",
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.jsonl"
    records = [
        {"product": "CCO", "reactant": "CC", "prior": 0.9},
        {"product": "CCO", "reactant": "C", "prior": 0.5},
        {"product": "CCC", "reactant": "CC", "prior": 0.9},
        {"product": "CCC", "reactant": "O", "prior": 0.4},
        {"product": "CO", "reactant": "C", "prior": 0.9},
        {"product": "CN", "reactant": "C", "prior": 0.9},
        {"product": "CCCC", "reactant": "C", "prior": 0.9},
        {"product": "CCN", "reactant": "N", "prior": 0.9},
        {"product": "CCN", "reactant": "C", "prior": 0.4},
    ]
    candidates.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    inventory = collect_official_feature_workload(source, metadata, candidates)

    assert inventory.smiles == ["C", "CC", "CCC", "CCN", "CCO", "N", "O"]
    assert inventory.audit["required_key_count"] == 7
    assert inventory.audit["eligible_train_products"] == 1
    assert inventory.audit["train_overlap_reactions_excluded"] == 1
    assert inventory.audit["train_products_uncovered"] == 1
    assert inventory.audit["train_products_without_negative"] == 1
    assert inventory.audit["evaluation"]["valid"]["reactions_covered"] == 1
    assert inventory.audit["evaluation"]["test"]["reactions_covered"] == 1


def test_fake_backend_pilot_is_stratified_and_excludes_warmup(tmp_path):
    sizes = {
        "s01-a": 5, "s01-b": 8,
        "s11-a": 11, "s11-b": 14, "s11-c": 17, "s11-d": 20,
        "s21-a": 21, "s21-b": 24, "s21-c": 27, "s21-d": 30,
        "s31-a": 31, "s31-b": 40,
        "s41-a": 41, "s41-b": 50,
        "s51-a": 51, "s51-b": 60,
        "s61-a": 61, "s61-b": 80,
        "s81-a": 81, "s81-b": 100, "s81-c": 128,
    }
    required = list(sizes)
    audit = {
        "required_key_count": len(required),
        "required_keys_sha256": "sha256:synthetic",
    }
    clock = FakeClock()
    backend = FakeBackend(clock)
    rss = FakeRss()
    report = run_timing_pilot(
        required_smiles=required,
        required_inventory_audit=audit,
        backend=backend,
        batch_size=2,
        sample_per_stratum=2,
        warmup_count=2,
        atom_counter=sizes.__getitem__,
        clock=clock,
        serializer=_fake_serializer(clock),
        rss_reader=rss,
    )

    assert report["conformer_label"] == "C1"
    assert report["conformer_seed"] == 42
    assert report["initialization"]["seconds"] == 5.0
    assert report["warmup"]["seconds"] == 7.0
    assert report["warmup"]["excluded_from_steady_state"] is True
    assert report["required_key_count_exact"] == len(required) == 21
    assert report["sample"]["count"] == 17
    assert report["warmup"]["count"] == 2
    assert report["warmup"]["disjoint_from_timed_sample"] is True
    assert report["warmup"]["stratum_counts"] == {"02_11_20": 1, "03_21_30": 1}
    assert report["steady_state"]["preprocess_seconds"] == 17.0
    assert report["steady_state"]["inference_seconds"] == 34.0
    assert report["steady_state"]["serialization_discard_seconds"] == 8.5
    assert report["steady_state"]["total_seconds"] == 59.5
    assert report["steady_state"]["failures"] == 1
    assert report["steady_state"]["fallbacks"] == 1
    assert report["embedding_artifacts_written"] is False
    assert report["backend"]["checkpoint_fingerprint"] == "sha256:fake-checkpoint"
    assert len(backend.preprocessed) == 19
    assert report["sample"]["batch_schedule"]["method"] == "round_robin_low_high_strata"
    assert report["steady_state"]["rss_observed_peak_bytes"] is not None
    assert report["system"]["process_rss_observed_peak_bytes"] is not None
    extrapolation = report["extrapolation_for_exact_required_key_count"]
    assert extrapolation["point_seconds"] == 73.5
    assert extrapolation["primary_estimator_name"] == "post_stratified_expansion"
    assert extrapolation["conservative_lower_seconds"] <= 73.5
    assert extrapolation["conservative_upper_seconds"] >= 73.5

    json_path = tmp_path / "timing.json"
    csv_path = tmp_path / "timing.csv"
    write_reports(report, json_path, csv_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["embedding_artifacts_written"] is False
    assert csv_path.read_text(encoding="utf-8").startswith("stratum,")
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".csv", ".json"]


def test_default_sampling_plan_is_489_timed_16_disjoint_and_62_batches():
    populations = [64, 72, 72, 64, 64, 64, 64, 41]
    atom_counts = [5, 15, 25, 35, 45, 55, 70, 100]
    sizes = {
        f"h{stratum_index}-{item_index}": atom_count
        for stratum_index, (population, atom_count) in enumerate(
            zip(populations, atom_counts), start=1
        )
        for item_index in range(population)
    }
    clock = FakeClock()
    report = run_timing_pilot(
        required_smiles=list(sizes),
        required_inventory_audit={"required_key_count": len(sizes)},
        backend=FakeBackend(clock),
        batch_size=8,
        sample_per_stratum=64,
        warmup_count=16,
        atom_counter=sizes.__getitem__,
        clock=clock,
        serializer=_fake_serializer(clock),
        rss_reader=lambda: None,
    )

    assert report["sample"]["count"] == 489
    assert report["warmup"]["count"] == 16
    assert report["warmup"]["stratum_counts"] == {
        "02_11_20": 8,
        "03_21_30": 8,
    }
    assert report["warmup"]["disjoint_from_timed_sample"] is True
    schedule = report["sample"]["batch_schedule"]
    assert schedule["batch_count"] == 62
    assert schedule["stratum_order"][:8] == [
        "01_1_10",
        "08_81_128",
        "02_11_20",
        "07_61_80",
        "03_21_30",
        "06_51_60",
        "04_31_40",
        "05_41_50",
    ]


def test_unimol_backend_is_lazy_and_c1_only():
    assert "unimol_tools" not in sys.modules
    backend = UniMolCpuTimingBackend(
        checkpoint_path="not-loaded.pt",
        batch_size=2,
        threads=1,
    )
    assert "unimol_tools" not in sys.modules
    with pytest.raises(ValueError, match="restricted to C1"):
        run_timing_pilot(
            required_smiles=["C"],
            required_inventory_audit={"required_key_count": 1},
            backend=backend,
            batch_size=1,
            sample_per_stratum=1,
            conformer_seed=43,
        )
    assert "unimol_tools" not in sys.modules


def _fake_unimol_backend(monkeypatch, tmp_path, inference_error=None):
    monkeypatch.delenv("UNIMOL_WEIGHT_DIR", raising=False)
    weight_dir = tmp_path / "weights"
    weight_dir.mkdir()
    checkpoint = weight_dir / REQUIRED_CHECKPOINT_BASENAME
    dictionary = weight_dir / REQUIRED_DICTIONARY_BASENAME
    checkpoint.write_bytes(b"synthetic-checkpoint")
    dictionary.write_text("[PAD]\n[CLS]\n", encoding="utf-8")
    expected_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    package_source = tmp_path / "unimol_tools_init.py"
    rdkit_source = tmp_path / "rdkit_init.py"
    package_source.write_text("# fake unimol_tools\n", encoding="utf-8")
    rdkit_source.write_text("# fake rdkit\n", encoding="utf-8")

    captures = {}

    class FakeUniMolRepr:
        def __init__(
            self,
            data_type="molecule",
            batch_size=32,
            remove_hs=False,
            model_name="unimolv1",
            model_size="84m",
            use_cuda=True,
            use_ddp=False,
            use_gpu="all",
        ):
            captures["repr_kwargs"] = {
                "data_type": data_type,
                "batch_size": batch_size,
                "remove_hs": remove_hs,
                "model_name": model_name,
                "model_size": model_size,
                "use_cuda": use_cuda,
                "use_ddp": use_ddp,
                "use_gpu": use_gpu,
            }
            captures["weight_dir_at_repr_init"] = os.environ.get(
                "UNIMOL_WEIGHT_DIR"
            )
            self.model = object()
            self.params = dict(captures["repr_kwargs"])

    class FakeConformerGen:
        def __init__(self, **params):
            captures["conformer_kwargs"] = dict(params)

        def single_process(self, smiles):
            captures.setdefault("preprocessed", []).append(smiles)
            return {
                "src_coord": np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32
                )
            }

    class FakeMolDataset:
        def __init__(self, data, label=None):
            self.data = data
            self.label = label

        def __len__(self):
            return len(self.data)

    class FakeTrainer:
        def __init__(self, save_path=None, **params):
            captures["trainer_kwargs"] = dict(params)

        def inference(
            self,
            model,
            dataset,
            return_repr=False,
            return_atomic_reprs=False,
            feature_name=None,
        ):
            captures["inference_call"] = {
                "model": model,
                "dataset": dataset,
                "return_repr": return_repr,
                "return_atomic_reprs": return_atomic_reprs,
                "feature_name": feature_name,
            }
            if inference_error is not None:
                raise inference_error
            return {
                "atomic_reprs": [
                    np.ones((2, 4), dtype=np.float32) for _ in dataset.data
                ]
            }

    torch_module = types.ModuleType("torch")
    torch_module.__version__ = "fake-torch"
    torch_state = {"threads": 0, "interop_threads": 0}
    torch_module.set_num_threads = lambda value: torch_state.update(threads=value)
    torch_module.set_num_interop_threads = lambda value: torch_state.update(
        interop_threads=value
    )
    torch_module.get_num_threads = lambda: torch_state["threads"]
    torch_module.get_num_interop_threads = lambda: torch_state["interop_threads"]
    torch_module.manual_seed = lambda value: captures.update(torch_seed=value)

    rdkit_module = types.ModuleType("rdkit")
    rdkit_module.__file__ = str(rdkit_source)
    unimol_module = types.ModuleType("unimol_tools")
    unimol_module.__file__ = str(package_source)
    unimol_module.__path__ = []
    unimol_module.UniMolRepr = FakeUniMolRepr
    data_module = types.ModuleType("unimol_tools.data")
    data_module.__path__ = []
    conformer_module = types.ModuleType("unimol_tools.data.conformer")
    conformer_module.ConformerGen = FakeConformerGen
    predictor_module = types.ModuleType("unimol_tools.predictor")
    predictor_module.MolDataset = FakeMolDataset
    tasks_module = types.ModuleType("unimol_tools.tasks")
    tasks_module.__path__ = []
    tasks_module.Trainer = FakeTrainer
    weights_module = types.ModuleType("unimol_tools.weights")
    weights_module.__path__ = []
    weights_module.WEIGHT_DIR = str(weight_dir.resolve())
    weighthub_module = types.ModuleType("unimol_tools.weights.weighthub")
    weighthub_module.WEIGHT_DIR = str(weight_dir.resolve())

    fake_modules = {
        "torch": torch_module,
        "rdkit": rdkit_module,
        "unimol_tools": unimol_module,
        "unimol_tools.data": data_module,
        "unimol_tools.data.conformer": conformer_module,
        "unimol_tools.predictor": predictor_module,
        "unimol_tools.tasks": tasks_module,
        "unimol_tools.weights": weights_module,
        "unimol_tools.weights.weighthub": weighthub_module,
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def fake_version(name):
        return {"unimol-tools": "0.1.3", "rdkit": "2025.9.6"}[name]

    class FakeDistribution:
        def read_text(self, name):
            assert name == "RECORD"
            return "synthetic,record\n"

    monkeypatch.setattr(
        "rerank.benchmarks.benchmark_conformer_timing.importlib.metadata.version",
        fake_version,
    )
    monkeypatch.setattr(
        "rerank.benchmarks.benchmark_conformer_timing.importlib.metadata.distribution",
        lambda name: FakeDistribution(),
    )

    backend = UniMolCpuTimingBackend(
        checkpoint_path=checkpoint,
        batch_size=2,
        threads=1,
        expected_checkpoint_sha256=expected_sha256,
    )
    return backend, captures, weight_dir


def test_unimol_013_api_contract_and_normalized_rdkit_version(monkeypatch, tmp_path):
    backend, captures, weight_dir = _fake_unimol_backend(monkeypatch, tmp_path)

    backend.initialize()
    prepared = backend.preprocess(["CC"], conformer_seed=C1_CONFORMER_SEED)
    inferred = backend.infer(prepared)

    assert captures["weight_dir_at_repr_init"] == str(weight_dir.resolve())
    assert captures["repr_kwargs"] == {
        "data_type": "molecule",
        "batch_size": 2,
        "remove_hs": True,
        "model_name": "unimolv1",
        "model_size": "84m",
        "use_cuda": False,
        "use_ddp": False,
        "use_gpu": "all",
    }
    assert "pretrained_dict_path" not in captures["conformer_kwargs"]
    assert prepared.statuses == ["ok"]
    assert isinstance(prepared.payloads[0], dict)
    assert inferred.atom_representations[0].shape == (2, 4)
    assert captures["inference_call"]["return_repr"] is True
    assert captures["inference_call"]["return_atomic_reprs"] is True
    assert captures["inference_call"]["feature_name"] is None
    metadata = backend.metadata()
    assert metadata["rdkit"]["version"] == "2025.9.6"
    assert metadata["unimol_weight_dir"] == str(weight_dir.resolve())
    assert metadata["checkpoint"]["path"].endswith(REQUIRED_CHECKPOINT_BASENAME)
    assert metadata["dictionary"]["path"].endswith(REQUIRED_DICTIONARY_BASENAME)


def test_unimol_inference_exception_is_not_silently_converted_to_failures(
    monkeypatch, tmp_path
):
    backend, _, _ = _fake_unimol_backend(
        monkeypatch, tmp_path, inference_error=ValueError("sentinel batch failure")
    )
    backend.initialize()
    prepared = backend.preprocess(["CC"], conformer_seed=C1_CONFORMER_SEED)

    with pytest.raises(
        RuntimeError, match="ValueError: sentinel batch failure"
    ) as error:
        backend.infer(prepared)
    assert isinstance(error.value.__cause__, ValueError)


def test_unimol_backend_rejects_noncanonical_weight_filename(monkeypatch, tmp_path):
    checkpoint = tmp_path / "renamed-checkpoint.pt"
    dictionary = tmp_path / REQUIRED_DICTIONARY_BASENAME
    checkpoint.write_bytes(b"checkpoint")
    dictionary.write_text("dictionary", encoding="utf-8")
    versions = {"unimol-tools": "0.1.3", "rdkit": "2025.9.6"}
    monkeypatch.setattr(
        "rerank.benchmarks.benchmark_conformer_timing.importlib.metadata.version",
        versions.__getitem__,
    )
    backend = UniMolCpuTimingBackend(
        checkpoint_path=checkpoint,
        dictionary_path=dictionary,
        batch_size=1,
        threads=1,
        expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )

    with pytest.raises(RuntimeError, match=REQUIRED_CHECKPOINT_BASENAME):
        backend.initialize()
