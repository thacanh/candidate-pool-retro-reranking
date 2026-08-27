import csv
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import rerank.experiments.run_conformer_seed as seed_runner
from rerank.data.build_conformer_sqlite import build_sqlite_cache, validate_conformer_seed
from rerank.data.collect_conformer_runs import collect_runs
from rerank.study_data import STUDY_CACHE_SCHEMA


class _FakeCudaProperties:
    def __init__(self, total_memory):
        self.total_memory = total_memory


class _FakeCuda:
    def __init__(self, available, total_memory=0):
        self._available = available
        self._total_memory = total_memory

    def is_available(self):
        return self._available

    def get_device_properties(self, index):
        assert index == 0
        return _FakeCudaProperties(self._total_memory)

    def get_device_name(self, index):
        assert index == 0
        return "Synthetic GPU"

    def get_device_capability(self, index):
        assert index == 0
        return (8, 6)


class _FakeTorch:
    __version__ = "test"

    def __init__(self, available, total_memory=0):
        self.cuda = _FakeCuda(available, total_memory)
        self.version = type("Version", (), {"cuda": "test-cuda"})()


class FakeBackend:
    def __init__(self):
        self.calls = []

    def metadata(self):
        return {"name": "fake", "seed": 42}

    def encode_batch(self, smiles):
        self.calls.append(list(smiles))
        arrays = [
            np.full((index + 1, 512), index + 0.25, dtype=np.float32)
            for index, _ in enumerate(smiles)
        ]
        statuses = ["fallback_2d" if value == "CC" else "ok" for value in smiles]
        return arrays, statuses, [None] * len(smiles)


class MustNotRunBackend:
    def metadata(self):
        raise AssertionError("completed cache should not initialize the backend")

    def encode_batch(self, smiles):
        raise AssertionError("completed cache should not encode")


class _OomTrainer:
    def inference(self, model, dataset, **kwargs):
        if len(dataset) > 2:
            raise RuntimeError("CUDA out of memory in synthetic test")
        return {
            "atomic_reprs": [np.zeros((1, 512), dtype=np.float32) for _ in dataset]
        }


def test_seed_validation_and_layout_are_isolated(tmp_path):
    assert validate_conformer_seed(42) == 42
    assert seed_runner.conformer_label(42) == "C1"
    assert seed_runner.conformer_label(51) == "C10"
    with pytest.raises(ValueError):
        validate_conformer_seed(52)

    first = seed_runner.build_layout(tmp_path, 42)
    second = seed_runner.build_layout(tmp_path, 43)
    assert first["root"].name == "seed_42"
    assert second["root"].name == "seed_43"
    assert first["atom_cache"] != second["atom_cache"]
    assert first["feature_cache"].is_relative_to(first["root"])


def test_runtime_auto_selects_cpu_and_physical_cores():
    runtime = seed_runner.resolve_runtime(
        torch_module=_FakeTorch(False),
        logical_cpu_count=24,
        physical_cpu_count=12,
    )
    assert runtime["resolved_device"] == "cpu"
    assert runtime["embedding_batch_size"] == 8
    assert runtime["embedding_threads"] == 12
    assert runtime["ranking_device"] == "cpu"


def test_runtime_auto_selects_cuda_and_scales_batch_from_vram():
    runtime = seed_runner.resolve_runtime(
        torch_module=_FakeTorch(True, 10 * 1024**3),
        logical_cpu_count=32,
        physical_cpu_count=20,
    )
    assert runtime["resolved_device"] == "cuda"
    assert runtime["embedding_batch_size"] == 16
    assert runtime["embedding_threads"] == 16
    assert runtime["gpu"]["name"] == "Synthetic GPU"
    assert runtime["ranking_device"] == "cuda"


def test_runtime_cuda_request_fails_closed_and_overrides_are_respected():
    with pytest.raises(RuntimeError, match="explicitly requested"):
        seed_runner.resolve_runtime("cuda", torch_module=_FakeTorch(False))
    runtime = seed_runner.resolve_runtime(
        "cpu",
        embedding_batch_size=3,
        embedding_threads=2,
        torch_module=_FakeTorch(True, 24 * 1024**3),
    )
    assert runtime["resolved_device"] == "cpu"
    assert runtime["embedding_batch_size"] == 3
    assert runtime["embedding_threads"] == 2


def test_cuda_oom_retries_only_smaller_inference_chunks():
    from rerank.data.build_conformer_sqlite import SeededUniMolBackend

    backend = object.__new__(SeededUniMolBackend)
    backend.device = "cuda"
    backend.batch_size = 8
    backend._dataset_class = list
    backend._trainer = _OomTrainer()
    backend._repr = type("Representation", (), {"model": object()})()
    backend._oom_split_count = 0
    backend._smallest_inference_chunk = 8
    result = backend._infer_payloads([{} for _ in range(5)])
    assert len(result) == 5
    assert backend._oom_split_count == 2
    assert backend._smallest_inference_chunk == 1


def test_sqlite_builder_is_resumable_and_records_non_ok(tmp_path):
    output = tmp_path / "scratch" / "atom_embeddings_seed_42.sqlite"
    status_csv = tmp_path / "embedding_non_ok.csv"
    summary_json = tmp_path / "embedding_summary.json"
    required = ["C", "CC", "CCC"]
    audit = {"required_key_count": 3, "required_keys_sha256": "inventory-audit"}
    backend = FakeBackend()

    summary = build_sqlite_cache(
        required,
        audit,
        output,
        42,
        backend,
        batch_size=2,
        status_csv_path=status_csv,
        summary_json_path=summary_json,
        input_fingerprints={"candidate": {"sha256": "a" * 64}},
    )

    assert summary["complete"] is True
    assert summary["stored_items"] == 3
    assert summary["status_counts"] == {"fallback_2d": 1, "ok": 2}
    assert len(backend.calls) == 2
    with sqlite3.connect(output) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key='complete'"
        ).fetchone()[0] == "1"
        rows = connection.execute(
            "SELECT smiles,n_rows,n_cols,length(data),status FROM embeddings ORDER BY smiles"
        ).fetchall()
    assert rows[0][2] == 512
    assert all(row[3] == row[1] * row[2] * 4 for row in rows)
    with open(status_csv, encoding="utf-8", newline="") as handle:
        non_ok = list(csv.DictReader(handle))
    assert non_ok == [{"smiles": "CC", "status": "fallback_2d", "error": ""}]

    resumed = build_sqlite_cache(
        required,
        audit,
        output,
        42,
        MustNotRunBackend(),
        batch_size=2,
        status_csv_path=status_csv,
        summary_json_path=summary_json,
    )
    assert resumed["stored_items"] == 3


def _write_synthetic_retained_results(layout, monkeypatch):
    monkeypatch.setattr(seed_runner, "EXPECTED_TRAIN_PRODUCTS", 1)
    monkeypatch.setattr(seed_runner, "EXPECTED_VALID_REACTIONS", 1)
    monkeypatch.setattr(seed_runner, "EXPECTED_VALID_COVERED", 1)
    monkeypatch.setattr(seed_runner, "EXPECTED_TEST_REACTIONS", 1)
    monkeypatch.setattr(seed_runner, "EXPECTED_TEST_COVERED", 1)
    layout["features"].mkdir(parents=True, exist_ok=True)
    layout["ranking"].mkdir(parents=True, exist_ok=True)
    feature_row = np.arange(14, dtype=np.float32).reshape(2, 7)
    payload = {
        "train_products": [{"features": feature_row}],
        "eval_features": [feature_row[:1]],
        "audit": {
            "train_products": 1,
            "train_overlap_reactions_excluded": 233,
            "eval_reactions_total": 1,
            "eval_reactions_covered": 1,
        },
    }
    with open(layout["feature_cache"], "wb") as handle:
        pickle.dump(
            {
                "schema_version": STUDY_CACHE_SCHEMA,
                "feature_mode": "3d+prior",
                "payload": payload,
            },
            handle,
        )
    with open(layout["validation_features"], "wb") as handle:
        pickle.dump(
            {
                "schema_version": STUDY_CACHE_SCHEMA,
                "audit": {"reactions_total": 1, "reactions_covered": 1},
                "payload": {"eval_features": [feature_row[:1]]},
            },
            handle,
        )
    metrics = {
        str(seed): {
            "top1": 0.5,
            "top3": 0.7,
            "top5": 0.8,
            "top10": 1.0,
            "mrr": 0.6,
        }
        for seed in seed_runner.TRAINING_SEEDS
    }
    (layout["ranking"] / "per_seed_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (layout["ranking"] / "manifest.json").write_text("{}", encoding="utf-8")
    for seed in seed_runner.TRAINING_SEEDS:
        with open(
            layout["ranking"] / f"eval_seed{seed}.csv",
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["reaction_id"])
            writer.writeheader()
            writer.writerow({"reaction_id": 1})


def test_validation_then_exact_cleanup(tmp_path, monkeypatch):
    layout = seed_runner.build_layout(tmp_path, 42)
    _write_synthetic_retained_results(layout, monkeypatch)
    validation = seed_runner.validate_retained_results(layout)
    assert validation["passed"] is True
    assert validation["feature_rows"] == 3

    layout["scratch"].mkdir(parents=True, exist_ok=True)
    layout["atom_cache"].write_bytes(b"sqlite")
    Path(str(layout["atom_cache"]) + "-journal").write_bytes(b"journal")
    unrelated = layout["root"] / "keep_me.txt"
    unrelated.write_text("keep", encoding="utf-8")

    removed = seed_runner.safe_cleanup_seed_cache(layout, 42)
    assert len(removed) == 2
    assert not layout["atom_cache"].exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_cleanup_refuses_a_cache_outside_seed_root(tmp_path):
    layout = seed_runner.build_layout(tmp_path / "runs", 42)
    layout["root"].mkdir(parents=True, exist_ok=True)
    layout["scratch"] = tmp_path / "outside"
    layout["atom_cache"] = layout["scratch"] / "atom_embeddings_seed_42.sqlite"
    with pytest.raises(RuntimeError, match="outside"):
        seed_runner.safe_cleanup_seed_cache(layout, 42)


def test_ten_clickable_seed_launchers_exist():
    jobs = Path(__file__).resolve().parents[1] / "conformer_jobs"
    launchers = sorted(jobs.glob("run_seed_*.cmd"))
    assert [path.stem for path in launchers] == [
        f"run_seed_{seed}" for seed in range(42, 52)
    ]
    for seed, path in zip(range(42, 52), launchers):
        assert f" {seed}" in path.read_text(encoding="utf-8")


def test_collector_verifies_complete_seed_folders(tmp_path):
    for seed in (42, 43):
        root = tmp_path / f"seed_{seed}"
        root.mkdir()
        manifest = root / "manifest.json"
        summary = root / "result_summary.json"
        manifest.write_text(json.dumps({"conformer_seed": seed}), encoding="utf-8")
        summary.write_text(
            json.dumps(
                {
                    "conformer_label": f"C{seed - 41}",
                    "confirmatory_role": "test",
                    "top1_mean": 0.5,
                    "top1_std": 0.01,
                    "mrr_mean": 0.6,
                    "mrr_std": 0.02,
                }
            ),
            encoding="utf-8",
        )
        completed = {
            "seed": seed,
            "runtime_seconds": 10.0,
            "large_atom_cache_retained": False,
            "retained_file_count": 2,
        }
        (root / "COMPLETED.json").write_text(
            json.dumps(completed), encoding="utf-8"
        )
        checksum_lines = []
        for path in (manifest, summary):
            checksum_lines.append(
                f"{seed_runner._sha256(path)}  {path.name}"
            )
        (root / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )

    rows, audit = collect_runs(tmp_path, seeds=(42, 43))
    assert audit["all_complete"] is True
    assert [row["conformer_seed"] for row in rows] == [42, 43]
    assert all(row["checksum_files_verified"] == 2 for row in rows)

    (tmp_path / "seed_43" / "result_summary.json").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        collect_runs(tmp_path, seeds=(42, 43))
