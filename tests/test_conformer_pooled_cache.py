import json
import sqlite3

import numpy as np

from rerank.data.build_conformer_pooled_sqlite import CACHE_KIND, build


class FakeBackend:
    def metadata(self):
        return {
            "name": "fake", "seed": 42, "device": "cpu", "cuda_device": None,
            "unimol_tools_version": "0.1.3", "rdkit_version": "2025.9.6",
            "torch_version": "test", "torch_cuda_version": None,
            "checkpoint": {"sha256": "checkpoint"},
            "dictionary": {"sha256": "dictionary"},
            "conformer": {"seed": 42},
        }

    def encode_batch(self, smiles):
        arrays = [np.arange((index + 1) * 512, dtype=np.float32).reshape(index + 1, 512) for index, _ in enumerate(smiles)]
        return arrays, ["ok"] * len(smiles), [None] * len(smiles)


def test_pooled_rows_retain_mean_and_atom_count(tmp_path, monkeypatch):
    from rerank.data.build_conformer_pooled_sqlite import _open_database, _write_metadata

    required = ["C", "CC"]
    path = tmp_path / "pooled.sqlite"
    connection = _open_database(path, 42, required)
    backend = FakeBackend()
    arrays, statuses, errors = backend.encode_batch(required)
    for smiles, array, status, error in zip(required, arrays, statuses, errors):
        pooled = array.mean(axis=0, dtype=np.float64).astype(np.float32)
        connection.execute(
            "INSERT INTO embeddings(smiles,atom_count,data,status,error) VALUES(?,?,?,?,?)",
            (smiles, len(array), pooled.tobytes(), status, error),
        )
    _write_metadata(connection, "complete", 1)
    _write_metadata(connection, "scientific_complete", True)
    connection.commit()
    rows = connection.execute("SELECT smiles,atom_count,data FROM embeddings ORDER BY smiles").fetchall()
    connection.close()
    assert [row[1] for row in rows] == [1, 2]
    np.testing.assert_allclose(np.frombuffer(rows[1][2], dtype=np.float32), arrays[1].mean(axis=0))

