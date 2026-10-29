"""N4 launch regressions: real Morgan encoder and historical checksum formats."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows/transfer"))
from run_morgan_comparison import _pool_inputs
from rerank.experiments import run_morgan_comparison as historical
from rerank.experiments.protocol_gate import fingerprint
from rerank.pool_streaming import atomic_json
from rerank.encoder_controls import MorganAtomEncoder, compute_query_control_features


def test_n4_actual_morgan_encoder_smoke():
    encoder = MorganAtomEncoder()
    encoder.initialize()
    matrix = compute_query_control_features(encoder, "CCO", ["CC.O", "C.CO"], 2)
    assert matrix.shape == (2, 3) and np.isfinite(matrix).all()


def test_n4_merged_uses_packaged_filename(tmp_path):
    paths = _pool_inputs(NS(pool_root=tmp_path, index_root=tmp_path,
        products_jsonl="p", source_csv="s", metadata_csv="m"), "merged")
    assert Path(paths["pool_jsonl"]).name == "merged_canonical_union.jsonl"


@pytest.mark.parametrize("prefix", ["", "sha256:"])
def test_historical_fit_accepts_both_hash_formats_not_changed_content(tmp_path, monkeypatch, prefix):
    monkeypatch.setattr(historical, "require_approval", lambda *_: {})
    monkeypatch.setattr(historical, "validate_selection_freeze", lambda *_: {"selection_fingerprint": "frozen"})
    monkeypatch.setattr(historical, "load_selection_bundle", lambda *_: {
        "representation_provenance": {"source_protocol_id": "C-MORGAN-ATOM"}})
    monkeypatch.setattr(historical, "transform_selection_cache", lambda *_: {})
    calls = []
    def fit(view, config, seed, device, checkpoint, normalizer, allowed_seeds):
        calls.append(seed)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic")
        normalizer.write_bytes(b"synthetic")
        return {"best_epoch": 1}
    monkeypatch.setattr(historical, "train_validation_trial", fit)
    bundle = tmp_path / "selection.pkl"
    bundle.write_bytes(b"synthetic-selection")
    record = fingerprint(bundle)
    record["sha256"] = prefix + record["sha256"]
    control = tmp_path / "control.json"
    atomic_json(control, dict(protocol_id="cap10-tuned-encoder-control-v1",
        primary_selection_fingerprint="frozen", prior_transform="raw", control_selection_bundle=record,
        primary_selected_config=dict(index=0, hidden_width=8, dropout=0.0, learning_rate=0.001, margin=0.1)))
    args = NS(seed=47, approval_record="unused", analysis_plan="unused", primary_freeze="unused",
        existing_control_freeze=control, control_selection_bundle=bundle,
        output_root=tmp_path / "models", device="cpu")
    result = historical.fit_one(args)
    assert result["seed"] == 47 and result["test_partition_loaded"] is False
    bundle.write_bytes(b"changed-selection")
    args.output_root = tmp_path / "other_models"
    with pytest.raises(ValueError, match="checksum"):
        historical.fit_one(args)
    assert calls == [47]
