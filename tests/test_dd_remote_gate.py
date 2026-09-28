from __future__ import annotations

import json
from pathlib import Path

import pytest

from rerank.experiments.dd_remote_gate import require_approval, resolve_frozen, sha256


def test_windows_absolute_path_relocates_only_with_matching_hash(tmp_path: Path) -> None:
    local = tmp_path / "shards"
    local.mkdir()
    artifact = local / "trial.json"
    artifact.write_bytes(b"immutable")
    record = {"path": r"C:\old\frozen\trial.json", "size_bytes": 9,
              "sha256": sha256(artifact)}
    assert resolve_frozen(record, local) == artifact.resolve()
    record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        resolve_frozen(record, local)


def test_scientific_job_cannot_run_without_exact_approved_runner(tmp_path: Path) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("approved plan", encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("# runner", encoding="utf-8")
    approval = tmp_path / "approval.json"
    protocol = "dd-n2-cross-inference-v1"
    payload = {"status": "approved", "approved_protocol_ids": [protocol],
               "supervisor": "Supervisor", "approval_date": "2026-09-23",
               "analysis_plan_commit": "abc123", "analysis_plan_sha256": sha256(plan),
               "runner_sha256": {protocol: sha256(runner)}}
    approval.write_text(json.dumps(payload), encoding="utf-8")
    assert require_approval(approval, plan, runner, protocol)["status"] == "approved"
    runner.write_text("# changed runner", encoding="utf-8")
    with pytest.raises(PermissionError, match="runner"):
        require_approval(approval, plan, runner, protocol)


def test_owner_override_requires_separate_quote_and_exact_runner(tmp_path: Path) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("committed plan", encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("# frozen N2", encoding="utf-8")
    approval = tmp_path / "approval.json"
    n2 = "dd-n2-cross-inference-v1"
    payload = {
        "status": "approved", "approved_protocol_ids": [n2],
        "authorization_basis": "repository_owner_explicit_override",
        "authorization_quotes": {n2: "thầy bảo k cần duyệt gì nữa rồi"},
        "supervisor": None, "supervisor_signature_present": False,
        "approval_date": "2026-09-23", "analysis_plan_commit": "abc123",
        "analysis_plan_sha256": sha256(plan),
        "runner_sha256": {n2: sha256(runner)},
    }
    approval.write_text(json.dumps(payload), encoding="utf-8")
    assert require_approval(approval, plan, runner, n2)["authorization_basis"] == \
        "repository_owner_explicit_override"
    payload["approved_protocol_ids"].append("dd-n3-pool-specific-selection-v1")
    payload["runner_sha256"]["dd-n3-pool-specific-selection-v1"] = sha256(runner)
    approval.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="override"):
        require_approval(approval, plan, runner, "dd-n3-pool-specific-selection-v1")
    payload["authorization_quotes"]["dd-n3-pool-specific-selection-v1"] = "giờ chạy N3 N4 đã"
    approval.write_text(json.dumps(payload), encoding="utf-8")
    assert require_approval(approval, plan, runner, "dd-n3-pool-specific-selection-v1")[
        "supervisor_signature_present"] is False
    runner.write_text("# changed runner", encoding="utf-8")
    with pytest.raises(PermissionError, match="runner"):
        require_approval(approval, plan, runner, "dd-n3-pool-specific-selection-v1")
