from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.dd_archive_stage import archive_stage, verify_stage


PROTOCOL = "dd-n2-cross-inference-v1"


def _args(tmp_path: Path) -> Namespace:
    result = tmp_path / "N2"
    result.mkdir()
    (result / "manifest.json").write_text(
        json.dumps({
            "status": "complete",
            "protocol_id": PROTOCOL,
            "comparator": "frozen diagonal cells",
            "single_intended_change": "frozen model on the other pool",
        }), encoding="utf-8"
    )
    (result / "effects.csv").write_text("effect\n0\n", encoding="utf-8")
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({"status": "approved", "approved_protocol_ids": [PROTOCOL]}),
                        encoding="utf-8")
    return Namespace(stage="N2", protocol_id=PROTOCOL, result_dir=str(result),
                     archive_dir=str(tmp_path / "archives"),
                     approval_record=str(approval), required=["effects.csv"],
                     log_file=None)


def test_complete_stage_is_archived_then_verified(tmp_path: Path) -> None:
    args = _args(tmp_path)
    receipt = archive_stage(args)
    assert receipt["status"] == "complete"
    checked = verify_stage(Namespace(receipt=str(tmp_path / "archives/N2.receipt.json")))
    assert checked["status"] == "verified"
    assert checked["member_count"] == 3
    with pytest.raises(FileExistsError):
        archive_stage(args)


def test_incomplete_or_unapproved_stage_cannot_be_archived(tmp_path: Path) -> None:
    args = _args(tmp_path)
    manifest_path = Path(args.result_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="status=complete"):
        archive_stage(args)

    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    Path(args.approval_record).write_text(json.dumps({"status": "approved",
        "blocked_protocol_ids": {PROTOCOL: "pending"},
        "approved_protocol_ids": [PROTOCOL]}), encoding="utf-8")
    with pytest.raises(PermissionError, match="blocked"):
        archive_stage(args)


def test_zip_corruption_is_detected(tmp_path: Path) -> None:
    args = _args(tmp_path)
    archive_stage(args)
    archive = tmp_path / "archives/N2.zip"
    with archive.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="byte size"):
        verify_stage(Namespace(receipt=str(tmp_path / "archives/N2.receipt.json")))
