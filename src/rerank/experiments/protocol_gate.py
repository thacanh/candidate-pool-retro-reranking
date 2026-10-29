"""Fail-closed approval and frozen-file checks for isolated DD remote jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: str | Path) -> dict:
    source = Path(path).resolve(strict=True)
    return {"path": str(source), "size_bytes": source.stat().st_size,
            "sha256": sha256(source)}


def require_approval(
    approval_path: str | Path, plan_path: str | Path,
    runner_path: str | Path, protocol_id: str,
) -> dict:
    """Require a recorded authorization bound to the current plan and runner bytes."""

    approval = json.loads(Path(approval_path).read_text(encoding="utf-8"))
    if approval.get("status") != "approved":
        raise PermissionError("Final-round job approval is not approved.")
    if protocol_id not in approval.get("approved_protocol_ids", ()):
        raise PermissionError(f"{protocol_id} is not in the approved protocol set.")
    if protocol_id in approval.get("blocked_protocol_ids", {}):
        raise PermissionError(f"{protocol_id} is still explicitly blocked.")
    if not approval.get("approval_date"):
        raise PermissionError("Authorization date is missing.")
    if approval.get("authorization_basis") == "repository_owner_explicit_override":
        # Each isolated job needs its own explicit owner instruction. A reported
        # verbal waiver is never represented as a signed supervisor approval.
        owner_jobs = {"dd-n2-cross-inference-v1", "dd-n3-pool-specific-selection-v1",
                      "dd-n4-morgan-atom-transfer-v1"}
        quotes = approval.get("authorization_quotes", {})
        if (protocol_id not in owner_jobs
                or approval.get("supervisor") is not None
                or approval.get("supervisor_signature_present") is not False
                or not isinstance(quotes, dict)
                or not quotes.get(protocol_id)):
            raise PermissionError("Invalid repository-owner override record for this job.")
    elif not approval.get("supervisor"):
        raise PermissionError("Supervisor identity is missing.")
    if not approval.get("analysis_plan_commit"):
        raise PermissionError("Committed analysis-plan revision is missing.")
    if approval.get("analysis_plan_sha256") != sha256(plan_path):
        raise PermissionError("Approved analysis-plan bytes differ from the current plan.")
    recorded_runner = approval.get("runner_sha256", {}).get(protocol_id)
    expected_runner = (recorded_runner.get(Path(runner_path).name)
                       if isinstance(recorded_runner, dict) else recorded_runner)
    if expected_runner != sha256(runner_path):
        raise PermissionError("Approved runner bytes differ from the current runner.")
    return approval


def resolve_frozen(record: dict, local_dir: str | Path) -> Path:
    """Relocate an absolute-path frozen artifact by basename, checking SHA/size."""

    source = str(record["path"])
    recorded = Path(source)
    basename = PureWindowsPath(source).name if "\\" in source else recorded.name
    candidate = recorded if recorded.is_file() else Path(local_dir) / basename
    if not candidate.is_file():
        raise FileNotFoundError(f"Frozen artifact is absent: {basename}")
    if candidate.stat().st_size != int(record["size_bytes"]):
        raise ValueError(f"Frozen artifact size mismatch: {candidate}")
    if sha256(candidate) != str(record["sha256"]):
        raise ValueError(f"Frozen artifact SHA-256 mismatch: {candidate}")
    return candidate.resolve()
