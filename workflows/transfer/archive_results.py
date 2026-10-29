"""Archive one *completed and approved* DD final-round job before the next job.

This is a transport/verification utility, not a scientific runner. It does not
authorize an experiment or infer that a job completed from a progress counter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


STAGES = {"N2", "N3_S1", "N3_S2", "N3_S3", "N4"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within(path: Path, parent: Path) -> Path:
    resolved, root = path.resolve(), parent.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path is outside the requested directory: {path}") from exc
    return resolved


def _files(root: Path) -> list[Path]:
    files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in dirs + names:
            if (base / name).is_symlink():
                raise ValueError(f"Symlink is not permitted in a frozen result: {base / name}")
        files.extend(base / name for name in names)
    return sorted(files)


def _check_approval(path: Path, protocol_id: str) -> dict:
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval.get("status") != "approved":
        raise PermissionError("Approval record is not approved.")
    if protocol_id not in approval.get("approved_protocol_ids", []):
        raise PermissionError(f"Protocol {protocol_id} is absent from the approved IDs.")
    if protocol_id in approval.get("blocked_protocol_ids", {}):
        raise PermissionError(f"Protocol {protocol_id} is still blocked.")
    return approval


def _verify_members(archive: Path, expected: dict[str, dict]) -> None:
    with zipfile.ZipFile(archive) as bundle:
        if set(bundle.namelist()) != set(expected):
            raise ValueError("ZIP member list differs from the frozen source list.")
        for name, record in expected.items():
            digest = hashlib.sha256()
            size = 0
            with bundle.open(name) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            if size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"ZIP member checksum differs: {name}")


def archive_stage(args: argparse.Namespace) -> dict:
    if args.stage not in STAGES:
        raise ValueError(f"Unknown stage {args.stage!r}.")
    if not re.fullmatch(r"dd-n[234]-[a-z0-9-]+-v[0-9]+", args.protocol_id):
        raise ValueError("Expected an explicit DD N2/N3/N4 protocol ID.")
    result_dir = Path(args.result_dir).resolve(strict=True)
    if not result_dir.is_dir():
        raise ValueError("Result directory is not a directory.")
    manifest_path = _within(result_dir / "manifest.json", result_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Scientific manifest does not say status=complete.")
    if manifest.get("protocol_id") != args.protocol_id:
        raise ValueError("Scientific manifest has a different protocol ID.")
    if not manifest.get("comparator") or not manifest.get("single_intended_change"):
        raise ValueError("Scientific manifest lacks comparator or single intended change.")
    approval_path = Path(args.approval_record).resolve(strict=True)
    _check_approval(approval_path, args.protocol_id)

    if not args.required:
        raise ValueError("At least one required scientific output must be named.")
    required = [Path(item) for item in args.required]
    for relative in required:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Required result path must be relative: {relative}")
        path = _within(result_dir / relative, result_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Required result is missing: {path}")

    archive_dir = Path(args.archive_dir).resolve()
    try:
        archive_dir.relative_to(result_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Archive directory must not be inside the result directory.")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / f"{args.stage}.zip"
    receipt = archive_dir / f"{args.stage}.receipt.json"
    if archive.exists() or receipt.exists():
        raise FileExistsError("Stage archive/receipt already exists; refusing overwrite.")

    sources = {f"{args.stage}/results/{path.relative_to(result_dir).as_posix()}": path
               for path in _files(result_dir)}
    sources[f"{args.stage}/provenance/approval_record.json"] = approval_path
    if args.log_file:
        log = Path(args.log_file).resolve(strict=True)
        if not log.is_file() or log.is_symlink():
            raise ValueError("Log must be an ordinary file.")
        sources[f"{args.stage}/logs/{log.name}"] = log
    if not sources:
        raise ValueError("No files to archive.")

    member_hashes = {
        name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for name, path in sources.items()
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.stage}.", suffix=".partial.zip", dir=archive_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True, compresslevel=6) as bundle:
            for name, path in sources.items():
                bundle.write(path, name)
        _verify_members(temporary, member_hashes)
        # A hard link creates the final name atomically and fails if it exists.
        os.link(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)

    data = {
        "schema_version": 1,
        "status": "complete",
        "stage": args.stage,
        "protocol_id": args.protocol_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": archive.name,
        "archive_sha256": _sha256(archive),
        "archive_size_bytes": archive.stat().st_size,
        "approval_record_sha256": _sha256(approval_path),
        "members": member_hashes,
    }
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return data


def verify_stage(args: argparse.Namespace) -> dict:
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise ValueError("Receipt is not complete.")
    archive = Path(args.receipt).resolve().parent / receipt["archive"]
    if archive.stat().st_size != receipt["archive_size_bytes"]:
        raise ValueError("ZIP byte size differs from receipt.")
    if _sha256(archive) != receipt["archive_sha256"]:
        raise ValueError("ZIP SHA-256 differs from receipt.")
    _verify_members(archive, receipt["members"])
    return {"status": "verified", "stage": receipt["stage"],
            "archive": str(archive), "sha256": receipt["archive_sha256"],
            "member_count": len(receipt["members"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    archive = commands.add_parser("archive")
    archive.add_argument("--stage", required=True)
    archive.add_argument("--protocol-id", required=True)
    archive.add_argument("--result-dir", required=True)
    archive.add_argument("--archive-dir", required=True)
    archive.add_argument("--approval-record", required=True)
    archive.add_argument("--required", action="append", default=[])
    archive.add_argument("--log-file")
    verify = commands.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    args = parser.parse_args()
    result = archive_stage(args) if args.command == "archive" else verify_stage(args)
    if args.command == "archive":
        result = {"status": result["status"], "stage": result["stage"],
                  "archive": result["archive"], "sha256": result["archive_sha256"],
                  "member_count": len(result["members"])}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
