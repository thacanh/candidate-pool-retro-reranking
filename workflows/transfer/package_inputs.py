"""Inventory, pack and verify three mutually independent DD remote inputs.

No experiment is run. The bundle retains original frozen file bytes and
relative repository paths; absolute paths inside historical JSON are resolved
by hash-checked relocation in the controlled runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath


JOBS = ("N2", "N3", "N4")
COMMON = (
    "docs/analysis_plan.md",
    "workflows/transfer/requirements-cpu.txt",
    "docs/dd_remote_approval.json",
    "data/uspto_smiles.csv",
    "data/uspto_reaction_metadata.csv",
    "outputs/jcheminform_revision/candidate_pools/ws_e_three_pools_filtered_v2/products.jsonl",
    "outputs/digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_transfer_loss_reactions.csv",
)
PRIMARY = "outputs/jcheminform_revision/tuned_primary/conformer_seed_42"
FEATURES = "outputs/jcheminform_revision/ws_e_streaming/pool_features_canonical_identity"
K1 = "outputs/digital_discovery_round_jk/k1"
POOLS = "outputs/jcheminform_revision/candidate_pools/ws_e_three_pools_filtered_v2"
INDEX = "outputs/jcheminform_revision/ws_e_streaming/index"
MORGAN = "outputs/jcheminform_revision/encoder_controls/morgan_atom"
COMMON_COHORT = (
    "outputs/digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_transfer_loss_reactions.csv"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _include(root: Path, target: str) -> list[Path]:
    path = root / target
    if path.is_symlink():
        raise ValueError(f"Symlink is not allowed in a remote input: {path}")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    for directory, dirs, names in os.walk(path, followlinks=False):
        for name in dirs + names:
            if (Path(directory) / name).is_symlink():
                raise ValueError(f"Symlink is not allowed in a remote input: {directory}/{name}")
        files.extend(Path(directory) / name for name in names if not name.endswith(".pyc"))
    return files


def selected_files(root: Path, job: str) -> list[Path]:
    if job not in JOBS:
        raise ValueError(f"Unknown DD remote job {job!r}.")
    paths = list(COMMON)
    paths.extend((f"{PRIMARY}/g1_20seed", f"{PRIMARY}/freeze/model_selection.json"))
    if job in {"N2", "N3"}:
        paths.append(
            "outputs/jcheminform_revision/conformers/seed_42/features/official_3d_prior_schema1.pkl"
        )
    if job == "N2":
        aiz = f"{K1}/aizynthfinder_only"
        paths.extend(
            (
                f"{aiz}/models",
                f"{aiz}/test_results",
                f"{aiz}/selection/manifest.json",
                f"{FEATURES}/aizynthfinder_only",
            )
        )
    if job == "N3":
        paths.extend((f"{PRIMARY}/selection/train_valid_only.pkl", K1, FEATURES, POOLS, INDEX))
        paths.append("docs/reproducibility.md")
    if job == "N4":
        paths.extend((FEATURES, POOLS, INDEX, MORGAN))
        paths.append("docs/reproducibility.md")
    files = []
    for target in paths:
        files.extend(_include(root, target))
    files.extend(path for path in (root / "src" / "rerank").rglob("*.py") if not path.is_symlink())
    scripts = (
        "workflows/transfer/package_inputs.py",
        "workflows/transfer/archive_results.py",
        "workflows/transfer/record_execution_plan.py",
        "workflows/transfer/setup.sh",
        "workflows/transfer/manage_jobs.sh",
        "workflows/transfer/check_prior_inputs.py",
        "workflows/transfer/run_prior_comparison.py",
        "workflows/transfer/check_morgan_inputs.py",
        "workflows/transfer/run_cross_inference.sh",
        "workflows/transfer/run_prior_selection.sh",
        "workflows/transfer/run_morgan_comparison.sh",
        "workflows/transfer/run_prior_selection.py",
        "workflows/transfer/run_morgan_comparison.py",
    )
    for target in scripts:
        files.extend(_include(root, target))
    files.extend(path for path in (root / "tests").glob("test_*.py"))
    files.extend(_include(root, "tests/test_pool_streaming.py"))
    unique = {path.resolve(): path for path in files}
    result = sorted(unique.values())
    if not result or any(not path.resolve().is_relative_to(root.resolve()) for path in result):
        raise ValueError("Remote package input escaped repository root.")
    return result


def inventory(root: Path, job: str) -> tuple[list[Path], dict]:
    files = selected_files(root, job)
    manifest = {
        "schema_version": 1,
        "record_kind": "dd_isolated_remote_input_bundle",
        "job": job,
        "scientific_experiment_executed": False,
        "approval_note": "The included job-specific authorization record is checked against the exact plan and runner bytes at execution; it does not claim a supervisor signature.",
        "files": {
            path.relative_to(root).as_posix(): {
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
            for path in files
        },
    }
    return files, manifest


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if name.startswith("/") or not name or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe ZIP member: {name!r}")


def pack(root: Path, job: str, output: Path) -> dict:
    files, manifest = inventory(root, job)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing remote ZIP: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"Existing partial ZIP requires inspection: {partial}")
    try:
        with zipfile.ZipFile(
            partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
        ) as archive:
            archive.writestr(
                "BUNDLE_INPUTS.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n"
            )
            for path in files:
                archive.write(path, path.relative_to(root).as_posix())
        verify(partial)
        os.link(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "job": job,
        "archive": str(output.resolve()),
        "sha256": _sha(output),
        "size_bytes": output.stat().st_size,
        "files": len(files),
    }


def verify(archive_path: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members.")
        for name in names:
            _safe_member(name)
        manifest = json.loads(archive.read("BUNDLE_INPUTS.json"))
        if set(names) != set(manifest["files"]) | {"BUNDLE_INPUTS.json"}:
            raise ValueError("Archive member list differs from its input manifest.")
        for name, record in manifest["files"].items():
            size = 0
            digest = hashlib.sha256()
            with archive.open(name) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    size += len(block)
                    digest.update(block)
            if size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Remote ZIP member differs: {name}")
    return {
        "status": "verified",
        "job": manifest["job"],
        "files": len(manifest["files"]),
        "archive_sha256": _sha(archive_path),
    }


def verify_extracted(root: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        _safe_member(name)
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Missing, linked, or wrong-sized extracted input: {name}")
        if _sha(path) != record["sha256"]:
            raise ValueError(f"Extracted input SHA-256 differs: {name}")
    return {"status": "verified", "job": manifest["job"], "files": len(manifest["files"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "pack"):
        action = commands.add_parser(name)
        action.add_argument("--root", default=".")
        action.add_argument("--job", choices=JOBS, required=True)
        if name == "pack":
            action.add_argument("--output", required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--archive", required=True)
    extracted = commands.add_parser("verify-extracted")
    extracted.add_argument("--root", required=True)
    extracted.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.command == "inventory":
        _, body = inventory(Path(args.root).resolve(), args.job)
        answer = {
            "job": args.job,
            "files": len(body["files"]),
            "size_bytes": sum(row["size_bytes"] for row in body["files"].values()),
        }
    elif args.command == "pack":
        answer = pack(Path(args.root).resolve(), args.job, Path(args.output).resolve())
    elif args.command == "verify":
        answer = verify(Path(args.archive).resolve())
    else:
        answer = verify_extracted(Path(args.root).resolve(), Path(args.manifest).resolve())
    print(json.dumps(answer, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
