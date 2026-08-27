"""Build and verify the clean, allowlist-based public research release.

This is a packaging workflow. It copies frozen artifacts without changing
their bytes and derives one compact plotting table from already-frozen test
manifests. It never runs a scientific model or modifies the research outputs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable


SCHEMA_VERSION = 1
MAX_GIT_FILE_BYTES = 100_000_000

ROOT_FILES = (
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "README.md",
    "constraints-revision-py310.txt",
    "environment-revision.yml",
    "pyproject.toml",
    "pytest.ini",
    "requirements-revision.txt",
)
DOC_FILES = (
    "docs/analysis_plan.md",
    "docs/dataset_source_audit.md",
    "docs/public_release.md",
    "docs/round_jk_approval.json",
)
PROVENANCE_FILES = (
    "data/candidate_generator_provenance.json",
    "data/grover_control_decision.json",
    "data/revision_external_assets.json",
    "data/uspto_provenance.json",
    "paper/digital_discovery/release_artifact_map.json",
)
SOURCE_TREES = (
    "src/rerank",
    "tests",
    "aizynth_jobs",
    "chemformer_jobs",
    "conformer_jobs",
)
REANALYSIS_DIRS = (
    "j1_j2_filtered_v2",
    "j3_j4",
    "k1",
    "k2_full_reporting_v2",
    "l1",
    "m1_heterogeneity",
    "m2_pool_shift",
    "m2b_transfer_inference",
)
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git", "render"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".aux", ".bbl", ".blg", ".log", ".out"}
FORBIDDEN_PAYLOAD_SUFFIXES = {".sqlite", ".pkl", ".pt", ".pth"}
TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".cmd",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_PATTERNS = (
    re.compile(r"C:\\Users\\[^\\\s]+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgh[oprsu]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in {".git", ".pytest_cache", "__pycache__"} for part in relative.parts):
            continue
        files.append(relative)
    return sorted(files)


def copy_file(repo: Path, output: Path, source_rel: str, destination_rel: str | None = None) -> None:
    source = repo / source_rel
    if not source.is_file():
        raise FileNotFoundError(f"Required release source is absent: {source_rel}")
    destination = output / (destination_rel or source_rel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(repo: Path, output: Path, source_rel: str, destination_rel: str | None = None) -> None:
    source_root = repo / source_rel
    if not source_root.is_dir():
        raise FileNotFoundError(f"Required release directory is absent: {source_rel}")
    destination_root = output / (destination_rel or source_rel)
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source.relative_to(source_root)
        if any(part in SKIP_PARTS for part in relative.parts) or source.suffix.lower() in SKIP_SUFFIXES:
            continue
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _mean_seed_metric(manifest: dict, arm: str, scope: str, metric: str) -> float:
    per_seed = manifest["per_seed_metrics"][arm]
    expected = {str(seed) for seed in range(42, 62)}
    if set(per_seed) != expected:
        raise ValueError(f"Expected seeds 42--61 for {arm}; found {sorted(per_seed)}")
    return sum(float(per_seed[seed][scope][metric]) for seed in sorted(per_seed)) / len(per_seed)


def build_operational_table(repo: Path, output: Path) -> None:
    historical_rel = (
        "outputs/jcheminform_revision/tuned_primary/conformer_seed_42/"
        "g1_20seed/test_results/manifest.json"
    )
    pool_sources = {
        pool: f"outputs/digital_discovery_round_jk/k1/{pool}/test_results/manifest.json"
        for pool in ("aizynthfinder_only", "localretro_only", "merged")
    }
    source_paths = [historical_rel, *pool_sources.values()]
    for relative in source_paths:
        if not (repo / relative).is_file():
            raise FileNotFoundError(f"Missing frozen figure input: {relative}")

    historical = json.loads((repo / historical_rel).read_text(encoding="utf-8"))
    per_seed = historical["per_seed_metrics"]
    expected = {str(seed) for seed in range(42, 62)}
    if set(per_seed["baseline"]) != expected or set(per_seed["augmented"]) != expected:
        raise ValueError("Historical operational manifest does not contain paired seeds 42--61")
    coverage = 3985 / 5004
    prior = float(next(iter(per_seed["baseline"].values()))["baseline_top1"])
    baseline = sum(float(row["top1"]) for row in per_seed["baseline"].values()) / 20
    augmented = sum(float(row["top1"]) for row in per_seed["augmented"].values()) / 20

    rows: list[dict[str, str | float]] = []
    for system, value in (
        ("candidate_prior", prior),
        ("prior_2d", baseline),
        ("prior_2d_unimol", augmented),
    ):
        rows.append(
            {
                "pool": "historical_cap10",
                "system": system,
                "within_pool_top1": value,
                "end_to_end_top1": value * coverage,
            }
        )

    for pool, relative in pool_sources.items():
        manifest = json.loads((repo / relative).read_text(encoding="utf-8"))
        values = (
            (
                "candidate_prior",
                float(manifest["prior_metrics"]["within_pool"]["top1"]),
                float(manifest["prior_metrics"]["end_to_end"]["top1"]),
            ),
            (
                "prior_2d",
                _mean_seed_metric(manifest, "baseline", "within_pool", "top1"),
                _mean_seed_metric(manifest, "baseline", "end_to_end", "top1"),
            ),
            (
                "prior_2d_unimol",
                _mean_seed_metric(manifest, "augmented", "within_pool", "top1"),
                _mean_seed_metric(manifest, "augmented", "end_to_end", "top1"),
            ),
        )
        for system, within, end_to_end in values:
            rows.append(
                {
                    "pool": pool,
                    "system": system,
                    "within_pool_top1": within,
                    "end_to_end_top1": end_to_end,
                }
            )

    target_dir = output / "outputs/transfer_analysis/figure_inputs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "operational_performance.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("pool", "system", "within_pool_top1", "end_to_end_top1"),
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": 1,
        "protocol_id": "dd-public-figure-inputs-v1",
        "single_intended_change": "model-free public packaging of frozen figure inputs",
        "scientific_computation": False,
        "row_count": len(rows),
        "output": {
            "path": "outputs/transfer_analysis/figure_inputs/operational_performance.csv",
            "size_bytes": target.stat().st_size,
            "sha256": sha256(target),
        },
        "sources": [
            {
                "research_worktree_path": relative,
                "size_bytes": (repo / relative).stat().st_size,
                "sha256": sha256(repo / relative),
            }
            for relative in source_paths
        ],
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_outputs_readme(output: Path) -> None:
    text = """# Frozen numerical outputs

This directory contains only compact, model-free numerical artifacts used by
the manuscript and figure code.

- `historical_anchor/` is a read-only numerical anchor from the same study.
- `transfer_analysis/` contains the candidate-pool transfer analyses.

Raw reactions, candidate-level prediction archives, checkpoints, and embedding
caches are intentionally excluded from Git. See `docs/public_release.md`.
"""
    path = output / "outputs/README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_release_text(output: Path) -> None:
    """Match Git EOL policy before checksums are written.

    Files below ``outputs/`` are excluded because their published hashes must
    remain byte-identical to the frozen research artifacts.
    """
    for relative in relative_files(output):
        if relative.parts and relative.parts[0] == "outputs":
            continue
        path = output / relative
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if path.suffix.lower() in {".cmd", ".ps1"}:
            text = text.replace("\n", "\r\n")
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)


def git_state(repo: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo}", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--short")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": "unknown", "branch": "unknown", "dirty": True}


def scan_release(output: Path) -> None:
    failures: list[str] = []
    for relative in relative_files(output):
        path = output / relative
        if path.suffix.lower() in FORBIDDEN_PAYLOAD_SUFFIXES:
            failures.append(f"forbidden heavyweight suffix: {relative.as_posix()}")
        if path.stat().st_size >= MAX_GIT_FILE_BYTES:
            failures.append(f"file reaches GitHub 100 MB limit: {relative.as_posix()}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                failures.append(
                    f"machine-local path or credential-like text in {relative.as_posix()}: {pattern.pattern}"
                )
    if failures:
        raise RuntimeError("Public release safety scan failed:\n- " + "\n- ".join(failures))


def verify_mapped_artifacts(output: Path) -> None:
    mapping_path = output / "data/provenance/release_artifact_map.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    for row in mapping["artifacts"]:
        public_path = output / row["public_path"]
        if not public_path.is_file():
            raise FileNotFoundError(f"Mapped public artifact is absent: {row['public_path']}")
        if public_path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Mapped artifact size differs: {row['public_path']}")
        if sha256(public_path) != row["sha256"]:
            raise ValueError(f"Mapped artifact hash differs: {row['public_path']}")


def write_release_ledgers(repo: Path, output: Path) -> None:
    payload = []
    for relative in relative_files(output):
        if relative.as_posix() in {"checksums.sha256", "release_manifest.json"}:
            continue
        path = output / relative
        payload.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_kind": "clean-public-code-and-compact-numerical-release",
        "publication_target": "Digital Discovery",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git": git_state(repo),
        "payload_file_count": len(payload),
        "payload_size_bytes": sum(row["size_bytes"] for row in payload),
        "excluded_classes": [
            "manuscript sources, journal template assets, and rendered paper figures",
            "raw reaction datasets",
            "third-party repositories and model weights",
            "SQLite, PKL, PT/PTH, and candidate-level caches",
            "full predictions and validation checkpoints",
            "logs, archives, scratch files, and abandoned drafts",
        ],
        "files": payload,
    }
    manifest_path = output / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for relative in relative_files(output):
        if relative.as_posix() == "checksums.sha256":
            continue
        rows.append(f"{sha256(output / relative)}  {relative.as_posix()}")
    (output / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_release(output: Path) -> dict[str, int]:
    output = output.resolve()
    checksum_path = output / "checksums.sha256"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing release checksum ledger: {checksum_path}")
    checked = 0
    bytes_checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = output / relative
        if not path.is_file():
            raise FileNotFoundError(f"Release file is absent: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Release checksum differs: {relative}")
        checked += 1
        bytes_checked += path.stat().st_size
    verify_mapped_artifacts(output)
    scan_release(output)
    return {"verified_files": checked, "verified_bytes": bytes_checked}


def build_release(repo: Path, output: Path) -> dict[str, int]:
    repo = repo.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty public release directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    for relative in ROOT_FILES + DOC_FILES:
        copy_file(repo, output, relative)
    for relative in PROVENANCE_FILES:
        copy_file(repo, output, relative, f"data/provenance/{Path(relative).name}")
    for relative in SOURCE_TREES:
        copy_tree(repo, output, relative)

    copy_tree(
        repo,
        output,
        "outputs/jcheminform_revision/numerical_freeze_v2",
        "outputs/historical_anchor/numerical_freeze_v2",
    )
    for directory in REANALYSIS_DIRS:
        copy_tree(
            repo,
            output,
            f"outputs/digital_discovery_round_jk/reanalysis/{directory}",
            f"outputs/transfer_analysis/{directory}",
        )

    build_operational_table(repo, output)
    write_outputs_readme(output)
    normalize_release_text(output)
    verify_mapped_artifacts(output)
    scan_release(output)
    write_release_ledgers(repo, output)
    result = verify_release(output)
    result["payload_size_bytes"] = json.loads(
        (output / "release_manifest.json").read_text(encoding="utf-8")
    )["payload_size_bytes"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Research worktree root (default: inferred from this module)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release/public_repository"),
        help="New clean release directory",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify an existing release instead of building one",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        result = verify_release(args.verify)
    else:
        output = args.output
        if not output.is_absolute():
            output = args.repo_root / output
        result = build_release(args.repo_root, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
