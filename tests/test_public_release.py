from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MARKER = ROOT / "outputs/historical_anchor/numerical_freeze_v2/manifest.json"


def _require_public_release() -> None:
    if not PUBLIC_MARKER.is_file():
        pytest.skip("clean public-release layout is not present in the research worktree")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_public_artifact_aliases_preserve_bytes() -> None:
    _require_public_release()
    mapping = json.loads(
        (ROOT / "data/provenance/release_artifact_map.json").read_text(encoding="utf-8")
    )
    for row in mapping["artifacts"]:
        path = ROOT / row["public_path"]
        assert path.is_file(), row["public_path"]
        assert path.stat().st_size == row["size_bytes"]
        assert _sha256(path) == row["sha256"]


def test_compact_operational_table_is_complete() -> None:
    _require_public_release()
    path = ROOT / "outputs/transfer_analysis/figure_inputs/operational_performance.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 12
    assert {row["pool"] for row in rows} == {
        "historical_cap10",
        "aizynthfinder_only",
        "localretro_only",
        "merged",
    }
    assert {row["system"] for row in rows} == {
        "candidate_prior",
        "prior_2d",
        "prior_2d_unimol",
    }
    for row in rows:
        within = float(row["within_pool_top1"])
        end_to_end = float(row["end_to_end_top1"])
        assert 0.0 <= end_to_end <= within <= 1.0


def test_release_has_no_heavy_payload_or_personal_windows_path() -> None:
    _require_public_release()
    assert not (ROOT / "paper").exists()
    forbidden_suffixes = {".sqlite", ".pkl", ".pt", ".pth"}
    personal_path = re.compile(r"C:\\Users\\[^\\\s]+", re.IGNORECASE)
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        assert path.suffix.lower() not in forbidden_suffixes, path
        assert path.stat().st_size < 100_000_000, path
        if path.suffix.lower() in {".md", ".txt", ".json", ".py", ".tex", ".csv"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            assert not personal_path.search(text), path


def test_transport_checksums_are_valid() -> None:
    _require_public_release()
    lines = (ROOT / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        assert path.is_file(), relative
        assert _sha256(path) == expected, relative
