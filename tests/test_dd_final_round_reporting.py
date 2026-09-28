"""Read-only manuscript reporting checks; no training or bootstrap execution."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/digital_discovery"
REPORT = ROOT / "outputs/dd_final_round/reporting_20260924/summary.json"
if not PAPER.exists() or not REPORT.exists():
    pytest.skip("Private manuscript/reporting inputs not installed", allow_module_level=True)


def test_reporting_hashes_preserve_frozen_inputs():
    data = json.loads(REPORT.read_text())
    for relative, expected in data["source_sha256"].items():
        path = ROOT / "outputs/dd_final_round" / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert data["ci_recomputed"] is False
    assert data["crossed_negative_transfer_intervals"] == {"N3": 3, "N4": 1}


def test_every_frozen_effect_and_interval_is_rendered():
    spec = importlib.util.spec_from_file_location("reporting", ROOT / "scripts/summarize_dd_final_round.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tex = (PAPER / "tables/final_round_n2_n4.tex").read_text()
    data = json.loads(REPORT.read_text())
    for family in ("n2", "n3", "n4"):
        for row in data[family]:
            for key in ("effect", "historical_effect", "expanded_effect", "transfer_loss"):
                if key in row:
                    assert module.pp(row[key]) in tex
            assert module.interval(row, "product") in tex
            assert module.interval(row, "crossed") in tex
    assert tex.count(r"\begin{table*}") == 7


def test_canonical_si_table_numbers_and_main_references():
    si = (PAPER / "supporting_information.tex").read_text()
    si = si.replace(r"\input{tables/final_round_n2_n4}", (PAPER / "tables/final_round_n2_n4.tex").read_text())
    labels = {}
    for index, table in enumerate(re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", si, re.S), 1):
        for label in re.findall(r"\\label\{([^}]+)\}", table):
            labels[label] = index
    assert labels["tab:s7"] == 8
    assert labels["tab:s6"] == 7
    assert labels["tab:n2_cross"] == 17
    assert labels["tab:n3_transfer"] == 19
    assert labels["tab:n4_transfer"] == 20
    assert labels["tab:n1_joint"] == 22
    assert labels["tab:n5_crossed"] == 23
    main = (PAPER / "main.tex").read_text()
    assert "ESI, Table S23" in main
    assert "1-\\mathrm{prior{+}2D\\ Top1}" in main
    assert "not failure of every adapted pipeline" in main
    assert "five of six crossed contrasts" in main


def test_n1_weighted_effect_agrees_with_n5():
    data = json.loads(REPORT.read_text())
    for metric in ("top1", "mrr"):
        selected = [r for r in data["n1"] if r["metric"] == metric]
        assert sum(int(r["n_reactions"]) for r in selected) == 3814
        weighted = sum(int(r["n_reactions"]) * float(r["aizynth_minus_historical_increment"])
                       for r in selected) / 3814
        target = next(r for r in data["n5"] if r["pool"] == "aizynthfinder_only" and r["metric"] == metric)
        assert abs(weighted - float(target["mean_transfer_loss"])) < 1e-8
