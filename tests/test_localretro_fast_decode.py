import json
from types import SimpleNamespace

import pytest

import rerank.data.decode_localretro_resumable as decoder


def test_fast_decoder_calls_template_once_per_proposal_and_keeps_upstream_order() -> None:
    calls: list[tuple] = []

    def decode_localtemplate(molecule, site, template, info):
        calls.append((site, template, info))
        return "C.O"

    fake = SimpleNamespace(
        get_edit_site=lambda molecule: ([0, 1], [(0, 1), (1, 0)]),
        get_idx_map=lambda molecule: {0: 0, 1: 1},
        decode_localtemplate=decode_localtemplate,
    )
    decoder._TD = fake
    decoder._RAW = {
        0: (
            "CC",
            [
                "(a, 0, 1, 0.9)",
                "(a, 0, 1, 0.9)",
            ],
        )
    }
    decoder._ATOM_TEMPLATES = {1: "[C:1]>>[C:1]"}
    decoder._BOND_TEMPLATES = {}
    decoder._TEMPLATE_INFOS = {
        "[C:1]>>[C:1]": {
            "edit_site": {},
            "change_H": {},
            "change_C": {},
            "change_S": {},
        }
    }

    test_id, predictions = decoder._decode_fast(0)
    assert test_id == 0
    assert predictions == ["('C.O', 0.9)"]
    assert len(calls) == 2
    assert all(call[1] == "([C:1])>>([C:1])" for call in calls)


def test_prediction_parser_accepts_pinned_bare_edit_names_without_eval() -> None:
    assert decoder._parse_prediction("(a, 2, 7, 0.932)") == (
        "a",
        2,
        7,
        0.932,
    )
    assert decoder._parse_prediction("(b, 3, 11, 0.125)") == (
        "b",
        3,
        11,
        0.125,
    )
    assert decoder._parse_prediction("('a', 0, 1, 1)") == ("a", 0, 1, 1.0)


def test_prediction_parser_rejects_executable_or_unknown_names() -> None:
    with pytest.raises(ValueError):
        decoder._parse_prediction("(__import__('os').system('echo unsafe'), 0, 1, 1)")
    with pytest.raises(ValueError):
        decoder._parse_prediction("(c, 0, 1, 1)")


def test_complete_progress_is_written_to_state_directory(tmp_path) -> None:
    decoder._write_complete_progress(tmp_path, 49_584)
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert progress["completed_products"] == 49_584
    assert progress["product_count"] == 49_584
    assert progress["percent"] == 100.0
    assert progress["eta_seconds"] == 0.0
