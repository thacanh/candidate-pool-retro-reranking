import json
from pathlib import Path

import pandas as pd
import pytest

from rerank.data.build_ws_e_candidate_pools import build_three_pools, normalized_rank
from rerank.data.prepare_localretro_current_dataset import (
    PROTOCOL_ID as MAPPING_PROTOCOL,
    _mapped_paths,
    compile_mapping,
    fingerprint,
    prepare_product_only_inference,
    reaction_identity,
    transfer_atom_maps_to_original,
    validate_mapped_reaction,
)
from rerank.experiments.run_localretro_revision import (
    PROTOCOL_ID as LOCALRETRO_PROTOCOL,
    compile_decoded_predictions,
)


def _jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_mapped_reaction_gate_preserves_chemistry_and_requires_product_maps() -> None:
    audit = validate_mapped_reaction(
        "CO.CN>>CO",
        "[CH3:1][OH:2].CN>>[CH3:1][OH:2]",
    )
    assert audit["product_atom_count"] == 2
    assert audit["reactant_unmapped_atom_count"] == 2
    with pytest.raises(ValueError, match="chemically identical"):
        validate_mapped_reaction("CO>>CO", "[CH3:1][OH:2]>>[CH3:1][NH2:2]")
    with pytest.raises(ValueError, match="positive atom-map"):
        validate_mapped_reaction("CO>>CO", "CO>>CO")
    with pytest.raises(ValueError, match="absent from the reactants"):
        validate_mapped_reaction(
            "CO>>CO", "[CH3:1][OH:2]>>[CH3:1][OH:3]"
        )
    audit = validate_mapped_reaction(
        "CO>>CO",
        "[CH3:1][OH:2]>>[CH3:1][OH:3]",
        allow_unbalanced_product=True,
    )
    assert audit["product_atom_maps_absent_from_reactants"] == 1


def test_atom_map_transfer_restores_exact_original_stereochemistry() -> None:
    unmapped = "N[C@@H](C)C(=O)O>>N[C@@H](C)C(=O)O"
    mapper_output = (
        "[NH2:1][C@H:2]([CH3:3])[C:4](=[O:5])[OH:6]"
        ">>[NH2:1][C@H:2]([CH3:3])[C:4](=[O:5])[OH:6]"
    )
    assert reaction_identity(unmapped) != reaction_identity(mapper_output)
    restored, transfer = transfer_atom_maps_to_original(unmapped, mapper_output)
    assert reaction_identity(restored) == reaction_identity(unmapped)
    assert transfer["original_stereochemistry_restored"] is True
    assert validate_mapped_reaction(unmapped, restored)["product_atom_count"] == 6


def test_identity_refreshes_rdkit_ranking_after_atom_maps_are_removed() -> None:
    """Regression for v2 shard-0 false failures such as reaction 217."""

    original = (
        "CCOC(=O)C1CC=C(c2ccc(N)c3c(=O)c(C)c[nH]c23)CC1"
        ">>"
        "CCOC(=O)[C@H]1CC[C@@H](c2ccc(N)c3c(=O)c(C)c[nH]c32)CC1"
    )
    mapped = (
        "[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH:6]1[CH2:7][CH:8]="
        "[C:9]([c:10]2[cH:11][cH:12][c:13]([NH2:14])[c:15]3[c:16]"
        "(=[O:17])[c:18]([CH3:19])[cH:20][nH:21][c:22]23)[CH2:23][CH2:24]1"
        ">>"
        "[CH3:1][CH2:2][O:3][C:4](=[O:5])[C@H:6]1[CH2:7][CH2:8]"
        "[C@@H:9]([c:10]2[cH:11][cH:12][c:13]([NH2:14])[c:15]3[c:16]"
        "(=[O:17])[c:18]([CH3:19])[cH:20][nH:21][c:22]23)[CH2:23][CH2:24]1"
    )

    restored, transfer = transfer_atom_maps_to_original(original, mapped)
    assert reaction_identity(restored) == reaction_identity(original)
    assert transfer["original_stereochemistry_restored"] is False
    assert validate_mapped_reaction(original, restored)["product_atom_count"] == 24


def test_compile_mapping_writes_train_valid_only(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "reactants_smiles": ["CO", "CN", "CC", "CCC"],
            "products_smiles": ["CO", "CN", "CO", "CCC"],
            "set": ["train", "valid", "train", "test"],
        }
    ).to_csv(source, index=False)
    pd.DataFrame(
        {
            "reaction_id": [0, 1, 2, 3],
            "source_split": ["train", "valid", "train", "test"],
            "reaction_class": [1, 2, 3, 4],
        }
    ).to_csv(metadata, index=False)
    mapping_root = tmp_path / "mapping"
    records = [
        {
            "reaction_id": 0,
            "split": "train",
            "reaction_class": 1,
            "unmapped_reaction": "CO>>CO",
            "mapped_reaction": "[CH3:1][OH:2]>>[CH3:1][OH:2]",
        },
        {
            "reaction_id": 1,
            "split": "valid",
            "reaction_class": 2,
            "unmapped_reaction": "CN>>CN",
            "mapped_reaction": "[CH3:1][NH2:2]>>[CH3:1][NH2:2]",
        },
    ]
    for index, record in enumerate(records):
        data_path, manifest_path = _mapped_paths(mapping_root, index, 2)
        _jsonl(data_path, [record])
        exclusion_path = (
            mapping_root
            / "shards"
            / f"mapping_exclusions_{index:03d}_of_002.json"
        )
        exclusions = []
        if index == 1:
            exclusions.append(
                {
                    "reaction_id": 2,
                    "split": "train",
                    "reason": "product_atoms_absent_from_reactants",
                    "missing_product_atom_count": 1,
                }
            )
        exclusion_path.write_text(
            json.dumps(
                {
                    "protocol_id": MAPPING_PROTOCOL,
                    "exclusions": exclusions,
                }
            ),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "protocol_id": MAPPING_PROTOCOL,
                    "output": fingerprint(data_path),
                    "exclusions": fingerprint(exclusion_path),
                }
            ),
            encoding="utf-8",
        )
    dataset = tmp_path / "dataset"
    result = compile_mapping(
        source_csv=source,
        metadata_csv=metadata,
        mapping_root=mapping_root,
        shard_count=2,
        dataset_dir=dataset,
    )
    assert result["counts"] == {
        "train": 1,
        "valid": 1,
        "total": 2,
        "source_train_valid_total": 3,
        "excluded_train": 1,
        "excluded_valid": 0,
        "excluded_total": 1,
        "test_rows_loaded": 0,
    }
    assert (dataset / "raw_train.csv").is_file()
    assert (dataset / "raw_val.csv").is_file()
    exclusion_audit = json.loads((dataset / "mapping_exclusions.json").read_text())
    assert exclusion_audit["total"] == 1
    assert exclusion_audit["exclusions"][0]["reaction_id"] == 2
    assert not (dataset / "raw_test.csv").exists()


def test_product_only_inference_requires_pretest_freeze(tmp_path: Path) -> None:
    anchored = tmp_path / "anchored.jsonl"
    _jsonl(
        anchored,
        [
            {"product": "CCO", "reactant": "C.O"},
            {"product": "CCO", "reactant": "N"},
            {"product": "CCN", "reactant": "C.N"},
        ],
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps({"protocol_id": LOCALRETRO_PROTOCOL, "test_partition_loaded": False}),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    inventory = dataset / "inference_products.jsonl"
    result = prepare_product_only_inference(
        anchored_pool=anchored,
        checkpoint_freeze=freeze,
        dataset_dir=dataset,
        inventory_output=inventory,
    )
    assert result["product_count"] == 2
    raw = pd.read_csv(dataset / "raw_test.csv")
    assert raw["reactants>reagents>production"].tolist() == ["CCO>>CCO", "CCN>>CCN"]


def test_compile_decoded_predictions_deduplicates_canonically(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    _jsonl(inventory, [{"test_id": 0, "product": "CCO"}])
    decoded = tmp_path / "decoded.txt"
    decoded.write_text(
        "0\t('C.O', 0.9)\t('O.C', 0.8)\t('N', 0.7)\n",
        encoding="utf-8",
    )
    output = tmp_path / "predictions.jsonl"
    counts = compile_decoded_predictions(
        decoded_path=decoded,
        inventory_path=inventory,
        output_path=output,
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert counts["canonical_duplicates_removed"] == 1
    assert [record["reactant"] for record in records] == ["C.O", "N"]
    assert [record["prior"] for record in records] == [1.0, 0.0]


def test_three_pool_builder_preserves_sources_and_coverage(tmp_path: Path) -> None:
    aizynth = tmp_path / "aizynth.jsonl"
    local = tmp_path / "local.jsonl"
    inventory = tmp_path / "inventory.jsonl"
    source = tmp_path / "source.csv"
    _jsonl(
        aizynth,
        [
            {
                "product": "CCO",
                "reactant": "C.O",
                "prior": 0.8,
                "candidate_rank": 1,
                "candidate_source": "legacy_cap10_anchor",
                "protocol_id": "cap50-legacy-anchored-v1",
            },
            {
                "product": "CCO",
                "reactant": "N",
                "prior": 0.5,
                "candidate_rank": 2,
                "candidate_source": "clean_cap50_extension",
                "protocol_id": "cap50-legacy-anchored-v1",
            },
            {
                "product": "CCN",
                "reactant": "C.N",
                "prior": 0.7,
                "candidate_rank": 1,
                "candidate_source": "legacy_cap10_anchor",
                "protocol_id": "cap50-legacy-anchored-v1",
            },
        ],
    )
    _jsonl(
        inventory,
        [{"test_id": 0, "product": "CCO"}, {"test_id": 1, "product": "CCN"}],
    )
    _jsonl(
        local,
        [
            {
                "product": "CCO",
                "reactant": "O.C",
                "raw_score": 0.9,
                "generator_rank": 1,
                "protocol_id": LOCALRETRO_PROTOCOL,
            },
            {
                "product": "CCO",
                "reactant": "CO",
                "raw_score": 0.4,
                "generator_rank": 2,
                "protocol_id": LOCALRETRO_PROTOCOL,
            },
        ],
    )
    pd.DataFrame(
        {
            "reactants_smiles": ["C.O", "C.N"],
            "products_smiles": ["CCO", "CCN"],
            "set": ["test", "test"],
        }
    ).to_csv(source, index=False)
    output = tmp_path / "pools"
    result = build_three_pools(
        aizynth_pool=aizynth,
        localretro_predictions=local,
        localretro_inventory=inventory,
        source_csv=source,
        output_root=output,
    )
    merged = [
        json.loads(line)
        for line in (output / "merged_canonical_union.jsonl").read_text().splitlines()
    ]
    both = [record for record in merged if record["product"] == "CCO" and record["reactant"] == "C.O"]
    assert len(both) == 1
    assert both[0]["source_aizynthfinder"] == 1
    assert both[0]["source_localretro"] == 1
    contribution = result["coverage"]["correct_candidate_contribution"]
    assert contribution["both_generators_correct"] == 1
    assert contribution["aizynth_only_correct"] == 1
    assert normalized_rank(1, 1) == 1.0
    assert json.loads((output / "POOL_RELEASE_GATE.json").read_text())["passed"]
