#!/usr/bin/env python
"""Verify Chemformer USPTO-50K provenance and export reaction-class metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from datetime import date
from pathlib import Path

import pandas as pd
from rdkit import Chem

CHEMFORMER_REPOSITORY = "https://github.com/MolecularAI/Chemformer"
CHEMFORMER_DATASET_PAGE = "https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq"
CHEMFORMER_BOX_FILE_ID = "854847820319"
T5CHEM_RECORD_URL = "https://zenodo.org/records/14280768"
T5CHEM_PAPER_DOI = "10.1021/acs.jcim.1c01467"
T5CHEM_ARCHIVE_NAME = "USPTO_50k.tar.bz2"
T5CHEM_ARCHIVE_MD5 = "44a5f3ae08fe55933404c9398be22f5b"
T5CHEM_ARCHIVE_SIZE_BYTES = 913111


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path: str | Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def reaction_class(value: object) -> int:
    match = re.fullmatch(r"<RX_(\d+)>", str(value))
    if match is None:
        raise ValueError(f"Unexpected Chemformer reaction_type: {value!r}")
    result = int(match.group(1))
    if not 1 <= result <= 10:
        raise ValueError(f"Reaction class outside USPTO-50K range: {result}")
    return result


def verify_t5chem_archive(
    source_csv: str | Path, archive_path: str | Path
) -> dict:
    """Verify the DOI-linked T5Chem mirror against the local split row by row."""
    archive_path = Path(archive_path)
    if archive_path.stat().st_size != T5CHEM_ARCHIVE_SIZE_BYTES:
        raise ValueError(
            "Unexpected T5Chem archive size: "
            f"{archive_path.stat().st_size} != {T5CHEM_ARCHIVE_SIZE_BYTES}"
        )
    archive_md5 = md5(archive_path)
    if archive_md5 != T5CHEM_ARCHIVE_MD5:
        raise ValueError(
            f"Unexpected T5Chem archive MD5: {archive_md5} != {T5CHEM_ARCHIVE_MD5}"
        )

    local = pd.read_csv(source_csv).reset_index(drop=True)
    split_files = {"train": "train", "valid": "val", "test": "test"}
    canonical_product_matches = 0
    canonical_reactant_matches = 0
    raw_pair_matches = 0
    split_counts: dict[str, int] = {}

    with tarfile.open(archive_path, "r:bz2") as archive:
        for local_split, archive_split in split_files.items():
            subset = local.loc[local["set"].astype(str) == local_split].reset_index(
                drop=True
            )
            prefix = f"data/USPTO_50k/{archive_split}"
            source_handle = archive.extractfile(f"{prefix}.source")
            target_handle = archive.extractfile(f"{prefix}.target")
            if source_handle is None or target_handle is None:
                raise ValueError(f"Missing T5Chem split files for {archive_split}")
            products = source_handle.read().decode("utf-8").splitlines()
            reactants = target_handle.read().decode("utf-8").splitlines()
            if len(products) != len(subset) or len(reactants) != len(subset):
                raise ValueError(
                    f"T5Chem row-count mismatch for {local_split}: "
                    f"local={len(subset)}, products={len(products)}, "
                    f"reactants={len(reactants)}"
                )

            split_counts[local_split] = len(subset)
            for index, (product, reactant) in enumerate(zip(products, reactants)):
                local_product = str(subset.at[index, "products_smiles"])
                local_reactant = str(subset.at[index, "reactants_smiles"])
                raw_pair_matches += int(
                    product == local_product and reactant == local_reactant
                )
                canonical_product_matches += int(
                    canonical(product) == canonical(local_product)
                )
                canonical_reactant_matches += int(
                    canonical(reactant) == canonical(local_reactant)
                )

    expected = len(local)
    if canonical_product_matches != expected or canonical_reactant_matches != expected:
        raise ValueError(
            "Canonical mismatch between the T5Chem archive and local CSV: "
            f"products={canonical_product_matches}/{expected}, "
            f"reactants={canonical_reactant_matches}/{expected}"
        )
    return {
        "record_url": T5CHEM_RECORD_URL,
        "associated_paper_doi": T5CHEM_PAPER_DOI,
        "archive_filename": T5CHEM_ARCHIVE_NAME,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_md5": archive_md5,
        "split_counts": split_counts,
        "raw_smiles_pair_matches": raw_pair_matches,
        "canonical_product_matches": canonical_product_matches,
        "canonical_reactant_matches": canonical_reactant_matches,
        "comparison_date": date.today().isoformat(),
        "comparison_note": (
            f"All rows match in split order after RDKit canonicalization; "
            f"{expected - raw_pair_matches} rows use non-byte-identical but "
            "canonically equivalent SMILES. The archive does not supply the "
            "Chemformer reaction-class labels."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemformer-pickle", required=True)
    parser.add_argument("--source-csv", default="data/uspto_smiles.csv")
    parser.add_argument("--output", default="data/uspto_reaction_metadata.csv")
    parser.add_argument(
        "--provenance-output", default="data/uspto_provenance.json"
    )
    parser.add_argument(
        "--t5chem-archive",
        help=(
            "Optional DOI-linked USPTO_50k.tar.bz2 from Zenodo record 14280768; "
            "when supplied, verify it row by row against the local CSV."
        ),
    )
    args = parser.parse_args()

    official = pd.read_pickle(args.chemformer_pickle).reset_index(drop=True)
    local = pd.read_csv(args.source_csv).reset_index(drop=True)
    required = {"reactants_mol", "products_mol", "reaction_type", "set"}
    if not required.issubset(official.columns):
        raise ValueError(f"Unexpected official schema: {official.columns.tolist()}")
    if len(official) != len(local):
        raise ValueError(
            f"Row-count mismatch: official={len(official)}, local={len(local)}"
        )
    if not official["set"].astype(str).equals(local["set"].astype(str)):
        raise ValueError("Official and local split columns do not match row-for-row.")

    product_matches = 0
    reactant_matches = 0
    for index in range(len(local)):
        official_product = Chem.MolToSmiles(
            official.at[index, "products_mol"], canonical=True
        )
        official_reactants = Chem.MolToSmiles(
            official.at[index, "reactants_mol"], canonical=True
        )
        if official_product == canonical(local.at[index, "products_smiles"]):
            product_matches += 1
        if official_reactants == canonical(local.at[index, "reactants_smiles"]):
            reactant_matches += 1
    if product_matches != len(local) or reactant_matches != len(local):
        raise ValueError(
            "Canonical molecule mismatch between official pickle and local CSV: "
            f"products={product_matches}/{len(local)}, "
            f"reactants={reactant_matches}/{len(local)}"
        )

    metadata = pd.DataFrame(
        {
            "reaction_id": range(len(official)),
            "source_split": official["set"].astype(str),
            "reaction_class": official["reaction_type"].map(reaction_class),
        }
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output, index=False)

    provenance = {
        "dataset": "Chemformer USPTO-50K",
        "repository": CHEMFORMER_REPOSITORY,
        "public_dataset_page": CHEMFORMER_DATASET_PAGE,
        "box_file_id": CHEMFORMER_BOX_FILE_ID,
        "official_filename": "uspto_50.pickle",
        "official_pickle_sha256": sha256(args.chemformer_pickle),
        "official_pickle_size_bytes": Path(args.chemformer_pickle).stat().st_size,
        "local_source_csv_sha256": sha256(args.source_csv),
        "n_reactions": len(local),
        "split_counts": local["set"].value_counts().to_dict(),
        "reaction_class_counts": metadata["reaction_class"].value_counts().sort_index().to_dict(),
        "canonical_product_matches": product_matches,
        "canonical_reactant_matches": reactant_matches,
        "source_lineage": {
            "raw_patent_reaction_corpus": {
                "title": "Chemical reactions from US patents (1976-Sep2016)",
                "doi": "10.6084/m9.figshare.5104873.v1",
                "license": "CC0",
            },
            "reaction_role_assignment_and_random_subset": {
                "paper_doi": "10.1021/acs.jcim.6b00564"
            },
            "retrosynthesis_benchmark": {
                "paper_doi": "10.1021/acscentsci.7b00303"
            },
            "chemformer_distribution": {
                "paper_doi": "10.1088/2632-2153/ac3ffb"
            },
        },
    }
    if args.t5chem_archive:
        provenance["independent_archive_crosscheck"] = verify_t5chem_archive(
            args.source_csv, args.t5chem_archive
        )
    with open(args.provenance_output, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
