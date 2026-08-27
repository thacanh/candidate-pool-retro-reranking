"""Fail-closed comparison of historical and regenerated candidate pools.

The comparison is intentionally limited to the normalized candidate-pool view
used by the controlled study.  Products are canonicalized with stereochemistry
enabled.  Dot-separated reactant fragments are canonicalized independently and
sorted while preserving multiplicity.  Canonical duplicate candidates retain
the largest prior; equal-prior duplicates retain the first occurrence.  The
resulting candidates are stably sorted by decreasing prior.

This module does not generate candidates or modify either input artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rdkit import Chem


ABSOLUTE_PRIOR_TOLERANCE = 1e-8
RELATIVE_PRIOR_TOLERANCE = 1e-6
SCHEMA_VERSION = 1

REQUIRED_MANIFEST_FIELDS = {
    "protocol_id",
    "comparator",
    "single_intended_change",
    "input_fingerprints",
    "settings",
    "environment",
    "failures",
    "counts",
    "output",
    "runtime",
}


@dataclass
class CandidateEntry:
    identity: str
    prior: float
    raw_reactant: str
    first_seen_line: int
    winner_line: int


@dataclass
class ProductPool:
    identity: str
    raw_product: str
    first_seen_line: int
    candidates_by_identity: Dict[str, CandidateEntry] = field(default_factory=dict)

    def add_candidate(
        self,
        candidate_identity: str,
        prior: float,
        raw_reactant: str,
        line_number: int,
    ) -> bool:
        """Add a candidate and return whether it was a canonical duplicate."""
        existing = self.candidates_by_identity.get(candidate_identity)
        if existing is None:
            self.candidates_by_identity[candidate_identity] = CandidateEntry(
                identity=candidate_identity,
                prior=prior,
                raw_reactant=raw_reactant,
                first_seen_line=line_number,
                winner_line=line_number,
            )
            return False
        if prior > existing.prior:
            existing.prior = prior
            existing.raw_reactant = raw_reactant
            existing.winner_line = line_number
        return True

    def ordered_candidates(self) -> List[CandidateEntry]:
        # Python's sort is stable.  The explicit first-seen key documents and
        # protects the tie rule even if the backing mapping changes later.
        return sorted(
            self.candidates_by_identity.values(),
            key=lambda item: (-item.prior, item.first_seen_line),
        )


@dataclass
class LoadedPools:
    path: str
    product_order: List[str]
    pools: Dict[str, ProductPool]
    audit: Dict[str, int]
    fatal_errors: List[dict]
    rejected_smiles_records: List[dict]


@lru_cache(maxsize=1_000_000)
def canonicalize_product(smiles: str) -> Optional[str]:
    """Return canonical isomeric SMILES for one product, or ``None``."""
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return None
        return Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return None


@lru_cache(maxsize=1_000_000)
def canonicalize_reactant_fragments(smiles: str) -> Optional[str]:
    """Canonicalize and sort dot-separated fragments, preserving multiplicity."""
    identities: List[str] = []
    for raw_fragment in str(smiles).split("."):
        fragment = raw_fragment.strip()
        if not fragment:
            continue
        identity = canonicalize_product(fragment)
        if identity is None:
            return None
        identities.append(identity)
    if not identities:
        return None
    return ".".join(sorted(identities))


def _fatal_error(line_number: int, error_type: str, detail: str) -> dict:
    return {
        "line_number": line_number,
        "type": error_type,
        "detail": detail,
    }


def load_normalized_pools(path: str | Path) -> LoadedPools:
    """Load a flat candidate JSONL into the prespecified normalized pool view."""
    source_path = Path(path)
    pools: Dict[str, ProductPool] = {}
    product_order: List[str] = []
    fatal_errors: List[dict] = []
    rejected_smiles_records: List[dict] = []
    audit = {
        "physical_lines": 0,
        "blank_lines": 0,
        "candidate_records": 0,
        "malformed_json_lines": 0,
        "missing_required_fields": 0,
        "invalid_prior_records": 0,
        "nonfinite_prior_records": 0,
        "invalid_product_records": 0,
        "invalid_reactant_records": 0,
        "canonical_duplicate_records": 0,
        "unique_products": 0,
        "unique_candidates": 0,
    }

    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            audit["physical_lines"] += 1
            stripped = raw_line.strip()
            if not stripped:
                audit["blank_lines"] += 1
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                audit["malformed_json_lines"] += 1
                fatal_errors.append(
                    _fatal_error(line_number, "malformed_json", str(error))
                )
                continue
            if not isinstance(record, Mapping):
                audit["malformed_json_lines"] += 1
                fatal_errors.append(
                    _fatal_error(line_number, "non_object_json", type(record).__name__)
                )
                continue

            required = ("product", "reactant", "prior")
            missing = [name for name in required if name not in record]
            if missing:
                audit["missing_required_fields"] += 1
                fatal_errors.append(
                    _fatal_error(
                        line_number,
                        "missing_required_fields",
                        ",".join(missing),
                    )
                )
                continue

            raw_product = str(record["product"])
            raw_reactant = str(record["reactant"])
            try:
                prior = float(record["prior"])
            except (TypeError, ValueError, OverflowError):
                audit["invalid_prior_records"] += 1
                fatal_errors.append(
                    _fatal_error(line_number, "invalid_prior", repr(record["prior"]))
                )
                continue
            if not math.isfinite(prior):
                audit["nonfinite_prior_records"] += 1
                fatal_errors.append(
                    _fatal_error(line_number, "nonfinite_prior", repr(prior))
                )
                continue

            audit["candidate_records"] += 1
            product_identity = canonicalize_product(raw_product)
            candidate_identity = canonicalize_reactant_fragments(raw_reactant)
            if product_identity is None or candidate_identity is None:
                if product_identity is None:
                    audit["invalid_product_records"] += 1
                    reason = "invalid_product"
                else:
                    audit["invalid_reactant_records"] += 1
                    reason = "invalid_reactant"
                # Historical cap-10 contains a few invalid candidates that the
                # controlled loader drops.  They are excluded from normalized
                # pools but their ordered raw signatures must reproduce exactly.
                rejected_smiles_records.append(
                    {
                        "type": reason,
                        "product_identity": product_identity,
                        "raw_product": raw_product,
                        "raw_reactant": raw_reactant,
                        "prior": prior,
                    }
                )
                continue

            product_pool = pools.get(product_identity)
            if product_pool is None:
                product_pool = ProductPool(
                    identity=product_identity,
                    raw_product=raw_product,
                    first_seen_line=line_number,
                )
                pools[product_identity] = product_pool
                product_order.append(product_identity)
            was_duplicate = product_pool.add_candidate(
                candidate_identity=candidate_identity,
                prior=prior,
                raw_reactant=raw_reactant,
                line_number=line_number,
            )
            audit["canonical_duplicate_records"] += int(was_duplicate)

    audit["unique_products"] = len(pools)
    audit["unique_candidates"] = sum(
        len(pool.candidates_by_identity) for pool in pools.values()
    )
    return LoadedPools(
        path=str(source_path.resolve()),
        product_order=product_order,
        pools=pools,
        audit=audit,
        fatal_errors=fatal_errors,
        rejected_smiles_records=rejected_smiles_records,
    )


def prior_within_tolerance(reference: float, regenerated: float) -> bool:
    limit = max(
        ABSOLUTE_PRIOR_TOLERANCE,
        RELATIVE_PRIOR_TOLERANCE * abs(reference),
    )
    return math.isfinite(reference) and math.isfinite(regenerated) and (
        abs(regenerated - reference) <= limit
    )


def _relative_difference(reference: float, regenerated: float) -> float:
    denominator = max(abs(reference), ABSOLUTE_PRIOR_TOLERANCE)
    return abs(regenerated - reference) / denominator


def _float32_ordered_integer(value: float) -> Optional[int]:
    """Map a finite float32 value to an integer ordered by numeric value."""
    if not math.isfinite(value):
        return None
    # Normalize signed zero so mathematically identical zero priors are 0 ULP
    # apart. Candidate priors are non-negative, but the signed mapping keeps
    # this diagnostic safe if a malformed finite negative value reaches it.
    normalized = 0.0 if value == 0.0 else value
    try:
        packed = struct.pack(">f", normalized)
    except (OverflowError, struct.error):
        return None
    rounded = struct.unpack(">f", packed)[0]
    if not math.isfinite(rounded):
        return None
    bits = struct.unpack(">I", packed)[0]
    if bits & 0x80000000:
        return (~bits) & 0xFFFFFFFF
    return bits | 0x80000000


def float32_ulp_distance(reference: float, regenerated: float) -> Optional[int]:
    """Return float32 ULP distance, or ``None`` for nonfinite/overflow values."""
    reference_integer = _float32_ordered_integer(reference)
    regenerated_integer = _float32_ordered_integer(regenerated)
    if reference_integer is None or regenerated_integer is None:
        return None
    return abs(regenerated_integer - reference_integer)


def _append_issue(
    discrepancy_map: Dict[str, dict],
    product_key: str,
    issue: dict,
) -> None:
    discrepancy = discrepancy_map.setdefault(
        product_key,
        {"product_key": product_key, "issues": []},
    )
    discrepancy["issues"].append(issue)


def _product_order_issues(
    reference_order: Sequence[str],
    regenerated_order: Sequence[str],
    discrepancy_map: Dict[str, dict],
) -> bool:
    if list(reference_order) == list(regenerated_order):
        return True
    reference_positions = {value: index for index, value in enumerate(reference_order)}
    regenerated_positions = {value: index for index, value in enumerate(regenerated_order)}
    for product_key in set(reference_positions).intersection(regenerated_positions):
        reference_index = reference_positions[product_key]
        regenerated_index = regenerated_positions[product_key]
        if reference_index != regenerated_index:
            _append_issue(
                discrepancy_map,
                product_key,
                {
                    "type": "product_order_mismatch",
                    "reference_index": reference_index,
                    "regenerated_index": regenerated_index,
                },
            )
    return False


def compare_loaded_pools(
    reference: LoadedPools,
    regenerated: LoadedPools,
) -> Tuple[dict, List[dict]]:
    """Compare two already loaded artifacts and return summary plus discrepancies."""
    discrepancy_map: Dict[str, dict] = {}
    global_issues: List[dict] = []
    reference_keys = set(reference.pools)
    regenerated_keys = set(regenerated.pools)

    for product_key in sorted(reference_keys - regenerated_keys):
        _append_issue(
            discrepancy_map,
            product_key,
            {"type": "missing_product"},
        )
    for product_key in sorted(regenerated_keys - reference_keys):
        _append_issue(
            discrepancy_map,
            product_key,
            {"type": "extra_product"},
        )

    product_order_exact = _product_order_issues(
        reference.product_order,
        regenerated.product_order,
        discrepancy_map,
    )
    identity_order_exact = True
    prior_tolerance_passed = True
    prior_inversion_count = 0
    max_absolute_difference = 0.0
    max_relative_difference = 0.0
    aligned_prior_count = 0
    exact_float_prior_count = 0
    max_float32_ulp_distance: Optional[int] = None
    float32_ulp_unavailable_count = 0

    for product_key in reference.product_order:
        if product_key not in regenerated.pools:
            identity_order_exact = False
            continue
        reference_candidates = reference.pools[product_key].ordered_candidates()
        regenerated_candidates = regenerated.pools[product_key].ordered_candidates()
        reference_identities = [item.identity for item in reference_candidates]
        regenerated_identities = [item.identity for item in regenerated_candidates]
        if reference_identities != regenerated_identities:
            identity_order_exact = False
            _append_issue(
                discrepancy_map,
                product_key,
                {
                    "type": "candidate_identity_or_order_mismatch",
                    "reference": reference_identities,
                    "regenerated": regenerated_identities,
                },
            )

        regenerated_by_identity = {
            item.identity: item for item in regenerated_candidates
        }
        aligned_regenerated_priors: List[float] = []
        for reference_candidate in reference_candidates:
            regenerated_candidate = regenerated_by_identity.get(
                reference_candidate.identity
            )
            if regenerated_candidate is None:
                continue
            aligned_prior_count += 1
            exact_float_prior_count += int(
                regenerated_candidate.prior == reference_candidate.prior
            )
            aligned_regenerated_priors.append(regenerated_candidate.prior)
            absolute_difference = abs(
                regenerated_candidate.prior - reference_candidate.prior
            )
            relative_difference = _relative_difference(
                reference_candidate.prior,
                regenerated_candidate.prior,
            )
            max_absolute_difference = max(
                max_absolute_difference,
                absolute_difference,
            )
            max_relative_difference = max(
                max_relative_difference,
                relative_difference,
            )
            ulp_distance = float32_ulp_distance(
                reference_candidate.prior,
                regenerated_candidate.prior,
            )
            if ulp_distance is None:
                float32_ulp_unavailable_count += 1
            elif max_float32_ulp_distance is None:
                max_float32_ulp_distance = ulp_distance
            else:
                max_float32_ulp_distance = max(
                    max_float32_ulp_distance,
                    ulp_distance,
                )
            if not prior_within_tolerance(
                reference_candidate.prior,
                regenerated_candidate.prior,
            ):
                prior_tolerance_passed = False
                _append_issue(
                    discrepancy_map,
                    product_key,
                    {
                        "type": "prior_tolerance_exceeded",
                        "candidate_identity": reference_candidate.identity,
                        "reference_prior": reference_candidate.prior,
                        "regenerated_prior": regenerated_candidate.prior,
                        "absolute_difference": absolute_difference,
                        "allowed_difference": max(
                            ABSOLUTE_PRIOR_TOLERANCE,
                            RELATIVE_PRIOR_TOLERANCE
                            * abs(reference_candidate.prior),
                        ),
                    },
                )

        # Compare regenerated scores in reference identity order.  A negative
        # adjacent difference is an explicit inversion of the frozen ranking.
        for left, right in zip(
            aligned_regenerated_priors,
            aligned_regenerated_priors[1:],
        ):
            if left < right:
                prior_inversion_count += 1
                _append_issue(
                    discrepancy_map,
                    product_key,
                    {
                        "type": "prior_order_inversion",
                        "left_prior": left,
                        "right_prior": right,
                    },
                )

    if reference.fatal_errors:
        global_issues.append(
            {"type": "reference_fatal_load_errors", "errors": reference.fatal_errors}
        )
    if regenerated.fatal_errors:
        global_issues.append(
            {
                "type": "regenerated_fatal_load_errors",
                "errors": regenerated.fatal_errors,
            }
        )
    audit_fields = sorted(set(reference.audit).union(regenerated.audit))
    audit_count_differences = {
        field_name: {
            "reference": reference.audit.get(field_name),
            "regenerated": regenerated.audit.get(field_name),
        }
        for field_name in audit_fields
        if reference.audit.get(field_name) != regenerated.audit.get(field_name)
    }
    audit_counts_exact = not audit_count_differences
    if not audit_counts_exact:
        global_issues.append(
            {
                "type": "raw_audit_counts_mismatch",
                "differences": audit_count_differences,
            }
        )
    rejected_records_exact = (
        reference.rejected_smiles_records == regenerated.rejected_smiles_records
    )
    if not rejected_records_exact:
        global_issues.append(
            {
                "type": "rejected_smiles_records_mismatch",
                "reference": reference.rejected_smiles_records,
                "regenerated": regenerated.rejected_smiles_records,
            }
        )

    product_identity_exact = reference_keys == regenerated_keys
    comparison_passed = all(
        (
            product_identity_exact,
            product_order_exact,
            identity_order_exact,
            prior_tolerance_passed,
            prior_inversion_count == 0,
            not reference.fatal_errors,
            not regenerated.fatal_errors,
            audit_counts_exact,
            rejected_records_exact,
            not discrepancy_map,
            not global_issues,
        )
    )
    discrepancies = list(discrepancy_map.values())
    if global_issues:
        discrepancies.append({"product_key": None, "issues": global_issues})

    summary = {
        "schema_version": SCHEMA_VERSION,
        "comparison_passed": comparison_passed,
        "reference": {
            "path": reference.path,
            "audit": reference.audit,
            "fatal_error_count": len(reference.fatal_errors),
            "rejected_smiles_record_count": len(
                reference.rejected_smiles_records
            ),
        },
        "regenerated": {
            "path": regenerated.path,
            "audit": regenerated.audit,
            "fatal_error_count": len(regenerated.fatal_errors),
            "rejected_smiles_record_count": len(
                regenerated.rejected_smiles_records
            ),
        },
        "tolerance": {
            "absolute": ABSOLUTE_PRIOR_TOLERANCE,
            "relative": RELATIVE_PRIOR_TOLERANCE,
            "rule": "abs(new-ref) <= max(1e-8, 1e-6*abs(ref))",
        },
        "canonicalization": {
            "product": "RDKit canonical isomeric SMILES",
            "reactant": "canonical isomeric fragments sorted with multiplicity preserved",
            "duplicate_winner": "highest prior; first occurrence on an equal-prior tie",
            "candidate_order": "stable decreasing prior",
            "ignored_fields": ["label"],
        },
        "checks": {
            "product_identity_exact": product_identity_exact,
            "product_order_exact": product_order_exact,
            "candidate_identity_and_order_exact": identity_order_exact,
            "audit_counts_exact": audit_counts_exact,
            "audit_count_differences": audit_count_differences,
            "prior_tolerance_passed": prior_tolerance_passed,
            "prior_inversion_count": prior_inversion_count,
            "rejected_smiles_records_exact": rejected_records_exact,
            "aligned_prior_count": aligned_prior_count,
            "exact_float_prior_count": exact_float_prior_count,
            "exact_float_prior_fraction": (
                exact_float_prior_count / aligned_prior_count
                if aligned_prior_count
                else None
            ),
            "max_absolute_prior_difference": max_absolute_difference,
            "max_relative_prior_difference": max_relative_difference,
            "max_float32_ulp_distance": max_float32_ulp_distance,
            "float32_ulp_unavailable_count": float32_ulp_unavailable_count,
        },
        "products_compared": len(reference_keys.intersection(regenerated_keys)),
        "discrepancy_product_count": len(discrepancy_map),
        "discrepancy_record_count": len(discrepancies),
    }
    return summary, discrepancies


def compare_candidate_pools(
    reference_path: str | Path,
    regenerated_path: str | Path,
) -> Tuple[dict, List[dict]]:
    reference = load_normalized_pools(reference_path)
    regenerated = load_normalized_pools(regenerated_path)
    return compare_loaded_pools(reference, regenerated)


def _nonempty_change(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    return False


def validate_manifest_schema(manifest: Any) -> List[str]:
    """Return human-readable schema errors for a candidate-generation manifest."""
    if not isinstance(manifest, Mapping):
        return ["manifest must be a JSON object"]
    errors: List[str] = []
    missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(manifest))
    errors.extend(f"missing required field: {field_name}" for field_name in missing)

    for field_name in ("protocol_id", "comparator"):
        if field_name in manifest and (
            not isinstance(manifest[field_name], str)
            or not manifest[field_name].strip()
        ):
            errors.append(f"{field_name} must be a non-empty string")
    if "single_intended_change" in manifest and not _nonempty_change(
        manifest["single_intended_change"]
    ):
        errors.append("single_intended_change must be a non-empty string or object")

    fingerprints = manifest.get("input_fingerprints")
    if fingerprints is not None:
        if not isinstance(fingerprints, Mapping) or not fingerprints:
            errors.append("input_fingerprints must be a non-empty object")
        else:
            for name, fingerprint in fingerprints.items():
                prefix = f"input_fingerprints.{name}"
                if not isinstance(fingerprint, Mapping):
                    errors.append(f"{prefix} must be an object")
                    continue
                for required in ("path", "size_bytes", "sha256"):
                    if required not in fingerprint:
                        errors.append(f"{prefix} missing {required}")
                path = fingerprint.get("path")
                if path is not None and (
                    not isinstance(path, str) or not path.strip()
                ):
                    errors.append(f"{prefix}.path must be a non-empty string")
                size = fingerprint.get("size_bytes")
                if size is not None and (
                    isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                ):
                    errors.append(f"{prefix}.size_bytes must be a non-negative integer")
                sha256 = fingerprint.get("sha256")
                if sha256 is not None and (
                    not isinstance(sha256, str)
                    or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None
                ):
                    errors.append(f"{prefix}.sha256 must contain 64 hexadecimal digits")

    for field_name in ("settings", "environment", "counts", "output", "runtime"):
        value = manifest.get(field_name)
        if value is not None and (not isinstance(value, Mapping) or not value):
            errors.append(f"{field_name} must be a non-empty object")
    failures = manifest.get("failures")
    if failures is not None and not isinstance(failures, (Mapping, list)):
        errors.append("failures must be an object or array")
    return errors


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_reports(
    summary: Mapping[str, Any],
    discrepancies: Iterable[Mapping[str, Any]],
    summary_path: str | Path,
    discrepancies_path: str | Path,
) -> None:
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    discrepancy_text = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in discrepancies
    )
    _atomic_write(Path(summary_path), summary_text)
    _atomic_write(Path(discrepancies_path), discrepancy_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a regenerated candidate JSONL with the frozen pool."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--regenerated", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--discrepancies-jsonl", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_paths = {
        Path(args.reference).resolve(),
        Path(args.regenerated).resolve(),
        Path(args.manifest_json).resolve(),
    }
    summary_path = Path(args.summary_json).resolve()
    discrepancies_path = Path(args.discrepancies_jsonl).resolve()
    if summary_path in input_paths or discrepancies_path in input_paths:
        raise ValueError("Report paths must not overwrite an input artifact.")
    if summary_path == discrepancies_path:
        raise ValueError("Summary and discrepancy reports require distinct paths.")

    summary, discrepancies = compare_candidate_pools(
        args.reference,
        args.regenerated,
    )
    try:
        with Path(args.manifest_json).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_errors = validate_manifest_schema(manifest)
    except (OSError, json.JSONDecodeError) as error:
        manifest_errors = [f"could not read manifest: {error}"]
    manifest_valid = not manifest_errors
    summary["manifest_validation"] = {
        "path": str(Path(args.manifest_json).resolve()),
        "valid": manifest_valid,
        "errors": manifest_errors,
    }
    summary["passed"] = bool(summary["comparison_passed"] and manifest_valid)
    if manifest_errors:
        discrepancies.append(
            {
                "product_key": None,
                "issues": [
                    {"type": "manifest_schema_invalid", "errors": manifest_errors}
                ],
            }
        )
        summary["discrepancy_record_count"] = len(discrepancies)
    write_reports(
        summary,
        discrepancies,
        summary_path,
        discrepancies_path,
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
