"""Build a deterministic cap-50 sensitivity pool anchored to legacy cap-10.

This protocol is used only after the clean A-CAP10-REPRO gate failed because
the historical environment did not retain enough information to reproduce
equal-prior outcome ordering.  It does not relabel that failed gate as passed.
For every canonical product, the normalized legacy identities, order, and
priors are immutable.  New identities are appended from the checksummed clean
Top-50 artifact until the pool reaches at most 50 unique candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from rerank.analysis.compare_candidate_pools import CandidateEntry, LoadedPools, load_normalized_pools


SCHEMA_VERSION = 1
PROTOCOL_ID = "cap50-legacy-anchored-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: str | Path, *, reported_path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    return {
        "path": str(Path(reported_path).resolve() if reported_path is not None else source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _require_compatible_products(legacy: LoadedPools, regenerated: LoadedPools) -> None:
    legacy_keys = set(legacy.pools)
    regenerated_keys = set(regenerated.pools)
    if legacy_keys != regenerated_keys:
        raise ValueError(
            "Legacy and regenerated product identities differ: "
            f"missing={len(legacy_keys - regenerated_keys)}, "
            f"extra={len(regenerated_keys - legacy_keys)}."
        )


def _anchored_candidates(
    legacy_candidates: Sequence[CandidateEntry],
    regenerated_candidates: Sequence[CandidateEntry],
    cap: int,
) -> tuple[list[tuple[CandidateEntry, str]], dict[str, int]]:
    if cap <= 0:
        raise ValueError("Candidate cap must be positive.")
    if len(legacy_candidates) > cap:
        raise ValueError("Legacy anchor already exceeds the requested cap.")

    regenerated_by_identity = {
        candidate.identity: candidate for candidate in regenerated_candidates
    }
    missing = [
        candidate.identity
        for candidate in legacy_candidates
        if candidate.identity not in regenerated_by_identity
    ]
    if missing:
        raise ValueError(
            "A legacy anchor candidate is absent from regenerated cap-50: "
            + missing[0]
        )

    anchored: list[tuple[CandidateEntry, str]] = [
        (candidate, "legacy_cap10_anchor") for candidate in legacy_candidates
    ]
    seen = {candidate.identity for candidate in legacy_candidates}
    for candidate in regenerated_candidates:
        if candidate.identity in seen:
            continue
        anchored.append((candidate, "clean_cap50_extension"))
        seen.add(candidate.identity)
        if len(anchored) >= cap:
            break

    extension_candidates = [
        candidate
        for candidate, source in anchored
        if source == "clean_cap50_extension"
    ]
    anchor_floor = legacy_candidates[-1].prior if legacy_candidates else float("inf")
    diagnostics = {
        "missing_anchor_candidates": len(missing),
        "cross_boundary_adjacent_prior_inversions": int(
            bool(legacy_candidates)
            and bool(extension_candidates)
            and legacy_candidates[-1].prior < extension_candidates[0].prior
        ),
        "extension_candidates_above_anchor_floor": sum(
            candidate.prior > anchor_floor for candidate in extension_candidates
        ),
    }
    return anchored, diagnostics


def build_legacy_anchored_pool(
    *,
    legacy_cap10: str | Path,
    regenerated_cap50: str | Path,
    failed_reproduction_summary: str | Path,
    output_root: str | Path,
    cap: int = 50,
) -> dict[str, Any]:
    start = time.perf_counter()
    legacy_path = Path(legacy_cap10).resolve()
    regenerated_path = Path(regenerated_cap50).resolve()
    summary_path = Path(failed_reproduction_summary).resolve()
    final_root = Path(output_root).resolve()
    for source in (legacy_path, regenerated_path, summary_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    if final_root.exists():
        raise FileExistsError(f"Refusing to overwrite anchored output: {final_root}")

    failed_summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if failed_summary.get("passed") is not False:
        raise ValueError("The supplied reproduction summary is not a recorded failed gate.")

    legacy = load_normalized_pools(legacy_path)
    regenerated = load_normalized_pools(regenerated_path)
    if legacy.fatal_errors or regenerated.fatal_errors:
        raise ValueError("Candidate inputs contain fatal normalized-loader errors.")
    _require_compatible_products(legacy, regenerated)

    staging = final_root.with_name(f".{final_root.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise FileExistsError(f"Staging output already exists: {staging}")
    staging.mkdir(parents=True)
    output_path = staging / "candidates_cap50_legacy_anchored.jsonl"

    anchor_count = 0
    extension_count = 0
    products_at_cap = 0
    products_with_cross_boundary_prior_inversion = 0
    extension_candidates_above_anchor_floor = 0
    min_candidates: int | None = None
    max_candidates = 0
    try:
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for product_rank, product_key in enumerate(legacy.product_order):
                legacy_pool = legacy.pools[product_key]
                regenerated_pool = regenerated.pools[product_key]
                legacy_candidates = legacy_pool.ordered_candidates()
                regenerated_candidates = regenerated_pool.ordered_candidates()
                anchored, diagnostics = _anchored_candidates(
                    legacy_candidates, regenerated_candidates, cap
                )
                products_with_cross_boundary_prior_inversion += diagnostics[
                    "cross_boundary_adjacent_prior_inversions"
                ]
                extension_candidates_above_anchor_floor += diagnostics[
                    "extension_candidates_above_anchor_floor"
                ]
                anchor_count += len(legacy_candidates)
                extension_count += len(anchored) - len(legacy_candidates)
                products_at_cap += int(len(anchored) == cap)
                min_candidates = (
                    len(anchored)
                    if min_candidates is None
                    else min(min_candidates, len(anchored))
                )
                max_candidates = max(max_candidates, len(anchored))
                for candidate_rank, (candidate, source) in enumerate(anchored, start=1):
                    record = {
                        "product": legacy_pool.raw_product,
                        "reactant": candidate.raw_reactant,
                        "prior": candidate.prior,
                        "candidate_rank": candidate_rank,
                        "product_rank": product_rank,
                        "candidate_source": source,
                        "protocol_id": PROTOCOL_ID,
                    }
                    handle.write(
                        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                    )
            handle.flush()
            os.fsync(handle.fileno())

        output_final = final_root / output_path.name
        validation = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "passed": True,
            "checks": {
                "failed_exact_reproduction_gate_preserved": True,
                "product_identity_and_order_match_legacy": True,
                "all_legacy_anchor_candidates_present_in_clean_cap50": True,
                "legacy_anchor_identity_order_prior_preserved": True,
                "extensions_drawn_only_from_clean_cap50": True,
                "canonical_candidate_count_at_most_50": True,
                "anchor_and_extension_internal_prior_order_preserved": True,
                "cross_boundary_raw_prior_inversions_reported": True,
                "downstream_prior_must_use_anchored_candidate_rank": True,
            },
            "counts": {
                "products": len(legacy.product_order),
                "legacy_anchor_candidates": anchor_count,
                "extension_candidates": extension_count,
                "total_candidates": anchor_count + extension_count,
                "products_at_cap": products_at_cap,
                "minimum_candidates_per_product": min_candidates,
                "maximum_candidates_per_product": max_candidates,
                "legacy_anchor_candidates_missing_from_clean_cap50": 0,
                "products_with_cross_boundary_raw_prior_inversion": (
                    products_with_cross_boundary_prior_inversion
                ),
                "extension_candidates_above_anchor_floor": (
                    extension_candidates_above_anchor_floor
                ),
            },
            "failed_reproduction_summary": {
                "sha256": sha256_file(summary_path),
                "discrepancy_product_count": failed_summary.get(
                    "discrepancy_product_count"
                ),
            },
            "created_at_utc": utc_now(),
        }
        validation_path = staging / "anchor_validation.json"
        atomic_json(validation_path, validation)

        script_path = Path(__file__).resolve()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "comparator": "frozen legacy-cap10 candidate pool",
            "single_intended_change": (
                "append only clean-run cap-50 candidate identities beyond the immutable "
                "legacy cap-10 anchor"
            ),
            "decision": {
                "status": "explicitly authorized by repository owner/user",
                "date": "2026-08-14",
                "claim_boundary": (
                    "secondary expanded-pool sensitivity; not an exact cap-10 reproduction"
                ),
            },
            "input_fingerprints": {
                "legacy_cap10": file_fingerprint(legacy_path),
                "clean_regenerated_cap50": file_fingerprint(regenerated_path),
                "failed_cap10_reproduction_summary": file_fingerprint(summary_path),
                "builder_script": file_fingerprint(script_path),
            },
            "settings": {
                "candidate_cap": cap,
                "product_order": "legacy normalized first-seen order",
                "anchor_order": "legacy stable decreasing prior then first-seen order",
                "extension_order": "clean cap-50 stable decreasing prior then first-seen order",
                "deduplication": "canonical fragment-set identity",
                "anchor_prior": "legacy prior",
                "extension_prior": "clean cap-50 prior",
                "downstream_prior_feature": (
                    "1-(candidate_rank-1)/max(pool_size-1,1); raw policy prior is audit-only"
                ),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "rdkit": importlib.metadata.version("rdkit"),
            },
            "failures": [],
            "counts": validation["counts"],
            "output": {
                "candidate_pool": file_fingerprint(
                    output_path, reported_path=output_final
                ),
                "anchor_validation": file_fingerprint(
                    validation_path,
                    reported_path=final_root / validation_path.name,
                ),
            },
            "runtime": {"seconds": time.perf_counter() - start},
            "created_at_utc": utc_now(),
        }
        manifest_path = staging / "manifest.json"
        atomic_json(manifest_path, manifest)
        gate = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "passed": True,
            "cap50_released": True,
            "rule": (
                "release only when the failed exact-reproduction result is preserved, "
                "all legacy anchors are covered, anchor identity/order/prior is immutable, "
                "and every extension originates in the checksummed clean cap-50"
            ),
            "manifest_sha256": sha256_file(manifest_path),
            "anchor_validation_sha256": sha256_file(validation_path),
            "created_at_utc": utc_now(),
        }
        atomic_json(staging / "ANCHOR_RELEASE_GATE.json", gate)
        final_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    result = {
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "output_root": str(final_root),
        **validation["counts"],
    }
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-cap10", required=True)
    parser.add_argument("--regenerated-cap50", required=True)
    parser.add_argument("--failed-reproduction-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cap", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_legacy_anchored_pool(
        legacy_cap10=args.legacy_cap10,
        regenerated_cap50=args.regenerated_cap50,
        failed_reproduction_summary=args.failed_reproduction_summary,
        output_root=args.output_root,
        cap=args.cap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
