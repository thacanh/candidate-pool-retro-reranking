"""Build the three prespecified WS-E candidate pools.

Inputs are the frozen AiZynthFinder legacy-anchored Top-50 pool and the
post-freeze LocalRetro Top-50 predictions.  Generator priors are normalized
within each product by rank.  The merged pool is a canonical union (up to 100
unique candidates), keeps the maximum normalized prior, and records both
source indicators without privileging a generator in tie-breaking.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

import pandas as pd

from rerank.data.prepare_localretro_current_dataset import (
    atomic_json,
    atomic_jsonl,
    canonical_fragments,
    fingerprint,
    sha256_file,
)


SCHEMA_VERSION = 1
PROTOCOL_ID = "ws-e-localretro-three-pools-filtered-v2"
EXPECTED_AIZ_PROTOCOL = "cap50-legacy-anchored-v1"
EXPECTED_LOCALRETRO_PROTOCOL = "localretro-top50-current-split-filtered-v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def product_key(smiles: str) -> tuple[str, ...]:
    return canonical_fragments(smiles)


def candidate_key(smiles: str) -> tuple[str, ...]:
    return canonical_fragments(smiles)


def _load_inventory(path: str | Path) -> tuple[list[tuple[str, ...]], dict[tuple[str, ...], str]]:
    order: list[tuple[str, ...]] = []
    raw: dict[tuple[str, ...], str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = product_key(str(record["product"]))
            if key in raw:
                raise RuntimeError("LocalRetro inventory contains a duplicate product.")
            raw[key] = str(record["product"])
            order.append(key)
    if not order:
        raise RuntimeError("LocalRetro inventory is empty.")
    return order, raw


def _load_aizynth(
    path: str | Path,
) -> tuple[list[tuple[str, ...]], dict[tuple[str, ...], str], dict[tuple[str, ...], list[dict[str, Any]]]]:
    order: list[tuple[str, ...]] = []
    raw_products: dict[tuple[str, ...], str] = {}
    pools: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    seen_candidates: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("protocol_id") != EXPECTED_AIZ_PROTOCOL:
                raise RuntimeError(f"AiZynth record protocol differs at line {line_number}.")
            product = str(record["product"])
            key = product_key(product)
            if key not in raw_products:
                raw_products[key] = product
                order.append(key)
            identity = candidate_key(str(record["reactant"]))
            if identity in seen_candidates[key]:
                raise RuntimeError("AiZynth pool contains a canonical duplicate.")
            seen_candidates[key].add(identity)
            pools[key].append({**record, "identity": identity})
    for key, candidates in pools.items():
        if len(candidates) > 50:
            raise RuntimeError("AiZynth pool exceeds Top-50.")
        ranks = [int(record["candidate_rank"]) for record in candidates]
        if ranks != list(range(1, len(candidates) + 1)):
            raise RuntimeError("AiZynth candidate ranks/order are not contiguous.")
    return order, raw_products, pools


def _load_localretro(
    path: str | Path,
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    pools: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    seen_candidates: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("protocol_id") != EXPECTED_LOCALRETRO_PROTOCOL:
                raise RuntimeError(f"LocalRetro record protocol differs at line {line_number}.")
            key = product_key(str(record["product"]))
            identity = candidate_key(str(record["reactant"]))
            if identity in seen_candidates[key]:
                raise RuntimeError("LocalRetro pool contains a canonical duplicate.")
            seen_candidates[key].add(identity)
            pools[key].append({**record, "identity": identity})
    for candidates in pools.values():
        if len(candidates) > 50:
            raise RuntimeError("LocalRetro pool exceeds Top-50.")
        ranks = [int(record["generator_rank"]) for record in candidates]
        if ranks != list(range(1, len(candidates) + 1)):
            raise RuntimeError("LocalRetro ranks/order are not contiguous.")
    return pools


def normalized_rank(rank: int, pool_size: int) -> float:
    if pool_size <= 0 or not 1 <= rank <= pool_size:
        raise ValueError("Rank is outside its pool.")
    return 1.0 - (rank - 1) / max(pool_size - 1, 1)


def _aiz_record(product: str, record: Mapping[str, Any], pool_size: int) -> dict[str, Any]:
    rank = int(record["candidate_rank"])
    return {
        "product": product,
        "reactant": str(record["reactant"]),
        "candidate_rank": rank,
        "generator_rank": rank,
        "prior": normalized_rank(rank, pool_size),
        "raw_generator_prior": float(record["prior"]),
        "source_aizynthfinder": 1,
        "source_localretro": 0,
        "aizynth_rank": rank,
        "localretro_rank": None,
        "aizynth_anchor_source": record.get("candidate_source"),
        "protocol_id": PROTOCOL_ID,
    }


def _local_record(product: str, record: Mapping[str, Any], pool_size: int) -> dict[str, Any]:
    rank = int(record["generator_rank"])
    return {
        "product": product,
        "reactant": str(record["reactant"]),
        "candidate_rank": rank,
        "generator_rank": rank,
        "prior": normalized_rank(rank, pool_size),
        "raw_generator_score": float(record["raw_score"]),
        "source_aizynthfinder": 0,
        "source_localretro": 1,
        "aizynth_rank": None,
        "localretro_rank": rank,
        "protocol_id": PROTOCOL_ID,
    }


def _merged_records(
    product: str,
    aizynth: list[dict[str, Any]],
    localretro: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in aizynth:
        identity = record["identity"]
        rank = int(record["candidate_rank"])
        merged[identity] = {
            "product": product,
            "reactant": str(record["reactant"]),
            "prior": normalized_rank(rank, len(aizynth)),
            "source_aizynthfinder": 1,
            "source_localretro": 0,
            "aizynth_rank": rank,
            "localretro_rank": None,
            "raw_aizynth_prior": float(record["prior"]),
            "raw_localretro_score": None,
            "identity": identity,
        }
    for record in localretro:
        identity = record["identity"]
        rank = int(record["generator_rank"])
        score = normalized_rank(rank, len(localretro))
        if identity in merged:
            merged[identity]["prior"] = max(float(merged[identity]["prior"]), score)
            merged[identity]["source_localretro"] = 1
            merged[identity]["localretro_rank"] = rank
            merged[identity]["raw_localretro_score"] = float(record["raw_score"])
        else:
            merged[identity] = {
                "product": product,
                "reactant": str(record["reactant"]),
                "prior": score,
                "source_aizynthfinder": 0,
                "source_localretro": 1,
                "aizynth_rank": None,
                "localretro_rank": rank,
                "raw_aizynth_prior": None,
                "raw_localretro_score": float(record["raw_score"]),
                "identity": identity,
            }
    ordered = sorted(
        merged.values(), key=lambda record: (-float(record["prior"]), record["identity"])
    )
    result: list[dict[str, Any]] = []
    for rank, record in enumerate(ordered, start=1):
        result.append(
            {
                **{key: value for key, value in record.items() if key != "identity"},
                "candidate_rank": rank,
                "protocol_id": PROTOCOL_ID,
            }
        )
    return result


def _coverage(
    source_csv: str | Path,
    pools: Mapping[str, Mapping[tuple[str, ...], set[tuple[str, ...]]]],
) -> dict[str, Any]:
    source = pd.read_csv(source_csv)
    results: dict[str, Any] = {}
    contribution = {
        "aizynth_only_correct": 0,
        "localretro_only_correct": 0,
        "both_generators_correct": 0,
        "neither_generator_correct": 0,
    }
    for pool_name, pool in pools.items():
        covered = 0
        by_split: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "covered": 0})
        for row in source.itertuples(index=False):
            product = product_key(str(row.products_smiles))
            reference = candidate_key(str(row.reactants_smiles))
            found = reference in pool.get(product, set())
            covered += int(found)
            split = str(row.set)
            by_split[split]["total"] += 1
            by_split[split]["covered"] += int(found)
        results[pool_name] = {
            "reactions_total": len(source),
            "reactions_covered": covered,
            "coverage": covered / len(source),
            "by_split": dict(by_split),
        }
    aiz = pools["aizynthfinder_only"]
    local = pools["localretro_only"]
    for row in source.itertuples(index=False):
        product = product_key(str(row.products_smiles))
        reference = candidate_key(str(row.reactants_smiles))
        in_aiz = reference in aiz.get(product, set())
        in_local = reference in local.get(product, set())
        if in_aiz and in_local:
            contribution["both_generators_correct"] += 1
        elif in_aiz:
            contribution["aizynth_only_correct"] += 1
        elif in_local:
            contribution["localretro_only_correct"] += 1
        else:
            contribution["neither_generator_correct"] += 1
    results["correct_candidate_contribution"] = contribution
    return results


def build_three_pools(
    *,
    aizynth_pool: str | Path,
    localretro_predictions: str | Path,
    localretro_inventory: str | Path,
    source_csv: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    order, raw_products, aizynth = _load_aizynth(aizynth_pool)
    inventory_order, inventory_products = _load_inventory(localretro_inventory)
    if order != inventory_order:
        raise RuntimeError("AiZynth and LocalRetro product identity/order differ.")
    localretro = _load_localretro(localretro_predictions)
    if not set(localretro).issubset(order):
        raise RuntimeError("LocalRetro predictions contain an unexpected product.")

    final = Path(output_root).resolve()
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite WS-E pools: {final}")
    staging = final.with_name(f".{final.name}.{os.getpid()}.tmp")
    staging.mkdir(parents=True)
    try:
        aiz_records: list[dict[str, Any]] = []
        local_records: list[dict[str, Any]] = []
        merged_records: list[dict[str, Any]] = []
        jaccards: list[float] = []
        global_overlap = 0
        pool_sets: dict[str, dict[tuple[str, ...], set[tuple[str, ...]]]] = {
            "aizynthfinder_only": {},
            "localretro_only": {},
            "merged": {},
        }
        for key in order:
            product = raw_products[key]
            aiz_pool = aizynth[key]
            local_pool = localretro.get(key, [])
            aiz_set = {record["identity"] for record in aiz_pool}
            local_set = {record["identity"] for record in local_pool}
            union = aiz_set | local_set
            overlap = aiz_set & local_set
            global_overlap += len(overlap)
            jaccards.append(len(overlap) / len(union) if union else 1.0)
            pool_sets["aizynthfinder_only"][key] = aiz_set
            pool_sets["localretro_only"][key] = local_set
            pool_sets["merged"][key] = union
            aiz_records.extend(_aiz_record(product, record, len(aiz_pool)) for record in aiz_pool)
            local_records.extend(_local_record(product, record, len(local_pool)) for record in local_pool)
            merged_records.extend(_merged_records(product, aiz_pool, local_pool))

        paths = {
            "aizynthfinder_only": staging / "aizynthfinder_only.jsonl",
            "localretro_only": staging / "localretro_only.jsonl",
            "merged": staging / "merged_canonical_union.jsonl",
        }
        atomic_jsonl(paths["aizynthfinder_only"], aiz_records)
        atomic_jsonl(paths["localretro_only"], local_records)
        atomic_jsonl(paths["merged"], merged_records)
        atomic_jsonl(
            staging / "products.jsonl",
            (
                {
                    "product_rank": index,
                    "product": raw_products[key],
                    "canonical_product": ".".join(key),
                }
                for index, key in enumerate(order)
            ),
        )
        coverage = _coverage(source_csv, pool_sets)
        overlap_report = {
            "products": len(order),
            "aizynth_candidates": len(aiz_records),
            "localretro_candidates": len(local_records),
            "merged_candidates": len(merged_records),
            "global_candidate_overlap": global_overlap,
            "global_jaccard": global_overlap / max(len(merged_records), 1),
            "mean_product_jaccard": mean(jaccards),
            "median_product_jaccard": median(jaccards),
            "products_without_localretro_candidate": sum(not localretro.get(key) for key in order),
            "maximum_merged_candidates_per_product": max(
                len(pool_sets["merged"][key]) for key in order
            ),
        }
        atomic_json(staging / "coverage.json", coverage)
        atomic_json(staging / "overlap.json", overlap_report)
        outputs = {
            name: fingerprint(path) for name, path in paths.items()
        }
        outputs.update(
            {
                "products": fingerprint(staging / "products.jsonl"),
                "coverage": fingerprint(staging / "coverage.json"),
                "overlap": fingerprint(staging / "overlap.json"),
            }
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "comparator": "AiZynthFinder-only cap50-legacy-anchored-v1 pool",
            "single_intended_change": "candidate generator/pool membership",
            "input_fingerprints": {
                "aizynthfinder": fingerprint(aizynth_pool),
                "localretro_predictions": fingerprint(localretro_predictions),
                "localretro_inventory": fingerprint(localretro_inventory),
                "source_csv_for_coverage": fingerprint(source_csv),
            },
            "settings": {
                "generator_cap": 50,
                "merged_cap": 100,
                "generator_prior": "1-(rank-1)/max(n-1,1)",
                "merged_prior": "maximum normalized generator prior",
                "canonical_identity": "isomeric; fragment-order invariant",
                "merged_tie_break": "canonical candidate identity; no generator priority",
                "source_indicators": ["source_aizynthfinder", "source_localretro"],
            },
            "counts": overlap_report,
            "coverage": coverage,
            "outputs": outputs,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "rdkit": importlib.metadata.version("rdkit"),
            },
            "runtime_seconds": time.perf_counter() - start,
            "created_at_utc": utc_now(),
        }
        atomic_json(staging / "manifest.json", manifest)
        gate = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "passed": True,
            "checks": {
                "product_identity_and_order_exact": True,
                "aizynthfinder_max_50": True,
                "localretro_max_50": True,
                "merged_max_100": overlap_report["maximum_merged_candidates_per_product"] <= 100,
                "canonical_deduplication": True,
                "source_indicators_present": True,
                "ground_truth_used_only_for_post_pool_coverage": True,
            },
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "created_at_utc": utc_now(),
        }
        atomic_json(staging / "POOL_RELEASE_GATE.json", gate)
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aizynth-pool", required=True)
    parser.add_argument("--localretro-predictions", required=True)
    parser.add_argument("--localretro-inventory", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = build_three_pools(
        aizynth_pool=args.aizynth_pool,
        localretro_predictions=args.localretro_predictions,
        localretro_inventory=args.localretro_inventory,
        source_csv=args.source_csv,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
