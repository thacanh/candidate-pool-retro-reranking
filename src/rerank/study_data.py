"""Data utilities for the controlled 2D versus 2D+3D study.

This module deliberately separates candidate generation coverage from
closed-set reranking.  The legacy experiment builder randomly split products
after candidate generation and therefore mixed the official Chemformer
train/validation/test partitions.  The helpers below reconstruct labels from
the source reaction table and preserve reaction-level identifiers throughout
evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from rerank.dataset import PairwiseRankingDataset

logger = logging.getLogger(__name__)

STUDY_CACHE_SCHEMA = 1
OFFICIAL_SPLIT_BUNDLE_SCHEMA = 1


@lru_cache(maxsize=500_000)
def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Return RDKit canonical SMILES, or ``None`` for an invalid molecule."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


@lru_cache(maxsize=500_000)
def canonicalize_reactant_set(smiles: str) -> Optional[str]:
    """Canonicalize a dot-separated reactant set independent of fragment order."""
    parts: List[str] = []
    for fragment in str(smiles).split("."):
        fragment = fragment.strip()
        if not fragment:
            continue
        canonical = canonicalize_smiles(fragment)
        if canonical is None:
            return None
        parts.append(canonical)
    return ".".join(sorted(parts)) if parts else None


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path, include_sha256: bool = True) -> dict:
    path = Path(path)
    stat = path.stat()
    result = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = file_sha256(path)
    return result


@dataclass(frozen=True)
class ReactionRecord:
    reaction_id: int
    source_split: str
    product_smiles: str
    product_key: str
    ground_truth: str
    ground_truth_key: str
    reaction_class: Optional[str] = None

    def metadata(self) -> dict:
        return {
            "reaction_id": self.reaction_id,
            "source_split": self.source_split,
            "reaction_class": self.reaction_class,
        }


def load_reactions(
    source_csv: str | Path,
    metadata_csv: Optional[str | Path] = None,
) -> List[ReactionRecord]:
    """Load reaction rows while retaining the official source split and class."""
    frame = pd.read_csv(source_csv)
    required = {"reactants_smiles", "products_smiles", "set"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")

    frame = frame.reset_index(drop=True)
    frame["reaction_id"] = np.arange(len(frame), dtype=np.int64)
    if "reaction_class" not in frame.columns:
        frame["reaction_class"] = None

    if metadata_csv is not None:
        metadata = pd.read_csv(metadata_csv)
        if "reaction_id" not in metadata.columns or "reaction_class" not in metadata.columns:
            raise ValueError(
                "Metadata CSV must contain reaction_id and reaction_class columns."
            )
        if metadata["reaction_id"].duplicated().any():
            raise ValueError("reaction_id is not unique in metadata CSV.")
        keep = ["reaction_id", "reaction_class"]
        if "source_split" in metadata.columns:
            keep.append("source_split")
        frame = frame.drop(columns=["reaction_class"]).merge(
            metadata[keep], on="reaction_id", how="left", validate="one_to_one"
        )
        if "source_split" in metadata.columns:
            mismatch = frame["source_split"].notna() & (
                frame["source_split"].astype(str) != frame["set"].astype(str)
            )
            if mismatch.any():
                raise ValueError("Metadata source_split does not match the source CSV.")

    records: List[ReactionRecord] = []
    malformed_products = 0
    malformed_ground_truths = 0
    for row in frame.itertuples(index=False):
        product = str(row.products_smiles)
        ground_truth = str(row.reactants_smiles)
        product_key = canonicalize_smiles(product)
        ground_truth_key = canonicalize_reactant_set(ground_truth)
        if product_key is None:
            malformed_products += 1
            continue
        if ground_truth_key is None:
            malformed_ground_truths += 1
            continue
        reaction_class = getattr(row, "reaction_class", None)
        if pd.isna(reaction_class):
            reaction_class = None
        records.append(
            ReactionRecord(
                reaction_id=int(row.reaction_id),
                source_split=str(row.set),
                product_smiles=product,
                product_key=product_key,
                ground_truth=ground_truth,
                ground_truth_key=ground_truth_key,
                reaction_class=None if reaction_class is None else str(reaction_class),
            )
        )

    logger.info(
        "Loaded %d reactions (%d malformed products, %d malformed ground truths).",
        len(records),
        malformed_products,
        malformed_ground_truths,
    )
    return records


def load_candidate_pools(candidate_jsonl: str | Path) -> Tuple[Dict[str, List[dict]], dict]:
    """Load, canonicalize, and deduplicate candidate pools by molecular identity."""
    pools: Dict[str, Dict[str, dict]] = {}
    n_lines = 0
    n_malformed_json = 0
    n_invalid = 0
    n_duplicate = 0

    with open(candidate_jsonl, "r", encoding="utf-8") as handle:
        for raw in tqdm(handle, desc="Loading candidate pools", unit="line"):
            raw = raw.strip()
            if not raw:
                continue
            n_lines += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                n_malformed_json += 1
                continue

            product = str(record.get("product", ""))
            candidate = str(record.get("reactant", ""))
            product_key = canonicalize_smiles(product)
            candidate_key = canonicalize_reactant_set(candidate)
            if product_key is None or candidate_key is None:
                n_invalid += 1
                continue

            prior = float(record.get("prior", 0.0))
            product_pool = pools.setdefault(product_key, {})
            existing = product_pool.get(candidate_key)
            if existing is not None:
                n_duplicate += 1
            if existing is None or prior > float(existing["prior"]):
                product_pool[candidate_key] = {
                    "smiles": candidate,
                    "prior": prior,
                    "canonical_smiles": candidate_key,
                }

    ordered = {
        product: sorted(pool.values(), key=lambda item: float(item["prior"]), reverse=True)
        for product, pool in pools.items()
    }
    audit = {
        "candidate_lines": n_lines,
        "candidate_products": len(ordered),
        "malformed_json_lines": n_malformed_json,
        "invalid_candidate_lines": n_invalid,
        "deduplicated_candidate_lines": n_duplicate,
        "unique_candidates": sum(len(pool) for pool in ordered.values()),
    }
    logger.info("Candidate-pool audit: %s", audit)
    return ordered, audit


def candidate_rank(reaction: ReactionRecord, pool: Sequence[dict]) -> int:
    """One-based ground-truth rank in a prior-sorted pool; zero means absent."""
    for rank, candidate in enumerate(pool, start=1):
        if candidate.get("canonical_smiles") == reaction.ground_truth_key:
            return rank
    return 0


def compute_coverage(
    reactions: Sequence[ReactionRecord],
    pools: Dict[str, List[dict]],
    ks: Sequence[int] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Return reaction-level candidate coverage for every official split."""
    rows = []
    split_groups = [("all", list(reactions))]
    split_groups.extend(
        (split, [reaction for reaction in reactions if reaction.source_split == split])
        for split in sorted({reaction.source_split for reaction in reactions})
    )
    for split, subset in split_groups:
        ranks = np.asarray(
            [candidate_rank(reaction, pools.get(reaction.product_key, [])) for reaction in subset],
            dtype=np.int32,
        )
        counts = np.asarray(
            [len(pools.get(reaction.product_key, [])) for reaction in subset],
            dtype=np.int32,
        )
        row = {
            "source_split": split,
            "n_reactions": len(subset),
            "n_products": len({reaction.product_key for reaction in subset}),
            "missing_pool_count": int(np.sum(counts == 0)),
            "coverage_all": float(np.mean(ranks > 0)) if len(ranks) else 0.0,
            "not_in_pool_rate": float(np.mean(ranks == 0)) if len(ranks) else 0.0,
            "mean_candidate_count": float(np.mean(counts)) if len(counts) else 0.0,
        }
        for k in ks:
            row[f"coverage_at_{k}"] = (
                float(np.mean((ranks > 0) & (ranks <= k))) if len(ranks) else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _extract_training_feature_payload(
    reactions: Sequence[ReactionRecord],
    pools: Dict[str, List[dict]],
    feature_extractor,
    train_split: str,
    exclude_cross_split_train_products: bool,
) -> Tuple[List[dict], dict]:
    """Extract the shared training features and return their audit counts."""
    non_train_products = {
        reaction.product_key
        for reaction in reactions
        if reaction.source_split != train_split
    }

    train_ground_truths: Dict[str, set[str]] = {}
    train_source_smiles: Dict[str, str] = {}
    n_train_overlap_excluded = 0
    for reaction in reactions:
        if reaction.source_split != train_split:
            continue
        if exclude_cross_split_train_products and reaction.product_key in non_train_products:
            n_train_overlap_excluded += 1
            continue
        train_ground_truths.setdefault(reaction.product_key, set()).add(
            reaction.ground_truth_key
        )
        train_source_smiles.setdefault(reaction.product_key, reaction.product_smiles)

    train_products: List[dict] = []
    train_uncovered = 0
    train_without_negative = 0
    for product_key in tqdm(
        sorted(train_ground_truths), desc="Extracting train features", unit="product"
    ):
        pool = pools.get(product_key, [])
        positive_keys = train_ground_truths[product_key]
        positive_indices = [
            index
            for index, candidate in enumerate(pool)
            if candidate["canonical_smiles"] in positive_keys
        ]
        if not positive_indices:
            train_uncovered += 1
            continue
        negative_indices = [
            index for index in range(len(pool)) if index not in set(positive_indices)
        ]
        if not negative_indices:
            train_without_negative += 1
            continue
        candidates = [str(candidate["smiles"]) for candidate in pool]
        priors = [float(candidate["prior"]) for candidate in pool]
        features = feature_extractor.extract_features_batch(
            product_smiles=train_source_smiles[product_key],
            candidates=candidates,
            priors=priors,
            ranks=list(range(len(candidates))),
        )
        train_products.append(
            {
                "product_key": product_key,
                "product_smiles": train_source_smiles[product_key],
                "candidates": pool,
                "positive_indices": positive_indices,
                "negative_indices": negative_indices,
                "features": features.astype(np.float32),
            }
        )

    return train_products, {
        "train_products": len(train_products),
        "train_overlap_reactions_excluded": n_train_overlap_excluded,
        "train_products_uncovered": train_uncovered,
        "train_products_without_negative": train_without_negative,
    }


def _extract_evaluation_feature_payload(
    reactions: Sequence[ReactionRecord],
    pools: Dict[str, List[dict]],
    feature_extractor,
    eval_split: str,
) -> Tuple[dict, dict]:
    """Build one isolated official-split evaluation payload.

    Feature extraction is cached by canonical product within this split. The
    returned metadata retains both the source reaction ID and source split.
    """

    eval_reactions = [
        reaction for reaction in reactions if reaction.source_split == eval_split
    ]
    eval_feature_by_product: Dict[str, np.ndarray] = {}
    eval_pwc: List[Tuple[str, List[dict]]] = []
    eval_ground_truths: List[str] = []
    eval_metadata: List[dict] = []
    eval_features: List[np.ndarray] = []
    eval_uncovered = 0
    covered_product_keys: set[str] = set()
    uncovered_product_keys: set[str] = set()

    for reaction in tqdm(
        eval_reactions,
        desc=f"Extracting {eval_split} features",
        unit="reaction",
    ):
        pool = pools.get(reaction.product_key, [])
        rank = candidate_rank(reaction, pool)
        if rank == 0:
            eval_uncovered += 1
            uncovered_product_keys.add(reaction.product_key)
            continue
        covered_product_keys.add(reaction.product_key)
        if reaction.product_key not in eval_feature_by_product:
            candidates = [str(candidate["smiles"]) for candidate in pool]
            priors = [float(candidate["prior"]) for candidate in pool]
            eval_feature_by_product[reaction.product_key] = (
                feature_extractor.extract_features_batch(
                    product_smiles=reaction.product_smiles,
                    candidates=candidates,
                    priors=priors,
                    ranks=list(range(len(candidates))),
                ).astype(np.float32)
            )
        clean_pool = [
            {"smiles": candidate["smiles"], "prior": candidate["prior"]}
            for candidate in pool
        ]
        eval_pwc.append((reaction.product_smiles, clean_pool))
        eval_ground_truths.append(reaction.ground_truth)
        metadata = reaction.metadata()
        metadata.update(
            {
                "candidate_count": len(pool),
                "coverage_rank": rank,
            }
        )
        eval_metadata.append(metadata)
        eval_features.append(eval_feature_by_product[reaction.product_key])

    all_product_keys = {reaction.product_key for reaction in eval_reactions}
    payload = {
        "source_split": eval_split,
        "eval_pwc": eval_pwc,
        "eval_ground_truths": eval_ground_truths,
        "eval_metadata": eval_metadata,
        "eval_features": eval_features,
    }
    audit = {
        "source_split": eval_split,
        "reactions_total": len(eval_reactions),
        "reactions_covered": len(eval_pwc),
        "reactions_uncovered": eval_uncovered,
        "unique_products_total": len(all_product_keys),
        "unique_products_covered": len(covered_product_keys),
        "unique_products_with_uncovered_reactions": len(uncovered_product_keys),
        "unique_products_fully_uncovered": len(
            all_product_keys.difference(covered_product_keys)
        ),
    }
    return payload, audit


def _legacy_audit(
    feature_mode: str,
    train_split: str,
    eval_split: str,
    exclude_cross_split_train_products: bool,
    train_audit: dict,
    eval_audit: dict,
) -> dict:
    """Preserve the exact legacy audit keys used by frozen callers."""
    return {
        "schema_version": STUDY_CACHE_SCHEMA,
        "feature_mode": feature_mode,
        "train_split": train_split,
        "eval_split": eval_split,
        "exclude_cross_split_train_products": exclude_cross_split_train_products,
        "train_products": train_audit["train_products"],
        "train_overlap_reactions_excluded": train_audit[
            "train_overlap_reactions_excluded"
        ],
        "train_products_uncovered": train_audit["train_products_uncovered"],
        "train_products_without_negative": train_audit[
            "train_products_without_negative"
        ],
        "eval_reactions_total": eval_audit["reactions_total"],
        "eval_reactions_covered": eval_audit["reactions_covered"],
        "eval_reactions_uncovered": eval_audit["reactions_uncovered"],
        "eval_unique_products_covered": eval_audit["unique_products_covered"],
    }


def build_official_feature_cache(
    reactions: Sequence[ReactionRecord],
    pools: Dict[str, List[dict]],
    feature_extractor,
    feature_mode: str,
    train_split: str = "train",
    eval_split: str = "test",
    exclude_cross_split_train_products: bool = True,
) -> dict:
    """Extract features once using official splits, before seed-specific sampling.

    This is the frozen legacy entry point. Its top-level keys and audit schema
    intentionally remain unchanged; tuned protocols should use
    :func:`build_official_split_feature_bundle` instead.
    """
    train_products, train_audit = _extract_training_feature_payload(
        reactions,
        pools,
        feature_extractor,
        train_split,
        exclude_cross_split_train_products,
    )
    eval_payload, eval_audit = _extract_evaluation_feature_payload(
        reactions,
        pools,
        feature_extractor,
        eval_split,
    )

    audit = _legacy_audit(
        feature_mode,
        train_split,
        eval_split,
        exclude_cross_split_train_products,
        train_audit,
        eval_audit,
    )
    logger.info("Official-split feature-cache audit: %s", audit)
    return {
        "schema_version": STUDY_CACHE_SCHEMA,
        "feature_mode": feature_mode,
        "train_products": train_products,
        "eval_pwc": eval_payload["eval_pwc"],
        "eval_ground_truths": eval_payload["eval_ground_truths"],
        "eval_metadata": eval_payload["eval_metadata"],
        "eval_features": eval_payload["eval_features"],
        "audit": audit,
    }


def build_official_split_feature_bundle(
    reactions: Sequence[ReactionRecord],
    pools: Dict[str, List[dict]],
    feature_extractor,
    feature_mode: str,
    train_split: str = "train",
    validation_split: str = "valid",
    exclude_cross_split_train_products: bool = True,
) -> dict:
    """Build a validation-only bundle for hyperparameter selection.

    Training features are extracted exactly once, followed only by official
    validation features. Test reactions are neither inspected nor represented
    in the returned object. They can be extracted only after selection is
    frozen via :func:`attach_post_selection_test_payload`.
    """
    if train_split == validation_split:
        raise ValueError("train and validation split names must be distinct")

    train_products, train_audit = _extract_training_feature_payload(
        reactions,
        pools,
        feature_extractor,
        train_split,
        exclude_cross_split_train_products,
    )
    validation_payload, validation_audit = _extract_evaluation_feature_payload(
        reactions,
        pools,
        feature_extractor,
        validation_split,
    )

    return {
        "bundle_schema_version": OFFICIAL_SPLIT_BUNDLE_SCHEMA,
        "feature_cache_schema_version": STUDY_CACHE_SCHEMA,
        "feature_mode": feature_mode,
        "train_split": train_split,
        "validation_split": validation_split,
        "exclude_cross_split_train_products": exclude_cross_split_train_products,
        "train_products": train_products,
        "validation_payload": validation_payload,
        "audit": {
            "train": train_audit,
            "validation": validation_audit,
        },
    }


def _validated_selection_freeze_record(record: dict) -> dict:
    """Require explicit evidence that model selection has been frozen."""
    if not isinstance(record, dict):
        raise PermissionError(
            "Post-selection test extraction requires a frozen-selection record."
        )
    fingerprint_fields = (
        "selected_config_fingerprint",
        "checkpoint_fingerprint",
    )
    fingerprint_pattern = re.compile(r"sha256:[0-9a-fA-F]{64}")
    fingerprints = [
        str(record.get(field, "")).strip() for field in fingerprint_fields
    ]
    if not any(fingerprint_pattern.fullmatch(value) for value in fingerprints):
        raise PermissionError(
            "Post-selection test extraction requires a selected configuration "
            "or checkpoint fingerprint formatted as sha256:<64 hex digits>."
        )
    return dict(record)


def attach_post_selection_test_payload(
    selection_bundle: dict,
    reactions: Sequence[ReactionRecord],
    pools: Dict[str, List[dict]],
    feature_extractor,
    frozen_selection_record: dict,
    test_split: str = "test",
) -> dict:
    """Return a new bundle with test data attached after frozen selection.

    Freeze evidence is validated before any test feature extraction occurs.
    The validation-only input bundle is not mutated.
    """
    freeze_record = _validated_selection_freeze_record(frozen_selection_record)
    if selection_bundle.get("bundle_schema_version") != OFFICIAL_SPLIT_BUNDLE_SCHEMA:
        raise ValueError("Unsupported official-split selection bundle schema.")
    if test_split in {
        selection_bundle.get("train_split"),
        selection_bundle.get("validation_split"),
    }:
        raise ValueError("Test split must be distinct from train and validation splits.")
    if "post_selection_test" in selection_bundle:
        raise ValueError("A post-selection test payload is already attached.")

    test_payload, test_audit = _extract_evaluation_feature_payload(
        reactions,
        pools,
        feature_extractor,
        test_split,
    )
    result = dict(selection_bundle)
    result["audit"] = dict(selection_bundle["audit"])
    result["audit"]["post_selection_test"] = test_audit
    result["post_selection_test"] = {
        "test_split": test_split,
        "frozen_selection_record": freeze_record,
        "payload": test_payload,
    }
    return result


def selection_feature_cache_view(
    bundle: dict,
    selection_split: Optional[str] = None,
) -> dict:
    """Return a validation-only legacy-shaped cache for model selection.

    The explicit split guard fails closed if selection code requests the test
    payload. The returned object contains no test payload or test metadata.
    """
    validation_split = str(bundle.get("validation_split", ""))
    requested_split = validation_split if selection_split is None else selection_split
    if requested_split != validation_split:
        raise PermissionError(
            "Model selection is restricted to the official validation split; "
            f"requested {requested_split!r}, expected {validation_split!r}."
        )

    payload = bundle.get("validation_payload")
    if not isinstance(payload, dict):
        raise ValueError("Bundle does not contain its declared validation payload.")
    metadata = payload.get("eval_metadata", [])
    if any(item.get("source_split") != validation_split for item in metadata):
        raise ValueError("Validation payload contains a non-validation reaction.")

    train_audit = bundle["audit"]["train"]
    eval_audit = bundle["audit"]["validation"]
    audit = _legacy_audit(
        bundle["feature_mode"],
        bundle["train_split"],
        validation_split,
        bool(bundle["exclude_cross_split_train_products"]),
        train_audit,
        eval_audit,
    )
    return {
        "schema_version": STUDY_CACHE_SCHEMA,
        "feature_mode": bundle["feature_mode"],
        "train_products": bundle["train_products"],
        "eval_pwc": payload["eval_pwc"],
        "eval_ground_truths": payload["eval_ground_truths"],
        "eval_metadata": payload["eval_metadata"],
        "eval_features": payload["eval_features"],
        "audit": audit,
    }


def make_pairwise_dataset(
    feature_cache: dict,
    seed: int,
    max_neg_per_pos: Optional[int] = 5,
    negative_mining: Literal["random", "mixed"] = "random",
) -> PairwiseRankingDataset:
    """Construct seed-specific training pairs from a shared feature cache."""
    rng = random.Random(seed)
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for product in feature_cache["train_products"]:
        features = product["features"]
        negatives = list(product["negative_indices"])
        for positive_index in product["positive_indices"]:
            selected = negatives
            if max_neg_per_pos is not None and len(negatives) > max_neg_per_pos:
                if negative_mining == "mixed":
                    # Candidates are prior-sorted, so lower indices are harder.
                    hard_count = max_neg_per_pos // 2
                    random_count = max_neg_per_pos - hard_count
                    hard = sorted(negatives)[:hard_count]
                    remaining = [index for index in negatives if index not in set(hard)]
                    selected = hard + rng.sample(
                        remaining, min(random_count, len(remaining))
                    )
                else:
                    selected = rng.sample(negatives, max_neg_per_pos)
            for negative_index in selected:
                pairs.append(
                    (
                        features[positive_index].copy(),
                        features[negative_index].copy(),
                    )
                )
    if not pairs:
        raise ValueError("No pairwise training samples could be constructed.")
    logger.info("Constructed %d training pairs for seed %d.", len(pairs), seed)
    return PairwiseRankingDataset(pairs, seed=seed)


def reaction_records_as_dicts(records: Iterable[ReactionRecord]) -> List[dict]:
    """Small serialization helper used by audit scripts and tests."""
    return [asdict(record) for record in records]
