"""
MODULE 9 — Evaluation (AiZynthFinder edition)
===============================================
Compares baseline AiZynthFinder ranking (by prior) vs. MLP-reranked results.

Metrics:
    top_k_accuracy(k): fraction of products where ground-truth reactant
                       appears within the first k candidates.
    mrr:               Mean Reciprocal Rank — more informative than Top-k
                       because it rewards getting the answer higher up.

FIX #6 (evaluation):
    Top-k = 1.0 is vacuous when train set = top-10 candidates (closed set).
    Added MRR as the primary metric.
    Added a warning when Top-k equals the candidate set size (always 1.0).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

AiZynCandidate = Dict[str, object]



# SMILES matching helpers


from functools import lru_cache

@lru_cache(maxsize=None)
def _canonicalize(smiles: str) -> str:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        c = Chem.MolToSmiles(mol, canonical=True)
        return c if c else smiles
    except Exception:
        return smiles


@lru_cache(maxsize=None)
def _sorted_canonical(smiles: str) -> str:
    parts = sorted(_canonicalize(p.strip()) for p in smiles.split(".") if p.strip())
    return ".".join(parts)


def _is_match(candidate: str, ground_truth: str) -> bool:
    try:
        return _sorted_canonical(candidate) == _sorted_canonical(ground_truth)
    except Exception:
        return False



# Core metric functions


def top_k_accuracy(
    ground_truths: List[str],
    candidate_lists: List[List[str]],
    k: int,
) -> float:
    """
    Compute top-k accuracy.

    Args:
        ground_truths:   Ground-truth reactant SMILES per product.
        candidate_lists: Ordered candidate SMILES per product (best → worst).
        k:               The k in top-k accuracy.

    Returns:
        Float in [0, 1].
    """
    if not ground_truths:
        return 0.0
    hits = sum(
        1 for gt, cands in zip(ground_truths, candidate_lists)
        if any(_is_match(c, gt) for c in cands[:k])
    )
    return hits / len(ground_truths)


def mean_reciprocal_rank(
    ground_truths: List[str],
    candidate_lists: List[List[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR = mean(1 / rank_of_first_correct_candidate).
    More informative than Top-k because it penalises correct answers
    at rank 2, 3, etc. rather than treating all hits equally.

    Returns 0.0 if no ground truth is found in any candidate list.
    """
    if not ground_truths:
        return 0.0
    rrs = []
    for gt, cands in zip(ground_truths, candidate_lists):
        rr = 0.0
        for rank, cand in enumerate(cands, start=1):
            if _is_match(cand, gt):
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return float(np.mean(rrs))


def _check_topk_validity(k: int, candidate_lists: List[List[str]]) -> None:
    """
    FIX #6: Warn when k >= candidate set size, which makes Top-k vacuous.
    e.g. Top-10 = 1.0 when each product has exactly 10 candidates.
    """
    if not candidate_lists:
        return
    max_cands = max(len(c) for c in candidate_lists)
    if k >= max_cands:
        logger.warning(
            "⚠️  Top-%d with max %d candidates per product: this metric is VACUOUS "
            "(always 1.0 if any positive exists). Use Top-1, Top-3, or MRR instead.",
            k, max_cands,
        )



# Result dataclass


@dataclass
class EvaluationResult:
    """Holds per-k accuracy for AiZynthFinder baseline and MLP-reranked."""

    ks:                  List[int]
    baseline_accuracy:   Dict[int, float]
    reranked_accuracy:   Dict[int, float]
    baseline_mrr:        float
    reranked_mrr:        float
    n_products:          int
    product_smiles:      Optional[List[str]]          = None
    ground_truths:       Optional[List[str]]          = None
    baseline_candidates: Optional[List[List[str]]]   = None
    reranked_candidates: Optional[List[List[str]]]   = None
    reranked_scores:     Optional[List[List[float]]] = None
    reaction_metadata:   Optional[List[dict]]        = None

    @property
    def delta(self) -> Dict[int, float]:
        return {k: self.reranked_accuracy[k] - self.baseline_accuracy[k] for k in self.ks}

    @property
    def mrr_delta(self) -> float:
        return self.reranked_mrr - self.baseline_mrr

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"Within-Beam Closed-Set Reranking Evaluation Summary ({self.n_products} products)",
            "Note: This is a closed-set reranking task where the ground truth is",
            "already present in the candidate beam.",
            "=" * 70,
            f"{'k':>4}  {'Baseline (AiZyn)':>16}  {'Reranked (MLP)':>14}  {'diff':>10}",
            "-" * 70,
        ]
        for k in self.ks:
            b = self.baseline_accuracy[k]
            r = self.reranked_accuracy[k]
            d = r - b
            lines.append(f"{k:>4}  {b:>16.4f}  {r:>14.4f}  {d:>+10.4f}")

        lines.append("-" * 70)
        lines.append(
            f"{'MRR':>4}  {self.baseline_mrr:>16.4f}  "
            f"{self.reranked_mrr:>14.4f}  {self.mrr_delta:>+10.4f}"
        )
        lines.append("=" * 70)
        if any(self.baseline_accuracy.get(k, 0) >= 0.99 for k in self.ks):
            lines.append(
                "NOTE: Top-k = 1.0 likely means k >= candidate set size (vacuous). "
                "Focus on Top-1 and MRR."
            )
        return "\n".join(lines)



# Main evaluation function


def evaluate_reranking(
    products_with_candidates: List[Tuple[str, List[AiZynCandidate]]],
    ground_truths: List[str],
    reranker,
    ks: Optional[List[int]] = None,
    output_csv: Optional[str] = None,
    precomputed_reranked_results=None,
    reaction_metadata: Optional[List[dict]] = None,
) -> EvaluationResult:
    """
    Compare AiZynthFinder baseline (prior-sorted) vs. MLP-reranked accuracy.

    Args:
        products_with_candidates:
            List of (product_smiles, candidates) where
            candidates = [{"smiles": str, "prior": float}, …].
        ground_truths:
            Ground-truth reactant SMILES aligned with products_with_candidates.
        reranker:
            A Reranker instance (Module 8).
        ks:
            k values for top-k accuracy. Default: [1, 3, 5, 10].
        output_csv:
            If given, save per-product hit/miss CSV to this path.

    Returns:
        An EvaluationResult instance.
    """
    if ks is None:
        ks = [1, 3, 5, 10]
    if reaction_metadata is not None and len(reaction_metadata) != len(ground_truths):
        raise ValueError(
            "reaction_metadata must be aligned one-to-one with ground_truths."
        )

    products       = [p for p, _ in products_with_candidates]
    all_candidates = [cands for _, cands in products_with_candidates]

    logger.info("Evaluating on %d products …", len(products))

    # Baseline: prior-sorted order
    baseline_cand_lists: List[List[str]] = [
        [str(c["smiles"]) for c in cands]
        for cands in all_candidates
    ]

    for k in ks:
        _check_topk_validity(k, baseline_cand_lists)

    baseline_acc: Dict[int, float] = {
        k: top_k_accuracy(ground_truths, baseline_cand_lists, k) for k in ks
    }
    baseline_mrr = mean_reciprocal_rank(ground_truths, baseline_cand_lists)
    logger.info(
        "Baseline top-k: %s  |  MRR: %.4f",
        {k: f"{v:.4f}" for k, v in baseline_acc.items()},
        baseline_mrr,
    )

    # Reranking via MLP
    if precomputed_reranked_results is not None:
        logger.info("Using pre-computed reranked results (no encoder needed) …")
        reranked_results = precomputed_reranked_results
    else:
        logger.info("Running MLP reranker …")
        reranked_results = reranker.rerank_batch(products_with_candidates)

    reranked_cand_lists: List[List[str]] = [
        [str(c["smiles"]) for c in ranked]
        for ranked, _ in reranked_results
    ]
    reranked_scores: List[List[float]] = [
        np.asarray(scores, dtype=float).tolist()
        for _, scores in reranked_results
    ]

    reranked_acc: Dict[int, float] = {
        k: top_k_accuracy(ground_truths, reranked_cand_lists, k) for k in ks
    }
    reranked_mrr = mean_reciprocal_rank(ground_truths, reranked_cand_lists)
    logger.info(
        "Reranked top-k: %s  |  MRR: %.4f",
        {k: f"{v:.4f}" for k, v in reranked_acc.items()},
        reranked_mrr,
    )

    result = EvaluationResult(
        ks=ks,
        baseline_accuracy=baseline_acc,
        reranked_accuracy=reranked_acc,
        baseline_mrr=baseline_mrr,
        reranked_mrr=reranked_mrr,
        n_products=len(products),
        product_smiles=products,
        ground_truths=ground_truths,
        baseline_candidates=baseline_cand_lists,
        reranked_candidates=reranked_cand_lists,
        reranked_scores=reranked_scores,
        reaction_metadata=reaction_metadata,
    )

    print(result.summary())

    if output_csv and _PANDAS_AVAILABLE:
        _save_csv(result, output_csv)

    return result



# CSV export helper


def _save_csv(result: EvaluationResult, path: str) -> None:
    rows = []
    for i, (product, gt) in enumerate(
        zip(result.product_smiles or [], result.ground_truths or [])
    ):
        baseline_cands = (result.baseline_candidates or [[]])[i]
        reranked_cands = (result.reranked_candidates or [[]])[i]

        metadata = (
            result.reaction_metadata[i]
            if result.reaction_metadata is not None
            else {}
        )
        row: Dict = dict(metadata)
        row.update({"product_smiles": product, "ground_truth": gt})

        for k in result.ks:
            row[f"baseline_hit@{k}"] = int(
                any(_is_match(c, gt) for c in baseline_cands[:k])
            )
            row[f"reranked_hit@{k}"] = int(
                any(_is_match(c, gt) for c in reranked_cands[:k])
            )

        # Reciprocal rank
        for prefix, cands in [("baseline", baseline_cands), ("reranked", reranked_cands)]:
            rr = 0.0
            matched_rank = 0
            for rank, cand in enumerate(cands, start=1):
                if _is_match(cand, gt):
                    rr = 1.0 / rank
                    matched_rank = rank
                    break
            row[f"{prefix}_rr"] = rr
            row[f"{prefix}_rank"] = matched_rank

        row["baseline_top1"] = baseline_cands[0] if baseline_cands else ""
        row["reranked_top1"] = reranked_cands[0] if reranked_cands else ""
        row["baseline_candidates_json"] = json.dumps(
            baseline_cands, ensure_ascii=False, separators=(",", ":")
        )
        row["reranked_candidates_json"] = json.dumps(
            reranked_cands, ensure_ascii=False, separators=(",", ":")
        )
        scores = (
            result.reranked_scores[i]
            if result.reranked_scores is not None
            else []
        )
        row["reranked_scores_json"] = json.dumps(
            scores, separators=(",", ":")
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Evaluation results saved to %s", path)
