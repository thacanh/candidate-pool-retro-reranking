"""
MODULE 8 — Inference / Reranking (v2 — two-model ablation)
============================================================
Fair comparison matching the paper design:

    1. Baseline    — prior-sorted (AiZynthFinder output, no ML used)
    2. Reranker2D  — MLP trained on 5 2D features [f1,f3,f4,f9,f10]
                     Does NOT use UniMol embeddings
    3. Reranker3D  — MLP trained on 10 3D features [f1,f3,f4,f5,f6,f7,f8,f9,f10,f11]
                     USES UniMol embeddings

Usage in the paper:
    # Train 2 separate models (see train_colab_v3.ipynb):
    # model_2d = train(extractor_2d, feature_mode='2d')
    # model_3d = train(extractor_3d, feature_mode='3d')

    extractor_2d = FeatureExtractor(encoder, feature_mode='2d')
    extractor_3d = FeatureExtractor(encoder, feature_mode='3d')

    reranker_2d = Reranker2D(extractor_2d, model_2d)
    reranker_3d = Reranker3D(extractor_3d, model_3d)

    cmp = TripleComparison(reranker_2d, reranker_3d)
    result = cmp.compare(products_with_candidates, ground_truths)
    print(result.summary())

This isolates the true impact of the 3D embeddings because:
  - model_2d never sees the 3D features
  - model_3d is trained with all 3D features from scratch
  - Δ(3D − 2D) = net contribution of UniMol 3D embeddings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias
AiZynCandidate = Dict[str, object]



# Base Reranker


class Reranker:
    """
    Reranks AiZynthFinder candidates using a trained MLP scorer.

    This is the FULL reranker: it uses all 11 features including
    the 3D embedding features (f5, f6, f7, f8, f11).

    Args:
        feature_extractor: FeatureExtractor instance (from features.py).
        model:             Trained RankerMLP instance.
        name:              Name to identify in outputs (default "Reranker-3D").
    """

    def __init__(
        self,
        feature_extractor,
        model,
        name: str = "Reranker-3D",
    ) -> None:
        self.extractor = feature_extractor
        self.model = model
        self.name = name


    # Hook for subclasses to modify features before scoring


    def _postprocess_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Override in subclasses to transform the feature matrix before
        passing it to the MLP. Default: no-op (pass-through).

        Args:
            feature_matrix: shape (N, FEATURE_DIM)

        Returns:
            shape (N, FEATURE_DIM)
        """
        return feature_matrix


    # Core reranking logic


    def rerank(
        self,
        product_smiles: str,
        candidates: List[AiZynCandidate],
    ) -> Tuple[List[AiZynCandidate], np.ndarray]:
        """
        Rerank candidates for a single product.

        Args:
            product_smiles: Canonical product SMILES.
            candidates:     [{"smiles": str, "prior": float}, …]

        Returns:
            (ranked_candidates, scores) — sorted by MLP score DESC.
        """
        if not candidates:
            logger.warning("[%s] No candidates for: %s", self.name, product_smiles)
            return [], np.array([], dtype=np.float32)

        # Step 1: Sort by prior DESC → assign rank
        sorted_cands = sorted(
            candidates, key=lambda c: float(c["prior"]), reverse=True
        )
        cand_smiles = [str(c["smiles"]) for c in sorted_cands]
        cand_priors = [float(c["prior"]) for c in sorted_cands]
        ranks = list(range(len(sorted_cands)))

        # Step 2: Batch extract features (a single UniMol forward pass)
        feature_matrix = self.extractor.extract_features_batch(
            product_smiles=product_smiles,
            candidates=cand_smiles,
            log_probs=cand_priors,
            ranks=ranks,
        )
        # shape: (N, 11)

        # Step 3: Hook for subclass (e.g., zero out 3D features)
        feature_matrix = self._postprocess_features(feature_matrix)

        # Step 4: Score using the MLP
        scores = self.model.score_numpy(feature_matrix)  # (N,)

        # Step 5: Sort by score DESC
        order = np.argsort(scores)[::-1]
        ranked_candidates = [sorted_cands[i] for i in order]
        ranked_scores = scores[order]

        return ranked_candidates, ranked_scores


    # Batch API


    def rerank_batch(
        self,
        products_with_candidates: List[Tuple[str, List[AiZynCandidate]]],
    ) -> List[Tuple[List[AiZynCandidate], np.ndarray]]:
        """
        Rerank candidates for multiple products.

        Returns:
            List of (ranked_candidates, scores) aligned with the input.
        """
        results = []
        n = len(products_with_candidates)

        for i, (product_smi, candidates) in enumerate(products_with_candidates):
            logger.debug(
                "[%s] [%d/%d] Reranking: %s", self.name, i + 1, n, product_smi
            )
            ranked, scores = self.rerank(product_smi, candidates)
            results.append((ranked, scores))

        return results


    # Convenience


    def top_k(
        self,
        product_smiles: str,
        candidates: List[AiZynCandidate],
        k: int = 1,
    ) -> List[AiZynCandidate]:
        """Top-k reranked candidates."""
        ranked, _ = self.rerank(product_smiles, candidates)
        return ranked[:k]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"



# Reranker2D (2D-only model — does not use UniMol)


class Reranker2D(Reranker):
    """
    Reranker 2D-ONLY: uses a model trained with 5 2D features.

    Features: [prior, morgan_sim, atom_count_penalty, n_frags, heavy_atom_ratio]

    This is the ablation baseline — completely free of 3D embeddings.
    Comparing Reranker2D vs Reranker3D highlights the value of UniMol.

    IMPORTANT: Must be used with a model trained specifically with feature_extractor(mode='2d').
    Do not share the same model weights with Reranker3D.
    """

    def __init__(
        self,
        feature_extractor,   # FeatureExtractor(mode='2d')
        model,               # RankerMLP.for_2d()
        name: str = "Reranker-2D",
    ) -> None:
        super().__init__(feature_extractor, model, name=name)


# Backward-compat alias
RerankerNo3D = Reranker2D



# Baseline (prior-sorted, no reranking)


class BaselineRanker:
    """
    Pure baseline: returns candidates sorted by prior DESC (original AiZynthFinder output).
    Does not use the MLP model or extracted features.

    Used for comparison against RerankerNo3D and Reranker in the paper.
    """

    name: str = "Baseline-Prior"

    def rerank(
        self,
        product_smiles: str,
        candidates: List[AiZynCandidate],
    ) -> Tuple[List[AiZynCandidate], np.ndarray]:
        """
        Sort candidates by prior DESC — this matches the raw AiZynthFinder output.
        """
        if not candidates:
            return [], np.array([], dtype=np.float32)

        sorted_cands = sorted(
            candidates, key=lambda c: float(c["prior"]), reverse=True
        )
        scores = np.array(
            [float(c["prior"]) for c in sorted_cands], dtype=np.float32
        )
        return sorted_cands, scores

    def rerank_batch(
        self,
        products_with_candidates: List[Tuple[str, List[AiZynCandidate]]],
    ) -> List[Tuple[List[AiZynCandidate], np.ndarray]]:
        return [self.rerank(p, cands) for p, cands in products_with_candidates]

    def top_k(
        self,
        product_smiles: str,
        candidates: List[AiZynCandidate],
        k: int = 1,
    ) -> List[AiZynCandidate]:
        ranked, _ = self.rerank(product_smiles, candidates)
        return ranked[:k]

    def __repr__(self) -> str:
        return f"BaselineRanker(name={self.name!r})"



# Reranker3D alias (full 3D model)


# The Reranker class supports full 3D features by default.
# Define an alias here to make naming clearer in notebooks.
Reranker3D = Reranker

@dataclass
class EvaluationResult3Way:
    """
    Three-way comparison results: Baseline / No-3D Reranker / 3D Reranker.

    Used for presenting results in the paper in a clear table format.
    """

    ks: List[int]
    n_products: int

    # Top-k accuracy
    baseline_accuracy: Dict[int, float] = field(default_factory=dict)
    no3d_accuracy: Dict[int, float] = field(default_factory=dict)
    full3d_accuracy: Dict[int, float] = field(default_factory=dict)

    # MRR
    baseline_mrr: float = 0.0
    no3d_mrr: float = 0.0
    full3d_mrr: float = 0.0

    # System names (for display in the paper)
    baseline_name: str = "Baseline (Prior)"
    no3d_name: str = "Rerank (No-3D)"
    full3d_name: str = "Rerank (3D)"

    @property
    def delta_no3d_vs_baseline(self) -> Dict[int, float]:
        return {k: self.no3d_accuracy[k] - self.baseline_accuracy[k] for k in self.ks}

    @property
    def delta_3d_vs_baseline(self) -> Dict[int, float]:
        return {k: self.full3d_accuracy[k] - self.baseline_accuracy[k] for k in self.ks}

    @property
    def delta_3d_vs_no3d(self) -> Dict[int, float]:
        """This is the most important number in the paper — the net contribution of 3D features."""
        return {k: self.full3d_accuracy[k] - self.no3d_accuracy[k] for k in self.ks}

    @property
    def mrr_delta_no3d_vs_baseline(self) -> float:
        return self.no3d_mrr - self.baseline_mrr

    @property
    def mrr_delta_3d_vs_baseline(self) -> float:
        return self.full3d_mrr - self.baseline_mrr

    @property
    def mrr_delta_3d_vs_no3d(self) -> float:
        """Net contribution of 3D embeddings to MRR."""
        return self.full3d_mrr - self.no3d_mrr

    def summary(self, latex: bool = False) -> str:
        """
        Print comparison table for the 3 systems.

        Args:
            latex: If True, also print the LaTeX table representation to paste into the paper.
        """
        col_w = 18
        lines = [
            "=" * 80,
            f"  Within-Beam Closed-Set Reranking Evaluation Summary ({self.n_products} products)",
            "  Note: This is a closed-set reranking task where the ground truth is",
            "  already present in the candidate beam.",
            "=" * 80,
            (
                f"{'k':>4}  "
                f"{self.baseline_name:>{col_w}}  "
                f"{self.no3d_name:>{col_w}}  "
                f"{self.full3d_name:>{col_w}}  "
                f"{'d(3D-No3D)':>12}"
            ),
            "-" * 80,
        ]

        for k in self.ks:
            b  = self.baseline_accuracy[k]
            n  = self.no3d_accuracy[k]
            f  = self.full3d_accuracy[k]
            d  = self.delta_3d_vs_no3d[k]
            sign = "+" if d >= 0 else ""
            lines.append(
                f"{k:>4}  "
                f"{b:>{col_w}.4f}  "
                f"{n:>{col_w}.4f}  "
                f"{f:>{col_w}.4f}  "
                f"{sign}{d:>11.4f}"
            )

        lines.append("-" * 80)
        mrr_d = self.mrr_delta_3d_vs_no3d
        mrr_sign = "+" if mrr_d >= 0 else ""
        lines.append(
            f"{'MRR':>4}  "
            f"{self.baseline_mrr:>{col_w}.4f}  "
            f"{self.no3d_mrr:>{col_w}.4f}  "
            f"{self.full3d_mrr:>{col_w}.4f}  "
            f"{mrr_sign}{mrr_d:>11.4f}"
        )
        lines.append("=" * 80)

        # Highlight the main finding of the paper
        lines.append("\n  KEY CLAIM (3D vs No-3D improvement):")
        for k in self.ks:
            d = self.delta_3d_vs_no3d[k]
            rel = d / max(self.no3d_accuracy[k], 1e-9) * 100
            lines.append(
                f"    Top-{k:<2}: {'+' if d>=0 else ''}{d:.4f} "
                f"({'+' if d>=0 else ''}{rel:.1f}% relative)"
            )
        lines.append(
            f"    MRR  : {'+' if mrr_d>=0 else ''}{mrr_d:.4f} "
            f"({'+' if mrr_d>=0 else ''}"
            f"{mrr_d / max(self.no3d_mrr, 1e-9) * 100:.1f}% relative)"
        )

        text = "\n".join(lines)

        if latex:
            text += "\n\n" + self._to_latex()

        return text

    def _to_latex(self) -> str:
        """Generate LaTeX table to copy-paste directly into the paper."""
        rows = []
        header = (
            r"\begin{table}[h]" "\n"
            r"\centering" "\n"
            r"\caption{Retrosynthesis reranking results. "
            r"$\Delta$ denotes improvement of 3D over No-3D reranker.}" "\n"
            r"\label{tab:reranking_results}" "\n"
            r"\begin{tabular}{lcccccc}" "\n"
            r"\toprule" "\n"
            r"Method & Top-1 & Top-3 & Top-5 & Top-10 & MRR \\" "\n"
            r"\midrule"
        )
        rows.append(header)

        def _fmt(d: Dict[int, float]) -> str:
            vals = []
            for k in [1, 3, 5, 10]:
                vals.append(f"{d.get(k, 0.0):.4f}" if k in d else "--")
            return " & ".join(vals)

        rows.append(
            f"{self.baseline_name} & {_fmt(self.baseline_accuracy)} "
            f"& {self.baseline_mrr:.4f} \\\\"
        )
        rows.append(
            f"{self.no3d_name} & {_fmt(self.no3d_accuracy)} "
            f"& {self.no3d_mrr:.4f} \\\\"
        )
        rows.append(r"\midrule")
        rows.append(
            f"{self.full3d_name} & {_fmt(self.full3d_accuracy)} "
            f"& {self.full3d_mrr:.4f} \\\\"
        )
        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")
        rows.append(r"\end{table}")
        return "\n".join(rows)



# TripleComparison — main orchestrator for the paper


class TripleComparison:
    """
    Compares the 3 systems on the same dataset:
        1. Baseline (prior-sorted)
        2. Reranker No-3D (MLP without 3D features)
        3. Reranker 3D (MLP with full 11 features)

    This is the main class used in the inference pipeline of the paper.

    Args:
        reranker_2d: Reranker2D instance.
        reranker_3d: Reranker3D (full 3D Reranker) instance.
        ks:          k values for top-k accuracy.
    """

    def __init__(
        self,
        reranker_2d: Reranker,    # Reranker2D instance
        reranker_3d: Reranker,    # Reranker3D (full Reranker) instance
        ks: Optional[List[int]] = None,
    ) -> None:
        self.baseline = BaselineRanker()
        self.no3d = reranker_2d    # backward compat name
        self.full3d = reranker_3d
        self.ks = ks or [1, 3, 5, 10]

    def compare(
        self,
        products_with_candidates: List[Tuple[str, List[AiZynCandidate]]],
        ground_truths: List[str],
        output_csv: Optional[str] = None,
    ) -> EvaluationResult3Way:
        """
        Run evaluation on all 3 systems and return EvaluationResult3Way.

        Args:
            products_with_candidates: List of (product_smiles, candidates).
            ground_truths:            Aligned ground-truth reactant SMILES.
            output_csv:               Optional path to save detailed per-product hits/misses.

        Returns:
            EvaluationResult3Way containing all evaluation metrics.
        """
        from rerank.evaluate import (
            top_k_accuracy,
            mean_reciprocal_rank,
            _is_match,
        )

        n = len(products_with_candidates)
        logger.info("TripleComparison: evaluating %d products …", n)

        # Run evaluation on all 3 systems
        logger.info("  [1/3] Baseline (prior-sorted) …")
        baseline_results = self.baseline.rerank_batch(products_with_candidates)

        logger.info("  [2/3] Reranker No-3D …")
        no3d_results = self.no3d.rerank_batch(products_with_candidates)

        logger.info("  [3/3] Reranker 3D …")
        full3d_results = self.full3d.rerank_batch(products_with_candidates)

        # Extract candidate SMILES in order
        def _to_smiles_list(
            results: List[Tuple[List[AiZynCandidate], np.ndarray]]
        ) -> List[List[str]]:
            return [[str(c["smiles"]) for c in ranked] for ranked, _ in results]

        baseline_lists = _to_smiles_list(baseline_results)
        no3d_lists     = _to_smiles_list(no3d_results)
        full3d_lists   = _to_smiles_list(full3d_results)

        # Compute metrics
        result = EvaluationResult3Way(
            ks=self.ks,
            n_products=n,
            baseline_name=self.baseline.name,
            no3d_name=self.no3d.name,
            full3d_name=self.full3d.name,
        )

        for k in self.ks:
            result.baseline_accuracy[k] = top_k_accuracy(
                ground_truths, baseline_lists, k
            )
            result.no3d_accuracy[k] = top_k_accuracy(
                ground_truths, no3d_lists, k
            )
            result.full3d_accuracy[k] = top_k_accuracy(
                ground_truths, full3d_lists, k
            )

        result.baseline_mrr = mean_reciprocal_rank(ground_truths, baseline_lists)
        result.no3d_mrr     = mean_reciprocal_rank(ground_truths, no3d_lists)
        result.full3d_mrr   = mean_reciprocal_rank(ground_truths, full3d_lists)

        # Log summary
        print(result.summary())

        # Optional CSV export
        if output_csv:
            self._save_csv(
                result, output_csv,
                products_with_candidates, ground_truths,
                baseline_lists, no3d_lists, full3d_lists,
            )

        return result

    def _save_csv(
        self,
        result: EvaluationResult3Way,
        path: str,
        products_with_candidates,
        ground_truths,
        baseline_lists,
        no3d_lists,
        full3d_lists,
    ) -> None:
        """Export CSV with per-product hit/miss stats for all 3 systems."""
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed — skipping CSV export.")
            return

        from rerank.evaluate import _is_match

        rows = []
        for i, ((product, _), gt) in enumerate(
            zip(products_with_candidates, ground_truths)
        ):
            row: Dict = {"product_smiles": product, "ground_truth": gt}
            for k in self.ks:
                row[f"baseline_hit@{k}"] = int(
                    any(_is_match(c, gt) for c in baseline_lists[i][:k])
                )
                row[f"no3d_hit@{k}"] = int(
                    any(_is_match(c, gt) for c in no3d_lists[i][:k])
                )
                row[f"full3d_hit@{k}"] = int(
                    any(_is_match(c, gt) for c in full3d_lists[i][:k])
                )

            # Per-product reciprocal rank
            for prefix, cands in [
                ("baseline", baseline_lists[i]),
                ("no3d",     no3d_lists[i]),
                ("full3d",   full3d_lists[i]),
            ]:
                rr = 0.0
                for rank, cand in enumerate(cands, start=1):
                    if _is_match(cand, gt):
                        rr = 1.0 / rank
                        break
                row[f"{prefix}_rr"] = rr

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("Per-product results saved to %s", path)