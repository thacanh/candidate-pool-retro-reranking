import pandas as pd
import pytest

from rerank.analysis.analyze_salt_sensitivity import EXPECTED_COVERED_REACTIONS, align_predictions


def _frame(reaction_ids=range(EXPECTED_COVERED_REACTIONS)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": reaction_ids,
            "product_smiles": ["CCO"] * len(reaction_ids),
            "ground_truth": ["CC"] * len(reaction_ids),
            "reranked_top1": ["CC"] * len(reaction_ids),
            "reranked_hit@1": [1] * len(reaction_ids),
            "reranked_rr": [1.0] * len(reaction_ids),
        }
    )


def test_align_predictions_requires_identical_pairing_and_ground_truth() -> None:
    aligned = align_predictions(_frame(), _frame())
    assert len(aligned) == EXPECTED_COVERED_REACTIONS

    mismatched = _frame()
    mismatched.loc[0, "ground_truth"] = "CN"
    with pytest.raises(ValueError, match="ground_truth"):
        align_predictions(_frame(), mismatched)
