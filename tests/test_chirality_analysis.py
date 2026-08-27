import pandas as pd

from rerank.analysis.analyze_chirality_sensitivity import (
    _align,
    canonical_fragment_set,
    is_stereo_only_reference_difference,
)


def test_stereo_only_reference_difference_is_fragment_order_invariant() -> None:
    reference = "CCO.F[C@H](Cl)Br"
    stereo_mismatch = "F[C@@H](Cl)Br.CCO"
    constitutional_mismatch = "F[C@@H](Cl)I.CCO"

    assert canonical_fragment_set(
        reference, preserve_stereochemistry=False
    ) == canonical_fragment_set(stereo_mismatch, preserve_stereochemistry=False)
    assert is_stereo_only_reference_difference(stereo_mismatch, reference)
    assert not is_stereo_only_reference_difference(reference, reference)
    assert not is_stereo_only_reference_difference(constitutional_mismatch, reference)


def test_align_requires_identical_scientific_identity() -> None:
    common = {
        "reaction_id": [1],
        "product_smiles": ["CCO"],
        "ground_truth": ["CC.O"],
        "reranked_top1": ["CC.O"],
        "reranked_hit@1": [1],
        "reranked_rr": [1.0],
    }
    false_frame = pd.DataFrame(common)
    true_frame = pd.DataFrame(common)

    # The production loader establishes the 3,985-row denominator; this unit
    # isolates the merge semantics by temporarily repeating unique identities.
    false_frame = pd.concat(
        [false_frame.assign(reaction_id=index) for index in range(3_985)],
        ignore_index=True,
    )
    true_frame = false_frame.copy()
    aligned = _align(false_frame, true_frame)

    assert len(aligned) == 3_985
    assert aligned["reaction_id"].is_unique
