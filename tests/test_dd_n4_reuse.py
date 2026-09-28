"""Approved frozen-prediction reuse never silently accepts a new rank discrepancy."""
import numpy as np
import pytest
from rerank.experiments.run_dd_n4_historical import checked_frozen_reuse


def test_approved_reuse_keeps_frozen_not_favourable_rank():
    audit = {'mismatches': [{'reaction_id': 2, 'computed': 1, 'frozen': 2},
                            {'reaction_id': 3, 'computed': 2, 'frozen': 1}]}
    actual = np.array([3, 1, 2], dtype=np.uint8)
    result = checked_frozen_reuse([1, 2, 3], actual, {1: 3, 2: 2, 3: 1}, audit)
    assert result.tolist() == [3, 2, 1]
    assert actual.tolist() == [3, 1, 2]


def test_new_discrepancy_stops():
    with pytest.raises(ValueError, match='approved audit'):
        checked_frozen_reuse([1], [2], {1: 1}, {'mismatches': []})


def test_disappeared_discrepancy_stops():
    with pytest.raises(ValueError, match='approved audit'):
        checked_frozen_reuse([1], [1], {1: 1},
            {'mismatches': [{'reaction_id': 1, 'computed': 2, 'frozen': 1}]})


def test_changed_coverage_stops():
    with pytest.raises(ValueError, match='coverage'):
        checked_frozen_reuse([1], [1], {2: 1}, {'mismatches': []})


def test_exact_seed_unchanged():
    assert checked_frozen_reuse([1], [1], {1: 1}, {'mismatches': []}).tolist() == [1]


def test_real_five_frozen_models_when_requested(tmp_path, monkeypatch):
    """Opt-in read-only replay; outputs go to pytest temp, never scientific roots."""
    import os
    import json
    from pathlib import Path
    from types import SimpleNamespace as NS
    from rerank.experiments import run_dd_n4_historical as h
    from rerank.experiments.dd_remote_gate import fingerprint
    if os.environ.get('DD_N4_REAL_REPLAY_QA') != '1':
        pytest.skip('explicit local five-model replay QA only')
    root = Path('outputs/jcheminform_revision/encoder_controls/morgan_atom')
    old = json.loads((root / 'ranking/freeze.json').read_text())
    audit = Path(os.environ['DD_N4_AUDIT_PATH'])
    monkeypatch.setattr(h, 'SEEDS', tuple(range(42, 47)))
    monkeypatch.setattr(h, 'require_approval', lambda *_: {'n4_replay_reuse': {'audit_sha256': h.REUSE_AUDIT_SHA256}})
    frozen = dict(protocol_id=h.PROTOCOL_ID, seeds=list(range(42, 47)),
        existing_control_freeze=fingerprint(root / 'ranking/freeze.json'),
        comparator='QA ONLY', single_intended_change='QA replay only',
        trials={str(s): {'model_root': str((root / 'ranking/artifacts').resolve()),
                         'trial': old['selected_control']['trials'][str(s)]} for s in range(42, 47)})
    (tmp_path / 'freeze.json').write_text(json.dumps(frozen))
    result = h.evaluate(NS(approval_record='unused', analysis_plan='unused', output_root=tmp_path,
        existing_control_freeze=root / 'ranking/freeze.json', control_cache=root / 'cache/train_test.pkl',
        result_dir=tmp_path / 'qa_results', device='cpu', reproduction_audit=audit,
        historical_baseline_predictions='outputs/jcheminform_revision/tuned_primary/conformer_seed_42/g1_20seed/test_results/predictions',
        existing_control_predictions=root / 'ranking/test_results/predictions'))
    assert result['replay_rank_disagreements'] == 7
    assert result['first_five_exact_reproduction'] is False
    with np.load(tmp_path / 'qa_results/reaction_ranks.npz') as arrays:
        assert arrays['augmented_ranks'].shape == (5, 3985)
        assert np.count_nonzero(arrays['augmented_ranks'] != arrays['recomputed_augmented_ranks']) == 7
