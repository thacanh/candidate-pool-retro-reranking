"""Synthetic S4 gates; no scientific dataset or official-test access."""
import importlib.util
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rerank.experiments import run_pool_ranking as trainer
from rerank.tuning import enumerate_d1_grid

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    's4', ROOT / 'workflows/transfer/search_hyperparameters.py'
)
s4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s4)


def test_grid_and_tie_rule():
    assert len(s4.GRID) == 81
    rows = [{'config': c, 'mean_validation_mrr': 0.5} for c in s4.GRID]
    rows[3]['mean_validation_mrr'] += 5e-13
    assert s4.select_winner(rows)['config']['index'] == 0
    rows[3]['mean_validation_mrr'] = 0.6
    assert s4.select_winner(rows)['config']['index'] == 3
    with pytest.raises(ValueError):
        s4.select_winner(rows[:-1])


def test_config_protocol_is_not_a_forged_primary():
    c = asdict(enumerate_d1_grid()[0])
    primary = {
        'protocol_id': s4.PROTOCOL,
        'record_kind': 'validation_grid_candidate',
        'test_partition_loaded': False,
        'selected_baseline': {'config': c},
    }
    selection = {'protocol_id': s4.PROTOCOL, 'pool_name': 'merged', 'test_partition_loaded': False}
    assert trainer._fit_config(primary, selection, 'baseline') == c
    with pytest.raises(ValueError):
        trainer._fit_config(primary, {**selection, 'protocol_id': 'other'}, 'baseline')
    with pytest.raises(ValueError):
        trainer._fit_config({**primary, 'test_partition_loaded': True}, selection, 'baseline')
    legacy = {'protocol_id': 'cap10-tuned-v1', 'selected_baseline': {'config': c}}
    assert trainer._fit_config(legacy, {}, 'baseline') == c


def test_synthetic_fit_and_resume_tamper_gate(tmp_path, monkeypatch):
    import torch

    torch.set_num_threads(1)
    monkeypatch.setattr(trainer, 'MAX_EPOCHS', 2)
    monkeypatch.setattr(trainer, 'PATIENCE', 2)
    pool = 'merged'
    c = s4.GRID[0]
    arm = 'baseline'
    seed = 42
    selection_root = tmp_path / 'inputs' / pool
    selection_root.mkdir(parents=True)
    rng = np.random.default_rng(4)
    np.savez(
        selection_root / 'train_pairs_seed_42.npz',
        positive=rng.normal(size=(16, 7)).astype('f'),
        negative=rng.normal(size=(16, 7)).astype('f'),
    )
    np.savez(
        selection_root / 'official_valid.npz',
        features=rng.normal(size=(4, 7)).astype('f'),
        offsets=np.array([0, 2, 4]),
        match_mask=np.array([True, False, True, False]),
    )

    def rec(name):
        p = selection_root / name
        return {
            'path': '/nonexistent_remote/' + name,
            'sha256': s4.sha(p),
            'size_bytes': p.stat().st_size,
        }

    selection = {
        'protocol_id': s4.PROTOCOL,
        'pool_name': pool,
        'seeds': [42],
        'test_partition_loaded': False,
        'candidate_cap': 10,
        'baseline_columns': [0, 1, 5, 6],
        'augmented_columns': list(range(7)),
        'source_indicators_in_both_arms': False,
        'train_pairs': {'42': rec('train_pairs_seed_42.npz')},
        'validation': rec('official_valid.npz'),
    }
    s4.write(selection_root / 'manifest.json', selection)
    attempt = (
        tmp_path / 'results/N3_S4' / pool / 'trials' / s4.task_id(c, arm, seed) / 'attempt_001'
    )
    candidate = attempt / 'candidate.json'
    s4.write(
        candidate,
        {
            'protocol_id': s4.PROTOCOL,
            'record_kind': 'validation_grid_candidate',
            'test_partition_loaded': False,
            'selected_baseline': {'config': c},
        },
    )
    result = trainer.fit_validation(
        SimpleNamespace(
            selection_root=selection_root,
            primary_freeze=candidate,
            output_root=attempt,
            arm=arm,
            seed=seed,
            device='cpu',
        )
    )
    assert result['epochs_completed'] == 2 and result['test_partition_loaded'] is False
    s4.write(tmp_path / 'S4_INPUTS.json', {'synthetic': True})
    trialpath = attempt / 'validation/baseline/seed_42/trial.json'
    s4.write(
        attempt / 'accepted.json',
        {
            'input_contract_sha256': s4.sha(tmp_path / 'S4_INPUTS.json'),
            'trial_sha256': s4.sha(trialpath),
        },
    )
    assert s4.accepted(tmp_path, pool, c, arm, seed) is not None
    trialpath.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        s4.accepted(tmp_path, pool, c, arm, seed)


def test_incomplete_grid_cannot_freeze(tmp_path):
    with pytest.raises(ValueError, match='incomplete'):
        s4.freeze(tmp_path, 'merged')


def test_no_test_evaluation_entrypoint():
    source = (ROOT / 'workflows/transfer/search_hyperparameters.py').read_text()
    assert 'evaluate_test' not in source
    assert 'official_test' not in source
