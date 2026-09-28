"""Read-only replay of the five frozen Morgan models; never fit or change results."""
import csv
import json
import pickle
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from rerank.experiments.run_dd_n4_historical import _artifact, _load
from rerank.experiments.run_dd_n2_cross_inference import _score_lists, _checked_diagonal_rank
from rerank.features import FeatureNormalizer
from rerank.model import RankerMLP
from rerank.study_data import canonicalize_reactant_set
from rerank.revision_tuning import transform_feature_matrix, feature_columns_for_arm
from rerank.experiments.dd_remote_gate import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-layout', action='store_true')
    args = parser.parse_args()
    root = Path('outputs/jcheminform_revision/encoder_controls/morgan_atom')
    frozen = _load(root / 'ranking/freeze.json')
    if sha256(root / 'cache/train_test.pkl') != frozen['retained_control_train_test_cache_sha256']:
        raise ValueError('Frozen Morgan test cache hash differs')
    with (root / 'cache/train_test.pkl').open('rb') as handle:
        payload = pickle.load(handle)['payload']
    matrices = [np.asarray(x, dtype=np.float32) for x in payload['eval_features']]
    if args.legacy_layout:
        matrices = [transform_feature_matrix(x, 'raw', feature_columns_for_arm('augmented'))
                    for x in matrices]
    masks = [np.asarray([canonicalize_reactant_set(c['smiles']) == canonicalize_reactant_set(t)
                         for c in candidates])
             for (_, candidates), t in zip(payload['eval_pwc'], payload['eval_ground_truths'])]
    report = []
    for seed in range(42, 47):
        trial = frozen['selected_control']['trials'][str(seed)]
        modelroot = root / 'ranking/artifacts'
        normalizer = FeatureNormalizer.load(str(_artifact(modelroot, trial, 'normalizer')))
        model = RankerMLP(input_dim=7, hidden_dims=[trial['config']['hidden_width']],
                          dropout=trial['config']['dropout'], use_batch_norm=False)
        model.load_state_dict(torch.load(_artifact(modelroot, trial, 'checkpoint'),
                                         map_location='cpu', weights_only=True))
        model.eval()
        scores = _score_lists(model, normalizer, matrices, 'cpu')
        with (root / f'ranking/test_results/predictions/augmented_seed_{seed}.csv').open(
                encoding='utf-8', newline='') as handle:
            old = {int(r['reaction_id']): r for r in csv.DictReader(handle)}
        if len(old) != len(matrices):
            raise ValueError('Frozen prediction coverage differs')
        mismatches = []
        max_error = 0.0
        score_comparisons = 0
        outside_tolerance = 0
        for meta, (_, candidates), truth, matrix, mask, score in zip(
                payload['eval_metadata'], payload['eval_pwc'], payload['eval_ground_truths'],
                matrices, masks, scores, strict=True):
            rid = int(meta['reaction_id'])
            row = old[rid]
            names = [c['smiles'] for c in candidates]
            if names != json.loads(row['baseline_candidates_json']):
                raise ValueError(f'Candidate input/order differs: {seed}/{rid}')
            if canonicalize_reactant_set(truth) != canonicalize_reactant_set(row['ground_truth']):
                raise ValueError(f'Reference chemistry differs: {seed}/{rid}')
            # Compare score multisets for duplicate exact SMILES; no invented
            # mapping between indistinguishable duplicate string occurrences.
            current_groups, old_groups = defaultdict(list), defaultdict(list)
            for name, value in zip(names, score, strict=True):
                current_groups[name].append(float(value))
            refscores = np.asarray(json.loads(row['reranked_scores_json']))
            for name, value in zip(json.loads(row['reranked_candidates_json']), refscores, strict=True):
                old_groups[name].append(float(value))
            if current_groups.keys() != old_groups.keys():
                raise ValueError(f'Frozen reranked candidates differ: {seed}/{rid}')
            for name, values in current_groups.items():
                a, b = np.array(sorted(values)), np.array(sorted(old_groups[name]))
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('Score count or finiteness differs')
                error = float(np.max(np.abs(a - b)))
                max_error = max(max_error, error)
                score_comparisons += len(a)
                # Diagnostic only: these tolerances do not authorize a rank change.
                outside_tolerance += int(np.sum(~np.isclose(a, b, rtol=1e-6, atol=1e-6)))
            rank = int(np.flatnonzero(mask[np.argsort(-score, kind='stable')])[0]) + 1
            target = int(old[rid]['reranked_rank'])
            if rank == target:
                continue
            item = {'reaction_id': rid, 'computed': rank, 'frozen': target,
                    'positive_count': int(mask.sum())}
            try:
                _, audit = _checked_diagonal_rank(rank, target, matrix, mask, score, refscores)
                item['identical_input_tie'] = audit
            except ValueError as exc:
                item['error'] = str(exc)
                item['matrix'] = matrix.tolist()
                item['scores'] = score.tolist()
                item['frozen_scores'] = refscores.tolist()
                item['mask'] = mask.tolist()
            mismatches.append(item)
        report.append({'seed': seed, 'reactions': len(matrices), 'mismatches': mismatches,
                       'score_comparisons': score_comparisons, 'max_absolute_score_error': max_error,
                       'scores_outside_diagnostic_tolerance': outside_tolerance,
                       'diagnostic_tolerance': {'rtol': 1e-6, 'atol': 1e-6},
                       'checkpoint_sha256': sha256(_artifact(modelroot, trial, 'checkpoint')),
                       'normalizer_sha256': sha256(_artifact(modelroot, trial, 'normalizer')),
                       'predictions_sha256': sha256(root / f'ranking/test_results/predictions/augmented_seed_{seed}.csv'),
                       'result_mutation_performed': False, 'training_performed': False})
        print(json.dumps(report[-1]), flush=True)


if __name__ == '__main__':
    main()
