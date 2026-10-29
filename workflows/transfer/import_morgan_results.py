"""Verify N4 archive, numerical point estimates and provenance; import without rewriting bytes."""
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from archive_results import verify_stage
from archive_paths import imported_archive
from package_inputs import _safe_member, PRIMARY, MORGAN, COMMON_COHORT
from rerank.analysis.transfer_statistics import common_cohort
from rerank.experiments.protocol_gate import sha256
from rerank.pool_streaming import atomic_json


def main():
    root = Path.cwd()
    target = root / 'outputs/dd_final_round/N4'
    report_path = root / 'outputs/dd_final_round/N4_return_audit_20260924/manifest.json'
    if target.exists() or report_path.exists():
        raise FileExistsError('Refusing to replace an existing N4 import or audit')
    receipt_path = imported_archive(root, 'N4.receipt.json')
    verified = verify_stage(SimpleNamespace(receipt=str(receipt_path)))
    assert verified['sha256'] == '6151e9faf29ef63420bd72a8ad7cad99b1bb7714887286802b9a333dfc519208'
    receipt = json.loads(receipt_path.read_text())
    z = zipfile.ZipFile(imported_archive(root, 'N4.zip'))
    for name in z.namelist():
        _safe_member(name)
    def read(name):
        return json.loads(z.read('N4/results/' + name))
    def bind(record):
        path = record['path'].replace('\\', '/')
        marker = '/outputs/dd_final_round/N4/'
        if marker in path:
            member = 'N4/results/' + path.split(marker, 1)[1]
            info = receipt['members'][member]
            assert info['sha256'] == record['sha256'] and info['size_bytes'] == record['size_bytes'], member
        elif '/outputs/' in path:
            local = root / ('outputs/' + path.split('/outputs/', 1)[1])
            assert sha256(local) == record['sha256'], path
            assert local.stat().st_size == record['size_bytes'], path
        else:
            raise ValueError(f'Unresolved provenance record: {path}')
    seeds = list(range(42, 62))
    pools = ('aizynthfinder_only', 'localretro_only', 'merged')
    manifest = read('manifest.json')
    assert manifest['status'] == 'complete' and manifest['protocol_id'] == 'dd-n4-morgan-atom-transfer-v1'
    assert manifest['seeds'] == seeds and manifest['test_partition_loaded_only_after_all_four_model_freezes']
    assert manifest['approval']['sha256'] == receipt['approval_record_sha256']
    for key in ('historical_model_freeze', 'historical_test', 'transfer_effects'):
        bind(manifest[key])
    for key in ('feature_freezes', 'model_freezes', 'expanded_tests'):
        for rec in manifest[key].values():
            bind(rec)
    historical_freeze = read('historical/models/freeze.json')
    assert historical_freeze['seeds'] == seeds and historical_freeze['test_partition_loaded'] is False
    for key in ('g1_freeze', 'control_selection_bundle', 'existing_control_freeze'):
        bind(historical_freeze[key])
    for seed in seeds:
        record = historical_freeze['trials'][str(seed)]
        trial = record['trial']
        assert trial['seed'] == seed and trial['test_partition_loaded'] is False
        if record['trial_file']:
            bind(record['trial_file'])
        for kind in ('checkpoint', 'normalizer'):
            if seed < 47:
                path = Path(MORGAN) / 'ranking/artifacts' / trial[f'{kind}_relpath']
                actual = sha256(path)
            else:
                actual = receipt['members']['N4/results/historical/models/' + trial[f'{kind}_relpath']]['sha256']
            assert trial[f'{kind}_sha256'] == 'sha256:' + actual
    htest = read('historical/test_results/manifest.json')
    assert htest['status'] == 'complete' and htest['first_five_exact_reproduction'] is False
    assert htest['replay_rank_disagreements'] == 7
    for key in ('model_freeze', 'control_cache', 'reaction_ranks', 'replay_audit'):
        bind(htest[key])
    with np.load(io.BytesIO(z.read('N4/results/historical/test_results/reaction_ranks.npz'))) as a:
        arrays = {key: a[key].copy() for key in a.files}
    assert arrays['seeds'].tolist() == seeds
    hids = arrays['reaction_ids'].tolist()
    assert len(hids) == len(set(hids)) == 3985
    assert arrays['augmented_ranks'].shape == arrays['baseline_ranks'].shape == (20, 3985)
    assert np.count_nonzero(arrays['recomputed_augmented_ranks'] != arrays['augmented_ranks']) == 7
    assert np.array_equal(arrays['recomputed_augmented_ranks'][5:], arrays['augmented_ranks'][5:])
    for i, seed in enumerate(seeds):
        paths = [('baseline', Path(PRIMARY) / f'g1_20seed/test_results/predictions/baseline_seed_{seed}.csv')]
        if seed < 47:
            paths.append(('augmented', Path(MORGAN) / f'ranking/test_results/predictions/augmented_seed_{seed}.csv'))
        for arm, path in paths:
            with path.open(newline='', encoding='utf-8') as f:
                old = {int(r['reaction_id']): int(r['reranked_rank']) for r in csv.DictReader(f)}
            assert set(old) == set(hids)
            assert np.array_equal(arrays[f'{arm}_ranks'][i], [old[rid] for rid in hids])
    ids, _ = common_cohort(COMMON_COHORT)
    positions = [hids.index(rid) for rid in ids]
    hist = (arrays['baseline_ranks'][:, positions], arrays['augmented_ranks'][:, positions])
    table = list(csv.DictReader(io.StringIO(z.read('N4/results/transfer_effects.csv').decode())))
    assert len(table) == 6 and manifest['common_reactions'] == len(ids) == 3814
    primary = json.loads(Path(PRIMARY, 'freeze/model_selection.json').read_text())
    counts = {}
    for pool in pools:
        freeze = read(f'{pool}/models/freeze.json')
        assert freeze['complete'] and freeze['seeds'] == seeds and freeze['test_partition_loaded'] is False
        bind(freeze['selection_manifest'])
        bind(freeze['primary_freeze'])
        feature = read(f'{pool}/features/manifest.json')
        assert feature['complete'] and feature['shard_count'] == 64
        for rec in feature['shard_manifests']:
            bind(rec)
            shard = read(f"{pool}/features/shards/" + Path(rec['path']).name)
            bind(shard['output'])
        for arm in ('baseline', 'augmented'):
            assert set(freeze['trials'][arm]) == set(map(str, seeds))
            for seed in seeds:
                bind(freeze['trials'][arm][str(seed)])
                trial = read(f'{pool}/models/validation/{arm}/seed_{seed}/trial.json')
                assert trial['seed'] == seed and trial['arm'] == arm and trial['pool_name'] == pool
                assert trial['test_partition_loaded'] is False
                assert trial['config'] == primary[f'selected_{arm}']['config']
                for key in ('checkpoint', 'normalizer', 'selection_manifest'):
                    bind(trial[key])
        test = read(f'{pool}/test_results/manifest.json')
        assert test['test_partition_loaded_only_after_model_freeze']
        for key in ('model_freeze', 'feature_freeze', 'predictions'):
            bind(test[key])
        rows = [json.loads(line) for line in z.read(f'N4/results/{pool}/test_results/reaction_ranks.jsonl').splitlines()]
        by_id = {r['reaction_id']: r for r in rows}
        assert len(rows) == len(by_id) == test['test_reactions_total'] == 5004
        covered = [r for r in rows if r['covered']]
        assert len(covered) == test['test_reactions_covered']
        assert all(by_id[rid]['covered'] for rid in ids)
        expanded = []
        for arm in ('baseline', 'augmented'):
            ranks = np.array([[by_id[rid][f'{arm}_rank_seed_{seed}'] for rid in ids] for seed in seeds])
            assert np.all((ranks >= 1) & (ranks <= 10))
            expanded.append(ranks)
            for seed in seeds:
                rr = np.array([r[f'{arm}_rank_seed_{seed}'] for r in covered])
                values = {'top1': np.mean(rr == 1), 'mrr': np.mean(1.0 / rr)}
                for metric, val in values.items():
                    stored = test['per_seed_metrics'][arm][str(seed)]
                    assert abs(val - stored['within_pool'][metric]) < 1e-12
                    assert abs(val * len(covered) / len(rows) - stored['end_to_end'][metric]) < 1e-12
        for metric in ('top1', 'mrr'):
            score = (lambda r: (r == 1).astype(float)) if metric == 'top1' else (lambda r: 1.0/r)
            old = float((score(hist[1]) - score(hist[0])).mean())
            new = float((score(expanded[1]) - score(expanded[0])).mean())
            stored = next(r for r in table if r['pool'] == pool and r['metric'] == metric)
            for key, val in [('historical_effect', old), ('expanded_effect', new), ('transfer_loss', new-old)]:
                assert abs(float(stored[key]) - val) < 1e-12
        counts[pool] = {'feature_shards': 64, 'trials': 40, 'covered_test_reactions': len(covered)}
    assert len([n for n in z.namelist() if n.endswith('/trial.json')]) == 135
    imported = 0
    for name in z.namelist():
        prefix = 'N4/results/'
        if not name.startswith(prefix):
            continue
        dest = target / name[len(prefix):]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with z.open(name) as source, dest.open('xb') as output:
            shutil.copyfileobj(source, output)
        assert sha256(dest) == receipt['members'][name]['sha256']
        imported += 1
    report = dict(status='verified_and_imported', protocol_id=manifest['protocol_id'],
        archive=verified, files_imported=imported, pools=counts, new_trials=135, total_models_including_reused=140,
        common_reactions=3814, seeds=seeds, first_five_frozen_predictions_exactly_preserved=True,
        replay_disagreements_audited=7, effects_point_estimates_recalculated=True,
        within_pool_and_end_to_end_metrics_recalculated=True, ci_recomputed=False,
        scientific_artifact_bytes_unchanged=True, manuscript_modified=False,
        instance_needed_to_recover_n4_results=False,
        crossed_transfer_intervals_excluding_zero=sum(float(r['crossed_ci95_high']) < 0 or float(r['crossed_ci95_low']) > 0 for r in table),
        effects=table,
        caveats=['Only merged MRR crossed 95% transfer-loss interval excludes zero; do not claim all-pool confirmed loss.',
                 'Continuation runtime excludes prior feature construction/training.',
                 'Archive N4_run.log preserves the original failed run; final manifests and replay audit document successful continuation.'])
    atomic_json(report_path, report)
    print(json.dumps(dict(status=report['status'], report=str(report_path), imported_files=imported,
                         pools=counts, instance_needed=False, crossed_intervals_excluding_zero=report['crossed_transfer_intervals_excluding_zero']), indent=2))


if __name__ == '__main__':
    main()
