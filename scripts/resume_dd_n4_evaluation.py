"""Approved N4 evaluation-only continuation; never build features or fit models."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

from dd_pack_remote import PRIMARY, MORGAN, POOLS as POOL_ROOT, INDEX, FEATURES, COMMON_COHORT
from run_dd_n4_remote import finish, POOLS, SEEDS, PROTOCOL_ID
from rerank.experiments.dd_remote_gate import sha256, fingerprint, require_approval
from rerank.experiments.run_dd_n4_historical import _artifact, REUSE_AUDIT_SHA256
from rerank.ws_e_streaming import atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_record(record, path):
    if sha256(path) != record['sha256'] or Path(path).stat().st_size != record['size_bytes']:
        raise ValueError(f'Frozen artifact differs: {path}')


def validate_models(root):
    """Verify all 140 checkpoints (5 reused + 135 fits) before evaluation."""
    historical = read(root / 'historical/models/freeze.json')
    if (historical['protocol_id'] != PROTOCOL_ID or historical['seeds'] != list(SEEDS)
            or historical['test_partition_loaded'] is not False):
        raise ValueError('Historical model freeze differs')
    for seed in SEEDS:
        rec = historical['trials'][str(seed)]
        for kind in ('checkpoint', 'normalizer'):
            _artifact(Path(rec['model_root']), rec['trial'], kind)
        if rec['trial_file']:
            check_record(rec['trial_file'], root / f'historical/models/validation/seed_{seed}/trial.json')
    for key, path in [('g1_freeze', f'{PRIMARY}/g1_20seed/freeze.json'),
                      ('existing_control_freeze', f'{MORGAN}/ranking/freeze.json'),
                      ('control_selection_bundle', f'{MORGAN}/ranking/selection.pkl')]:
        check_record(historical[key], path)
    for pool in POOLS:
        item = root / pool
        frozen = read(item / 'models/freeze.json')
        if (frozen['protocol_id'] != PROTOCOL_ID or frozen['seeds'] != list(SEEDS)
                or frozen['test_partition_loaded'] is not False or not frozen['complete']):
            raise ValueError('Expanded model freeze differs')
        check_record(frozen['primary_freeze'], f'{PRIMARY}/freeze/model_selection.json')
        check_record(frozen['selection_manifest'], item / 'selection/manifest.json')
        selection = read(item / 'selection/manifest.json')
        check_record(selection['inputs']['feature_freeze'], item / 'features/manifest.json')
        for arm in ('baseline', 'augmented'):
            for seed in SEEDS:
                path = item / f'models/validation/{arm}/seed_{seed}/trial.json'
                check_record(frozen['trials'][arm][str(seed)], path)
                trial = read(path)
                if trial['seed'] != seed or trial['arm'] != arm or trial['pool_name'] != pool:
                    raise ValueError('Trial identity differs')
                for kind in ('checkpoint', 'normalizer'):
                    check_record(trial[kind], trial[kind]['path'])
    if len(list(root.glob('*/models/validation/**/trial.json'))) != 135:
        raise ValueError('Expected 135 new validation trial files')


def main():
    started = time.perf_counter()
    require_approval('docs/dd_remote_approval.json', 'docs/analysis_plan.md', __file__, PROTOCOL_ID)
    root = Path('outputs/dd_final_round/N4').resolve()
    if (root / 'manifest.json').exists():
        raise FileExistsError('N4 already has a final manifest')
    for pool in ('historical', *POOLS):
        out = root / pool / 'test_results'
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Existing evaluation output requires review: {out}')
    print('Checking all 140 frozen models; no training will run.', flush=True)
    validate_models(root)
    expected_path = Path('docs/n4_reproduction_audit_approved.jsonl')
    if sha256(expected_path) != REUSE_AUDIT_SHA256:
        raise ValueError('Wrong approved replay audit')
    replay = subprocess.run([sys.executable, 'scripts/diagnose_dd_n4_reproduction.py'],
                            capture_output=True, text=True, check=True)
    actual = [json.loads(line) for line in replay.stdout.splitlines() if line.strip()]
    expected = [json.loads(line) for line in expected_path.read_text().splitlines() if line.strip()]
    if actual != expected:
        raise ValueError('Full five-model score audit changed; resume refused')
    print('157120 score comparisons match approved audit; frozen first-five predictions retained.', flush=True)
    audit_dir = root / 'replay_audit'
    audit_dir.mkdir(exist_ok=False)
    shutil.copy2(expected_path, audit_dir / expected_path.name)
    shutil.copy2('docs/dd_remote_approval.json', audit_dir / 'resume_approval.json')
    backup = Path('.patch_backups/n4_frozen_reuse_20260924')
    for name in ('patch_audit.json', 'docs/dd_remote_approval.json'):
        if (backup / name).exists():
            shutil.copy2(backup / name, audit_dir / ('original_approval.json' if name.startswith('docs/') else name))
    atomic_json(audit_dir / 'continuation.json', dict(protocol_id=PROTOCOL_ID,
        policy='Owner-approved reuse of frozen first-five predictions, not exact rank reproduction',
        full_score_replay_matches_reviewed_audit=True, preserved_fitted_models=135,
        reused_historical_models=5, training_performed=False,
        model_freezes={pool: fingerprint(root / pool / 'models/freeze.json') for pool in ('historical', *POOLS)}))
    args = NS(approval_record='docs/dd_remote_approval.json', analysis_plan='docs/analysis_plan.md',
        primary_freeze=f'{PRIMARY}/freeze/model_selection.json',
        morgan_control_freeze=f'{MORGAN}/ranking/freeze.json',
        morgan_test_cache=f'{MORGAN}/cache/train_test.pkl',
        historical_prediction_root=f'{PRIMARY}/g1_20seed/test_results/predictions',
        morgan_old_predictions=f'{MORGAN}/ranking/test_results/predictions',
        reproduction_audit=str(audit_dir / expected_path.name),
        pool_root=POOL_ROOT, index_root=INDEX, products_jsonl=f'{POOL_ROOT}/products.jsonl',
        source_csv='data/uspto_smiles.csv', metadata_csv='data/uspto_reaction_metadata.csv',
        common_reactions_csv=COMMON_COHORT, device='cpu', bootstrap_samples=10000, rng_seed=2230)
    result = finish(args, root, started)
    print(json.dumps({'status': result['status'], 'training_performed': False}), flush=True)


if __name__ == '__main__':
    main()
