"""Audit S4 returned ZIPs and validation selection, without loading official test."""
from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from local_archive_paths import imported_archive

from dd_n3_s4 import ARMS, GRID, POOLS, PROTOCOL, SEEDS, read, select_winner, sha, task_id, write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name
            and ':' not in name, f'Unsafe archive member: {name}')
    return p


def audit_pool(root, pool):
    archive = imported_archive(root, f'N3_S4_{pool}.zip')
    receipt_path = archive.with_suffix('.receipt.json')
    receipt = read(receipt_path)
    contract_path = root / 'release/dd_n3_s4_20260924/S4_INPUTS.json'
    contract_hash = sha(contract_path)
    contract = read(contract_path)
    require(receipt['protocol_id'] == PROTOCOL and receipt['status'] == 'verified', 'Wrong receipt')
    require(sha(archive) == receipt['sha256'], 'Archive checksum mismatch')
    prefix = f'N3_S4/{pool}/'
    with ZipFile(archive) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'Duplicate archive members')
        require(set(names) == {prefix + n for n in receipt['members']}, 'Member inventory differs')
        for name, digest in receipt['members'].items():
            safe_name(prefix + name)
            require(hashlib.sha256(z.read(prefix + name)).hexdigest() == digest, f'Hash differs: {name}')
        def record(name):
            import json
            return json.loads(z.read(prefix + name))
        m = record('manifest.json')
        execution = record('execution_contract.json')
        require(execution['input_contract'] == contract, 'Source/input contract differs')
        require(m['protocol_id'] == PROTOCOL and m['pool'] == pool
                and m['stage'] == 'validation_search_only' and m['status'] == 'complete'
                and m['test_partition_loaded'] is False and m['test_evaluation_complete'] is False
                and m['seeds'] == list(SEEDS) and m['trial_count'] == 810
                and m['input_contract_sha256'] == contract_hash, 'Invalid validation freeze')
        require(m['prior_transform'] == ('log' if pool == 'localretro_only' else 'rank'), 'Prior changed')
        expected_ids = {task_id(c, a, s) for c in GRID for a in ARMS for s in SEEDS}
        require(set(m['trials']) == expected_ids, 'Not the exact complete 810-trial grid')
        means, selected, runtimes = [], {}, []
        for arm in ARMS:
            rows = []
            for config in GRID:
                values = []
                for seed in SEEDS:
                    identity = task_id(config, arm, seed)
                    rec = m['trials'][identity]
                    name = rec['path']
                    require(name.startswith('results/' + prefix + 'trials/' + identity + '/attempt_'), 'Wrong trial path')
                    name = name.removeprefix('results/')
                    raw = z.read(name)
                    require(hashlib.sha256(raw).hexdigest() == rec['sha256'], 'Trial hash differs')
                    import json
                    t = json.loads(raw)
                    require(t['protocol_id'] == PROTOCOL and t['pool_name'] == pool and t['arm'] == arm
                            and t['seed'] == seed and t['config'] == config and t['test_partition_loaded'] is False,
                            'Trial identity/protocol differs')
                    selection_sha = contract['files'][f'inputs/{pool}/manifest.json']['sha256']
                    require(t['selection_manifest']['sha256'] == selection_sha, 'Selection input differs')
                    trial_parent = PurePosixPath(name).parent
                    attempt = trial_parent.parent.parent.parent
                    accepted = json.loads(z.read(str(attempt / 'accepted.json')))
                    require(accepted['trial_sha256'] == rec['sha256']
                            and accepted['input_contract_sha256'] == contract_hash, 'Acceptance receipt differs')
                    for key in ('checkpoint', 'normalizer'):
                        leaf = t[key]['path'].replace('\\', '/').rsplit('/', 1)[-1]
                        data = z.read(str(trial_parent / leaf))
                        require(len(data) == t[key]['size_bytes'] and hashlib.sha256(data).hexdigest() == t[key]['sha256'],
                                f'Model artifact differs: {key}')
                    score = t['best_validation_mrr']
                    require(math.isfinite(score) and 0 <= score <= 1, 'Invalid validation MRR')
                    values.append(score)
                    runtimes.append(t['runtime_seconds'])
                row = dict(arm=arm, config=config, per_seed_validation_mrr=values,
                           mean_validation_mrr=sum(values) / len(values))
                rows.append(row)
                means.append(row)
            selected[arm] = select_winner(rows)
        require(means == m['all_validation_results'], 'Recomputed validation summary differs')
        require(selected == m['selected'], 'Winner/tie rule differs')
        return dict(pool=pool, archive_sha256=receipt['sha256'], receipt_sha256=sha(receipt_path),
                    member_count=len(names), verified_trials=810, selected=selected,
                    validation_augmented_minus_baseline=selected['augmented']['mean_validation_mrr']
                    - selected['baseline']['mean_validation_mrr'], sum_trial_runtime_seconds=sum(runtimes),
                    input_contract_sha256=contract_hash, test_partition_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.monotonic()
    pools = [audit_pool(root, p) for p in POOLS]
    out = root / 'outputs/dd_final_round/N3_S4_return_audit/manifest.json'
    result = dict(status='pass', protocol_id=PROTOCOL, stage='returned_validation_search_audit',
                  comparator='archived validation trials versus frozen S4 input/grid contract',
                  single_intended_change='none; verification only', pools=pools, total_verified_trials=1620,
                  test_partition_loaded=False, training_performed=False,
                  interpretation='Validation selection scores are not official-test effects.',
                  runtime_seconds=time.monotonic() - started)
    if out.exists():
        old = read(out)
        require(old['pools'] == pools and old['status'] == 'pass', 'Existing audit differs; do not overwrite')
    else:
        write(out, result)
    import json
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
