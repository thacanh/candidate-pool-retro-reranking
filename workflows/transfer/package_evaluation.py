"""Assemble isolated final S4 job; copy frozen data, never train or score test."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import copy
import hashlib
import json
import shutil

from import_search_results import audit_pool
from archive_paths import imported_archive
from search_hyperparameters import POOLS, ARMS, PROTOCOL, read, write, sha, task_id

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'release/dd_s4_finish_20260924'
POOL_ROOT = 'outputs/jcheminform_revision/candidate_pools/ws_e_three_pools_filtered_v2'
INDEX = 'outputs/jcheminform_revision/ws_e_streaming/index'
COMMON = (
    'outputs/digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_transfer_loss_reactions.csv'
)
PRIMARY = 'outputs/jcheminform_revision/tuned_primary/conformer_seed_42/g1_20seed/test_results/predictions'


def copy_path(source, dest):
    if source.is_dir():
        for p in source.rglob('*'):
            if p.is_file() and p.suffix != '.pyc':
                copy_path(p, dest / p.relative_to(source))
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        if sha(source) != sha(dest):
            raise ValueError(f'Copy failed: {source}')


def main():
    audits = {p: audit_pool(ROOT, p) for p in POOLS}
    if TARGET.exists():
        raise FileExistsError(TARGET)
    TARGET.mkdir(parents=True)
    frozen = ROOT / 'release/dd_n3_s4_20260924'
    source_contract = read(frozen / 'S4_INPUTS.json')
    for name, rec in source_contract['files'].items():
        if name.startswith('src/'):
            assert sha(frozen / name) == rec['sha256']
            copy_path(frozen / name, TARGET / name)
    for name in [
        'workflows/transfer/search_hyperparameters.py',
        'workflows/transfer/evaluate_selected_models.py',
        'workflows/transfer/evaluate_selected_models.sh',
        'tests/test_selected_model_evaluation.py',
        'workflows/transfer/requirements-cpu.txt',
        'docs/reproducibility.md',
    ]:
        copy_path(ROOT / name, TARGET / name)
    evaluation = {}
    for pool in POOLS:
        inp = TARGET / 'inputs' / pool
        with ZipFile(imported_archive(ROOT, f'N3_S4_{pool}.zip')) as z:
            prefix = f'N3_S4/{pool}/'
            search = json.loads(z.read(prefix + 'manifest.json'))
            inp.mkdir(parents=True)
            (inp / 'search_freeze.json').write_bytes(z.read(prefix + 'manifest.json'))
            for arm in ARMS:
                for seed in range(42, 47):
                    rec = search['trials'][task_id(search['selected'][arm]['config'], arm, seed)]
                    name = rec['path'].removeprefix('results/')
                    t = json.loads(z.read(name))
                    dst = inp / 'selected' / arm / f'seed_{seed}'
                    dst.mkdir(parents=True)
                    (dst / 'trial.json').write_bytes(z.read(name))
                    for key in ('checkpoint', 'normalizer'):
                        leaf = t[key]['path'].replace('\\', '/').rsplit('/', 1)[-1]
                        (dst / leaf).write_bytes(z.read(name.rsplit('/', 1)[0] + '/' + leaf))
            write(
                inp / 'selected_config.json',
                dict(
                    protocol_id=PROTOCOL,
                    record_kind='validation_grid_candidate',
                    test_partition_loaded=False,
                    selected_baseline=search['selected']['baseline'],
                    selected_augmented=search['selected']['augmented'],
                    frozen_search_sha256=sha(inp / 'search_freeze.json'),
                ),
            )
        source = ROOT / 'outputs/dd_final_round/N3_S3' / pool / 'selection'
        selection = read(source / 'manifest.json')
        for rec in [selection['validation'], *selection['train_pairs'].values()]:
            leaf = rec['path'].replace('\\', '/').rsplit('/', 1)[-1]
            assert (
                sha(source / leaf) == rec['sha256']
                and (source / leaf).stat().st_size == rec['size_bytes']
            )
            copy_path(source / leaf, inp / 'selection' / leaf)
            rec['path'] = leaf
        # First-five arrays must exactly match the already completed search.
        old = read(frozen / 'inputs' / pool / 'manifest.json')
        assert selection['validation']['sha256'] == old['validation']['sha256']
        for s in range(42, 47):
            assert (
                selection['train_pairs'][str(s)]['sha256'] == old['train_pairs'][str(s)]['sha256']
            )
        selection.update(
            protocol_id=PROTOCOL,
            seeds=list(range(42, 62)),
            source_selection_sha256=sha(source / 'manifest.json'),
        )
        write(inp / 'selection/manifest.json', selection)
        transform = selection['prior_transform']
        feature = f'outputs/dd_final_round/N3_S2/{pool}/{transform}/features'
        assert (
            sha(ROOT / feature / 'manifest.json') == selection['inputs']['feature_freeze']['sha256']
        )
        copy_path(ROOT / feature, TARGET / feature)
        poolfile = 'merged_canonical_union.jsonl' if pool == 'merged' else f'{pool}.jsonl'
        for name in [
            f'{POOL_ROOT}/{poolfile}',
            f'{INDEX}/{pool}_index.npz',
            f'{INDEX}/{pool}_index.json',
        ]:
            copy_path(ROOT / name, TARGET / name)
        evaluation[pool] = dict(
            feature_freeze=f'{feature}/manifest.json',
            pool_jsonl=f'{POOL_ROOT}/{poolfile}',
            pool_index_npz=f'{INDEX}/{pool}_index.npz',
            pool_index_manifest=f'{INDEX}/{pool}_index.json',
            products_jsonl=f'{POOL_ROOT}/products.jsonl',
            source_csv='data/uspto_smiles.csv',
            metadata_csv='data/uspto_reaction_metadata.csv',
        )
    for name in [
        f'{POOL_ROOT}/products.jsonl',
        COMMON,
        'data/uspto_smiles.csv',
        'data/uspto_reaction_metadata.csv',
    ]:
        copy_path(ROOT / name, TARGET / name)
    for arm in ARMS:
        for seed in range(42, 62):
            name = f'{arm}_seed_{seed}.csv'
            copy_path(ROOT / PRIMARY / name, TARGET / 'historical_predictions' / name)
    contract = dict(
        protocol_id=PROTOCOL,
        stage='frozen-choice-20seed-extension-and-test',
        seeds=list(range(42, 62)),
        new_training_trials=60,
        reused_models=20,
        search_audits=audits,
        evaluation_inputs=evaluation,
        common_cohort=COMMON,
        test_data_packaged=True,
        test_scoring_during_packaging=False,
        authorization={
            'basis': 'repository_owner_instruction',
            'quote': 'ok tiếp đi',
            'date': '2026-09-24',
        },
        settings={
            'bootstrap_draws': 10000,
            'sign_flip_draws': 100000,
            'statistics_rng_seeds': [
                2410,
                2411,
                2420,
                2421,
                2430,
                2431,
                2440,
                2441,
                2540,
                2541,
                2450,
                2451,
            ],
        },
        files={
            p.relative_to(TARGET).as_posix(): dict(sha256=sha(p), size_bytes=p.stat().st_size)
            for p in sorted(TARGET.rglob('*'))
            if p.is_file()
        },
    )
    write(TARGET / 'FINISH_INPUTS.json', contract)
    archive = TARGET.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with ZipFile(archive, 'w', ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(TARGET.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(TARGET).as_posix())
    with ZipFile(archive) as z:
        for name, rec in contract['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == rec['sha256']
        assert z.read('FINISH_INPUTS.json') == (TARGET / 'FINISH_INPUTS.json').read_bytes()
    receipt = dict(
        archive=archive.name,
        sha256=sha(archive),
        size_bytes=archive.stat().st_size,
        status='verified_inputs_only',
        training_performed=False,
        test_scoring_performed=False,
    )
    write(archive.with_suffix('.receipt.json'), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
