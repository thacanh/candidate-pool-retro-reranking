"""Frozen S4 choices: 60 additional fits, then 20-seed official-test evaluation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from search_hyperparameters import PROTOCOL, POOLS, ARMS, read, sha, write
from rerank.experiments.protocol_gate import fingerprint
from rerank.experiments.run_pool_ranking import evaluate_test
from rerank.analysis.transfer_statistics import common_cohort, historical_g1_ranks, k1_ranks, transfer_summary
from rerank.analysis.analyze_predictions import (
    product_cluster_bootstrap, crossed_product_seed_bootstrap, clustered_sign_flip,
)

SEEDS = tuple(range(42, 62))
PACKAGES = {'torch':'2.2.2+cpu','numpy':'1.26.4','pandas':'1.5.3','scipy':'1.12.0','rdkit':'2025.9.6'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def immutable(path, value):
    if path.exists():
        require(read(path) == value, f'Existing output differs: {path}')
    else:
        write(path, value)


def local_record(record, parent):
    leaf = record['path'].replace('\\','/').rsplit('/',1)[-1]
    path = parent / leaf
    require(path.is_file() and path.stat().st_size == record['size_bytes']
            and sha(path) == record['sha256'], f'Artifact differs: {path}')
    return path


def verify(root):
    contract = read(root/'FINISH_INPUTS.json')
    require(contract['protocol_id'] == PROTOCOL and contract['seeds'] == list(SEEDS)
            and contract['new_training_trials'] == 60, 'Wrong final-stage contract')
    for name, rec in contract['files'].items():
        p = root/name
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == rec['size_bytes']
                and sha(p) == rec['sha256'], f'Input differs: {name}')
    installed = {n:importlib.metadata.version(n) for n in PACKAGES}
    require(installed == PACKAGES and platform.python_version() == '3.10.14',
            f'Pinned CPU runtime required: {installed}; Python {platform.python_version()}')
    for pool in POOLS:
        selection = read(root/'inputs'/pool/'selection/manifest.json')
        search = read(root/'inputs'/pool/'search_freeze.json')
        require(selection['seeds'] == list(SEEDS) and selection['protocol_id'] == PROTOCOL
                and selection['test_partition_loaded'] is False and search['trial_count'] == 810
                and search['test_partition_loaded'] is False, 'Invalid frozen selection')
        for rec in [selection['validation'],*selection['train_pairs'].values()]:
            local_record(rec,root/'inputs'/pool/'selection')
        ev=contract['evaluation_inputs'][pool]
        require(all((root/name).is_file() for name in ev.values()),'Missing evaluation file')
        feature_path=root/ev['feature_freeze'];feature=read(feature_path)
        require(sha(feature_path)==selection['inputs']['feature_freeze']['sha256']
                and feature['complete'] and feature['pool_name']==pool
                and feature['prior_transform']==selection['prior_transform'], 'Feature freeze differs')
        require(sha(root/ev['pool_index_manifest'])==feature['pool_index_manifest']['sha256'], 'Index differs')
        index=read(root/ev['pool_index_manifest'])
        for key,arg in [('pool','pool_jsonl'),('products','products_jsonl'),('index','pool_index_npz')]:
            require(sha(root/ev[arg])==index[key]['sha256'],f'Index input differs: {key}')
        for rec in feature['shard_manifests']:
            manifest_path=local_record(rec,feature_path.parent/'shards')
            shard=read(manifest_path)
            local_record(shard['output'],manifest_path.parent)
        for arm in ARMS:
            for seed in range(42,47):
                path=root/'inputs'/pool/'selected'/arm/f'seed_{seed}'/'trial.json'
                check_trial(path, pool, arm, seed, search['selected'][arm]['config'])
    return contract


def check_trial(path, pool, arm, seed, config):
    t = read(path)
    require(t['protocol_id'] == PROTOCOL and t['pool_name'] == pool and t['arm'] == arm
            and t['seed'] == seed and t['config'] == config and t['test_partition_loaded'] is False,
            f'Trial identity differs: {path}')
    require(np.isfinite(t['best_validation_mrr']) and 0 <= t['best_validation_mrr'] <= 1,
            'Invalid validation score')
    for key in ('checkpoint','normalizer'):
        local_record(t[key], path.parent)
    return t


def fit_one(root, pool, arm, seed):
    require(seed in range(47,62), 'Never retrain search seeds 42--46')
    selection = root/'inputs'/pool/'selection'
    config = read(root/'inputs'/pool/'search_freeze.json')['selected'][arm]['config']
    folder = root/'results'/pool/'attempts'/arm/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    receipts = list(folder.glob('attempt_*/accepted.json'))
    require(len(receipts) <= 1, 'Duplicate accepted trials')
    if receipts:
        r=read(receipts[0]); trial=receipts[0].parent/'validation'/arm/f'seed_{seed}'/'trial.json'
        require(r['input_contract_sha256']==sha(root/'FINISH_INPUTS.json') and r['trial_sha256']==sha(trial), 'Accepted trial changed')
        check_trial(trial,pool,arm,seed,config)
        return trial
    number=max([int(p.name.split('_')[1]) for p in folder.glob('attempt_*')] or [0])+1
    attempt=folder/f'attempt_{number:03d}';attempt.mkdir()
    env=os.environ.copy()
    env.update(PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    command=[sys.executable,'-m','rerank.experiments.run_pool_ranking','fit-validation',
             '--selection-root',str(selection),'--primary-freeze',str(root/'inputs'/pool/'selected_config.json'),
             '--output-root',str(attempt),'--arm',arm,'--seed',str(seed),'--device','cpu']
    with (attempt/'fit.log').open('x',encoding='utf-8') as log:
        subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    trial=attempt/'validation'/arm/f'seed_{seed}'/'trial.json'
    t=check_trial(trial,pool,arm,seed,config)
    require(t['selection_manifest']['sha256']==sha(selection/'manifest.json'),'Fit input changed')
    write(attempt/'accepted.json',dict(input_contract_sha256=sha(root/'FINISH_INPUTS.json'),trial_sha256=sha(trial)))
    return trial


def freeze_models(root, pool, trials):
    """Make checksummed path-relocated views; never mutate original search trials."""
    selection_path=root/'inputs'/pool/'selection/manifest.json';selection=read(selection_path)
    search_path=root/'inputs'/pool/'search_freeze.json';search=read(search_path)
    model_root=root/'results'/pool/'models'
    records,relocations={},{}
    for arm in ARMS:
        records[arm]={}
        for seed in SEEDS:
            source=trials[(arm,seed)]
            original=check_trial(source,pool,arm,seed,search['selected'][arm]['config'])
            destination=model_root/'validation'/arm/f'seed_{seed}'
            destination.mkdir(parents=True,exist_ok=True)
            view=copy.deepcopy(original)
            for key in ('checkpoint','normalizer'):
                src=local_record(original[key],source.parent);dst=destination/src.name
                if dst.exists():require(sha(dst)==sha(src),'Existing model differs')
                else:shutil.copy2(src,dst)
                view[key]['path']=str(dst.resolve())
            immutable(destination/'trial.json',view)
            records[arm][str(seed)]=fingerprint(destination/'trial.json')
            relocations[f'{arm}/{seed}']={'source_trial':fingerprint(source),
                'view_trial':records[arm][str(seed)],'changed_fields':['checkpoint.path','normalizer.path'],
                'reused_search_model':seed < 47}
    immutable(model_root/'relocation_audit.json',relocations)
    freeze=dict(schema_version=1,record_kind='s4_final_model_freeze',protocol_id=PROTOCOL,
        pool_name=pool,seeds=list(SEEDS),candidate_cap=10,candidate_ordering=selection['candidate_ordering'],
        baseline_config=search['selected']['baseline']['config'],augmented_config=search['selected']['augmented']['config'],
        selected_prior_transform=selection['prior_transform'],config_selection_performed_for_ws_e=True,
        config_selection_source=fingerprint(search_path),selection_manifest=fingerprint(selection_path),
        trials=records,test_partition_loaded=False,complete=True,reused_search_models=10,new_models=30,
        input_contract_sha256=sha(root/'FINISH_INPUTS.json'))
    immutable(model_root/'freeze.json',freeze)
    return model_root/'freeze.json'


def test_gate(root):
    """Both 40-model freezes and every checkpoint must exist before test access."""
    for pool in POOLS:
        model_root=root/'results'/pool/'models';freeze=read(model_root/'freeze.json')
        search=read(root/'inputs'/pool/'search_freeze.json')
        require(freeze['complete'] and freeze['test_partition_loaded'] is False
                and freeze['seeds']==list(SEEDS) and freeze['protocol_id']==PROTOCOL
                and freeze['input_contract_sha256']==sha(root/'FINISH_INPUTS.json'), 'Incomplete test gate')
        require(freeze['config_selection_source']['sha256']==sha(root/'inputs'/pool/'search_freeze.json'), 'Choice changed')
        for arm in ARMS:
            require(set(freeze['trials'][arm])==set(map(str,SEEDS)), 'Missing frozen seed')
            for seed in SEEDS:
                p=model_root/'validation'/arm/f'seed_{seed}'/'trial.json'
                require(sha(p)==freeze['trials'][arm][str(seed)]['sha256'], 'Frozen trial changed')
                check_trial(p,pool,arm,seed,search['selected'][arm]['config'])


def evaluate_pool(root,pool):
    test_gate(root)
    result=root/'results'/pool/'test_results'
    manifest=result/'manifest.json'
    if manifest.exists():
        m=read(manifest)
        require(m['model_freeze']['sha256']==sha(root/'results'/pool/'models/freeze.json'), 'Test model freeze changed')
        local_record(m['predictions'],result)
        return m
    # An interrupted evaluation is preserved, never silently overwritten.
    require(not result.exists() or not any(result.iterdir()), 'Partial test output exists; inspect before retrying')
    inputs=read(root/'FINISH_INPUTS.json')['evaluation_inputs'][pool]
    kwargs={k:str(root/v) for k,v in inputs.items()}
    return evaluate_test(SimpleNamespace(**kwargs,model_freeze=str(root/'results'/pool/'models/freeze.json'),
        selection_root=str(root/'inputs'/pool/'selection'),primary_freeze=str(root/'inputs'/pool/'selected_config.json'),
        output_root=str(root/'results'/pool/'models'),result_dir=str(result),device='cpu'))


def statistics(root,pool,draws=10000,flips=100000):
    out=root/'results'/pool/'statistics.json'
    if out.exists():
        result=read(out)
        require(result['predictions']['sha256']==sha(root/'results'/pool/'test_results/reaction_ranks.jsonl'), 'Statistics input changed')
        return result
    from rerank.study_data import load_reactions
    reactions=load_reactions(root/'data/uspto_smiles.csv',root/'data/uspto_reaction_metadata.csv')
    clusters_by_id={r.reaction_id:r.product_key for r in reactions if r.source_split=='test'}
    prediction=root/'results'/pool/'test_results/reaction_ranks.jsonl'
    rows=[json.loads(line) for line in prediction.read_text().splitlines()]
    covered=sorted(r['reaction_id'] for r in rows if r['covered'])
    require(len(rows)==5004 and len(set(r['reaction_id'] for r in rows))==5004,'Wrong test cohort')
    b,a=k1_ranks(prediction,covered);clusters=np.asarray([clusters_by_id[i] for i in covered],dtype=object)
    endpoints={}
    for i,metric in enumerate(('top1','mrr')):
        diff=(a==1).astype(float)-(b==1) if metric=='top1' else 1./a-1./b
        endpoints[metric]={'product':product_cluster_bootstrap(diff,clusters,draws,2410+i),
            'crossed':crossed_product_seed_bootstrap(diff,clusters,draws,2420+i),
            'sign_flip_p':clustered_sign_flip(diff,clusters,flips,2430+i),
            'positive_seeds':int(np.sum(diff.mean(axis=1)>0)),
            'negative_seeds':int(np.sum(diff.mean(axis=1)<0)),
            'zero_seeds':int(np.sum(diff.mean(axis=1)==0))}
    common=read(root/'FINISH_INPUTS.json')['common_cohort'];ids,c=common_cohort(root/common)
    historical=historical_g1_ranks(root/'historical_predictions',ids)
    transfer=transfer_summary(historical,{pool:k1_ranks(prediction,ids)},c,draws=draws,seed=2440)
    for i,row in enumerate(transfer):
        old=(historical[1]==1).astype(float)-(historical[0]==1) if row['metric']=='top1' else 1./historical[1]-1./historical[0]
        bb,aa=k1_ranks(prediction,ids)
        new=(aa==1).astype(float)-(bb==1) if row['metric']=='top1' else 1./aa-1./bb
        row['sign_flip_p']=clustered_sign_flip(new-old,c,flips,2450+i)
    result=dict(protocol_id=PROTOCOL,scope='secondary post-hoc sensitivity; not the primary Holm family',
        seeds=list(SEEDS),bootstrap_draws=draws,sign_flip_draws=flips,pool=pool,
        own_pool_effects=endpoints,common_transfer=transfer,predictions=fingerprint(prediction))
    write(out,result);return result


def archive(root,pool,label):
    source=root/'results'/pool;output=root/'archives';output.mkdir(exist_ok=True)
    target=output/f'N3_S4_{label}_{pool}.zip';receipt=target.with_suffix('.receipt.json')
    if receipt.exists():
        rec=read(receipt)
        require(sha(target)==rec['sha256'],'Archive checksum changed')
        if label=='final':
            require(rec['members']=={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()},'Final archived files changed')
        return
    require(not target.exists(),'Archive without receipt: preserve and inspect')
    files=sorted(p for p in source.rglob('*') if p.is_file())
    members={p.relative_to(source).as_posix():sha(p) for p in files}
    tmp=target.with_suffix('.partial.zip')
    with ZipFile(tmp,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,f'N3_S4_final/{pool}/{p.relative_to(source).as_posix()}')
    with ZipFile(tmp) as z:
        for n,h in members.items():require(hashlib.sha256(z.read(f'N3_S4_final/{pool}/{n}')).hexdigest()==h,'ZIP readback failed')
    os.replace(tmp,target)
    write(receipt,dict(status='verified',protocol_id=PROTOCOL,stage=label,archive=target.name,sha256=sha(target),members=members))


def run(root,workers):
    contract=verify(root);require(1<=workers<=16,'Use 1--16 workers')
    lock=root/'FINISH_RUNNING.lock'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),host=platform.node()),f)
    started=time.monotonic()
    try:
        for pool in POOLS:
            immutable(root/'results'/pool/'execution_contract.json',contract)
            trials={(a,s):root/'inputs'/pool/'selected'/a/f'seed_{s}'/'trial.json' for a in ARMS for s in range(42,47)}
            count=0
            write(root/'progress.json',dict(stage='training',pool=pool,completed=0,expected=30,workers=workers))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures={executor.submit(fit_one,root,pool,a,s):(a,s) for a in ARMS for s in range(47,62)}
                for f in as_completed(futures):
                    trials[futures[f]]=f.result();count+=1
                    write(root/'progress.json',dict(stage='training',pool=pool,completed=count,expected=30,workers=workers))
            freeze_models(root,pool,trials)
            archive(root,pool,'models')
        test_gate(root)
        for pool in POOLS:
            write(root/'progress.json',dict(stage='official_test',pool=pool))
            evaluate_pool(root,pool)
            write(root/'progress.json',dict(stage='paired_statistics',pool=pool))
            statistics(root,pool)
            manifest=root/'results'/pool/'manifest.json'
            if not manifest.exists():
                write(manifest,dict(status='complete',protocol_id=PROTOCOL,stage='S4_20seed_test',pool=pool,
                    comparator='N3-S3 frozen prior and features; historical G1 for transfer',
                    single_intended_change='validation-selected 81-grid per-arm configuration',
                    reused_models=10,new_models=30,seeds=list(SEEDS),
                    test_partition_loaded_only_after_both_model_freezes=True,
                    input_contract_sha256=sha(root/'FINISH_INPUTS.json'),
                    models=fingerprint(root/'results'/pool/'models/freeze.json'),
                    test=fingerprint(root/'results'/pool/'test_results/manifest.json'),
                    statistics=fingerprint(root/'results'/pool/'statistics.json'),
                    packages=PACKAGES,runtime_seconds_this_invocation=time.monotonic()-started))
            archive(root,pool,'final')
        write(root/'progress.json',dict(status='complete',new_models=60,reused_models=20,official_test_complete=True))
    except BaseException as e:
        write(root/'progress.json',dict(status='stopped',error=str(e)));raise
    finally:lock.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','run','status']);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    if args.command=='verify':print(json.dumps(dict(status='verified',files=len(verify(root)['files']),new_trials=60)))
    elif args.command=='status':print(json.dumps(read(root/'progress.json'),indent=2) if (root/'progress.json').exists() else 'Not started')
    else:run(root,args.workers)


if __name__=='__main__':main()
