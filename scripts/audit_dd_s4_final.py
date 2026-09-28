"""Verify/import returned S4 final archives; no training or model inference."""
import copy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from local_archive_paths import imported_archive

import numpy as np

from audit_dd_n3_s4_returned import safe_name, require
from dd_n3_s4 import POOLS, ARMS, PROTOCOL, read, sha, write
from rerank.experiments.run_ws_e_ranking import _rank_metrics
from rerank.analysis.dd_remote_effects import common_cohort, historical_g1_ranks


def close(a,b):
    require(np.isclose(a,b,rtol=0,atol=1e-12),f'Numerical mismatch: {a} vs {b}')


def checked_archive(root,pool,stage):
    path=imported_archive(root, f'N3_S4_{stage}_{pool}.zip');receipt=read(path.with_suffix('.receipt.json'))
    require(receipt['status']=='verified' and receipt['protocol_id']==PROTOCOL
            and receipt['stage']==stage and sha(path)==receipt['sha256'],'Archive receipt mismatch')
    prefix=f'N3_S4_final/{pool}/'
    with ZipFile(path) as z:
        names=z.namelist()
        require(len(names)==len(set(names)) and set(names)=={prefix+n for n in receipt['members']},'ZIP inventory mismatch')
        data={}
        for n,h in receipt['members'].items():
            safe_name(n);raw=z.read(prefix+n)
            require(hashlib.sha256(raw).hexdigest()==h,f'Member hash differs: {n}')
            data[n]=raw
    return receipt,data


def main():
    root=Path.cwd();bundle=root/'release/dd_s4_finish_20260924'
    contract=read(bundle/'FINISH_INPUTS.json');contract_sha=sha(bundle/'FINISH_INPUTS.json')
    ids,clusters=common_cohort(root/contract['common_cohort'])
    historical=historical_g1_ranks(bundle/'historical_predictions',ids)
    results={};imports={}
    for pool in POOLS:
        backup,backup_data=checked_archive(root,pool,'models')
        receipt,data=checked_archive(root,pool,'final');imports[pool]=data
        for name,h in backup['members'].items():require(receipt['members'].get(name)==h,'Model backup differs from final')
        def get(name):return json.loads(data[name])
        def bind(rec):
            marker=f'/results/{pool}/'
            path=rec['path'].replace('\\','/')
            require(marker in path,'Unexpected remote result path')
            name=path.split(marker,1)[1];raw=data[name]
            require(len(raw)==rec['size_bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256'],'Referenced artifact differs')
            return name
        manifest=get('manifest.json');freeze=get('models/freeze.json');test=get('test_results/manifest.json');stats=get('statistics.json')
        require(get('execution_contract.json')==contract,'Execution contract differs')
        require(manifest['status']=='complete' and manifest['protocol_id']==PROTOCOL
                and manifest['new_models']==30 and manifest['reused_models']==10
                and manifest['test_partition_loaded_only_after_both_model_freezes'] is True
                and manifest['input_contract_sha256']==contract_sha,'Invalid final manifest')
        for key in ('models','test','statistics'):bind(manifest[key])
        require(freeze['complete'] and freeze['test_partition_loaded'] is False
                and freeze['input_contract_sha256']==contract_sha and freeze['seeds']==list(range(42,62)), 'Invalid model gate')
        search=read(bundle/'inputs'/pool/'search_freeze.json')
        selection=read(bundle/'inputs'/pool/'selection/manifest.json')
        require(freeze['config_selection_source']['sha256']==sha(bundle/'inputs'/pool/'search_freeze.json')
                and freeze['selection_manifest']['sha256']==sha(bundle/'inputs'/pool/'selection/manifest.json'),'Model selection source differs')
        require(freeze['selected_prior_transform']==selection['prior_transform'],'Prior changed')
        audit=get('models/relocation_audit.json');new_count=reused_count=0
        for arm in ARMS:
            require(set(freeze['trials'][arm])==set(map(str,range(42,62))), 'Incomplete model seeds')
            for seed in range(42,62):
                name=bind(freeze['trials'][arm][str(seed)]);trial=get(name)
                require(trial['pool_name']==pool and trial['arm']==arm and trial['seed']==seed
                        and trial['config']==search['selected'][arm]['config']
                        and trial['test_partition_loaded'] is False,'Model config/identity differs')
                for key in ('checkpoint','normalizer'):bind(trial[key])
                relocation=audit[f'{arm}/{seed}'];bind(relocation['view_trial'])
                if seed<47:
                    source_path=bundle/'inputs'/pool/'selected'/arm/f'seed_{seed}'/'trial.json'
                    original=read(source_path)
                    require(sha(source_path)==relocation['source_trial']['sha256'],'Reused original trial differs')
                    for key in ('checkpoint','normalizer'):
                        leaf=original[key]['path'].replace('\\','/').rsplit('/',1)[-1]
                        require(data[bind(trial[key])]==(source_path.parent/leaf).read_bytes(),'Reused model bytes changed')
                    reused_count+=1
                else:
                    original_name=bind(relocation['source_trial']);original=get(original_name)
                    attempt=str(Path(original_name).parent.parent.parent.parent).replace('\\','/')
                    accepted=get(attempt+'/accepted.json')
                    require(accepted['input_contract_sha256']==contract_sha and accepted['trial_sha256']==relocation['source_trial']['sha256'],'Fit receipt differs')
                    require(original['selection_manifest']['sha256']==sha(bundle/'inputs'/pool/'selection/manifest.json'),'Training input differs')
                    for key in ('checkpoint','normalizer'):bind(original[key])
                    new_count+=1
                restored=copy.deepcopy(trial)
                for key in ('checkpoint','normalizer'):restored[key]['path']=original[key]['path']
                require(restored==original,'More than paths changed in trial view')
        bind(test['model_freeze']);bind(test['predictions']);bind(stats['predictions'])
        require(test['test_partition_loaded_only_after_model_freeze'] is True
                and stats['seeds']==list(range(42,62)) and stats['bootstrap_draws']==10000
                and stats['sign_flip_draws']==100000,'Test/statistical gate differs')
        rows=[json.loads(line) for line in data['test_results/reaction_ranks.jsonl'].splitlines()]
        mapping={r['reaction_id']:r for r in rows}
        require(len(rows)==len(mapping)==test['test_reactions_total']==5004,'Wrong test cohort')
        covered=[r for r in rows if r['covered']]
        require(len(covered)==test['test_reactions_covered'],'Wrong coverage')
        close(test['coverage'],len(covered)/5004)
        original_rows={json.loads(line)['reaction_id']:json.loads(line) for line in
            (root/'outputs/dd_final_round/N3_S3'/pool/'test_results/reaction_ranks.jsonl').read_text().splitlines()}
        for rid,row in mapping.items():
            require(all(row[k]==original_rows[rid][k] for k in ('covered','candidate_count','prior_rank')), 'Frozen candidate cohort changed')
        arrays={}
        for arm in ARMS:
            arrays[arm]=np.array([[r[f'{arm}_rank_seed_{seed}'] for r in covered] for seed in range(42,62)])
            require(np.all(arrays[arm]>=1) and np.all(arrays[arm]<=np.array([r['candidate_count'] for r in covered])), 'Invalid reference ranks')
            for i,seed in enumerate(range(42,62)):
                for scope,denom in [('within_pool',None),('end_to_end',5004)]:
                    actual=_rank_metrics(arrays[arm][i],denom)
                    for key,value in actual.items():close(value,test['per_seed_metrics'][arm][str(seed)][scope][key])
        effects={}
        for metric in ('top1','mrr'):
            score=lambda x:(x==1).astype(float) if metric=='top1' else 1./x
            b,a=arrays['baseline'],arrays['augmented'];diff=score(a)-score(b);s=stats['own_pool_effects'][metric]
            close(diff.mean(),s['product']['effect']);close(diff.mean(),s['crossed']['effect'])
            require(s['positive_seeds']==int(sum(diff.mean(axis=1)>0)) and s['negative_seeds']==int(sum(diff.mean(axis=1)<0)), 'Seed counts differ')
            for kind in ('product','crossed'):require(s[kind]['ci95_low']<=s[kind]['ci95_high'],'Reversed interval')
            effects[metric]={'baseline':float(score(b).mean()),'augmented':float(score(a).mean()),'delta':float(diff.mean()),
                'crossed_ci95':[s['crossed']['ci95_low'],s['crossed']['ci95_high']]}
            common_arrays=[np.array([[mapping[rid][f'{arm}_rank_seed_{seed}'] for rid in ids] for seed in range(42,62)]) for arm in ARMS]
            old=score(historical[1])-score(historical[0]);new=score(common_arrays[1])-score(common_arrays[0])
            transfer=next(r for r in stats['common_transfer'] if r['metric']==metric)
            close(old.mean(),transfer['historical_effect']);close(new.mean(),transfer['expanded_effect']);close((new-old).mean(),transfer['transfer_loss'])
        results[pool]=dict(status='pass',models_archive_sha256=backup['sha256'],final_archive_sha256=receipt['sha256'],
            verified_members=len(data),new_models=new_count,reused_models=reused_count,test_total=5004,
            test_covered=len(covered),effects=effects,common_transfer=stats['common_transfer'],
            runtime_seconds_this_invocation=manifest['runtime_seconds_this_invocation'])
    # All pools pass before importing any result. Existing bytes are never replaced.
    for pool,data in imports.items():
        target=root/'outputs/dd_final_round/N3_S4_final'/pool
        for name,raw in data.items():
            dst=target/name
            if dst.exists():require(dst.read_bytes()==raw,f'Existing local result differs: {dst}')
            else:
                dst.parent.mkdir(parents=True,exist_ok=True)
                with dst.open('xb') as f:f.write(raw)
    report=dict(status='pass',protocol_id=PROTOCOL,scope='archive, provenance, frozen-model, pairing and numerical point-estimate audit',
        statistical_intervals_recomputed=False,training_performed=False,model_inference_performed=False,
        total_models=80,new_models=60,reused_models=20,pools=results)
    out=root/'outputs/dd_final_round/N3_S4_final_return_audit/manifest.json'
    if out.exists():require(read(out)==report,'Existing audit differs')
    else:write(out,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
