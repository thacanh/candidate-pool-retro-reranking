"""Recompute S4 paired uncertainty from frozen ranks; never fit or infer models."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from dd_n3_s4 import POOLS, read, sha, write
from rerank.study_data import load_reactions
from rerank.analysis.dd_remote_effects import common_cohort, historical_g1_ranks, k1_ranks, transfer_summary
from rerank.analysis.analyze_revision_predictions import product_cluster_bootstrap, crossed_product_seed_bootstrap, clustered_sign_flip

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/dd_final_round/N3_S4_final'
OUTPUT=ROOT/'outputs/dd_final_round/N3_S4_statistics_recheck'


def compare(actual,expected,path=''):
    if isinstance(expected,dict):
        for k,v in expected.items():compare(actual[k],v,path+'/'+k)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for i,(a,e) in enumerate(zip(actual,expected)):compare(a,e,path+f'/{i}')
    elif isinstance(expected,(float,int)):
        assert np.isclose(actual,expected,atol=1e-12,rtol=0), (path,actual,expected)
    else:assert actual==expected,(path,actual,expected)


def main():
    started=time.monotonic()
    bundle=ROOT/'release/dd_s4_finish_20260924';contract=read(bundle/'FINISH_INPUTS.json')
    ids,common_clusters=common_cohort(ROOT/contract['common_cohort'])
    historical=historical_g1_ranks(bundle/'historical_predictions',ids)
    reactions=load_reactions(ROOT/'data/uspto_smiles.csv',ROOT/'data/uspto_reaction_metadata.csv')
    products={r.reaction_id:r.product_key for r in reactions if r.source_split=='test'}
    results={};sources={}
    sources[str((bundle/'FINISH_INPUTS.json').relative_to(ROOT))]=sha(bundle/'FINISH_INPUTS.json')
    for p in (bundle/'historical_predictions').glob('*.csv'):sources[str(p.relative_to(ROOT))]=sha(p)
    for p in [ROOT/contract['common_cohort'],ROOT/'data/uspto_smiles.csv',ROOT/'data/uspto_reaction_metadata.csv']:
        sources[str(p.relative_to(ROOT))]=sha(p)
    for pool in POOLS:
        path=DATA/pool/'test_results/reaction_ranks.jsonl';expected=read(DATA/pool/'statistics.json')
        for p in [path,DATA/pool/'statistics.json']:sources[str(p.relative_to(ROOT))]=sha(p)
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        covered=sorted(r['reaction_id'] for r in rows if r['covered'])
        b,a=k1_ranks(path,covered);clusters=np.asarray([products[r] for r in covered],dtype=object)
        effects={}
        for i,metric in enumerate(('top1','mrr')):
            print(pool,metric,'bootstrap and sign flip',flush=True)
            diff=(a==1).astype(float)-(b==1) if metric=='top1' else 1./a-1./b
            effects[metric]={'product':product_cluster_bootstrap(diff,clusters,10000,2410+i),
                'crossed':crossed_product_seed_bootstrap(diff,clusters,10000,2420+i),
                'sign_flip_p':clustered_sign_flip(diff,clusters,100000,2430+i),
                'positive_seeds':int(sum(diff.mean(axis=1)>0)), 'negative_seeds':int(sum(diff.mean(axis=1)<0)),
                'zero_seeds':int(sum(diff.mean(axis=1)==0))}
        bb,aa=k1_ranks(path,ids)
        transfer=transfer_summary(historical,{pool:(bb,aa)},common_clusters,draws=10000,seed=2440)
        for i,row in enumerate(transfer):
            metric=row['metric']
            old=(historical[1]==1).astype(float)-(historical[0]==1) if metric=='top1' else 1./historical[1]-1./historical[0]
            new=(aa==1).astype(float)-(bb==1) if metric=='top1' else 1./aa-1./bb
            row['sign_flip_p']=clustered_sign_flip(new-old,common_clusters,100000,2450+i)
        compare(effects,expected['own_pool_effects']);compare(transfer,expected['common_transfer'])
        results[pool]={'own_pool_effects':effects,'common_transfer':transfer,'status':'pass'}
    report={'status':'pass','protocol_id':'dd-n3-s4-validation-grid-v1','scope':'frozen-rank numerical replay',
        'ci_and_p_recomputed':True,'absolute_tolerance':1e-12,'relative_tolerance':0,
        'bootstrap_draws':10000,'sign_flip_draws':100000,'sources':sources,'results':results,
        'training_performed':False,'model_inference_performed':False,'python':sys.version,
        'numpy':np.__version__,'runtime_seconds':time.monotonic()-started}
    out=OUTPUT/'manifest.json'
    if out.exists():
        previous=read(out);compare(results,previous['results']);assert previous['sources']==sources
    else:write(out,report)
    print(json.dumps({'status':'pass','report':str(out),'runtime_seconds':report['runtime_seconds']}),flush=True)


if __name__=='__main__':main()
