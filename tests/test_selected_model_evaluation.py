"""Synthetic final-stage gates: no scientific data or scientific test scoring."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workflows/transfer'))
import evaluate_selected_models as finish
from search_hyperparameters import GRID, write, read, sha
from rerank.experiments import run_pool_ranking as wse
from rerank.model import RankerMLP
from rerank.experiments.run_pool_ranking import FeatureNormalizer
from rerank.experiments.protocol_gate import fingerprint


def fixture_models(root):
    write(root/'FINISH_INPUTS.json',{'synthetic':True})
    config=GRID[0]
    for pool in finish.POOLS:
        inp=root/'inputs'/pool
        write(inp/'search_freeze.json',{'selected':{arm:{'config':config} for arm in finish.ARMS}})
        write(inp/'selected_config.json',{'selected_'+arm:{'config':config} for arm in finish.ARMS})
        selection={'protocol_id':finish.PROTOCOL,'pool_name':pool,'seeds':list(finish.SEEDS),
            'candidate_cap':10,'candidate_ordering':'frozen','prior_transform':'log' if pool=='localretro_only' else 'rank',
            'baseline_columns':[0,1],'augmented_columns':[0,1,2],'source_indicators_in_both_arms':False}
        write(inp/'selection/manifest.json',selection)
        trials={}
        for arm in finish.ARMS:
            for seed in finish.SEEDS:
                dest=inp/'selected'/arm/f'seed_{seed}';dest.mkdir(parents=True)
                n=2 if arm=='baseline' else 3
                model=RankerMLP(input_dim=n,hidden_dims=[config['hidden_width']],dropout=config['dropout'],use_batch_norm=False)
                for p in model.parameters():p.data.zero_()
                torch.save(model.state_dict(),dest/'best_checkpoint.pt')
                FeatureNormalizer().fit(np.array([[0.]*n,[1.]*n],dtype='f')).save(str(dest/'normalizer.npz'))
                t={'protocol_id':finish.PROTOCOL,'pool_name':pool,'arm':arm,'seed':seed,'config':config,
                    'test_partition_loaded':False,'best_validation_mrr':.5}
                for key,name in [('checkpoint','best_checkpoint.pt'),('normalizer','normalizer.npz')]:
                    t[key]=fingerprint(dest/name);t[key]['path']='/old/remote/'+name
                write(dest/'trial.json',t);trials[arm,seed]=dest/'trial.json'
        finish.freeze_models(root,pool,trials)
    return root


def test_reused_seeds_cannot_be_trained(tmp_path):
    with pytest.raises(ValueError,match='Never retrain'):
        finish.fit_one(tmp_path,'merged','baseline',42)


def test_both_freezes_required_before_any_test_loader(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Test evaluator must not be called')
    monkeypatch.setattr(finish,'evaluate_test',forbidden)
    with pytest.raises(FileNotFoundError):finish.evaluate_pool(tmp_path,'merged')


def test_synthetic_freeze_relocation_tamper_and_archive(tmp_path):
    fixture_models(tmp_path);finish.test_gate(tmp_path)
    original=tmp_path/'inputs/merged/selected/baseline/seed_42/trial.json'
    assert read(original)['checkpoint']['path']=='/old/remote/best_checkpoint.pt'
    finish.archive(tmp_path,'merged','models')
    finish.archive(tmp_path,'merged','models')
    p=tmp_path/'results/merged/models/validation/baseline/seed_42/best_checkpoint.pt'
    p.write_bytes(b'bad')
    with pytest.raises(ValueError,match='Artifact differs'):finish.test_gate(tmp_path)


def test_real_evaluator_synthetic_ranking_and_completed_resume(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    fixture_models(tmp_path)
    pool='merged'
    write(tmp_path/'feature.json',{'pool_index_manifest':fingerprint(tmp_path/'FINISH_INPUTS.json')})
    # feature.json fingerprint must bind the synthetic index, before changing contract.
    write(tmp_path/'index.json',{'synthetic':True})
    write(tmp_path/'feature.json',{'pool_index_manifest':fingerprint(tmp_path/'index.json')})
    contract={'evaluation_inputs':{pool:{'feature_freeze':'feature.json','pool_index_manifest':'index.json',
        'pool_index_npz':'unused.npz','pool_jsonl':'unused.jsonl','products_jsonl':'unused_products.jsonl',
        'source_csv':'source.csv','metadata_csv':'metadata.csv'}}}
    # Stable final input contract, then regenerate freezes to bind it.
    write(tmp_path/'FINISH_INPUTS.json',contract)
    for p in finish.POOLS:
        frozen=tmp_path/'results'/p/'models/freeze.json'
        m=read(frozen);m['input_contract_sha256']=sha(tmp_path/'FINISH_INPUTS.json');write(frozen,m)
    (tmp_path/'source.csv').write_text('synthetic');(tmp_path/'metadata.csv').write_text('synthetic')
    reactions=[SimpleNamespace(reaction_id=i,product_key=f'p{i}',ground_truth_key='C',reaction_class=None,source_split='test') for i in range(3)]
    digest=wse.identity_digest('C');wrong=wse.identity_digest('CC')
    lookup={0:(np.array([list(wrong),list(digest)],dtype='uint8'),np.array([[0.,0.,0.],[1.,1.,1.]],dtype='f')),
            1:(np.array([list(digest)],dtype='uint8'),np.array([[0.,0.,0.]],dtype='f'))}
    monkeypatch.setattr(wse,'_load_pool_index_once',lambda *a:None)
    monkeypatch.setattr(wse,'_reaction_maps',lambda *a:(reactions,[],{f'p{i}':i for i in range(3)}))
    monkeypatch.setattr(wse,'_product_feature_lookup',lambda *a:lookup)
    m=finish.evaluate_pool(tmp_path,pool)
    assert m['test_reactions_total']==3 and m['test_reactions_covered']==2
    assert m['per_seed_metrics']['baseline']['42']['within_pool']['mrr']==.75
    assert m['per_seed_metrics']['baseline']['42']['end_to_end']['top1']==pytest.approx(1/3)
    monkeypatch.setattr(finish,'evaluate_test',lambda *a:pytest.fail('Must reuse completed predictions'))
    assert finish.evaluate_pool(tmp_path,pool)==m
