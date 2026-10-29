"""Build a validation-only S4 bundle from audited N3 selections, without fitting."""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from search_hyperparameters import PROTOCOL, GRID, SEEDS, POOLS, read, sha, verify, write

ROOT = Path(__file__).resolve().parents[2]


def main():
    target=ROOT/'release/dd_n3_s4_20260924'
    if target.exists():
        raise FileExistsError(f"Preserve existing bundle, use a new dated destination: {target}")
    target.mkdir(parents=True)
    choicepath=ROOT/'outputs/dd_final_round/N3_S2/choice_freeze.json'
    choice=read(choicepath)
    assert choice['test_partition_loaded'] is False
    provenance={}
    for pool,transform in zip(POOLS,('log','rank')):
        assert choice['choices'][pool]['selected']==transform
        source=ROOT/'outputs/dd_final_round/N3_S3'/pool/'selection'
        source_manifest=source/'manifest.json'
        d=read(source_manifest)
        assert d['prior_transform']==transform and d['candidate_cap']==10
        assert d['test_partition_loaded'] is False and d['test_ground_truth_loaded'] is False
        destination=target/'inputs'/pool
        destination.mkdir(parents=True)
        d['train_pairs']={str(s):d['train_pairs'][str(s)] for s in SEEDS}
        for rec in [d['validation'],*d['train_pairs'].values()]:
            name=Path(rec['path']).name
            file=source/name
            assert file.stat().st_size==rec['size_bytes'] and sha(file)==rec['sha256']
            with np.load(file,allow_pickle=False) as a:
                keys=('features',) if name=='official_valid.npz' else ('positive','negative')
                for k in keys:
                    assert a[k].ndim==2 and np.isfinite(a[k]).all()
                if name!='official_valid.npz':assert a['positive'].shape==a['negative'].shape
            shutil.copy2(file,destination/name)
            rec['path']=name
        provenance[pool]={'original_manifest_sha256':sha(source_manifest),
                          'original_protocol_id':d['protocol_id'],'prior_transform':transform}
        d.update(protocol_id=PROTOCOL,seeds=list(SEEDS),
                 source_selection_manifest=provenance[pool],
                 comparator='N3-S3 fixed ranker on identical frozen transformed input arrays')
        write(destination/'manifest.json',d)
    paths=list((ROOT/'src/rerank').rglob('*.py'))
    paths += [ROOT/p for p in ['workflows/transfer/search_hyperparameters.py','workflows/transfer/search_hyperparameters.sh',
                'docs/reproducibility.md','workflows/transfer/requirements-cpu.txt','docs/analysis_plan.md',
                'tests/test_hyperparameter_search.py']]
    for source in paths:
        dest=target/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    contract={
        'protocol_id':PROTOCOL,'stage':'validation_search_only','expected_trials':1620,
        'grid':list(GRID),'seeds':list(SEEDS),'pools':list(POOLS),'test_data_packaged':False,
        'base_analysis_plan_sha256':sha(ROOT/'docs/analysis_plan.md'),
        'repository_head_at_packaging':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'authorization':{'date':'2026-09-24','basis':'repository_owner_instruction',
                         'quote':'làm có khó k, nếu k quá khó thì triển khai luôn cũng được',
                         'supervisor_signature_present':False,
                         'scope':'S4 staged implementation; validation search first, no automatic official-test evaluation'},
        'source_selections':provenance,'n3_s2_choice_sha256':sha(choicepath),
        'files':{p.relative_to(target).as_posix():{'size_bytes':p.stat().st_size,'sha256':sha(p)}
                 for p in sorted(target.rglob('*')) if p.is_file()}}
    write(target/'S4_INPUTS.json',contract)
    verify(target)
    archive=target.with_suffix('.zip')
    if archive.exists():raise FileExistsError(archive)
    with ZipFile(archive,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(target.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(target).as_posix())
    with ZipFile(archive) as z:
        assert z.testzip() is None
        import hashlib
        for name,rec in contract['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==rec['sha256']
    receipt={'status':'verified_inputs_only','archive':archive.name,'sha256':sha(archive),
             'size_bytes':archive.stat().st_size,'files':len(contract['files']),
             'training_performed':False,'test_data_packaged':False}
    write(target.with_suffix('.receipt.json'),receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
