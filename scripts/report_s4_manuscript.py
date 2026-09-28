"""Generate private S4 manuscript tables only after statistical replay passes."""
import hashlib
import json
from pathlib import Path

from dd_n3_s4 import read, sha, write
from summarize_dd_final_round import table, pp

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/dd_final_round'


def ci(values):
    return '['+pp(values[0])+', '+pp(values[1])+']'


def main():
    replay=read(DATA/'N3_S4_statistics_recheck/manifest.json')
    assert replay['status']=='pass' and replay['ci_and_p_recomputed']
    audit=read(DATA/'N3_S4_final_return_audit/manifest.json')
    assert audit['status']=='pass'
    sources={};choices=[];own=[];transfer=[];hash_rows=[]
    for pool,name in [('localretro_only','LocalRetro'),('merged','Merged')]:
        base=DATA/'N3_S4_final'/pool
        freeze=base/'models/freeze.json';stats_path=base/'statistics.json';test_path=base/'test_results/manifest.json'
        stats=read(stats_path);test=read(test_path);f=read(freeze)
        for p in [freeze,stats_path,test_path,base/'test_results/reaction_ranks.jsonl',base/'manifest.json']:
            sources[p.relative_to(ROOT).as_posix()]=sha(p)
        for arm,label in [('baseline','2D'),('augmented','2D + Uni-Mol')]:
            c=f[arm+'_config']
            choices.append([name,label,f['selected_prior_transform'],str(c['hidden_width']),str(c['dropout']),
                            f"{c['learning_rate']:g}",str(c['margin'])])
        for metric,label in [('top1','Top-1'),('mrr','MRR')]:
            e=stats['own_pool_effects'][metric]
            means=[sum(test['per_seed_metrics'][a][str(s)]['within_pool'][metric] for s in range(42,62))/20 for a in ('baseline','augmented')]
            own.append([name,label,f'{100*means[0]:.3f}',f'{100*means[1]:.3f}',pp(e['product']['effect']),
                ci([e['product']['ci95_low'],e['product']['ci95_high']]),
                ci([e['crossed']['ci95_low'],e['crossed']['ci95_high']]),f"{e['sign_flip_p']:.5f}"])
        for row in stats['common_transfer']:
            transfer.append([name,'Top-1' if row['metric']=='top1' else 'MRR',pp(row['expanded_effect']),pp(row['transfer_loss']),
                ci([row['product_ci95_low'],row['product_ci95_high']]),ci([row['crossed_ci95_low'],row['crossed_ci95_high']]),f"{row['sign_flip_p']:.5f}"])
        for p,label in [(base/'manifest.json','Final manifest'),(stats_path,'Paired statistics')]:
            h=sha(p)
            hash_rows.append([name,label,r'\texttt{'+h[:32]+r'}\newline\texttt{'+h[32:]+'}'])
    parts=[r'% Generated from frozen N3-S4 artifacts after CI/p-value replay.']
    parts.append(table('N3-S4 configurations selected independently for each arm by five-seed official-validation MRR. Both arms received the same 81-configuration budget; prior transforms remain those selected by N3-S2.',
        'tab:s4_choices','lllrrrr',['Pool','Arm','Prior','Width','Dropout','Learning rate','Margin'],choices))
    parts.append(table('N3-S4 own-pool official-test results over 20 paired seeds. Baseline and augmented means are conditional on reference coverage: 4,462 reactions for LocalRetro and 4,584 for merged. Means are percentages; effects and intervals are percentage points. Product-cluster and crossed product--seed intervals use 10,000 draws; two-sided sign-flip tests use 100,000 draws and are unadjusted secondary tests.',
        'tab:s4_own','llrrrrrr',['Pool','Endpoint','2D','+ Uni-Mol',r'$\Delta$','Product 95\% CI','Crossed 95\% CI',r'$p$'],own))
    parts.append(table('N3-S4 common-cohort effects and transfer loss on the same 3,814 reactions, seeds 42--61. Values are percentage points. Historical increments are +0.704 Top-1 and +0.524 MRR points; transfer loss is the adapted increment minus the historical increment. Both LocalRetro crossed intervals exclude zero; neither merged interval does. Tests are pointwise and are not added to the primary Holm family.',
        'tab:s4_transfer','llrrrrr',['Pool','Endpoint','Adapted effect','Transfer loss','Product 95\% CI','Crossed 95\% CI',r'$p$'],transfer))
    parts.append(table('N3-S4 artifact provenance. The final manifests and paired statistics reside in the pool subdirectories of '+r'\path{outputs/dd_final_round/N3_S4_final/}'+'. Full SHA256 strings are split across two lines without changing their order. The replay report is '+r'\path{outputs/dd_final_round/N3_S4_statistics_recheck/manifest.json}'+'.',
        'tab:s4_hash',r'llp{0.48\textwidth}',['Pool','Artifact','SHA256'],hash_rows))
    text='\n'.join(parts).replace(r'\path{',r'\nolinkurl{')
    (ROOT/'paper/digital_discovery/tables/n3_s4_final.tex').write_text(text,encoding='utf-8')
    report={'status':'pass','sources':sources,'replay_manifest_sha256':sha(DATA/'N3_S4_statistics_recheck/manifest.json'),
            'ci_and_p_recomputed':True,'new_training_performed':False,'own_pool_rows':own,'transfer_rows':transfer,
            'generated_table_sha256':hashlib.sha256(text.encode()).hexdigest()}
    out=DATA/'N3_S4_reporting/manifest.json'
    # This is a regenerable typesetting report, not a frozen scientific artifact.
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('Four S4 tables generated from audited artifacts.')


if __name__=='__main__':main()
