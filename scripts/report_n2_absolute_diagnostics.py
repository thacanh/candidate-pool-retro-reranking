"""Read frozen N2 ranks/clipping; render absolute metrics without model inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'outputs/dd_final_round/N2'
    manifest = json.loads((source/'manifest.json').read_text())
    pools = manifest['scope']['training_pools']
    assert pools == ['historical_cap10', 'aizynthfinder_only']
    inputs = {n: source/n for n in ('cell_ranks.npz','cross_inference_effects.csv','clipping.csv')}
    hashes = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n,p in inputs.items()}
    for key, name in [('cell_ranks','cell_ranks.npz'),('effects','cross_inference_effects.csv'),('clipping','clipping.csv')]:
        assert manifest['outputs'][key]['sha256'].removeprefix('sha256:') == hashes[name]
    with np.load(inputs['cell_ranks.npz'], allow_pickle=False) as saved:
        ranks = saved['ranks']
        assert ranks.shape == (2,2,2,20,3814)
        assert np.array_equal(saved['seeds'], np.arange(42,62))
        assert np.all((ranks >= 1) & (ranks <= 10))
    effects = pd.read_csv(inputs['cross_inference_effects.csv'])
    clipping = pd.read_csv(inputs['clipping.csv'])
    rows = []
    labels = {'historical_cap10':'Historical', 'aizynthfinder_only':'Anchored AiZ'}
    for i, train in enumerate(pools):
        for j, evaluate in enumerate(pools):
            audit = clipping[(clipping.training_pool==train)&(clipping.evaluation_pool==evaluate)]
            assert len(audit)==40
            for metric in ('top1','mrr'):
                values = (ranks[i,j]==1).astype(float) if metric=='top1' else 1./ranks[i,j]
                b,a=values.mean(axis=(1,2))
                row=effects[(effects.training_pool==train)&(effects.evaluation_pool==evaluate)&(effects.metric==metric)].iloc[0]
                assert abs((a-b)-row.effect)<1e-12
                rows.append((labels[train],labels[evaluate], 'Top-1' if metric=='top1' else 'MRR',
                             b*100,a*100,(a-b)*100,audit.prior_total_clipped_fraction.max()*100,
                             audit.any_feature_clipped_fraction.max()*100))
    lines=[r'\begin{table*}[t]',r'\scriptsize',
      r'\caption{Absolute N2 common-cohort metrics and clipping diagnostics. Top-1 and MRR are on a 0--100 scale; the increment is in percentage points. All metrics average over 20 seeds and the same 3,814 reactions. Clipping columns give the maximum candidate-row fraction across both arms and all seeds, not reaction error rates.}',
      r'\label{tab:n2_absolute}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrrr}',r'\hline',
      r'Train / normalizer & Evaluate & Metric & Baseline & Augmented & Increment & Prior clip (\%) & Any-feature clip (\%) \\',r'\hline']
    for train,evaluate,metric,b,a,d,p,c in rows:
        lines.append(f'{train} & {evaluate} & {metric} & {b:.3f} & {a:.3f} & {d:+.3f} & {p:.3f} & {c:.3f} '+r'\\')
    lines += [r'\hline',r'\end{tabular*}',r'\par\smallskip\scriptsize Each cell uses its training-pool, arm- and seed-specific frozen normalizer: training-pair mean and standard deviation with clipping to $[-5,5]$. No target-pool refit was performed. Historical probabilities and anchored rank-normalized priors remain non-commensurate. Raw and standardized prior distributions are reported in Table~\ref{tab:n2_prior_distribution}; zero prior clipping is not evidence of calibration. Interpret off-diagonal cells as diagnostics, not deployable zero-shot gains.',r'\end{table*}']
    (ROOT/'paper/digital_discovery/tables/n2_absolute_diagnostics.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    auditdir=ROOT/'tmp/pdfs/dd_20260924_n2complete'
    auditdir.mkdir(parents=True,exist_ok=True)
    (auditdir/'n2_reporting_audit.json').write_text(json.dumps({'source_sha256':hashes,'all_effects_match_frozen':True,'training_or_inference_performed':False},indent=2)+'\n')
    print('N2: eight absolute metric rows agree with frozen effects; source hashes verified.')


if __name__ == '__main__': main()
