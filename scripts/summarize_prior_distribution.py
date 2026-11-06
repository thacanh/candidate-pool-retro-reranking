"""Read-only N2 prior-distribution supplement; no training or model scoring.

Run with PYTHONPATH=src in the original Python 3.10/Numpy 1.26.4 environment.
Original N2 outputs are never written. Candidate rows are the descriptive
unit; pooled standardized distributions repeat those rows for all 20 seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'dd-n2-prior-distribution-report-v1'
QUANTILES = ('min', 'p01', 'q25', 'median', 'q75', 'p99', 'max')


def describe(values):
    x = np.asarray(values)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite vector.')
    q = np.quantile(x, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1], method='linear')
    return dict(
        count=len(x),
        mean=float(np.mean(x, dtype=np.float64)),
        std=float(np.std(x, dtype=np.float64)),
        **dict(zip(QUANTILES, map(float, q))),
    )


def digest(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(
        path=path.relative_to(ROOT).as_posix(), size_bytes=path.stat().st_size, sha256=digest(path)
    )


def check_record(item):
    path = Path(item['path'])
    if (
        not path.is_file()
        or path.stat().st_size != item['size_bytes']
        or digest(path) != item['sha256'].removeprefix('sha256:')
    ):
        raise ValueError(f'Frozen input changed: {path}')
    return path


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows, raw_rows):
    labels = {'historical_cap10': 'Historical', 'aizynthfinder_only': 'Anchored AiZ'}
    lines = [
        r'\begin{table*}[t]',
        r'\scriptsize',
        r'\caption{N2 prior distributions on the 3,814 common reactions. Raw summaries use each candidate row once. Standardized summaries pool the same rows over the 20 frozen training-seed normalizers, separately by arm; these repetitions are not independent observations. Quantiles use linear interpolation.}',
        r'\label{tab:n2_prior_distribution}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrrrrr}',
        r'\hline',
        r'Train & Evaluate & Arm & Min & P1 & Q1 & Median & Q3 & P99 & Max \\',
        r'\hline',
        r'\multicolumn{10}{l}{Raw stored prior (before normalization)} \\',
    ]
    for row in raw_rows:
        lines.append(
            '-- & '
            + labels[row['evaluation_pool']]
            + ' & -- & '
            + ' & '.join(f'{row[k]:.3f}' for k in QUANTILES)
            + r' \\'
        )
    lines += [
        r'\hline',
        r'\multicolumn{10}{l}{Standardized prior: $(p-\mu_{\mathrm{train}})/\sigma_{\mathrm{train}}$} \\',
    ]
    for row in rows:
        if row['stage'] != 'standardized':
            continue
        lines.append(
            ' & '.join(
                [
                    labels[row['training_pool']],
                    labels[row['evaluation_pool']],
                    '2D' if row['arm'] == 'baseline' else '+Uni-Mol',
                ]
            )
            + ' & '
            + ' & '.join(f'{row[k]:.3f}' for k in QUANTILES)
            + r' \\'
        )
    counts = {r['evaluation_pool']: r['count'] for r in raw_rows}
    lines += [
        r'\hline',
        r'\end{tabular*}',
        r'\par\smallskip\scriptsize '
        + f"Candidate counts: Historical {counts['historical_cap10']:,}; Anchored AiZ {counts['aizynthfinder_only']:,}. "
        + r'Both N2 freezes specify raw priors: no log or rank transform is introduced here. Normalization uses each original training-pool/arm/seed normalizer without refitting. No prior values exceed $[-5,5]$; post-clipping summaries equal the standardized summaries exactly. All 160 cell-level prior and any-feature clipping audits reproduce the original N2 report. Raw scales remain semantically different; these descriptive distributions do not establish calibration or explain the cross-inference effect causally. Per-seed summaries, normalizer parameters and SHA256 provenance are recorded under \path{N2_prior_distribution_v1/}.',
        r'\end{table*}',
    ]
    return '\n'.join(lines) + '\n'


def run(output):
    # Reuse only frozen input readers and audit code, never the N2 run/scoring path.
    from rerank.experiments.run_cross_inference import (
        _check_freezes,
        _common_reactions,
        _load_lists,
        _frozen_trial,
        POOLS,
        ARMS,
        SEEDS,
    )
    from rerank.experiments.cross_inference_features import training_arm_features, clipping_audit
    from rerank.features import FeatureNormalizer

    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f'Preserving existing report: {output}')
    source = ROOT / 'outputs/dd_final_round/N2'
    before = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['protocol_id'] != 'dd-n2-cross-inference-v1':
        raise ValueError('Wrong/incomplete original N2.')
    inputs = {name: check_record(row) for name, row in manifest['inputs'].items()}
    for row in manifest['outputs'].values():
        check_record(row)
    args = argparse.Namespace(**inputs)
    args.historical_model_root = inputs['historical_freeze'].parent
    args.aiz_model_root = inputs['aiz_freeze'].parent
    h, a = _check_freezes(inputs['historical_freeze'], inputs['aiz_freeze'])
    ids, products = _common_reactions(inputs['common_reactions_csv'])
    print('Loading frozen candidate feature lists; no model inference.', flush=True)
    lists = _load_lists(args, ids, products)
    with (source / 'clipping.csv').open(newline='') as f:
        clipping = {
            (r['training_pool'], r['evaluation_pool'], r['arm'], int(r['seed'])): r
            for r in csv.DictReader(f)
        }
    if len(clipping) != 160:
        raise ValueError('Wrong original clipping row count.')
    raw_rows = [
        dict(
            evaluation_pool=pool, **describe(np.concatenate([lists[pool][i][0][:, 0] for i in ids]))
        )
        for pool in POOLS
    ]
    per_seed, pooled, normalizers, checks = [], {}, [], 0
    for train in POOLS:
        for arm in ARMS:
            flats = {
                ev: np.concatenate(
                    [
                        training_arm_features(
                            lists[ev][i][0], training_pool=train, evaluation_pool=ev, arm=arm
                        )
                        for i in ids
                    ]
                )
                for ev in POOLS
            }
            for seed in SEEDS:
                _, _, norm_path = _frozen_trial(args, train, arm, seed, h, a)
                norm = FeatureNormalizer.load(str(norm_path))
                normalizers.append(
                    dict(
                        training_pool=train,
                        arm=arm,
                        seed=seed,
                        prior_mean=float(norm.mean_[0]),
                        prior_std=float(norm.std_[0]),
                        clip_sigma=norm.clip_sigma,
                        **record(norm_path),
                    )
                )
                for ev, flat in flats.items():
                    observed = clipping_audit(flat, norm.mean_, norm.std_, norm.clip_sigma)
                    frozen = clipping[(train, ev, arm, seed)]
                    for name, value in observed.items():
                        if isinstance(value, bool):
                            if value != (frozen[name] == 'True'):
                                raise ValueError('Clipping flag mismatch')
                        elif not np.isclose(value, float(frozen[name]), rtol=0, atol=1e-15):
                            raise ValueError(f'Clipping differs: {train}/{ev}/{arm}/{seed}/{name}')
                    checks += 1
                    standardized = ((flat - norm.mean_) / norm.std_)[:, 0]
                    transformed = norm.transform(flat)[:, 0]
                    if not np.array_equal(standardized, transformed):
                        raise ValueError('Expected frozen N2 zero prior clipping.')
                    for stage, values in [('standardized', standardized), ('clipped', transformed)]:
                        key = (train, ev, arm, stage)
                        pooled.setdefault(key, []).append(values)
                        per_seed.append(
                            dict(
                                training_pool=train,
                                evaluation_pool=ev,
                                arm=arm,
                                seed=seed,
                                stage=stage,
                                **describe(values),
                            )
                        )
            print(f'Verified {train}/{arm}: 20 normalizers, 40 clipping cells.', flush=True)
    pooled_rows = [
        dict(
            training_pool=k[0],
            evaluation_pool=k[1],
            arm=k[2],
            stage=k[3],
            **describe(np.concatenate(v)),
        )
        for k, v in pooled.items()
    ]
    after = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    if before != after:
        raise ValueError('Original N2 files changed during reporting.')
    output.mkdir(parents=True)
    for name, rows in [
        ('raw_prior.csv', raw_rows),
        ('per_seed_prior.csv', per_seed),
        ('pooled_prior.csv', pooled_rows),
        ('normalizers.csv', normalizers),
    ]:
        write_csv(output / name, rows)
    (output / 'n2_prior_distribution.tex').write_text(
        table(pooled_rows, raw_rows), encoding='utf-8'
    )
    report = dict(
        protocol_id=PROTOCOL,
        status='complete',
        comparator='unchanged frozen N2 2x2 cross-inference',
        single_intended_change='descriptive prior-distribution reporting from original caches and normalizers',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        common_reactions=len(ids),
        seeds=list(SEEDS),
        weighting='candidate-row weighted; pooled standardized rows repeat across 20 seeds; no inferential sample-size claim',
        quantile_method='linear',
        clipping_cells_matched=checks,
        original_n2_unchanged=True,
        original_n2_sha256=before,
        training_performed=False,
        model_inference_performed=False,
        normalizer_refit_performed=False,
        test_partition_read_for='post-freeze descriptive reporting only; no selection or tuning',
        source_inputs=manifest['inputs'],
        normalizers=normalizers,
        script=record(Path(__file__)),
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        outputs={p.name: record(p) for p in output.iterdir() if p.is_file()},
        runtime_seconds=time.perf_counter() - started,
    )
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(
        json.dumps(
            dict(
                status='complete',
                clipping_cells_matched=checks,
                runtime_seconds=report['runtime_seconds'],
            )
        ),
        flush=True,
    )


if __name__ == '__main__':
    run(ROOT / 'outputs/dd_final_round/N2_prior_distribution_v1')
