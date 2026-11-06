"""Descriptive prior reporting must not call the model or alter N2."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'n2_prior_report', ROOT / 'scripts/summarize_prior_distribution.py'
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_quantiles_and_population_std():
    values = np.array([0, 1, 2, 3], dtype=np.float32)
    row = report.describe(values)
    assert row['count'] == 4 and row['median'] == 1.5
    assert row['q25'] == 0.75 and row['q75'] == 2.25
    assert row['std'] == np.std(values, dtype=np.float64)


@pytest.mark.parametrize('values', [[], [1, np.nan], [[1, 2]]])
def test_bad_inputs_fail_closed(values):
    with pytest.raises(ValueError):
        report.describe(values)


def test_no_model_scoring_or_normalizer_fit_calls():
    import ast

    tree = ast.parse(Path(report.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden = {
        '_score_lists',
        'RankerMLP',
        'fit',
        'fit_transform',
        'load_state_dict',
        'run_search',
    }
    for call in calls:
        name = getattr(call.func, 'id', getattr(call.func, 'attr', ''))
        assert name not in forbidden


def test_completed_report_retains_original_n2():
    import json

    output = ROOT / 'outputs/dd_final_round/N2_prior_distribution_v1'
    if not (output / 'manifest.json').exists():
        pytest.skip('Reporting run not present')
    m = json.loads((output / 'manifest.json').read_text())
    assert m['clipping_cells_matched'] == 160
    assert len(m['normalizers']) == 80
    assert m['training_performed'] is m['model_inference_performed'] is False
    for name, sha in m['original_n2_sha256'].items():
        assert report.digest(ROOT / 'outputs/dd_final_round/N2' / name) == sha
    for rec in m['outputs'].values():
        assert report.digest(ROOT / rec['path']) == rec['sha256']
    assert (ROOT / 'paper/digital_discovery/tables/n2_prior_distribution.tex').read_bytes() == (
        output / 'n2_prior_distribution.tex'
    ).read_bytes()
    si = (ROOT / 'paper/digital_discovery/supporting_information.tex').read_text()
    assert r'\input{tables/n2_prior_distribution}' in si
