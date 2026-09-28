"""Read-only numerical and manuscript consistency checks for the S4 addendum."""
import hashlib
import json
import re
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper/digital_discovery'
REPORT=ROOT/'outputs/dd_final_round/N3_S4_reporting/manifest.json'
if not REPORT.exists():
    pytest.skip('Private S4 artifacts not installed',allow_module_level=True)

def test_frozen_sources_and_statistical_replay():
    report=json.loads(REPORT.read_text())
    assert report['ci_and_p_recomputed'] and not report['new_training_performed']
    for path,digest in report['sources'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    replay=ROOT/'outputs/dd_final_round/N3_S4_statistics_recheck/manifest.json'
    assert hashlib.sha256(replay.read_bytes()).hexdigest()==report['replay_manifest_sha256']
    assert json.loads(replay.read_text())['status']=='pass'

def test_generated_values_and_table_labels():
    report=json.loads(REPORT.read_text())
    tex=(PAPER/'tables/n3_s4_final.tex').read_text()
    assert hashlib.sha256(tex.encode()).hexdigest()==report['generated_table_sha256']
    for row in report['own_pool_rows']+report['transfer_rows']:
        assert ' & '.join(row) in tex
    si=(PAPER/'supporting_information.tex').read_text()
    for stem in ('final_round_n2_n4','n3_s4_final'):
        si=si.replace(r'\input{tables/'+stem+'}',(PAPER/'tables'/f'{stem}.tex').read_text())
    labels={}
    for n,tab in enumerate(re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',si,re.S),1):
        for label in re.findall(r'\\label\{([^}]+)\}',tab):labels[label]=n
    assert [labels[x] for x in ('tab:s4_choices','tab:s4_own','tab:s4_transfer','tab:s4_hash')]==[24,25,26,27]

def test_no_unperformed_full_grid_claim():
    main=(PAPER/'main.tex').read_text()
    si=(PAPER/'supporting_information.tex').read_text()
    assert 'N3-S4' in main and 'N3-S4' in si
    assert 'full-grid retuning was not performed' not in main+si
