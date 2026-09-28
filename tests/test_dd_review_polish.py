"""Frozen-number and reference checks for the reviewer-driven presentation update."""
from pathlib import Path
import json,re,hashlib
import numpy as np
import pytest
from rerank.figures import plot_digital_discovery_figures as plot

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'release/dd_figure_notebook_20260924_unified'

def test_adaptation_compact_matches_frozen_sources():
    if not BUNDLE.exists():pytest.skip('Private figure distribution absent')
    live=plot.adaptation_frame(ROOT)
    portable=plot.adaptation_frame(BUNDLE)
    assert list(live[['pool','metric','stage']].itertuples(index=False,name=None))==list(portable[['pool','metric','stage']].itertuples(index=False,name=None))
    np.testing.assert_allclose(live[['effect','low','high']],portable[['effect','low','high']],atol=1e-15,rtol=0)
    assert len(live)==12

def test_notebook_has_six_figures_and_hash_verified_data():
    if not BUNDLE.exists():pytest.skip('Private figure distribution absent')
    notebook=json.loads((BUNDLE/'DD_Figures_Custom.ipynb').read_text(encoding='utf-8'))
    code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    assert code.count("preview('figure")==6
    for row in json.loads((BUNDLE/'figure_data_manifest.json').read_text())['inputs']:
        assert hashlib.sha256((BUNDLE/row['path']).read_bytes()).hexdigest()==row['sha256']

def test_manuscript_evidence_hierarchy_and_reference_style():
    paper=ROOT/'paper/digital_discovery'
    if not paper.exists():pytest.skip('Private manuscript absent')
    main=(paper/'main.tex').read_text()
    assert 'current, independently frozen generators' not in main
    assert 'AiZynthFinder-only' not in main
    assert 'figure4_dd_adaptation' in main
    assert 'figure4_dd_descriptor_diagnostics' not in main
    assert 'secondary, post hoc sensitivities' in main
    assert 'transformed-prior' not in main.lower() # uncertainty detail belongs in SI
    for stem in ('main','supporting_information'):
        assert r'\bibliographystyle{rsc_links}' in (paper/(stem+'.tex')).read_text()
    bbl=ROOT/'tmp/pdfs/dd_20260924_unified/main/main.bbl'
    if bbl.exists():
        text=bbl.read_text()
        assert len(re.findall(r'\\bibitem\[',text))==30
        assert 'Computer-Assisted Design' in text
        assert r'\href{https://doi.org/10.1126/science.166.3902.178}' in text
