"""Cosmetic redesign checks; scientific inputs and inference are read-only."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pytest
from rerank.figures import plot_digital_discovery_figures as plot

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'release/dd_figure_notebook_20260924_unified'
pytestmark=pytest.mark.skipif(not BUNDLE.exists(),reason='Private notebook bundle absent')

@pytest.fixture
def capture(monkeypatch):
    figs=[]
    plot.configure_style()
    monkeypatch.setattr(plot,'save',lambda fig,*args:figs.append(fig))
    yield figs
    plot.plt.close('all')

def test_figure2_tick_labels_do_not_overlap(capture,tmp_path):
    plot.figure_coverage_effects(BUNDLE,tmp_path)
    fig=capture[0];fig.canvas.draw()
    bounds=[x.get_window_extent(fig.canvas.get_renderer()) for x in fig.axes[0].get_xticklabels()]
    assert all(not a.overlaps(b) for a,b in zip(bounds,bounds[1:]))
    assert fig.axes[0].get_ylim()==(0,100)

def test_no_hatching_and_grid_below_data(capture,tmp_path):
    plot.figure_transfer_shift(BUNDLE,tmp_path)
    fig=capture[0]
    assert all(not p.get_hatch() for ax in fig.axes for p in ax.patches)
    assert all(ax.get_axisbelow() is True for ax in fig.axes)
    # The two series retain their frozen values, with colour rather than patterns.
    frame=plot.pd.read_csv(BUNDLE/'outputs/transfer_analysis/m2_pool_shift/m2_pool_summary.csv')
    frame=frame.set_index('pool').loc[['aizynthfinder_only','localretro_only','merged']]
    actual=[p.get_height() for p in fig.axes[1].patches]
    expected=list(frame.jaccard_vs_historical_mean)+list(frame.kendall_order_vs_historical_mean)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)

def test_heatmap_scale_and_inputs_are_unchanged(capture,tmp_path):
    plot.figure_descriptor_diagnostics(BUNDLE,tmp_path)
    heat=capture[0].axes[0].images[0]
    assert heat.get_clim()==(-.30,.30)
    assert np.max(np.abs(heat.get_array()))<=.031
    for row in json.loads((BUNDLE/'figure_data_manifest.json').read_text())['inputs']:
        assert hashlib.sha256((BUNDLE/row['path']).read_bytes()).hexdigest()==row['sha256']

def test_notebook_has_individual_figures_and_safe_custom_output():
    book=json.loads((BUNDLE/'DD_Figures_Custom.ipynb').read_text(encoding='utf-8'))
    code='\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
    assert "OUTPUT = HERE / 'custom_figures'" in code
    assert 'STYLE = {' in code
    assert code.count('preview(\'figure')==6
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in book['cells'])
