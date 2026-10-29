"""Presentation changes must retain the frozen data and a code-based Fig. 4."""
from pathlib import Path
import json
import numpy as np
from rerank.figures import plot_figures as plot

ROOT = Path(__file__).resolve().parents[1]


def test_adaptation_markers_and_intervals_are_csv_derived(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(plot, 'save', lambda fig, *_: captured.append(fig))
    plot.configure_style()
    plot.figure_adaptation(ROOT, tmp_path)
    fig = captured[0]
    data = plot.adaptation_frame(ROOT)
    for i, metric in enumerate(('top1', 'mrr')):
        for j, pool in enumerate(('localretro_only', 'merged')):
            axis = fig.axes[i * 2 + j]
            for k, stage in enumerate(('original', 'prior_reselected', 'full_grid')):
                row = plot.one(data, pool=pool, metric=metric, stage=stage)
                marker, _, collections = axis.containers[k].lines
                np.testing.assert_allclose(
                    np.asarray(marker.get_xdata(), dtype=float), [row.effect * 100]
                )
                segment = collections[0].get_segments()[0]
                np.testing.assert_allclose(segment[:, 0], [row.low * 100, row.high * 100])
                assert marker.get_color() == (plot.BLUE, plot.ORANGE, plot.TEAL)[k]
            assert axis.get_xlim() == (-3, 1.2)
    plot.plt.close(fig)


def test_shared_palette_and_separate_conceptual_diagram():
    assert (plot.BLUE, plot.ORANGE, plot.TEAL, plot.PURPLE) == (
        '#4477AA',
        '#CC8844',
        '#559977',
        '#8877AA',
    )
    assert plot.FONT_FAMILY == 'Arial'
    notebook = json.loads(
        (ROOT / 'notebooks/rebuild_data_figures.ipynb').read_text(encoding='utf8')
    )
    code = '\n'.join(
        ''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code'
    )
    assert 'from rerank.figures import plot_figures' in code
    assert 'figures.figure_adaptation(ROOT, OUT)' in code
    assert '.pptx' not in code and 'figure1' not in code
