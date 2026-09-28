"""Create a portable, private figure notebook from unchanged frozen summaries."""
import hashlib
import argparse
import importlib.util
import json
import re
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/rerank/figures/plot_digital_discovery_figures.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', action='store_true', help='Package after notebook Run All QA')
    parser.add_argument('--tag', default='20260924_unified')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9]{8}(?:_[a-z0-9]+)*', args.tag):
        parser.error('Invalid tag')
    DEST = ROOT / 'release' / ('dd_figure_notebook_' + args.tag)
    DEST.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('dd_figures', SOURCE)
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    records = []
    pairs = [
        ('transfer_analysis/j1_j2_filtered_v2/j2_headroom_capture.csv', 'digital_discovery_round_jk/reanalysis/j1_j2_filtered_v2/j2_headroom_capture.csv'),
        ('transfer_analysis/k1/k1_truncated_pool_effects.csv', 'digital_discovery_round_jk/reanalysis/k1/k1_truncated_pool_effects.csv'),
        ('historical_anchor/numerical_freeze_v2/paired_effects.csv', 'jcheminform_revision/numerical_freeze_v2/paired_effects.csv'),
        ('transfer_analysis/m2b_transfer_inference/m2b_transfer_loss_inference.csv', 'digital_discovery_round_jk/reanalysis/m2b_transfer_inference/m2b_transfer_loss_inference.csv'),
        ('transfer_analysis/m2_pool_shift/m2_pool_summary.csv', 'digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_pool_summary.csv'),
        ('transfer_analysis/m2_pool_shift/m2_transfer_loss_associations.csv', 'digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_transfer_loss_associations.csv'),
    ]
    for public, local in pairs:
        src = plot.artifact_path(ROOT, 'outputs/'+public, 'outputs/'+local)
        dst = DEST / 'outputs' / public
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        records.append({'source': src.relative_to(ROOT).as_posix(), 'path': dst.relative_to(DEST).as_posix(), 'sha256': digest(src)})
    # Capture the identical plotting inputs through the existing reader, without
    # rerunning models or statistical inference. Persist round-trippable floats.
    captured = {}
    plot._plot_operational_performance = lambda out, labels, within, end: captured.update(labels=labels, within=within, end=end)
    plot.figure_operational_performance(ROOT, DEST)
    rows = []
    for i, pool in enumerate(('historical_cap10', 'aizynthfinder_only', 'localretro_only', 'merged')):
        for j, system in enumerate(('candidate_prior', 'prior_2d', 'prior_2d_unimol')):
            rows.append(dict(pool=pool, system=system, within_pool_top1=captured['within'][i][j], end_to_end_top1=captured['end'][i][j]))
    compact = DEST/'outputs/transfer_analysis/figure_inputs/operational_performance.csv'
    compact.parent.mkdir(parents=True, exist_ok=True)
    plot.pd.DataFrame(rows).to_csv(compact, index=False, float_format='%.17g')
    sources = [ROOT/'outputs/jcheminform_revision/tuned_primary/conformer_seed_42/g1_20seed/test_results/manifest.json']
    sources += [ROOT/f'outputs/digital_discovery_round_jk/k1/{p}/test_results/manifest.json' for p in ('aizynthfinder_only','localretro_only','merged')]
    records.append({'path': compact.relative_to(DEST).as_posix(), 'sha256': digest(compact),
                    'derivation': 'Existing plot reader; paired seed means and coverage, no new inference',
                    'sources': {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}})
    adaptation = DEST/'outputs/transfer_analysis/figure_inputs/adaptation.csv'
    plot.adaptation_frame(ROOT).to_csv(adaptation,index=False,float_format='%.17g')
    sources = [ROOT/'outputs/digital_discovery_final_round/n5_crossed_transfer/n5_crossed_transfer_intervals.csv',
               ROOT/'outputs/dd_final_round/N3_S3/transfer_effects.csv']
    sources += [ROOT/f'outputs/dd_final_round/N3_S4_final/{p}/statistics.json' for p in ('localretro_only','merged')]
    records.append({'path':adaptation.relative_to(DEST).as_posix(), 'sha256':digest(adaptation),
                    'derivation':'Frozen common-cohort crossed intervals; no new inference',
                    'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in sources}})
    shutil.copy2(SOURCE, DEST/SOURCE.name)
    design_dir = DEST/'conceptual_design'
    design_dir.mkdir(exist_ok=True)
    for suffix in ('pdf','png'):
        shutil.copy2(ROOT/f'paper/digital_discovery/figures/figure1_dd_design.{suffix}',
                     design_dir/f'figure1_dd_design.{suffix}')
    cells = []

    def cell(kind, source):
        c = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(True)}
        if kind == 'code': c.update(execution_count=None, outputs=[])
        cells.append(c)

    cell('markdown', '''# Digital Discovery — tự chỉnh và vẽ lại figures

Gói độc lập: giữ notebook, `plot_digital_discovery_figures.py` và thư mục `outputs/` cùng nhau.
Mở bằng Jupyter / VS Code, chọn Python rồi **Run All**. Nếu thiếu thư viện, chạy trong terminal:
`python -m pip install matplotlib==3.10.3 numpy==1.26.4 pandas==2.2.3 ipykernel pillow`

Chỉ vẽ từ numerical summaries đã đóng băng; không training, inference, bootstrap hoặc đổi protocol.
Figure 2, 3, 5 giữ primary analyses. Figure 4 so sánh primary và adaptation sensitivities,
cùng estimand và common cohort; không thay primary bằng S4. Heatmap chuyển sang Figure S1.
Kết quả vẽ lưu vào `custom_figures/`, không ghi đè paper.
Chỉ Figure 1 là sơ đồ khái niệm chỉnh trong PowerPoint; notebook giữ bản xuất đã duyệt.
Figures 2–5 và S1 đều được vẽ bằng Matplotlib từ CSV, không dùng PowerPoint.
''')
    cell('code', '''from pathlib import Path
import importlib.util
import json, hashlib
from IPython.display import display, Image

HERE = Path.cwd().resolve()
# Nếu kernel mở ở thư mục khác, sửa HERE thành thư mục chứa notebook.
assert (HERE / 'plot_digital_discovery_figures.py').is_file(), 'Hãy chỉnh HERE.'
spec = importlib.util.spec_from_file_location('dd_figures_custom', HERE / 'plot_digital_discovery_figures.py')
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)
OUTPUT = HERE / 'custom_figures'
OUTPUT.mkdir(exist_ok=True)

audit = json.loads((HERE / 'figure_data_manifest.json').read_text())
for row in audit['inputs']:
    assert hashlib.sha256((HERE / row['path']).read_bytes()).hexdigest() == row['sha256']
print('Frozen input hashes verified. Output:', OUTPUT)
''')
    cell('markdown', '''## 1. Chỉnh style tại đây

Màu: mã HEX. `FONT_SCALE` tăng/giảm tất cả chữ; `HEIGHT_SCALE` tăng khoảng trống theo chiều dọc.
`FULL_WIDTH_IN` là chiều rộng figure tính bằng inch (174 mm mặc định).
`GRID_ALPHA=0` tắt nét lưới. Giữ thang heatmap ±0.30 để không phóng đại tương quan nhỏ.
Chỉnh xong chạy lại cell này và các figure bên dưới. Để đổi layout hoặc nội dung nhãn,
sửa hàm tương ứng trong `plot_digital_discovery_figures.py`, rồi chạy lại cell import đầu tiên.
''')
    cell('code', '''STYLE = {
    'NAVY': '#253344',       # Historical pool
    'BLUE': '#4477AA',       # AiZ / 2D / original procedure
    'ORANGE': '#CC8844',     # LocalRetro / augmented / prior reselection
    'PURPLE': '#8877AA',     # Merged
    'TEAL': '#559977',       # Full-grid adaptation
    'GREY': '#83909B',
    'PALE_BLUE': '#B9CCDF',
    'PANEL_FILL': '#F2F6FA',
    'LIGHT': '#E9EDF0',
    'INK': '#253344',
    'FONT_FAMILY': 'Arial',  # DejaVu Sans cũng có thể dùng trên Linux
    'FONT_SCALE': 1.0,
    'HEIGHT_SCALE': 1.0,
    'GRID_ALPHA': 0.35,
    'FULL_WIDTH_IN': 174 / 25.4,
    'RASTER_DPI': 600,
}
figures.configure_style(STYLE)
(OUTPUT / 'style.json').write_text(json.dumps(STYLE, indent=2))

def preview(stem):
    display(Image(filename=str(OUTPUT / (stem + '.png')), width=900))
''')
    jobs = [
        ('1 — Sơ đồ khái niệm (bản xuất PowerPoint đã duyệt)', None, 'figure1_dd_design', False),
        ('2 — Coverage và matched ranking', 'figure_coverage_effects', 'figure2_dd_coverage_effects', True),
        ('3 — Transfer loss và candidate-list shift', 'figure_transfer_shift', 'figure3_dd_transfer_shift', True),
        ('4 — Primary và adaptation sensitivities', 'figure_adaptation', 'figure4_dd_adaptation', True),
        ('5 — Conditional và end-to-end performance', 'figure_operational_performance', 'figure5_dd_operational_performance', True),
        ('S1 — Descriptor diagnostics (không phải causal mechanism)', 'figure_descriptor_diagnostics', 'figure4_dd_descriptor_diagnostics', True),
    ]
    for title, fn, stem, uses_data in jobs:
        cell('markdown', '## Figure '+title)
        if fn is None:
            cell('code', "import shutil\nfor ext in ('pdf', 'png'):\n    shutil.copy2(HERE / 'conceptual_design' / ('figure1_dd_design.' + ext), OUTPUT / ('figure1_dd_design.' + ext))\npreview('figure1_dd_design')\n")
        else:
            cell('code', f"figures.{fn}({'HERE, ' if uses_data else ''}OUTPUT)\npreview('{stem}')\n")
    cell('markdown', '''## Đưa hình custom vào Overleaf

Upload 6 PDF trong `custom_figures/` vào `figures/` của project Overleaf mới, giữ nguyên tên.
File `figure4_dd_descriptor_diagnostics.pdf` là Figure S1 trong SI (giữ tên file để truy vết).
LaTeX sẽ dùng hình mới mà không cần sửa `main.tex`. PNG chỉ để xem nhanh; PDF là vector.
Notebook và dữ liệu không cần upload lên Overleaf. Luôn kiểm tra lại chữ/legend sau khi tự đổi style.
''')
    notebook={'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.10'}}, 'nbformat':4, 'nbformat_minor':4}
    (DEST/'DD_Figures_Custom.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    (DEST/'figure_data_manifest.json').write_text(json.dumps({'inputs':records, 'plot_code_sha256':digest(SOURCE), 'scope':'Cosmetic redesign only; scientific artifacts unchanged'},indent=2)+'\n',encoding='utf-8')
    print(DEST/'DD_Figures_Custom.ipynb')
    if args.archive:
        assert len(list((DEST/'custom_figures').glob('figure*.pdf'))) == 6
        files = [DEST/'DD_Figures_Custom.ipynb', DEST/SOURCE.name, DEST/'figure_data_manifest.json']
        files += sorted((DEST/'outputs').rglob('*.csv'))
        files += sorted((DEST/'conceptual_design').glob('*'))
        files += sorted(p for p in (DEST/'custom_figures').iterdir() if p.suffix in {'.pdf','.png','.json'})
        archive = DEST.parent/f'DD_Figures_Notebook_{args.tag}.zip'
        # A regenerable figure distribution, never a scientific freeze.
        with ZipFile(archive,'w',compression=ZIP_DEFLATED,compresslevel=9) as z:
            for p in files:z.write(p,p.relative_to(DEST).as_posix())
        with ZipFile(archive) as z:
            assert z.testzip() is None
            for p in files:assert hashlib.sha256(z.read(p.relative_to(DEST).as_posix())).hexdigest()==digest(p)
        print(json.dumps({'archive':str(archive),'sha256':digest(archive),'members':len(files)}))


if __name__=='__main__': main()
