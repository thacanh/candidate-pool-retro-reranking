"""Private archive relocation must not change scientific payloads."""
import importlib.util
from pathlib import Path
import pytest

helper = Path(__file__).resolve().parents[1] / 'scripts/local_archive_paths.py'
if not helper.exists():
    pytest.skip('Private archive tooling is not part of this payload', allow_module_level=True)
spec = importlib.util.spec_from_file_location('local_archive_paths', helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('name,group', [('N3_S1.zip','N3'), ('N3_S2.receipt.json','N3'), ('N3_S4_merged.zip','N3_S4'), ('N4.zip','N4')])
def test_organized_and_legacy_locations(tmp_path, name, group):
    with pytest.raises(FileNotFoundError):
        module.imported_archive(tmp_path, name)
    legacy = tmp_path / name
    legacy.write_bytes(b'payload')
    assert module.imported_archive(tmp_path, name) == legacy
    organized = tmp_path / 'archive/imports/dd_final_round' / group / name
    organized.parent.mkdir(parents=True)
    legacy.rename(organized)
    assert module.imported_archive(tmp_path, name) == organized
    legacy.write_bytes(b'other payload')
    with pytest.raises(FileExistsError):
        module.imported_archive(tmp_path, name)


@pytest.mark.parametrize('name', ['../N4.zip', 'unknown.zip', 'x/N4.zip', 'x\\N4.zip'])
def test_reject_unsafe_or_unknown_names(tmp_path, name):
    with pytest.raises(ValueError):
        module.imported_archive(tmp_path, name)
