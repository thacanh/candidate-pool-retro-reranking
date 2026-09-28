import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def fixture(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/apply_dd_n3_tie_patch.py'
    spec = importlib.util.spec_from_file_location('n3_patch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, patch = tmp_path / 'bundle', tmp_path / 'patch'
    patch.mkdir()
    record = {'patch_id': module.PATCH_ID, 'files': {}}
    manifest = {'job': 'N3', 'files': {}}
    for name in module.TARGETS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'old')
        source = patch / 'payload' / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b'new')
        record['files'][name] = {'before_sha256': module.digest(b'old'), 'after_sha256': module.digest(b'new')}
        manifest['files'][name] = {'size_bytes': 3, 'sha256': module.digest(b'old')}
    original = json.dumps(manifest).encode()
    (root / 'BUNDLE_INPUTS.json').write_bytes(original)
    record['base_manifest_sha256'] = module.digest(original)
    (patch / 'PATCH.json').write_text(json.dumps(record))
    logs = root / 'logs/dd_final_round'
    logs.mkdir(parents=True)
    (logs / 'N3.exit').write_text('1')
    (logs / 'N3_S1.log').write_text('N3-S1 frozen rank reproduction failed: historical_cap10/baseline/43/46527')
    monkeypatch.setattr(module, '__file__', str(patch / 'apply.py'))
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1))
    return module, root, original


def test_patch_preserves_old_bytes_logs_and_updates_only_transport_hashes(tmp_path, monkeypatch):
    module, root, original = fixture(tmp_path, monkeypatch)
    module.apply(root)
    backup = root / '.patch_backups' / module.PATCH_ID
    assert (backup / 'BUNDLE_INPUTS.json').read_bytes() == original
    assert (backup / 'logs/N3.exit').read_text() == '1'
    assert not (root / 'logs/dd_final_round/N3.exit').exists()
    manifest = json.loads((root / 'BUNDLE_INPUTS.json').read_text())
    for name in module.TARGETS:
        assert (backup / name).read_bytes() == b'old'
        assert (root / name).read_bytes() == b'new'
        assert manifest['files'][name]['sha256'] == hashlib.sha256(b'new').hexdigest()


def test_patch_refuses_changed_bundle_before_any_write(tmp_path, monkeypatch):
    module, root, original = fixture(tmp_path, monkeypatch)
    (root / module.TARGETS[0]).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Base bundle differs'):
        module.apply(root)
    assert not (root / '.patch_backups').exists()
    assert (root / 'BUNDLE_INPUTS.json').read_bytes() == original
