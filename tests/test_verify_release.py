"""Integrity checks are optional, but must reject missing or changed files."""
import hashlib

import pytest

from rerank.data.verify_release import verify


def fixture(root):
    data = b'original data\n'
    (root / 'data.txt').write_bytes(data)
    (root / 'checksums.sha256').write_text(
        f'{hashlib.sha256(data).hexdigest()}  data.txt\n', encoding='utf-8')


def test_tampered_payload_is_rejected(tmp_path):
    fixture(tmp_path)
    assert verify(tmp_path)['files'] == 1
    (tmp_path / 'data.txt').write_text('changed')
    with pytest.raises(ValueError, match='Changed file'):
        verify(tmp_path)


def test_outside_path_is_rejected(tmp_path):
    (tmp_path / 'checksums.sha256').write_text('0'*64 + '  ../outside.txt\n')
    with pytest.raises(ValueError, match='Unsafe'):
        verify(tmp_path)


def test_editable_install_metadata_is_not_a_payload_change(tmp_path):
    fixture(tmp_path)
    metadata = tmp_path / 'src/example.egg-info'
    metadata.mkdir(parents=True)
    (metadata / 'PKG-INFO').write_text('installed metadata')
    assert verify(tmp_path)['files'] == 1
