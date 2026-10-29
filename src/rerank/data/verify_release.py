"""Check the downloaded files against the repository's SHA256 ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

IGNORED_PARTS = {'.git', '__pycache__', '.pytest_cache', '.ipynb_checkpoints',
                 '.venv', 'venv', 'build', 'dist', 'figures_rebuilt'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def payload_files(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob('*')
        if p.is_file() and not any(
            part in IGNORED_PARTS or part.endswith('.egg-info') or part.startswith('.pytest_tmp')
            for part in p.relative_to(root).parts
        )
    }


def verify(root: Path) -> dict:
    root = root.resolve()
    ledger = root / 'checksums.sha256'
    expected_paths = set()
    size = 0
    for line in ledger.read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        relative = PurePosixPath(name)
        path = root / name
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name
                or name in expected_paths or path.is_symlink()
                or not path.resolve().is_relative_to(root)):
            raise ValueError(f'Unsafe or duplicate ledger path: {name}')
        if sha256(path) != expected:
            raise ValueError(f'Changed file: {name}')
        expected_paths.add(name)
        size += path.stat().st_size
    actual_paths = payload_files(root) - {'checksums.sha256'}
    if actual_paths != expected_paths:
        raise ValueError(f'Unlisted or missing files: {sorted(actual_paths ^ expected_paths)}')
    return dict(status='verified', files=len(expected_paths), bytes=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    print(json.dumps(verify(parser.parse_args().repo_root), indent=2))


if __name__ == '__main__':
    main()
