"""Build/apply the hash-bound N3-S1 numerical-tie transport patch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

TARGETS = ('src/rerank/experiments/run_dd_n3_diagnostic.py', 'docs/dd_remote_approval.json')
PATCH_ID = 'n3_s1_identical_input_ties_20260923'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(base, output):
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(base) as archive:
        old_manifest = archive.read('BUNDLE_INPUTS.json')
        records = {}
        payloads = {}
        for name in TARGETS:
            before = archive.read(name)
            after = (root / name).read_bytes()
            records[name] = {'before_sha256': digest(before), 'after_sha256': digest(after)}
            payloads['payload/' + name] = after
    record = {'patch_id': PATCH_ID, 'base_manifest_sha256': digest(old_manifest),
              'scope': 'N3-S1 identical-input numerical-tie audit; no scientific artifact edited',
              'files': records}
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('PATCH.json', json.dumps(record, indent=2) + '\n')
        z.write(__file__, 'apply.py')
        z.write(root / 'scripts/resume_dd_n3_after_patch.sh', 'resume.sh')
        for name, content in payloads.items():
            z.writestr(name, content)
    print(json.dumps({'archive': str(output), 'sha256': digest(Path(output).read_bytes()),
                      'size_bytes': Path(output).stat().st_size}, indent=2))


def apply(root):
    root = root.resolve(strict=True)
    patchdir = Path(__file__).resolve().parent
    record = json.loads((patchdir / 'PATCH.json').read_text())
    if record['patch_id'] != PATCH_ID:
        raise ValueError('Wrong patch ID')
    if subprocess.run(['tmux', 'has-session', '-t', 'dd-n3'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('dd-n3 is still running; patch refused')
    logdir = root / 'logs/dd_final_round'
    if ((logdir / 'N3.exit').read_text().strip() != '1'
            or 'N3-S1 frozen rank reproduction failed: historical_cap10/baseline/43/46527'
            not in (logdir / 'N3_S1.log').read_text()):
        raise ValueError('Expected failed N3-S1 run was not found')
    for stage in ('N3_S1', 'N3_S2', 'N3_S3'):
        out = root / 'outputs/dd_final_round' / stage
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'{stage} has outputs; manual audit required')
    manifest_path = root / 'BUNDLE_INPUTS.json'
    original = manifest_path.read_bytes()
    if digest(original) != record['base_manifest_sha256']:
        raise ValueError('Wrong base bundle manifest; no files changed')
    manifest = json.loads(original)
    for name, item in manifest['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Unsafe bundle path')
        data = path.read_bytes()
        if len(data) != item['size_bytes'] or digest(data) != item['sha256']:
            raise ValueError(f'Base bundle differs: {name}')
    replacements = {}
    for name, item in record['files'].items():
        if name not in TARGETS:
            raise ValueError('Unexpected patch target')
        data = (patchdir / 'payload' / name).read_bytes()
        if (digest((root / name).read_bytes()) != item['before_sha256']
                or digest(data) != item['after_sha256']):
            raise ValueError(f'Patch checksum differs: {name}')
        replacements[name] = data
        manifest['files'][name] = {'size_bytes': len(data), 'sha256': digest(data)}
    backup = root / '.patch_backups' / PATCH_ID
    backup.mkdir(parents=True, exist_ok=False)
    for name in (*TARGETS, 'BUNDLE_INPUTS.json'):
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    replacements['BUNDLE_INPUTS.json'] = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    for name, data in replacements.items():
        target = root / name
        temporary = target.with_name(target.name + '.tie-patch.tmp')
        with temporary.open('xb') as handle:
            handle.write(data)
        os.replace(temporary, target)
    for name in ('N3_master.log', 'N3.exit', 'N3_S1.log', 'N3_preflight.log'):
        source = logdir / name
        if source.exists():
            destination = backup / 'logs' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
    record['updated_bundle_manifest_sha256'] = digest(manifest_path.read_bytes())
    (backup / 'patch_audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'PATCH VERIFIED AND APPLIED; original files/logs retained in {backup}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--build-from', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.build_from and args.output:
        build(args.build_from, args.output)
    elif args.root:
        apply(args.root)
    else:
        parser.error('Use --root, or --build-from and --output')
