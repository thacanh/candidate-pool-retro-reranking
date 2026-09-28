"""Build/apply one reviewed N4 continuation patch without changing scientific artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

PATCH_ID = 'n4_frozen_reuse_20260924'
TARGETS = (
    'src/rerank/experiments/run_dd_n4_historical.py',
    'scripts/run_dd_n4_remote.py', 'scripts/resume_dd_n4_evaluation.py',
    'scripts/resume_dd_n4_after_patch.sh', 'scripts/diagnose_dd_n4_reproduction.py',
    'tests/test_dd_n4_reuse.py', 'docs/dd_remote_approval.json',
    'docs/n4_reproduction_audit_approved.jsonl',
)
AUDIT_SHA = '3ee8816ea0962c42d2121580d7175c55e2829cb606d3b1e781ccf21ab236199d'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def build(base, output, audit):
    root = Path(__file__).resolve().parents[1]
    audit_bytes = Path(audit).read_bytes().replace(b'\r\n', b'\n')
    if digest(audit_bytes) != AUDIT_SHA:
        raise ValueError('Wrong owner-reviewed audit')
    with zipfile.ZipFile(base) as old:
        before_manifest = old.read('BUNDLE_INPUTS.json')
        payload = {name: (root / name).read_bytes() for name in TARGETS
                   if not name.startswith('docs/')}
        payload['scripts/resume_dd_n4_after_patch.sh'] = payload['scripts/resume_dd_n4_after_patch.sh'].replace(b'\r\n', b'\n')
        approval = json.loads(old.read('docs/dd_remote_approval.json'))
        approval['n4_replay_reuse'] = {
            'approval_date': '2026-09-24', 'owner_quote': 'ok tôi đồng ý',
            'audit_sha256': AUDIT_SHA,
            'policy': 'Retain all frozen first-five predictions after exact replay-audit verification; no retraining; disclose non-exact rank reproduction.',
            'scope': 'N4 historical seeds 42--46 only; no altered fitting, features, models or prior scientific files',
            'supervisor_signature_present': False,
        }
        for name, data in payload.items():
            if Path(name).name in ('run_dd_n4_historical.py', 'run_dd_n4_remote.py', 'resume_dd_n4_evaluation.py'):
                approval['runner_sha256']['dd-n4-morgan-atom-transfer-v1'][Path(name).name] = digest(data)
        payload['docs/dd_remote_approval.json'] = (json.dumps(approval, indent=2, ensure_ascii=False) + '\n').encode()
        payload['docs/n4_reproduction_audit_approved.jsonl'] = audit_bytes
        records = {name: {'before_sha256': digest(old.read(name)) if name in old.namelist() else None,
                          'after_sha256': digest(data)} for name, data in payload.items()}
    manifest = dict(patch_id=PATCH_ID, base_manifest_sha256=digest(before_manifest), files=records,
                    scope='Approved audited frozen-prediction reuse and evaluation-only continuation')
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('PATCH.json', json.dumps(manifest, indent=2))
        z.write(__file__, 'apply.py')
        for name, data in payload.items():
            z.writestr('payload/' + name, data)
    print(json.dumps(dict(archive=str(output), size_bytes=Path(output).stat().st_size, sha256=filehash(output))))


def apply(root, patchdir=None, check_session=True):
    root = Path(root).resolve(strict=True)
    patchdir = Path(patchdir or Path(__file__).resolve().parent)
    record = json.loads((patchdir / 'PATCH.json').read_text())
    if record['patch_id'] != PATCH_ID or set(record['files']) != set(TARGETS):
        raise ValueError('Wrong patch')
    if check_session and subprocess.run(['tmux', 'has-session', '-t', 'dd-n4'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('N4 session still active; patch refused')
    logs = root / 'logs/dd_final_round'
    if ((logs / 'N4.exit').read_text().strip() != '1' or
        'N4 first-five historical Morgan test ranks do not reproduce.' not in (logs / 'N4_run.log').read_text()):
        raise ValueError('Expected failed evaluation was not found')
    out = root / 'outputs/dd_final_round/N4'
    if (out / 'manifest.json').exists():
        raise ValueError('N4 already finalized')
    for pool in ('historical', 'aizynthfinder_only', 'localretro_only', 'merged'):
        if not (out / pool / 'models/freeze.json').is_file():
            raise ValueError(f'Missing model freeze: {pool}')
        results = out / pool / 'test_results'
        if results.exists() and any(results.iterdir()):
            raise ValueError('Existing test output; manual review required')
    manifest_path = root / 'BUNDLE_INPUTS.json'
    original = manifest_path.read_bytes()
    if digest(original) != record['base_manifest_sha256']:
        raise ValueError('Wrong base N4 bundle')
    manifest = json.loads(original)
    for name, info in manifest['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Unsafe path')
        if path.stat().st_size != info['size_bytes'] or filehash(path) != info['sha256']:
            raise ValueError(f'Base bundle changed: {name}')
    replacements = {}
    for name, info in record['files'].items():
        target = root / name
        if (filehash(target) if target.exists() else None) != info['before_sha256']:
            raise ValueError(f'Existing target differs: {name}')
        data = (patchdir / 'payload' / name).read_bytes()
        if digest(data) != info['after_sha256']:
            raise ValueError('Patch payload differs')
        replacements[name] = data
        manifest['files'][name] = dict(size_bytes=len(data), sha256=digest(data))
    backup = root / '.patch_backups' / PATCH_ID
    backup.mkdir(parents=True, exist_ok=False)
    for name in (*TARGETS, 'BUNDLE_INPUTS.json', 'logs/dd_final_round/N4.exit', 'logs/dd_final_round/N4_run.log'):
        if (root / name).exists():
            dest = backup / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, dest)
    replacements['BUNDLE_INPUTS.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    for name, data in replacements.items():
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + '.n4patch.tmp')
        with temp.open('xb') as handle:
            handle.write(data)
        os.replace(temp, dest)
    record['updated_manifest_sha256'] = filehash(manifest_path)
    (backup / 'patch_audit.json').write_text(json.dumps(record, indent=2))
    print('PATCH APPLIED. Scientific outputs unchanged; old source, approval and logs backed up.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path)
    p.add_argument('--build-from', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--audit', type=Path)
    a = p.parse_args()
    if a.build_from and a.output and a.audit:
        build(a.build_from, a.output, a.audit)
    elif a.root:
        apply(a.root)
    else:
        p.error('Use --root or --build-from/--output/--audit')
