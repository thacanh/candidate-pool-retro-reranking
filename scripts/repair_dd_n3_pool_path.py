"""Repair the original Vast N3 bundle's merged-pool alias and resume S2 only.

The live runner and its authorization bytes remain unchanged. The alias is
a hard link to the existing, index-checksummed pool, with no duplicate data.
Incomplete pre-training S2 outputs and failed logs are preserved in a backup.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PATCH_ID = 'n3_merged_path_20260923'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def resume_script(source):
    start = '"$PY" -m rerank.experiments.run_dd_n3_diagnostic'
    step2 = '"$PY" scripts/run_dd_n3_remote.py s2'
    if source.count(start) != 1 or source.count(step2) != 1:
        raise ValueError('Unrecognized N3 wrapper')
    prefix = source.split(start)[0]
    suffix = source[source.index(step2):]
    return prefix + ('"$PY" scripts/dd_archive_stage.py verify --receipt '
                     '"$ARCHIVES/N3_S1.receipt.json"\n\n') + suffix


def repair(root):
    root = root.resolve(strict=True)
    logs = root / 'logs/dd_final_round'
    if subprocess.run(['tmux', 'has-session', '-t', 'dd-n3'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('N3 session still running')
    if (logs / 'N3.exit').read_text().strip() != '1':
        raise ValueError('Expected failed run not found')
    error = (logs / 'N3_S2.log').read_text()
    if 'FileNotFoundError' not in error or '/merged.jsonl' not in error:
        raise ValueError('Unexpected failure; do not restart automatically')
    s2 = root / 'outputs/dd_final_round/N3_S2'
    s3 = root / 'outputs/dd_final_round/N3_S3'
    if ((s3.exists() and any(s3.iterdir()))
            or (s2 / 'choice_freeze.json').exists()
            or (s2 / 'manifest.json').exists()
            or any(s2.rglob('trial.json')) or any(s2.rglob('*.pt'))):
        raise ValueError('Training/frozen outputs exist; separate resume audit required')
    subprocess.run([sys.executable, 'scripts/dd_pack_remote.py', 'verify-extracted',
                    '--root', str(root), '--manifest', str(root/'BUNDLE_INPUTS.json')],
                   cwd=root, check=True)
    subprocess.run([sys.executable, 'scripts/dd_archive_stage.py', 'verify',
                    '--receipt', '/workspace/dd_n3_archives/N3_S1.receipt.json'],
                   cwd=root, check=True)
    pool = root / 'outputs/jcheminform_revision/candidate_pools/ws_e_three_pools_filtered_v2'
    original = pool / 'merged_canonical_union.jsonl'
    alias = pool / 'merged.jsonl'
    index = json.loads((root/'outputs/jcheminform_revision/ws_e_streaming/index/merged_index.json').read_text())
    expected = index['pool']
    if original.stat().st_size != expected['size_bytes'] or sha(original) != expected['sha256']:
        raise ValueError('Merged pool differs from frozen index')
    if alias.exists() and not os.path.samefile(alias, original):
        raise ValueError('An unrelated merged.jsonl already exists')
    wrapper = (root/'scripts/run_dd_n3_remote.sh').read_text()
    continuation = resume_script(wrapper)
    backup = root / '.patch_backups' / PATCH_ID
    backup.mkdir(parents=True, exist_ok=False)
    if not alias.exists():
        os.link(original, alias)
    moved = []
    if s2.exists():
        destination = backup / 'N3_S2_incomplete'
        # Both explicit paths are constrained to this bundle before moving.
        if not s2.resolve().is_relative_to(root) or not destination.resolve().is_relative_to(root):
            raise ValueError('Output path escaped bundle')
        shutil.move(str(s2), str(destination))
        moved.append(str(destination))
    for name in ('N3_master.log', 'N3.exit', 'N3_S2.log'):
        source = logs / name
        if source.exists():
            shutil.move(str(source), str(backup / name))
    pipeline = backup / 'resume_s2_pipeline.sh'
    pipeline.write_text(continuation, encoding='utf-8', newline='\n')
    audit = {'repair_id': PATCH_ID, 'scope': 'path compatibility and pre-training restart only',
             'pool_sha256': expected['sha256'], 'alias': str(alias),
             'alias_is_same_file': os.path.samefile(alias, original),
             'scientific_inputs_modified': False, 's1_rerun': False,
             'preserved_partial_outputs': moved, 'continuation_sha256': sha(pipeline)}
    (backup/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('REPAIR VERIFIED; S1 preserved; incomplete S2/logs backed up; ready to resume S2.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    repair(parser.parse_args().root)
