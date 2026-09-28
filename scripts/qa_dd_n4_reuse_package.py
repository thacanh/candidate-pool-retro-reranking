"""Transport QA using a private temporary extraction and synthetic failure markers."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    repo = Path.cwd()
    # No production trial or model is created. Only failure-state markers for
    # apply-patch transport testing are placed in this explicitly QA directory.
    root = Path(tempfile.mkdtemp(prefix='dd_n4_patch_transport_qa_'))
    with zipfile.ZipFile(repo / 'release/dd_remote/dd_n4_vast_v2_20260923.zip') as z:
        z.extractall(root)
    logs = root / 'logs/dd_final_round'
    logs.mkdir(parents=True, exist_ok=True)
    (logs / 'N4.exit').write_text('1\n')
    (logs / 'N4_run.log').write_text('N4 first-five historical Morgan test ranks do not reproduce.\n')
    for pool in ('historical', 'aizynthfinder_only', 'localretro_only', 'merged'):
        path = root / f'outputs/dd_final_round/N4/{pool}/models/freeze.json'
        path.parent.mkdir(parents=True)
        path.write_text('{"synthetic_transport_marker_only": true}')
    spec = importlib.util.spec_from_file_location('patch', repo / 'scripts/apply_dd_n4_reuse_patch.py')
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    patch.apply(root, repo / 'release/dd_remote/n4_reuse_patch_qa', check_session=False)
    subprocess.run([sys.executable, 'scripts/dd_pack_remote.py', 'verify-extracted',
                    '--root', '.', '--manifest', 'BUNDLE_INPUTS.json'], cwd=root, check=True)
    env = dict(os.environ, PYTHONPATH=str(root/'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', DD_N4_REAL_REPLAY_QA='1',
               DD_N4_AUDIT_PATH=str(root/'docs/n4_reproduction_audit_approved.jsonl'))
    subprocess.run([sys.executable, '-m', 'pytest', '-q', '--import-mode=importlib',
                   'tests/test_dd_n4_reuse.py', 'tests/test_dd_n4_release.py',
                   'tests/test_dd_n3_n4_preparation.py'], cwd=root, env=env, check=True)
    for name in ('run_dd_n4_historical.py', 'run_dd_n4_remote.py', 'resume_dd_n4_evaluation.py'):
        parent = 'src/rerank/experiments' if name == 'run_dd_n4_historical.py' else 'scripts'
        actual = patch.filehash(root/parent/name)
        approval = json.loads((root/'docs/dd_remote_approval.json').read_text(encoding='utf-8'))
        assert actual == approval['runner_sha256']['dd-n4-morgan-atom-transfer-v1'][name]
    try:
        patch.apply(root, repo/'release/dd_remote/n4_reuse_patch_qa', check_session=False)
        raise AssertionError('Second application should fail')
    except ValueError as exc:
        assert 'base' in str(exc).lower()
    print(json.dumps(dict(status='pass', scope='transport + real five-model replay; no new training',
                         qa_temp_dir=str(root), remote_full_continuation_executed=False)))


if __name__ == '__main__':
    main()
