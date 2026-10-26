"""Portable validation-only N3-S4 search. No official-test loader or evaluator."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from rerank.tuning import enumerate_d1_grid

PROTOCOL = "dd-n3-s4-validation-grid-v1"
POOLS = ("localretro_only", "merged")
SEEDS = tuple(range(42, 47))
ARMS = ("baseline", "augmented")
GRID = tuple(asdict(item) for item in enumerate_d1_grid())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def verify(root):
    manifest = read(root / "S4_INPUTS.json")
    if (manifest["protocol_id"] != PROTOCOL or manifest["grid"] != list(GRID)
            or manifest["seeds"] != list(SEEDS) or manifest["test_data_packaged"] is not False):
        raise ValueError("S4 input contract differs.")
    for name, rec in manifest["files"].items():
        p = root / name
        if (p.is_symlink() or not p.is_file() or p.stat().st_size != rec["size_bytes"]
                or sha(p) != rec["sha256"]):
            raise ValueError(f"Frozen input differs: {name}")
    for pool, transform in zip(POOLS, ("log", "rank")):
        selection = read(root / "inputs" / pool / "manifest.json")
        if (selection["protocol_id"] != PROTOCOL or selection["pool_name"] != pool
                or selection["prior_transform"] != transform
                or selection["test_partition_loaded"] is not False
                or selection["test_ground_truth_loaded"] is not False
                or selection["seeds"] != list(SEEDS)):
            raise ValueError("Wrong validation-only selection.")
        for rec in [selection["validation"], *selection["train_pairs"].values()]:
            p = root / "inputs" / pool / Path(rec["path"]).name
            if sha(p) != rec["sha256"]:
                raise ValueError("Selection fingerprint mismatch.")
    return manifest


def task_id(config, arm, seed):
    return f"cfg_{config['index']:03d}/{arm}/seed_{seed}"


def accepted(root, pool, config, arm, seed):
    folder = root / "results/N3_S4" / pool / "trials" / task_id(config, arm, seed)
    receipts = sorted(folder.glob("attempt_*/accepted.json"))
    if len(receipts) > 1:
        raise ValueError(f"Multiple completed attempts: {folder}")
    if not receipts:
        return None
    receipt = read(receipts[0])
    if receipt["input_contract_sha256"] != sha(root / "S4_INPUTS.json"):
        raise ValueError("Trial was run against a different input/code contract.")
    trialpath = receipts[0].parent / "validation" / arm / f"seed_{seed}" / "trial.json"
    if sha(trialpath) != receipt["trial_sha256"]:
        raise ValueError("Completed trial changed.")
    trial = read(trialpath)
    if (trial["protocol_id"] != PROTOCOL or trial["pool_name"] != pool
            or trial["arm"] != arm or trial["seed"] != seed
            or trial["config"] != config or trial["test_partition_loaded"] is not False
            or not math.isfinite(trial["best_validation_mrr"])
            or not 0 <= trial["best_validation_mrr"] <= 1):
        raise ValueError("Completed trial identity/metric invalid.")
    if trial["selection_manifest"]["sha256"] != sha(root / "inputs" / pool / "manifest.json"):
        raise ValueError("Trial selection input changed.")
    for key in ("checkpoint", "normalizer"):
        path = trialpath.parent / Path(trial[key]["path"]).name
        if path.stat().st_size != trial[key]["size_bytes"] or sha(path) != trial[key]["sha256"]:
            raise ValueError(f"Completed trial {key} changed.")
    return trialpath, trial


def fit(root, pool, config, arm, seed):
    folder = root / "results/N3_S4" / pool / "trials" / task_id(config, arm, seed)
    folder.mkdir(parents=True, exist_ok=True)
    number = max([int(p.name.split('_')[1]) for p in folder.glob("attempt_*")] or [0]) + 1
    attempt = folder / f"attempt_{number:03d}"
    attempt.mkdir()
    candidate = attempt / "candidate.json"
    write(candidate, {"protocol_id": PROTOCOL, "record_kind": "validation_grid_candidate",
                      "test_partition_loaded": False, "selected_baseline": {"config": config},
                      "selected_augmented": {"config": config}})
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    command = [sys.executable, "-m", "rerank.experiments.run_ws_e_ranking", "fit-validation",
               "--selection-root", str(root / "inputs" / pool), "--primary-freeze", str(candidate),
               "--output-root", str(attempt), "--arm", arm, "--seed", str(seed), "--device", "cpu"]
    with (attempt / "fit.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Trial stopped (rc={result.returncode}): {attempt / 'fit.log'}")
    trialpath = attempt / "validation" / arm / f"seed_{seed}" / "trial.json"
    write(attempt / "accepted.json", {"input_contract_sha256": sha(root / "S4_INPUTS.json"),
                                     "trial_sha256": sha(trialpath)})
    accepted(root, pool, config, arm, seed)


def select_winner(rows):
    if len(rows) != 81 or {r["config"]["index"] for r in rows} != set(range(81)):
        raise ValueError("Selection requires all 81 configurations.")
    best = max(r["mean_validation_mrr"] for r in rows)
    return min((r for r in rows if best - r["mean_validation_mrr"] <= 1e-12),
               key=lambda r: r["config"]["index"])


def freeze(root, pool):
    selected, all_rows, trials = {}, [], {}
    for arm in ARMS:
        rows = []
        for config in GRID:
            values = []
            for seed in SEEDS:
                item = accepted(root, pool, config, arm, seed)
                if item is None:
                    raise ValueError("Cannot freeze an incomplete grid.")
                path, trial = item
                values.append(trial["best_validation_mrr"])
                trials[task_id(config, arm, seed)] = {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}
            row = {"arm": arm, "config": config, "per_seed_validation_mrr": values,
                   "mean_validation_mrr": sum(values) / len(values)}
            rows.append(row)
            all_rows.append(row)
        selected[arm] = select_winner(rows)
    out = root / "results/N3_S4" / pool
    manifest = {"status": "complete", "stage": "validation_search_only", "protocol_id": PROTOCOL,
                "comparator": "N3-S3 fixed configuration on identical pool/prior/features",
                "single_intended_change": "81-point per-arm pool-specific validation hyperparameter selection",
                "pool": pool, "prior_transform": "log" if pool == "localretro_only" else "rank",
                "seeds": list(SEEDS), "trial_count": 810, "selected": selected,
                "tie_rule": "within 1e-12 of best mean: lowest original D1 index",
                "all_validation_results": all_rows, "trials": trials,
                "input_contract_sha256": sha(root / "S4_INPUTS.json"),
                "test_partition_loaded": False, "test_evaluation_complete": False,
                "environment": {"python": sys.version, "platform": platform.platform()}}
    path = out / "manifest.json"
    if path.exists():
        old = read(path)
        for key in ("selected", "trials", "input_contract_sha256"):
            if old[key] != manifest[key]:
                raise ValueError("Existing validation freeze changed.")
    else:
        write(path, manifest)
    return path


def archive(root, pool):
    source = root / "results/N3_S4" / pool
    output = root / "archives"
    output.mkdir(exist_ok=True)
    zip_path = output / f"N3_S4_{pool}.zip"
    receiptpath = zip_path.with_suffix(".receipt.json")
    if receiptpath.exists():
        receipt = read(receiptpath)
        if sha(zip_path) != receipt["sha256"]:
            raise ValueError("Archive hash changed.")
        current = {p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
        if current != receipt["members"]:
            raise ValueError("Archived pool files changed.")
        return
    if zip_path.exists():
        raise FileExistsError("Archive exists without a receipt; preserve it for inspection.")
    members = {p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
    temporary = zip_path.with_suffix(".partial.zip")
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=6) as z:
        for name in members:
            z.write(source / name, f"N3_S4/{pool}/{name}")
    with ZipFile(temporary) as z:
        assert z.testzip() is None
        for name,digest in members.items():
            if hashlib.sha256(z.read(f"N3_S4/{pool}/{name}")).hexdigest() != digest:
                raise ValueError("Archive verification failed.")
    os.replace(temporary, zip_path)
    write(receiptpath, {"status": "verified", "protocol_id": PROTOCOL, "stage": "validation_search_only",
                        "archive": zip_path.name, "sha256": sha(zip_path), "members": members})


def run(root, workers):
    verify(root)
    if workers < 1 or workers > 32:
        raise ValueError("Workers must be in 1..32.")
    expected = {"torch":"2.2.2+cpu", "numpy":"1.26.4", "pandas":"1.5.3",
                "scipy":"1.12.0", "rdkit":"2025.9.6"}
    installed = {name:importlib.metadata.version(name) for name in expected}
    if installed != expected or platform.python_version() != "3.10.14":
        raise RuntimeError(f"Use the pinned S4 runtime before fitting: {installed}, Python {platform.python_version()}")
    lock = root / "S4_RUNNING.lock"
    with lock.open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "host": platform.node()}, f)
    try:
        for pool in POOLS:
            contract_path = root / "results/N3_S4" / pool / "execution_contract.json"
            contract = {"input_contract": read(root / "S4_INPUTS.json"), "packages":installed,
                        "python":platform.python_version(), "threads_per_worker":1,
                        "trial_level_resume":True, "epoch_level_resume":False}
            if contract_path.exists():
                if read(contract_path) != contract:
                    raise ValueError("Existing pool execution contract differs.")
            else:
                write(contract_path,contract)
            tasks = [(c,a,s) for c in GRID for a in ARMS for s in SEEDS
                     if accepted(root,pool,c,a,s) is None]
            completed, started, initial = 810-len(tasks), time.monotonic(), 810-len(tasks)
            def progress():
                rate = (completed-initial)/max(time.monotonic()-started,1)
                write(root / "progress.json", {"pool": pool, "completed": completed, "expected":810,
                      "workers":workers, "status":"running", "trial_resume":True,
                      "eta_seconds": (810-completed)/rate if rate else None})
            progress()
            iterator = iter(tasks)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending = set()
                for _ in range(min(workers,len(tasks))):
                    pending.add(executor.submit(fit,root,pool,*next(iterator)))
                while pending:
                    done,pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                        completed += 1
                        progress()
                        spec = next(iterator,None)
                        if spec is not None:
                            pending.add(executor.submit(fit,root,pool,*spec))
            freeze(root,pool)
            archive(root,pool)
            print(f"{pool}: 810/810 validation trials; freeze and archive verified",flush=True)
        write(root / "progress.json", {"status":"complete_validation_only", "completed":1620,
              "expected":1620, "test_partition_loaded":False,
              "next":"Review frozen choices before the 20-seed extension/test stage."})
    except BaseException as exc:
        write(root / "progress.json", {"status":"stopped", "error":str(exc), "resume":"rerun after resolving error; completed trials retained"})
        raise
    finally:
        lock.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("verify","run","status"))
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args(); root=args.root.resolve()
    if args.command=="verify":
        m=verify(root); print(json.dumps({"status":"verified","files":len(m['files']),"expected_trials":1620}))
    elif args.command=="status":
        p=root/'progress.json'; print(json.dumps(read(p),indent=2) if p.exists() else "Not started")
        lock=root/'S4_RUNNING.lock'
        if lock.exists(): print('Lock:',lock.read_text())
    else:
        run(root,args.workers)


if __name__=="__main__":
    main()
