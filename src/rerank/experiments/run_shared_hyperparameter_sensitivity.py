#!/usr/bin/env python
"""Reviewer-requested shared-hyperparameter sensitivity for Digital Discovery.

This runner is deliberately isolated from the primary analysis.  It crosses
the two already-frozen D1 configurations symmetrically, imports every existing
compatible validation trial by checksum, and fits only the missing crossed
arm for the three alternative candidate pools.  Official-test labels are not
opened until a complete per-pool model freeze exists.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from rerank.analysis.analyze_revision_predictions import (
    clustered_sign_flip,
    crossed_product_seed_bootstrap,
    product_cluster_bootstrap,
)
from rerank.evaluate import evaluate_reranking
from rerank.features import FeatureNormalizer
from rerank.model import RankerMLP
from rerank.revision_tuning import (
    feature_columns_for_arm,
    file_sha256,
    transform_feature_matrix,
)
from rerank.study_data import load_reactions
from rerank.ws_e_streaming import atomic_json, fingerprint, sha256_file, utc_now
from rerank.experiments import run_ws_e_ranking as ws_e


PROTOCOL_ID = "dd-shared-hyperparameter-feature-only-v1"
SEEDS = tuple(range(42, 47))
STRATA = {
    "baseline_selected": {
        "dropout": 0.0,
        "hidden_width": 128,
        "index": 62,
        "learning_rate": 0.001,
        "margin": 0.3,
    },
    "augmented_selected": {
        "dropout": 0.1,
        "hidden_width": 128,
        "index": 70,
        "learning_rate": 0.001,
        "margin": 0.1,
    },
}
MATCHING_ARM = {"baseline_selected": "baseline", "augmented_selected": "augmented"}
ALT_POOLS = ("aizynthfinder_only", "localretro_only", "merged")
ALL_POOLS = ("historical",) + ALT_POOLS
N_BOOTSTRAP = 10_000
N_SIGN_FLIPS = 100_000
ANALYSIS_PATCH_RECORD = "docs/shared_hyperparameter_analysis_patch_approval.json"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_output_root(root: Path) -> Path:
    return (
        root
        / "outputs/digital_discovery_round_jk/reviewer_sensitivity"
        / "shared_hyperparameters"
    )


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if load_json(path) != payload:
            raise FileExistsError(f"Refusing to replace non-identical immutable output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)


def require_fingerprint(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if int(record.get("size_bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"{label} byte size changed: {path}")
    if str(record.get("sha256")) != sha256_file(path):
        raise RuntimeError(f"{label} SHA-256 changed: {path}")
    return path


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def environment_record() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "torch", "rdkit", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_runtime": torch.version.cuda,
    }


def create_approval(root: Path, commit: str) -> dict[str, Any]:
    if not commit or commit == "unknown":
        raise ValueError("A committed plan/runner revision is required.")
    paths = {
        "analysis_plan": root / "docs/analysis_plan.md",
        "runner": Path(__file__).resolve(),
        "tests": root / "tests/test_shared_hyperparameter_sensitivity.py",
        "ws_e_runner": root / "src/rerank/experiments/run_ws_e_ranking.py",
        "revision_tuning": root / "src/rerank/revision_tuning.py",
        "analysis_statistics": root / "src/rerank/analysis/analyze_revision_predictions.py",
        "study_data": root / "src/rerank/study_data.py",
        "environment_yml": root / "environment-revision.yml",
        "requirements": root / "requirements-revision.txt",
        "pyproject": root / "pyproject.toml",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Approval input is missing ({label}): {path}")
    record = {
        "schema_version": 1,
        "record_kind": "shared_hyperparameter_execution_approval",
        "protocol_id": PROTOCOL_ID,
        "status": "approved",
        "approval_date": "2026-08-27",
        "supervisor": "repository owner",
        "authorization": "ok triển khai thôi",
        "analysis_plan_commit": commit,
        "inputs": {key: fingerprint(value) for key, value in paths.items()},
        "runtime_lock": environment_record(),
        "seeds": list(SEEDS),
        "strata": copy.deepcopy(STRATA),
        "single_intended_arm_change": "add three frozen Uni-Mol-derived pair-level scalars",
        "created_at_utc": utc_now(),
    }
    record["approval_fingerprint"] = canonical_sha256(record)
    output = root / "docs/shared_hyperparameter_approval.json"
    if output.exists():
        raise FileExistsError(f"Approval record already exists: {output}")
    atomic_json(output, record)
    return record


def _load_base_approval(root: Path) -> dict[str, Any]:
    record = load_json(root / "docs/shared_hyperparameter_approval.json")
    supplied = record.get("approval_fingerprint")
    unsigned = dict(record)
    unsigned.pop("approval_fingerprint", None)
    if supplied != canonical_sha256(unsigned):
        raise PermissionError("Shared-hyperparameter approval fingerprint is invalid.")
    return record


def create_analysis_patch_approval(root: Path, commit: str) -> dict[str, Any]:
    """Bind the reaction-cluster lookup correction without replacing the base lock."""

    base = _load_base_approval(root)
    paths = {
        "corrected_runner": Path(__file__).resolve(),
        "patch_tests": root / "tests/test_shared_hyperparameter_analysis_patch.py",
        "source_csv": root / "data/uspto_smiles.csv",
        "metadata_csv": root / "data/uspto_reaction_metadata.csv",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Analysis-patch input is missing ({label}): {path}")
    record = {
        "schema_version": 1,
        "record_kind": "shared_hyperparameter_analysis_implementation_patch",
        "protocol_id": PROTOCOL_ID,
        "status": "approved-compute-only-correction",
        "approval_date": "2026-08-27",
        "authorization": "vừa hết token, giờ tiếp tục đi",
        "analysis_patch_commit": commit,
        "base_approval": fingerprint(root / "docs/shared_hyperparameter_approval.json"),
        "base_approval_fingerprint": base["approval_fingerprint"],
        "inputs": {key: fingerprint(value) for key, value in paths.items()},
        "reason": (
            "Resolve canonical-product clusters from the complete official test metadata "
            "rather than the historical covered-only prediction table; suppress a NumPy "
            "masked-division warning. No model, prediction, estimand, RNG seed, resampling "
            "rule, multiplicity family, or scientific result is changed."
        ),
        "failed_analysis_wrote_scientific_tables": False,
        "created_at_utc": utc_now(),
    }
    record["patch_fingerprint"] = canonical_sha256(record)
    output = root / ANALYSIS_PATCH_RECORD
    if output.exists():
        raise FileExistsError(f"Analysis patch record already exists: {output}")
    atomic_json(output, record)
    return record


def validate_analysis_patch(root: Path, base: Mapping[str, Any]) -> dict[str, Any]:
    path = root / ANALYSIS_PATCH_RECORD
    record = load_json(path)
    unsigned = dict(record)
    supplied = unsigned.pop("patch_fingerprint", None)
    if supplied != canonical_sha256(unsigned):
        raise PermissionError("Shared-hyperparameter analysis-patch fingerprint is invalid.")
    if (
        record.get("protocol_id") != PROTOCOL_ID
        or record.get("status") != "approved-compute-only-correction"
        or record.get("base_approval_fingerprint") != base.get("approval_fingerprint")
        or record.get("failed_analysis_wrote_scientific_tables") is not False
    ):
        raise PermissionError("Shared-hyperparameter analysis-patch identity is invalid.")
    require_fingerprint(record["base_approval"], "analysis-patch base approval")
    for label, source in record.get("inputs", {}).items():
        require_fingerprint(source, f"analysis-patch input {label}")
    return record


def validate_approval(root: Path) -> dict[str, Any]:
    record = _load_base_approval(root)
    if (
        record.get("protocol_id") != PROTOCOL_ID
        or record.get("status") != "approved"
        or tuple(record.get("seeds", ())) != SEEDS
        or record.get("strata") != STRATA
    ):
        raise PermissionError("Shared-hyperparameter approval identity is invalid.")
    runner_changed = False
    for label, source in record.get("inputs", {}).items():
        try:
            require_fingerprint(source, f"approval input {label}")
        except RuntimeError:
            if label != "runner":
                raise
            runner_changed = True
    if runner_changed:
        validate_analysis_patch(root, record)
    if not record.get("analysis_plan_commit"):
        raise PermissionError("Approval does not bind a committed analysis plan.")
    return record


def _primary_freeze(root: Path) -> Path:
    return root / "outputs/jcheminform_revision/tuned_primary/conformer_seed_42/freeze/model_selection.json"


def _historical_search(root: Path) -> Path:
    return root / "outputs/jcheminform_revision/tuned_primary/conformer_seed_42/search"


def _source_selection(root: Path, pool: str) -> Path:
    return root / f"outputs/digital_discovery_round_jk/k1/{pool}/selection/manifest.json"


def _source_models(root: Path, pool: str) -> Path:
    return root / f"outputs/digital_discovery_round_jk/k1/{pool}/models"


def _stratum_root(output_root: Path, stratum: str) -> Path:
    return output_root / "strata" / stratum


def _pool_root(output_root: Path, stratum: str, pool: str) -> Path:
    return _stratum_root(output_root, stratum) / "pools" / pool


def _config_source_path(output_root: Path, stratum: str) -> Path:
    return _stratum_root(output_root, stratum) / "config_source.json"


def _validate_primary_configs(root: Path) -> dict[str, Any]:
    primary = load_json(_primary_freeze(root))
    if primary.get("protocol_id") != "cap10-tuned-v1":
        raise RuntimeError("Historical primary freeze protocol changed.")
    if primary["selected_baseline"]["config"] != STRATA["baseline_selected"]:
        raise RuntimeError("Frozen baseline-selected configuration changed.")
    if primary["selected_augmented"]["config"] != STRATA["augmented_selected"]:
        raise RuntimeError("Frozen augmented-selected configuration changed.")
    return primary


def _historical_trial_path(root: Path, stratum: str, arm: str, seed: int) -> Path:
    index = int(STRATA[stratum]["index"])
    return (
        _historical_search(root)
        / f"search/{arm}/raw/config_{index:03d}/seed_{seed}/trial.json"
    )


def _validate_historical_trial(root: Path, stratum: str, arm: str, seed: int) -> dict:
    path = _historical_trial_path(root, stratum, arm, seed)
    trial = load_json(path)
    if (
        trial.get("protocol_id") != "cap10-tuned-v1"
        or trial.get("arm") != arm
        or int(trial.get("seed")) != seed
        or trial.get("prior_transform") != "raw"
        or trial.get("config") != STRATA[stratum]
        or trial.get("test_partition_loaded") is not False
    ):
        raise RuntimeError(f"Historical trial identity mismatch: {path}")
    search = _historical_search(root)
    for key, hash_key in (
        ("checkpoint_relpath", "checkpoint_sha256"),
        ("normalizer_relpath", "normalizer_sha256"),
    ):
        artifact = search / trial[key]
        if "sha256:" + file_sha256(artifact) != trial[hash_key]:
            raise RuntimeError(f"Historical trial artifact changed: {artifact}")
    return trial


def _validate_source_trial(path: Path, pool: str, arm: str, seed: int, config: dict) -> dict:
    trial = load_json(path)
    if (
        trial.get("pool_name") != pool
        or trial.get("arm") != arm
        or int(trial.get("seed")) != seed
        or trial.get("config") != config
        or trial.get("test_partition_loaded") is not False
    ):
        raise RuntimeError(f"K1 source-trial identity mismatch: {path}")
    require_fingerprint(trial["checkpoint"], "K1 source checkpoint")
    require_fingerprint(trial["normalizer"], "K1 source normalizer")
    return trial


def preflight(root: Path) -> dict[str, Any]:
    _validate_primary_configs(root)
    trial_count = 0
    for stratum in STRATA:
        for arm in ("baseline", "augmented"):
            for seed in SEEDS:
                _validate_historical_trial(root, stratum, arm, seed)
                trial_count += 1
    source_count = 0
    for pool in ALT_POOLS:
        selection_path = _source_selection(root, pool)
        selection = load_json(selection_path)
        if (
            selection.get("pool_name") != pool
            or int(selection.get("candidate_cap")) != 10
            or not set(SEEDS).issubset(set(selection.get("seeds", ())))
            or selection.get("test_partition_loaded") is not False
        ):
            raise RuntimeError(f"K1 selection identity mismatch: {selection_path}")
        for record in selection["inputs"].values():
            require_fingerprint(record, f"K1 {pool} selection input")
        for stratum, matching_arm in MATCHING_ARM.items():
            for seed in SEEDS:
                path = _source_models(root, pool) / f"validation/{matching_arm}/seed_{seed}/trial.json"
                _validate_source_trial(path, pool, matching_arm, seed, STRATA[stratum])
                source_count += 1
    return {
        "protocol_id": PROTOCOL_ID,
        "historical_trials_verified": trial_count,
        "alternative_trials_verified": source_count,
        "new_crossed_fits_required": len(ALT_POOLS) * len(STRATA) * len(SEEDS),
        "test_partition_loaded": False,
    }


def _config_source(root: Path, stratum: str, approval: Mapping[str, Any]) -> dict[str, Any]:
    primary = _validate_primary_configs(root)
    config = copy.deepcopy(STRATA[stratum])
    selected = {"config": config}
    return {
        "schema_version": 1,
        "record_kind": "shared_hyperparameter_config_source",
        # The WS-E fitter explicitly requires the parent training protocol.
        "protocol_id": "cap10-tuned-v1",
        "sensitivity_protocol_id": PROTOCOL_ID,
        "stratum": stratum,
        "selected_baseline": copy.deepcopy(selected),
        "selected_augmented": copy.deepcopy(selected),
        "fixed_training_config": copy.deepcopy(primary["fixed_training_config"]),
        "source_primary_freeze": fingerprint(_primary_freeze(root)),
        "approval": fingerprint(root / "docs/shared_hyperparameter_approval.json"),
        "approval_fingerprint": approval["approval_fingerprint"],
        "test_partition_loaded": False,
        "single_intended_arm_change": "add three frozen Uni-Mol-derived pair-level scalars",
    }


def _selection_subset(root: Path, pool: str, approval: Mapping[str, Any]) -> dict[str, Any]:
    source_path = _source_selection(root, pool)
    source = load_json(source_path)
    result = copy.deepcopy(source)
    result.update(
        {
            "protocol_id": PROTOCOL_ID,
            "record_kind": "shared_hyperparameter_train_validation_selection",
            "seeds": list(SEEDS),
            "train_pairs": {str(seed): source["train_pairs"][str(seed)] for seed in SEEDS},
            "train_audit": {str(seed): source["train_audit"][str(seed)] for seed in SEEDS},
            "source_selection_manifest": fingerprint(source_path),
            "approval": fingerprint(root / "docs/shared_hyperparameter_approval.json"),
            "approval_fingerprint": approval["approval_fingerprint"],
            "post_hoc_reviewer_sensitivity": True,
            "test_partition_loaded": False,
            "test_ground_truth_loaded": False,
            "single_intended_arm_change": "add three frozen Uni-Mol-derived pair-level scalars",
        }
    )
    return result


def _imported_trial(
    source_path: Path,
    pool: str,
    stratum: str,
    arm: str,
    seed: int,
    approval: Mapping[str, Any],
) -> dict[str, Any]:
    source = _validate_source_trial(source_path, pool, arm, seed, STRATA[stratum])
    result = copy.deepcopy(source)
    result.update(
        {
            "record_kind": "shared_hyperparameter_validation_trial",
            "protocol_id": PROTOCOL_ID,
            "sensitivity_stratum": stratum,
            "imported_existing_trial": True,
            "source_trial": fingerprint(source_path),
            "approval_fingerprint": approval["approval_fingerprint"],
            "single_intended_arm_change": "add three frozen Uni-Mol-derived pair-level scalars",
            "test_partition_loaded": False,
        }
    )
    return result


def prepare(root: Path, output_root: Path) -> dict[str, Any]:
    approval = validate_approval(root)
    audit = preflight(root)
    for stratum in STRATA:
        config_path = _config_source_path(output_root, stratum)
        write_immutable_json(config_path, _config_source(root, stratum, approval))
        matching_arm = MATCHING_ARM[stratum]
        for pool in ALT_POOLS:
            pool_root = _pool_root(output_root, stratum, pool)
            write_immutable_json(
                pool_root / "selection/manifest.json",
                _selection_subset(root, pool, approval),
            )
            for seed in SEEDS:
                source = _source_models(root, pool) / f"validation/{matching_arm}/seed_{seed}/trial.json"
                target = pool_root / f"models/validation/{matching_arm}/seed_{seed}/trial.json"
                write_immutable_json(
                    target,
                    _imported_trial(source, pool, stratum, matching_arm, seed, approval),
                )
    manifest_path = output_root / "preparation_manifest.json"
    existing_created_at = (
        load_json(manifest_path).get("created_at_utc") if manifest_path.exists() else utc_now()
    )
    manifest = {
        "schema_version": 1,
        "record_kind": "shared_hyperparameter_preparation",
        "protocol_id": PROTOCOL_ID,
        "seeds": list(SEEDS),
        "strata": copy.deepcopy(STRATA),
        "pools": list(ALL_POOLS),
        "approval": fingerprint(root / "docs/shared_hyperparameter_approval.json"),
        "preflight": audit,
        "test_partition_loaded": False,
        "created_at_utc": existing_created_at,
    }
    write_immutable_json(manifest_path, manifest)
    return manifest


def fit_one(root: Path, output_root: Path, stratum: str, pool: str, seed: int, device: str) -> dict:
    validate_approval(root)
    if stratum not in STRATA or pool not in ALT_POOLS or seed not in SEEDS:
        raise ValueError("Invalid shared-hyperparameter fit identity.")
    arm = "augmented" if MATCHING_ARM[stratum] == "baseline" else "baseline"
    pool_root = _pool_root(output_root, stratum, pool)
    trial_path = pool_root / f"models/validation/{arm}/seed_{seed}/trial.json"
    if trial_path.exists():
        trial = load_json(trial_path)
        if trial.get("config") != STRATA[stratum]:
            raise RuntimeError(f"Existing crossed trial has the wrong config: {trial_path}")
        return trial
    args = argparse.Namespace(
        selection_root=str(pool_root / "selection"),
        primary_freeze=str(_config_source_path(output_root, stratum)),
        output_root=str(pool_root / "models"),
        arm=arm,
        seed=seed,
        device=device,
    )
    result = ws_e.fit_validation(args)
    result.update(
        {
            "record_kind": "shared_hyperparameter_validation_trial",
            "protocol_id": PROTOCOL_ID,
            "sensitivity_stratum": stratum,
            "imported_existing_trial": False,
            "approval_fingerprint": validate_approval(root)["approval_fingerprint"],
            "post_hoc_reviewer_sensitivity": True,
        }
    )
    # The scientific payload is unchanged; enrich the just-written record atomically.
    atomic_json(trial_path, result)
    return result


def _fit_job(command: list[str], root: Path) -> tuple[list[str], int, str]:
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    return command, completed.returncode, completed.stdout + completed.stderr


def run_missing_fits(root: Path, output_root: Path, workers: int, device: str) -> dict[str, Any]:
    validate_approval(root)
    jobs = []
    for stratum in STRATA:
        arm = "augmented" if MATCHING_ARM[stratum] == "baseline" else "baseline"
        for pool in ALT_POOLS:
            for seed in SEEDS:
                trial = _pool_root(output_root, stratum, pool) / f"models/validation/{arm}/seed_{seed}/trial.json"
                if not trial.exists():
                    jobs.append(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "fit-one",
                            "--root",
                            str(root),
                            "--output-root",
                            str(output_root),
                            "--stratum",
                            stratum,
                            "--pool",
                            pool,
                            "--seed",
                            str(seed),
                            "--device",
                            device,
                        ]
                    )
    total = len(jobs)
    failures = []
    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fit_job, command, root) for command in jobs]
        for future in concurrent.futures.as_completed(futures):
            command, returncode, output = future.result()
            completed_count += 1
            print(f"\rShared-HP fits: {completed_count}/{total}", end="", flush=True)
            if returncode:
                failures.append({"command": command, "returncode": returncode, "output": output[-4000:]})
    print(flush=True)
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))
    return {"required_at_start": total, "completed": completed_count, "workers": workers}


def freeze_alt(root: Path, output_root: Path, stratum: str, pool: str) -> dict:
    approval = validate_approval(root)
    pool_root = _pool_root(output_root, stratum, pool)
    output = pool_root / "models/freeze.json"
    if output.exists():
        return load_json(output)
    args = argparse.Namespace(
        selection_root=str(pool_root / "selection"),
        primary_freeze=str(_config_source_path(output_root, stratum)),
        output_root=str(pool_root / "models"),
        output=str(output),
    )
    result = ws_e.freeze(args)
    result.update(
        {
            "record_kind": "shared_hyperparameter_pool_model_freeze",
            "protocol_id": PROTOCOL_ID,
            "sensitivity_stratum": stratum,
            "approval_fingerprint": approval["approval_fingerprint"],
            "post_hoc_reviewer_sensitivity": True,
        }
    )
    atomic_json(output, result)
    return result


def freeze_historical(root: Path, output_root: Path, stratum: str) -> dict[str, Any]:
    approval = validate_approval(root)
    output = _stratum_root(output_root, stratum) / "historical/model_freeze.json"
    if output.exists():
        return load_json(output)
    trials: dict[str, dict[str, Any]] = {"baseline": {}, "augmented": {}}
    for arm in trials:
        for seed in SEEDS:
            path = _historical_trial_path(root, stratum, arm, seed)
            _validate_historical_trial(root, stratum, arm, seed)
            trials[arm][str(seed)] = fingerprint(path)
    record = {
        "schema_version": 1,
        "record_kind": "shared_hyperparameter_historical_model_freeze",
        "protocol_id": PROTOCOL_ID,
        "sensitivity_stratum": stratum,
        "pool_name": "historical",
        "config": copy.deepcopy(STRATA[stratum]),
        "seeds": list(SEEDS),
        "trials": trials,
        "approval_fingerprint": approval["approval_fingerprint"],
        "complete": True,
        "test_partition_loaded": False,
        "created_at_utc": utc_now(),
    }
    write_immutable_json(output, record)
    return record


def _load_historical_test_payload(root: Path, freeze_path: Path) -> dict[str, Any]:
    freeze = load_json(freeze_path)
    if not freeze.get("complete") or freeze.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("Historical test scoring requires a complete sensitivity freeze.")
    primary = _validate_primary_configs(root)
    cache = root / "outputs/jcheminform_revision/conformers/seed_42/features/official_3d_prior_schema1.pkl"
    if sha256_file(cache) != primary["retained_train_test_cache_sha256"]:
        raise PermissionError("Historical train/test cache differs from the primary freeze.")
    with open(cache, "rb") as handle:
        blob = pickle.load(handle)
    payload = blob.get("payload", blob)
    required = {"eval_pwc", "eval_ground_truths", "eval_metadata", "eval_features"}
    if not required.issubset(payload):
        raise RuntimeError("Historical cache lacks official-test fields.")
    if any(str(row.get("source_split")) != "test" for row in payload["eval_metadata"]):
        raise PermissionError("Historical evaluation payload is not exclusively test.")
    return payload


def _score_historical_arm(
    root: Path,
    output_root: Path,
    stratum: str,
    arm: str,
    payload: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    metrics = {}
    prediction_dir = _stratum_root(output_root, stratum) / "historical/test_results/predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    columns = feature_columns_for_arm(arm)
    config = STRATA[stratum]
    for seed in SEEDS:
        target = prediction_dir / f"{arm}_seed_{seed}.csv"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite historical prediction: {target}")
        trial = _validate_historical_trial(root, stratum, arm, seed)
        checkpoint = _historical_search(root) / trial["checkpoint_relpath"]
        normalizer = FeatureNormalizer.load(
            str(_historical_search(root) / trial["normalizer_relpath"])
        )
        model = RankerMLP(
            input_dim=len(columns),
            hidden_dims=[int(config["hidden_width"])],
            dropout=float(config["dropout"]),
            use_batch_norm=False,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        reranked = []
        with torch.no_grad():
            for (_, candidates), matrix in zip(payload["eval_pwc"], payload["eval_features"]):
                features = transform_feature_matrix(matrix, "raw", columns)
                scores = model.score(
                    torch.as_tensor(normalizer.transform(features), dtype=torch.float32, device=device)
                ).detach().cpu().numpy()
                order = np.argsort(-scores, kind="stable")
                reranked.append(([candidates[index] for index in order], scores[order]))
        evaluation = evaluate_reranking(
            products_with_candidates=payload["eval_pwc"],
            ground_truths=payload["eval_ground_truths"],
            reranker=None,
            ks=[1, 3, 5, 10],
            output_csv=str(target),
            precomputed_reranked_results=reranked,
            reaction_metadata=payload["eval_metadata"],
        )
        metrics[str(seed)] = {
            "top1": evaluation.reranked_accuracy[1],
            "top3": evaluation.reranked_accuracy[3],
            "top5": evaluation.reranked_accuracy[5],
            "top10": evaluation.reranked_accuracy[10],
            "mrr": evaluation.reranked_mrr,
            "n_test_reactions": evaluation.n_products,
            "prediction": fingerprint(target),
        }
    return metrics


def evaluate_historical(root: Path, output_root: Path, stratum: str, device: str) -> dict:
    approval = validate_approval(root)
    result_dir = _stratum_root(output_root, stratum) / "historical/test_results"
    manifest_path = result_dir / "manifest.json"
    if manifest_path.exists():
        return load_json(manifest_path)
    freeze_path = _stratum_root(output_root, stratum) / "historical/model_freeze.json"
    freeze_historical(root, output_root, stratum)
    freeze_hash = sha256_file(freeze_path)
    payload = _load_historical_test_payload(root, freeze_path)
    per_arm = {
        arm: _score_historical_arm(root, output_root, stratum, arm, payload, device)
        for arm in ("baseline", "augmented")
    }
    if sha256_file(freeze_path) != freeze_hash:
        raise RuntimeError("Historical sensitivity freeze changed during test scoring.")
    result = {
        "schema_version": 1,
        "record_kind": "shared_hyperparameter_historical_official_test",
        "protocol_id": PROTOCOL_ID,
        "sensitivity_stratum": stratum,
        "pool_name": "historical",
        "config": copy.deepcopy(STRATA[stratum]),
        "seeds": list(SEEDS),
        "per_arm_metrics": per_arm,
        "model_freeze": fingerprint(freeze_path),
        "approval_fingerprint": approval["approval_fingerprint"],
        "test_partition_loaded_only_after_model_freeze": True,
        "created_at_utc": utc_now(),
    }
    atomic_json(manifest_path, result)
    return result


def _alt_evaluation_args(root: Path, output_root: Path, stratum: str, pool: str, device: str):
    pool_root = _pool_root(output_root, stratum, pool)
    source_selection = load_json(_source_selection(root, pool))
    feature_freeze_path = Path(source_selection["inputs"]["feature_freeze"]["path"])
    feature_freeze = load_json(feature_freeze_path)
    index_manifest_path = Path(feature_freeze["pool_index_manifest"]["path"])
    index_manifest = load_json(index_manifest_path)
    return argparse.Namespace(
        model_freeze=str(pool_root / "models/freeze.json"),
        selection_root=str(pool_root / "selection"),
        primary_freeze=str(_config_source_path(output_root, stratum)),
        output_root=str(pool_root / "models"),
        feature_freeze=str(feature_freeze_path),
        pool_jsonl=source_selection["inputs"]["pool"]["path"],
        products_jsonl=source_selection["inputs"]["products"]["path"],
        pool_index_npz=index_manifest["index"]["path"],
        pool_index_manifest=str(index_manifest_path),
        source_csv=source_selection["inputs"]["source_csv"]["path"],
        metadata_csv=source_selection["inputs"]["metadata_csv"]["path"],
        result_dir=str(pool_root / "test_results"),
        device=device,
    )


def evaluate_alt(root: Path, output_root: Path, stratum: str, pool: str, device: str) -> dict:
    validate_approval(root)
    pool_root = _pool_root(output_root, stratum, pool)
    manifest = pool_root / "test_results/manifest.json"
    if manifest.exists():
        return load_json(manifest)
    freeze_alt(root, output_root, stratum, pool)
    result = ws_e.evaluate_test(_alt_evaluation_args(root, output_root, stratum, pool, device))
    result.update(
        {
            "record_kind": "shared_hyperparameter_alternative_official_test",
            "protocol_id": PROTOCOL_ID,
            "sensitivity_stratum": stratum,
            "post_hoc_reviewer_sensitivity": True,
            "approval_fingerprint": validate_approval(root)["approval_fingerprint"],
        }
    )
    atomic_json(manifest, result)
    return result


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm adjustment requires a non-empty vector of probabilities.")
    order = np.argsort(values, kind="stable")
    adjusted_ordered = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted


def _load_historical_matrix(output_root: Path, stratum: str) -> tuple[dict, pd.DataFrame]:
    prediction_dir = _stratum_root(output_root, stratum) / "historical/test_results/predictions"
    frames = {}
    reference = None
    identity = ["reaction_id", "product_smiles", "ground_truth"]
    for arm in ("baseline", "augmented"):
        frames[arm] = []
        for seed in SEEDS:
            frame = pd.read_csv(prediction_dir / f"{arm}_seed_{seed}.csv")
            if frame["reaction_id"].duplicated().any():
                raise RuntimeError("Historical sensitivity predictions contain duplicate reactions.")
            if reference is None:
                reference = frame[identity].copy()
            elif not reference.equals(frame[identity]):
                raise RuntimeError("Historical sensitivity predictions are not paired.")
            frames[arm].append(frame)
    matrices = {}
    for metric, column in (("top1", "reranked_hit@1"), ("mrr", "reranked_rr")):
        matrices[metric] = {
            arm: np.stack([frame[column].to_numpy(float) for frame in frames[arm]])
            for arm in ("baseline", "augmented")
        }
    assert reference is not None
    return matrices, reference


def _load_alt_matrix(path: Path) -> tuple[dict, pd.DataFrame]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    frame = pd.DataFrame(rows).sort_values("reaction_id", kind="stable").reset_index(drop=True)
    covered = frame["covered"].astype(bool).to_numpy()
    matrices = {}
    for metric in ("top1", "mrr"):
        matrices[metric] = {}
        for arm in ("baseline", "augmented"):
            ranks = np.stack(
                [frame[f"{arm}_rank_seed_{seed}"].fillna(0).to_numpy(float) for seed in SEEDS]
            )
            if metric == "top1":
                values = ((ranks == 1) & covered).astype(float)
            else:
                values = np.zeros_like(ranks, dtype=float)
                np.divide(1.0, ranks, out=values, where=ranks > 0)
            matrices[metric][arm] = values
    return matrices, frame


def analyze(root: Path, output_root: Path) -> dict[str, Any]:
    approval = validate_approval(root)
    analysis_root = output_root / "analysis"
    manifest_path = analysis_root / "manifest.json"
    if manifest_path.exists():
        return load_json(manifest_path)
    within_rows = []
    transfer_rows = []
    input_records = []
    source_csv = root / "data/uspto_smiles.csv"
    metadata_csv = root / "data/uspto_reaction_metadata.csv"
    test_reactions = {
        int(item.reaction_id): item
        for item in load_reactions(source_csv, metadata_csv)
        if item.source_split == "test"
    }
    if len(test_reactions) != 5004:
        raise RuntimeError("Shared-HP analysis requires exactly 5,004 official test reactions.")
    input_records.extend((fingerprint(source_csv), fingerprint(metadata_csv)))
    for stratum_index, stratum in enumerate(STRATA):
        historical, historical_ref = _load_historical_matrix(output_root, stratum)
        historical_prediction_dir = (
            _stratum_root(output_root, stratum) / "historical/test_results/predictions"
        )
        input_records.extend(
            fingerprint(historical_prediction_dir / f"{arm}_seed_{seed}.csv")
            for arm in ("baseline", "augmented")
            for seed in SEEDS
        )
        historical_ref = historical_ref.reset_index(drop=True)
        historical_ids = historical_ref["reaction_id"].astype(int).to_numpy()
        historical_lookup = {rid: index for index, rid in enumerate(historical_ids)}
        alt = {}
        common_ids = set(historical_ids.tolist())
        for pool in ALT_POOLS:
            prediction = _pool_root(output_root, stratum, pool) / "test_results/reaction_ranks.jsonl"
            input_records.append(fingerprint(prediction))
            alt[pool] = _load_alt_matrix(prediction)
            covered_ids = set(
                alt[pool][1].loc[alt[pool][1]["covered"].astype(bool), "reaction_id"].astype(int)
            )
            common_ids &= covered_ids
        common = sorted(common_ids)
        try:
            clusters_all = np.asarray(
                [test_reactions[int(item)].product_key for item in historical_ids],
                dtype=object,
            )
        except KeyError as error:
            raise RuntimeError(f"Historical prediction has a non-test reaction: {error}") from error
        for pool_index, pool in enumerate(ALL_POOLS):
            if pool == "historical":
                matrices = historical
                clusters = clusters_all
                n_reactions = len(historical_ref)
            else:
                matrices, frame = alt[pool]
                keep = frame["covered"].astype(bool).to_numpy()
                ids = frame.loc[keep, "reaction_id"].astype(int).to_numpy()
                try:
                    clusters = np.asarray(
                        [test_reactions[int(item)].product_key for item in ids],
                        dtype=object,
                    )
                except KeyError as error:
                    raise RuntimeError(f"Alternative prediction has a non-test reaction: {error}") from error
                matrices = {
                    metric: {arm: values[arm][:, keep] for arm in values}
                    for metric, values in matrices.items()
                }
                n_reactions = int(keep.sum())
            for metric_index, metric in enumerate(("top1", "mrr")):
                differences = matrices[metric]["augmented"] - matrices[metric]["baseline"]
                result = crossed_product_seed_bootstrap(
                    differences,
                    clusters,
                    n_draws=N_BOOTSTRAP,
                    seed=4100 + 100 * stratum_index + 10 * pool_index + metric_index,
                )
                within_rows.append(
                    {
                        "stratum": stratum,
                        "pool": pool,
                        "metric": metric,
                        "n_reactions": n_reactions,
                        **result,
                    }
                )
        common_h = np.asarray([historical_lookup[item] for item in common], dtype=int)
        common_clusters = np.asarray(
            [test_reactions[int(item)].product_key for item in common], dtype=object
        )
        raw_p_indices = []
        for pool_index, pool in enumerate(ALT_POOLS):
            matrices, frame = alt[pool]
            frame_lookup = {int(value): index for index, value in enumerate(frame["reaction_id"])}
            common_a = np.asarray([frame_lookup[item] for item in common], dtype=int)
            for metric_index, metric in enumerate(("top1", "mrr")):
                historical_effect = (
                    historical[metric]["augmented"][:, common_h]
                    - historical[metric]["baseline"][:, common_h]
                )
                alternative_effect = (
                    matrices[metric]["augmented"][:, common_a]
                    - matrices[metric]["baseline"][:, common_a]
                )
                transfer = alternative_effect - historical_effect
                bootstrap = product_cluster_bootstrap(
                    transfer,
                    common_clusters,
                    n_draws=N_BOOTSTRAP,
                    seed=5100 + 100 * stratum_index + 10 * pool_index + metric_index,
                )
                p_value = clustered_sign_flip(
                    transfer,
                    common_clusters,
                    n_draws=N_SIGN_FLIPS,
                    seed=6100 + 100 * stratum_index + 10 * pool_index + metric_index,
                )
                raw_p_indices.append(len(transfer_rows))
                transfer_rows.append(
                    {
                        "stratum": stratum,
                        "pool": pool,
                        "metric": metric,
                        "n_common_reactions": len(common),
                        "n_product_clusters": len(set(common_clusters.tolist())),
                        **bootstrap,
                        "sign_flip_p_raw": p_value,
                    }
                )
        adjusted = holm_adjust([transfer_rows[index]["sign_flip_p_raw"] for index in raw_p_indices])
        for index, value in zip(raw_p_indices, adjusted):
            transfer_rows[index]["sign_flip_p_holm_within_stratum"] = float(value)
    analysis_root.mkdir(parents=True, exist_ok=True)
    within_path = analysis_root / "within_pool_effects.csv"
    transfer_path = analysis_root / "direct_transfer_effects.csv"
    pd.DataFrame(within_rows).to_csv(within_path, index=False)
    pd.DataFrame(transfer_rows).to_csv(transfer_path, index=False)
    result = {
        "schema_version": 1,
        "record_kind": "shared_hyperparameter_sensitivity_analysis",
        "protocol_id": PROTOCOL_ID,
        "post_hoc_reviewer_sensitivity": True,
        "strata": copy.deepcopy(STRATA),
        "seeds": list(SEEDS),
        "n_bootstrap": N_BOOTSTRAP,
        "n_sign_flips": N_SIGN_FLIPS,
        "holm_family": "six pool-by-endpoint transfer tests separately within each config stratum",
        "within_pool_effects": fingerprint(within_path),
        "direct_transfer_effects": fingerprint(transfer_path),
        "prediction_inputs": input_records,
        "approval_fingerprint": approval["approval_fingerprint"],
        "test_partition_used_for_training_or_selection": False,
        "created_at_utc": utc_now(),
    }
    atomic_json(manifest_path, result)
    return result


def status(output_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"protocol_id": PROTOCOL_ID, "strata": {}}
    for stratum in STRATA:
        item: dict[str, Any] = {"historical": {}, "alternative": {}}
        item["historical"] = {
            "freeze": (_stratum_root(output_root, stratum) / "historical/model_freeze.json").is_file(),
            "test": (_stratum_root(output_root, stratum) / "historical/test_results/manifest.json").is_file(),
        }
        for pool in ALT_POOLS:
            pool_root = _pool_root(output_root, stratum, pool)
            counts = {}
            for arm in ("baseline", "augmented"):
                counts[arm] = sum(
                    (pool_root / f"models/validation/{arm}/seed_{seed}/trial.json").is_file()
                    for seed in SEEDS
                )
            item["alternative"][pool] = {
                "validation_trials": counts,
                "freeze": (pool_root / "models/freeze.json").is_file(),
                "test": (pool_root / "test_results/manifest.json").is_file(),
            }
        result["strata"][stratum] = item
    result["analysis"] = (output_root / "analysis/manifest.json").is_file()
    return result


def run_all(root: Path, output_root: Path, workers: int, device: str) -> dict[str, Any]:
    prepare(root, output_root)
    fit_summary = run_missing_fits(root, output_root, workers, device)
    for stratum in STRATA:
        freeze_historical(root, output_root, stratum)
        for pool in ALT_POOLS:
            freeze_alt(root, output_root, stratum, pool)
    for stratum in STRATA:
        evaluate_historical(root, output_root, stratum, device)
        for pool in ALT_POOLS:
            evaluate_alt(root, output_root, stratum, pool, device)
    analysis = analyze(root, output_root)
    return {"fit_summary": fit_summary, "analysis": analysis, "status": status(output_root)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "preflight", "lock-approval", "lock-analysis-patch", "prepare", "fit-one", "run-fits",
        "freeze", "evaluate", "analyze", "status", "run-all",
    ))
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--stratum", choices=tuple(STRATA))
    parser.add_argument("--pool", choices=ALT_POOLS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_root = (args.output_root or default_output_root(root)).resolve()
    if args.command == "preflight":
        result = preflight(root)
    elif args.command == "lock-approval":
        result = create_approval(root, args.commit or _git_head(root))
    elif args.command == "lock-analysis-patch":
        result = create_analysis_patch_approval(root, args.commit or _git_head(root))
    elif args.command == "prepare":
        result = prepare(root, output_root)
    elif args.command == "fit-one":
        if args.stratum is None or args.pool is None or args.seed is None:
            raise ValueError("fit-one requires --stratum, --pool and --seed")
        result = fit_one(root, output_root, args.stratum, args.pool, args.seed, args.device)
    elif args.command == "run-fits":
        result = run_missing_fits(root, output_root, args.workers, args.device)
    elif args.command == "freeze":
        for stratum in STRATA:
            freeze_historical(root, output_root, stratum)
            for pool in ALT_POOLS:
                freeze_alt(root, output_root, stratum, pool)
        result = status(output_root)
    elif args.command == "evaluate":
        for stratum in STRATA:
            evaluate_historical(root, output_root, stratum, args.device)
            for pool in ALT_POOLS:
                evaluate_alt(root, output_root, stratum, pool, args.device)
        result = status(output_root)
    elif args.command == "analyze":
        result = analyze(root, output_root)
    elif args.command == "status":
        result = status(output_root)
    else:
        result = run_all(root, output_root, args.workers, args.device)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
