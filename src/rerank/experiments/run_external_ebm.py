#!/usr/bin/env python
"""Phased D5 runner for the pinned rxn-ebm FF-EBM comparison."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from rerank.external_ebm import (
    CANDIDATE_WIDTH,
    COMPARATOR,
    PROTOCOL_ID,
    PUBLISHED_EBM_SEEDS,
    PUBLISHED_FF_SETTINGS,
    REACTION_FP_DIM,
    RETRORANKER_COMMIT,
    RXN_EBM_COMMIT,
    SINGLE_INTENDED_CHANGE,
    UPSTREAM_CORE_FILES,
    SparseQueryDataset,
    adapter_audit,
    atomic_json_dump,
    atomic_sparse_save,
    build_fingerprint_matrix,
    canonical_fingerprint,
    evaluation_query_rows,
    file_fingerprint,
    file_sha256,
    immutable_target,
    import_pinned_rxn_ebm,
    model_args,
    prior_baseline_energies,
    ranking_metrics,
    seed_everything,
    training_query_rows,
    verify_pinned_repository,
)
from rerank.revision_tuning import load_selection_bundle


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_record(device: str) -> dict:
    packages = {}
    for name in ("torch", "numpy", "scipy", "rdkit", "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "device": device,
        "packages": packages,
    }
    if device.startswith("cuda"):
        result["cuda"] = {
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
        }
    return result


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return requested


def _unwrap(blob: dict) -> dict:
    payload = blob.get("payload", blob)
    if not isinstance(payload, dict):
        raise ValueError("Compact cache payload is not a mapping.")
    return payload


def run_audit(args) -> None:
    output = immutable_target(args.output, "D5 feasibility audit")
    rxn_repo = verify_pinned_repository(
        args.rxn_ebm_repo, RXN_EBM_COMMIT, UPSTREAM_CORE_FILES
    )
    retro_repo = verify_pinned_repository(
        args.retroranker_repo, RETRORANKER_COMMIT
    )
    bundle = load_selection_bundle(args.selection_bundle)
    train_rows = training_query_rows(bundle["train_products"])
    valid_rows = evaluation_query_rows(bundle["validation_payload"])
    audit = adapter_audit(train_rows, valid_rows)
    result = {
        "schema_version": 1,
        "record_kind": "external_reranker_feasibility_audit",
        "protocol_id": PROTOCOL_ID,
        "comparator": COMPARATOR,
        "single_intended_change": SINGLE_INTENDED_CHANGE,
        "created_at_utc": utc_now(),
        "selection_bundle": file_fingerprint(args.selection_bundle),
        "rxn_ebm": {
            "status": "feasible_requires_training",
            "repository": rxn_repo,
            "selected_public_variant": "FeedforwardEBM",
            "reason": (
                "The published hybrid-all reaction fingerprints use unmapped reactant "
                "and product SMILES, so the frozen cap-10 pool is directly constructible."
            ),
            "checkpoint_compatibility": (
                "No upstream pretrained FF-EBM checkpoint compatible with the current "
                "Chemformer split and AiZynthFinder cap-10 pool is published; train from scratch."
            ),
            "compatibility_shim": (
                "fp_utils imports nmslib but never references it, so an empty import-only "
                "module is supplied; RDKit's renamed useCountSimulation keyword is mapped "
                "to countSimulation without changing its value or fingerprint algorithm"
            ),
            "settings": PUBLISHED_FF_SETTINGS,
            "seeds": list(PUBLISHED_EBM_SEEDS),
            "adapter_audit": audit,
        },
        "retroranker": {
            "status": "documented_fallback_not_run",
            "repository": retro_repo,
            "specific_incompatible_input_requirement": (
                "RetroRanker preprocessing requires atom-mapped reactions and derived "
                "reaction-change graphs. The frozen candidate pool contains unmapped "
                "reactant sets; mapping it would add a new fallible transformation and "
                "its released checkpoints target different USPTO-full/AT/R-SMILES data."
            ),
            "decision": "Do not invoke fallback because pinned rxn-ebm FF-EBM is feasible.",
        },
        "test_partition_loaded": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))


def run_prepare(args) -> None:
    root = Path(args.output_root).resolve()
    manifest_path = immutable_target(root / "prepare_manifest.json", "D5 prepare manifest")
    train_path = immutable_target(root / "train_fingerprints.npz", "D5 train fingerprints")
    valid_path = immutable_target(root / "valid_fingerprints.npz", "D5 valid fingerprints")
    verify_pinned_repository(args.rxn_ebm_repo, RXN_EBM_COMMIT, UPSTREAM_CORE_FILES)
    bundle = load_selection_bundle(args.selection_bundle)
    train_rows = training_query_rows(bundle["train_products"])
    valid_rows = evaluation_query_rows(bundle["validation_payload"])
    started = time.time()
    print(f"Preparing {len(train_rows)} train FF-EBM rows...")
    train = build_fingerprint_matrix(
        train_rows,
        args.rxn_ebm_repo,
        workers=args.workers,
        progress=lambda it, total: tqdm(it, total=total, unit="query", desc="train fp"),
    )
    atomic_sparse_save(train, train_path)
    del train
    print(f"Preparing {len(valid_rows)} official-validation FF-EBM rows...")
    valid = build_fingerprint_matrix(
        valid_rows,
        args.rxn_ebm_repo,
        workers=args.workers,
        progress=lambda it, total: tqdm(it, total=total, unit="query", desc="valid fp"),
    )
    atomic_sparse_save(valid, valid_path)
    del valid
    manifest = {
        "schema_version": 1,
        "record_kind": "external_ebm_train_valid_prepare",
        "protocol_id": PROTOCOL_ID,
        "comparator": COMPARATOR,
        "single_intended_change": SINGLE_INTENDED_CHANGE,
        "created_at_utc": utc_now(),
        "runtime_seconds": time.time() - started,
        "selection_bundle": file_fingerprint(args.selection_bundle),
        "rxn_ebm_repository": verify_pinned_repository(
            args.rxn_ebm_repo, RXN_EBM_COMMIT, UPSTREAM_CORE_FILES
        ),
        "settings": PUBLISHED_FF_SETTINGS,
        "seeds": list(PUBLISHED_EBM_SEEDS),
        "adapter_audit": adapter_audit(train_rows, valid_rows),
        "outputs": {
            "train_fingerprints": file_fingerprint(train_path),
            "valid_fingerprints": file_fingerprint(valid_path),
        },
        "workers": args.workers,
        "test_partition_loaded": False,
    }
    atomic_json_dump(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _load_prepare(root: Path, selection_bundle: str | Path) -> dict:
    with open(root / "prepare_manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Prepared fingerprints have the wrong protocol ID.")
    actual = file_fingerprint(selection_bundle)
    expected = manifest.get("selection_bundle", {})
    if (actual["size_bytes"], actual["sha256"]) != (
        expected.get("size_bytes"),
        expected.get("sha256"),
    ):
        raise PermissionError("Prepared fingerprints belong to another selection bundle.")
    for name in ("train_fingerprints", "valid_fingerprints"):
        path = root / ("train_fingerprints.npz" if name.startswith("train") else "valid_fingerprints.npz")
        record = manifest["outputs"][name]
        current = file_fingerprint(path)
        if (current["size_bytes"], current["sha256"]) != (
            record["size_bytes"], record["sha256"]
        ):
            raise RuntimeError(f"Prepared {name} fingerprint differs from its manifest.")
    return manifest


def _masked_energies(model, batch, mask):
    energies = model(batch)
    return torch.where(mask, energies, torch.full_like(energies, float("inf")))


def _validation_metrics(model, loader, true_indices, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    offset = 0
    with torch.no_grad():
        for batch, mask in loader:
            batch = batch.to(device)
            mask = mask.to(device)
            energies = _masked_energies(model, batch, mask)
            truth = torch.as_tensor(
                true_indices[offset : offset + len(batch)], device=device, dtype=torch.long
            )
            rows = torch.arange(len(batch), device=device)
            total_loss += (
                energies[rows, truth] + torch.logsumexp(-energies, dim=1)
            ).sum().item()
            correct += (torch.argmin(energies, dim=1) == truth).sum().item()
            total += len(batch)
            offset += len(batch)
    return {"loss": total_loss / total, "top1": correct / total}


def run_fit(args) -> None:
    if args.seed not in PUBLISHED_EBM_SEEDS:
        raise ValueError(f"Seed must be one of {PUBLISHED_EBM_SEEDS}.")
    prepared = Path(args.prepared_root).resolve()
    _load_prepare(prepared, args.selection_bundle)
    verify_pinned_repository(args.rxn_ebm_repo, RXN_EBM_COMMIT, UPSTREAM_CORE_FILES)
    result_dir = Path(args.output_root).resolve() / f"seed_{args.seed}"
    trial_path = result_dir / "trial.json"
    if trial_path.exists() or trial_path.with_suffix(".json.tmp").exists():
        raise FileExistsError(f"D5 seed {args.seed} already has a complete or partial trial record.")
    checkpoint_path = result_dir / "best_checkpoint.pt"
    resume_path = result_dir / "resume_state.pt"
    resume_temporary = resume_path.with_suffix(".pt.tmp")
    if resume_temporary.exists():
        raise FileExistsError(
            f"Interrupted D5 resume-state write requires inspection: {resume_temporary}"
        )
    if checkpoint_path.exists() != resume_path.exists():
        raise FileExistsError(
            "D5 resume requires both best_checkpoint.pt and resume_state.pt, or neither."
        )
    result_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_selection_bundle(args.selection_bundle)
    valid_rows = evaluation_query_rows(bundle["validation_payload"])
    valid_true = np.asarray([row.true_index for row in valid_rows], dtype=np.int64)
    train_matrix = sparse.load_npz(prepared / "train_fingerprints.npz").tocsr()
    valid_matrix = sparse.load_npz(prepared / "valid_fingerprints.npz").tocsr()
    if len(valid_rows) != valid_matrix.shape[0]:
        raise ValueError("Validation fingerprint rows are misaligned.")
    device = resolve_device(args.device)
    seed_everything(args.seed)
    _, ff_module, model_utils = import_pinned_rxn_ebm(args.rxn_ebm_repo)
    model = ff_module.FeedforwardEBM(model_args()).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=0
    )
    generator = torch.Generator().manual_seed(args.seed)
    start_epoch = 0
    best_top1 = -float("inf")
    best_epoch = -1
    wait = 0
    history = []
    stopped_reason = "max_epochs"
    previous_runtime = 0.0
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        if state.get("protocol_id") != PROTOCOL_ID or int(state.get("seed")) != args.seed:
            raise RuntimeError("D5 resume state has the wrong protocol or seed.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        generator.set_state(state["generator_state"])
        start_epoch = int(state["next_epoch"])
        best_top1 = float(state["best_top1"])
        best_epoch = int(state["best_epoch"])
        wait = int(state["wait"])
        history = list(state["history"])
        stopped_reason = str(state.get("stopped_reason", "max_epochs"))
        previous_runtime = float(state.get("runtime_seconds", 0.0))
        print(f"RESUMING seed {args.seed} at epoch {start_epoch + 1}/40")
    train_loader = DataLoader(
        SparseQueryDataset(train_matrix),
        batch_size=96,
        shuffle=True,
        generator=generator,
        num_workers=args.loader_workers,
        pin_memory=device.startswith("cuda"),
    )
    valid_loader = DataLoader(
        SparseQueryDataset(valid_matrix),
        batch_size=96,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=device.startswith("cuda"),
    )
    started = time.time()
    for epoch in range(start_epoch, 40):
        model.train()
        train_loss = 0.0
        train_rows = 0
        iterator = tqdm(train_loader, desc=f"seed {args.seed} epoch {epoch + 1}/40", unit="batch")
        for batch, mask in iterator:
            batch = batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            energies = _masked_energies(model, batch, mask)
            loss = (energies[:, 0] + torch.logsumexp(-energies, dim=1)).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_rows += len(batch)
            iterator.set_postfix(loss=f"{train_loss / train_rows:.4f}")
        validation = _validation_metrics(model, valid_loader, valid_true, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / train_rows,
                "validation_loss": validation["loss"],
                "validation_top1": validation["top1"],
                "learning_rate": current_lr,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss / train_rows:.6f} "
            f"valid_loss={validation['loss']:.6f} valid_top1={validation['top1']:.6f}"
        )
        if validation["top1"] > best_top1:
            best_top1 = validation["top1"]
            best_epoch = epoch
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "protocol_id": PROTOCOL_ID,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_top1": best_top1,
                    "model_state_dict": model.state_dict(),
                    "model_settings": PUBLISHED_FF_SETTINGS,
                    "parameter_count": parameter_count,
                },
                temporary,
            )
            os.replace(temporary, checkpoint_path)
        should_stop = False
        if best_top1 - validation["top1"] > 0:
            if wait >= 2:
                stopped_reason = "published_early_stop_patience"
                should_stop = True
            else:
                wait += 1
        else:
            wait = 0
        if not should_stop:
            scheduler.step(validation["top1"])
            if optimizer.param_groups[0]["lr"] < 8e-7:
                stopped_reason = "published_lr_floor"
                should_stop = True
        torch.save(
            {
                "protocol_id": PROTOCOL_ID,
                "seed": args.seed,
                "next_epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "generator_state": generator.get_state(),
                "best_top1": best_top1,
                "best_epoch": best_epoch,
                "wait": wait,
                "history": history,
                "stopped_reason": stopped_reason,
                "runtime_seconds": previous_runtime + time.time() - started,
            },
            resume_temporary,
        )
        os.replace(resume_temporary, resume_path)
        if should_stop:
            break
    trial = {
        "schema_version": 1,
        "record_kind": "external_ebm_validation_trial",
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "comparator": COMPARATOR,
        "single_intended_change": SINGLE_INTENDED_CHANGE,
        "created_at_utc": utc_now(),
        "runtime_seconds": previous_runtime + time.time() - started,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_top1": best_top1,
        "stopped_reason": stopped_reason,
        "settings": PUBLISHED_FF_SETTINGS,
        "parameter_count": parameter_count,
        "selection_bundle": file_fingerprint(args.selection_bundle),
        "prepare_manifest": file_fingerprint(prepared / "prepare_manifest.json"),
        "checkpoint": file_fingerprint(checkpoint_path),
        "resume_state_retained_on_compute_instance": True,
        "resume_state_policy": (
            "retained on the compute instance for crash audit; excluded from the final result archive"
        ),
        "history": history,
        "environment": environment_record(device),
        "upstream_commit": RXN_EBM_COMMIT,
        "test_partition_loaded": False,
    }
    atomic_json_dump(trial, trial_path)
    print(json.dumps(trial, indent=2, sort_keys=True))


def run_freeze(args) -> None:
    output = immutable_target(args.output, "D5 model-selection freeze")
    trials = []
    root = Path(args.fit_root).resolve()
    bundle_fp = file_fingerprint(args.selection_bundle)
    for seed in PUBLISHED_EBM_SEEDS:
        path = root / f"seed_{seed}" / "trial.json"
        with open(path, "r", encoding="utf-8") as handle:
            trial = json.load(handle)
        if trial.get("status") != "complete" or trial.get("seed") != seed:
            raise RuntimeError(f"Incomplete D5 trial for seed {seed}.")
        if trial.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError("D5 trial has the wrong protocol ID.")
        expected = trial.get("selection_bundle", {})
        if (expected.get("size_bytes"), expected.get("sha256")) != (
            bundle_fp["size_bytes"], bundle_fp["sha256"]
        ):
            raise RuntimeError("D5 trial belongs to another selection bundle.")
        checkpoint = Path(trial["checkpoint"]["path"])
        if file_sha256(checkpoint) != trial["checkpoint"]["sha256"]:
            raise RuntimeError(f"D5 checkpoint changed for seed {seed}.")
        trials.append(trial)
    freeze = {
        "schema_version": 1,
        "record_kind": "external_ebm_model_selection_freeze",
        "protocol_id": PROTOCOL_ID,
        "comparator": COMPARATOR,
        "single_intended_change": SINGLE_INTENDED_CHANGE,
        "created_at_utc": utc_now(),
        "selection_bundle": bundle_fp,
        "seeds": list(PUBLISHED_EBM_SEEDS),
        "trials": trials,
        "freeze_fingerprint": canonical_fingerprint(
            [
                {
                    "seed": trial["seed"],
                    "best_epoch": trial["best_epoch"],
                    "checkpoint_sha256": trial["checkpoint"]["sha256"],
                }
                for trial in trials
            ]
        ),
        "test_partition_loaded": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(freeze, output)
    print(json.dumps(freeze, indent=2, sort_keys=True))


def _score_model(model, loader, device) -> np.ndarray:
    parts = []
    model.eval()
    with torch.no_grad():
        for batch, mask in tqdm(loader, desc="scoring test", unit="batch"):
            batch = batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            parts.append(_masked_energies(model, batch, mask).cpu().numpy())
    return np.concatenate(parts, axis=0)


def run_evaluate(args) -> None:
    result_dir = Path(args.result_dir).resolve()
    manifest_path = immutable_target(result_dir / "manifest.json", "D5 test manifest")
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"D5 result directory is not empty: {result_dir}")
    with open(args.selection_freeze, "r", encoding="utf-8") as handle:
        freeze = json.load(handle)
    if freeze.get("protocol_id") != PROTOCOL_ID or freeze.get("record_kind") != "external_ebm_model_selection_freeze":
        raise PermissionError("Official test requires the immutable D5 selection freeze.")
    with open(args.train_test_cache, "rb") as handle:
        payload = _unwrap(pickle.load(handle))
    required = {"eval_pwc", "eval_ground_truths", "eval_metadata"}
    if not required.issubset(payload):
        raise ValueError("Train/test compact cache lacks its official-test payload.")
    if any(str(row.get("source_split")) != "test" for row in payload["eval_metadata"]):
        raise PermissionError("D5 post-freeze payload contains a non-test reaction.")
    rows = evaluation_query_rows(payload)
    result_dir.mkdir(parents=True, exist_ok=False)
    test_fp_path = result_dir / "test_fingerprints.npz"
    started = time.time()
    matrix = build_fingerprint_matrix(
        rows,
        args.rxn_ebm_repo,
        workers=args.workers,
        progress=lambda it, total: tqdm(it, total=total, unit="query", desc="test fp"),
    )
    atomic_sparse_save(matrix, test_fp_path)
    loader = DataLoader(
        SparseQueryDataset(matrix),
        batch_size=96,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=resolve_device(args.device).startswith("cuda"),
    )
    true_indices = np.asarray([row.true_index for row in rows], dtype=np.int64)
    counts = np.asarray([len(row.candidate_smiles) for row in rows], dtype=np.int64)
    baseline = ranking_metrics(prior_baseline_energies(rows), true_indices, counts)
    device = resolve_device(args.device)
    _, ff_module, _ = import_pinned_rxn_ebm(args.rxn_ebm_repo)
    per_seed = {}
    prediction_files = {}
    for trial in freeze["trials"]:
        seed = int(trial["seed"])
        model = ff_module.FeedforwardEBM(model_args()).to(device)
        checkpoint = torch.load(trial["checkpoint"]["path"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        energies = _score_model(model, loader, device)
        metrics = ranking_metrics(energies, true_indices, counts)
        prediction_path = result_dir / f"predictions_seed_{seed}.csv"
        with open(prediction_path, "x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "reaction_id",
                    "seed",
                    "candidate_count",
                    "prior_true_rank",
                    "external_true_rank",
                    "external_top1_candidate",
                ),
            )
            writer.writeheader()
            for index, row in enumerate(rows):
                top_index = int(np.argmin(energies[index, : counts[index]]))
                writer.writerow(
                    {
                        "reaction_id": row.reaction_id,
                        "seed": seed,
                        "candidate_count": counts[index],
                        "prior_true_rank": int(baseline["true_ranks"][index]),
                        "external_true_rank": int(metrics["true_ranks"][index]),
                        "external_top1_candidate": row.candidate_smiles[top_index],
                    }
                )
        per_seed[str(seed)] = {
            key: metrics[key] for key in ("top1", "top3", "top5", "mrr")
        }
        prediction_files[str(seed)] = file_fingerprint(prediction_path)
    mean_metrics = {
        key: float(np.mean([per_seed[str(seed)][key] for seed in PUBLISHED_EBM_SEEDS]))
        for key in ("top1", "top3", "top5", "mrr")
    }
    manifest = {
        "schema_version": 1,
        "record_kind": "external_ebm_official_test",
        "protocol_id": PROTOCOL_ID,
        "comparator": COMPARATOR,
        "single_intended_change": SINGLE_INTENDED_CHANGE,
        "created_at_utc": utc_now(),
        "runtime_seconds": time.time() - started,
        "selection_freeze": file_fingerprint(args.selection_freeze),
        "train_test_cache": file_fingerprint(args.train_test_cache),
        "test_fingerprints": file_fingerprint(test_fp_path),
        "test_reactions": len(rows),
        "baseline_prior_metrics": {
            key: baseline[key] for key in ("top1", "top3", "top5", "mrr")
        },
        "external_per_seed_metrics": per_seed,
        "external_mean_metrics": mean_metrics,
        "prediction_files": prediction_files,
        "environment": environment_record(device),
        "test_partition_loaded_only_after_freeze": True,
    }
    atomic_json_dump(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--selection-bundle", required=True)
    audit.add_argument("--rxn-ebm-repo", required=True)
    audit.add_argument("--retroranker-repo", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=run_audit)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--selection-bundle", required=True)
    prepare.add_argument("--rxn-ebm-repo", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--workers", type=int, default=1)
    prepare.set_defaults(func=run_prepare)

    fit = sub.add_parser("fit")
    fit.add_argument("--selection-bundle", required=True)
    fit.add_argument("--prepared-root", required=True)
    fit.add_argument("--rxn-ebm-repo", required=True)
    fit.add_argument("--output-root", required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    fit.add_argument("--loader-workers", type=int, default=0)
    fit.set_defaults(func=run_fit)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--selection-bundle", required=True)
    freeze.add_argument("--fit-root", required=True)
    freeze.add_argument("--output", required=True)
    freeze.set_defaults(func=run_freeze)

    evaluate = sub.add_parser("evaluate-test")
    evaluate.add_argument("--selection-freeze", required=True)
    evaluate.add_argument("--train-test-cache", required=True)
    evaluate.add_argument("--rxn-ebm-repo", required=True)
    evaluate.add_argument("--result-dir", required=True)
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--loader-workers", type=int, default=0)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate.set_defaults(func=run_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
