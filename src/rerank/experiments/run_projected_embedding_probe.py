"""Phased runner for the prespecified C2 projected pooled-embedding probe."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import random
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from tqdm import tqdm

from rerank.cached_encoder import canon
from rerank.evaluate import _is_match, evaluate_reranking
from rerank.projected_probe import (
    BASE_DIM, CONTROL_ID, POOLED_DIM, PROTOCOL_ID, RAW_INPUT_DIM, SCHEMA_VERSION,
    ProjectionNormalizer, ProjectedRanker, canonical_fingerprint,
    expected_parameter_count, file_sha256, fit_pair_normalizer, make_pair_indices,
    materialize_rows, materialize_validation, selected_base_features,
    validate_projected_selection,
)
from rerank.revision_tuning import MAX_EPOCHS, MIN_IMPROVEMENT, PATIENCE, SEEDS


COMPARATOR = "frozen cap10-tuned-v1 validation-tuned prior+2D baseline"
INTENDED_CHANGE = "replace three Uni-Mol pair scalars with learned 16D product and 16D reactant projections"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: Mapping, path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {target.resolve()}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, target)


def atomic_pickle(payload: Mapping, path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {target.resolve()}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, target)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def file_record(path: str | Path, *, hash_content: bool = True) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    result = {"path": str(resolved), "size_bytes": stat.st_size}
    if hash_content:
        result["sha256"] = file_sha256(resolved)
    return result


def validate_primary_freeze(path: str | Path) -> dict:
    freeze = load_json(path)
    selected = freeze.get("selected_augmented", {}).get("config", {})
    expected = {
        "index": 70, "hidden_width": 128, "dropout": 0.1,
        "learning_rate": 0.001, "margin": 0.1,
    }
    if freeze.get("protocol_id") != "cap10-tuned-v1" or freeze.get("selected_prior_transform") != "raw":
        raise PermissionError("C2 requires the frozen cap10-tuned-v1 raw-prior selection.")
    if selected != expected or tuple(freeze.get("seeds", ())) != SEEDS:
        raise PermissionError("C2 requires the exact frozen augmented D1 configuration and seeds.")
    if freeze.get("test_partition_loaded") is not False:
        raise PermissionError("Primary selection freeze does not attest test isolation.")
    return freeze


class PoolReader:
    """Mean-pool cached atom rows while retaining bounded fragment reuse."""

    def __init__(self, sqlite_path: str | Path):
        self.path = Path(sqlite_path).resolve()
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        metadata = {
            key: json.loads(value)
            for key, value in self.connection.execute("SELECT key,value FROM metadata")
        }
        if (
            metadata.get("cache_kind") != "unimol_atom_mean_with_atom_count"
            or metadata.get("schema_version") != 1
            or metadata.get("conformer_seed") != 42
            or metadata.get("complete") != 1
            or metadata.get("scientific_complete") is not True
        ):
            raise PermissionError(
                "C2 requires a complete seed-42 pooled cache with explicit conformer provenance."
            )
        backend = metadata.get("backend", {})
        checkpoint = backend.get("checkpoint", {})
        if checkpoint.get("sha256") != "da27196af09a8c6d089e10b7764b6a716bcc33da227fc118f5b45b0e484585e9":
            raise PermissionError("C2 pooled cache uses the wrong Uni-Mol checkpoint.")
        self.metadata = metadata
        self.fragment_cache: dict[str, tuple[np.ndarray, int] | None] = {}
        self.molecules = 0
        self.molecules_all_missing = 0

    def pool(self, smiles: str) -> np.ndarray:
        total = np.zeros(POOLED_DIM, dtype=np.float64)
        atoms = 0
        self.molecules += 1
        for fragment in [value.strip() for value in str(smiles).split(".") if value.strip()]:
            key = canon(fragment)
            if key is None:
                continue
            cached = self.fragment_cache.get(key, "not-cached")
            if isinstance(cached, str):
                row = self.connection.execute(
                    "SELECT atom_count,data FROM embeddings WHERE smiles=?", (key,)
                ).fetchone()
                if row is None or row[1] is None:
                    self.fragment_cache[key] = None
                    cached = None
                else:
                    count = int(row[0])
                    mean = np.frombuffer(row[1], dtype=np.float32)
                    if count < 1 or mean.shape != (POOLED_DIM,) or not np.isfinite(mean).all():
                        raise RuntimeError(f"Invalid pooled embedding row for {key}.")
                    cached = (mean.astype(np.float64) * count, count)
                    self.fragment_cache[key] = cached
            if cached is not None:
                subtotal, count = cached
                total += subtotal
                atoms += count
        if atoms == 0:
            self.molecules_all_missing += 1
            return np.zeros(POOLED_DIM, dtype=np.float32)
        return (total / atoms).astype(np.float32)

    def coverage(self) -> dict:
        hits = sum(value is not None for value in self.fragment_cache.values())
        misses = len(self.fragment_cache) - hits
        return {
            "molecules_pooled": self.molecules,
            "molecules_all_fragments_missing": self.molecules_all_missing,
            "unique_fragments_hit": hits,
            "unique_fragments_missing": misses,
        }


def project_train_item(item: Mapping, pooler: PoolReader) -> dict:
    candidates = list(item["candidates"])
    return {
        "product_key": item["product_key"],
        "product_smiles": item["product_smiles"],
        "candidates": candidates,
        "positive_indices": list(item["positive_indices"]),
        "negative_indices": list(item["negative_indices"]),
        "base_features": selected_base_features(item["features"]),
        "product_embedding": pooler.pool(item["product_smiles"]),
        "reactant_embeddings": np.stack(
            [pooler.pool(candidate["smiles"]) for candidate in candidates]
        ).astype(np.float32),
    }


def project_validation_payload(validation: Mapping, pooler: PoolReader, limit: int | None) -> dict:
    count = len(validation["eval_pwc"]) if limit is None else min(limit, len(validation["eval_pwc"]))
    pwc = list(validation["eval_pwc"][:count])
    base, products, reactants = [], [], []
    for index, (product, candidates) in enumerate(tqdm(pwc, desc="C2 validation pooling", unit="rxn")):
        base.append(selected_base_features(validation["eval_features"][index]))
        products.append(pooler.pool(product))
        reactants.append(np.stack([pooler.pool(candidate["smiles"]) for candidate in candidates]))
    return {
        "eval_pwc": pwc,
        "eval_ground_truths": list(validation["eval_ground_truths"][:count]),
        "eval_metadata": list(validation["eval_metadata"][:count]),
        "base_features": base,
        "product_embeddings": products,
        "reactant_embeddings": reactants,
    }


def run_prepare(args) -> None:
    started = time.perf_counter()
    freeze = validate_primary_freeze(args.primary_freeze)
    primary = load_pickle(args.primary_selection_bundle)
    if primary.get("protocol_id") != "cap10-tuned-v1" or "validation_payload" not in primary:
        raise ValueError("C2 preparation requires the restricted primary train/validation bundle.")
    forbidden = {"eval_pwc", "eval_features", "eval_ground_truths", "eval_metadata"}.intersection(primary)
    if forbidden:
        raise PermissionError(f"Primary selection bundle leaks top-level test-like fields: {sorted(forbidden)}")
    expected_sha = freeze.get("selected_augmented", {}).get("trials", {}).get("42", {}).get("selection_bundle_sha256")
    actual_sha = "sha256:" + file_sha256(args.primary_selection_bundle)
    if expected_sha != actual_sha:
        raise PermissionError("Primary selection bundle differs from its frozen D1 input.")

    pooler = PoolReader(args.sqlite_cache)
    train_limit = len(primary["train_products"]) if args.max_train_products is None else min(
        args.max_train_products, len(primary["train_products"])
    )
    train = [
        project_train_item(item, pooler)
        for item in tqdm(primary["train_products"][:train_limit], desc="C2 train pooling", unit="product")
    ]
    validation = project_validation_payload(primary["validation_payload"], pooler, args.max_validation_products)
    complete = train_limit == len(primary["train_products"]) and len(validation["eval_pwc"]) == len(primary["validation_payload"]["eval_pwc"])
    sqlite_record = file_record(args.sqlite_cache, hash_content=args.hash_sqlite)
    repair = Path(str(args.sqlite_cache) + ".repair.sqlite")
    record = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "control_id": CONTROL_ID,
        "comparator": COMPARATOR,
        "single_intended_change": INTENDED_CHANGE,
        "probe_label": "information-bottleneck upper-bound; not parameter matched",
        "status": "complete" if complete else "partial_benchmark",
        "test_partition_loaded": False,
        "base_columns": [0, 1, 5, 6],
        "raw_input_dim": RAW_INPUT_DIM,
        "projection_dims": {"product": 16, "aggregate_reactant": 16},
        "head_input_dim": 36,
        "train_products": train,
        "validation_payload": validation,
        "input_fingerprints": {
            "primary_selection_bundle": file_record(args.primary_selection_bundle),
            "primary_selection_freeze": file_record(args.primary_freeze),
            "sqlite_atom_embeddings": sqlite_record,
            "sqlite_repair": file_record(repair, hash_content=args.hash_sqlite) if repair.exists() else None,
        },
        "coverage": pooler.coverage(),
        "runtime_seconds": time.perf_counter() - started,
        "created_at_utc": utc_now(),
    }
    atomic_pickle(record, args.output)
    summary = {
        "status": record["status"], "train_products": len(train),
        "validation_reactions": len(validation["eval_pwc"]),
        "coverage": record["coverage"], "runtime_seconds": record["runtime_seconds"],
        "output": str(Path(args.output).resolve()), "output_sha256": file_sha256(args.output),
    }
    print(json.dumps(summary, indent=2))


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return requested


def validation_masks(validation: Mapping) -> list[np.ndarray]:
    masks = []
    for (_, candidates), ground_truth in zip(validation["eval_pwc"], validation["eval_ground_truths"]):
        masks.append(np.asarray([_is_match(candidate["smiles"], ground_truth) for candidate in candidates], dtype=bool))
    return masks


def validation_mrr(model, values: np.ndarray, offsets: Sequence[int], masks: Sequence[np.ndarray], device: str) -> float:
    import torch
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 1024):
            batch = torch.from_numpy(values[start : start + 1024]).to(device)
            scores.append(model.score(batch).cpu().numpy())
    merged = np.concatenate(scores)
    reciprocal = []
    for index, mask in enumerate(masks):
        start, stop = offsets[index : index + 2]
        order = np.argsort(-merged[start:stop], kind="stable")
        positions = np.flatnonzero(mask[order])
        reciprocal.append(0.0 if len(positions) == 0 else 1.0 / (positions[0] + 1))
    return float(np.mean(reciprocal))


def train_one(bundle: Mapping, primary_freeze: Mapping, seed: int, device: str, trial_dir: Path) -> dict:
    import torch
    from rerank.loss import PairwiseRankingLoss

    if trial_dir.exists():
        raise FileExistsError(f"Refusing to overwrite C2 trial directory: {trial_dir.resolve()}")
    trial_dir.mkdir(parents=True)
    config = primary_freeze["selected_augmented"]["config"]
    product_indices, positive_indices, negative_indices = make_pair_indices(bundle["train_products"], seed)
    positive = materialize_rows(bundle["train_products"], product_indices, positive_indices)
    negative = materialize_rows(bundle["train_products"], product_indices, negative_indices)
    normalizer = fit_pair_normalizer(positive, negative)
    normalizer_path = trial_dir / "normalizer.npz"
    normalizer.save(normalizer_path)
    for start in range(0, len(positive), 4096):
        positive[start : start + 4096] = normalizer.transform(positive[start : start + 4096])
        negative[start : start + 4096] = normalizer.transform(negative[start : start + 4096])
    pos_tensor = torch.from_numpy(positive)
    neg_tensor = torch.from_numpy(negative)
    validation_values, offsets = materialize_validation(bundle["validation_payload"])
    validation_values = normalizer.transform(validation_values)
    masks = validation_masks(bundle["validation_payload"])

    seed_everything(seed)
    model = ProjectedRanker.build(int(config["hidden_width"]), float(config["dropout"])).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != expected_parameter_count(int(config["hidden_width"])):
        raise AssertionError("C2 model parameter count changed.")
    criterion = PairwiseRankingLoss(margin=float(config["margin"]), reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(MAX_EPOCHS - 5, 1), eta_min=float(config["learning_rate"]) * 0.01
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    checkpoint_path = trial_dir / "best_checkpoint.pt"
    best_mrr, best_epoch, stale = float("-inf"), 0, 0
    history = []
    started = time.perf_counter()
    for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc=f"C2 seed {seed}", unit="epoch"):
        model.train()
        order = torch.randperm(len(pos_tensor), generator=generator)
        loss_sum = 0.0
        batches = 0
        for start in range(0, len(order), 256):
            indices = order[start : start + 256]
            x_pos = pos_tensor[indices].to(device)
            x_neg = neg_tensor[indices].to(device)
            optimizer.zero_grad()
            loss = criterion(model.score(x_pos), model.score(x_neg))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
        mrr = validation_mrr(model, validation_values, offsets, masks, device)
        improved = best_epoch == 0 or mrr > best_mrr + MIN_IMPROVEMENT
        history.append({
            "epoch": epoch, "train_loss": loss_sum / max(batches, 1),
            "validation_mrr": mrr, "selected_improvement": improved,
        })
        if improved:
            best_mrr, best_epoch, stale = mrr, epoch, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
        if epoch > 5:
            scheduler.step()
        if stale >= PATIENCE:
            break
    trial = {
        "trial_schema": 1, "status": "completed", "protocol_id": PROTOCOL_ID,
        "control_id": CONTROL_ID, "comparator": COMPARATOR,
        "single_intended_change": INTENDED_CHANGE, "probe_label": "upper-bound; not parameter matched",
        "seed": seed, "config": config, "best_epoch": best_epoch,
        "best_validation_mrr": best_mrr, "epochs_completed": len(history),
        "early_stopped": len(history) < MAX_EPOCHS, "history": history,
        "n_train_pairs": len(pos_tensor), "n_validation_reactions": len(masks),
        "n_validation_candidate_rows": len(validation_values),
        "model_parameters": expected_parameter_count(int(config["hidden_width"])),
        "checkpoint_relpath": str(checkpoint_path.relative_to(trial_dir.parent.parent)).replace("\\", "/"),
        "checkpoint_sha256": "sha256:" + file_sha256(checkpoint_path),
        "normalizer_relpath": str(normalizer_path.relative_to(trial_dir.parent.parent)).replace("\\", "/"),
        "normalizer_sha256": "sha256:" + file_sha256(normalizer_path),
        "selection_cache_sha256": "sha256:" + file_sha256(bundle["_selection_path"]),
        "test_partition_loaded": False, "runtime_seconds": time.perf_counter() - started,
        "created_at_utc": utc_now(),
    }
    atomic_json(trial, trial_dir / "trial.json")
    return trial


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in SEEDS for seed in seeds):
        raise ValueError("C2 seeds must be a unique subset of 42,43,44,45,46.")
    return seeds


def run_fit(args) -> None:
    bundle = load_pickle(args.selection_cache)
    validate_projected_selection(bundle)
    bundle["_selection_path"] = str(Path(args.selection_cache).resolve())
    freeze = validate_primary_freeze(args.primary_freeze)
    device = resolve_device(args.device)
    root = Path(args.output_root).resolve()
    for seed in parse_seeds(args.seeds):
        trial = train_one(bundle, freeze, seed, device, root / "validation" / f"seed_{seed}")
        print(f"C2 seed {seed}: best validation MRR={trial['best_validation_mrr']:.6f} epoch={trial['best_epoch']}")


def validate_trial(root: Path, seed: int, selection_sha: str) -> dict:
    trial_path = root / "validation" / f"seed_{seed}" / "trial.json"
    trial = load_json(trial_path)
    if trial.get("protocol_id") != PROTOCOL_ID or trial.get("seed") != seed or trial.get("test_partition_loaded") is not False:
        raise PermissionError(f"Invalid C2 validation trial for seed {seed}.")
    if trial.get("selection_cache_sha256") != selection_sha:
        raise PermissionError(f"C2 seed {seed} belongs to a different selection cache.")
    for key, rel_key, sha_key in (
        ("checkpoint", "checkpoint_relpath", "checkpoint_sha256"),
        ("normalizer", "normalizer_relpath", "normalizer_sha256"),
    ):
        path = root / trial[rel_key]
        if "sha256:" + file_sha256(path) != trial[sha_key]:
            raise RuntimeError(f"C2 {key} changed for seed {seed}.")
    return trial


def run_freeze(args) -> None:
    bundle = load_pickle(args.selection_cache)
    validate_projected_selection(bundle)
    primary = validate_primary_freeze(args.primary_freeze)
    selection_sha = "sha256:" + file_sha256(args.selection_cache)
    root = Path(args.output_root).resolve()
    trials = {str(seed): validate_trial(root, seed, selection_sha) for seed in SEEDS}
    curves_path = Path(args.output).with_name("train_validation_curves.csv")
    if curves_path.exists():
        raise FileExistsError(f"Refusing to overwrite C2 curves: {curves_path.resolve()}")
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    with open(curves_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "epoch", "train_loss", "validation_mrr", "selected_improvement"])
        writer.writeheader()
        for seed in SEEDS:
            for row in trials[str(seed)]["history"]:
                writer.writerow({"seed": seed, **row})
    retained_sha = primary.get("retained_train_test_cache_sha256")
    record = {
        "record_kind": "c2_validation_selection_freeze", "protocol_id": PROTOCOL_ID,
        "control_id": CONTROL_ID, "comparator": COMPARATOR,
        "single_intended_change": INTENDED_CHANGE, "probe_label": "upper-bound; not parameter matched",
        "seeds": list(SEEDS), "selected_config": primary["selected_augmented"]["config"],
        "selection_cache_sha256": selection_sha,
        "selection_cache": file_record(args.selection_cache),
        "primary_selection_freeze": file_record(args.primary_freeze),
        "retained_train_test_cache_sha256": retained_sha,
        "trials": trials, "train_validation_curves": file_record(curves_path),
        "test_partition_loaded": False, "created_at_utc": utc_now(),
    }
    record["freeze_fingerprint"] = canonical_fingerprint(record)
    atomic_json(record, args.output)
    print(json.dumps({"status": "frozen", "freeze": str(Path(args.output).resolve()), "mean_validation_mrr": float(np.mean([trials[str(seed)]["best_validation_mrr"] for seed in SEEDS]))}, indent=2))


def validate_c2_freeze(path: str | Path) -> dict:
    record = load_json(path)
    supplied = record.get("freeze_fingerprint")
    unsigned = dict(record)
    unsigned.pop("freeze_fingerprint", None)
    if record.get("record_kind") != "c2_validation_selection_freeze" or supplied != canonical_fingerprint(unsigned):
        raise PermissionError("C2 test evaluation requires an authentic immutable freeze.")
    if tuple(record.get("seeds", ())) != SEEDS or record.get("test_partition_loaded") is not False:
        raise PermissionError("C2 freeze has the wrong seeds or test gate.")
    return record


def build_test_projection(payload: Mapping, pooler: PoolReader) -> dict:
    result = {
        "eval_pwc": payload["eval_pwc"], "eval_ground_truths": payload["eval_ground_truths"],
        "eval_metadata": payload["eval_metadata"], "base_features": [],
        "product_embeddings": [], "reactant_embeddings": [],
    }
    for index, (product, candidates) in enumerate(tqdm(payload["eval_pwc"], desc="C2 test pooling", unit="rxn")):
        result["base_features"].append(selected_base_features(payload["eval_features"][index]))
        result["product_embeddings"].append(pooler.pool(product))
        result["reactant_embeddings"].append(np.stack([pooler.pool(candidate["smiles"]) for candidate in candidates]))
    return result


def environment_record(device: str) -> dict:
    import torch
    return {
        "python": sys.version, "platform": platform.platform(), "device": device,
        "torch": torch.__version__, "numpy": np.__version__,
    }


PREDICTION_METRIC_COLUMNS = (
    "baseline_hit@1", "reranked_hit@1", "baseline_hit@3", "reranked_hit@3",
    "baseline_hit@5", "reranked_hit@5", "baseline_hit@10", "reranked_hit@10",
    "baseline_rr", "reranked_rr",
)


def read_prediction_metrics(path: str | Path) -> tuple[dict, tuple[tuple[str, str], ...]]:
    """Read one immutable C2 prediction CSV without loading or scoring test data."""

    source = Path(path)
    with open(source, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"reaction_id", "source_split", *PREDICTION_METRIC_COLUMNS}
        available = set(reader.fieldnames or ())
        if missing := required.difference(available):
            raise ValueError(f"C2 prediction CSV is missing columns {sorted(missing)}: {source}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"C2 prediction CSV is empty: {source}")
    identities = tuple((str(row["reaction_id"]), str(row["source_split"])) for row in rows)
    if len(set(identities)) != len(identities) or any(split != "test" for _, split in identities):
        raise PermissionError(f"C2 prediction CSV lacks unique official-test identities: {source}")
    values = {column: np.asarray([float(row[column]) for row in rows], dtype=np.float64)
              for column in PREDICTION_METRIC_COLUMNS}
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError(f"C2 prediction CSV has non-finite metric values: {source}")
    return {
        "top1": float(values["reranked_hit@1"].mean()),
        "top3": float(values["reranked_hit@3"].mean()),
        "top5": float(values["reranked_hit@5"].mean()),
        "top10": float(values["reranked_hit@10"].mean()),
        "mrr": float(values["reranked_rr"].mean()),
        "baseline_top1": float(values["baseline_hit@1"].mean()),
        "baseline_mrr": float(values["baseline_rr"].mean()),
        "n_test_reactions": len(rows),
    }, identities


def collect_existing_prediction_metrics(prediction_dir: str | Path) -> tuple[dict, dict]:
    """Validate five paired CSVs and recover their metrics without rescoring."""

    directory = Path(prediction_dir)
    metrics: dict[str, dict] = {}
    reference_identities: tuple[tuple[str, str], ...] | None = None
    baseline: tuple[float, float] | None = None
    for seed in SEEDS:
        source = directory / f"projected_seed_{seed}.csv"
        if not source.is_file():
            raise FileNotFoundError(f"Missing immutable C2 prediction CSV for seed {seed}: {source}")
        seed_metrics, identities = read_prediction_metrics(source)
        if reference_identities is None:
            reference_identities = identities
            baseline = (seed_metrics["baseline_top1"], seed_metrics["baseline_mrr"])
        elif identities != reference_identities:
            raise PermissionError(f"C2 prediction identities/order differ for seed {seed}.")
        elif baseline != (seed_metrics["baseline_top1"], seed_metrics["baseline_mrr"]):
            raise PermissionError(f"C2 baseline metrics differ across paired seed CSVs ({seed}).")
        metrics[str(seed)] = seed_metrics
    records = {str(seed): file_record(directory / f"projected_seed_{seed}.csv") for seed in SEEDS}
    return metrics, records


def c2_test_manifest(
    *, freeze: Mapping, primary_train_test_cache: str | Path, sqlite_cache: str | Path,
    metrics: Mapping[str, Mapping], environment: Mapping, recovery: Mapping | None = None,
) -> dict:
    repair = Path(str(sqlite_cache) + ".repair.sqlite")
    manifest = {
        "manifest_kind": "c2_post_selection_test_evaluation", "protocol_id": PROTOCOL_ID,
        "control_id": CONTROL_ID, "comparator": COMPARATOR,
        "single_intended_change": INTENDED_CHANGE, "probe_label": "upper-bound; not parameter matched",
        "freeze": file_record(freeze["_source_path"]), "freeze_fingerprint": freeze["freeze_fingerprint"],
        "primary_train_test_cache": file_record(primary_train_test_cache),
        "sqlite_atom_embeddings": file_record(sqlite_cache, hash_content=False),
        "sqlite_repair": file_record(repair, hash_content=False) if repair.exists() else None,
        "per_seed_metrics": metrics,
        "descriptive_summary": {metric: {
            "mean": float(np.mean([metrics[str(seed)][metric] for seed in SEEDS])),
            "sample_std": float(np.std([metrics[str(seed)][metric] for seed in SEEDS], ddof=1)),
        } for metric in ("top1", "top3", "top5", "top10", "mrr")},
        "test_partition_loaded_only_after_c2_freeze": True,
        "environment": dict(environment), "created_at_utc": utc_now(),
    }
    if recovery is None:
        manifest["test_pooling_coverage"] = None
    else:
        manifest["prediction_recovery"] = dict(recovery)
        manifest["test_pooling_coverage"] = None
    return manifest


def run_evaluate(args) -> None:
    import torch
    freeze = validate_c2_freeze(args.freeze)
    freeze["_source_path"] = str(Path(args.freeze).resolve())
    expected = str(freeze["retained_train_test_cache_sha256"]).removeprefix("sha256:")
    if file_sha256(args.primary_train_test_cache) != expected:
        raise PermissionError("C2 official-test cache differs from the pre-test freeze.")
    result_dir = Path(args.result_dir).resolve()
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite C2 test results: {result_dir}")
    blob = load_pickle(args.primary_train_test_cache)
    payload = blob.get("payload", blob)
    if any(str(row.get("source_split")) != "test" for row in payload["eval_metadata"]):
        raise PermissionError("C2 evaluation payload is not exclusively official test.")
    pooler = PoolReader(args.sqlite_cache)
    projected = build_test_projection(payload, pooler)
    values, offsets = materialize_validation(projected)
    device = resolve_device(args.device)
    root = Path(args.output_root).resolve()
    prediction_dir = result_dir / "predictions"
    metrics = {}
    for seed in SEEDS:
        trial = freeze["trials"][str(seed)]
        normalizer_path = root / trial["normalizer_relpath"]
        checkpoint_path = root / trial["checkpoint_relpath"]
        if "sha256:" + file_sha256(normalizer_path) != trial["normalizer_sha256"] or "sha256:" + file_sha256(checkpoint_path) != trial["checkpoint_sha256"]:
            raise RuntimeError(f"C2 frozen artifacts changed for seed {seed}.")
        normalizer = ProjectionNormalizer.load(normalizer_path)
        normalized = normalizer.transform(values)
        config = freeze["selected_config"]
        model = ProjectedRanker.build(int(config["hidden_width"]), float(config["dropout"])).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        score_parts = []
        with torch.no_grad():
            for start in range(0, len(normalized), 1024):
                score_parts.append(model.score(torch.from_numpy(normalized[start:start+1024]).to(device)).cpu().numpy())
        scores = np.concatenate(score_parts)
        reranked = []
        for index, (_, candidates) in enumerate(projected["eval_pwc"]):
            start, stop = offsets[index:index+2]
            order = np.argsort(-scores[start:stop], kind="stable")
            reranked.append(([candidates[position] for position in order], scores[start:stop][order]))
        prediction_dir.mkdir(parents=True, exist_ok=True)
        evaluation = evaluate_reranking(
            projected["eval_pwc"], projected["eval_ground_truths"], None,
            ks=[1, 3, 5, 10], output_csv=str(prediction_dir / f"projected_seed_{seed}.csv"),
            precomputed_reranked_results=reranked, reaction_metadata=projected["eval_metadata"],
        )
        metrics[str(seed)] = {
            "top1": evaluation.reranked_accuracy[1], "top3": evaluation.reranked_accuracy[3],
            "top5": evaluation.reranked_accuracy[5], "top10": evaluation.reranked_accuracy[10],
            "mrr": evaluation.reranked_mrr, "baseline_top1": evaluation.baseline_accuracy[1],
            "baseline_mrr": evaluation.baseline_mrr, "n_test_reactions": evaluation.n_products,
        }
    manifest = c2_test_manifest(
        freeze=freeze, primary_train_test_cache=args.primary_train_test_cache,
        sqlite_cache=args.sqlite_cache, metrics=metrics, environment=environment_record(device),
    )
    manifest["test_pooling_coverage"] = pooler.coverage()
    atomic_json(manifest, result_dir / "manifest.json")
    print(json.dumps(manifest["descriptive_summary"], indent=2))


def run_finalize_existing_test(args) -> None:
    """Recover an interrupted manifest write from complete immutable prediction CSVs.

    This command is deliberately score-free: it reads the already-written five
    CSVs, re-derives their metrics, checks paired identity/order, and writes
    only the missing immutable manifest.
    """

    freeze = validate_c2_freeze(args.freeze)
    freeze["_source_path"] = str(Path(args.freeze).resolve())
    expected = str(freeze["retained_train_test_cache_sha256"]).removeprefix("sha256:")
    if file_sha256(args.primary_train_test_cache) != expected:
        raise PermissionError("C2 official-test cache differs from the pre-test freeze.")
    result_dir = Path(args.result_dir).resolve()
    manifest_path = result_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"C2 result manifest already exists: {manifest_path}")
    metrics, prediction_records = collect_existing_prediction_metrics(result_dir / "predictions")
    manifest = c2_test_manifest(
        freeze=freeze, primary_train_test_cache=args.primary_train_test_cache,
        sqlite_cache=args.sqlite_cache, metrics=metrics,
        environment={"recovery": "no-model-rescoring", "python": sys.version},
        recovery={
            "mode": "recovered-from-complete-immutable-prediction-csvs",
            "single_intended_change": "manifest-only recovery after optional-repair-sidecar bug",
            "prediction_files": prediction_records,
        },
    )
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest["descriptive_summary"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-selection")
    prepare.add_argument("--primary-selection-bundle", required=True)
    prepare.add_argument("--primary-freeze", required=True)
    prepare.add_argument("--sqlite-cache", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--max-train-products", type=int)
    prepare.add_argument("--max-validation-products", type=int)
    prepare.add_argument("--hash-sqlite", action="store_true")
    fit = sub.add_parser("fit-validation")
    fit.add_argument("--selection-cache", required=True)
    fit.add_argument("--primary-freeze", required=True)
    fit.add_argument("--output-root", required=True)
    fit.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    fit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--selection-cache", required=True)
    freeze.add_argument("--primary-freeze", required=True)
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate-test")
    evaluate.add_argument("--freeze", required=True)
    evaluate.add_argument("--primary-train-test-cache", required=True)
    evaluate.add_argument("--sqlite-cache", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--result-dir", required=True)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    finalize = sub.add_parser("finalize-existing-test")
    finalize.add_argument("--freeze", required=True)
    finalize.add_argument("--primary-train-test-cache", required=True)
    finalize.add_argument("--sqlite-cache", required=True)
    finalize.add_argument("--result-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-selection":
        run_prepare(args)
    elif args.command == "fit-validation":
        run_fit(args)
    elif args.command == "freeze":
        run_freeze(args)
    elif args.command == "evaluate-test":
        run_evaluate(args)
    elif args.command == "finalize-existing-test":
        run_finalize_existing_test(args)


if __name__ == "__main__":
    main()
