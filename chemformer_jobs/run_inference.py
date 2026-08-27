#!/usr/bin/env python
"""Run pinned AiZynthModels Chemformer beam inference for prepared F1 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROTOCOL_ID = "f1-chemformer-forward-inference-v1"
EXPECTED_ROWS = 4_762
EXPECTED_BEAMS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_config(input_csv: Path, checkpoint: Path, vocabulary: Path, device: str, batch_size: int):
    from omegaconf import OmegaConf
    from aizynthmodels.utils.configs.chemformer.predict import Predict

    config = OmegaConf.structured(Predict)
    config.data_path = str(input_csv)
    config.vocabulary_path = str(vocabulary)
    config.model_path = str(checkpoint)
    config.task = "forward_prediction"
    config.device = device
    config.mode = "eval"
    config.dataset_part = "test"
    config.batch_size = batch_size
    config.n_predictions = EXPECTED_BEAMS
    config.sample_unique = False
    config.seed = 1
    config.n_devices = 1
    config.n_chunks = 1
    config.i_chunk = 0
    config.trainer = None
    config.callbacks = None
    config.scorers = None
    return config


def run(args: argparse.Namespace) -> dict:
    import pytorch_lightning as pl
    import torch
    from aizynthmodels.chemformer import Chemformer
    from aizynthmodels.utils.writing import predictions_to_file

    source = Path(args.input).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    vocabulary = Path(args.vocabulary).resolve()
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite F1 inference output/manifest.")
    frame = pd.read_csv(source, sep="\t", keep_default_na=False)
    if list(frame.columns) != ["reactants", "products", "set"]:
        raise RuntimeError("Prepared Chemformer input has an unexpected schema/order.")
    if len(frame) != EXPECTED_ROWS or set(frame["set"]) != {"test"}:
        raise RuntimeError("Prepared Chemformer input is not the frozen 4,762-row test set.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    work_dir = output.parent / ".f1_chemformer_runtime"
    if work_dir.exists():
        raise FileExistsError("Partial F1 runtime directory exists; preserve it for audit.")
    work_dir.mkdir(parents=True)
    runtime_csv = work_dir / "chemformer_input.csv"
    shutil.copyfile(source, runtime_csv)

    pl.seed_everything(1, workers=True)
    config = build_config(
        runtime_csv, checkpoint, vocabulary, args.device, args.batch_size
    )
    model = Chemformer(config)
    model.datamodule._num_workers = args.workers
    predictions = model.predict()

    temporary = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_to_file(
        str(temporary),
        predictions["predictions"],
        predictions["log_likelihoods"],
        predictions.get("ground_truth"),
        prediction_col="sampled_smiles",
        ranking_metric_col="log_likelihood",
    )
    produced = pd.read_csv(temporary, sep="\t", keep_default_na=False)
    required = {"ground_truth"} | {
        f"sampled_smiles_{index}" for index in range(1, EXPECTED_BEAMS + 1)
    }
    if len(produced) != EXPECTED_ROWS or not required.issubset(produced.columns):
        raise RuntimeError("Chemformer output failed its row/schema gate.")
    if produced["ground_truth"].tolist() != frame["products"].tolist():
        raise RuntimeError("Chemformer output target/order differs from frozen input.")
    os.replace(temporary, output)
    shutil.rmtree(work_dir)

    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "comparator": "identical forward model/settings for all frozen systems",
        "single_intended_change": "forward round-trip scoring of frozen precursor sets",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "task": "forward_prediction",
            "beam_count": EXPECTED_BEAMS,
            "beam1_primary": True,
            "beam5_sensitivity": True,
            "seed": 1,
            "sample_unique": False,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "device": args.device,
        },
        "input": {"path": str(source), "size_bytes": source.stat().st_size, "sha256": sha256(source)},
        "checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint)},
        "vocabulary": {"path": str(vocabulary), "size_bytes": vocabulary.stat().st_size, "sha256": sha256(vocabulary)},
        "output": {"path": str(output), "size_bytes": output.stat().st_size, "sha256": sha256(output)},
        "rows": EXPECTED_ROWS,
        "test_partition_used_for_training_or_selection": False,
        "torch": torch.__version__,
        "pytorch_lightning": pl.__version__,
    }
    atomic_json(manifest_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
