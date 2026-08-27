#!/usr/bin/env python
"""Load the migrated checkpoint through AiZynthModels without scientific inference."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from run_inference import build_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    from aizynthmodels.chemformer import Chemformer

    with tempfile.TemporaryDirectory(prefix="chemformer_f1_smoke_") as directory:
        path = Path(directory) / "smoke.csv"
        pd.DataFrame({"reactants": ["C"], "products": ["C"], "set": ["test"]}).to_csv(path, sep="\t", index=False)
        config = build_config(
            path,
            Path(args.checkpoint).resolve(),
            Path(args.vocabulary).resolve(),
            args.device,
            1,
        )
        model = Chemformer(config)
        model.datamodule._num_workers = 0
        state = model.model.state_dict()
        if len(model.tokenizer) != 523 or len(state) != 188:
            raise RuntimeError(
                f"Unexpected model structure: vocab={len(model.tokenizer)}, tensors={len(state)}"
            )
        payload = {
            "status": "pass",
            "model_initialization": "pass; one synthetic row loaded; no prediction",
            "vocabulary_size": len(model.tokenizer),
            "state_tensor_count": len(state),
            "device": args.device,
            "scientific_input_loaded": False,
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
