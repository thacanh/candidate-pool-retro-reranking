#!/usr/bin/env python
"""Export official-validation features from one completed conformer cache.

The legacy fixed-50 runner intentionally evaluates test directly and therefore
does not retain official-validation features.  This small companion artifact,
together with the legacy train/test feature cache, preserves all split features
needed for later prespecified tuning and conformer figures without regenerating
the expensive conformer.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

from rerank.cached_encoder import SqliteCachedUniMolEncoder
from rerank.features import FeatureExtractor
from rerank.study_data import (
    STUDY_CACHE_SCHEMA,
    _extract_evaluation_feature_payload,
    file_fingerprint,
    load_candidate_pools,
    load_reactions,
)


def export_validation_features(
    source_csv: str | Path,
    metadata_csv: str | Path,
    candidate_jsonl: str | Path,
    atom_cache: str | Path,
    output: str | Path,
    conformer_seed: int,
) -> dict:
    output = Path(output)
    reactions = load_reactions(source_csv, metadata_csv)
    pools, candidate_audit = load_candidate_pools(candidate_jsonl)
    encoder = SqliteCachedUniMolEncoder(
        str(atom_cache), log_misses=False, strict=False
    )
    extractor = FeatureExtractor(encoder, feature_mode="3d+prior")
    payload, audit = _extract_evaluation_feature_payload(
        reactions, pools, extractor, "valid"
    )
    blob = {
        "schema_version": STUDY_CACHE_SCHEMA,
        "artifact_kind": "official_validation_conformer_features",
        "protocol_id": "cap10-conformer-features-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "conformer_seed": int(conformer_seed),
        "feature_mode": "3d+prior",
        "input_fingerprints": {
            "source_csv": file_fingerprint(source_csv),
            "metadata_csv": file_fingerprint(metadata_csv),
            "candidate_jsonl": file_fingerprint(candidate_jsonl),
            "atom_cache": file_fingerprint(atom_cache, include_sha256=False),
        },
        "candidate_audit": candidate_audit,
        "encoder_coverage": encoder.get_coverage_metrics(),
        "audit": audit,
        "payload": payload,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(blob, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output)
    return {
        "output": str(output.resolve()),
        "audit": audit,
        "encoder_coverage": blob["encoder_coverage"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", default="data/uspto_smiles.csv")
    parser.add_argument("--metadata-csv", default="data/uspto_reaction_metadata.csv")
    parser.add_argument("--candidate-jsonl", default="outputs/rerank_dataset.jsonl")
    parser.add_argument("--atom-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conformer-seed", type=int, required=True)
    args = parser.parse_args()
    result = export_validation_features(
        args.source_csv,
        args.metadata_csv,
        args.candidate_jsonl,
        args.atom_cache,
        args.output,
        args.conformer_seed,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
