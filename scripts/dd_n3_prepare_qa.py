"""Release QA: prepare all real N3 S2 inputs, without fitting or selecting models."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from dd_pack_remote import PRIMARY, FEATURES, K1, POOLS, INDEX
from run_dd_n3_remote import prepare_s2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    tasks = prepare_s2(SimpleNamespace(
        approval_record="docs/dd_remote_approval.json", analysis_plan="docs/analysis_plan.md",
        s2_root=args.output, feature_root=FEATURES, k1_root=K1, pool_root=POOLS,
        index_root=INDEX, products_jsonl=f"{POOLS}/products.jsonl",
        source_csv="data/uspto_smiles.csv", metadata_csv="data/uspto_reaction_metadata.csv",
        primary_freeze=f"{PRIMARY}/freeze/model_selection.json", device="cpu"))
    assert len(tasks) == 30
    counts = {}
    for pool in ("aizynthfinder_only", "localretro_only", "merged"):
        for transform in ("log", "rank"):
            p = Path(args.output) / pool / transform / "selection/manifest.json"
            m = json.loads(p.read_text(encoding="utf-8"))
            assert m["train_overlap_reactions_excluded"] == 233
            assert not m["test_ground_truth_loaded"]
            counts[f"{pool}/{transform}"] = m["validation_audit"]
    print(json.dumps(dict(status="pass", training_performed=False, model_selection_performed=False,
        validation_inputs=counts, planned_fit_tasks=len(tasks), runtime_seconds=time.perf_counter()-started), indent=2))


if __name__ == "__main__":
    main()
