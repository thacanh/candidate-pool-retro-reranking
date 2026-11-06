"""Verify and import returned N3 archives without altering frozen artifact bytes."""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from archive_results import verify_stage
from archive_paths import imported_archive
from package_inputs import _safe_member, PRIMARY, K1, COMMON_COHORT
from rerank.analysis.transfer_statistics import common_cohort, historical_g1_ranks
from rerank.experiments.protocol_gate import sha256
from rerank.experiments.select_prior_transform import choose_transform
from rerank.pool_streaming import atomic_json


def main():
    root = Path.cwd()
    report_path = root / "outputs/dd_final_round/N3_return_audit_20260923/manifest.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    archives, receipts, checked = {}, {}, []
    for stage in ("N3_S1", "N3_S2", "N3_S3"):
        receipt_path = imported_archive(root, f"{stage}.receipt.json")
        checked.append(verify_stage(SimpleNamespace(receipt=str(receipt_path))))
        receipts[stage] = json.loads(receipt_path.read_text(encoding="utf-8"))
        archives[stage] = zipfile.ZipFile(imported_archive(root, f"{stage}.zip"))
        for name in archives[stage].namelist():
            _safe_member(name)

    def read(stage, path):
        return json.loads(archives[stage].read(f"{stage}/results/{path}"))

    def bind(record):
        path = record["path"].replace("\\", "/")
        for stage in archives:
            marker = f"/outputs/dd_final_round/{stage}/"
            if marker in path:
                member = f"{stage}/results/" + path.split(marker, 1)[1]
                actual = receipts[stage]["members"][member]
                assert record["sha256"] == actual["sha256"], member
                assert record["size_bytes"] == actual["size_bytes"], member
                return
        raise ValueError(f"Expected an archived N3 artifact: {path}")

    manifests = {stage: read(stage, "manifest.json") for stage in archives}
    for stage, m in manifests.items():
        assert m["status"] == "complete"
        assert m["protocol_id"] == "dd-n3-pool-specific-selection-v1"
        assert m["approval"]["sha256"] == sha256("docs/dd_remote_approval.json")
    s1, s2, s3 = (manifests[f"N3_S{i}"] for i in (1, 2, 3))
    assert s1["training_performed"] is False and s1["normalizer_refit_performed"] is False
    assert s2["test_partition_loaded"] is False and s2["test_references_loaded"] is False
    assert s3["test_partition_loaded_only_after_choice_and_model_freezes"] is True
    bind(s3["s2_choice_freeze"])
    bind(s3["transfer_effects"])
    assert archives["N3_S2"].read("N3_S2/results/choice_freeze.json") == archives["N3_S2"].read(
        "N3_S2/results/manifest.json"
    )
    primary = json.loads(Path(PRIMARY, "freeze/model_selection.json").read_text(encoding="utf-8"))
    counts = {}
    for stage, expected in (("N3_S2", 30), ("N3_S3", 120)):
        trial_names = [n for n in archives[stage].namelist() if n.endswith("/trial.json")]
        assert len(trial_names) == expected
        for name in trial_names:
            trial = json.loads(archives[stage].read(name))
            assert trial["test_partition_loaded"] is False
            assert trial["config"] == primary[f"selected_{trial['arm']}"]["config"]
            for key in ("checkpoint", "normalizer", "selection_manifest"):
                bind(trial[key])
        counts[stage] = len(trial_names)
    choices = {}
    ids, _ = common_cohort(COMMON_COHORT)
    historical = historical_g1_ranks(Path(PRIMARY) / "g1_20seed/test_results/predictions", ids)
    csv_rows = list(
        csv.DictReader(
            io.StringIO(archives["N3_S3"].read("N3_S3/results/transfer_effects.csv").decode())
        )
    )
    assert len(csv_rows) == 6 and s3["common_reactions"] == len(ids) == 3814
    effects = []
    for pool, chosen in s2["choices"].items():
        means = {}
        for transform, stats in chosen["validation"].items():
            actual = []
            for seed in range(42, 47):
                relative = f"models/validation/baseline/seed_{seed}/trial.json"
                if transform == "raw":
                    trial = json.loads(Path(K1, pool, relative).read_text(encoding="utf-8"))
                else:
                    trial = read("N3_S2", f"{pool}/{transform}/{relative}")
                actual.append(trial["best_validation_mrr"])
                assert actual[-1] == stats["per_seed"][str(seed)]
            means[transform] = float(np.mean(actual))
            assert abs(means[transform] - stats["mean_best_validation_mrr"]) < 1e-12
        assert choose_transform(means) == chosen["selected"] == s3["choices"][pool]
        choices[pool] = dict(
            selected=chosen["selected"],
            validation_means=means,
            validation_gain_over_raw=means[chosen["selected"]] - means["raw"],
        )
        frozen = read("N3_S3", f"{pool}/models/freeze.json")
        assert frozen["complete"] and frozen["seeds"] == list(range(42, 62))
        assert frozen["selected_prior_transform"] == chosen["selected"]
        for arm in ("baseline", "augmented"):
            assert set(frozen["trials"][arm]) == set(map(str, range(42, 62)))
            for record in frozen["trials"][arm].values():
                bind(record)
        test = read("N3_S3", f"{pool}/test_results/manifest.json")
        assert test["test_partition_loaded_only_after_model_freeze"]
        for key in ("model_freeze", "feature_freeze", "predictions"):
            bind(test[key])
        bind(s3["result_predictions"][pool])
        rows = [
            json.loads(line)
            for line in archives["N3_S3"]
            .read(f"N3_S3/results/{pool}/test_results/reaction_ranks.jsonl")
            .decode()
            .splitlines()
        ]
        by_id = {r["reaction_id"]: r for r in rows}
        assert len(rows) == len(by_id) == test["test_reactions_total"] == 5004
        covered = [r for r in rows if r["covered"]]
        assert len(covered) == test["test_reactions_covered"]
        assert all(by_id[rid]["covered"] for rid in ids)
        arrays = {}
        for arm in ("baseline", "augmented"):
            arrays[arm] = np.asarray(
                [[by_id[rid][f"{arm}_rank_seed_{seed}"] for rid in ids] for seed in range(42, 62)]
            )
            assert np.all((arrays[arm] >= 1) & (arrays[arm] <= 10))
            for seed in range(42, 62):
                ranks = np.asarray([r[f"{arm}_rank_seed_{seed}"] for r in covered])
                for metric, val in (
                    ("top1", float(np.mean(ranks == 1))),
                    ("mrr", float(np.mean(1 / ranks))),
                ):
                    stored = test["per_seed_metrics"][arm][str(seed)]
                    assert abs(val - stored["within_pool"][metric]) < 1e-12
                    assert (
                        abs(val * len(covered) / len(rows) - stored["end_to_end"][metric]) < 1e-12
                    )
        for metric in ("top1", "mrr"):
            score = (lambda r: (r == 1).astype(float)) if metric == "top1" else (lambda r: 1 / r)
            old = float((score(historical[1]) - score(historical[0])).mean())
            new = float((score(arrays["augmented"]) - score(arrays["baseline"])).mean())
            row = next(r for r in csv_rows if r["pool"] == pool and r["metric"] == metric)
            for key, val in (
                ("historical_effect", old),
                ("expanded_effect", new),
                ("transfer_loss", new - old),
            ):
                assert abs(val - float(row[key])) < 1e-12
            effects.append(
                {k: (v if k in ("pool", "metric") else float(v)) for k, v in row.items()}
            )
    # Import only after every check passed. Never rewrite embedded Linux paths.
    targets = {s: root / "outputs/dd_final_round" / s for s in archives}
    for target in targets.values():
        if target.exists():
            raise FileExistsError(f"Refusing existing result directory: {target}")
    imported_files = 0
    for stage, z in archives.items():
        prefix = f"{stage}/results/"
        for name in z.namelist():
            if not name.startswith(prefix):
                continue
            target = targets[stage] / name[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as source, target.open("xb") as dest:
                import shutil

                shutil.copyfileobj(source, dest)
            assert sha256(target) == receipts[stage]["members"][name]["sha256"]
            imported_files += 1
    clipping = [
        d["training_pair_prior_clipping"]["prior_total_clipped_fraction"]
        for pool in s1["pools"].values()
        for arm in pool["arms"].values()
        for d in arm.values()
    ]
    report = dict(
        status="verified_and_imported",
        protocol_id=s3["protocol_id"],
        archives=checked,
        files_imported=imported_files,
        trial_counts=counts,
        choices=choices,
        common_reactions=3814,
        seeds=list(range(42, 62)),
        effects_recalculated_from_predictions=effects,
        maximum_s1_prior_clipping=max(clipping),
        s1_numerical_tie_audit_count=len(s1["numerical_tie_audit"]),
        scientific_artifact_bytes_unchanged=True,
        ci_recomputed=False,
        manuscript_modified=False,
        instance_needed_to_recover_n3_results=False,
        caveats=[
            "S2 runtime_seconds times choice calculation only, not preparation/training.",
            "Own-pool effect intervals are not in transfer_effects.csv; positive point estimates alone are not evidence of resolved gains.",
            "Transfer-loss crossed intervals exclude zero in 3/6 contrasts after prior reselection, not 6/6.",
        ],
    )
    atomic_json(report_path, report)
    print(
        json.dumps(
            dict(
                status=report["status"],
                report=str(report_path),
                imported_files=imported_files,
                choices=choices,
                effects=effects,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
