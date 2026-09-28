"""Post hoc prior supplement; frozen predictions only, no model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = "dd-prior-increment-gap-supplement-v1"
SEEDS = np.arange(42, 62)
DRAWS = 10000
ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def record(path):
    p = Path(path).resolve()
    return {"path": str(p), "size_bytes": p.stat().st_size, "sha256": digest(p)}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def gap_groups(priors):
    """Covariate-only groups; -1 singleton, 0..3 quartiles; ties go lower."""
    gaps = np.full(len(priors), np.nan)
    for i, p in enumerate(priors):
        p = np.asarray(p, dtype=float)
        if p.ndim != 1 or not len(p) or not np.isfinite(p).all():
            raise ValueError("Invalid frozen prior list")
        if len(p) > 1:
            ordered = np.sort(p)
            gaps[i] = ordered[-1] - ordered[-2]
    valid = np.isfinite(gaps)
    if not valid.any():
        raise ValueError("No nonsingleton prior gap")
    cuts = np.quantile(gaps[valid], [.25, .5, .75], method="linear")
    groups = np.full(len(gaps), -1, dtype=int)
    groups[valid] = np.searchsorted(cuts, gaps[valid], side="left")
    return gaps, cuts, groups


def weights(rng, batch, n):
    # Multinomial counts are equivalent to resampling n observations uniformly.
    return rng.multinomial(n, np.full(n, 1.0 / n), size=batch) / n


def bootstrap_pools(values, draws=DRAWS, seed=2026092701):
    """values: pool x seed x distinct-product; product draws shared by pool."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 3 or v.shape[0] != 2 or not np.isfinite(v).all():
        raise ValueError("Expected two finite pool matrices")
    rng = np.random.default_rng(seed)
    out = np.empty((draws, 3))
    for start in range(0, draws, 100):
        b = min(100, draws-start)
        pw = weights(rng, b, v.shape[2])
        for pool in range(2):
            sw = weights(rng, b, v.shape[1])
            out[start:start+b, pool] = np.sum((pw @ v[pool].T) * sw, axis=1)
        out[start:start+b, 2] = out[start:start+b, 1] - out[start:start+b, 0]
    return out


def bootstrap_groups(values, groups, draws=DRAWS, seed=2026092702):
    """Shared paired seed draw; independent product draws inside fixed strata."""
    v = np.asarray(values, dtype=float)  # endpoint x seed x product
    groups = np.asarray(groups)
    if v.ndim != 3 or v.shape[2] != len(groups) or not np.isfinite(v).all():
        raise ValueError("Invalid group matrix pairing")
    rng = np.random.default_rng(seed)
    out = np.full((draws, 4, v.shape[0]), np.nan)
    for start in range(0, draws, 100):
        b = min(100, draws-start)
        sw = weights(rng, b, v.shape[1])
        for g in range(4):
            ix = np.flatnonzero(groups == g)
            if not len(ix):
                continue
            pw = weights(rng, b, len(ix))
            for e in range(v.shape[0]):
                out[start:start+b, g, e] = np.sum((pw @ v[e][:, ix].T) * sw, axis=1)
    return out


def interval_row(label, effect, draws, **extra):
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(contrast=label, effect_pp=float(effect)*100,
                ci95_low_pp=float(lo)*100, ci95_high_pp=float(hi)*100, **extra)


def run(output):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    plan_path = ROOT / "docs/analysis_plan.md"
    # Fail closed unless current amendment and code are committed.
    owned = ["docs/analysis_plan.md", str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
             "tests/test_dd_prior_gap.py"]
    for p in owned:
        saved = subprocess.check_output(["git", "show", f"HEAD:{p}"], cwd=ROOT)
        actual = (ROOT / p).read_bytes()
        if saved.replace(b"\r\n", b"\n") != actual.replace(b"\r\n", b"\n"):
            raise ValueError(f"Uncommitted execution input: {p}")
    if PROTOCOL not in plan_path.read_text(encoding="utf-8"):
        raise PermissionError("Plan amendment absent")
    inputs = {}

    def read_json(path):
        p = ROOT / path
        inputs[str(p)] = record(p)
        return json.loads(p.read_text(encoding="utf-8-sig"))

    def checked(path, expected):
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        r = record(p)
        if r["sha256"] != expected["sha256"].removeprefix("sha256:"):
            raise ValueError(f"Hash mismatch: {p}")
        if "size_bytes" in expected and r["size_bytes"] != expected["size_bytes"]:
            raise ValueError(f"Size mismatch: {p}")
        inputs[str(p)] = r
        return p

    n2root = ROOT / "outputs/dd_final_round/N2"
    n2 = read_json(n2root / "manifest.json")
    if n2["status"] != "complete" or n2["protocol_id"] != "dd-n2-cross-inference-v1":
        raise ValueError("Wrong N2 family")
    common_path = checked(n2["inputs"]["common_reactions_csv"]["path"], n2["inputs"]["common_reactions_csv"])
    common = pd.read_csv(common_path)
    common = common[common.pool == "aizynthfinder_only"].sort_values("reaction_id")
    ids = common.reaction_id.to_numpy()
    if len(ids) != 3814 or len(set(ids)) != 3814 or common.canonical_product.nunique() != 3814:
        raise ValueError("Expected 3814 distinct product clusters")
    cache_path = checked(n2["inputs"]["historical_cache"]["path"], n2["inputs"]["historical_cache"])
    from rerank.study_data import canonicalize_smiles
    with cache_path.open("rb") as f:
        cache = pickle.load(f)["payload"]
    wanted = dict(zip(ids, common.canonical_product))
    prior_lookup = {}
    for meta, pwc, features in zip(cache["eval_metadata"], cache["eval_pwc"], cache["eval_features"], strict=True):
        rid = int(meta["reaction_id"])
        if rid not in wanted:
            continue
        product, candidates = pwc
        if meta["source_split"] != "test" or canonicalize_smiles(product) != wanted[rid]:
            raise ValueError("Historical product/split mismatch")
        if rid in prior_lookup or len(features) != len(candidates) or not 1 <= len(features) <= 10:
            raise ValueError("Historical candidate alignment mismatch")
        prior_lookup[rid] = np.asarray(features)[:, 0].astype(float)
    if set(prior_lookup) != set(ids):
        raise ValueError("Missing historical feature lists")
    priors = [prior_lookup[i] for i in ids]
    gaps, cuts, groups = gap_groups(priors)
    output.mkdir(parents=True)
    design = dict(protocol_id=PROTOCOL, status="bin_design_frozen_before_prediction_analysis",
                  cutpoints=cuts.tolist(), counts={str(g): int(np.sum(groups == g)) for g in range(-1, 4)},
                  exact_top_prior_ties=int(np.sum(gaps == 0)), quantile_method="linear",
                  boundary_rule="ties at cutpoint go to lower bin; no tie splitting",
                  seeds=SEEDS.tolist(), bootstrap_draws=DRAWS, rng_seeds=[2026092701, 2026092702],
                  pointwise_intervals=True, test_access="post-freeze post-hoc analysis only",
                  approval_quote="thế cứ chạy đi", approval_date="2026-09-27",
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  analysis_plan=record(plan_path), runner=record(__file__),
                  training_performed=False, model_inference_performed=False)
    write_json(output / "design_freeze.json", design)
    print("Input identity gates passed; gap grouping frozen. Loading frozen outcomes.", flush=True)

    n2ranks = checked(n2root / "cell_ranks.npz", n2["outputs"]["cell_ranks"])
    with np.load(n2ranks, allow_pickle=False) as z:
        ranks = z["ranks"].copy()
        if not np.array_equal(z["reaction_ids"], ids) or not np.array_equal(z["seeds"], SEEDS):
            raise ValueError("N2 seed/reaction pairing mismatch")
    if ranks.shape != (2, 2, 2, 20, 3814) or not np.all((ranks >= 1) & (ranks <= 10)):
        raise ValueError("N2 rank dimensions/range mismatch")
    stats = read_json("outputs/jcheminform_revision/final_statistics_20seed/manifest.json")
    predroot = ROOT / "outputs/jcheminform_revision/tuned_primary/conformer_seed_42/g1_20seed/test_results/predictions"
    prior_rank = None
    for si, seed in enumerate(SEEDS):
        for ai, arm in enumerate(("baseline", "augmented")):
            name = f"{arm}_seed_{seed}.csv"
            p = checked(predroot / name, {"sha256": stats["prediction_sha256"][name]})
            f = pd.read_csv(p, usecols=["reaction_id", "baseline_rank", "reranked_rank"]).set_index("reaction_id").loc[ids]
            if not np.array_equal(f.reranked_rank, ranks[0, 0, ai, si]):
                raise ValueError("Historical diagonal replay mismatch")
            if prior_rank is None:
                prior_rank = f.baseline_rank.to_numpy()
            if not np.array_equal(prior_rank, f.baseline_rank):
                raise ValueError("Prior ranks vary by arm/seed")
    k1path = checked(n2["inputs"]["aiz_prediction_jsonl"]["path"], n2["inputs"]["aiz_prediction_jsonl"])
    with k1path.open() as f:
        k1 = {int(r["reaction_id"]): r for r in map(json.loads, f) if int(r["reaction_id"]) in wanted}
    if set(k1) != set(ids):
        raise ValueError("Anchored cohort mismatch")
    anchored_prior = np.asarray([k1[i]["prior_rank"] for i in ids])
    for ai, arm in enumerate(("baseline", "augmented")):
        for si, seed in enumerate(SEEDS):
            expected = [k1[i][f"{arm}_rank_seed_{seed}"] for i in ids]
            if not np.array_equal(expected, ranks[1, 1, ai, si]):
                raise ValueError("Anchored diagonal replay mismatch")
    hp, ap = prior_rank == 1, anchored_prior == 1
    hb, ha = ranks[0, 0, 0] == 1, ranks[0, 0, 1] == 1
    aa = ranks[1, 1, 1] == 1
    increments = np.stack([ha.astype(float)-hp, aa.astype(float)-ap])
    if not np.allclose(increments.mean(axis=(1, 2)), [0.017973256423702177, 0.004457262716308308], atol=1e-12, rtol=0):
        raise ValueError("Reviewed common-cohort point estimates differ")
    single = groups == -1
    if single.any() and not (hp[single].all() and hb[:, single].all() and ha[:, single].all()):
        raise ValueError("Singleton Top-1 gate failed")
    print("Frozen diagonal ranks and prior endpoints verified; computing 10,000-draw intervals.", flush=True)
    boot = bootstrap_pools(increments)
    effects = increments.mean(axis=(1, 2))
    labels = ["historical_prior_to_augmented", "anchored_prior_to_augmented", "anchored_minus_historical_prior_to_augmented"]
    allrows = [interval_row(labels[i], e, boot[:, i], reactions=3814) for i, e in enumerate([*effects, effects[1]-effects[0]])]
    pd.DataFrame(allrows).to_csv(output / "prior_increment_intervals.csv", index=False)
    values = np.stack([ha.astype(float)-hp, ha.astype(float)-hb])
    gb = bootstrap_groups(values, groups)
    group_rows = []
    endpoints = ["prior_to_augmented", "baseline_to_augmented"]
    for g in range(-1, 4):
        mask = groups == g
        n = int(mask.sum())
        for e, endpoint in enumerate(endpoints):
            extra = dict(group="singleton" if g == -1 else f"Q{g+1}", reactions=n,
                         status="complete" if n else "empty_stratum",
                         prior_top1_percent=float(hp[mask].mean()*100) if n else None,
                         baseline_top1_percent=float(hb[:, mask].mean()*100) if n else None,
                         augmented_top1_percent=float(ha[:, mask].mean()*100) if n else None,
                         gap_min=float(np.nanmin(gaps[mask])) if n and g >= 0 else None,
                         gap_max=float(np.nanmax(gaps[mask])) if n and g >= 0 else None,
                         zero_gap_count=int(np.sum(gaps[mask] == 0)))
            if n and g >= 0:
                row = interval_row(endpoint, values[e][:, mask].mean(), gb[:, g, e], **extra)
            else:
                row = dict(contrast=endpoint, effect_pp=0.0 if n else None,
                           ci95_low_pp=0.0 if n else None, ci95_high_pp=0.0 if n else None, **extra)
            group_rows.append(row)
    pd.DataFrame(group_rows).to_csv(output / "prior_gap_groups.csv", index=False)
    contrast_rows = []
    if (groups == 0).any() and (groups == 3).any():
        for e, endpoint in enumerate(endpoints):
            diff = values[e][:, groups == 3].mean() - values[e][:, groups == 0].mean()
            contrast_rows.append(interval_row("Q4_minus_Q1_"+endpoint, diff, gb[:, 3, e]-gb[:, 0, e]))
    else:
        contrast_rows.append(dict(contrast="Q4_minus_Q1", status="unavailable_empty_extreme_group"))
    pd.DataFrame(contrast_rows).to_csv(output / "prior_gap_contrasts.csv", index=False)
    per_reaction = common[["reaction_id", "canonical_product"]].copy()
    per_reaction["candidate_count"] = [len(p) for p in priors]
    per_reaction["raw_prior_gap"] = gaps
    per_reaction["gap_group"] = groups
    per_reaction["prior_rank_historical"] = prior_rank
    per_reaction["prior_rank_anchored"] = anchored_prior
    for e, endpoint in enumerate(endpoints):
        per_reaction[endpoint+"_mean_top1_effect"] = values[e].mean(axis=0)
    per_reaction.to_csv(output / "reaction_diagnostics.csv", index=False)
    np.savez_compressed(output / "bootstrap_draws.npz", prior=boot, strata=gb)
    np.savez_compressed(output / "paired_effects.npz", ids=ids, prior_increments=increments,
                        historical_gap_increments=values, groups=groups)
    for p, frozen in inputs.items():
        if digest(p) != frozen["sha256"]:
            raise ValueError(f"Scientific source changed during analysis: {p}")
    report = dict(protocol_id=PROTOCOL, status="complete", comparator="unchanged historical/anchored frozen diagonal Top-1 predictions",
                  single_intended_change="post hoc prior-increment uncertainty and historical raw-prior-gap reporting",
                  created_at_utc=datetime.now(timezone.utc).isoformat(), design=design, inputs=inputs,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, platform=platform.platform()),
                  runtime_seconds=time.perf_counter()-started, unchanged_input_hashes_verified=True,
                  test_partition_access="frozen post-selection diagnostics; no tuning",
                  training_performed=False, model_inference_performed=False, original_outputs_modified=False,
                  inference_scope="pointwise exploratory CIs, not causal or multiplicity-adjusted discovery",
                  all_prior_ranks_equal_across_pools=bool(np.array_equal(prior_rank, anchored_prior)),
                  output_files={p.name:record(p) for p in output.iterdir() if p.is_file()})
    write_json(output / "manifest.json", report)
    print(json.dumps({"status":"complete", "runtime_seconds":report["runtime_seconds"], "output":str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dd_final_round/prior_gap_supplement_v1")
    run(parser.parse_args().output.resolve())
