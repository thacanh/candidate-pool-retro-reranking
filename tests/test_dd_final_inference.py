import hashlib
import json

import numpy as np

from rerank.analysis.analyze_dd_mechanisms import _transfer_inference
from rerank.analysis.analyze_round_jk import (
    J12_SYNC_PROTOCOL_ID,
    _j12_sync_approval,
)
from rerank.experiments.run_round_jk import PROTOCOL_ID as K1_PROTOCOL_ID


def test_m2b_direct_inference_is_paired_and_deterministic():
    values = np.linspace(-0.03, -0.01, 40)
    products = [f"product-{index}" for index in range(len(values))]
    left = _transfer_inference(values, products, 500, 2_000, 2050, 2060)
    right = _transfer_inference(values, products, 500, 2_000, 2050, 2060)
    assert left == right
    assert left["effect"] == np.mean(values)
    assert left["ci95_high"] < 0
    assert left["paired_sign_flip_p_two_sided"] < 0.01


def test_final_j12_approval_is_bound_to_plan(tmp_path):
    plan = tmp_path / "analysis_plan.md"
    plan.write_text("approved final analysis", encoding="utf-8")
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "protocol_id": K1_PROTOCOL_ID,
                "status": "approved",
                "width128_d1_resolution": "approved",
                "j4_bin_assignment": "median_across_seeds_worse_bin_on_half_tie",
                "supervisor": "repository owner",
                "approval_date": "2026-08-27",
                "analysis_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                "analysis_plan_commit": "abc123",
                "platform_lock": {"test": {"sha256": "abc"}},
                "final_inference_status": "approved_frozen_artifacts_only",
                "j12_sync_protocol_id": J12_SYNC_PROTOCOL_ID,
                "j12_sync_approval_date": "2026-08-27",
                "j12_sync_approval_quote": "approved",
            }
        ),
        encoding="utf-8",
    )
    record = _j12_sync_approval(approval, plan)
    assert record["j12_sync_protocol_id"] == J12_SYNC_PROTOCOL_ID

