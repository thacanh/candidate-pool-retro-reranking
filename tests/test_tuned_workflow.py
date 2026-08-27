from types import SimpleNamespace

import rerank.experiments.orchestrate_tuned_revision as workflow


def _args(stage):
    return SimpleNamespace(
        stage=stage,
        conformer_seed=42,
        conformer_root="outputs/jcheminform_revision/conformers",
        shard_index=2,
        shard_count=9,
        device="cpu",
        capacity_dropout=None,
        capacity_learning_rate=None,
        capacity_margin=None,
        capacity_decision_note=None,
        stop_after_epoch=None,
        stop_margin_seconds=600.0,
    )


def test_baseline_shard_covers_all_prior_transforms_and_never_mentions_test(tmp_path):
    paths = workflow.workflow_paths(tmp_path / "tuned")
    commands = workflow.stage_commands(_args("baseline-shard"), paths)
    assert len(commands) == 3
    assert {command[command.index("--prior-transform") + 1] for command in commands} == {
        "raw", "log", "rank"
    }
    assert all("evaluate-test" not in command for command in commands)
    assert all(command[command.index("--shard-index") + 1] == "2" for command in commands)
    assert all("--compact-progress" in command for command in commands)


def test_search_budget_deadline_is_forwarded_without_test_access(tmp_path):
    paths = workflow.workflow_paths(tmp_path / "tuned")
    args = _args("baseline-shard")
    args.stop_after_epoch = 1234567890.0
    args.stop_margin_seconds = 900.0
    commands = workflow.stage_commands(args, paths)
    assert all("--stop-after-epoch" in command for command in commands)
    assert all(command[command.index("--stop-margin-seconds") + 1] == "900.0" for command in commands)
    assert all("evaluate-test" not in command for command in commands)


def test_augmented_search_requires_prior_freeze_and_test_is_post_freeze(tmp_path):
    paths = workflow.workflow_paths(tmp_path / "tuned")
    paths.prior_freeze.parent.mkdir(parents=True)
    paths.prior_freeze.write_text(
        '{"selected_prior_transform":"log"}', encoding="utf-8"
    )
    augmented = workflow.stage_commands(_args("augmented-shard"), paths)[0]
    assert "--prior-freeze" in augmented
    assert augmented[augmented.index("--prior-transform") + 1] == "log"
    evaluate = workflow.stage_commands(_args("evaluate-test"), paths)[0]
    assert "--selection-freeze" in evaluate
    assert "--train-test-cache" in evaluate
    assert "search" not in evaluate


def test_capacity_plan_requires_explicit_non_width_settings(tmp_path):
    paths = workflow.workflow_paths(tmp_path / "tuned")
    args = _args("prepare-capacity")
    try:
        workflow.stage_commands(args, paths)
        assert False, "under-specified capacity plan must fail"
    except ValueError:
        pass
    args.capacity_dropout = 0.1
    args.capacity_learning_rate = 3e-4
    args.capacity_margin = 0.1
    args.capacity_decision_note = "Frozen before capacity-control fitting"
    command = workflow.stage_commands(args, paths)[0]
    assert "prepare-capacity" in command
    assert "--decision-note" in command
