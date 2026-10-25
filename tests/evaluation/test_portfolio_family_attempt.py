from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation.test_portfolio_family_result_artifact import _identity
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance


@pytest.fixture(scope="module")
def actual_attempt_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")

    from tests.evaluation.test_portfolio_family_comparison import (
        _actual_ppo_bundle_case,
    )
    from trade_rl.evaluation.runs import portfolio_family_attempt as attempt_module
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        FourFamilyPortfolioAttemptContext,
        run_four_family_portfolio_attempt,
    )

    root = tmp_path_factory.mktemp("stage4b-actual-ppo")
    inputs, dataset, publication = _actual_ppo_bundle_case(tmp_path_factory)
    provenance = build_candidate_run_provenance(research_context_digest="c" * 64)
    identity = replace(
        _identity(),
        expected_implementation_digest=provenance["implementation_digest"],
        expected_runtime_environment_digest=provenance["runtime_environment_digest"],
        expected_dataset_artifact_sha256=inputs["expected_dataset_artifact_sha256"],
        expected_symbols=inputs["expected_symbols"],
        ridge_artifact_sha256=inputs["expected_ridge_artifact_sha256"],
        expected_ridge_binding=inputs["expected_ridge_binding"],
        expected_bundle_sha256=publication.bundle_sha256,
        expected_manifest_sha256=publication.training_manifest_sha256,
        expected_completion_sha256=publication.training_completion_sha256,
        expected_training_context_sha256=publication.training_context_sha256,
        expected_checkpoint_sha256=publication.checkpoint_sha256,
        bound=inputs["bound"],
        start_index=inputs["start_index"],
        stop_index=inputs["stop_index"],
        initial_capital=inputs["initial_capital"],
        target_abs_weights=inputs["target_abs_weights"],
        trend_config=inputs["trend_config"],
        ridge_entry_threshold=inputs["ridge_entry_threshold"],
        ridge_exit_threshold=inputs["ridge_exit_threshold"],
        ridge_one_way_switch_cost=inputs["ridge_one_way_switch_cost"],
        execution_cost=inputs["execution_cost"],
        risk_config=inputs["risk_config"],
        market_order_profile=inputs["market_order_profile"],
    )
    context = FourFamilyPortfolioAttemptContext(
        identity=identity,
        dataset_artifact_root=inputs["dataset_artifact_root"],
        ridge_artifact_root=inputs["ridge_artifact_root"],
        bundle_root=inputs["bundle_root"],
    )

    observed: list[object] = []
    result_load_completion_state: list[bool] = []
    original_runner = attempt_module.run_four_family_shared_portfolio_replay
    original_result_loader = attempt_module.load_four_family_portfolio_result_artifact

    def observe_completion(**kwargs: object) -> object:
        original_observer = kwargs["on_family_completed"]

        def observer(outcome: object) -> None:
            observed.append(outcome)
            original_observer(outcome)  # type: ignore[operator]

        kwargs["on_family_completed"] = observer
        return original_runner(**kwargs)  # type: ignore[arg-type]

    def record_result_load_state(root: str | Path, **kwargs: object) -> object:
        result_load_completion_state.append(
            (Path(root) / "complete" / "record.json").exists()
        )
        return original_result_loader(root, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            attempt_module,
            "run_four_family_shared_portfolio_replay",
            observe_completion,
        )
        monkeypatch.setattr(
            attempt_module,
            "load_four_family_portfolio_result_artifact",
            record_result_load_state,
        )
        attempt = run_four_family_portfolio_attempt(
            attempt_root=root / "attempt", context=context
        )

    return {
        "attempt": attempt,
        "context": context,
        "dataset": dataset,
        "identity": identity,
        "observed": tuple(observed),
        "result_load_completion_state": tuple(result_load_completion_state),
        "root": root / "attempt",
    }


def test_actual_ppo_attempt_writes_and_strictly_reloads_complete_result(
    actual_attempt_case: dict[str, Any],
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        load_four_family_portfolio_attempt,
    )

    root = actual_attempt_case["root"]
    identity = actual_attempt_case["identity"]
    observed = actual_attempt_case["observed"]

    loaded = load_four_family_portfolio_attempt(root, expected_identity=identity)

    assert tuple(item.family for item in observed) == (
        "cash",
        "fixed_trend",
        "ridge_forecast",
        "joint_ppo",
    )
    assert loaded.attempt.root == root
    assert (
        loaded.result.artifact_digest
        == actual_attempt_case["attempt"].result_artifact_digest
    )
    assert (
        loaded.result.payload["dataset"]["dataset_id"]
        == actual_attempt_case["dataset"].dataset_id
    )
    assert actual_attempt_case["result_load_completion_state"] == (False, True)
    assert (root / "started" / "record.json").is_file()
    assert (root / "families" / "cash" / "record.json").is_file()
    assert (root / "families" / "joint_ppo" / "record.json").is_file()
    assert (root / "result" / "manifest.json").is_file()
    assert (root / "complete" / "record.json").is_file()
    assert not (root / "failure" / "record.json").exists()


def test_completed_arm_is_preserved_if_a_later_arm_interruption_occurs(
    actual_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_attempt as attempt_module
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        run_four_family_portfolio_attempt,
    )

    outcomes = actual_attempt_case["observed"]
    context = actual_attempt_case["context"]
    root = tmp_path / "interrupted-attempt"
    invoked: list[str] = []

    def interrupt_after_first_family(**kwargs: object) -> None:
        invoked.append("comparison")
        observer = kwargs["on_family_completed"]
        observer(outcomes[0])  # type: ignore[operator]
        raise KeyboardInterrupt("synthetic stop after first completed family")

    monkeypatch.setattr(
        attempt_module,
        "run_four_family_shared_portfolio_replay",
        interrupt_after_first_family,
    )

    with pytest.raises(KeyboardInterrupt, match="synthetic stop"):
        run_four_family_portfolio_attempt(attempt_root=root, context=context)

    assert invoked == ["comparison"]
    assert (root / "families" / "cash" / "record.json").is_file()
    assert not (root / "families" / "fixed_trend" / "record.json").exists()
    assert not (root / "result" / "manifest.json").exists()
    assert not (root / "complete" / "record.json").exists()
    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["exception_type"] == "KeyboardInterrupt"
    assert failure["phase"] == "four_family_comparison"
    assert failure["completed_families"] == ["cash"]


def test_family_row_write_failure_keeps_prior_rows_and_failure_phase(
    actual_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_attempt as attempt_module
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        run_four_family_portfolio_attempt,
    )

    outcomes = actual_attempt_case["observed"]
    context = actual_attempt_case["context"]
    root = tmp_path / "row-write-failure"
    callback_families: list[str] = []
    original_publish = attempt_module._publish_record

    def fail_fixed_trend_row(path: Path, record: object) -> bytes:
        if path.parent.name == "fixed_trend":
            raise OSError("synthetic family row write failure")
        return original_publish(path, record)

    def emit_until_callback_fails(**kwargs: object) -> object:
        observer = kwargs["on_family_completed"]
        for outcome in outcomes:
            callback_families.append(outcome.family)
            observer(outcome)  # type: ignore[operator]
        raise AssertionError("failed callback must stop before the next family")

    monkeypatch.setattr(attempt_module, "_publish_record", fail_fixed_trend_row)
    monkeypatch.setattr(
        attempt_module,
        "run_four_family_shared_portfolio_replay",
        emit_until_callback_fails,
    )

    with pytest.raises(OSError, match="synthetic family row write failure"):
        run_four_family_portfolio_attempt(attempt_root=root, context=context)

    assert callback_families == ["cash", "fixed_trend"]
    assert (root / "families" / "cash" / "record.json").is_file()
    assert not (root / "families" / "fixed_trend" / "record.json").exists()
    assert not (root / "families" / "ridge_forecast" / "record.json").exists()
    assert not (root / "families" / "joint_ppo" / "record.json").exists()
    assert not (root / "result" / "manifest.json").exists()
    assert not (root / "complete" / "record.json").exists()
    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["phase"] == "four_family_comparison"
    assert failure["completed_families"] == ["cash"]


def test_failure_journal_error_does_not_replace_original_callback_error(
    actual_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_attempt as attempt_module
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        run_four_family_portfolio_attempt,
    )

    outcomes = actual_attempt_case["observed"]
    context = actual_attempt_case["context"]
    root = tmp_path / "failure-journal-write-error"
    original_publish = attempt_module._publish_record

    def fail_both_writes(path: Path, record: object) -> bytes:
        if path.parent.name == "fixed_trend":
            raise OSError("primary family row error")
        if path.parent.name == "failure":
            raise OSError("secondary failure journal error")
        return original_publish(path, record)

    def emit_until_callback_fails(**kwargs: object) -> object:
        observer = kwargs["on_family_completed"]
        observer(outcomes[0])  # type: ignore[operator]
        observer(outcomes[1])  # type: ignore[operator]
        raise AssertionError("failed callback must stop before the next family")

    monkeypatch.setattr(attempt_module, "_publish_record", fail_both_writes)
    monkeypatch.setattr(
        attempt_module,
        "run_four_family_shared_portfolio_replay",
        emit_until_callback_fails,
    )

    with pytest.raises(OSError, match="primary family row error") as captured:
        run_four_family_portfolio_attempt(attempt_root=root, context=context)

    assert any(
        "secondary failure journal error" in note for note in captured.value.__notes__
    )
    assert (root / "families" / "cash" / "record.json").is_file()
    assert not (root / "families" / "fixed_trend" / "record.json").exists()
    assert not (root / "failure" / "record.json").exists()


def test_failed_final_attempt_reload_keeps_complete_but_is_not_loadable(
    actual_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_attempt as attempt_module
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        load_four_family_portfolio_attempt,
        run_four_family_portfolio_attempt,
    )

    root = tmp_path / "final-attempt-reload-failure"
    context = actual_attempt_case["context"]

    def reject_final_reload(*args: object, **kwargs: object) -> object:
        raise ValueError("synthetic final attempt reload failure")

    monkeypatch.setattr(
        attempt_module, "load_four_family_portfolio_attempt", reject_final_reload
    )

    with pytest.raises(ValueError, match="synthetic final attempt reload failure"):
        run_four_family_portfolio_attempt(attempt_root=root, context=context)

    assert (root / "complete" / "record.json").is_file()
    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["phase"] == "strict_attempt_reload"
    assert "completion_marker_note" in failure
    with pytest.raises(ValueError, match="failed attempt"):
        load_four_family_portfolio_attempt(root, expected_identity=context.identity)


def test_expected_implementation_mismatch_stops_before_stage4a_input_loading(
    actual_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_attempt as attempt_module
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        run_four_family_portfolio_attempt,
    )

    context = actual_attempt_case["context"]
    mismatched = replace(
        context,
        identity=replace(context.identity, expected_implementation_digest="f" * 64),
    )
    calls: list[str] = []

    def must_not_load_inputs(**kwargs: object) -> object:
        calls.append("stage4a")
        raise AssertionError("artifact/replay loader ran before provenance pin check")

    monkeypatch.setattr(
        attempt_module,
        "run_four_family_shared_portfolio_replay",
        must_not_load_inputs,
    )

    with pytest.raises(ValueError, match="implementation"):
        run_four_family_portfolio_attempt(
            attempt_root=tmp_path / "provenance-mismatch", context=mismatched
        )

    assert calls == []


def test_existing_attempt_root_is_rejected_before_provenance_or_input_load(
    actual_attempt_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_attempt as attempt_module
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        run_four_family_portfolio_attempt,
    )

    def must_not_build_provenance(**kwargs: object) -> object:
        raise AssertionError("existing attempt was not rejected before preflight")

    monkeypatch.setattr(
        attempt_module, "build_candidate_run_provenance", must_not_build_provenance
    )

    with pytest.raises(FileExistsError):
        run_four_family_portfolio_attempt(
            attempt_root=actual_attempt_case["root"],
            context=actual_attempt_case["context"],
        )


def test_strict_load_rejects_a_tampered_completed_family_row(
    actual_attempt_case: dict[str, Any],
    tmp_path: Path,
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        load_four_family_portfolio_attempt,
    )

    copied = tmp_path / "tampered-copy"
    shutil.copytree(actual_attempt_case["root"], copied)
    family_row = copied / "families" / "cash" / "record.json"
    family_row.write_bytes(family_row.read_bytes() + b" ")

    with pytest.raises(ValueError, match="digest|canonical|size"):
        load_four_family_portfolio_attempt(
            copied,
            expected_identity=actual_attempt_case["identity"],
        )
