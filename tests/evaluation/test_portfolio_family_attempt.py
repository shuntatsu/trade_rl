from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
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
    comparison_results: list[object] = []
    result_load_completion_state: list[bool] = []
    original_runner = attempt_module.run_four_family_shared_portfolio_replay
    original_result_loader = attempt_module.load_four_family_portfolio_result_artifact

    def observe_completion(**kwargs: object) -> object:
        original_observer = kwargs["on_family_completed"]

        def observer(outcome: object) -> None:
            observed.append(outcome)
            original_observer(outcome)  # type: ignore[operator]

        kwargs["on_family_completed"] = observer
        result = original_runner(**kwargs)  # type: ignore[arg-type]
        comparison_results.append(result)
        return result

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
        "comparison": comparison_results[0],
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

    loaded = load_four_family_portfolio_attempt(
        root,
        expected_identity=identity,
        expected_result_digest=actual_attempt_case["attempt"].result_artifact_digest,
    )

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


def test_public_attempt_loader_requires_external_result_digest(
    actual_attempt_case: dict[str, Any],
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        load_four_family_portfolio_attempt,
    )

    with pytest.raises(ValueError, match="result digest differs from expected"):
        load_four_family_portfolio_attempt(
            actual_attempt_case["root"],
            expected_identity=actual_attempt_case["identity"],
            expected_result_digest="e" * 64,
        )


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


def test_wrong_callback_family_order_fails_before_persisting_the_row(
    actual_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_attempt as attempt_module
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        run_four_family_portfolio_attempt,
    )

    outcomes = actual_attempt_case["observed"]
    root = tmp_path / "wrong-family-order"
    emitted: list[str] = []

    def emit_wrong_first_outcome(**kwargs: object) -> object:
        observer = kwargs["on_family_completed"]
        emitted.append(outcomes[1].family)
        observer(outcomes[1])  # type: ignore[operator]
        raise AssertionError("family-order failure must stop the comparison")

    monkeypatch.setattr(
        attempt_module,
        "run_four_family_shared_portfolio_replay",
        emit_wrong_first_outcome,
    )

    with pytest.raises(ValueError, match="fixed family order"):
        run_four_family_portfolio_attempt(
            attempt_root=root, context=actual_attempt_case["context"]
        )

    assert emitted == ["fixed_trend"]
    assert not (root / "families" / "cash" / "record.json").exists()
    assert not (root / "families" / "fixed_trend" / "record.json").exists()
    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["completed_families"] == []
    assert failure["phase"] == "four_family_comparison"


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


def test_mismatched_callback_dataset_is_rejected_before_next_family(
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
    root = tmp_path / "callback-dataset-mismatch"
    emitted: list[str] = []
    original_payload = attempt_module._family_payload

    def mismatch_fixed_trend_dataset(outcome: object) -> dict[str, object]:
        payload = original_payload(outcome)  # type: ignore[arg-type]
        if payload["family"] == "fixed_trend":
            ledger = dict(payload["ledger"])
            ledger["dataset_id"] = "unexpected-dataset"
            payload["ledger"] = ledger
        return payload

    def emit_outcomes(**kwargs: object) -> object:
        observer = kwargs["on_family_completed"]
        for outcome in outcomes:
            emitted.append(outcome.family)
            observer(outcome)  # type: ignore[operator]
        raise AssertionError("invalid callback row must stop before the next family")

    monkeypatch.setattr(attempt_module, "_family_payload", mismatch_fixed_trend_dataset)
    monkeypatch.setattr(
        attempt_module, "run_four_family_shared_portfolio_replay", emit_outcomes
    )

    with pytest.raises(ValueError, match="Dataset identity"):
        run_four_family_portfolio_attempt(attempt_root=root, context=context)

    assert emitted == ["cash", "fixed_trend"]
    assert (root / "families" / "cash" / "record.json").is_file()
    assert not (root / "families" / "fixed_trend" / "record.json").exists()
    assert not (root / "families" / "ridge_forecast" / "record.json").exists()
    assert not (root / "families" / "joint_ppo" / "record.json").exists()
    assert not (root / "result" / "manifest.json").exists()
    assert not (root / "complete" / "record.json").exists()
    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["phase"] == "four_family_comparison"
    assert failure["completed_families"] == ["cash"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("actual_interval_count", 0, "family interval count differs"),
        ("terminal_flat", False, "terminal flat flag differs"),
        ("no_active_orders", False, "pending-order flag differs"),
        ("termination_reason", "synthetic-termination", "termination reason differs"),
    ],
)
def test_family_payload_rejects_conflicting_native_count_and_flags(
    actual_attempt_case: dict[str, Any],
    field: str,
    value: object,
    message: str,
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        _family_payload,
        _validate_family_payload,
    )

    outcome = actual_attempt_case["observed"][0]
    identity = actual_attempt_case["identity"]
    row = _family_payload(outcome)
    row[field] = value

    with pytest.raises(ValueError, match=message):
        _validate_family_payload(row, expected_family="cash", identity=identity)


def test_family_payload_rejects_diagnostic_termination_reason_mismatch(
    actual_attempt_case: dict[str, Any],
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        _family_payload,
        _validate_family_payload,
    )

    outcome = actual_attempt_case["observed"][0]
    identity = actual_attempt_case["identity"]
    row = _family_payload(outcome)
    assert row["termination_reason"] is None
    assert row["diagnostics"]["termination_reasons"] == []
    row["diagnostics"]["termination_reasons"] = ["insolvency"]

    with pytest.raises(ValueError, match="diagnostic termination reasons differ"):
        _validate_family_payload(row, expected_family="cash", identity=identity)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing_prefix", "form a prefix from the requested replay index"),
        ("multi_bar_interval", "must advance exactly one native bar"),
        ("interior_termination", "only the final native interval may terminate"),
        ("final_reason_mismatch", "final native interval termination differs"),
    ],
)
def test_ledger_rejects_non_native_interval_prefix_and_termination(
    actual_attempt_case: dict[str, Any], corruption: str, message: str
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        _family_payload,
        _validate_ledger_shape,
    )

    outcome = actual_attempt_case["observed"][0]
    identity = actual_attempt_case["identity"]
    family = _family_payload(outcome)
    ledger = cast(dict[str, object], family["ledger"])
    intervals = cast(list[object], ledger["intervals"])
    first = cast(dict[str, object], intervals[0])

    if corruption == "missing_prefix":
        first["start_index"] = identity.start_index + 1
        first["next_index"] = identity.start_index + 2
        intervals[:] = intervals[:1]
    elif corruption == "multi_bar_interval":
        first["next_index"] = identity.start_index + 2
        intervals[:] = intervals[:1]
    elif corruption == "interior_termination":
        ledger["termination_reason"] = "insolvency"
        first["termination_reason"] = "insolvency"
    else:
        ledger["termination_reason"] = "insolvency"

    with pytest.raises(ValueError, match=message):
        _validate_ledger_shape(ledger, identity=identity)


def test_native_insolvency_is_retained_as_an_incomplete_attempt_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.evaluation.test_portfolio_family_comparison import (
        _FixedPPOPolicy,
        _prepared_inputs,
    )
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        FourFamilyPortfolioAttemptContext,
        load_four_family_portfolio_attempt,
        run_four_family_portfolio_attempt,
    )

    inputs, _ = _prepared_inputs(tmp_path, force_insolvency=True)
    monkeypatch.setattr(
        comparison_module,
        "load_portfolio_inference_bundle",
        lambda *args, **kwargs: _FixedPPOPolicy(),
    )
    provenance = build_candidate_run_provenance(research_context_digest="c" * 64)
    identity = replace(
        _identity(),
        expected_implementation_digest=provenance["implementation_digest"],
        expected_runtime_environment_digest=provenance["runtime_environment_digest"],
        expected_dataset_artifact_sha256=inputs["expected_dataset_artifact_sha256"],
        expected_symbols=inputs["expected_symbols"],
        ridge_artifact_sha256=inputs["expected_ridge_artifact_sha256"],
        expected_ridge_binding=inputs["expected_ridge_binding"],
        expected_bundle_sha256=inputs["expected_bundle_sha256"],
        expected_manifest_sha256=inputs["expected_manifest_sha256"],
        expected_completion_sha256=inputs["expected_completion_sha256"],
        expected_training_context_sha256=inputs["expected_training_context_sha256"],
        expected_checkpoint_sha256=inputs["expected_checkpoint_sha256"],
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

    attempt = run_four_family_portfolio_attempt(
        attempt_root=tmp_path / "insolvency-attempt", context=context
    )
    loaded = load_four_family_portfolio_attempt(
        attempt.root,
        expected_identity=identity,
        expected_result_digest=attempt.result_artifact_digest,
    )
    rows = {
        family: json.loads(
            (attempt.root / "families" / family / "record.json").read_text("utf-8")
        )
        for family in ("cash", "fixed_trend", "ridge_forecast", "joint_ppo")
    }

    assert len(loaded.result.payload["families"]) == 4
    retained = [row for row in rows.values() if row["insolvent"]]
    assert retained
    for row in retained:
        assert row["window_complete"] is False
        assert row["termination_reason"] == "insolvency"
        assert row["diagnostics"]["termination_reasons"] == ["insolvency"]
        assert row["actual_interval_count"] < row["expected_interval_count"]
        assert row["book"]["insolvent"] is True

    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        _validate_family_payload,
    )

    inconsistent = json.loads(json.dumps(retained[0]))
    inconsistent["diagnostics"]["termination_reasons"] = []
    with pytest.raises(ValueError, match="diagnostic termination reasons differ"):
        _validate_family_payload(
            inconsistent,
            expected_family=inconsistent["family"],
            identity=identity,
        )


@pytest.mark.parametrize("annual_rate", (-0.05, 0.05))
def test_signed_cash_carry_survives_four_family_attempt_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    annual_rate: float,
) -> None:
    from tests.evaluation.test_portfolio_family_comparison import (
        _FixedPPOPolicy,
        _prepared_inputs,
    )
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        FourFamilyPortfolioAttemptContext,
        load_four_family_portfolio_attempt,
        run_four_family_portfolio_attempt,
    )

    inputs, _ = _prepared_inputs(
        tmp_path, cash_rate=np.full((7,), annual_rate, dtype=np.float64)
    )
    monkeypatch.setattr(
        comparison_module,
        "load_portfolio_inference_bundle",
        lambda *args, **kwargs: _FixedPPOPolicy(),
    )
    provenance = build_candidate_run_provenance(research_context_digest="c" * 64)
    identity = replace(
        _identity(),
        expected_implementation_digest=provenance["implementation_digest"],
        expected_runtime_environment_digest=provenance["runtime_environment_digest"],
        expected_dataset_artifact_sha256=inputs["expected_dataset_artifact_sha256"],
        expected_symbols=inputs["expected_symbols"],
        ridge_artifact_sha256=inputs["expected_ridge_artifact_sha256"],
        expected_ridge_binding=inputs["expected_ridge_binding"],
        expected_bundle_sha256=inputs["expected_bundle_sha256"],
        expected_manifest_sha256=inputs["expected_manifest_sha256"],
        expected_completion_sha256=inputs["expected_completion_sha256"],
        expected_training_context_sha256=inputs["expected_training_context_sha256"],
        expected_checkpoint_sha256=inputs["expected_checkpoint_sha256"],
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

    attempt = run_four_family_portfolio_attempt(
        attempt_root=tmp_path / "signed-cash-attempt", context=context
    )
    load_four_family_portfolio_attempt(
        attempt.root,
        expected_identity=identity,
        expected_result_digest=attempt.result_artifact_digest,
    )
    cash = json.loads(
        (attempt.root / "families" / "cash" / "record.json").read_text("utf-8")
    )
    ledger = cash["ledger"]
    intervals = ledger["intervals"]
    returns = cash["returns"]["values"]

    assert cash["window_complete"] is True
    assert cash["terminal_flat"] is True
    assert cash["no_active_orders"] is True
    assert cash["actual_interval_count"] == cash["expected_interval_count"]
    assert cash["book"]["fill_count"] == 0
    assert cash["book"]["total_cost"] == 0.0
    assert cash["book"]["turnover_total"] == 0.0
    assert ledger["active_order_remainders"] == []
    assert ledger["terminal_exact_quantities"] == ["0", "0"]
    assert ledger["final_total_cost"] == 0.0
    assert ledger["final_funding_pnl"] == 0.0
    assert ledger["final_borrow_cost"] == 0.0
    assert len(intervals) == len(returns) == cash["actual_interval_count"]

    for interval, realized_return in zip(intervals, returns, strict=True):
        interest = interval["interval_cash_interest"]
        assert interest * annual_rate > 0.0
        assert interval["interval_cost"] == 0.0
        assert interval["interval_funding"] == 0.0
        assert interval["interval_borrow_cost"] == 0.0
        assert interval["cash_before"] == pytest.approx(
            interval["portfolio_value_before"], rel=0.0, abs=1e-10
        )
        assert interval["cash_after"] == pytest.approx(
            interval["portfolio_value_after"], rel=0.0, abs=1e-10
        )
        assert interval["cash_after"] - interval["cash_before"] == pytest.approx(
            interest, rel=0.0, abs=1e-10
        )
        assert realized_return == pytest.approx(
            interval["interval_net_return"], rel=0.0, abs=1e-12
        )
        assert realized_return == pytest.approx(
            interest / interval["portfolio_value_before"],
            rel=0.0,
            abs=1e-12,
        )

    total_interest = math.fsum(
        interval["interval_cash_interest"] for interval in intervals
    )
    cash_change = ledger["final_cash"] - identity.initial_capital
    assert total_interest == pytest.approx(cash_change, rel=0.0, abs=1e-10)
    assert ledger["final_portfolio_value"] == pytest.approx(
        ledger["final_cash"], rel=0.0, abs=1e-10
    )
    assert cash_change * annual_rate > 0.0
    assert math.prod(1.0 + value for value in returns) - 1.0 == pytest.approx(
        cash_change / identity.initial_capital, rel=0.0, abs=1e-12
    )


@pytest.mark.parametrize("missing_or_extra", ["missing", "extra"])
def test_public_attempt_loader_rejects_incomplete_or_extra_family_inventory(
    actual_attempt_case: dict[str, Any],
    tmp_path: Path,
    missing_or_extra: str,
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        load_four_family_portfolio_attempt,
    )

    copied = tmp_path / f"{missing_or_extra}-family-inventory"
    shutil.copytree(actual_attempt_case["root"], copied)
    if missing_or_extra == "missing":
        (copied / "families" / "ridge_forecast" / "record.json").unlink()
    else:
        (copied / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="missing|unexpected"):
        load_four_family_portfolio_attempt(
            copied,
            expected_identity=actual_attempt_case["identity"],
            expected_result_digest=actual_attempt_case[
                "attempt"
            ].result_artifact_digest,
        )


def test_public_attempt_loader_rejects_symlinked_tree_entry(
    actual_attempt_case: dict[str, Any], tmp_path: Path
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        load_four_family_portfolio_attempt,
    )

    copied = tmp_path / "symlinked-attempt"
    shutil.copytree(actual_attempt_case["root"], copied)
    link = copied / "unexpected-link.json"
    try:
        os.symlink(copied / "started" / "record.json", link)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable on this platform: {error}")

    with pytest.raises(ValueError, match="symlink"):
        load_four_family_portfolio_attempt(
            copied,
            expected_identity=actual_attempt_case["identity"],
            expected_result_digest=actual_attempt_case[
                "attempt"
            ].result_artifact_digest,
        )


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
        load_four_family_portfolio_attempt(
            root,
            expected_identity=context.identity,
            expected_result_digest="e" * 64,
        )


def test_after_provenance_drift_keeps_rows_but_does_not_publish_result(
    actual_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_attempt as attempt_module
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        run_four_family_portfolio_attempt,
    )

    root = tmp_path / "after-provenance-drift"
    outcomes = actual_attempt_case["observed"]

    def emit_valid_comparison(**kwargs: object) -> object:
        observer = kwargs["on_family_completed"]
        for outcome in outcomes:
            observer(outcome)  # type: ignore[operator]
        return actual_attempt_case["comparison"]

    original_provenance = attempt_module.build_candidate_run_provenance
    provenance_calls = 0

    def drift_runtime_provenance(*, research_context_digest: str) -> dict[str, object]:
        nonlocal provenance_calls
        provenance_calls += 1
        provenance = original_provenance(
            research_context_digest=research_context_digest
        )
        if provenance_calls == 2:
            return {**provenance, "runtime_environment_digest": "f" * 64}
        return dict(provenance)

    monkeypatch.setattr(
        attempt_module,
        "run_four_family_shared_portfolio_replay",
        emit_valid_comparison,
    )
    monkeypatch.setattr(
        attempt_module,
        "build_candidate_run_provenance",
        drift_runtime_provenance,
    )

    with pytest.raises(ValueError, match="runtime provenance"):
        run_four_family_portfolio_attempt(
            attempt_root=root, context=actual_attempt_case["context"]
        )

    assert provenance_calls == 2
    assert all(
        (root / "families" / family / "record.json").is_file()
        for family in ("cash", "fixed_trend", "ridge_forecast", "joint_ppo")
    )
    assert not (root / "provenance" / "after" / "record.json").exists()
    assert not (root / "result" / "manifest.json").exists()
    assert not (root / "complete" / "record.json").exists()
    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["phase"] == "after_provenance"
    assert failure["completed_families"] == [
        "cash",
        "fixed_trend",
        "ridge_forecast",
        "joint_ppo",
    ]


@pytest.mark.parametrize(
    ("digest_field", "message"),
    [
        ("expected_implementation_digest", "implementation"),
        ("expected_runtime_environment_digest", "runtime"),
    ],
)
def test_expected_provenance_mismatch_stops_before_stage4a_input_loading(
    actual_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    digest_field: str,
    message: str,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_attempt as attempt_module
    from trade_rl.evaluation.runs.portfolio_family_attempt import (
        run_four_family_portfolio_attempt,
    )

    context = actual_attempt_case["context"]
    mismatched = replace(
        context,
        identity=replace(context.identity, **{digest_field: "f" * 64}),
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

    with pytest.raises(ValueError, match=message):
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
            expected_result_digest=actual_attempt_case[
                "attempt"
            ].result_artifact_digest,
        )
