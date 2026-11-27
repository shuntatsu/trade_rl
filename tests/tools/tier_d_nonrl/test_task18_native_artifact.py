"""Actual native ledger publication/reopen; only closed artificial v7 inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.tools.tier_d_nonrl.test_task18_frozen_inputs import (
    _binding,
    _identity,
    _market,
)
from tools.tier_d_nonrl.task18_receipts import (
    ExternalSourceVerificationReport,
    SourceBinding,
    SourceFilePin,
    evaluate_run_receipt,
)
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from trade_rl.artifacts.canonical import canonical_json_bytes

P = build_task18_protocol()


@pytest.fixture(scope="module")
def persisted_arm(tmp_path_factory):
    from tools.tier_d_nonrl.task18_frozen_inputs import (
        prepare_frozen_inputs,
        publish_frozen_inputs,
    )
    from tools.tier_d_nonrl.task18_native import open_synthetic_native_attempt
    from tools.tier_d_nonrl.task18_native_artifact import (
        new_synthetic_run_receipt,
        publish_completed_native_arm,
        synthetic_receipt_protocol,
    )

    market = _market()
    market = _identity(
        replace(
            market,
            dataset_id="0" * 64,
            identity_payload_json=None,
            lot_size=np.full_like(market.close, 0.001),
        )
    )
    source = SourceBinding(
        "c" * 40,
        (
            SourceFilePin(
                "tests/tools/tier_d_nonrl/test_task18_native_artifact.py",
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
        ),
        hashlib.sha256(
            b"artificial-only source declaration; not authenticated"
        ).hexdigest(),
    )
    source_raw = canonical_json_bytes(
        {"scope": "synthetic fixture", "source_authentication": False}
    )
    report = ExternalSourceVerificationReport(
        source.commit_sha,
        True,
        source.files,
        source.import_origin_manifest_sha256,
        1,
        hashlib.sha256(source_raw).hexdigest(),
    )
    binding = replace(_binding(market), source_manifest_sha256=source.manifest_sha256)
    frozen = prepare_frozen_inputs(market, binding=binding)
    base = tmp_path_factory.mktemp("native-receipt").resolve()
    root = base / "original"
    pin = publish_frozen_inputs(root, frozen)
    attempt = open_synthetic_native_attempt(
        market, root / "frozen-plan.bin", expected_binding=binding, expected_pin=pin
    )
    run = new_synthetic_run_receipt(
        attempt,
        run_id="artificial-native-original",
        source_binding=source,
        source_report=report,
    )
    protocol = synthetic_receipt_protocol(attempt, source)
    assert protocol.scope == "SYNTHETIC_NATIVE_CONFORMANCE_ONLY"
    assert run.protocol_digest != P.adapted_protocol_digest
    assert evaluate_run_receipt(run, protocol).declaration_consistent
    assert not evaluate_run_receipt(run, P).declaration_consistent
    arm = replace(
        run.arms[0],
        status="RUNNING",
        reason_code="RUN_IN_PROGRESS",
        initial_equity_usdt=100000.0,
    )
    run = replace(run, arms=(arm, *run.arms[1:]))
    actual = next(attempt.iter_arms((arm.arm_id,)))
    root.joinpath("arms").mkdir()
    published = publish_completed_native_arm(
        root, attempt=attempt, run=run, observation=actual
    )
    return root, attempt, run, published, protocol


def test_actual_completed_receipt_reopens_and_only_pending_cash_executes(persisted_arm):
    from tools.tier_d_nonrl.task18_native_artifact import load_completed_native_arm

    root, attempt, run, published, protocol = persisted_arm
    old = {p.path: (root / p.path).read_bytes() for p in published.pins}
    assert len(old) == 4
    assert published.receipt.status == "COMPLETED"
    assert (
        published.receipt.native_ledger_sha256
        == hashlib.sha256(old["arms/PAIR-01-NONRL/native-ledger.json"]).hexdigest()
    )
    with (
        patch(
            "tools.tier_d_nonrl.task18_frozen_inputs.fit_simple_return_ridge",
            side_effect=AssertionError("must not fit"),
        ),
        patch(
            "tools.tier_d_nonrl.task18_frozen_inputs.build_cost_rates",
            side_effect=AssertionError("must not recost"),
        ),
        patch(
            "trade_rl.strategies.forecasts.simple_return.SimpleReturnRidgeModel.predict_simple_return",
            side_effect=AssertionError("must not predict"),
        ),
        patch(
            "tools.tier_d_nonrl.task18_native.run_shared_portfolio_replay",
            side_effect=AssertionError("must not replay on reopen"),
        ),
    ):
        loaded = load_completed_native_arm(
            root,
            attempt=attempt,
            run=run,
            arm_id=published.receipt.arm_id,
            expected_pins=tuple(reversed(published.pins)),
        )
    assert loaded.receipt == published.receipt
    assert loaded.verification == published.verification
    assert not loaded.execution_authorized
    stopped = replace(run, arms=(loaded.receipt, *run.arms[1:]))
    evaluation = evaluate_run_receipt(stopped, protocol)
    assert (
        evaluation.declaration_consistent
        and evaluation.arm_reports_match
        and evaluation.declared_metrics_well_formed
    )
    assert not evaluation.declared_runtime_completed
    # Actual native CASH run after disk-only reopen; no completed NONRL rerun.
    cash_id = stopped.arms[1].arm_id
    cash = next(attempt.iter_arms((cash_id,)))
    assert cash.arm_id == cash_id and cash.verification.passed
    assert cash.result.book.portfolio_value == 100000.0
    assert old == {p.path: (root / p.path).read_bytes() for p in published.pins}


def test_native_artifact_rejects_wrong_external_pin_before_decoder(persisted_arm):
    from tools.tier_d_nonrl import task18_native_artifact as artifact

    root, attempt, run, published, _ = persisted_arm
    pins = list(published.pins)
    pins[0] = replace(pins[0], sha256="f" * 64)
    with patch.object(
        artifact, "decode_ledger", side_effect=AssertionError("unverified decode")
    ):
        with pytest.raises(ValueError, match="digest mismatch"):
            artifact.load_completed_native_arm(
                root,
                attempt=attempt,
                run=run,
                arm_id=published.receipt.arm_id,
                expected_pins=tuple(pins),
            )


@pytest.mark.parametrize(
    "change",
    (
        "forged_receipt",
        "wrong_context",
        "bool_index",
        "unknown_field",
        "truncated_ledger",
        "verification_authority",
    ),
)
def test_native_artifact_refuses_resealed_corruption(persisted_arm, tmp_path, change):
    from tools.tier_d_nonrl.task18_native_artifact import load_completed_native_arm

    root, attempt, run, published, _ = persisted_arm
    pins = []
    for pin in published.pins:
        raw = (root / pin.path).read_bytes()
        if change == "forged_receipt" and pin.path.endswith("/receipt.json"):
            value = json.loads(raw)
            value["terminal_equity_usdt"] += 1
            raw = canonical_json_bytes(value)
        elif change == "wrong_context" and pin.path.endswith("/result.json"):
            value = json.loads(raw)
            value["run_id"] = "foreign"
            raw = canonical_json_bytes(value)
        elif change == "verification_authority" and pin.path.endswith(
            "/verification.json"
        ):
            value = json.loads(raw)
            value["execution_authorized"] = True
            raw = canonical_json_bytes(value)
        elif change in (
            "bool_index",
            "unknown_field",
            "truncated_ledger",
        ) and pin.path.endswith("/native-ledger.json"):
            if change == "truncated_ledger":
                raw = raw[:-1]
            else:
                value = json.loads(raw)
                if change == "bool_index":
                    value["intervals"][0]["start_index"] = True
                else:
                    value["intervals"][0]["unknown"] = 0
                raw = canonical_json_bytes(value)
        dest = tmp_path / pin.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        pins.append(
            replace(pin, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        )
    with pytest.raises(ValueError):
        load_completed_native_arm(
            tmp_path,
            attempt=attempt,
            run=run,
            arm_id=published.receipt.arm_id,
            expected_pins=tuple(pins),
        )


def test_native_artifact_refuses_foreign_run_and_completed_redeclaration(persisted_arm):
    from tools.tier_d_nonrl.task18_native_artifact import load_completed_native_arm

    root, attempt, run, published, _ = persisted_arm
    for invalid in (
        replace(run, run_id="foreign-run"),
        replace(
            run,
            arms=(
                replace(published.receipt, terminal_equity_usdt=123.0),
                *run.arms[1:],
            ),
        ),
    ):
        with pytest.raises(ValueError):
            load_completed_native_arm(
                root,
                attempt=attempt,
                run=invalid,
                arm_id=published.receipt.arm_id,
                expected_pins=published.pins,
            )


def test_late_completed_artifact_must_match_observed_partial_frontier(persisted_arm):
    from tools.tier_d_nonrl.task18_native_artifact import load_completed_native_arm
    from tools.tier_d_nonrl.task18_receipts import ArmAttemptProgress

    root, attempt, run, published, _ = persisted_arm
    ledger = json.loads(
        (root / "arms" / published.receipt.arm_id / "native-ledger.json").read_bytes()
    )
    interval = ledger["intervals"][63]
    progress = ArmAttemptProgress(
        64,
        interval["next_index"],
        interval["portfolio_value_after"],
        interval["max_drawdown_after"],
    )
    running = replace(run.arms[0], progress=progress)
    correct = replace(run, arms=(running, *run.arms[1:]))
    assert (
        load_completed_native_arm(
            root,
            attempt=attempt,
            run=correct,
            arm_id=running.arm_id,
            expected_pins=published.pins,
        ).receipt
        == published.receipt
    )
    bad = replace(
        correct,
        arms=(
            replace(
                running,
                progress=replace(
                    progress,
                    last_observed_equity_usdt=progress.last_observed_equity_usdt + 1.0,
                ),
            ),
            *run.arms[1:],
        ),
    )
    with pytest.raises(ValueError, match="contradicts observed"):
        load_completed_native_arm(
            root,
            attempt=attempt,
            run=bad,
            arm_id=running.arm_id,
            expected_pins=published.pins,
        )


def test_native_v1_decision_closure_survives_new_replay_modes():
    from tools.tier_d_nonrl.task18_native_decode import _decision
    from trade_rl.evaluation.replay import SharedCashPortfolioReplayDecision

    count = len(P.dataset.symbols)
    original = SharedCashPortfolioReplayDecision(
        index=0,
        proposal_weights=(0.0,) * count,
        target_weights=(0.0,) * count,
        risk_reasons=(),
        requested_hold_symbols=tuple(range(count)),
        effective_hold_symbols=tuple(range(count)),
    )
    raw = json.loads(canonical_json_bytes(original.to_mapping()))
    assert _decision(raw) == original
    with pytest.raises(ValueError, match="portfolio decision field closure differs"):
        _decision(raw | {"pending_continuation_status": None})
