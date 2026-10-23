from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.tools.market_b4_oracle.test_frozen_h_witness import _frozen_h_plan


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _component_spec() -> dict[str, object]:
    return {
        "schema": "issue810_artificial_h_component_spec_v1",
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "dataset_label": "synthetic-issue810-component-v1",
        "dataset_identity_binding": "computed_by_pinned_H_MarketDatasetBuilder",
        "protocol_id": "synthetic-issue810-component-protocol-v1",
        "configuration_id": "H24_RIDGE_A1",
        "symbol": "BTCUSDT",
        "scenario": "BASEL0",
        "actor_start": 2903,
        "actor_stop_exclusive": 9430,
        "actor_advances": 6527,
        "reserve_advances": 72,
        "role_pair": ["NONRL", "CASH"],
        "row_count": 9504,
        "pulse_indices": [
            2905,
            3995,
            5082,
            5089,
            6191,
            7290,
            7297,
            8362,
            9425,
            9429,
        ],
        "feature_names": [
            "fixture_pulse",
            "fixture_log_return",
            "fixture_body_return",
            "fixture_volume_change",
            "fixture_close_location",
            "fixture_lower_wick",
        ],
        "dataset_recipe": {
            "timestamp_origin_utc": "2024-01-01T00:00:00Z",
            "interval_seconds": 3600,
            "base_timeframe": "1h",
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "volume_unit": "QUOTE_NOTIONAL",
            "availability": "exact_row_timestamp",
            "tradable": True,
            "ohlc": {
                "open": 100.0,
                "close": 100.0,
                "ordinary_high": 101.0,
                "ordinary_low": 99.0,
                "selected_pulse_high": 151.0,
                "selected_pulse_low": 49.0,
                "pulse_symbol": "BTCUSDT",
            },
            "quote_volume_per_bar": 10000000.0,
            "funding_rate": 0.0,
            "funding_price_rate": 0.0,
            "funding_event_stride": 8,
            "instrument_rules": {
                "BTCUSDT": {
                    "tick_size": 0.1,
                    "lot_size": 0.001,
                    "minimum_notional": 50.0,
                },
                "ETHUSDT": {
                    "tick_size": 0.01,
                    "lot_size": 0.001,
                    "minimum_notional": 20.0,
                },
            },
            "feature_kinds": [
                "HIGH_LOW_RANGE",
                "LOG_RETURN",
                "BODY_RETURN",
                "VOLUME_LOG_CHANGE",
                "CLOSE_LOCATION_VALUE",
                "LOWER_WICK_RATIO",
            ],
        },
        "account_recipe": {
            "initial_cash": 100000.0,
            "initial_quantities": [0.0, 0.0],
            "selected_symbol": "BTCUSDT",
            "other_symbol_always_zero": True,
        },
        "cost_recipe": {
            "fee_rate": 0.0005,
            "spread_rate": 0.0002,
            "max_participation_rate": 0.05,
            "borrow_available": True,
            "borrow_rate": 0.0,
            "market_fee_overlay": 0.0,
            "market_spread_overlay": 0.0,
            "impact_coefficient": 0.0001,
            "cash_rate": 0.0,
            "funding_rate": 0.0,
            "stochastic_slippage": 0.0,
            "random_seed": 0,
            "reference_notional": 100000.0,
            "funding_lookback_hours": 24,
            "funding_minimum_event_count": 2,
            "funding_maximum_counted_close_gap_hours": 9.0,
            "funding_maximum_counted_close_age_hours": 8.0,
        },
        "risk_recipe": {
            "max_gross": 1.0,
            "max_abs_weight": 1.0,
            "drawdown_start": 0.1,
            "drawdown_stop": 0.2,
            "emergency_turnover_override": True,
            "fail_closed_tolerance": 1e-10,
        },
        "allocator_recipe": {
            "lower_weight": -1.0,
            "upper_weight": 1.0,
            "max_turnover": None,
            "risk_aversion": 0.0,
        },
        "forecast_recipe": {
            "method": "fixed_projection_from_stored_float32_inputs_no_fit",
            "fit_called": False,
            "horizon_hours": 24,
            "ridge_alpha": 1.0,
            "vintage_starts": [2903, 5087, 7295],
            "feature_indices": [0, 1, 2, 3, 4, 5],
            "fit_symbol_indices": [0],
            "prediction_symbols": ["BTCUSDT", "ETHUSDT"],
            "training_trace_only": True,
            "mean": "first_feature_at_actor_start_then_zero",
            "scale": "ones",
            "intercept": -0.02,
            "pulse_expected_simple_return": 0.02,
            "nonpulse_expected_simple_return": -0.02,
        },
        "execution_recipe": {
            "order_type": "MARKET",
            "time_in_force": "GTC",
            "latency_bars": 0,
            "capacity_source": "prior_bar_quote_notional_volume",
            "market_fill_price_source": "next_processing_row_open",
            "decision_mark_source": "decision_row_close",
            "actor_stop_exclusive": 9430,
            "reserve_stop_inclusive": 9502,
            "row_9503_used": False,
        },
        "full_b4_authorized": False,
        "outer_attempt_authenticated": False,
    }


def test_fixed_component_row_counts_match_actor_and_reserve_bounds() -> None:
    import tools.market_b4_artificial_component_operator as operator

    assert operator._ACTOR_ROWS == 6527
    assert operator._RESERVE_ROWS == 72
    assert operator._ACTOR_ROWS + operator._RESERVE_ROWS == 6599


def test_fixed_forecast_recipe_flips_after_cost_target_from_opposite_holding() -> None:
    """Prove each vintage's fixed signal overcomes its pinned execution costs."""
    import tools.market_b4_oracle.artificial_component as artificial
    from tools.market_b4_oracle.frozen_decision import compose_frozen_allocation_target
    from tools.market_b4_oracle.frozen_forecast_index import FrozenForecastIndex
    from tools.market_b4_oracle.frozen_market_replay import FrozenMarketReplay

    spec = _component_spec()
    inputs = artificial.build_artificial_component_inputs(spec)
    binding, policy = artificial.artificial_component_binding(inputs, "NONRL")
    forecast_bytes = inputs.forecast_payload_canonical
    forecast = FrozenForecastIndex(
        forecast_bytes,
        pinned_bytes=len(forecast_bytes),
        pinned_sha256=hashlib.sha256(forecast_bytes).hexdigest(),
    )
    replay = FrozenMarketReplay(
        content=inputs.content,
        policy=policy,
        binding=binding,
        reserve_plan_digest="b" * 64,
    )

    # Each fixed vintage starts with a nonpulse baseline; the adjacent chosen
    # pulse lies inside that vintage. The last pulse is the actor's final
    # decision and therefore must fill on processing row 9430.
    cases = (
        (2903, 1.0, -0.02),
        (2905, -1.0, 0.02),
        (5087, 1.0, -0.02),
        (5089, -1.0, 0.02),
        (7295, 1.0, -0.02),
        (7297, -1.0, 0.02),
        (9429, -1.0, 0.02),
    )
    for decision_index, held_weight, expected_return in cases:
        source = replay._decision_snapshot(decision_index)
        costs, _ = replay._derive_cost_assumptions(
            source, decision_index=decision_index
        )
        assumptions = forecast.lookup(source, cost_assumptions=costs)

        # With the fixed equal buy/sell/exit recipe, each one-way rate is
        # 0.00071. The return magnitude .02 exceeds the largest combined
        # two-cost slope bound (.00142), so the unique after-cost target is
        # the signal's endpoint regardless of the opposite current holding.
        assert costs.buy_cost == costs.sell_cost == costs.exit_cost == 0.00071
        assert abs(assumptions.expected_simple_return) > (
            2.0 * max(costs.buy_cost, costs.sell_cost, costs.exit_cost)
        )
        assert assumptions.expected_simple_return == expected_return
        target = compose_frozen_allocation_target(
            assumptions,
            current_weight=held_weight,
            lower_weight=-1.0,
            upper_weight=1.0,
            drawdown=0.0,
            role="NONRL",
        )
        assert target.baseline_weight == (1.0 if expected_return > 0.0 else -1.0)


def test_q_transition_oracle_proves_pair_replay_fills_and_boundary_close() -> None:
    """Replay every fixed actor/reserve row for both roles without the H child."""
    import tools.market_b4_oracle.artificial_component as artificial
    from tools.market_b4_oracle.frozen_forecast_index import FrozenForecastIndex
    from tools.market_b4_oracle.frozen_market_replay import FrozenMarketReplay

    spec = _component_spec()
    inputs = artificial.build_artificial_component_inputs(spec)
    forecast_bytes = inputs.forecast_payload_canonical
    forecast = FrozenForecastIndex(
        forecast_bytes,
        pinned_bytes=len(forecast_bytes),
        pinned_sha256=hashlib.sha256(forecast_bytes).hexdigest(),
    )
    for role in ("NONRL", "CASH"):
        binding, policy = artificial.artificial_component_binding(inputs, role)
        replay = FrozenMarketReplay(
            content=inputs.content,
            policy=policy,
            binding=binding,
            reserve_plan_digest=("a" if role == "NONRL" else "c") * 64,
        )
        actor_fill_indices: list[int] = []
        reserve_fill_indices: list[int] = []
        for _ in range(binding.window.actor_advances + binding.window.reserve_advances):
            advance = replay.advance(forecast_index=forecast, role=role)
            if advance.owned_fills:
                target = (
                    actor_fill_indices
                    if advance.phase == "actor"
                    else reserve_fill_indices
                )
                target.append(advance.processing_index)

        completion = replay.completion()
        assert completion["status"] == "comparison_window_consumed"
        assert completion["processed_actor_advances"] == 6527
        assert completion["processed_reserve_advances"] == 72
        assert completion["terminal_flat"] is True
        assert completion["no_active_orders"] is True
        if role == "CASH":
            assert actor_fill_indices == []
            assert reserve_fill_indices == []
        else:
            assert all(
                required in actor_fill_indices for required in (2906, 5090, 7298, 9430)
            )
            assert 9431 in reserve_fill_indices


def test_artificial_inputs_do_not_expose_mutable_authority() -> None:
    import tools.market_b4_oracle.artificial_component as artificial

    inputs = artificial.build_artificial_component_inputs(_component_spec())
    volume_units = inputs.content.dataset["volume_units"]
    original_volume_unit = volume_units[0]
    try:
        volume_units[0] = "MUTATED"
    except TypeError:
        pass
    else:
        volume_units[0] = original_volume_unit
        pytest.fail("Q Dataset metadata must be recursively immutable")

    protocol_view = inputs.protocol_document
    protocol_view["scope"] = "MUTATED"
    forecast_view = inputs.forecast_payload
    forecast_view["schema"] = "MUTATED"

    assert inputs.protocol_document["scope"] == "ARTIFICIAL_COMPARISON_ONLY"
    assert inputs.forecast_payload["schema"] == "simple_return_ridge_stream_v1"


def test_coherently_rehashed_artificial_forecast_fails_before_receipt_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_oracle.artificial_component as artificial
    import tools.market_b4_oracle.offline_replay_consumer as consumer
    from tools.market_b4_oracle.decision_identity import canonical_json_bytes
    from tools.market_b4_oracle.output_index import (
        capture_output_manifest_pin,
        load_pinned_output_index,
    )

    inputs = artificial.build_artificial_component_inputs(_component_spec())
    binding, policy = artificial.artificial_component_binding(inputs, "NONRL")
    output_root = tmp_path / "output"
    forecast_path = output_root / "forecasts" / "simple-return-stream.json"
    forecast_path.parent.mkdir(parents=True)

    def write_index(raw: bytes):
        forecast_path.write_bytes(raw)
        (output_root / "output-manifest.json").write_bytes(
            _canonical(
                {
                    "forecasts/simple-return-stream.json": {
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                }
            )
        )
        return load_pinned_output_index(
            output_root, capture_output_manifest_pin(output_root)
        )

    def fail_if_receipt_reader_reached(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("receipt scan reached after exact forecast validation")

    valid_index = write_index(inputs.forecast_payload_canonical)
    monkeypatch.setattr(
        type(valid_index), "receipt_rows", fail_if_receipt_reader_reached
    )
    with pytest.raises(AssertionError, match="receipt scan reached"):
        consumer.consume_indexed_market_arm(
            output_index=valid_index,
            receipt_name="receipts/nonrl.json.gz",
            forecast_name="forecasts/simple-return-stream.json",
            content=inputs.content,
            policy=policy,
            binding=binding,
            role="NONRL",
            reserve_plan_digest="a" * 64,
            artificial_inputs=inputs,
        )

    payload = json.loads(inputs.forecast_payload_canonical)
    vintage_digests: dict[str, str] = {}
    models: dict[str, dict[str, object]] = {}
    for vintage in payload["vintages"]:
        old_digest = vintage["digest"]
        model = vintage["model"]
        model["coefficients"][0] = float(model["coefficients"][0]) + 0.125
        body = {key: value for key, value in vintage.items() if key != "digest"}
        new_digest = artificial._stream_packet_digest(body)
        vintage_digests[old_digest] = new_digest
        models[new_digest] = model
        vintage["digest"] = new_digest
    for packet in payload["packets"]:
        old_digest = packet["vintage_digest"]
        new_digest = vintage_digests[old_digest]
        model = models[new_digest]
        feature_values = packet["feature_values"]
        expected_return = float(model["intercept"])
        for value, mean, scale, coefficient in zip(
            feature_values,
            model["feature_mean"],
            model["feature_scale"],
            model["coefficients"],
            strict=True,
        ):
            expected_return += (
                (float(value) - float(mean)) / float(scale) * float(coefficient)
            )
        packet["expected_simple_return"] = expected_return
        packet["vintage_digest"] = new_digest
        body = {key: value for key, value in packet.items() if key != "digest"}
        packet["digest"] = artificial._stream_packet_digest(body)
    causal_content = {key: payload[key] for key in ("vintages", "packets")}
    payload["causal_scope_digest"] = artificial._stream_packet_digest(causal_content)

    forecast_raw = canonical_json_bytes(payload)
    output_index = write_index(forecast_raw)
    with pytest.raises(ValueError, match="fixed recipe"):
        consumer.consume_indexed_market_arm(
            output_index=output_index,
            receipt_name="receipts/nonrl.json.gz",
            forecast_name="forecasts/simple-return-stream.json",
            content=inputs.content,
            policy=policy,
            binding=binding,
            role="NONRL",
            reserve_plan_digest="a" * 64,
            artificial_inputs=inputs,
        )


def _artificial_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    progress_jsonl: bool = False,
):
    import tools.market_b4_outer_runner as outer
    from tools.market_replay_oracle.source import NativeSourcePin

    frozen, _ = _frozen_h_plan(tmp_path)
    observer = frozen.observer_repository
    observer.mkdir(parents=True)
    operator = observer / "tools" / "market_b4_artificial_component_operator.py"
    operator.parent.mkdir(parents=True)
    operator.write_text("# fixed artificial H component operator\n", encoding="utf-8")
    verify = observer / "output" / "verify"
    verify.mkdir(parents=True)
    frozen.child_source_root.mkdir(parents=True)
    plan_path = verify / "component-input.json"
    output_root = verify / "component-output"
    witness_root = verify / "component-witness"

    monkeypatch.setattr(
        outer,
        "implementation_source_identity",
        lambda _: {"closure_sha256": "0" * 64, "files": []},
    )
    monkeypatch.setattr(outer, "_git_readback", lambda _: ("1" * 40, True, "2" * 64))
    monkeypatch.setattr(
        outer,
        "verify_native_source",
        lambda root, pin: {
            "head_sha": pin.head_sha,
            "snapshot_sha256": pin.snapshot_sha256,
            "scope": "STATIC_SOURCE_BYTES_NOT_RUNTIME_ATTESTATION",
        },
    )
    monkeypatch.setattr(outer, "_assert_frozen_h_runtime_pin_current", lambda *_: None)
    monkeypatch.setattr(
        outer, "capture_frozen_h_runtime_pin", lambda _: frozen.runtime_pin
    )

    plan_kwargs: dict[str, Any] = {}
    if progress_jsonl:
        plan_kwargs["progress_jsonl_path"] = verify / "component-progress.jsonl"
    return outer.build_artificial_h_component_plan(
        observer_repository=observer,
        child_source_root=frozen.child_source_root,
        source_pin=NativeSourcePin("e" * 40, "f" * 64),
        runtime_pin=frozen.runtime_pin,
        component_spec=_component_spec(),
        input_plan_path=plan_path,
        output_root=output_root,
        witness_root=witness_root,
        **plan_kwargs,
    )


def test_artificial_h_progress_journal_is_optional_and_pinned_in_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    plan = _artificial_plan(tmp_path / "enabled", monkeypatch, progress_jsonl=True)
    progress_path = plan.output_root.parent / "component-progress.jsonl"
    document = json.loads(plan.input_plan_canonical)
    authority = plan.input_authority
    assert plan.progress_jsonl_path == progress_path
    assert not progress_path.exists()
    assert document["progress_jsonl_path"] == str(progress_path)
    assert document["witness_root"] == str(plan.witness_root)
    assert authority["progress_jsonl_path"] == str(progress_path)
    assert "--progress-jsonl" in plan.argv
    progress_position = plan.argv.index("--progress-jsonl")
    assert plan.argv[progress_position + 1] == str(progress_path)
    assert plan.scope == "ARTIFICIAL_H_COMPONENT_PLAN_ONLY"
    assert plan.outer_attempt_authenticated is False
    assert plan.full_b4_authorized is False

    default_plan = _artificial_plan(tmp_path / "default", monkeypatch)
    assert default_plan.progress_jsonl_path is None
    assert "progress_jsonl_path" not in json.loads(default_plan.input_plan_canonical)
    assert "--progress-jsonl" not in default_plan.argv


def test_artificial_h_progress_path_rejects_authoritative_or_existing_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_outer_runner as outer

    plan = _artificial_plan(tmp_path, monkeypatch)
    verify_root = plan.output_root.parent
    existing = verify_root / "already-owned.jsonl"
    existing.write_bytes(b"retain")
    cases = (
        plan.output_root / "progress.jsonl",
        plan.witness_root / "progress.jsonl",
        tmp_path / "outside-verify.jsonl",
        plan.input_plan_path.with_name(f"{plan.input_plan_path.name}:progress.jsonl"),
        existing,
    )
    for progress_path in cases:
        with pytest.raises((ValueError, FileExistsError), match="progress|diagnostic"):
            outer.build_artificial_h_component_plan(
                observer_repository=plan.observer_repository,
                child_source_root=plan.child_source_root,
                source_pin=plan.child_source_pin,
                runtime_pin=plan.runtime_pin,
                component_spec=plan.component_spec,
                input_plan_path=plan.input_plan_path,
                output_root=plan.output_root,
                witness_root=plan.witness_root,
                progress_jsonl_path=progress_path,
            )
    assert existing.read_bytes() == b"retain"
    assert not plan.output_root.exists()
    assert not plan.witness_root.exists()


def test_artificial_h_progress_path_requires_absolute_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_outer_runner as outer

    plan = _artificial_plan(tmp_path, monkeypatch)
    monkeypatch.chdir(plan.output_root.parent)

    with pytest.raises(ValueError, match="absolute|progress"):
        outer.build_artificial_h_component_plan(
            observer_repository=plan.observer_repository,
            child_source_root=plan.child_source_root,
            source_pin=plan.child_source_pin,
            runtime_pin=plan.runtime_pin,
            component_spec=plan.component_spec,
            input_plan_path=plan.input_plan_path,
            output_root=plan.output_root,
            witness_root=plan.witness_root,
            progress_jsonl_path=Path("relative-progress.jsonl"),
        )


def test_artificial_h_progress_path_rejects_original_parent_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_outer_runner as outer

    plan = _artificial_plan(tmp_path, monkeypatch)
    verify_root = plan.output_root.parent
    aliased_path = verify_root / ".." / "verify" / "parent-progress.jsonl"

    with pytest.raises(ValueError, match="absolute|progress|non-aliased"):
        outer.build_artificial_h_component_plan(
            observer_repository=plan.observer_repository,
            child_source_root=plan.child_source_root,
            source_pin=plan.child_source_pin,
            runtime_pin=plan.runtime_pin,
            component_spec=plan.component_spec,
            input_plan_path=plan.input_plan_path,
            output_root=plan.output_root,
            witness_root=plan.witness_root,
            progress_jsonl_path=aliased_path,
        )


def test_component_progress_journal_rejects_ads_alias_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_artificial_component_operator as operator

    _, plan = _prepare_operator_main(tmp_path, monkeypatch)
    ads_path = plan.input_plan_path.with_name(
        f"{plan.input_plan_path.name}:progress.jsonl"
    )
    opened: list[Path] = []

    def forbid_open(path: Path, *_args: object, **_kwargs: object) -> int:
        opened.append(Path(path))
        raise AssertionError("ADS-like progress paths must be rejected before open")

    monkeypatch.setattr(operator.os, "open", forbid_open)
    with pytest.raises(ValueError, match="progress"):
        operator._ProgressJournal.create(
            ads_path,
            observer_root=plan.observer_repository,
            plan_path=plan.input_plan_path,
            output_root=plan.output_root,
            witness_root=plan.witness_root,
        )

    assert opened == []
    assert plan.input_plan_path.read_bytes() == plan.input_plan_canonical
    assert not ads_path.exists()


def test_operator_main_rejects_ads_progress_before_output_or_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator, plan = _prepare_operator_main(tmp_path, monkeypatch, progress_jsonl=True)
    ads_path = plan.input_plan_path.with_name(
        f"{plan.input_plan_path.name}:progress.jsonl"
    )
    document = json.loads(plan.input_plan_canonical)
    document["progress_jsonl_path"] = str(ads_path)
    plan.input_plan_path.write_bytes(_canonical(document))
    opened: list[Path] = []

    def forbid_open(path: Path, *_args: object, **_kwargs: object) -> int:
        opened.append(Path(path))
        raise AssertionError("ADS-like progress paths must be rejected before open")

    monkeypatch.setattr(operator.os, "open", forbid_open)
    with pytest.raises(ValueError, match="progress"):
        operator.main(
            [
                "--plan",
                str(plan.input_plan_path),
                "--progress-jsonl",
                str(ads_path),
            ]
        )

    assert opened == []
    assert not plan.output_root.exists()
    assert plan.input_plan_path.read_bytes() == _canonical(document)


def test_component_progress_rejects_reparse_ancestor_added_after_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import tools.market_b4_artificial_component_operator as operator

    plan = _artificial_plan(tmp_path, monkeypatch, progress_jsonl=True)
    assert plan.progress_jsonl_path is not None
    linked_ancestor = plan.progress_jsonl_path.parent.parent
    original_lstat = Path.lstat

    def lstat_with_reparse(path: Path):
        info = original_lstat(path)
        if path == linked_ancestor:
            return SimpleNamespace(
                st_mode=info.st_mode,
                st_file_attributes=0x400,
                st_nlink=info.st_nlink,
            )
        return info

    monkeypatch.setattr(Path, "lstat", lstat_with_reparse)
    with pytest.raises(ValueError, match="reparse|plain directory|ancestor"):
        operator._assert_progress_path(
            plan.progress_jsonl_path,
            observer_root=plan.observer_repository,
            plan_path=plan.input_plan_path,
            output_root=plan.output_root,
            witness_root=plan.witness_root,
            require_absent=True,
        )


def test_artificial_root_authority_binds_q_protocol_and_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_oracle.artificial_component as artificial
    import tools.market_b4_oracle.offline_replay_consumer as consumer
    import tools.market_b4_outer_runner as outer
    from tools.market_b4_oracle.outer_witness import (
        WitnessFilePin,
        _make_root_observed_attempt,
    )
    from tools.market_b4_oracle.output_index import (
        ArtifactPin,
        load_pinned_output_index,
    )

    plan = _artificial_plan(tmp_path, monkeypatch)
    inputs = artificial.build_artificial_component_inputs(plan.component_spec)
    nonrl_binding, _ = artificial.artificial_component_binding(inputs, "NONRL")
    cash_binding, _ = artificial.artificial_component_binding(inputs, "CASH")
    assert nonrl_binding.protocol_canonical == cash_binding.protocol_canonical
    expected_binding_digest = hashlib.sha256(
        nonrl_binding.protocol_canonical
    ).hexdigest()

    current_runtime = {"packages": {"numpy": "1.26.4", "trade-rl": "0.3.0"}}
    monkeypatch.setattr(
        consumer,
        "implementation_source_identity",
        lambda: {"closure_sha256": "c" * 64, "runtime": current_runtime},
    )

    def validate(authority: dict[str, object], case: str) -> None:
        output_root = tmp_path / case / "output"
        output_root.mkdir(parents=True)
        payload = b"{}"
        (output_root / "placeholder.json").write_bytes(payload)
        manifest = _canonical(
            {
                "placeholder.json": {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            }
        )
        (output_root / "output-manifest.json").write_bytes(manifest)
        manifest_pin = ArtifactPin(len(manifest), hashlib.sha256(manifest).hexdigest())
        output_index = load_pinned_output_index(output_root, manifest_pin)

        source = {
            "closure_sha256": "c" * 64,
            "runtime": current_runtime,
            "repository_head": "d" * 40,
            "working_tree_clean": True,
            "git_status_sha256": "e" * 64,
        }
        argv = ["python", "-m", "synthetic-artificial-component"]
        attempt_uuid = hashlib.sha256(case.encode("utf-8")).hexdigest()[:32]
        start = {
            "schema": "issue810_root_owned_attempt_start_v1",
            "attempt_uuid": attempt_uuid,
            "clock": {"wall_utc_ns": 10, "monotonic_ns": 20},
            "argv": argv,
            "output_root": str(output_root.resolve()),
            "implementation_source": source,
            "input_authority": authority,
        }
        finish = {
            "schema": "issue810_root_owned_attempt_finish_v1",
            "attempt_uuid": attempt_uuid,
            "clock": {"wall_utc_ns": 11, "monotonic_ns": 21},
            "argv_sha256": hashlib.sha256(_canonical(argv)).hexdigest(),
            "output_root": str(output_root.resolve()),
            "implementation_source": source,
            "child": {
                "pid": 101,
                "birth_ticks": 102,
                "exit_code": 0,
                "owned_tree_empty": True,
                "observed_process_identities": [[101, 102]],
            },
            "output_manifest": {
                "bytes": manifest_pin.bytes,
                "sha256": manifest_pin.sha256,
            },
        }
        witness_root = tmp_path / case / "witness"
        witness_root.mkdir()
        start_path = witness_root / "start.json"
        finish_path = witness_root / "finish.json"
        start_bytes = _canonical(start)
        finish_bytes = _canonical(finish)
        start_path.write_bytes(start_bytes)
        finish_path.write_bytes(finish_bytes)
        root_witness = _make_root_observed_attempt(
            start_path,
            finish_path,
            repository_head="d" * 40,
            implementation_source_sha256="c" * 64,
        )
        consumer._validate_root_observed_attempt(
            output_index=output_index,
            content=inputs.content,
            binding=nonrl_binding,
            artificial_inputs=inputs,
            static_inputs=None,
            root_witness=root_witness,
            expected_start_pin=WitnessFilePin(
                len(start_bytes), hashlib.sha256(start_bytes).hexdigest()
            ),
            expected_finish_pin=WitnessFilePin(
                len(finish_bytes), hashlib.sha256(finish_bytes).hexdigest()
            ),
        )

    authority = outer._artificial_h_plan_authority_payload(plan)
    validate(authority, "valid-authority")
    assert authority["binding_digest"] == expected_binding_digest
    assert authority["dataset"] == {"dataset_id": inputs.content.dataset_id}

    missing_fields = dict(authority)
    missing_fields.pop("binding_digest")
    missing_fields.pop("dataset")
    with pytest.raises(ValueError, match="root-observed binding or Dataset differs"):
        validate(missing_fields, "missing-authority-fields")

    wrong_binding = dict(authority)
    wrong_binding["binding_digest"] = "0" * 64
    with pytest.raises(ValueError, match="root-observed binding or Dataset differs"):
        validate(wrong_binding, "wrong-binding-digest")

    wrong_dataset = dict(authority)
    wrong_dataset["dataset"] = {"dataset_id": "0" * 64}
    with pytest.raises(ValueError, match="root-observed binding or Dataset differs"):
        validate(wrong_dataset, "wrong-dataset-id")


def _prepare_operator_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    progress_jsonl: bool = False,
):
    import tools.market_b4_artificial_component_operator as operator
    from trade_rl.strategies.forecasts import simple_stream_io

    plan = _artificial_plan(tmp_path, monkeypatch, progress_jsonl=progress_jsonl)
    plan.input_plan_path.write_bytes(plan.input_plan_canonical)
    monkeypatch.setattr(operator, "__file__", str(plan.operator_path))
    monkeypatch.setattr(operator.Path, "cwd", lambda: plan.child_source_root)
    monkeypatch.setattr(
        operator,
        "sys",
        SimpleNamespace(executable=str(plan.child_python)),
    )
    monkeypatch.setattr(operator, "_load_h_modules", lambda *_: None)
    monkeypatch.setattr(
        operator,
        "_runtime_report",
        lambda **_: {"scope": "ARTIFICIAL_COMPARISON_ONLY"},
    )
    monkeypatch.setattr(operator, "_build_dataset", lambda _spec: object())
    monkeypatch.setattr(operator, "_build_stream", lambda *_: object())

    def publish_forecast(path: Path, _stream: object) -> SimpleNamespace:
        raw = b"synthetic forecast bytes"
        path.write_bytes(raw)
        return SimpleNamespace(
            digest=hashlib.sha256(raw).hexdigest(),
            dataset_id="synthetic-test-dataset",
        )

    monkeypatch.setattr(
        simple_stream_io,
        "publish_simple_return_stream_artifact",
        publish_forecast,
    )
    return operator, plan


def test_operator_main_retains_bounded_failure_after_forecast_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator, plan = _prepare_operator_main(tmp_path, monkeypatch)
    original_error = RuntimeError("component failed: " + "診断🙂" * 10_000)

    def fail_pair(*_args: object) -> object:
        def raise_deep(remaining: int) -> None:
            if remaining:
                raise_deep(remaining - 1)
                return
            raise original_error

        raise_deep(30)

    monkeypatch.setattr(operator, "_run_pair", fail_pair)

    with pytest.raises(RuntimeError) as caught:
        operator.main(["--plan", str(plan.input_plan_path)])

    assert caught.value is original_error
    failure_path = plan.output_root / "operator-failure.json"
    assert failure_path.is_file()
    raw = failure_path.read_bytes()
    assert len(raw) <= 16 * 1024
    record = json.loads(raw)
    assert raw == _canonical(record)
    assert record["schema"] == "issue810_artificial_h_operator_failure_v1"
    assert record["scope"] == "ARTIFICIAL_COMPARISON_ONLY"
    assert record["attempt_uuid"] == plan.attempt_uuid
    assert (
        record["input_plan_sha256"]
        == hashlib.sha256(plan.input_plan_canonical).hexdigest()
    )
    assert record["operator_sha256"] == plan.operator_sha256
    assert record["phase"] == "run_pair"
    assert record["outer_attempt_authenticated"] is False
    assert record["full_b4_authorized"] is False
    assert record["research_authorized"] is False
    failure = record["failure"]
    assert failure["type"] == "RuntimeError"
    assert "診断🙂" in failure["message"]
    assert len(failure["message"].encode("utf-8")) <= 2048
    assert failure["message"].endswith("...[truncated]")
    trace = failure["traceback"]
    assert len(trace["frames"]) == 24
    assert trace["truncated"] is True
    assert any(frame["function"] == "fail_pair" for frame in trace["frames"])
    assert (plan.output_root / "operator-runtime.json").is_file()
    assert (plan.output_root / "forecasts" / "simple-return-stream.json").is_file()
    assert not (plan.output_root / "component-result.json").exists()
    assert not (plan.output_root / "output-manifest.json").exists()


def test_operator_main_does_not_write_to_an_unvalidated_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator, plan = _prepare_operator_main(tmp_path, monkeypatch)
    document = json.loads(plan.input_plan_canonical)
    attacker_path = tmp_path / "outside-verify"
    document["output_root"] = str(attacker_path)
    plan.input_plan_path.write_bytes(_canonical(document))

    with pytest.raises(
        ValueError, match="component output root must be fresh under verify"
    ):
        operator.main(["--plan", str(plan.input_plan_path)])

    assert not attacker_path.exists()
    assert not plan.output_root.exists()


def test_operator_main_rejects_output_root_with_parent_components_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator, plan = _prepare_operator_main(tmp_path, monkeypatch)
    document = json.loads(plan.input_plan_canonical)
    verify_root = plan.output_root.parent
    attacker_path = verify_root / ".." / ".." / "escaped-output"
    document["output_root"] = str(attacker_path)
    plan.input_plan_path.write_bytes(_canonical(document))
    monkeypatch.setattr(operator, "_run_pair", lambda *_args: {})
    monkeypatch.setattr(operator, "_write_manifest", lambda *_args: None)

    with pytest.raises(
        ValueError, match="component output root must be fresh under verify"
    ):
        operator.main(["--plan", str(plan.input_plan_path)])

    assert not attacker_path.exists()
    assert not plan.output_root.exists()


def test_operator_main_rejects_progress_cli_or_existing_path_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator, plan = _prepare_operator_main(tmp_path, monkeypatch, progress_jsonl=True)
    assert plan.progress_jsonl_path is not None
    wrong_path = plan.progress_jsonl_path.with_name("other-progress.jsonl")
    with pytest.raises(ValueError, match="progress"):
        operator.main(
            [
                "--plan",
                str(plan.input_plan_path),
                "--progress-jsonl",
                str(wrong_path),
            ]
        )
    assert not plan.output_root.exists()
    assert not wrong_path.exists()

    retained = b"existing diagnostic target"
    plan.progress_jsonl_path.write_bytes(retained)
    with pytest.raises(FileExistsError, match="progress"):
        operator.main(
            [
                "--plan",
                str(plan.input_plan_path),
                "--progress-jsonl",
                str(plan.progress_jsonl_path),
            ]
        )
    assert plan.progress_jsonl_path.read_bytes() == retained


@pytest.mark.parametrize("binding", ["attempt", "plan_hash", "witness_path"])
def test_operator_main_rejects_mismatched_parent_start_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: str,
) -> None:
    import tools.market_b4_outer_runner as outer

    operator, plan = _prepare_operator_main(tmp_path, monkeypatch, progress_jsonl=True)
    assert plan.progress_jsonl_path is not None
    plan.witness_root.mkdir()
    start = outer._artificial_h_start_record(plan)
    authority = start["input_authority"]
    assert isinstance(authority, dict)
    if binding == "attempt":
        start["attempt_uuid"] = "0" * 32
    elif binding == "plan_hash":
        input_plan = authority["input_plan"]
        assert isinstance(input_plan, dict)
        input_plan["sha256"] = "0" * 64
    else:
        authority["witness_root"] = str(plan.output_root)
    raw_start = _canonical(start)
    start_path = plan.witness_root / "start.json"
    start_path.write_bytes(raw_start)

    with pytest.raises(ValueError, match="parent start witness"):
        operator.main(
            [
                "--plan",
                str(plan.input_plan_path),
                "--progress-jsonl",
                str(plan.progress_jsonl_path),
            ]
        )

    assert start_path.read_bytes() == raw_start
    assert not plan.output_root.exists()
    assert not plan.progress_jsonl_path.exists()


def test_operator_main_rejects_aliased_parent_start_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_outer_runner as outer

    operator, plan = _prepare_operator_main(tmp_path, monkeypatch, progress_jsonl=True)
    assert plan.progress_jsonl_path is not None
    plan.witness_root.mkdir()
    start_path = plan.witness_root / "start.json"
    start_path.write_bytes(_canonical(outer._artificial_h_start_record(plan)))
    alias_path = tmp_path / "start-alias.json"
    try:
        os.link(start_path, alias_path)
    except OSError as error:
        pytest.skip(f"hard-linked test files are unavailable: {error}")

    with pytest.raises(ValueError, match="parent start witness"):
        operator.main(
            [
                "--plan",
                str(plan.input_plan_path),
                "--progress-jsonl",
                str(plan.progress_jsonl_path),
            ]
        )

    assert start_path.read_bytes() == alias_path.read_bytes()
    assert not plan.output_root.exists()
    assert not plan.progress_jsonl_path.exists()


def test_operator_main_rejects_parent_witness_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_outer_runner as outer

    operator, plan = _prepare_operator_main(tmp_path, monkeypatch, progress_jsonl=True)
    assert plan.progress_jsonl_path is not None
    target = tmp_path / "witness-target"
    target.mkdir()
    start_path = target / "start.json"
    start_path.write_bytes(_canonical(outer._artificial_h_start_record(plan)))
    try:
        os.symlink(target, plan.witness_root, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")

    with pytest.raises(ValueError, match="parent witness root"):
        operator.main(
            [
                "--plan",
                str(plan.input_plan_path),
                "--progress-jsonl",
                str(plan.progress_jsonl_path),
            ]
        )

    assert start_path.is_file()
    assert not plan.output_root.exists()
    assert not plan.progress_jsonl_path.exists()
    assert not plan.output_root.exists()


def test_operator_main_rejects_overlapping_parent_witness_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_outer_runner as outer

    operator, plan = _prepare_operator_main(tmp_path, monkeypatch, progress_jsonl=True)
    assert plan.progress_jsonl_path is not None
    witness_root = plan.output_root.parent / "shared-root"
    output_root = witness_root / "component-output"
    witness_root.mkdir()

    document = json.loads(plan.input_plan_canonical)
    document["witness_root"] = str(witness_root)
    document["output_root"] = str(output_root)
    raw_plan = _canonical(document)
    plan.input_plan_path.write_bytes(raw_plan)

    start = outer._artificial_h_start_record(plan)
    start["output_root"] = str(output_root)
    authority = start["input_authority"]
    assert isinstance(authority, dict)
    authority["output_root"] = str(output_root)
    authority["witness_root"] = str(witness_root)
    input_plan = authority["input_plan"]
    assert isinstance(input_plan, dict)
    input_plan["sha256"] = hashlib.sha256(raw_plan).hexdigest()
    start_raw = _canonical(start)
    (witness_root / "start.json").write_bytes(start_raw)

    with pytest.raises(ValueError, match="component witness root overlaps"):
        operator.main(
            [
                "--plan",
                str(plan.input_plan_path),
                "--progress-jsonl",
                str(plan.progress_jsonl_path),
            ]
        )

    assert (witness_root / "start.json").read_bytes() == start_raw
    assert not output_root.exists()
    assert not plan.progress_jsonl_path.exists()


@pytest.mark.parametrize("diagnostic_conflict", ["already_exists", "write_error"])
def test_operator_main_preserves_original_failure_if_diagnostic_cannot_be_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    diagnostic_conflict: str,
) -> None:
    operator, plan = _prepare_operator_main(tmp_path, monkeypatch)
    original_error = RuntimeError("primary operator failure")
    failure_path = plan.output_root / "operator-failure.json"
    preexisting = b"preserve existing bytes"

    if diagnostic_conflict == "write_error":
        write_json = operator._write_json

        def fail_diagnostic_write(path: Path, value: object) -> None:
            if path.name == "operator-failure.json":
                raise OSError("diagnostic storage unavailable")
            write_json(path, value)

        monkeypatch.setattr(operator, "_write_json", fail_diagnostic_write)

    def fail_pair(output_root: Path, *_args: object) -> object:
        if diagnostic_conflict == "already_exists":
            (output_root / failure_path.name).write_bytes(preexisting)
        raise original_error

    monkeypatch.setattr(operator, "_run_pair", fail_pair)

    with pytest.raises(RuntimeError) as caught:
        operator.main(["--plan", str(plan.input_plan_path)])

    assert caught.value is original_error
    if diagnostic_conflict == "already_exists":
        assert failure_path.read_bytes() == preexisting
    else:
        assert not failure_path.exists()


@pytest.mark.parametrize("progress_enabled", [False, True])
def test_operator_main_success_keeps_the_existing_five_file_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    progress_enabled: bool,
) -> None:
    operator, plan = _prepare_operator_main(
        tmp_path, monkeypatch, progress_jsonl=progress_enabled
    )
    parent_start_bytes: bytes | None = None
    if progress_enabled:
        import tools.market_b4_outer_runner as outer

        # Match the real launch order: the parent owns and writes the start
        # witness before it starts the child operator.
        plan.witness_root.mkdir()
        parent_start = outer._artificial_h_start_record(plan)
        parent_start_bytes = _canonical(parent_start)
        (plan.witness_root / "start.json").write_bytes(parent_start_bytes)

    def complete_pair(
        output_root: Path,
        *_args: object,
        progress_journal: Any = None,
    ) -> dict[str, object]:
        if progress_journal is not None:
            for role in ("nonrl", "cash"):
                observe = progress_journal.observer_for(role)
                observe("actor", 2903, 2904)
                observe("reserve", 9430, 9431)
        receipt_root = output_root / "receipts"
        receipt_root.mkdir()
        (receipt_root / "nonrl.json.gz").write_bytes(b"nonrl receipt")
        (receipt_root / "cash.json.gz").write_bytes(b"cash receipt")
        return {"scope": "ARTIFICIAL_COMPARISON_ONLY"}

    monkeypatch.setattr(operator, "_run_pair", complete_pair)

    argv = ["--plan", str(plan.input_plan_path)]
    if progress_enabled:
        assert plan.progress_jsonl_path is not None
        argv.extend(["--progress-jsonl", str(plan.progress_jsonl_path)])
    assert operator.main(argv) == 0
    manifest = json.loads((plan.output_root / "output-manifest.json").read_bytes())
    assert set(manifest) == {
        "component-result.json",
        "forecasts/simple-return-stream.json",
        "operator-runtime.json",
        "receipts/cash.json.gz",
        "receipts/nonrl.json.gz",
    }
    assert not (plan.output_root / "operator-failure.json").exists()
    if progress_enabled:
        assert parent_start_bytes is not None
        assert (plan.witness_root / "start.json").read_bytes() == parent_start_bytes
        assert plan.progress_jsonl_path is not None
        records = [
            json.loads(line)
            for line in plan.progress_jsonl_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert [(record["role"], record["phase"]) for record in records] == [
            ("nonrl", "actor"),
            ("nonrl", "reserve"),
            ("cash", "actor"),
            ("cash", "reserve"),
        ]
        assert all(
            set(record) == {"role", "phase", "start_index", "processing_index"}
            for record in records
        )
        assert not plan.progress_jsonl_path.is_relative_to(plan.output_root)


def test_progress_journal_is_passive_and_disables_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_artificial_component_operator as operator
    from tests.evaluation.test_allocation_terminal_execution import (
        capability as terminal_capability,
    )
    from tests.evaluation.test_allocation_terminal_execution import (
        carrier,
        plans,
    )

    verify_root = tmp_path / "output" / "verify"
    verify_root.mkdir(parents=True)
    plan_path = verify_root / "input-plan.json"
    output_root = verify_root / "attempt"
    witness_root = verify_root / "witness"
    api = terminal_capability()

    def execute(observer=None) -> bytes:
        folds, environment = carrier(kind="cash")
        plan = plans(api, environment, folds, kind="cash", stop=9)
        if observer is None:
            result = api.run_terminal_allocation_execution(folds, environment, plan)
        else:
            result = api.run_terminal_allocation_execution(
                folds, environment, plan, on_advance=observer
            )
        return _canonical(result.payload)

    baseline = execute()
    path = verify_root / "progress.jsonl"
    journal = operator._ProgressJournal.create(
        path,
        observer_root=tmp_path,
        plan_path=plan_path,
        output_root=output_root,
        witness_root=witness_root,
    )
    try:
        instrumented = execute(journal.observer_for("cash"))
    finally:
        journal.close()
    assert instrumented == baseline
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert {record["phase"] for record in records} == {"actor", "reserve"}
    assert all(
        set(record) == {"role", "phase", "start_index", "processing_index"}
        and record["role"] == "cash"
        and type(record["start_index"]) is int
        and record["processing_index"] == record["start_index"] + 1
        for record in records
    )

    failing_path = verify_root / "write-failure.jsonl"
    failing_journal = operator._ProgressJournal.create(
        failing_path,
        observer_root=tmp_path,
        plan_path=plan_path,
        output_root=output_root,
        witness_root=witness_root,
    )
    assert failing_journal._descriptor is not None
    failed_descriptor = failing_journal._descriptor
    original_write = operator.os.write
    attempted_writes = 0

    def fail_progress_write(descriptor: int, data: Any) -> int:
        nonlocal attempted_writes
        if descriptor == failed_descriptor:
            attempted_writes += 1
            raise OSError("diagnostic storage unavailable")
        return original_write(descriptor, data)

    monkeypatch.setattr(operator.os, "write", fail_progress_write)
    try:
        after_write_failure = execute(failing_journal.observer_for("cash"))
    finally:
        failing_journal.close()
    assert after_write_failure == baseline
    assert attempted_writes == 1
    assert failing_path.read_bytes() == b""


def test_operator_main_removes_success_markers_if_manifest_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator, plan = _prepare_operator_main(tmp_path, monkeypatch)
    original_error = OSError("manifest storage unavailable")

    def complete_pair(output_root: Path, *_args: object) -> dict[str, object]:
        receipt_root = output_root / "receipts"
        receipt_root.mkdir()
        (receipt_root / "nonrl.json.gz").write_bytes(b"nonrl receipt")
        (receipt_root / "cash.json.gz").write_bytes(b"cash receipt")
        return {"scope": "ARTIFICIAL_COMPARISON_ONLY"}

    def fail_manifest(_root: Path) -> None:
        raise original_error

    monkeypatch.setattr(operator, "_run_pair", complete_pair)
    monkeypatch.setattr(operator, "_write_manifest", fail_manifest)

    with pytest.raises(OSError) as caught:
        operator.main(["--plan", str(plan.input_plan_path)])

    assert caught.value is original_error
    assert not (plan.output_root / "component-result.json").exists()
    assert not (plan.output_root / "output-manifest.json").exists()
    assert (plan.output_root / "operator-failure.json").is_file()


def test_artificial_h_component_plan_is_typed_and_never_native_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_outer_runner as outer

    plan = _artificial_plan(tmp_path, monkeypatch)

    assert type(plan) is outer.ArtificialHComponentPlan
    assert plan.scope == "ARTIFICIAL_H_COMPONENT_PLAN_ONLY"
    assert plan.process_started is False
    assert plan.outer_attempt_authenticated is False
    assert plan.full_b4_authorized is False
    assert plan.component_spec["dataset_label"] == "synthetic-issue810-component-v1"
    assert (
        plan.component_spec["protocol_id"] == "synthetic-issue810-component-protocol-v1"
    )
    assert plan.argv == (
        str(plan.child_python),
        "-I",
        "-B",
        str(plan.operator_path),
        "--plan",
        str(plan.input_plan_path),
    )
    assert "tools.tier_d_nonrl" not in plan.argv
    assert plan.input_authority["scope"] == "ARTIFICIAL_H_COMPONENT_ONLY"
    assert plan.input_authority["full_b4_authorized"] is False


def test_artificial_h_plan_start_record_must_match_exact_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_outer_runner as outer

    plan = _artificial_plan(tmp_path, monkeypatch)
    start = outer._artificial_h_start_record(plan)
    outer._assert_artificial_h_start_matches_plan(plan, start)

    start["argv"][-1] = "--mode"
    with pytest.raises(ValueError, match="artificial H start witness differs"):
        outer._assert_artificial_h_start_matches_plan(plan, start)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("dataset", {"dataset_id": "0" * 64}),
        ("binding_digest", "0" * 64),
    ],
)
def test_artificial_h_preflight_rejects_q_input_authority_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    changed_value: object,
) -> None:
    from dataclasses import replace

    import tools.market_b4_outer_runner as outer

    plan = _artificial_plan(tmp_path, monkeypatch)
    outer._assert_artificial_h_plan_current(plan)

    authority = outer._artificial_h_plan_authority_payload(plan)
    authority[field] = changed_value
    tampered_plan = replace(
        plan,
        input_authority_canonical=_canonical(authority),
    )

    def fail_if_spawned(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Q authority mismatch must fail before child creation")

    monkeypatch.setattr(outer.subprocess, "Popen", fail_if_spawned)
    with pytest.raises(ValueError, match="Q input authority differs"):
        outer.run_artificial_h_component_attempt(plan=tampered_plan)


def test_artificial_operator_runtime_report_must_match_same_process_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.market_b4_outer_runner as outer

    plan = _artificial_plan(tmp_path, monkeypatch)
    runtime_pin = plan.runtime_pin.authority_payload()
    runtime = runtime_pin["runtime_environment"]
    assert type(runtime) is dict
    packages = runtime["packages"]
    assert type(packages) is dict
    report = {
        "schema": "issue810_artificial_h_operator_runtime_v1",
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "attempt_uuid": plan.attempt_uuid,
        "input_plan_sha256": hashlib.sha256(plan.input_plan_canonical).hexdigest(),
        "operator_sha256": plan.operator_sha256,
        "python_executable": str(plan.child_python),
        "prefix": str(plan.child_source_root / ".venv"),
        "source_root": str(plan.child_source_root),
        "runtime_environment": runtime,
        "runtime_environment_sha256": runtime_pin["runtime_environment_sha256"],
        "numpy_version": packages["numpy"],
        "numpy_origin": str(
            plan.child_source_root
            / ".venv"
            / "Lib"
            / "site-packages"
            / "numpy"
            / "__init__.py"
        ),
        "trade_rl_origin": str(plan.child_source_root / "trade_rl" / "__init__.py"),
        "tier_d_nonrl_origin": str(
            plan.child_source_root / "tools" / "tier_d_nonrl" / "__init__.py"
        ),
        "outer_attempt_authenticated": False,
        "full_b4_authorized": False,
    }
    index = SimpleNamespace(
        read_pinned_bytes=lambda *_args, **_kwargs: _canonical(report)
    )

    outer._assert_artificial_h_operator_runtime_report(plan, index)
    report["runtime_environment_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from its typed plan"):
        outer._assert_artificial_h_operator_runtime_report(plan, index)


@pytest.mark.skipif(
    __import__("os").name != "nt",
    reason="root-owned process-tree observer is Windows-only",
)
@pytest.mark.parametrize("child_exit_code", [0, 7])
def test_artificial_typed_runner_observes_and_reaps_synthetic_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, child_exit_code: int
) -> None:
    import venv

    import tools.market_b4_outer_runner as outer
    from tools.market_replay_oracle.source import NativeSourcePin

    frozen, _ = _frozen_h_plan(tmp_path)
    observer = frozen.observer_repository
    child = frozen.child_source_root
    observer.mkdir(parents=True)
    child.mkdir(parents=True)
    venv.EnvBuilder(with_pip=False).create(child / ".venv")
    verify = observer / "output" / "verify"
    verify.mkdir(parents=True)
    numpy_origin = child / ".venv" / "Lib" / "site-packages" / "numpy" / "__init__.py"
    numpy_origin.parent.mkdir(parents=True)
    numpy_origin.write_text("# synthetic runtime fixture\n", encoding="utf-8")
    (child / "trade_rl").mkdir()
    (child / "trade_rl" / "__init__.py").write_text("", encoding="utf-8")
    (child / "tools" / "tier_d_nonrl").mkdir(parents=True)
    (child / "tools" / "tier_d_nonrl" / "__init__.py").write_text("", encoding="utf-8")

    operator_path = observer / "tools" / "market_b4_artificial_component_operator.py"
    operator_path.parent.mkdir(parents=True)
    operator_source = """from pathlib import Path
import hashlib
import json
import sys
import time

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

plan_path = Path(sys.argv[sys.argv.index("--plan") + 1])
plan_bytes = plan_path.read_bytes()
plan = json.loads(plan_bytes)
source = Path(plan["child_source"]["root"])
runtime_pin = plan["runtime_pin"]
runtime = runtime_pin["runtime_environment"]
output = Path(plan["output_root"])
output.mkdir(parents=True)
report = {
    "schema": "issue810_artificial_h_operator_runtime_v1",
    "scope": "ARTIFICIAL_COMPARISON_ONLY",
    "attempt_uuid": plan["attempt_uuid"],
    "input_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
    "operator_sha256": plan["operator"]["sha256"],
    "python_executable": sys.executable,
    "prefix": sys.prefix,
    "source_root": str(source),
    "runtime_environment": runtime,
    "runtime_environment_sha256": runtime_pin["runtime_environment_sha256"],
    "numpy_version": runtime["packages"]["numpy"],
    "numpy_origin": str(source / ".venv" / "Lib" / "site-packages" / "numpy" / "__init__.py"),
    "trade_rl_origin": str(source / "trade_rl" / "__init__.py"),
    "tier_d_nonrl_origin": str(source / "tools" / "tier_d_nonrl" / "__init__.py"),
    "outer_attempt_authenticated": False,
    "full_b4_authorized": False,
}
artifacts = {
    "operator-runtime.json": canonical(report),
    "synthetic-result.json": canonical({"scope": "ARTIFICIAL_COMPARISON_ONLY", "outer_attempt_authenticated": False, "full_b4_authorized": False}),
}
for name, raw in artifacts.items():
    target = output / name
    target.write_bytes(raw)
manifest = {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in artifacts.items()}
(output / "output-manifest.json").write_bytes(canonical(manifest))
time.sleep(3.0)
sys.exit(__EXIT_CODE__)
""".replace("__EXIT_CODE__", str(child_exit_code))
    operator_path.write_text(operator_source, encoding="utf-8")
    (verify / "typed-child").mkdir()

    source_identity = {
        "closure_sha256": "c" * 64,
        "files": [],
        "runtime": {"packages": {"numpy": "1.26.4", "trade-rl": "0.3.0"}},
    }
    monkeypatch.setattr(
        outer, "implementation_source_identity", lambda _: source_identity
    )
    monkeypatch.setattr(outer, "_git_readback", lambda _: ("d" * 40, True, "e" * 64))
    monkeypatch.setattr(
        outer,
        "verify_native_source",
        lambda _root, pin: {
            "head_sha": pin.head_sha,
            "snapshot_sha256": pin.snapshot_sha256,
            "scope": "STATIC_SOURCE_BYTES_NOT_RUNTIME_ATTESTATION",
        },
    )
    monkeypatch.setattr(outer, "_assert_frozen_h_runtime_pin_current", lambda *_: None)
    monkeypatch.setattr(
        outer,
        "capture_frozen_h_runtime_pin",
        lambda _: frozen.runtime_pin,
    )
    plan = outer.build_artificial_h_component_plan(
        observer_repository=observer,
        child_source_root=child,
        source_pin=NativeSourcePin("e" * 40, "f" * 64),
        runtime_pin=frozen.runtime_pin,
        component_spec=_component_spec(),
        input_plan_path=verify / "typed-child" / "input.json",
        output_root=verify / "typed-child-output",
        witness_root=verify / "typed-child-witness",
    )

    with pytest.raises(ValueError, match="exact typed plan"):
        outer._run_frozen_h_attempt(plan, timeout_seconds=1.0)  # type: ignore[arg-type]

    if child_exit_code != 0:
        with pytest.raises(RuntimeError, match="exited unsuccessfully"):
            outer.run_artificial_h_component_attempt(plan=plan, timeout_seconds=20.0)
        failure = json.loads(
            (plan.witness_root / "failure.json").read_text(encoding="utf-8")
        )
        assert failure["attempt_uuid"] == plan.attempt_uuid
        assert failure["child"]["exit_code"] == child_exit_code
        assert failure["child"]["owned_tree_empty"] is True
        assert "output_manifest" not in failure
        assert (plan.witness_root / "start.json").is_file()
        assert not (plan.witness_root / "finish.json").exists()
        return

    execution = outer.run_artificial_h_component_attempt(
        plan=plan, timeout_seconds=20.0
    )

    # A root-owned process witness does not mean the two roles were replayed
    # independently. The comparison wrapper below is the only path allowed to
    # publish a completed pair result.
    assert execution.scope == "ARTIFICIAL_H_COMPONENT_PROCESS_OBSERVED_ONLY"
    assert execution.input_authority["scope"] == "ARTIFICIAL_H_COMPONENT_ONLY"
    assert execution.input_authority["outer_attempt_authenticated"] is False
    assert execution.input_authority["full_b4_authorized"] is False
    assert execution.output_index.files == (
        "operator-runtime.json",
        "synthetic-result.json",
    )
    start, finish = execution.root_witness.readback(
        expected_start_pin=execution.expected_start_pin,
        expected_finish_pin=execution.expected_finish_pin,
    )
    assert start["argv"] == list(plan.argv)
    assert start["cwd"] == str(child)
    assert finish["child"]["exit_code"] == 0
    assert type(finish["child"]["pid"]) is int
    assert type(finish["child"]["birth_ticks"]) is int
    assert finish["child"]["owned_tree_empty"] is True
    assert (
        finish["output_manifest"]["sha256"]
        == execution.output_index.manifest_pin.sha256
    )
    result = json.loads(
        execution.output_index.read_pinned_bytes(
            "synthetic-result.json", maximum=16 * 1024
        )
    )
    assert result == {
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "outer_attempt_authenticated": False,
        "full_b4_authorized": False,
    }


def test_native_h_builder_rejects_comparison_only_artificial_binding(
    tmp_path: Path,
) -> None:
    import tools.market_b4_outer_runner as outer

    with pytest.raises(ValueError, match="artificial inputs cannot produce"):
        outer.build_frozen_h_launch_plan(
            observer_repository=tmp_path,
            supervisor_attempt_root=tmp_path / "native-attempt",
            witness_root=tmp_path / "witness",
            content=object(),
            policy=object(),
            binding=SimpleNamespace(comparison_only=True),
            runtime_pin=object(),
            static_inputs=object(),
            review_record_path=tmp_path / "review.json",
            review_record_pin=object(),
        )


def test_component_input_plan_digest_covers_the_complete_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _artificial_plan(tmp_path, monkeypatch)
    canonical = _canonical(plan.component_spec)

    assert hashlib.sha256(canonical).hexdigest() == plan.component_spec_sha256
    assert plan.component_spec_sha256 in plan.input_authority_canonical.decode("utf-8")


@pytest.mark.parametrize(
    ("section", "field", "changed"),
    [
        ("dataset_recipe", "availability", "before_event"),
        ("dataset_recipe", "quote_volume_per_bar", 1.0),
        ("cost_recipe", "fee_rate", 0.0),
        ("risk_recipe", "drawdown_stop", 0.5),
        ("forecast_recipe", "fit_called", True),
        ("execution_recipe", "row_9503_used", True),
    ],
)
def test_artificial_h_component_plan_rejects_mutated_fixture_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    changed: object,
) -> None:
    import tools.market_b4_outer_runner as outer
    from tools.market_replay_oracle.source import NativeSourcePin

    frozen, _ = _frozen_h_plan(tmp_path)
    spec = _component_spec()
    nested = spec[section]
    assert type(nested) is dict
    nested[field] = changed

    with pytest.raises(ValueError, match="not the fixed scope"):
        outer.build_artificial_h_component_plan(
            observer_repository=frozen.observer_repository,
            child_source_root=frozen.child_source_root,
            source_pin=NativeSourcePin("e" * 40, "f" * 64),
            runtime_pin=frozen.runtime_pin,
            component_spec=spec,
            input_plan_path=frozen.observer_repository
            / "output"
            / "verify"
            / "input.json",
            output_root=frozen.observer_repository / "output" / "verify" / "output",
            witness_root=frozen.observer_repository / "output" / "verify" / "witness",
        )


def test_outer_and_h_operator_accept_the_same_exact_synthetic_recipe() -> None:
    import tools.market_b4_artificial_component_operator as operator
    import tools.market_b4_outer_runner as outer

    spec = _component_spec()
    assert outer._is_artificial_h_component_spec(spec)
    assert operator._is_fixed_spec(spec)
    for section, field, changed in (
        ("dataset_recipe", "availability", "before_event"),
        ("cost_recipe", "fee_rate", 0.0),
        ("forecast_recipe", "fit_called", True),
        ("execution_recipe", "row_9503_used", True),
    ):
        altered = json.loads(_canonical(spec))
        nested = altered[section]
        nested[field] = changed
        assert not outer._is_artificial_h_component_spec(altered)
        assert not operator._is_fixed_spec(altered)


def test_operator_rejects_active_orders_for_nonselected_symbol() -> None:
    import tools.market_b4_artificial_component_operator as operator

    selected = [{"intent": {"symbol_index": 0}}]
    operator._assert_selected_symbol_orders(selected, selected_symbol_index=0)
    operator._assert_selected_symbol_orders([], selected_symbol_index=0)

    with pytest.raises(ValueError, match="nonselected symbol"):
        operator._assert_selected_symbol_orders(
            [{"intent": {"symbol_index": 1}}], selected_symbol_index=0
        )

    with pytest.raises(ValueError, match="malformed"):
        operator._assert_selected_symbol_orders(
            [{"intent": {"symbol_index": True}}], selected_symbol_index=0
        )


def test_operator_requires_late_actor_fill_and_first_reserve_flatten_fill() -> None:
    import tools.market_b4_artificial_component_operator as operator

    operator._assert_reserve_boundary_fills([9430], [9431], actor_stop_exclusive=9430)

    with pytest.raises(ValueError, match="late actor exposure"):
        operator._assert_reserve_boundary_fills([], [9431], actor_stop_exclusive=9430)

    with pytest.raises(ValueError, match="reserve flatten"):
        operator._assert_reserve_boundary_fills([9430], [], actor_stop_exclusive=9430)


def test_operator_reads_actor_fills_and_filters_cost_rows_by_symbol(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tools.market_b4_artificial_component_operator as operator
    import tools.tier_d_nonrl.tier_d_native as native_api
    import tools.tier_d_nonrl.tier_d_records as records_api
    import trade_rl.evaluation.allocation_terminal_execution as terminal_api
    import trade_rl.evaluation.runs as runs_api

    spec = _component_spec()
    plans = (SimpleNamespace(digest="nonrl"), SimpleNamespace(digest="cash"))
    pair = SimpleNamespace(controls=SimpleNamespace(plans=plans))
    dataset = SimpleNamespace(
        symbols=("BTCUSDT", "ETHUSDT"), dataset_id="synthetic-test-dataset"
    )
    estimates = tuple(
        SimpleNamespace(
            symbol=symbol,
            decision_time=decision_index,
            fee_rate=0.0005 if symbol == "BTCUSDT" else 0.001,
        )
        for symbol in dataset.symbols
        for decision_index in range(2903, 9430)
    )
    expected_selected_estimates = tuple(
        row for row in estimates if row.symbol == "BTCUSDT"
    )
    declared_estimate_args: list[tuple[object, ...]] = []
    role_estimate_args: list[tuple[object, ...]] = []
    actor_fill_indices = {2906, 5090, 7298, 9430}

    def receipt_for(plan: object) -> SimpleNamespace:
        role = getattr(plan, "digest")
        fills = actor_fill_indices if role == "nonrl" else set()
        rows = [
            {
                "native": {
                    "execution": {
                        "fill_count": int(processing_index in fills),
                        "next_index": processing_index,
                    },
                    "active_orders": [],
                    "book": {"quantities": [0.0, 0.0]},
                }
            }
            for processing_index in range(2904, 9431)
        ]
        reserve_fills = {9431} if role == "nonrl" else set()
        closing_rows = [
            {
                "processing_index": processing_index,
                "native": {
                    "execution": {"fill_count": int(processing_index in reserve_fills)},
                    "active_orders": [],
                    "book": {"quantities": [0.0, 0.0]},
                },
            }
            for processing_index in range(9431, 9503)
        ]
        return SimpleNamespace(
            payload={
                "native_receipt": {"rows": rows},
                "closing_rows": closing_rows,
                "settlement_complete": True,
                "final_active_orders": [],
                "final_book": {"quantities": [0.0, 0.0]},
            }
        )

    monkeypatch.setattr(
        operator,
        "_build_prepared",
        lambda *_args: (object(), estimates, {}),
    )

    def capture_declare_pair(
        _prepared: object,
        passed_estimates: tuple[object, ...],
        *_args: object,
        **_kwargs: object,
    ) -> object:
        declared_estimate_args.append(passed_estimates)
        return pair

    def capture_carrier(
        _prepared: object,
        passed_estimates: tuple[object, ...],
        *_args: object,
        **_kwargs: object,
    ) -> object:
        role_estimate_args.append(passed_estimates)
        return object()

    monkeypatch.setattr(native_api, "declare_pair", capture_declare_pair)
    monkeypatch.setattr(native_api, "execution_folds", lambda *_args: ())
    monkeypatch.setattr(native_api, "carrier", capture_carrier)
    monkeypatch.setattr(records_api, "save_gzip", lambda *_args: None)
    monkeypatch.setattr(
        terminal_api,
        "run_terminal_allocation_execution",
        lambda _folds, _environment, plan: receipt_for(plan),
    )
    monkeypatch.setattr(runs_api, "build_candidate_run_provenance", lambda: object())

    result = operator._run_pair(tmp_path, spec, dataset, object(), object())

    roles = result["roles"]
    assert type(roles) is dict
    assert len(declared_estimate_args) == 1
    selected_estimates = declared_estimate_args[0]
    assert len(selected_estimates) == 9430 - 2903
    assert selected_estimates == expected_selected_estimates
    assert all(row.symbol == "BTCUSDT" for row in selected_estimates)
    assert tuple(row.decision_time for row in selected_estimates) == tuple(
        range(2903, 9430)
    )
    assert len(role_estimate_args) == 2
    assert all(passed is selected_estimates for passed in role_estimate_args)
    nonrl = roles["nonrl"]
    assert type(nonrl) is dict
    assert nonrl["actor_fill_processing_indices"] == [2906, 5090, 7298, 9430]
    assert nonrl["reserve_fill_processing_indices"] == [9431]


def test_synthetic_recipe_has_tick_valid_bars_and_terminal_boundary() -> None:
    spec = _component_spec()
    dataset = spec["dataset_recipe"]
    cost = spec["cost_recipe"]
    execution = spec["execution_recipe"]
    assert type(dataset) is dict and type(cost) is dict and type(execution) is dict
    ohlc = dataset["ohlc"]
    rules = dataset["instrument_rules"]
    assert type(ohlc) is dict and type(rules) is dict
    for symbol, rule in rules.items():
        assert type(rule) is dict
        tick = float(rule["tick_size"])
        prices = [
            float(ohlc[key])
            for key in ("open", "close", "ordinary_high", "ordinary_low")
        ]
        if symbol == "BTCUSDT":
            prices.extend(
                float(ohlc[key])
                for key in ("selected_pulse_high", "selected_pulse_low")
            )
        assert all((price / tick).is_integer() for price in prices)
        assert all(
            float(ohlc["ordinary_low"]) <= price <= float(ohlc["ordinary_high"])
            for price in (float(ohlc["open"]), float(ohlc["close"]))
        )
    assert (
        float(dataset["quote_volume_per_bar"]) * float(cost["max_participation_rate"])
        == 500000.0
    )
    assert execution["actor_stop_exclusive"] == 9430
    assert execution["reserve_stop_inclusive"] == 9502
    assert execution["row_9503_used"] is False
    assert spec["pulse_indices"][-1] == execution["actor_stop_exclusive"] - 1
    assert 6527 + 72 == 6599


def _first_artificial_active_order() -> tuple[Any, Any, Any, set[str]]:
    import tools.market_b4_oracle.artificial_component as artificial
    from tools.market_b4_oracle.frozen_forecast_index import FrozenForecastIndex
    from tools.market_b4_oracle.frozen_market_replay import FrozenMarketReplay

    inputs = artificial.build_artificial_component_inputs(_component_spec())
    binding, policy = artificial.artificial_component_binding(inputs, "NONRL")
    forecast_bytes = inputs.forecast_payload_canonical
    forecast = FrozenForecastIndex(
        forecast_bytes,
        pinned_bytes=len(forecast_bytes),
        pinned_sha256=hashlib.sha256(forecast_bytes).hexdigest(),
    )
    replay = FrozenMarketReplay(
        content=inputs.content,
        policy=policy,
        binding=binding,
        reserve_plan_digest="a" * 64,
    )
    advance = None
    prior_terminal_ids: set[str] = set()
    while not replay._active:
        prior_terminal_ids = set(replay._terminal)
        advance = replay.advance(forecast_index=forecast, role="NONRL")
    assert advance is not None
    assert advance.decision_index == 2905
    return replay, advance, next(iter(replay._active.values())), prior_terminal_ids


def _native_active_order(order: Any, intent: dict[str, object]) -> dict[str, object]:
    from tools.market_b4_oracle.expected_intent import project_quantity

    return {
        "intent": intent,
        "status": order.status,
        "trigger_index": None,
        "last_processed_index": order.last_processed_index,
        "terminal_reason": order.terminal_reason,
        "evidence_version": order.evidence_version,
        "exact_cumulative_filled_quantity": str(order.exact_filled_quantity),
        "remaining_quantity": order.remaining_quantity,
        "cumulative_filled_quantity": project_quantity(order.exact_filled_quantity),
        "cumulative_filled_notional": order.cumulative_filled_notional,
    }


def test_order_comparison_accepts_native_intent_with_default_reduce_only_omitted() -> (
    None
):
    from dataclasses import asdict

    import tools.market_b4_oracle.offline_replay_consumer as consumer

    replay, advance, order, prior_terminal_ids = _first_artificial_active_order()
    native_intent = asdict(order.intent)
    assert native_intent.pop("reduce_only") is False

    consumer._compare_orders(
        {
            "active_orders": [_native_active_order(order, native_intent)],
            "terminal_order_reasons": [
                {"order_id": order_id, "reason": terminal.terminal_reason}
                for order_id, terminal in replay._terminal.items()
                if order_id not in prior_terminal_ids
            ],
        },
        replay,
        advance,
        prior_terminal_ids,
    )


def test_order_comparison_retains_true_reduce_only_and_rejects_intent_mutations() -> (
    None
):
    from dataclasses import replace as dataclass_replace

    import tools.market_b4_oracle.offline_replay_consumer as consumer
    from tools.market_b4_oracle.expected_intent import ExpectedIntent

    replay, advance, order, _ = _first_artificial_active_order()
    original = order.intent
    intent = ExpectedIntent.create(
        dataset_id=original.dataset_id,
        target_identity=original.target_identity,
        execution_policy_digest=original.execution_policy_digest,
        symbol_index=original.symbol_index,
        requested_quantity=original.requested_quantity,
        order_type=original.order_type,
        time_in_force=original.time_in_force,
        limit_price=original.limit_price,
        stop_price=original.stop_price,
        submit_index=original.submit_index,
        eligible_index=original.eligible_index,
        expiry_index=original.expiry_index,
        submission_reference_price=original.submission_reference_price,
        decision_equity=original.decision_equity,
        replaced_order_id=original.replaced_order_id,
        reduce_only=True,
    )
    true_order = dataclass_replace(order, intent=intent)
    true_replay = SimpleNamespace(
        _active={intent.order_id: true_order},
        _terminal={},
    )
    native_intent = intent.canonical_payload()
    assert native_intent["reduce_only"] is True
    native_order = _native_active_order(true_order, native_intent)
    native = {"active_orders": [native_order], "terminal_order_reasons": []}

    consumer._compare_orders(native, true_replay, advance, set())

    wrong_reduce_only = json.loads(json.dumps(native))
    wrong_reduce_only["active_orders"][0]["intent"]["reduce_only"] = False
    with pytest.raises(ValueError, match="native active order intent"):
        consumer._compare_orders(wrong_reduce_only, true_replay, advance, set())

    changed_identity = json.loads(json.dumps(native))
    changed_identity["active_orders"][0]["intent"]["target_identity"] = "0" * 64
    with pytest.raises(ValueError, match="native active order intent"):
        consumer._compare_orders(changed_identity, true_replay, advance, set())
