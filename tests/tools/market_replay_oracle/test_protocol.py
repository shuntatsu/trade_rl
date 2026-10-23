from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from tools.market_replay_oracle.artifact import VerifiedDatasetContent

FROZEN_PROTOCOL_JSON = '{"accounts":["H24_RIDGE_A1_BTCUSDT","H24_RIDGE_A1_ETHUSDT","H24_RIDGE_A10_BTCUSDT","H24_RIDGE_A10_ETHUSDT","H72_RIDGE_A1_BTCUSDT","H72_RIDGE_A1_ETHUSDT","H72_RIDGE_A10_BTCUSDT","H72_RIDGE_A10_ETHUSDT"],"actor":{"fit_cutoff":"vintage start minus 1 ns; endpoints/publication strictly earlier","inference_delay_seconds":0,"model_fit_time":"vintage start; declared simulation, not measured latency","rows":6527,"start":2903,"start_utc":"2025-04-01T00:00:00","stop":9430,"stop_utc":"2025-12-28T23:00:00","vintage_start_utc":["2025-04-01T00:00:00","2025-07-01T00:00:00","2025-10-01T00:00:00"],"vintage_starts":[2903,5087,7295]},"attempts":"normal technical debugging/retries; retain each failed attempt and unchanged thresholds","budget":{"advances_per_arm":6599,"cost_producers":4,"dataset_loads":5,"hpo":false,"matched_pairs":24,"native_arms":48,"public_prepare_fits":4,"regression_features":6,"ridge_vintages":12,"seeds":[],"total_native_advances":316752,"unattempted":"retain UNATTEMPTED; never fabricate a cash-zero receipt"},"capital":{"combination":"two nominal USDT books for ONE config/scenario only","combined_denominator":200000.0,"currency":"USDT","fx_parity":1.0,"mode":"independent_symbol","per_account":100000.0},"concurrency":{"max_workers":4},"configurations":[{"horizon_hours":24,"id":"H24_RIDGE_A1","ridge_alpha":1.0},{"horizon_hours":24,"id":"H24_RIDGE_A10","ridge_alpha":10.0},{"horizon_hours":72,"id":"H72_RIDGE_A1","ridge_alpha":1.0},{"horizon_hours":72,"id":"H72_RIDGE_A10","ridge_alpha":10.0}],"dataset":{"build":"market_build_v3","dataset_id":"e916ff716fb5e4b90d5805eaa79200824b09f103984fa845bbeaedb57bef194f","development_start":"2024-12-01T01:00:00","development_stop_exclusive":"2026-01-01T01:00:00","dimensions":[9504,2,6],"feature_names":["1h__log_return_24bar","4h__log_return_1bar","1d__log_return_1bar","1h__realized_volatility_24bar","1h__volume_zscore_24bar","1h__funding_bps"],"fit_symbols":["BTCUSDT","ETHUSDT"],"identity":"market_dataset_identity_v7","input_sha256":{"arrays.npz":"25c97a4ebd1e81ac64f4bc7e78dce4ab102c522e548e30a2154fcfa0143842a3","manifest.json":"6ac5099b98a32bd33e4f98c5ac5fb2dbffd19b0b9e3c7c3a9b5a8f6860810ee6"},"root":"C:\\\\dev\\\\trade_rl\\\\.worktrees\\\\nonrl-market-development-20261012\\\\output\\\\nonrl-market-development-20261012\\\\task18-selected-six\\\\task18-new-selected-six-feature-witness","schema":"market_dataset_artifact_v4","unused_status":"CONSUMED_DEVELOPMENT_NEVER_UNUSED"},"decision":{"adoption":"NOT_ESTABLISHED","decomposition":"returned native price PnL / fill fee-spread-impact / funding / borrow-interest; terminal closing cost disclosed as subset","guardrails":"every receipt verified, flat, no pending; maximum DD<=20% per book","minimum_operational_profit_threshold":"UNKNOWN; no adoption claim","operating_costs":"fixed infrastructure costs UNKNOWN; account net is pre-tax and before unknown operating costs","primary":"terminal after-cost NONRL minus matched CASH wealth / V0","rank":"per config/scenario combined BTC+ETH difference / (2*V0)","stress":"BASEL0, COST2L0, BASEL2 each required and disclosed; no crossed cell","winner":"NOT_DECLARED"},"economics":{"borrow_available":true,"borrow_rate":0.0,"cash_rate":0.0,"cost2":"multiplier2 doubles fee/spread/impact only; carry unchanged","dataset_fee":0.0005,"dataset_spread":0.0002,"execution":"one native MarketExecutor/BookState ledger; prior-bar capacity","impact_coefficient":0.0001,"latency2":"BASE beliefs/forecasts; MARKET latency2 actual execution","overlay_fee":0.0,"overlay_spread":0.0,"participation":0.05,"reference_notional":100000.0,"stochastic_slippage":0.0,"venue_rules":"current snapshot assumption, not historical PIT"},"funding":{"clock_limit":"counted bar-close proxy may understate actual settlement age","coverage_bounds":{"lookback_hours":24,"maximum_counted_close_age_hours":8,"maximum_counted_close_gap_hours":9,"minimum_counted_events":2},"marks":"bar-close proxy; rates times bar-close mark all rows","observed_24h_counts":{"maximum":4,"minimum":2},"source":"Task18 rates-only feature witness, not authenticated venue marks","zero_rate_events":"counted by event_count, not rate!=0; original dates unchanged"},"policy":{"allocator":{"lower_weight":-1.0,"max_turnover":null,"risk_aversion":0.0,"upper_weight":1.0},"cash_action":"direct HOLD (code0) from flat initial book, scale1","fold_metadata":"execution segments only; checkpoint/selection placeholders UNUSED","forecasts":"raw close-ratio simple-return labels; carry/costs in allocator","gae_lambda_unused_nonrl":0.95,"gamma":1.0,"max_active_order_slots":8,"nonrl_action":"residual KEEP_BASELINE (code2), scale1","observation_schema":"allocation_account_observation_v2","resets_per_arm":1,"reward":"equity_delta_v1 / fixed V0","risk":{"drawdown_start":0.1,"drawdown_stop":0.2,"emergency_turnover_override":true,"fail_closed_tolerance":1e-10,"max_abs_weight":1.0,"max_gross":1.0,"max_turnover":null},"rollout_steps":32,"vintage_boundary":"continuous quantity/cash/pending/risk"},"progress_policy":{"no_progress_seconds":600,"sample_seconds":1,"schema":"issue810_native_progress_v1"},"research":{"can_prove":"conditional development comparison under declared financial proxies","cannot_prove":"unused-future, maximum profit, historical calibration, live readiness","causal_story":"persistence/reversion in consumed crypto prices is a conjecture","falsifiers":"nonpositive CASH delta, DD breach, incomplete settlement/coverage","hypothesis":"lagged selected observations may predict H24/H72 price returns","legacy":"six-fit PPO learnability stays FAILED incl fee-only seed17; Phase1 source unchanged","no":["new features","new dataset","new algorithm","LLM trading/teachers","second ledger","RL market fit","sealed/unused reads","network market","live orders"],"objective":"screen fixed NONRL terminal net profit against CASH","review":"ROOT different-model result-blind review BEFORE actual costs/fits/PnL"},"resource_proposal":{"disk_bytes":4294967296,"limitations":"native APIs retain arm receipts in active workers; full-matrix feasibility and the proposed resource/progress bounds are not established","maximum_inflight_pairs":4,"measured":false,"rss_bytes":2147483648,"sample_seconds":10,"semantics":"Windows owned process-tree RSS including redirector descendants; sampled stop; overshoot possible","storage":"gzip full raw receipts; matched pairs retained by at most the declared in-flight worker count","wall_seconds":3600},"roles":["nonrl","cash"],"scenarios":["BASEL0","COST2L0","BASEL2"],"schema":"issue810_tier_d_nonrl48_result_blind_v3","scope":"CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY","source_head":"8bc2fd0cba83a02f5040159c1fc71e95fab774db","terminal":{"advances":72,"last_row_9503":"unused by actor/reserve","stop_inclusive":9502,"utc":"2025-12-31T23:00:00"}}'
SOURCE_HEAD = "8bc2fd0cba83a02f5040159c1fc71e95fab774db"
MAX_PROTOCOL_BYTES = 1024 * 1024


def _api() -> tuple[type[Any], Callable[..., Any], Callable[..., Any]]:
    from tools.market_replay_oracle.protocol import (
        ProtocolPin,
        bind_schedule,
        load_protocol,
    )

    return ProtocolPin, load_protocol, bind_schedule


def _document() -> dict[str, Any]:
    assert FROZEN_PROTOCOL_JSON is not None
    return json.loads(FROZEN_PROTOCOL_JSON)


def _wire(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load(
    tmp_path: Path,
    document: dict[str, Any],
    *,
    pin_source_head: str = SOURCE_HEAD,
) -> dict[str, Any]:
    ProtocolPin, load_protocol, _ = _api()
    raw = _wire(document)
    path = tmp_path / "protocol.json"
    path.write_bytes(raw)
    pin = ProtocolPin(
        sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw),
        source_head=pin_source_head,
    )
    return load_protocol(path, pin)


def _content(
    protocol: dict[str, Any],
    *,
    symbols: tuple[str, ...] | None = None,
    features: tuple[str, ...] | None = None,
    timestamp_shift_hours: int = 0,
    timestamp_change: tuple[int, int] | None = None,
    dataset_id: str | None = None,
    feature_shape: tuple[int, int, int] = (9504, 2, 6),
    calendar_kind: str = "continuous_24_7",
    nominal_bar_hours: int | float | None = 1.0,
) -> VerifiedDatasetContent:
    dataset = protocol["dataset"]
    start = np.datetime64(dataset["development_start"], "ns")
    timestamps = start + np.arange(9504) * np.timedelta64(1, "h")
    timestamps = timestamps + np.timedelta64(timestamp_shift_hours, "h")
    if timestamp_change is not None:
        index, hours = timestamp_change
        timestamps[index] += np.timedelta64(hours, "h")
    arrays = MappingProxyType(
        {
            "timestamps": timestamps,
            "features": np.zeros(feature_shape, dtype=np.float64),
        }
    )
    scalars = MappingProxyType(
        {
            "symbols": symbols or tuple(dataset["fit_symbols"]),
            "feature_names": features or tuple(dataset["feature_names"]),
            "calendar_kind": calendar_kind,
            "nominal_bar_hours": nominal_bar_hours,
        }
    )
    return VerifiedDatasetContent(
        dataset_id or dataset["dataset_id"],
        arrays,
        scalars,
    )


def test_protocol_module_is_available() -> None:
    assert importlib.util.find_spec("tools.market_replay_oracle.protocol") is not None


def test_load_protocol_accepts_pinned_frozen_v3_spec(tmp_path: Path) -> None:
    protocol = _load(tmp_path, _document())

    assert protocol["schema"] == "issue810_tier_d_nonrl48_result_blind_v3"
    assert protocol["scope"] == "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY"
    assert protocol["source_head"] == SOURCE_HEAD
    assert protocol["budget"]["native_arms"] == 48


def test_load_protocol_requires_the_frozen_source_head_pin(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source head pin"):
        _load(tmp_path, _document(), pin_source_head="1" * 40)


def test_protocol_pin_requires_bounded_lowercase_digests() -> None:
    ProtocolPin, _, _ = _api()

    with pytest.raises(ValueError):
        ProtocolPin(sha256="A" * 64, raw_bytes=1, source_head=SOURCE_HEAD)
    with pytest.raises(ValueError):
        ProtocolPin(
            sha256="a" * 64, raw_bytes=MAX_PROTOCOL_BYTES + 1, source_head=SOURCE_HEAD
        )
    with pytest.raises(ValueError):
        ProtocolPin(sha256="a" * 64, raw_bytes=1, source_head="bad")


def test_raw_pin_is_checked_before_json_parse(tmp_path: Path) -> None:
    ProtocolPin, load_protocol, _ = _api()
    raw = b'{"invalid":NaN'
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    pin = ProtocolPin(sha256="0" * 64, raw_bytes=len(raw), source_head=SOURCE_HEAD)

    with pytest.raises(ValueError, match="SHA-256"):
        load_protocol(path, pin)

    size_pin = ProtocolPin(
        sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw) + 1,
        source_head=SOURCE_HEAD,
    )
    with pytest.raises(ValueError, match="byte count"):
        load_protocol(path, size_pin)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema":"first","schema":"second"}',
        b'{"schema":NaN}',
        b'{"schema":1e999}',
    ],
)
def test_strict_json_rejects_duplicate_and_nonfinite_values(
    tmp_path: Path, raw: bytes
) -> None:
    ProtocolPin, load_protocol, _ = _api()
    path = tmp_path / "strict.json"
    path.write_bytes(raw)
    pin = ProtocolPin(
        sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw),
        source_head=SOURCE_HEAD,
    )

    with pytest.raises(ValueError):
        load_protocol(path, pin)


def test_protocol_size_limit_fails_before_parsing(tmp_path: Path) -> None:
    ProtocolPin, load_protocol, _ = _api()
    raw = b" " * (MAX_PROTOCOL_BYTES + 1)
    path = tmp_path / "oversized.json"
    path.write_bytes(raw)
    pin = ProtocolPin(
        sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=MAX_PROTOCOL_BYTES,
        source_head=SOURCE_HEAD,
    )

    with pytest.raises(ValueError, match="1 MiB"):
        load_protocol(path, pin)


def test_load_protocol_rejects_symlinked_parent(tmp_path: Path) -> None:
    ProtocolPin, load_protocol, _ = _api()
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    raw = _wire(_document())
    (real_parent / "protocol.json").write_bytes(raw)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    pin = ProtocolPin(
        sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw),
        source_head=SOURCE_HEAD,
    )

    with pytest.raises(ValueError, match="link|ancestor"):
        load_protocol(linked_parent / "protocol.json", pin)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc.pop("budget"),
        lambda doc: doc.__setitem__("new_field", "unsupported"),
        lambda doc: doc.__setitem__("scope", "OTHER_SCOPE"),
        lambda doc: doc.__setitem__("source_head", "0" * 40),
    ],
)
def test_protocol_closure_scope_and_source_head_fail_closed(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], object]
) -> None:
    document = _document()
    mutation(document)

    with pytest.raises(ValueError):
        _load(tmp_path, document)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc["budget"].__setitem__("advances_per_arm", 6598),
        lambda doc: doc["capital"].__setitem__("per_account", 99999.0),
        lambda doc: doc["policy"]["risk"].__setitem__("max_gross", 0.9),
        lambda doc: doc["economics"].__setitem__("impact_coefficient", 0.0002),
        lambda doc: doc["funding"].__setitem__("cash_rate", 0.001),
        lambda doc: doc.__setitem__("scenarios", list(reversed(doc["scenarios"]))),
        lambda doc: doc.__setitem__("roles", ["cash", "nonrl"]),
        lambda doc: doc["configurations"].reverse(),
        lambda doc: doc["accounts"].reverse(),
    ],
)
def test_fixed_financial_and_arm_contract_rejects_rehashed_changes(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], object]
) -> None:
    document = _document()
    mutation(document)

    with pytest.raises(ValueError):
        _load(tmp_path, document)


def test_bind_schedule_emits_fixed_chronology_and_ordered_48_arms(
    tmp_path: Path,
) -> None:
    protocol = _load(tmp_path, _document())
    _, _, bind_schedule = _api()
    schedule = bind_schedule(protocol, _content(protocol))

    assert set(schedule) == {"actor", "terminal", "vintages", "arms"}
    assert schedule["actor"]["start_index"] == 2903
    assert schedule["actor"]["stop_exclusive_index"] == 9430
    assert schedule["actor"]["advances"] == 6527
    assert schedule["terminal"]["stop_inclusive_index"] == 9502
    assert schedule["terminal"]["advances"] == 72
    assert schedule["terminal"]["total_advances_per_arm"] == 6599
    assert schedule["terminal"]["first_unused_index"] == 9503
    assert schedule["terminal"]["first_unused_utc"] == "2026-01-01T00:00:00"
    assert [
        (item["start_index"], item["stop_exclusive_index"])
        for item in schedule["vintages"]
    ] == [(2903, 5087), (5087, 7295), (7295, 9430)]
    assert [item["fit_cutoff_utc"] for item in schedule["vintages"]] == [
        "2025-03-31T23:59:59.999999999",
        "2025-06-30T23:59:59.999999999",
        "2025-09-30T23:59:59.999999999",
    ]

    arms = schedule["arms"]
    assert len(arms) == 48
    assert [
        (arm["configuration"], arm["symbol"], arm["scenario"], arm["role"])
        for arm in arms[:8]
    ] == [
        ("H24_RIDGE_A1", "BTCUSDT", "BASEL0", "nonrl"),
        ("H24_RIDGE_A1", "BTCUSDT", "BASEL0", "cash"),
        ("H24_RIDGE_A1", "BTCUSDT", "COST2L0", "nonrl"),
        ("H24_RIDGE_A1", "BTCUSDT", "COST2L0", "cash"),
        ("H24_RIDGE_A1", "BTCUSDT", "BASEL2", "nonrl"),
        ("H24_RIDGE_A1", "BTCUSDT", "BASEL2", "cash"),
        ("H24_RIDGE_A1", "ETHUSDT", "BASEL0", "nonrl"),
        ("H24_RIDGE_A1", "ETHUSDT", "BASEL0", "cash"),
    ]
    assert arms[0]["account"] == "H24_RIDGE_A1_BTCUSDT"
    assert arms[0]["forecast_horizon_seconds"] == 86400
    assert (
        arms[0]["action_mode"],
        arms[0]["action_code"],
        arms[0]["action_scale"],
    ) == (
        "residual",
        2,
        1.0,
    )
    assert (
        arms[1]["action_mode"],
        arms[1]["action_code"],
        arms[1]["action_scale"],
    ) == (
        "direct",
        0,
        1.0,
    )
    assert arms[-1]["configuration"] == "H72_RIDGE_A10"
    assert arms[-1]["symbol"] == "ETHUSDT"
    assert arms[-1]["scenario"] == "BASEL2"
    assert arms[-1]["role"] == "cash"
    assert arms[-1]["forecast_horizon_seconds"] == 259200


@pytest.mark.parametrize(
    "content_change",
    [
        {"dataset_id": "0" * 64},
        {"symbols": ("ETHUSDT", "BTCUSDT")},
        {"features": tuple(reversed(_document()["dataset"]["feature_names"]))},
        {"feature_shape": (9503, 2, 6)},
        {"calendar_kind": "daily"},
        {"nominal_bar_hours": 2.0},
        {"timestamp_shift_hours": 1},
        {"timestamp_change": (5087, 1)},
        {"timestamp_change": (9502, -1)},
    ],
)
def test_bind_schedule_rejects_content_identity_or_clock_mismatch(
    tmp_path: Path, content_change: dict[str, object]
) -> None:
    protocol = _load(tmp_path, _document())
    _, _, bind_schedule = _api()

    with pytest.raises(ValueError):
        bind_schedule(protocol, _content(protocol, **content_change))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc["budget"].__setitem__("total_native_advances", 316751),
        lambda doc: doc["capital"].__setitem__("combined_denominator", 100000.0),
        lambda doc: doc["policy"]["risk"].__setitem__("drawdown_stop", 0.25),
        lambda doc: doc["economics"].__setitem__("dataset_fee", 0.0004),
        lambda doc: doc["scenarios"].pop(),
    ],
)
def test_bind_schedule_revalidates_fixed_protocol_before_expansion(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], object]
) -> None:
    protocol = _load(tmp_path, _document())
    mutation(protocol)
    _, _, bind_schedule = _api()

    with pytest.raises(ValueError):
        bind_schedule(protocol, _content(protocol))
