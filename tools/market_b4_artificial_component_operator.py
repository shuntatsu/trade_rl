"""Run one fixed synthetic H component pair under the pinned H runtime.

This operator is deliberately outside the comparator package. It imports no
Q replay, receipt, witness, or comparison code; its only output is an artificial
NONRL/CASH component pair and bounded runtime facts. It never enters the native
market CLI or reads the frozen Task18 dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import numpy as np

_PLAN_SCHEMA = "issue810_artificial_h_component_plan_v1"
_SPEC_SCHEMA = "issue810_artificial_h_component_spec_v1"
_RUNTIME_SCHEMA = "issue810_artificial_h_operator_runtime_v1"
_ROWS = 9504
_ACTOR_START = 2903
_ACTOR_STOP = 9430
_ACTOR_ROWS = _ACTOR_STOP - _ACTOR_START
_RESERVE_STOP = 9502
_RESERVE_ROWS = _RESERVE_STOP - _ACTOR_STOP
_VINTAGE_STARTS = (2903, 5087, 7295)
_PULSES = (2905, 3995, 5082, 5089, 6191, 7290, 7297, 8362, 9425, 9429)
_FEATURES = (
    "fixture_pulse",
    "fixture_log_return",
    "fixture_body_return",
    "fixture_volume_change",
    "fixture_close_location",
    "fixture_lower_wick",
)
_MAX_PLAN_BYTES = 1024 * 1024
_MAX_OUTPUT_FILE_BYTES = 2 * 1024**3
_MAX_FAILURE_BYTES = 16 * 1024
_MAX_FAILURE_MESSAGE_BYTES = 2048
_MAX_FAILURE_TRACEBACK_FRAMES = 24


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError("artificial component JSON contains a duplicate key")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ValueError(f"artificial component JSON contains non-finite {value}")


def _read_plan(path: Path) -> tuple[bytes, dict[str, object]]:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("component input plan path must be absolute and confined")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or getattr(before, "st_nlink", 1) != 1:
        raise ValueError("component input plan must be a regular unaliased file")
    if getattr(before, "st_file_attributes", 0) & 0x400:
        raise ValueError("component input plan cannot be a reparse point")
    if not 0 < before.st_size <= _MAX_PLAN_BYTES:
        raise ValueError("component input plan exceeds its fixed byte bound")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError("component input plan changed while opening")
        raw = stream.read(_MAX_PLAN_BYTES + 1)
        after = os.fstat(stream.fileno())
    final = path.lstat()
    if (
        len(raw) != before.st_size
        or len(raw) > _MAX_PLAN_BYTES
        or (after.st_dev, after.st_ino, after.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_size)
        or (final.st_dev, final.st_ino, final.st_size)
        != (before.st_dev, before.st_ino, before.st_size)
    ):
        raise ValueError("component input plan changed while reading")
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    if type(value) is not dict or _canonical(value) != raw:
        raise ValueError("component input plan must be a canonical JSON object")
    return raw, value


def _sha256(path: Path, *, maximum: int = _MAX_OUTPUT_FILE_BYTES) -> tuple[int, str]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or getattr(before, "st_nlink", 1) != 1:
        raise ValueError("component artifact must be a regular unaliased file")
    if getattr(before, "st_file_attributes", 0) & 0x400:
        raise ValueError("component artifact cannot be a reparse point")
    if not 0 < before.st_size <= maximum:
        raise ValueError("component artifact exceeds its fixed byte bound")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise ValueError("component artifact changed while opening")
        while chunk := stream.read(64 * 1024):
            size += len(chunk)
            if size > maximum:
                raise ValueError("component artifact grew past its bound")
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    final = path.lstat()

    def stamp(item: os.stat_result) -> tuple[int, int, int, int]:
        return item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns

    if (
        stamp(after) != stamp(opened)
        or stamp(final) != stamp(before)
        or size != before.st_size
    ):
        raise ValueError("component artifact changed while hashing")
    return size, digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    raw = _canonical(value)
    if not raw or len(raw) > _MAX_PLAN_BYTES:
        raise ValueError("component JSON output exceeds its fixed metadata bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _assert_output_root(plan: dict[str, object], observer_root: Path) -> Path:
    output_value = plan.get("output_root")
    if type(output_value) is not str:
        raise ValueError("component plan has no output root")
    output = Path(output_value)
    verify_root = observer_root / "output" / "verify"
    if (
        not output.is_absolute()
        or output == verify_root
        or not output.is_relative_to(verify_root)
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("component output root must be fresh under verify")
    parent = output.parent
    while parent != verify_root:
        info = parent.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("component output ancestor is not a plain directory")
        parent = parent.parent
    return output


def _assert_plan(
    plan: dict[str, object], *, plan_raw: bytes, operator_path: Path, plan_path: Path
) -> tuple[Path, dict[str, object], str]:
    if set(plan) != {
        "schema",
        "scope",
        "attempt_uuid",
        "child_source",
        "runtime_pin",
        "operator",
        "component_spec",
        "component_spec_sha256",
        "output_root",
        "outer_attempt_authenticated",
        "full_b4_authorized",
    }:
        raise ValueError("component input plan has an unexpected field roster")
    child = plan.get("child_source")
    runtime_pin = plan.get("runtime_pin")
    operator = plan.get("operator")
    spec = plan.get("component_spec")
    if (
        plan.get("schema") != _PLAN_SCHEMA
        or plan.get("scope") != "ARTIFICIAL_H_COMPONENT_ONLY"
        or plan.get("outer_attempt_authenticated") is not False
        or plan.get("full_b4_authorized") is not False
        or type(plan.get("attempt_uuid")) is not str
        or type(child) is not dict
        or type(runtime_pin) is not dict
        or type(operator) is not dict
        or type(spec) is not dict
        or child.get("root") != str(Path.cwd().resolve())
        or runtime_pin.get("source_root") != child.get("root")
        or runtime_pin.get("python_executable") != str(Path(sys.executable).resolve())
        or operator.get("path") != str(operator_path)
        or type(operator.get("sha256")) is not str
        or _sha256(operator_path, maximum=4 * 1024**2)[1] != operator.get("sha256")
        or hashlib.sha256(_canonical(spec)).hexdigest()
        != plan.get("component_spec_sha256")
        or plan_path.resolve() == operator_path.resolve()
    ):
        raise ValueError("component input plan differs from this H process")
    try:
        UUID(str(plan["attempt_uuid"]))
    except ValueError as error:
        raise ValueError("component attempt UUID is malformed") from error
    if not _is_fixed_spec(spec):
        raise ValueError(
            "component specification is outside its fixed artificial scope"
        )
    observer = plan.get("operator")
    assert type(observer) is dict
    observer_root = Path(str(observer["path"])).parents[1]
    output_root = _assert_output_root(plan, observer_root)
    return output_root, runtime_pin, str(plan["attempt_uuid"])


def _is_fixed_spec(spec: dict[str, object]) -> bool:
    expected = {
        "schema": _SPEC_SCHEMA,
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "dataset_label": "synthetic-issue810-component-v1",
        "dataset_identity_binding": "computed_by_pinned_H_MarketDatasetBuilder",
        "protocol_id": "synthetic-issue810-component-protocol-v1",
        "configuration_id": "H24_RIDGE_A1",
        "symbol": "BTCUSDT",
        "scenario": "BASEL0",
        "actor_start": _ACTOR_START,
        "actor_stop_exclusive": _ACTOR_STOP,
        "actor_advances": _ACTOR_ROWS,
        "reserve_advances": 72,
        "role_pair": ["NONRL", "CASH"],
        "row_count": _ROWS,
        "pulse_indices": list(_PULSES),
        "feature_names": list(_FEATURES),
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
            "actor_stop_exclusive": _ACTOR_STOP,
            "reserve_stop_inclusive": _RESERVE_STOP,
            "row_9503_used": False,
        },
        "full_b4_authorized": False,
        "outer_attempt_authenticated": False,
    }
    try:
        return _canonical(spec) == _canonical(expected)
    except (TypeError, ValueError):
        return False


def _runtime_report(
    *,
    plan: dict[str, object],
    plan_raw: bytes,
    runtime_pin: dict[str, object],
    attempt_uuid: str,
) -> dict[str, object]:
    from trade_rl.evaluation.runs.provenance import runtime_environment_manifest

    packages = runtime_pin.get("runtime_environment")
    package_versions = packages.get("packages") if type(packages) is dict else None
    if type(package_versions) is not dict:
        raise ValueError("pinned H runtime does not contain package versions")
    operator = plan["operator"]
    assert type(operator) is dict
    child = plan["child_source"]
    assert type(child) is dict
    numpy_file = Path(str(np.__file__)).resolve()
    h_site = Path(str(child["root"])) / ".venv" / "Lib" / "site-packages"
    if not numpy_file.is_relative_to(h_site):
        raise ValueError("NumPy was not imported from the pinned H virtual environment")
    trade_rl = sys.modules.get("trade_rl")
    nonrl = sys.modules.get("tools.tier_d_nonrl")
    trade_rl_file = getattr(trade_rl, "__file__", None)
    nonrl_file = getattr(nonrl, "__file__", None)
    expected_root = Path(str(child["root"]))
    observed_runtime = runtime_environment_manifest()
    runtime_digest = hashlib.sha256(_canonical(observed_runtime)).hexdigest()
    if (
        type(trade_rl_file) is not str
        or Path(trade_rl_file).resolve() != expected_root / "trade_rl" / "__init__.py"
        or type(nonrl_file) is not str
        or Path(nonrl_file).resolve()
        != expected_root / "tools" / "tier_d_nonrl" / "__init__.py"
        or type(packages) is not dict
        or observed_runtime != packages
        or runtime_digest != runtime_pin.get("runtime_environment_sha256")
        or np.__version__ != package_versions.get("numpy")
    ):
        raise ValueError(
            "H component imports did not resolve inside the pinned checkout"
        )
    return {
        "schema": _RUNTIME_SCHEMA,
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "attempt_uuid": attempt_uuid,
        "input_plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "operator_sha256": operator["sha256"],
        "python_executable": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
        "source_root": str(expected_root),
        "runtime_environment": observed_runtime,
        "runtime_environment_sha256": runtime_digest,
        "numpy_version": np.__version__,
        "numpy_origin": str(numpy_file),
        "trade_rl_origin": trade_rl_file,
        "tier_d_nonrl_origin": nonrl_file,
        "outer_attempt_authenticated": False,
        "full_b4_authorized": False,
    }


def _write_manifest(root: Path) -> None:
    entries: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if path.name == "output-manifest.json":
            continue
        if path.is_symlink():
            raise ValueError("component output contains a symbolic link")
        if path.is_file():
            size, digest = _sha256(path)
            entries[path.relative_to(root).as_posix()] = {
                "bytes": size,
                "sha256": digest,
            }
    if not entries:
        raise ValueError("component output manifest cannot be empty")
    _write_json(root / "output-manifest.json", entries)


def _bounded_diagnostic_text(value: str, *, maximum_bytes: int) -> str:
    raw = value.encode("utf-8", errors="replace")
    if len(raw) <= maximum_bytes:
        return raw.decode("utf-8")
    marker = b"...[truncated]"
    prefix = raw[: maximum_bytes - len(marker)]
    return prefix.decode("utf-8", errors="ignore") + marker.decode("ascii")


def _diagnostic_traceback(error: BaseException) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    traceback = error.__traceback__
    while traceback is not None and len(frames) < _MAX_FAILURE_TRACEBACK_FRAMES:
        code = traceback.tb_frame.f_code
        frames.append(
            {
                "file": _bounded_diagnostic_text(code.co_filename, maximum_bytes=256),
                "function": _bounded_diagnostic_text(code.co_name, maximum_bytes=128),
                "line": traceback.tb_lineno,
            }
        )
        traceback = traceback.tb_next
    return {"frames": frames, "truncated": traceback is not None}


def _write_failure_diagnostic(
    *,
    output_root: Path,
    plan_raw: bytes,
    plan: dict[str, object],
    attempt_uuid: str,
    phase: str,
    error: BaseException,
) -> None:
    operator = plan.get("operator")
    if type(operator) is not dict or type(operator.get("sha256")) is not str:
        raise ValueError("validated artificial plan lost its operator source pin")
    try:
        message = str(error)
    except BaseException:
        message = "<exception message unavailable>"
    record: dict[str, object] = {
        "schema": "issue810_artificial_h_operator_failure_v1",
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "attempt_uuid": attempt_uuid,
        "input_plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "operator_sha256": operator["sha256"],
        "phase": phase,
        "failure": {
            "type": _bounded_diagnostic_text(type(error).__name__, maximum_bytes=128),
            "message": _bounded_diagnostic_text(
                message, maximum_bytes=_MAX_FAILURE_MESSAGE_BYTES
            ),
            "traceback": _diagnostic_traceback(error),
        },
        "outer_attempt_authenticated": False,
        "full_b4_authorized": False,
        "research_authorized": False,
    }
    raw = _canonical(record)
    if not raw or len(raw) > _MAX_FAILURE_BYTES:
        raise ValueError("artificial operator failure diagnostic exceeds its bound")
    _write_json(output_root / "operator-failure.json", record)


def _protocol(spec: dict[str, Any], dataset: Any) -> dict[str, object]:
    dataset_recipe = spec["dataset_recipe"]
    cost_recipe = spec["cost_recipe"]
    risk_recipe = spec["risk_recipe"]
    allocator_recipe = spec["allocator_recipe"]
    forecast_recipe = spec["forecast_recipe"]
    execution_recipe = spec["execution_recipe"]
    assert type(dataset_recipe) is dict
    assert type(cost_recipe) is dict
    assert type(risk_recipe) is dict
    assert type(allocator_recipe) is dict
    assert type(forecast_recipe) is dict
    assert type(execution_recipe) is dict
    account_recipe = spec["account_recipe"]
    assert type(account_recipe) is dict
    stop = int(spec["actor_stop_exclusive"])
    return {
        "schema": "issue810_artificial_h_component_protocol_v1",
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "protocol_id": spec["protocol_id"],
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "dataset_label": spec["dataset_label"],
            "identity_binding": spec["dataset_identity_binding"],
            "dimensions": [
                int(dataset.n_bars),
                int(dataset.n_symbols),
                int(dataset.n_features),
            ],
            "feature_names": list(dataset.feature_names),
            "fit_symbols": [str(spec["symbol"])],
            "prediction_symbols": list(dataset.symbols),
            "start": str(dataset.timestamps[0]),
            "stop_exclusive": str(
                dataset.timestamps[-1]
                + np.timedelta64(int(dataset_recipe["interval_seconds"]), "s")
            ),
            "source": "in_memory_synthetic_bars_no_external_dataset_read",
        },
        "configuration": {
            "id": spec["configuration_id"],
            "horizon_hours": forecast_recipe["horizon_hours"],
            "ridge_alpha": forecast_recipe["ridge_alpha"],
            "forecast_origin": "predeclared_fixture_no_fit",
        },
        "account": f"{spec['configuration_id']}_{spec['symbol']}",
        "capital": {
            "currency": "USDT",
            "mode": "independent_symbol",
            "per_account": account_recipe["initial_cash"],
            "initial_quantities": account_recipe["initial_quantities"],
        },
        "scenario": spec["scenario"],
        "roles": [str(value).lower() for value in spec["role_pair"]],
        "actor": {
            "start": int(spec["actor_start"]),
            "stop": stop,
            "rows": int(spec["actor_advances"]),
            "vintage_starts": list(forecast_recipe["vintage_starts"]),
        },
        "terminal": {
            "stop_inclusive": int(execution_recipe["reserve_stop_inclusive"]),
            "advances": int(spec["reserve_advances"]),
        },
        "economics": {
            "dataset_fee": cost_recipe["fee_rate"],
            "dataset_spread": cost_recipe["spread_rate"],
            "impact_coefficient": cost_recipe["impact_coefficient"],
            "quote_notional_volume": dataset_recipe["quote_volume_per_bar"],
            "borrow_available": cost_recipe["borrow_available"],
            "borrow_rate": cost_recipe["borrow_rate"],
            "cash_rate": cost_recipe["cash_rate"],
            "funding_rate": cost_recipe["funding_rate"],
            "funding_event_interval_hours": int(dataset_recipe["funding_event_stride"]),
        },
        "policy": {
            "allocator": {
                "lower_weight": allocator_recipe["lower_weight"],
                "upper_weight": allocator_recipe["upper_weight"],
                "max_turnover": allocator_recipe["max_turnover"],
                "risk_aversion": allocator_recipe["risk_aversion"],
            },
            "risk": {
                "max_gross": risk_recipe["max_gross"],
                "max_abs_weight": risk_recipe["max_abs_weight"],
                "max_turnover": None,
                "drawdown_start": risk_recipe["drawdown_start"],
                "drawdown_stop": risk_recipe["drawdown_stop"],
                "emergency_turnover_override": risk_recipe[
                    "emergency_turnover_override"
                ],
                "fail_closed_tolerance": risk_recipe["fail_closed_tolerance"],
            },
        },
        "execution": {
            "kind": f"{execution_recipe['order_type']}/{execution_recipe['time_in_force']}",
            "latency_bars": execution_recipe["latency_bars"],
            "capacity": execution_recipe["capacity_source"],
            "max_participation_rate": cost_recipe["max_participation_rate"],
            "market_fee_overlay": cost_recipe["market_fee_overlay"],
            "market_spread_overlay": cost_recipe["market_spread_overlay"],
            "impact_coefficient": cost_recipe["impact_coefficient"],
            "random_seed": cost_recipe["random_seed"],
            "stochastic_slippage": cost_recipe["stochastic_slippage"],
            "decision_mark_source": execution_recipe["decision_mark_source"],
            "market_fill_price_source": execution_recipe["market_fill_price_source"],
        },
        "component_spec_sha256": hashlib.sha256(_canonical(spec)).hexdigest(),
        "full_b4_authorized": False,
        "outer_attempt_authenticated": False,
        "research_authorized": False,
    }


def _build_dataset(spec: dict[str, Any]) -> Any:
    from trade_rl.data.build.builder import MarketDatasetBuilder
    from trade_rl.data.build.economics import ExecutionEconomicsProfile
    from trade_rl.data.contracts import (
        FeatureKind,
        FeatureSpec,
        InstrumentContract,
        MarketBuildConfig,
        VolumeUnit,
    )
    from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries

    recipe = spec["dataset_recipe"]
    costs = spec["cost_recipe"]
    assert type(recipe) is dict
    assert type(costs) is dict
    row_count = int(spec["row_count"])
    symbols = tuple(str(value) for value in recipe["symbols"])
    ohlc = recipe["ohlc"]
    rules = recipe["instrument_rules"]
    assert type(ohlc) is dict and type(rules) is dict
    origin = str(recipe["timestamp_origin_utc"]).removesuffix("Z")
    timestamps = np.datetime64(origin, "ns") + np.arange(
        row_count, dtype=np.int64
    ) * np.timedelta64(int(recipe["interval_seconds"]), "s")
    pulses = set(_PULSES)
    source: dict[str, RawMarketSeries] = {}
    for symbol in symbols:
        selected = symbol == str(ohlc["pulse_symbol"])
        open_price = np.full(row_count, float(ohlc["open"]), dtype=np.float64)
        high = np.full(row_count, float(ohlc["ordinary_high"]), dtype=np.float64)
        low = np.full(row_count, float(ohlc["ordinary_low"]), dtype=np.float64)
        close = np.full(row_count, float(ohlc["close"]), dtype=np.float64)
        if selected:
            indices = np.asarray(sorted(pulses), dtype=np.int64)
            high[indices] = float(ohlc["selected_pulse_high"])
            low[indices] = float(ohlc["selected_pulse_low"])
        event_count = np.zeros(row_count, dtype=np.int32)
        event_count[:: int(recipe["funding_event_stride"])] = 1
        available_at = timestamps.copy()
        source[symbol] = RawMarketSeries(
            timestamps=timestamps,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=np.full(
                row_count, float(recipe["quote_volume_per_bar"]), dtype=np.float64
            ),
            funding_rate=np.full(
                row_count, float(recipe["funding_rate"]), dtype=np.float64
            ),
            tradable=np.full(row_count, recipe["tradable"], dtype=np.bool_),
            funding_available=event_count > 0,
            available_at=available_at,
            funding_event_count=event_count,
            funding_price_rate=np.full(
                row_count, float(recipe["funding_price_rate"]), dtype=np.float64
            ),
        )
    names = tuple(str(value) for value in spec["feature_names"])
    feature_kinds = tuple(str(value) for value in recipe["feature_kinds"])
    feature_specs = tuple(
        FeatureSpec(name, FeatureKind[kind])
        for name, kind in zip(names, feature_kinds, strict=True)
    )
    origin_datetime = datetime.fromisoformat(origin).replace(tzinfo=UTC)
    instruments_list = []
    for symbol in symbols:
        rule = rules[symbol]
        if type(rule) is not dict:
            raise ValueError("artificial instrument rule is malformed")
        instruments_list.append(
            InstrumentContract(
                symbol,
                origin_datetime,
                volume_unit=VolumeUnit[str(recipe["volume_unit"])],
                tick_size=float(rule["tick_size"]),
                lot_size=float(rule["lot_size"]),
                minimum_notional=float(rule["minimum_notional"]),
            )
        )
    instruments = tuple(instruments_list)
    economics = ExecutionEconomicsProfile(
        name="Artificial Task18-shaped component economics",
        fee_rate=float(costs["fee_rate"]),
        spread_rate=float(costs["spread_rate"]),
        max_participation_rate=float(costs["max_participation_rate"]),
        borrow_available=costs["borrow_available"],
        borrow_rate=float(costs["borrow_rate"]),
    )
    dataset = MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe=str(recipe["base_timeframe"]), features=feature_specs
        )
    ).build(
        InMemoryMarketDataSource(source),
        instruments,
        execution_economics=economics,
    )
    expected_available_at = np.broadcast_to(
        timestamps[:, None], (row_count, len(symbols))
    )
    if (
        dataset.n_bars != row_count
        or dataset.symbols != symbols
        or dataset.feature_names != names
        or dataset.dataset_id
        == "e916ff716fb5e4b90d5805eaa79200824b09f103984fa845bbeaedb57bef194f"
        or not np.array_equal(dataset.timestamps, timestamps)
        or not np.array_equal(
            dataset.resolved_array("available_at"), expected_available_at
        )
    ):
        raise ValueError("synthetic Dataset does not satisfy the fixed component shape")
    selected_index = dataset.symbols.index(str(spec["symbol"]))
    selected_symbol_active = dataset.resolved_array("asset_active")[:, selected_index]
    if not bool(np.all(selected_symbol_active)):
        raise ValueError("synthetic selected asset is not active for the full window")
    return dataset


def _build_stream(dataset: Any, spec: dict[str, Any]) -> Any:
    from trade_rl.strategies.forecasts.simple_return import (
        SimpleReturnRidgeModel,
        SimpleReturnTrainingSet,
    )
    from trade_rl.strategies.forecasts.simple_stream import (
        FrozenSimpleReturnStream,
        SimpleReturnPacket,
        SimpleReturnVintage,
    )
    from trade_rl.strategies.forecasts.stream import ForecastBlock
    from trade_rl.strategies.forecasts.supervised import (
        build_causal_forecast_training_set,
    )

    recipe = spec["forecast_recipe"]
    assert type(recipe) is dict
    horizon_hours = int(recipe["horizon_hours"])
    alpha = float(recipe["ridge_alpha"])
    expected_pulse_return = float(recipe["pulse_expected_simple_return"])
    expected_nonpulse_return = float(recipe["nonpulse_expected_simple_return"])
    intercept = float(recipe["intercept"])
    vintage_starts = tuple(int(value) for value in recipe["vintage_starts"])
    fit_symbol_indices = tuple(int(value) for value in recipe["fit_symbol_indices"])
    prediction_symbols = tuple(str(value) for value in recipe["prediction_symbols"])
    blocks = tuple(
        ForecastBlock(
            fit_cutoff=dataset.timestamps[start] - np.timedelta64(1, "ns"),
            model_fit_time=dataset.timestamps[start],
            prediction_start=dataset.timestamps[start],
            prediction_stop=dataset.timestamps[stop],
            inference_delay_seconds=0,
        )
        for start, stop in zip(
            vintage_starts, (*vintage_starts[1:], _ACTOR_STOP), strict=True
        )
    )
    pulses = set(_PULSES)
    vintages = []
    packets = []
    feature_indices = tuple(int(value) for value in recipe["feature_indices"])
    selected_index = dataset.symbols.index(str(spec["symbol"]))
    base_pulse = float(dataset.features[_ACTOR_START, selected_index, 0])
    pulse_value = float(dataset.features[min(pulses), selected_index, 0])
    if (
        not math.isfinite(base_pulse)
        or not math.isfinite(pulse_value)
        or pulse_value <= base_pulse
    ):
        raise ValueError("synthetic pulse feature is not the predeclared float32 range")
    if intercept != expected_nonpulse_return:
        raise ValueError("fixed projection does not reproduce the nonpulse return")
    coefficient = (expected_pulse_return - intercept) / (pulse_value - base_pulse)
    for block in blocks:
        training = SimpleReturnTrainingSet(
            build_causal_forecast_training_set(
                dataset,
                feature_indices=feature_indices,
                fit_symbol_indices=fit_symbol_indices,
                fit_cutoff=block.fit_cutoff,
                horizon_hours=horizon_hours,
            )
        )
        model = SimpleReturnRidgeModel(
            feature_indices=feature_indices,
            feature_mean=np.asarray(
                (base_pulse, *([0.0] * (len(feature_indices) - 1)))
            ),
            feature_scale=np.ones(len(feature_indices), dtype=np.float64),
            coefficients=np.asarray((coefficient, 0.0, 0.0, 0.0, 0.0, 0.0)),
            intercept=intercept,
            horizon_hours=horizon_hours,
            alpha=alpha,
            n_samples=training.n_samples,
            fit_cutoff=training.fit_cutoff,
            fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
        )
        vintage = SimpleReturnVintage(
            block=block,
            training=training,
            model=model,
            prediction_symbols=prediction_symbols,
        )
        vintages.append(vintage)
        start = int(np.searchsorted(dataset.timestamps, block.prediction_start))
        stop = int(np.searchsorted(dataset.timestamps, block.prediction_stop))
        for index in range(start, stop):
            as_of = dataset.timestamps[index]
            horizon_end = as_of + np.timedelta64(horizon_hours, "h")
            if horizon_end > dataset.timestamps[-1]:
                raise ValueError("synthetic forecast horizon escapes the input tape")
            for symbol_index, symbol in enumerate(dataset.symbols):
                features = tuple(
                    float(value)
                    for value in np.asarray(
                        dataset.features[index, symbol_index], dtype=np.float64
                    )
                )
                predicted = model.predict_selected(
                    np.asarray(features, dtype=np.float64)
                )
                packets.append(
                    SimpleReturnPacket(
                        symbol=symbol,
                        as_of=as_of,
                        source_available_at=dataset.resolved_array("available_at")[
                            index, symbol_index
                        ],
                        forecast_available_at=max(as_of, block.model_fit_time),
                        horizon_end=horizon_end,
                        horizon_seconds=horizon_hours * 3600,
                        expected_simple_return=predicted,
                        fit_prefix_marginal_variance=model.fit_prefix_marginal_variance,
                        vintage_digest=vintage.digest,
                        feature_values=features,
                        decision_close=float(dataset.close[index, symbol_index]),
                    )
                )
    stream = FrozenSimpleReturnStream(
        dataset.dataset_id, tuple(vintages), tuple(packets)
    )
    if len(stream.packets) != _ACTOR_ROWS * 2:
        raise ValueError(
            "synthetic forecast stream does not cover both symbols and all actors"
        )
    for packet in stream.packets:
        index = int(np.searchsorted(dataset.timestamps, packet.as_of))
        expected_return = (
            expected_pulse_return
            if packet.symbol == str(spec["symbol"]) and index in pulses
            else expected_nonpulse_return
        )
        if not math.isclose(
            packet.expected_simple_return,
            expected_return,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("synthetic forecast differs from its fixed projection")
    return stream


def _build_prepared(
    dataset: Any, stream: Any, artifact: Any, spec: dict[str, Any]
) -> tuple[Any, tuple[Any, ...], dict[str, object]]:
    from tools.tier_d_nonrl.tier_d_native import cost_rows
    from trade_rl.evaluation.allocation_market_inputs import (
        PreparedAllocationMarketInputs,
    )

    forecast_recipe = spec["forecast_recipe"]
    assert type(forecast_recipe) is dict
    horizon = int(forecast_recipe["horizon_hours"])
    scenario = str(spec["scenario"])
    protocol = _protocol(spec, dataset)
    estimates = cost_rows(dataset, protocol, horizon, scenario)
    by_symbol = tuple(
        tuple(row for row in estimates if row.symbol == symbol)
        for symbol in dataset.symbols
    )
    prepared = PreparedAllocationMarketInputs(dataset, stream, artifact, by_symbol)
    return prepared, estimates, protocol


def _assert_selected_symbol_orders(
    active_orders: object, *, selected_symbol_index: int
) -> None:
    if type(selected_symbol_index) is not int or selected_symbol_index < 0:
        raise ValueError("selected symbol index is malformed")
    if type(active_orders) is not list:
        raise ValueError("native active-order facts are malformed")
    for order in active_orders:
        if type(order) is not dict:
            raise ValueError("native active-order facts are malformed")
        intent = order.get("intent")
        if type(intent) is not dict:
            raise ValueError("native active-order intent is malformed")
        symbol_index = intent.get("symbol_index")
        if type(symbol_index) is not int or symbol_index < 0:
            raise ValueError("native active-order symbol index is malformed")
        if symbol_index != selected_symbol_index:
            raise ValueError("native active order targets a nonselected symbol")


def _assert_reserve_boundary_fills(
    actor_fill_indices: list[int],
    reserve_fill_indices: list[int],
    *,
    actor_stop_exclusive: int,
) -> None:
    if (
        type(actor_stop_exclusive) is not int
        or any(type(index) is not int for index in actor_fill_indices)
        or any(type(index) is not int for index in reserve_fill_indices)
    ):
        raise ValueError("native fill index facts are malformed")
    if actor_stop_exclusive not in actor_fill_indices:
        raise ValueError("late actor exposure did not fill at the actor boundary")
    if actor_stop_exclusive + 1 not in reserve_fill_indices:
        raise ValueError("reserve flatten did not fill on its first processing bar")


def _run_pair(
    output_root: Path,
    spec: dict[str, Any],
    dataset: Any,
    stream: Any,
    artifact: Any,
) -> dict[str, object]:
    from tools.tier_d_nonrl.tier_d_native import carrier, declare_pair, execution_folds
    from tools.tier_d_nonrl.tier_d_records import save_gzip
    from trade_rl.evaluation.allocation_terminal_execution import (
        run_terminal_allocation_execution,
    )
    from trade_rl.evaluation.runs import build_candidate_run_provenance

    prepared, estimates, protocol = _build_prepared(dataset, stream, artifact, spec)
    account = spec["account_recipe"]
    forecast = spec["forecast_recipe"]
    execution = spec["execution_recipe"]
    assert type(account) is dict
    assert type(forecast) is dict
    assert type(execution) is dict
    if (
        float(account["initial_cash"]) != 100000.0
        or account["initial_quantities"] != [0.0, 0.0]
        or account["selected_symbol"] != spec["symbol"]
        or account["other_symbol_always_zero"] is not True
        or int(execution["actor_stop_exclusive"]) != int(spec["actor_stop_exclusive"])
        or int(execution["reserve_stop_inclusive"])
        - int(execution["actor_stop_exclusive"])
        != int(spec["reserve_advances"])
        or execution["row_9503_used"] is not False
    ):
        raise ValueError(
            "synthetic account or terminal recipe differs from the fixed scope"
        )
    provenance = build_candidate_run_provenance()
    selected_symbol = str(spec["symbol"])
    selected_symbol_index = dataset.symbols.index(selected_symbol)
    selected_estimates = tuple(
        row for row in estimates if row.symbol == selected_symbol
    )
    # These declarations only create native plan identities for the two
    # artificial arms. They carry no review or research authority.
    pair = declare_pair(
        prepared,
        selected_estimates,
        protocol,
        horizon=int(forecast["horizon_hours"]),
        symbol_index=selected_symbol_index,
        account=f"{spec['configuration_id']}_{spec['symbol']}",
        scenario=str(spec["scenario"]),
        configuration=str(spec["configuration_id"]),
        provenance=provenance,
    )
    folds = execution_folds(
        tuple(int(value) for value in forecast["vintage_starts"]),
        int(spec["actor_stop_exclusive"]),
    )
    closures: dict[str, object] = {}
    plans: dict[str, object] = {}
    roles = tuple(str(value).lower() for value in spec["role_pair"])
    for role, plan in zip(roles, pair.controls.plans, strict=True):
        environment = carrier(
            prepared,
            selected_estimates,
            protocol,
            horizon=int(forecast["horizon_hours"]),
            symbol_index=selected_symbol_index,
            account=f"{spec['configuration_id']}_{spec['symbol']}",
            scenario=str(spec["scenario"]),
            role=role,
        )
        native = run_terminal_allocation_execution(folds, environment, plan)
        payload = native.payload
        receipt_path = output_root / "receipts" / f"{role}.json.gz"
        receipt_path.parent.mkdir(exist_ok=True)
        save_gzip(output_root, f"receipts/{role}.json.gz", payload)
        rows = payload["native_receipt"]["rows"]
        closing = payload["closing_rows"]
        fill_indices = [
            int(row["native"]["execution"]["next_index"])
            for row in rows
            if row["native"] is not None
            and row["native"]["execution"]["fill_count"] > 0
        ]
        closing_fill_indices = [
            int(row["processing_index"])
            for row in closing
            if row["native"]["execution"]["fill_count"] > 0
        ]
        if len(rows) != int(spec["actor_advances"]) or len(closing) != int(
            spec["reserve_advances"]
        ):
            raise ValueError(
                "native artificial role did not complete the fixed row roster"
            )
        if payload["settlement_complete"] is not True or payload["final_active_orders"]:
            raise ValueError(
                "native artificial role did not settle with no active orders"
            )
        if any(float(value) != 0.0 for value in payload["final_book"]["quantities"]):
            raise ValueError("native artificial role did not finish flat")
        selected_index = dataset.symbols.index(str(spec["symbol"]))
        for row in (*rows, *closing):
            native_row = row.get("native")
            if type(native_row) is not dict:
                raise ValueError("completed artificial row has no native facts")
            _assert_selected_symbol_orders(
                native_row.get("active_orders"),
                selected_symbol_index=selected_index,
            )
        nonselected = [
            float(value)
            for row in (*rows, *closing)
            if row.get("native") is not None
            for index, value in enumerate(row["native"]["book"]["quantities"])
            if index != selected_index
        ]
        if any(value != 0.0 for value in nonselected):
            raise ValueError("native artificial role traded the nonselected asset")
        if role == "nonrl":
            for start, stop in zip(
                tuple(int(value) for value in forecast["vintage_starts"]),
                (
                    *tuple(int(value) for value in forecast["vintage_starts"])[1:],
                    _ACTOR_STOP,
                ),
                strict=True,
            ):
                if not any(start < index < stop for index in fill_indices):
                    raise ValueError(
                        "fixed projection produced no NONRL fill in a declared vintage"
                    )
            _assert_reserve_boundary_fills(
                fill_indices,
                closing_fill_indices,
                actor_stop_exclusive=int(execution["actor_stop_exclusive"]),
            )
        elif fill_indices or closing_fill_indices:
            raise ValueError("CASH direct-HOLD control unexpectedly filled an order")
        closures[role] = {
            "actor_rows": len(rows),
            "reserve_rows": len(closing),
            "actor_fill_processing_indices": fill_indices,
            "reserve_fill_processing_indices": closing_fill_indices,
            "terminal_flat": True,
            "active_orders": 0,
            "terminal_plan_digest": plan.digest,
        }
        plans[role.upper()] = plan.digest
        del rows, closing, native, payload, environment
    return {
        "schema": "issue810_artificial_h_component_result_v1",
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "component_spec_sha256": hashlib.sha256(_canonical(spec)).hexdigest(),
        "synthetic_dataset_id": dataset.dataset_id,
        "synthetic_protocol_id": spec["protocol_id"],
        "roles": closures,
        "terminal_plan_digests": plans,
        "outer_attempt_authenticated": False,
        "full_b4_authorized": False,
        "research_authorized": False,
    }


def _load_h_modules(source_root: Path, observer_root: Path) -> None:
    if (
        Path.cwd().resolve() != source_root
        or Path(sys.executable).resolve()
        != source_root / ".venv" / "Scripts" / "python.exe"
    ):
        raise ValueError("operator must run from the exact pinned H checkout and venv")
    resolved_paths = [Path(entry or Path.cwd()).resolve() for entry in sys.path]
    if any(
        path == observer_root or observer_root in path.parents
        for path in resolved_paths
    ):
        raise ValueError("Q comparator source leaked into isolated H import search")
    sys.path.insert(0, str(source_root))
    import tools.tier_d_nonrl  # noqa: PLC0415
    import trade_rl  # noqa: PLC0415

    if (
        Path(trade_rl.__file__).resolve() != source_root / "trade_rl" / "__init__.py"
        or Path(tools.tier_d_nonrl.__file__).resolve()
        != source_root / "tools" / "tier_d_nonrl" / "__init__.py"
    ):
        raise ValueError(
            "isolated H API imports resolved outside the pinned source root"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--plan", required=True)
    arguments = parser.parse_args(argv)
    plan_path = Path(arguments.plan)
    plan_raw, plan = _read_plan(plan_path)
    operator_path = Path(__file__).resolve()
    output_root, runtime_pin, attempt_uuid = _assert_plan(
        plan, plan_raw=plan_raw, operator_path=operator_path, plan_path=plan_path
    )
    output_root.mkdir(parents=True, exist_ok=False)
    phase = "operator_setup"
    try:
        child = plan["child_source"]
        operator = plan["operator"]
        spec_value = plan["component_spec"]
        assert (
            type(child) is dict and type(operator) is dict and type(spec_value) is dict
        )
        spec = cast(dict[str, Any], spec_value)
        source_root = Path(str(child["root"])).resolve()
        observer_root = Path(str(operator["path"])).resolve().parents[1]
        _load_h_modules(source_root, observer_root)

        phase = "runtime_report"
        runtime_report = _runtime_report(
            plan=plan,
            plan_raw=plan_raw,
            runtime_pin=runtime_pin,
            attempt_uuid=attempt_uuid,
        )
        _write_json(output_root / "operator-runtime.json", runtime_report)

        phase = "build_dataset"
        dataset = _build_dataset(spec)
        phase = "build_forecast"
        stream = _build_stream(dataset, spec)
        from trade_rl.strategies.forecasts.simple_stream_io import (
            publish_simple_return_stream_artifact,
        )

        forecast_path = output_root / "forecasts" / "simple-return-stream.json"
        forecast_path.parent.mkdir(exist_ok=False)
        phase = "publish_forecast"
        forecast = publish_simple_return_stream_artifact(forecast_path, stream)

        phase = "run_pair"
        result = _run_pair(output_root, spec, dataset, stream, forecast)
        result["forecast_stream"] = {
            "path": forecast_path.relative_to(output_root).as_posix(),
            "sha256": forecast.digest,
            "dataset_id": forecast.dataset_id,
        }
        phase = "write_component_result"
        _write_json(output_root / "component-result.json", result)
        phase = "write_output_manifest"
        _write_manifest(output_root)
    except BaseException as error:
        for success_marker in ("component-result.json", "output-manifest.json"):
            try:
                (output_root / success_marker).unlink(missing_ok=True)
            except BaseException:
                # A cleanup failure must not replace the original operator error.
                pass
        try:
            _write_failure_diagnostic(
                output_root=output_root,
                plan_raw=plan_raw,
                plan=plan,
                attempt_uuid=attempt_uuid,
                phase=phase,
                error=error,
            )
        except BaseException:
            # The primary operator failure remains authoritative if journaling fails.
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
