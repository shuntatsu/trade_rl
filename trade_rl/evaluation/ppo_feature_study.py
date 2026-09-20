"""Write-once paired PPO feature ablation for development data only."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.artifacts import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.data.features.price_channels import with_price_channels
from trade_rl.evaluation.directional import evaluate_directional_arm
from trade_rl.evaluation.directional_candidates import (
    PPO_TIMESTEPS,
    fit_directional_candidate,
)
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.evaluation.directional_study import (
    SOURCE_ARTIFACT_DIGEST,
    SOURCE_DATASET_ID,
    SOURCE_STUDY_DIGEST,
    development_indices,
)
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.evaluation.metrics import compound_return
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.quantities import parse_quantity, project_quantity

SCHEMA = "ppo_feature_ablation_protocol_v1"
SEEDS = (0, 1, 2, 3, 4)
BASELINE_FACTOR = "baseline"
CANDIDATE_FACTOR = "btc_relative"
FACTORS = (BASELINE_FACTOR, CANDIDATE_FACTOR)
RELATIVE_FEATURE_NAMES = (
    "1h__relative_return_to_btc_1bar",
    "4h__relative_return_to_btc_1bar",
    "1d__relative_return_to_btc_1bar",
)
TRAINING_RISK = PreTradeRiskConfig(
    max_gross=0.5,
    max_abs_weight=0.1,
    max_turnover=None,
    drawdown_start=0.1,
    drawdown_stop=0.2,
)
_FLAT_TOLERANCE = 1e-10


def _sha256(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _write_once(path: Path, payload: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload))


def _interval_year_slices(
    timestamps: np.ndarray, start: int, stop: int
) -> dict[str, list[int]]:
    years = timestamps[start:stop].astype("datetime64[Y]").astype(str)
    result: dict[str, list[int]] = {}
    cursor = 0
    for year in np.unique(years):
        count = int(np.count_nonzero(years == year))
        if not np.all(years[cursor : cursor + count] == year):
            raise ValueError("interval-start year buckets are not contiguous")
        result[str(year)] = [cursor, cursor + count]
        cursor += count
    if cursor != stop - start:
        raise ValueError("interval-start year slices do not cover the evaluation")
    return result


def _expected_year_slices(
    timestamps: np.ndarray, start: int, stop: int
) -> dict[str, list[int]]:
    result = _interval_year_slices(timestamps, start, stop)
    if result != {"2023": [0, 8_760], "2024": [8_760, 17_544]}:
        raise ValueError("interval-start year coverage differs from preregistration")
    return result


def _source_snapshot_bytes(provenance: dict[str, Any]) -> bytes:
    package_root = Path(__file__).resolve().parents[1]
    manifest = provenance["implementation"]["files"]
    buffer = io.BytesIO()
    with ZipFile(
        buffer, mode="w", compression=ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for row in manifest:
            relative = row["path"]
            path = package_root / relative
            raw = path.read_bytes()
            if _sha256(raw) != row["sha256"]:
                raise ValueError(f"source changed while snapshotting: {relative}")
            info = ZipInfo(f"trade_rl/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw, compress_type=ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def _verify_source_snapshot(root: Path, protocol: dict[str, Any]) -> None:
    path = root / "source-snapshot.zip"
    raw = path.read_bytes()
    if _sha256(raw) != protocol["source_snapshot_sha256"]:
        raise ValueError("source snapshot digest mismatch")
    expected = {
        f"trade_rl/{row['path']}": row["sha256"]
        for row in protocol["provenance"]["implementation"]["files"]
    }
    with ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("source snapshot file roster mismatch")
        for name, digest in expected.items():
            if _sha256(archive.read(name)) != digest:
                raise ValueError(f"source snapshot file digest mismatch: {name}")


def expected_protocol(source: Path) -> dict[str, Any]:
    """Resolve and bind the one-factor trial before candidate economics run."""
    identity = inspect_published_market_dataset_artifact(source / "dataset")
    original = load_market_dataset_artifact(source / "dataset")
    if (
        original.dataset_id != SOURCE_DATASET_ID
        or identity.artifact_digest != SOURCE_ARTIFACT_DIGEST
    ):
        raise ValueError("source is not the frozen successor Dataset artifact")
    plan = inspect_study(source / "study").plan
    if plan.digest != SOURCE_STUDY_DIGEST or plan.dataset_id != SOURCE_DATASET_ID:
        raise ValueError("source StudyPlan is not the frozen successor")
    baseline = plan.baseline_config
    if (
        tuple(original.feature_names[index] for index in baseline.feature_indices)
        != baseline.feature_names
    ):
        raise ValueError("baseline feature indices do not resolve to their names")
    if (
        tuple(original.symbols[index] for index in baseline.fit_symbol_indices)
        != baseline.fit_symbol_names
    ):
        raise ValueError("baseline fit-symbol indices do not resolve to their names")
    if baseline.fit_symbol_indices != tuple(range(original.n_symbols)):
        raise ValueError("the fixed PPO fit scope must include the full Dataset roster")
    if np.datetime64(baseline.fit_cutoff) > np.datetime64("2023-01-01T00"):
        raise ValueError("training cutoff overlaps the development evaluation")

    candidate_names = baseline.feature_names + RELATIVE_FEATURE_NAMES
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("candidate feature roster contains duplicate names")
    try:
        candidate_indices = tuple(
            original.feature_names.index(name) for name in candidate_names
        )
        relative_indices = tuple(
            original.feature_names.index(name) for name in RELATIVE_FEATURE_NAMES
        )
    except ValueError as error:
        raise ValueError(
            "a preregistered BTC-relative feature is unavailable"
        ) from error
    if len(relative_indices) != 3:
        raise ValueError("candidate must add the 1h, 4h and 1d relative features")

    dataset = with_price_channels(original)
    del original
    start, stop = development_indices(dataset)
    if stop - start != 17_544:
        raise ValueError("development interval count differs from preregistration")
    year_slices = _expected_year_slices(dataset.timestamps, start, stop)
    scenario_specs = {
        "base": {"cost_multiplier": 1.0, "latency_bars": 0},
        "cost_2x": {"cost_multiplier": 2.0, "latency_bars": 0},
        "latency_1": {"cost_multiplier": 1.0, "latency_bars": 1},
    }
    scenarios: dict[str, dict[str, Any]] = {}
    for name, spec in scenario_specs.items():
        latency_bars = int(spec["latency_bars"])
        execution = replace(
            DIRECTIONAL_BASE_EXECUTION_COST,
            multiplier=spec["cost_multiplier"],
            order_latency_bars=latency_bars,
        )
        scenarios[name] = {
            **spec,
            "latency_bars": latency_bars,
            "execution_policy_digest": MarketExecutor(
                dataset, execution
            ).execution_policy_digest,
        }
    provenance = build_candidate_run_provenance()
    snapshot = _source_snapshot_bytes(provenance)
    risk = asdict(TRAINING_RISK)
    initial_capital = 10_000.0
    return {
        "schema": SCHEMA,
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_artifact_digest": identity.artifact_digest,
        "source_study_digest": plan.digest,
        "source_plan_sha256": _sha256((source / "study" / "plan.json").read_bytes()),
        "evaluation_dataset_id": dataset.dataset_id,
        "symbols": list(dataset.symbols),
        "baseline_config": baseline.to_payload(),
        "factors": {
            BASELINE_FACTOR: {
                "feature_names": list(baseline.feature_names),
                "feature_indices": list(baseline.feature_indices),
                "observation_width": 3 * len(baseline.feature_names) + 2,
            },
            CANDIDATE_FACTOR: {
                "feature_names": list(candidate_names),
                "feature_indices": list(candidate_indices),
                "added_feature_names": list(RELATIVE_FEATURE_NAMES),
                "added_feature_indices": list(relative_indices),
                "observation_width": 3 * len(candidate_names) + 2,
            },
        },
        "seeds": list(SEEDS),
        "training": {
            "requested_timesteps": PPO_TIMESTEPS,
            "layout": "sequential",
            "fit_symbol_indices": list(baseline.fit_symbol_indices),
            "fit_cutoff": baseline.fit_cutoff,
            "gross_budget": 0.1,
            "initial_capital": initial_capital,
            "settle_terminal_position": True,
            "normalize_features": False,
            "risk": risk,
        },
        "evaluation": {
            "start_index": start,
            "stop_index": stop,
            "start_timestamp": str(dataset.timestamps[start]),
            "stop_timestamp": str(dataset.timestamps[stop]),
            "interval_count": stop - start,
            "interval_year_basis": "decision interval start timestamp",
            "interval_year_slices": year_slices,
            "interval_year_counts": {
                year: bounds[1] - bounds[0] for year, bounds in year_slices.items()
            },
            "interval_year_mapping_oracle": "return[i] belongs to timestamps[start+i]; terminal endpoint at 2025-01-01T00 closes the 2024 interval",
            "evaluator_year_returns_use": "diagnostic only; evaluate_directional_arm reports by interval-end timestamp and its qualified flag is not the admission oracle",
            "one_independent_10000_account_per_symbol": True,
            "initial_capital": initial_capital,
            "gross_budget": 0.1,
            "risk": risk,
            "execution_cost": asdict(DIRECTIONAL_BASE_EXECUTION_COST),
            "terminal_flat_required": True,
            "zero_terminations_required": True,
            "no_active_order_remainder_required": True,
            "ledger_drawdown_limit": 0.2,
            "scenarios": scenarios,
            "stress_names": ["cost_2x", "latency_1"],
        },
        "admission": {
            "required_passing_seeds_per_symbol": 4,
            "required_symbols": 4,
            "required_positive_paired_seed_deltas": 4,
            "candidate_hard_guards": "all candidate base and stress seed-symbol cells;zero terminations;terminal flat;no active order remainder;ledger DD<=0.20",
            "candidate_seed_pass": "same >=4/5 seeds have positive full and every-year returns in base and both stresses after global hard guards",
            "candidate_medians": "median of all five seeds;full and every year return positive",
            "relative_pass": "same symbol;>=4/5 strictly positive paired full-return deltas;positive median full and each-year delta;baseline and candidate complete, nonterminated and flat;candidate DD<=0.20",
            "stress_pass": "same >=4/5 seeds pass full and each-year positive-return votes in base and both stresses with ledger DD<=0.20, no termination, terminal flatness and complete clock; all-five-seed medians are positive in full and each year in base and both stresses",
            "common_symbol_set": "at least four symbols must pass absolute, paired-relative and stress screens together",
            "decision_on_pass": "PROSPECTIVE_PAPER_REQUIRED",
            "development_only": True,
            "production_eligible": False,
        },
        "provenance": provenance,
        "source_snapshot_sha256": _sha256(snapshot),
        "development_data_reused": True,
        "unused_data_accessed": False,
        "production_eligible": False,
    }


def reserve_study(source: Path, root: Path) -> dict[str, Any]:
    protocol = expected_protocol(source)
    snapshot = _source_snapshot_bytes(protocol["provenance"])
    root.mkdir(parents=True, exist_ok=False)
    with (root / "source-snapshot.zip").open("xb") as stream:
        stream.write(snapshot)
    _write_once(root / "protocol.json", protocol)
    _write_once(root / "protocol.digest.json", {"digest": content_digest(protocol)})
    for factor in FACTORS:
        (root / factor).mkdir(exist_ok=False)
    print(
        f"Prepared fixed PPO feature ablation: {content_digest(protocol)}", flush=True
    )
    return protocol


def validate_protocol(source: Path, root: Path) -> dict[str, Any]:
    expected = expected_protocol(source)
    if (root / "protocol.json").read_bytes() != canonical_json_bytes(expected):
        raise ValueError(
            "study protocol differs from current source/data/runtime contract"
        )
    digest = content_digest(expected)
    if (root / "protocol.digest.json").read_bytes() != canonical_json_bytes(
        {"digest": digest}
    ):
        raise ValueError("study protocol digest changed")
    _verify_source_snapshot(root, expected)
    return expected


def interval_start_year_returns(
    returns: tuple[float, ...] | list[float],
    year_slices: dict[str, list[int]] | dict[str, tuple[int, int]],
) -> dict[str, float]:
    """Compound returns by decision-interval start year, excluding endpoint drift."""
    values = tuple(returns)
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("returns must contain a non-empty finite numeric sequence")
    ranges: list[tuple[int, int, str]] = []
    for year, bounds in year_slices.items():
        if (
            not isinstance(year, str)
            or not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(
                isinstance(bound, bool) or not isinstance(bound, int)
                for bound in bounds
            )
        ):
            raise ValueError("year slices must be year-to-two-integer-bound mappings")
        start, stop = bounds
        if not 0 <= start < stop <= len(values):
            raise ValueError("year slices must be non-empty and within returns")
        ranges.append((start, stop, year))
    ranges.sort()
    cursor = 0
    result: dict[str, float] = {}
    for start, stop, year in ranges:
        if start != cursor:
            raise ValueError("year slices must cover every interval exactly once")
        result[year] = compound_return(
            tuple(float(value) for value in values[start:stop])
        )
        cursor = stop
    if cursor != len(values):
        raise ValueError("year slices must cover every interval exactly once")
    return result


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite numeric evidence")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite numeric evidence")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-11)


def _validate_ledger_payload(
    ledger: object,
    *,
    row: dict[str, Any],
    protocol: dict[str, Any],
    scenario_name: str,
    symbol: str,
) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise ValueError(f"{symbol} ledger evidence must be an object")
    evaluation = protocol["evaluation"]
    scenario = evaluation["scenarios"][scenario_name]
    symbols = tuple(protocol["symbols"])
    interval_count = evaluation["interval_count"]
    intervals = ledger.get("intervals")
    if (
        ledger.get("schema_version") != "shared_cash_replay_ledger_v1"
        or ledger.get("dataset_id") != protocol["evaluation_dataset_id"]
        or ledger.get("start_index") != evaluation["start_index"]
        or ledger.get("stop_index") != evaluation["stop_index"]
        or ledger.get("execution_policy_digest") != scenario["execution_policy_digest"]
        or not isinstance(intervals, list)
        or len(intervals) != interval_count
    ):
        raise ValueError(f"{symbol} ledger schema/identity/interval count mismatch")

    row_returns = tuple(row["returns"])
    previous: dict[str, Any] | None = None
    for offset, interval in enumerate(intervals):
        expected_start = evaluation["start_index"] + offset
        if (
            not isinstance(interval, dict)
            or interval.get("start_index") != expected_start
            or interval.get("next_index") != expected_start + 1
        ):
            raise ValueError(f"{symbol} ledger interval clock is incomplete")
        if not _close(
            _finite_number(
                interval.get("interval_net_return"), field=f"{symbol} ledger return"
            ),
            row_returns[offset],
        ):
            raise ValueError(f"{symbol} ledger return differs from replay returns")
        for field in (
            "cash_before",
            "cash_after",
            "portfolio_value_before",
            "portfolio_value_after",
            "total_cost_before",
            "total_cost_after",
            "interval_cost",
            "interval_funding",
            "interval_borrow_cost",
            "interval_dividend",
            "interval_cash_interest",
            "funding_pnl_before",
            "funding_pnl_after",
            "borrow_cost_before",
            "borrow_cost_after",
            "turnover_total_before",
            "turnover_total_after",
            "max_drawdown_before",
            "max_drawdown_after",
        ):
            _finite_number(interval.get(field), field=f"{symbol} ledger {field}")
        before_value = float(interval["portfolio_value_before"])
        after_value = float(interval["portfolio_value_after"])
        if before_value <= 0.0 or after_value <= 0.0:
            raise ValueError(f"{symbol} ledger portfolio values must stay positive")
        implied_return = after_value / before_value - 1.0
        if not _close(implied_return, row_returns[offset]):
            raise ValueError(f"{symbol} ledger equity chain differs from returns")
        for interval_field, before_field, after_field in (
            ("interval_cost", "total_cost_before", "total_cost_after"),
            ("interval_funding", "funding_pnl_before", "funding_pnl_after"),
            ("interval_borrow_cost", "borrow_cost_before", "borrow_cost_after"),
        ):
            if not _close(
                float(interval[after_field]) - float(interval[before_field]),
                float(interval[interval_field]),
            ):
                raise ValueError(
                    f"{symbol} ledger {interval_field} disagrees with cumulative total"
                )
        before_dd = float(interval["max_drawdown_before"])
        after_dd = float(interval["max_drawdown_after"])
        if not 0.0 <= before_dd <= after_dd <= 1.0:
            raise ValueError(f"{symbol} ledger drawdown trace is invalid")
        quantities_before = interval.get("exact_quantities_before")
        quantities_after = interval.get("exact_quantities_after")
        if (
            not isinstance(quantities_before, list)
            or not isinstance(quantities_after, list)
            or len(quantities_before) != len(symbols)
            or len(quantities_after) != len(symbols)
        ):
            raise ValueError(
                f"{symbol} ledger quantity vectors differ from symbol roster"
            )
        try:
            parsed_before = [parse_quantity(value) for value in quantities_before]
            parsed_after = [parse_quantity(value) for value in quantities_after]
        except (TypeError, ValueError) as error:
            raise ValueError(f"{symbol} ledger exact quantity is invalid") from error
        if previous is None:
            if (
                before_dd != 0.0
                or before_value != evaluation["initial_capital"]
                or float(interval["cash_before"]) != evaluation["initial_capital"]
                or any(
                    not _close(float(interval[field]), 0.0)
                    for field in (
                        "total_cost_before",
                        "funding_pnl_before",
                        "borrow_cost_before",
                        "turnover_total_before",
                    )
                )
                or any(parsed_before)
            ):
                raise ValueError(f"{symbol} ledger initial account state is invalid")
        else:
            for before_field, after_field in (
                ("cash_before", "cash_after"),
                ("portfolio_value_before", "portfolio_value_after"),
                ("total_cost_before", "total_cost_after"),
                ("funding_pnl_before", "funding_pnl_after"),
                ("borrow_cost_before", "borrow_cost_after"),
                ("turnover_total_before", "turnover_total_after"),
                ("max_drawdown_before", "max_drawdown_after"),
            ):
                if not _close(
                    float(interval[before_field]), float(previous[after_field])
                ):
                    raise ValueError(f"{symbol} ledger book-state chain is broken")
            if parsed_before != previous["parsed_quantities_after"]:
                raise ValueError(f"{symbol} ledger quantity chain is broken")
        previous = {
            **interval,
            "parsed_quantities_after": parsed_after,
        }

    assert previous is not None
    final_dd = _finite_number(
        ledger.get("final_max_drawdown"), field=f"{symbol} final ledger drawdown"
    )
    if (
        not 0.0 <= final_dd <= 1.0
        or not _close(final_dd, float(previous["max_drawdown_after"]))
        or ledger.get("terminal_exact_quantities") != previous["exact_quantities_after"]
        or not _close(
            _finite_number(
                ledger.get("final_portfolio_value"),
                field=f"{symbol} final portfolio value",
            ),
            float(previous["portfolio_value_after"]),
        )
        or not _close(
            _finite_number(ledger.get("final_cash"), field=f"{symbol} final cash"),
            float(previous["cash_after"]),
        )
        or not _close(
            _finite_number(
                ledger.get("final_total_cost"),
                field=f"{symbol} final total cost",
            ),
            float(previous["total_cost_after"]),
        )
        or not _close(
            _finite_number(
                ledger.get("final_funding_pnl"),
                field=f"{symbol} final funding PnL",
            ),
            float(previous["funding_pnl_after"]),
        )
        or not _close(
            _finite_number(
                ledger.get("final_borrow_cost"),
                field=f"{symbol} final borrow cost",
            ),
            float(previous["borrow_cost_after"]),
        )
        or not _close(
            _finite_number(
                ledger.get("final_turnover_total"),
                field=f"{symbol} final turnover",
            ),
            float(previous["turnover_total_after"]),
        )
    ):
        raise ValueError(f"{symbol} final ledger state disagrees with interval trace")
    remainders = ledger.get("active_order_remainders")
    if not isinstance(remainders, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not math.isfinite(float(item[1]))
        for item in remainders
    ):
        raise ValueError(f"{symbol} active-order remainder evidence is malformed")
    terminal_exact = ledger.get("terminal_exact_quantities")
    if not isinstance(terminal_exact, list) or len(terminal_exact) != len(symbols):
        raise ValueError(
            f"{symbol} terminal exact quantities differ from symbol roster"
        )
    try:
        terminal_parsed = [parse_quantity(value) for value in terminal_exact]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{symbol} terminal exact quantity is invalid") from error
    terminal_values = tuple(row["terminal_quantities"])
    if any(
        not _close(project_quantity(exact), value)
        for exact, value in zip(terminal_parsed, terminal_values, strict=True)
    ):
        raise ValueError(f"{symbol} ledger terminal quantities differ from replay")
    termination_reason = ledger.get("termination_reason")
    if termination_reason != previous.get("termination_reason"):
        raise ValueError(f"{symbol} ledger termination evidence is inconsistent")
    return {
        "schema_version": ledger["schema_version"],
        "dataset_id": ledger["dataset_id"],
        "start_index": ledger["start_index"],
        "stop_index": ledger["stop_index"],
        "execution_policy_digest": ledger["execution_policy_digest"],
        "interval_count": len(intervals),
        "final_max_drawdown": final_dd,
        "terminal_exact_quantities": terminal_exact,
        "termination_reason": termination_reason,
        "active_order_remainders": remainders,
    }


def _validate_replay(
    row: dict[str, Any],
    *,
    protocol: dict[str, Any],
    scenario_name: str,
    symbol: str,
    symbol_index: int,
) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    scenario = evaluation["scenarios"].get(scenario_name)
    if not isinstance(scenario, dict):
        raise ValueError(f"{scenario_name} is not a preregistered scenario")
    if row.get("dataset_id") != protocol["evaluation_dataset_id"]:
        raise ValueError(f"{symbol} replay dataset identity mismatch")
    integer_identities = (
        (row.get("start_index"), evaluation["start_index"]),
        (row.get("stop_index"), evaluation["stop_index"]),
        (row.get("symbol_index"), symbol_index),
    )
    if any(
        isinstance(actual, bool) or not isinstance(actual, int) or actual != expected
        for actual, expected in integer_identities
    ):
        raise ValueError(f"{symbol} replay identity/window mismatch")
    try:
        initial_capital = _finite_number(
            row.get("initial_capital"), field=f"{symbol} capital"
        )
        gross_budget = _finite_number(
            row.get("gross_budget"), field=f"{symbol} gross budget"
        )
        cost_multiplier = _finite_number(
            row.get("cost_multiplier"), field=f"{symbol} cost multiplier"
        )
    except ValueError as error:
        raise ValueError(f"{symbol} replay scenario contract mismatch") from error
    if (
        row.get("schema") != "directional_arm_v1"
        or row.get("scenario") != scenario_name
        or not _close(initial_capital, evaluation["initial_capital"])
        or not _close(gross_budget, evaluation["gross_budget"])
        or isinstance(row.get("latency_bars"), bool)
        or not isinstance(row.get("latency_bars"), int)
        or row.get("latency_bars") != scenario["latency_bars"]
        or not _close(cost_multiplier, scenario["cost_multiplier"])
        or row.get("execution_policy_digest") != scenario["execution_policy_digest"]
        or canonical_json_bytes(row.get("risk"))
        != canonical_json_bytes(evaluation["risk"])
    ):
        raise ValueError(f"{symbol} replay scenario contract mismatch")
    raw_returns = row.get("returns")
    if (
        not isinstance(raw_returns, list)
        or len(raw_returns) != evaluation["interval_count"]
    ):
        raise ValueError(f"{symbol} replay interval count is incomplete")
    returns = tuple(
        _finite_number(value, field=f"{symbol} return") for value in raw_returns
    )
    if any(value <= -1.0 for value in returns):
        raise ValueError(f"{symbol} simple returns must be greater than -1")
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{symbol} replay metrics are missing")
    total_return = _finite_number(
        metrics.get("total_return"), field=f"{symbol} total return"
    )
    recomputed = compound_return(returns)
    if not math.isclose(total_return, recomputed, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{symbol} total return disagrees with interval returns")
    endpoint_drawdown = 0.0
    wealth = peak = 1.0
    for value in returns:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        endpoint_drawdown = max(endpoint_drawdown, 1.0 - wealth / peak)
    metric_drawdown = _finite_number(
        metrics.get("max_drawdown"), field=f"{symbol} metric drawdown"
    )
    if not _close(metric_drawdown, endpoint_drawdown):
        raise ValueError(f"{symbol} metric drawdown disagrees with returns")
    drawdown = _finite_number(
        row.get("ledger_max_drawdown"), field=f"{symbol} ledger drawdown"
    )
    if not 0.0 <= drawdown <= 1.0:
        raise ValueError(f"{symbol} ledger drawdown must be within [0, 1]")
    if drawdown + 1e-11 < endpoint_drawdown:
        raise ValueError(f"{symbol} ledger drawdown understates endpoint drawdown")
    reasons = row.get("termination_reasons")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) for reason in reasons
    ):
        raise ValueError(f"{symbol} termination evidence is malformed")
    terminal_flat = row.get("terminal_flat")
    quantities = row.get("terminal_quantities")
    if (
        not isinstance(terminal_flat, bool)
        or not isinstance(quantities, list)
        or len(quantities) != len(protocol["symbols"])
    ):
        raise ValueError(
            f"{symbol} terminal-position evidence must match the exact symbol roster"
        )
    terminal_values = tuple(
        _finite_number(value, field=f"{symbol} terminal quantity")
        for value in quantities
    )
    actual_flat = all(abs(value) <= _FLAT_TOLERANCE for value in terminal_values)
    if terminal_flat != actual_flat:
        raise ValueError(f"{symbol} terminal-flat flag contradicts terminal quantities")
    summary = row.get("ledger_trace_summary")
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != "shared_cash_replay_ledger_v1"
        or summary.get("dataset_id") != protocol["evaluation_dataset_id"]
        or summary.get("start_index") != evaluation["start_index"]
        or summary.get("stop_index") != evaluation["stop_index"]
        or summary.get("execution_policy_digest") != scenario["execution_policy_digest"]
        or summary.get("interval_count") != evaluation["interval_count"]
        or not _close(
            _finite_number(
                summary.get("final_max_drawdown"),
                field=f"{symbol} trace drawdown",
            ),
            drawdown,
        )
    ):
        raise ValueError(f"{symbol} ledger trace summary is missing or mismatched")
    exact_quantities = summary.get("terminal_exact_quantities")
    if not isinstance(exact_quantities, list) or len(exact_quantities) != len(
        protocol["symbols"]
    ):
        raise ValueError(
            f"{symbol} terminal exact quantities differ from symbol roster"
        )
    try:
        exact_values = tuple(
            project_quantity(parse_quantity(value)) for value in exact_quantities
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{symbol} terminal exact quantity is invalid") from error
    if any(
        not _close(exact, value)
        for exact, value in zip(exact_values, terminal_values, strict=True)
    ):
        raise ValueError(f"{symbol} ledger terminal quantities differ from replay")
    active_remainders = summary.get("active_order_remainders")
    if not isinstance(active_remainders, list):
        raise ValueError(f"{symbol} active-order remainder evidence is malformed")
    ledger_termination = summary.get("termination_reason")
    if ledger_termination != (reasons[0] if reasons else None) or len(reasons) > 1:
        raise ValueError(f"{symbol} ledger termination evidence is inconsistent")
    yearly = interval_start_year_returns(
        raw_returns, evaluation["interval_year_slices"]
    )
    return {
        "returns": returns,
        "total_return": total_return,
        "year_returns": yearly,
        "ledger_max_drawdown": drawdown,
        "termination_reasons": tuple(reasons),
        "terminal_flat": terminal_flat,
        "active_order_remainders": tuple(tuple(item) for item in active_remainders),
    }


def _valid_candidate_row(replay: dict[str, Any], *, maximum_drawdown: float) -> bool:
    return bool(
        not replay["termination_reasons"]
        and replay["terminal_flat"]
        and not replay["active_order_remainders"]
        and replay["ledger_max_drawdown"] <= maximum_drawdown
        and replay["total_return"] > 0.0
        and replay["year_returns"]
        and all(value > 0.0 for value in replay["year_returns"].values())
    )


def _validate_arm_roster(
    rows: dict[int, dict[str, Any]], factor: str, protocol: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    expected_seeds = tuple(protocol["seeds"])
    if set(rows) != set(expected_seeds) or len(rows) != len(expected_seeds):
        raise ValueError("evidence must contain the exact seed roster")
    symbols = tuple(protocol["symbols"])
    stress_names = tuple(protocol["evaluation"]["stress_names"])
    validated: dict[int, dict[str, Any]] = {}
    for seed in expected_seeds:
        arm = rows[seed]
        if (
            arm.get("schema") != "ppo_feature_arm_v1"
            or arm.get("factor") != factor
            or isinstance(arm.get("seed"), bool)
            or not isinstance(arm.get("seed"), int)
            or arm.get("seed") != seed
        ):
            raise ValueError("factor/seed identity mismatch")
        if (
            arm.get("dataset_id") != protocol["evaluation_dataset_id"]
            or arm.get("start_index") != protocol["evaluation"]["start_index"]
            or arm.get("stop_index") != protocol["evaluation"]["stop_index"]
            or arm.get("requested_timesteps")
            != protocol["training"]["requested_timesteps"]
            or canonical_json_bytes(arm.get("feature_names"))
            != canonical_json_bytes(protocol["factors"][factor]["feature_names"])
            or canonical_json_bytes(arm.get("feature_indices"))
            != canonical_json_bytes(protocol["factors"][factor]["feature_indices"])
            or canonical_json_bytes(arm.get("training_risk"))
            != canonical_json_bytes(protocol["training"]["risk"])
        ):
            raise ValueError(
                "arm dataset, feature, training or evaluation identity mismatch"
            )
        by_symbol = arm.get("by_symbol")
        if not isinstance(by_symbol, dict) or set(by_symbol) != set(symbols):
            raise ValueError("evidence must contain the exact symbol roster")
        base = {
            symbol: _validate_replay(
                by_symbol[symbol],
                protocol=protocol,
                scenario_name="base",
                symbol=symbol,
                symbol_index=index,
            )
            for index, symbol in enumerate(symbols)
        }
        raw_stress = arm.get("stress_by_name")
        expected_stress_names = stress_names if factor == CANDIDATE_FACTOR else ()
        if not isinstance(raw_stress, dict) or set(raw_stress) != set(
            expected_stress_names
        ):
            raise ValueError("candidate evidence stress roster mismatch")
        stress: dict[str, dict[str, dict[str, Any]]] = {}
        for stress_name in expected_stress_names:
            stress_roster = raw_stress[stress_name]
            if not isinstance(stress_roster, dict) or set(stress_roster) != set(
                symbols
            ):
                raise ValueError("candidate evidence stress symbol roster mismatch")
            stress[stress_name] = {
                symbol: _validate_replay(
                    stress_roster[symbol],
                    protocol=protocol,
                    scenario_name=stress_name,
                    symbol=symbol,
                    symbol_index=index,
                )
                for index, symbol in enumerate(symbols)
            }
        validated[seed] = {"base": base, "stress": stress}
    return validated


def compare_feature_ablation(
    baseline_rows: dict[int, dict[str, Any]],
    candidate_rows: dict[int, dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply fixed absolute, paired-relative, and stress admission screens."""
    baseline = _validate_arm_roster(baseline_rows, BASELINE_FACTOR, protocol)
    candidate = _validate_arm_roster(candidate_rows, CANDIDATE_FACTOR, protocol)
    symbols = tuple(protocol["symbols"])
    seeds = tuple(protocol["seeds"])
    required_seeds = protocol["admission"]["required_passing_seeds_per_symbol"]
    required_symbols = protocol["admission"]["required_symbols"]
    required_deltas = protocol["admission"]["required_positive_paired_seed_deltas"]
    max_drawdown = protocol["evaluation"]["ledger_drawdown_limit"]
    stress_names = tuple(protocol["evaluation"]["stress_names"])

    hard_guard_violations: list[dict[str, Any]] = []
    for seed in seeds:
        for symbol in symbols:
            scenario_rows = [("base", candidate[seed]["base"][symbol])]
            scenario_rows.extend(
                (name, candidate[seed]["stress"][name][symbol]) for name in stress_names
            )
            for scenario_name, replay in scenario_rows:
                reasons: list[str] = []
                if replay["ledger_max_drawdown"] > max_drawdown:
                    reasons.append("ledger_drawdown_exceeded")
                if replay["termination_reasons"]:
                    reasons.append("terminated")
                if not replay["terminal_flat"]:
                    reasons.append("terminal_position_not_flat")
                if replay["active_order_remainders"]:
                    reasons.append("active_order_remainder")
                if reasons:
                    hard_guard_violations.append(
                        {
                            "seed": seed,
                            "symbol": symbol,
                            "scenario": scenario_name,
                            "reasons": reasons,
                        }
                    )

    absolute_symbols: list[str] = []
    relative_symbols: list[str] = []
    stress_symbols: list[str] = []
    symbol_report: dict[str, Any] = {}
    for symbol in symbols:
        old = {seed: baseline[seed]["base"][symbol] for seed in seeds}
        new = {seed: candidate[seed]["base"][symbol] for seed in seeds}
        absolute_seed_pass = {
            seed: _valid_candidate_row(new[seed], maximum_drawdown=max_drawdown)
            for seed in seeds
        }
        median_full = float(np.median([new[seed]["total_return"] for seed in seeds]))
        median_yearly = {
            year: float(np.median([new[seed]["year_returns"][year] for seed in seeds]))
            for year in protocol["evaluation"]["interval_year_slices"]
        }
        absolute = bool(
            sum(absolute_seed_pass.values()) >= required_seeds
            and median_full > 0.0
            and all(value > 0.0 for value in median_yearly.values())
        )
        if absolute:
            absolute_symbols.append(symbol)

        deltas = {
            seed: new[seed]["total_return"] - old[seed]["total_return"]
            for seed in seeds
        }
        year_deltas = {
            year: {
                seed: new[seed]["year_returns"][year] - old[seed]["year_returns"][year]
                for seed in seeds
            }
            for year in protocol["evaluation"]["interval_year_slices"]
        }
        comparable = {
            seed: (
                not old[seed]["termination_reasons"]
                and not new[seed]["termination_reasons"]
                and old[seed]["terminal_flat"]
                and new[seed]["terminal_flat"]
                and not old[seed]["active_order_remainders"]
                and not new[seed]["active_order_remainders"]
                and new[seed]["ledger_max_drawdown"] <= max_drawdown
            )
            for seed in seeds
        }
        positive_delta_seeds = sum(
            comparable[seed] and deltas[seed] > 0.0 for seed in seeds
        )
        median_delta = float(np.median(list(deltas.values())))
        median_year_deltas = {
            year: float(np.median(list(values.values())))
            for year, values in year_deltas.items()
        }
        relative = bool(
            positive_delta_seeds >= required_deltas
            and median_delta > 0.0
            and all(value > 0.0 for value in median_year_deltas.values())
        )
        if relative:
            relative_symbols.append(symbol)

        stress_details: dict[str, Any] = {}
        stress_seed_pass: dict[int, bool] = {seed: True for seed in seeds}
        for stress_name in stress_names:
            per_seed = {
                seed: candidate[seed]["stress"][stress_name][symbol] for seed in seeds
            }
            for seed in seeds:
                stress_seed_pass[seed] = stress_seed_pass[
                    seed
                ] and _valid_candidate_row(
                    per_seed[seed], maximum_drawdown=max_drawdown
                )
            stress_median_full = float(
                np.median([per_seed[seed]["total_return"] for seed in seeds])
            )
            stress_median_years = {
                year: float(
                    np.median([per_seed[seed]["year_returns"][year] for seed in seeds])
                )
                for year in protocol["evaluation"]["interval_year_slices"]
            }
            stress_details[stress_name] = {
                "passing_seeds": [
                    seed
                    for seed in seeds
                    if _valid_candidate_row(
                        per_seed[seed], maximum_drawdown=max_drawdown
                    )
                ],
                "median_full_return": stress_median_full,
                "median_year_returns": stress_median_years,
                "median_pass": stress_median_full > 0.0
                and all(value > 0.0 for value in stress_median_years.values()),
            }
        same_seed_pass = {
            seed: absolute_seed_pass[seed] and stress_seed_pass[seed] for seed in seeds
        }
        stress = bool(
            sum(same_seed_pass.values()) >= required_seeds
            and all(item["median_pass"] for item in stress_details.values())
        )
        if stress:
            stress_symbols.append(symbol)
        symbol_report[symbol] = {
            "absolute_passing_seeds": [
                seed for seed in seeds if absolute_seed_pass[seed]
            ],
            "absolute_median_full_return": median_full,
            "absolute_median_year_returns": median_yearly,
            "absolute_pass": absolute,
            "paired_positive_delta_seeds": [
                seed for seed in seeds if comparable[seed] and deltas[seed] > 0.0
            ],
            "paired_median_full_delta": median_delta,
            "paired_median_year_deltas": median_year_deltas,
            "relative_pass": relative,
            "stress_seed_pass_both": [seed for seed in seeds if stress_seed_pass[seed]],
            "base_and_both_stress_passing_seeds": [
                seed for seed in seeds if same_seed_pass[seed]
            ],
            "stress": stress_details,
            "stress_pass": stress,
        }

    accepted_symbols = sorted(
        set(absolute_symbols) & set(relative_symbols) & set(stress_symbols)
    )
    hard_guards_pass = not hard_guard_violations
    if not hard_guards_pass:
        accepted_symbols = []
        decision = "HARD_GUARD_FAILURE"
    elif len(accepted_symbols) >= required_symbols:
        decision = "PROSPECTIVE_PAPER_REQUIRED"
    elif len(set(absolute_symbols) & set(relative_symbols)) >= required_symbols:
        decision = "KEEP_BASELINE"
    elif (
        len(absolute_symbols) >= required_symbols
        and len(relative_symbols) < required_symbols
    ):
        decision = "RELATIVE_IMPROVEMENT_NOT_ESTABLISHED"
    elif (
        len(relative_symbols) >= required_symbols
        and len(absolute_symbols) < required_symbols
    ):
        decision = "RELATIVE_IMPROVEMENT_ONLY"
    else:
        decision = "KEEP_BASELINE"
    return {
        "schema": "ppo_feature_ablation_comparison_v1",
        "decision": decision,
        "absolute_symbols": absolute_symbols,
        "relative_symbols": relative_symbols,
        "stress_symbols": stress_symbols,
        "accepted_symbols": accepted_symbols,
        "hard_guard_violations": hard_guard_violations,
        "hard_guards_pass": hard_guards_pass,
        "required_symbols": required_symbols,
        "symbol_results": symbol_report,
        "development_only": True,
        "production_eligible": False,
    }


def _read_ledger_artifact(
    directory: Path,
    row: dict[str, Any],
    *,
    protocol: dict[str, Any],
    scenario_name: str,
    symbol: str,
) -> dict[str, Any]:
    scenario = protocol["evaluation"]["scenarios"][scenario_name]
    symbol_index = tuple(protocol["symbols"]).index(symbol)
    filename = f"ledger-{scenario_name}-symbol-{symbol_index}.json.gz"
    if row.get("ledger_evidence_file") != filename:
        raise ValueError(f"{symbol} ledger evidence file identity mismatch")
    compressed = (directory / filename).read_bytes()
    if _sha256(compressed) != row.get("ledger_evidence_gzip_sha256"):
        raise ValueError(f"{symbol} ledger compressed digest mismatch")
    try:
        raw = gzip.decompress(compressed)
    except (OSError, EOFError) as error:
        raise ValueError(f"{symbol} ledger evidence compression is invalid") from error
    if _sha256(raw) != row.get("ledger_evidence_raw_sha256"):
        raise ValueError(f"{symbol} ledger raw digest mismatch")
    try:
        ledger = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{symbol} ledger evidence JSON is invalid") from error
    if canonical_json_bytes(ledger) != raw:
        raise ValueError(f"{symbol} ledger evidence JSON is not canonical")
    summary = _validate_ledger_payload(
        ledger,
        row=row,
        protocol=protocol,
        scenario_name=scenario_name,
        symbol=symbol,
    )
    if summary["execution_policy_digest"] != scenario["execution_policy_digest"]:
        raise ValueError(f"{symbol} ledger execution policy differs from scenario")
    return summary


def _read_arm(
    root: Path, factor: str, seed: int, protocol: dict[str, Any]
) -> dict[str, Any]:
    directory = root / factor / f"ppo{seed}"
    if (directory / "failed.json").exists():
        raise ValueError(f"{factor}/ppo{seed} failed; complete evidence is required")
    raw = (directory / "result.json").read_bytes()
    if json.loads((directory / "result.sha256.json").read_bytes()) != {
        "sha256": _sha256(raw)
    }:
        raise ValueError(f"{factor}/ppo{seed} result hash mismatch")
    row = json.loads(raw)
    if (
        row.get("schema") != "ppo_feature_arm_v1"
        or row.get("factor") != factor
        or isinstance(row.get("seed"), bool)
        or not isinstance(row.get("seed"), int)
        or row.get("seed") != seed
        or row.get("protocol_digest") != content_digest(protocol)
        or row.get("provenance") != protocol["provenance"]
    ):
        raise ValueError(f"{factor}/ppo{seed} protocol/provenance mismatch")
    expected_features = protocol["factors"][factor]
    if (
        row.get("dataset_id") != protocol["evaluation_dataset_id"]
        or row.get("start_index") != protocol["evaluation"]["start_index"]
        or row.get("stop_index") != protocol["evaluation"]["stop_index"]
        or row.get("requested_timesteps") != protocol["training"]["requested_timesteps"]
        or canonical_json_bytes(row.get("feature_names"))
        != canonical_json_bytes(expected_features["feature_names"])
        or canonical_json_bytes(row.get("feature_indices"))
        != canonical_json_bytes(expected_features["feature_indices"])
        or canonical_json_bytes(row.get("training_risk"))
        != canonical_json_bytes(protocol["training"]["risk"])
    ):
        raise ValueError(f"{factor}/ppo{seed} feature/training configuration mismatch")
    model_hashes = row.get("model_sha256")
    if not isinstance(model_hashes, dict) or set(model_hashes) != {"model.zip"}:
        raise ValueError(f"{factor}/ppo{seed} model evidence is incomplete")
    model_raw = (directory / "model.zip").read_bytes()
    if model_hashes["model.zip"] != _sha256(model_raw):
        raise ValueError(f"{factor}/ppo{seed} model hash mismatch")
    symbols = tuple(protocol["symbols"])
    scenario_rosters: list[tuple[str, dict[str, Any]]] = [("base", row["by_symbol"])]
    raw_stress = row.get("stress_by_name")
    expected_stress_names = (
        tuple(protocol["evaluation"]["stress_names"])
        if factor == CANDIDATE_FACTOR
        else ()
    )
    if not isinstance(raw_stress, dict) or set(raw_stress) != set(
        expected_stress_names
    ):
        raise ValueError(f"{factor}/ppo{seed} stress roster mismatch")
    scenario_rosters.extend((name, raw_stress[name]) for name in expected_stress_names)
    for scenario_name, roster in scenario_rosters:
        if not isinstance(roster, dict) or set(roster) != set(symbols):
            raise ValueError(
                f"{factor}/ppo{seed} {scenario_name} symbol roster mismatch"
            )
        for symbol_index, symbol in enumerate(symbols):
            replay = roster[symbol]
            replay["ledger_trace_summary"] = _read_ledger_artifact(
                directory,
                replay,
                protocol=protocol,
                scenario_name=scenario_name,
                symbol=symbol,
            )
            _validate_replay(
                replay,
                protocol=protocol,
                scenario_name=scenario_name,
                symbol=symbol,
                symbol_index=symbol_index,
            )
    return row


def _persist_replay_ledger(
    directory: Path,
    row: dict[str, Any],
    *,
    scenario_name: str,
    symbol_index: int,
) -> dict[str, Any]:
    ledger = row.pop("ledger_evidence", None)
    if not isinstance(ledger, dict):
        raise ValueError("directional replay did not produce requested ledger evidence")
    raw = canonical_json_bytes(ledger)
    compressed = gzip.compress(raw, mtime=0)
    filename = f"ledger-{scenario_name}-symbol-{symbol_index}.json.gz"
    with (directory / filename).open("xb") as stream:
        stream.write(compressed)
    row["scenario"] = scenario_name
    row["ledger_evidence_file"] = filename
    row["ledger_evidence_raw_sha256"] = _sha256(raw)
    row["ledger_evidence_gzip_sha256"] = _sha256(compressed)
    return row


def run_arm(source: Path, root: Path, factor: str, seed: int) -> None:
    if factor not in FACTORS or seed not in SEEDS:
        raise ValueError("run requires a preregistered factor and seed")
    protocol = validate_protocol(source, root)
    directory = root / factor / f"ppo{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    _write_once(
        directory / "started.json",
        {"factor": factor, "seed": seed, "protocol_digest": content_digest(protocol)},
    )
    try:
        original = load_market_dataset_artifact(source / "dataset")
        dataset = with_price_channels(original)
        del original
        base_config = inspect_study(source / "study").plan.baseline_config
        feature = protocol["factors"][factor]
        config = replace(
            base_config,
            feature_names=tuple(feature["feature_names"]),
            feature_indices=tuple(feature["feature_indices"]),
        )
        start = protocol["evaluation"]["start_index"]
        stop = protocol["evaluation"]["stop_index"]
        print(f"{factor}/ppo{seed}: fitting", flush=True)
        factory = fit_directional_candidate(
            f"ppo{seed}",
            dataset,
            config,
            directory,
            ppo_risk_config=TRAINING_RISK,
            initial_capital=protocol["training"]["initial_capital"],
            gross_budget=protocol["training"]["gross_budget"],
        )
        print(f"{factor}/ppo{seed}: replaying every symbol", flush=True)
        by_symbol: dict[str, dict[str, Any]] = {}
        for index, symbol in enumerate(dataset.symbols):
            replay = evaluate_directional_arm(
                dataset,
                factory,
                start_index=start,
                stop_index=stop,
                symbol_index=index,
                initial_capital=protocol["evaluation"]["initial_capital"],
                gross_budget=protocol["evaluation"]["gross_budget"],
                capture_ledger_evidence=True,
            )
            by_symbol[symbol] = _persist_replay_ledger(
                directory, replay, scenario_name="base", symbol_index=index
            )
        stress_by_name: dict[str, dict[str, Any]] = {}
        if factor == CANDIDATE_FACTOR:
            for stress_name in protocol["evaluation"]["stress_names"]:
                scenario = protocol["evaluation"]["scenarios"][stress_name]
                stress_by_name[stress_name] = {}
                for index, symbol in enumerate(dataset.symbols):
                    replay = evaluate_directional_arm(
                        dataset,
                        factory,
                        start_index=start,
                        stop_index=stop,
                        symbol_index=index,
                        initial_capital=protocol["evaluation"]["initial_capital"],
                        gross_budget=protocol["evaluation"]["gross_budget"],
                        cost_multiplier=scenario["cost_multiplier"],
                        latency_bars=scenario["latency_bars"],
                        capture_ledger_evidence=True,
                    )
                    stress_by_name[stress_name][symbol] = _persist_replay_ledger(
                        directory,
                        replay,
                        scenario_name=stress_name,
                        symbol_index=index,
                    )
        if build_candidate_run_provenance() != protocol["provenance"]:
            raise ValueError("source/runtime changed during training or replay")
        model = directory / "model.zip"
        result = {
            "schema": "ppo_feature_arm_v1",
            "factor": factor,
            "seed": seed,
            "dataset_id": dataset.dataset_id,
            "start_index": start,
            "stop_index": stop,
            "feature_names": feature["feature_names"],
            "feature_indices": feature["feature_indices"],
            "training_risk": protocol["training"]["risk"],
            "requested_timesteps": PPO_TIMESTEPS,
            "by_symbol": by_symbol,
            "stress_by_name": stress_by_name,
            "model_sha256": {"model.zip": _sha256(model.read_bytes())},
            "protocol_digest": content_digest(protocol),
            "provenance": protocol["provenance"],
        }
        _write_once(directory / "result.json", result)
        _write_once(
            directory / "result.sha256.json",
            {"sha256": _sha256((directory / "result.json").read_bytes())},
        )
        print(f"{factor}/ppo{seed}: evidence published", flush=True)
    except BaseException as error:
        _write_once(
            directory / "failed.json",
            {"factor": factor, "seed": seed, "error": repr(error)},
        )
        raise


def finalize_study(source: Path, root: Path) -> None:
    protocol = validate_protocol(source, root)
    rows = {
        factor: {seed: _read_arm(root, factor, seed, protocol) for seed in SEEDS}
        for factor in FACTORS
    }
    comparison = compare_feature_ablation(
        rows[BASELINE_FACTOR], rows[CANDIDATE_FACTOR], protocol
    )
    comparison["protocol_digest"] = content_digest(protocol)
    comparison["result_sha256"] = {
        factor: {
            f"ppo{seed}": _sha256(
                (root / factor / f"ppo{seed}" / "result.json").read_bytes()
            )
            for seed in SEEDS
        }
        for factor in FACTORS
    }
    comparison["model_sha256"] = {
        factor: {
            f"ppo{seed}": rows[factor][seed]["model_sha256"]["model.zip"]
            for seed in SEEDS
        }
        for factor in FACTORS
    }
    _write_once(root / "comparison.json", comparison)
    _write_once(
        root / "comparison.digest.json",
        {"digest": content_digest(comparison)},
    )
    print(f"Published feature-ablation decision: {comparison['decision']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "finalize"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", choices=FACTORS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == "prepare":
        reserve_study(args.source, args.output)
    elif args.action == "finalize":
        finalize_study(args.source, args.output)
    elif args.factor is None or args.seed is None:
        parser.error("run requires --factor and --seed")
    else:
        run_arm(args.source, args.output, args.factor, args.seed)


if __name__ == "__main__":
    main()


__all__ = [
    "BASELINE_FACTOR",
    "CANDIDATE_FACTOR",
    "RELATIVE_FEATURE_NAMES",
    "SEEDS",
    "TRAINING_RISK",
    "compare_feature_ablation",
    "expected_protocol",
    "finalize_study",
    "interval_start_year_returns",
    "main",
    "reserve_study",
    "run_arm",
    "validate_protocol",
]
