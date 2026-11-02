"""Frozen, metadata-only Task18 NONRL48 recovery recipe declaration.

The historical protocol and consumed-data hashes here are reference pins. They
are not permission to load the historical arrays or execute the old protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Literal

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceFilePin:
    """Repository-relative source path bound to a raw SHA-256 digest."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise ValueError("source pin path must be a normalized repo-relative path")
        pure_path = PurePosixPath(self.path)
        if (
            not self.path
            or "\\" in self.path
            or "\x00" in self.path
            or ":" in self.path
            or pure_path.is_absolute()
            or pure_path.as_posix() != self.path
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            raise ValueError("source pin path must be a normalized repo-relative path")
        if not isinstance(self.sha256, str) or not _HEX_64.fullmatch(self.sha256):
            raise ValueError(
                "source pin sha256 must be 64 lowercase hexadecimal characters"
            )


@dataclass(frozen=True)
class HistoricalReference:
    source_commit: str
    protocol_source_path: str
    protocol_source_sha256: str
    protocol_digest: str
    dataset_id: str
    manifest_sha256: str
    arrays_sha256: str
    artifact_schema: int
    identity_schema: int
    market_build_version: int
    source_file_pins: tuple[SourceFilePin, ...]


@dataclass(frozen=True)
class DatasetDeclaration:
    artifact_schema: int
    identity_schema: int
    market_build_version: int
    row_count: int
    symbols: tuple[str, ...]
    features_per_symbol: int
    feature_names: tuple[str, ...]
    development_start: str
    development_end: str
    consumed_development_only: bool


@dataclass(frozen=True)
class RidgeConfiguration:
    name: str
    horizon_hours: int
    alpha: float


@dataclass(frozen=True)
class ScenarioDeclaration:
    name: str
    fee_multiplier: float
    spread_multiplier: float
    impact_multiplier: float
    execution_latency_bars: int
    description: str


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    pair_index: int
    configuration: str
    symbol: str
    scenario: str


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    arm_index: int
    pair_id: str
    role: Literal["NONRL", "CASH"]
    configuration: str
    symbol: str
    scenario: str


@dataclass(frozen=True)
class IndexPlan:
    actor_start_inclusive: int
    actor_stop_exclusive: int
    actor_start_utc: str
    actor_last_bar_utc: str
    actor_stop_exclusive_utc: str
    vintage_start_indices: tuple[int, ...]
    ridge_vintage_count: int
    public_cost_preparation_fit_count: int
    vintages_are_pair_dimensions: bool
    account_continuity_across_vintages: bool
    cutoff_rule: str
    fit_time_semantics: str
    inference_delay_bars: int
    terminal_stop_inclusive: int
    terminal_advance_count: int
    terminal_time_utc: str
    unused_final_row: int
    native_advances_per_arm: int
    total_native_advances: int
    resets_per_arm: int
    declared_seeds: int


@dataclass(frozen=True)
class Economics:
    initial_capital_per_symbol_usdt: float
    books_are_independent_by_symbol: bool
    base_fee_rate: float
    base_spread_rate: float
    impact_coefficient: float
    impact_reference_notional_usdt: float
    maximum_participation_rate: float
    borrow_available: bool
    borrow_rate: float
    cash_rate: float
    stochastic_slippage: float
    funding_mark_semantics: str
    funding_observation_window_hours: int
    funding_observed_event_count_minimum: int
    funding_observed_event_count_maximum: int
    maximum_funding_close_gap_hours: int
    maximum_funding_age_hours: int
    zero_rate_funding_events_count: bool


@dataclass(frozen=True)
class RiskDeclaration:
    allocator_weight_minimum: float
    allocator_weight_maximum: float
    turnover_cap: float | None
    max_gross: float
    max_abs_weight: float
    drawdown_start: float
    drawdown_stop: float
    emergency_turnover_override: bool
    tolerance: float
    maximum_active_slots: int


@dataclass(frozen=True)
class MetricAndGateDeclaration:
    primary_pair_metric: str
    combined_metric: str
    combined_metric_denominator: str
    required_scenarios: tuple[str, ...]
    maximum_drawdown_per_book: float
    terminal_flat_required: bool
    no_pending_orders_required: bool
    minimum_operational_threshold: str
    winner_status: str
    adoption_status: str
    unknown_costs: tuple[str, ...]
    claim_boundary: str


@dataclass(frozen=True)
class Task18Protocol:
    recipe_schema: str
    scope: str
    roster_dimension_note: str
    historical_reference: HistoricalReference
    dataset: DatasetDeclaration
    configurations: tuple[RidgeConfiguration, ...]
    scenarios: tuple[ScenarioDeclaration, ...]
    pairs: tuple[PairSpec, ...]
    arms: tuple[ArmSpec, ...]
    index_plan: IndexPlan
    economics: Economics
    risk: RiskDeclaration
    gates: MetricAndGateDeclaration
    rollout_length: int
    gamma: float
    gae_lambda: float
    gae_lambda_usage: str
    reward_name: str
    nonrl_policy: str
    cash_policy: str
    label_semantics: str
    hyperparameter_optimization: bool

    @property
    def adapted_protocol_digest(self) -> str:
        payload = asdict(self)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _historical_reference() -> HistoricalReference:
    source_pins = (
        SourceFilePin(
            "tools/tier_d_nonrl/__init__.py",
            "e512388cffd43367cb2b3c15f45020e92b20ed5ecd3d080b87d2553123e8b429",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/__main__.py",
            "4465bb24509a2a27d917744c75a0cc276294bd74a0bf512955a89cab06f12166",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/progress.py",
            "f82d48cd17865b623cb07994c006e32411c3620569fb551006e3de65fc7f721b",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/progress_io.py",
            "1c748d8e8994350ce623e3237e014e9e230c95d7a18940aae4c75ab06aa91d36",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/restore.py",
            "fa057d6e6c4231abd6ff1882edfe3a3a315cd66eacfb7bf5bfb3e8974e6ba8d7",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/resume.py",
            "7f82565f554e41dd6fd9aee7ec3605809df6b3185ca03aa44c02332e2bb0c07f",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/tier_d_decomposition.py",
            "2c4a9ac75799a77fe9e908189eeff9736ee6d5da462c8fd867b3427d9b694f10",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/tier_d_fixture.py",
            "4a0f1c527a7a6d7abcfd2cf43fd5b93a75dd71b8ea058358e865454de83d068e",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/tier_d_native.py",
            "dcf6dacf2d82ecec6ef957d5df13006fa4df972cb96e3d2d2ae03042afa1658e",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/tier_d_process_tree.py",
            "c923fe16c98584e6e58bd2da4263fd5e6f712e723e2c507a5a6196e9c0a332db",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/tier_d_protocol.py",
            "9d679316cc4ffe614bdd8237271610c3168915f4b66993a16010b2fa812c6641",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/tier_d_records.py",
            "cd92688d40c24eec09a1c9c620cfb5e58dbe67588e3dd01539aabc9e5fac3562",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/tier_d_runner.py",
            "749feebae592b3ca853a40647695ffef27942d9eb6fd77faa5517887a8b39dda",
        ),
        SourceFilePin(
            "tools/tier_d_nonrl/tier_d_supervisor.py",
            "795777ef9e1549d60994a36ee2d5738bd4a86b8cafef3968bbd6641a78168581",
        ),
    )
    return HistoricalReference(
        source_commit="9f40cc93aec8439a3740aec4e2aa2b842ad893a3",
        protocol_source_path="tools/tier_d_nonrl/tier_d_protocol.py",
        protocol_source_sha256="9d679316cc4ffe614bdd8237271610c3168915f4b66993a16010b2fa812c6641",
        protocol_digest="9ce03a1a3361e77edb2a111ebb1364f83994e09400128a920bb4b3c38b6c9b79",
        dataset_id="e916ff716fb5e4b90d5805eaa79200824b09f103984fa845bbeaedb57bef194f",
        manifest_sha256="6ac5099b98a32bd33e4f98c5ac5fb2dbffd19b0b9e3c7c3a9b5a8f6860810ee6",
        arrays_sha256="25c97a4ebd1e81ac64f4bc7e78dce4ab102c522e548e30a2154fcfa0143842a3",
        artifact_schema=4,
        identity_schema=7,
        market_build_version=3,
        source_file_pins=source_pins,
    )


def _pair_roster(
    configurations: tuple[RidgeConfiguration, ...],
    symbols: tuple[str, ...],
    scenarios: tuple[ScenarioDeclaration, ...],
) -> tuple[PairSpec, ...]:
    pairs: list[PairSpec] = []
    for configuration in configurations:
        for symbol in symbols:
            for scenario in scenarios:
                pair_index = len(pairs) + 1
                pairs.append(
                    PairSpec(
                        pair_id=f"PAIR-{pair_index:02d}",
                        pair_index=pair_index,
                        configuration=configuration.name,
                        symbol=symbol,
                        scenario=scenario.name,
                    )
                )
    return tuple(pairs)


def _arm_roster(pairs: tuple[PairSpec, ...]) -> tuple[ArmSpec, ...]:
    arms: list[ArmSpec] = []
    for pair in pairs:
        for role in ("NONRL", "CASH"):
            arm_index = len(arms) + 1
            arms.append(
                ArmSpec(
                    arm_id=f"{pair.pair_id}-{role}",
                    arm_index=arm_index,
                    pair_id=pair.pair_id,
                    role=role,
                    configuration=pair.configuration,
                    symbol=pair.symbol,
                    scenario=pair.scenario,
                )
            )
    return tuple(arms)


def build_task18_protocol() -> Task18Protocol:
    """Return the fixed 24-pair/48-arm Task18 recovery declaration."""
    configurations = (
        RidgeConfiguration("H24_RIDGE_A1", 24, 1.0),
        RidgeConfiguration("H24_RIDGE_A10", 24, 10.0),
        RidgeConfiguration("H72_RIDGE_A1", 72, 1.0),
        RidgeConfiguration("H72_RIDGE_A10", 72, 10.0),
    )
    scenarios = (
        ScenarioDeclaration(
            "BASEL0", 1.0, 1.0, 1.0, 0, "Base costs and zero-bar execution latency."
        ),
        ScenarioDeclaration(
            "COST2L0",
            2.0,
            2.0,
            2.0,
            0,
            "Double fee, spread, and impact; carry unchanged.",
        ),
        ScenarioDeclaration(
            "BASEL2", 1.0, 1.0, 1.0, 2, "Two-bar execution latency; beliefs unchanged."
        ),
    )
    symbols = ("BTCUSDT", "ETHUSDT")
    pairs = _pair_roster(configurations, symbols, scenarios)
    arms = _arm_roster(pairs)
    return Task18Protocol(
        recipe_schema="task18_nonrl48_adapted_recovery_v1",
        scope="CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY",
        roster_dimension_note=(
            "4 configurations x 2 symbols x 3 scenarios = 24 pairs; three vintages "
            "are sequential model updates, not a pair dimension."
        ),
        historical_reference=_historical_reference(),
        dataset=DatasetDeclaration(
            artifact_schema=4,
            identity_schema=7,
            market_build_version=3,
            row_count=9504,
            symbols=symbols,
            features_per_symbol=6,
            feature_names=(
                "1h__log_return_24bar",
                "4h__log_return_1bar",
                "1d__log_return_1bar",
                "1h__realized_volatility_24bar",
                "1h__volume_zscore_24bar",
                "1h__funding_bps",
            ),
            development_start="2024-12-01T01:00:00",
            development_end="2026-01-01T01:00:00",
            consumed_development_only=True,
        ),
        configurations=configurations,
        scenarios=scenarios,
        pairs=pairs,
        arms=arms,
        index_plan=IndexPlan(
            actor_start_inclusive=2903,
            actor_stop_exclusive=9430,
            actor_start_utc="2025-04-01T00:00:00Z",
            actor_last_bar_utc="2025-12-28T22:00:00Z",
            actor_stop_exclusive_utc="2025-12-28T23:00:00Z",
            vintage_start_indices=(2903, 5087, 7295),
            ridge_vintage_count=12,
            public_cost_preparation_fit_count=4,
            vintages_are_pair_dimensions=False,
            account_continuity_across_vintages=True,
            cutoff_rule="vintage start timestamp minus 1 ns; endpoints/publication precede cutoff",
            fit_time_semantics="declared simulated time only; not measured compute readiness",
            inference_delay_bars=0,
            terminal_stop_inclusive=9502,
            terminal_advance_count=72,
            terminal_time_utc="2025-12-31T23:00:00Z",
            unused_final_row=9503,
            native_advances_per_arm=6599,
            total_native_advances=316752,
            resets_per_arm=1,
            declared_seeds=0,
        ),
        economics=Economics(
            initial_capital_per_symbol_usdt=100_000.0,
            books_are_independent_by_symbol=True,
            base_fee_rate=0.0005,
            base_spread_rate=0.0002,
            impact_coefficient=0.0001,
            impact_reference_notional_usdt=100_000.0,
            maximum_participation_rate=0.05,
            borrow_available=True,
            borrow_rate=0.0,
            cash_rate=0.0,
            stochastic_slippage=0.0,
            funding_mark_semantics=(
                "Rates-only Task18 witness; bar-close proxy rate multiplied by bar-close mark; "
                "not authenticated venue mark data."
            ),
            funding_observation_window_hours=24,
            funding_observed_event_count_minimum=2,
            funding_observed_event_count_maximum=4,
            maximum_funding_close_gap_hours=9,
            maximum_funding_age_hours=8,
            zero_rate_funding_events_count=True,
        ),
        risk=RiskDeclaration(
            allocator_weight_minimum=-1.0,
            allocator_weight_maximum=1.0,
            turnover_cap=None,
            max_gross=1.0,
            max_abs_weight=1.0,
            drawdown_start=0.10,
            drawdown_stop=0.20,
            emergency_turnover_override=True,
            tolerance=1e-10,
            maximum_active_slots=8,
        ),
        gates=MetricAndGateDeclaration(
            primary_pair_metric="terminal after-cost NONRL minus matched CASH wealth / V0",
            combined_metric="BTC and ETH wealth deltas combined per configuration and scenario",
            combined_metric_denominator="2 * V0, with independent 100000 USDT books per symbol",
            required_scenarios=("BASEL0", "COST2L0", "BASEL2"),
            maximum_drawdown_per_book=0.20,
            terminal_flat_required=True,
            no_pending_orders_required=True,
            minimum_operational_threshold="UNKNOWN",
            winner_status="NOT_DECLARED",
            adoption_status="NOT_ESTABLISHED",
            unknown_costs=(
                "fixed infrastructure costs",
                "taxes",
                "other unpriced costs",
            ),
            claim_boundary=(
                "Conditional comparison on consumed development data only; no unused-period, "
                "maximum-profit, historical-calibration, or live-readiness claim."
            ),
        ),
        rollout_length=32,
        gamma=1.0,
        gae_lambda=0.95,
        gae_lambda_usage="declared but unused for the NONRL baseline",
        reward_name="equity_delta_v1 / fixed V0",
        nonrl_policy="residual KEEP_BASELINE code 2, scale 1",
        cash_policy="direct HOLD code 0 from a flat book, scale 1",
        label_semantics=(
            "raw close-ratio simple-return labels; carry and costs are accounted "
            "in allocation/execution"
        ),
        hyperparameter_optimization=False,
    )
