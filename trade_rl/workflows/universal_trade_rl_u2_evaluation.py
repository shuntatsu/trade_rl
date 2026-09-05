"""Metadata-only Development evaluation scopes for Universal Trade RL U2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.workflows.universal_trade_rl_u2_contract import UniversalTradeRLU2Contract
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    UniversalTradeRLU2TimePartition,
)
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

UNIVERSAL_TRADE_RL_U2_EVALUATION_SCOPE_SCHEMA: Final = (
    "universal_trade_rl_u2_evaluation_scope_v1"
)
UNIVERSAL_TRADE_RL_U2_DEVELOPMENT_SCOPE_CLOSURE_SCHEMA: Final = (
    "universal_trade_rl_u2_development_scope_closure_v1"
)

_CELL_DEFINITIONS: Final = (
    (
        "A",
        UniversalTradeRLSymbolRole.TRAIN,
        "seen_time_probe",
        "diagnostic_only",
    ),
    (
        "B",
        UniversalTradeRLSymbolRole.DEVELOPMENT,
        "seen_time_probe",
        "mandatory",
    ),
    (
        "C1",
        UniversalTradeRLSymbolRole.TRAIN,
        "development_future_1",
        "mandatory",
    ),
    (
        "C2",
        UniversalTradeRLSymbolRole.TRAIN,
        "development_future_2",
        "mandatory",
    ),
    (
        "D1",
        UniversalTradeRLSymbolRole.DEVELOPMENT,
        "development_future_1",
        "mandatory",
    ),
    (
        "D2",
        UniversalTradeRLSymbolRole.DEVELOPMENT,
        "development_future_2",
        "mandatory",
    ),
)
_CELL_ORDER: Final = tuple(item[0] for item in _CELL_DEFINITIONS)
_CELL_RULES: Final = {item[0]: item[1:] for item in _CELL_DEFINITIONS}


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2EvaluationScope:
    """One immutable U2 symbol × time × tile Development evaluation scope."""

    u2_contract_digest: str
    universe_manifest_digest: str
    time_partition_digest: str
    u1_contract_digest: str
    cell: str
    selection_use: str
    symbol_role: UniversalTradeRLSymbolRole
    concrete_symbol: str
    source_dataset_digest: str
    source_window: str
    tile_index: int
    outcome_start_bar_index: int
    outcome_stop_bar_index_exclusive: int
    evaluation_start_bar_index: int
    evaluation_stop_bar_index: int
    schema_version: str = UNIVERSAL_TRADE_RL_U2_EVALUATION_SCOPE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != UNIVERSAL_TRADE_RL_U2_EVALUATION_SCOPE_SCHEMA:
            raise ValueError("unsupported Universal Trade RL U2 evaluation scope schema")
        for field_name, value in (
            ("u2_contract_digest", self.u2_contract_digest),
            ("universe_manifest_digest", self.universe_manifest_digest),
            ("time_partition_digest", self.time_partition_digest),
            ("u1_contract_digest", self.u1_contract_digest),
            ("source_dataset_digest", self.source_dataset_digest),
        ):
            require_sha256(value, field=f"U2 evaluation scope {field_name}")

        rule = _CELL_RULES.get(self.cell)
        if rule is None:
            raise ValueError("U2 Development evaluation scope cell is invalid")
        expected_role, expected_window, expected_selection_use = rule
        if self.symbol_role is not expected_role:
            raise ValueError("U2 evaluation scope symbol role does not match its cell")
        if self.source_window != expected_window:
            raise ValueError("U2 evaluation scope time window does not match its cell")
        if self.selection_use != expected_selection_use:
            raise ValueError("U2 evaluation scope Selection role does not match its cell")
        if not isinstance(self.concrete_symbol, str) or not self.concrete_symbol:
            raise ValueError("U2 evaluation scope concrete symbol must be non-empty")

        tile_index = _non_negative_integer(self.tile_index, field="U2 tile index")
        outcome_start = _non_negative_integer(
            self.outcome_start_bar_index,
            field="U2 outcome start bar index",
        )
        outcome_stop = _non_negative_integer(
            self.outcome_stop_bar_index_exclusive,
            field="U2 outcome stop bar index",
        )
        evaluation_start = _non_negative_integer(
            self.evaluation_start_bar_index,
            field="U2 evaluation start bar index",
        )
        evaluation_stop = _non_negative_integer(
            self.evaluation_stop_bar_index,
            field="U2 evaluation stop bar index",
        )
        if outcome_stop <= outcome_start:
            raise ValueError("U2 evaluation scope outcome range is empty")
        if evaluation_start != outcome_start - 1:
            raise ValueError("U2 evaluation scope initial-state endpoint drifted")
        if evaluation_stop != outcome_stop:
            raise ValueError("U2 evaluation scope terminal endpoint drifted")
        if evaluation_stop - evaluation_start - 1 != outcome_stop - outcome_start:
            raise ValueError("U2 evaluation scope decision count drifted")
        object.__setattr__(self, "tile_index", tile_index)

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 evaluation scope artifact digest")
            if self.digest != expected:
                raise ValueError("Universal Trade RL U2 evaluation scope digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def decision_count(self) -> int:
        return self.outcome_stop_bar_index_exclusive - self.outcome_start_bar_index

    @property
    def evaluation_range(self) -> tuple[int, int]:
        return (self.evaluation_start_bar_index, self.evaluation_stop_bar_index)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "u2_contract_digest": self.u2_contract_digest,
            "universe_manifest_digest": self.universe_manifest_digest,
            "time_partition_digest": self.time_partition_digest,
            "u1_contract_digest": self.u1_contract_digest,
            "cell": self.cell,
            "selection_use": self.selection_use,
            "symbol_role": self.symbol_role.value,
            "concrete_symbol": self.concrete_symbol,
            "source_dataset_digest": self.source_dataset_digest,
            "source_window": self.source_window,
            "tile_index": self.tile_index,
            "outcome_start_bar_index": self.outcome_start_bar_index,
            "outcome_stop_bar_index_exclusive": self.outcome_stop_bar_index_exclusive,
            "evaluation_start_bar_index": self.evaluation_start_bar_index,
            "evaluation_stop_bar_index": self.evaluation_stop_bar_index,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2DevelopmentScopeClosure:
    """Canonical A/B/C1/C2/D1/D2 metadata closure; Admission is excluded."""

    universe_manifest_digest: str
    time_partition_digest: str
    u2_contract_digest: str
    scopes: tuple[UniversalTradeRLU2EvaluationScope, ...]
    schema_version: str = UNIVERSAL_TRADE_RL_U2_DEVELOPMENT_SCOPE_CLOSURE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != UNIVERSAL_TRADE_RL_U2_DEVELOPMENT_SCOPE_CLOSURE_SCHEMA:
            raise ValueError("unsupported U2 Development scope closure schema")
        for field_name, value in (
            ("universe_manifest_digest", self.universe_manifest_digest),
            ("time_partition_digest", self.time_partition_digest),
            ("u2_contract_digest", self.u2_contract_digest),
        ):
            require_sha256(value, field=f"U2 Development scope closure {field_name}")
        scopes = tuple(self.scopes)
        if not scopes:
            raise ValueError("U2 Development scope closure must not be empty")
        if any(not isinstance(scope, UniversalTradeRLU2EvaluationScope) for scope in scopes):
            raise TypeError("U2 Development scope closure contains invalid scopes")
        if any(
            scope.universe_manifest_digest != self.universe_manifest_digest
            or scope.time_partition_digest != self.time_partition_digest
            or scope.u2_contract_digest != self.u2_contract_digest
            for scope in scopes
        ):
            raise ValueError("U2 Development scope identity closure drifted")
        if any(scope.cell == "E" or scope.source_window == "admission_future" for scope in scopes):
            raise PermissionError("U2 Development scope closure must exclude Admission")
        order = {cell: index for index, cell in enumerate(_CELL_ORDER)}
        expected_order = tuple(
            sorted(
                scopes,
                key=lambda scope: (
                    order[scope.cell],
                    scope.concrete_symbol,
                    scope.tile_index,
                ),
            )
        )
        if scopes != expected_order:
            raise ValueError("U2 Development scopes are not in canonical order")
        if len({scope.digest for scope in scopes}) != len(scopes):
            raise ValueError("U2 Development scope closure contains duplicate scopes")
        object.__setattr__(self, "scopes", scopes)

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 Development scope closure digest")
            if self.digest != expected:
                raise ValueError("U2 Development scope closure digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "universe_manifest_digest": self.universe_manifest_digest,
            "time_partition_digest": self.time_partition_digest,
            "u2_contract_digest": self.u2_contract_digest,
            "scopes": tuple(scope.to_payload() for scope in self.scopes),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_universal_trade_rl_u2_development_scope_closure(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    time_partition: UniversalTradeRLU2TimePartition,
    u2_contract: UniversalTradeRLU2Contract,
) -> UniversalTradeRLU2DevelopmentScopeClosure:
    """Derive preregistered U2 A-D scopes without opening numeric market data."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U2 Development scope closure requires a U0 manifest")
    if not isinstance(time_partition, UniversalTradeRLU2TimePartition):
        raise TypeError("U2 Development scope closure requires a U2 time partition")
    if not isinstance(u2_contract, UniversalTradeRLU2Contract):
        raise TypeError("U2 Development scope closure requires a U2 contract")
    if time_partition.universe_manifest_digest != manifest.digest:
        raise ValueError("U2 Development time partition universe identity mismatch")
    if u2_contract.universe_manifest_digest != manifest.digest:
        raise ValueError("U2 Development contract universe identity mismatch")
    if u2_contract.time_partition_digest != time_partition.digest:
        raise ValueError("U2 Development contract time partition identity mismatch")

    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.DEVELOPMENT,
    )
    symbols_by_role = {
        UniversalTradeRLSymbolRole.TRAIN: access.train_symbols,
        UniversalTradeRLSymbolRole.DEVELOPMENT: access.development_symbols,
    }

    scopes: list[UniversalTradeRLU2EvaluationScope] = []
    for cell, role, window_name, selection_use in _CELL_DEFINITIONS:
        tiles = time_partition.tiles_for(window_name)
        for symbol in symbols_by_role[role]:
            entry = manifest.entry_for(symbol)
            for tile in tiles:
                evaluation_start, evaluation_stop = tile.evaluation_range
                scopes.append(
                    UniversalTradeRLU2EvaluationScope(
                        u2_contract_digest=u2_contract.digest,
                        universe_manifest_digest=manifest.digest,
                        time_partition_digest=time_partition.digest,
                        u1_contract_digest=u2_contract.u1_contract_digest,
                        cell=cell,
                        selection_use=selection_use,
                        symbol_role=role,
                        concrete_symbol=symbol,
                        source_dataset_digest=entry.dataset_digest,
                        source_window=window_name,
                        tile_index=tile.tile_index,
                        outcome_start_bar_index=tile.start_bar_index,
                        outcome_stop_bar_index_exclusive=(
                            tile.stop_bar_index_exclusive
                        ),
                        evaluation_start_bar_index=evaluation_start,
                        evaluation_stop_bar_index=evaluation_stop,
                    )
                )

    return UniversalTradeRLU2DevelopmentScopeClosure(
        universe_manifest_digest=manifest.digest,
        time_partition_digest=time_partition.digest,
        u2_contract_digest=u2_contract.digest,
        scopes=tuple(scopes),
    )


__all__ = [
    "UNIVERSAL_TRADE_RL_U2_DEVELOPMENT_SCOPE_CLOSURE_SCHEMA",
    "UNIVERSAL_TRADE_RL_U2_EVALUATION_SCOPE_SCHEMA",
    "UniversalTradeRLU2DevelopmentScopeClosure",
    "UniversalTradeRLU2EvaluationScope",
    "build_universal_trade_rl_u2_development_scope_closure",
]
