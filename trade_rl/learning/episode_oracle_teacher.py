"""Episode-aligned bounded Oracle teacher contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.learning.oracle_bellman_contracts import (
    OracleEpisodeInputs,
    OracleSolverConfig,
    OracleSolverProvenance,
)
from trade_rl.learning.oracle_solver import OracleBatchBackend, solve_oracle_episodes
from trade_rl.learning.oracle_teacher import (
    OracleTeacherConfig,
    _validate_train_range,
    portfolio_states,
)

EPISODE_ORACLE_TEACHER_SCHEMA: Final = "episode_aligned_oracle_teacher_v1"
EPISODE_ORACLE_CONTRACT_SCHEMA: Final = "oracle_episode_contract_v1"
EPISODE_ORACLE_BATCH_SCHEMA: Final = "episode_oracle_batch_v1"
_ALLOWED_INITIAL_STATE_MODES: Final = frozenset({"cash", "baseline"})
_EPSILON: Final = 1e-12

InitialWeightProvider: TypeAlias = Callable[[str, int], np.ndarray]


def _array_identity(value: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "shape": tuple(int(size) for size in array.shape),
    }


def _readonly_weight_vector(
    value: object,
    *,
    n_symbols: int,
    field: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (n_symbols,):
        raise ValueError(f"{field} must match dataset symbols")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{field} must be numeric")
    weights = np.asarray(raw, dtype=np.float64).copy(order="C")
    if not np.isfinite(weights).all():
        raise ValueError(f"{field} must be finite")
    weights.setflags(write=False)
    return weights


def _validated_oracle_initial_weights(
    dataset: MarketDataset,
    config: OracleTeacherConfig,
    value: object,
) -> np.ndarray:
    weights = _readonly_weight_vector(
        value,
        n_symbols=dataset.n_symbols,
        field="oracle initial weights",
    )
    if np.any(np.abs(weights) > config.max_abs_weight + _EPSILON):
        raise ValueError("oracle initial weights exceed max_abs_weight")
    if float(np.abs(weights).sum()) > config.max_gross + _EPSILON:
        raise ValueError("oracle initial weights exceed max_gross")
    if not config.execution_cost.allow_short and np.any(weights < -_EPSILON):
        raise ValueError("oracle initial weights contain a disallowed short position")
    return weights


def episode_oracle_target_path(
    dataset: MarketDataset,
    train_range: tuple[int, int],
    config: OracleTeacherConfig,
    *,
    initial_weights: np.ndarray,
    solver_config: OracleSolverConfig | None = None,
    accelerator_backend: OracleBatchBackend | None = None,
) -> np.ndarray:
    """Return an Oracle path seeded from one explicit episode initial state."""

    if config.execution_cost.margin_mode != "cross":
        raise ValueError("oracle currently supports cross margin only")
    start, stop = _validate_train_range(dataset, train_range)
    initial = _validated_oracle_initial_weights(dataset, config, initial_weights)
    result = solve_oracle_episodes(
        dataset,
        states=portfolio_states(dataset, config),
        episode_inputs=OracleEpisodeInputs(
            episode_indices=np.array([0], dtype=np.int64),
            starts=np.array([start], dtype=np.int64),
            stops=np.array([stop], dtype=np.int64),
            initial_weights=initial[None, :],
        ),
        parameters=config.bellman_parameters,
        solver_config=solver_config or OracleSolverConfig(),
        accelerator_backend=accelerator_backend,
    )
    return result.targets[0]


@dataclass(frozen=True, slots=True)
class OracleEpisodeSamplingConfig:
    """Deterministic PPO-aligned episode sampling contract."""

    episode_bars: int
    episode_count: int
    initial_state_modes: tuple[str, ...] = ("cash",)
    seed: int = 0
    sampling_mode: str = "uniform_with_replacement"
    schema_version: str = EPISODE_ORACLE_TEACHER_SCHEMA

    def __post_init__(self) -> None:
        for name, value in (
            ("episode_bars", self.episode_bars),
            ("episode_count", self.episode_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        modes = tuple(str(mode) for mode in self.initial_state_modes)
        if (
            not modes
            or len(set(modes)) != len(modes)
            or any(mode not in _ALLOWED_INITIAL_STATE_MODES for mode in modes)
        ):
            raise ValueError("initial_state_modes must contain unique maintained modes")
        if self.sampling_mode != "uniform_with_replacement":
            raise ValueError("unsupported Oracle episode sampling mode")
        if self.schema_version != EPISODE_ORACLE_TEACHER_SCHEMA:
            raise ValueError("unsupported episode Oracle teacher schema")
        object.__setattr__(self, "initial_state_modes", modes)

    @property
    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class OracleEpisodeContract:
    """One immutable finite-horizon Oracle episode contract."""

    dataset_id: str
    episode_index: int
    start: int
    stop: int
    initial_state_mode: str
    initial_weights: np.ndarray
    digest: str = ""
    schema_version: str = EPISODE_ORACLE_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        dataset_id = require_sha256(self.dataset_id, field="dataset_id")
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or self.episode_index < 0
        ):
            raise ValueError("episode_index must be a non-negative integer")
        if (
            isinstance(self.start, bool)
            or isinstance(self.stop, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.stop, int)
            or self.start < 0
            or self.stop <= self.start + 1
        ):
            raise ValueError("Oracle episode bounds must contain at least one decision")
        if self.initial_state_mode not in _ALLOWED_INITIAL_STATE_MODES:
            raise ValueError("unsupported Oracle episode initial state mode")
        weights = np.asarray(self.initial_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0 or not np.isfinite(weights).all():
            raise ValueError("Oracle episode initial weights must be a finite vector")
        weights = weights.copy(order="C")
        weights.setflags(write=False)
        if self.schema_version != EPISODE_ORACLE_CONTRACT_SCHEMA:
            raise ValueError("unsupported Oracle episode contract schema")
        expected = content_digest(
            {
                "dataset_id": dataset_id,
                "episode_index": self.episode_index,
                "initial_state_mode": self.initial_state_mode,
                "initial_weights": _array_identity(weights),
                "schema_version": self.schema_version,
                "start": self.start,
                "stop": self.stop,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("Oracle episode contract digest mismatch")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "initial_weights", weights)
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class EpisodeOracleBatch:
    """Immutable per-episode Oracle targets with explicit boundaries."""

    dataset_id: str
    teacher_config_digest: str
    sampling_config_digest: str
    contracts: tuple[OracleEpisodeContract, ...]
    targets: tuple[np.ndarray, ...]
    solver_provenance: OracleSolverProvenance | None = None
    digest: str = ""
    schema_version: str = EPISODE_ORACLE_BATCH_SCHEMA

    def __post_init__(self) -> None:
        dataset_id = require_sha256(self.dataset_id, field="dataset_id")
        teacher_digest = require_sha256(
            self.teacher_config_digest,
            field="teacher_config_digest",
        )
        sampling_digest = require_sha256(
            self.sampling_config_digest,
            field="sampling_config_digest",
        )
        contracts = tuple(self.contracts)
        raw_targets = tuple(self.targets)
        if not contracts or len(contracts) != len(raw_targets):
            raise ValueError(
                "Oracle batch contracts and targets must be non-empty and equal"
            )
        resolved_targets: list[np.ndarray] = []
        for contract, value in zip(contracts, raw_targets, strict=True):
            if contract.dataset_id != dataset_id:
                raise ValueError("Oracle batch contract dataset identity mismatch")
            target = np.asarray(value, dtype=np.float32)
            expected_shape = (
                contract.stop - contract.start - 1,
                contract.initial_weights.size,
            )
            if target.shape != expected_shape or not np.isfinite(target).all():
                raise ValueError("Oracle batch target shape or values are invalid")
            target = target.copy(order="C")
            target.setflags(write=False)
            resolved_targets.append(target)
        if self.schema_version != EPISODE_ORACLE_BATCH_SCHEMA:
            raise ValueError("unsupported episode Oracle batch schema")
        if self.solver_provenance is not None and not isinstance(
            self.solver_provenance, OracleSolverProvenance
        ):
            raise ValueError("solver_provenance must be OracleSolverProvenance")
        targets = tuple(resolved_targets)
        expected = content_digest(
            {
                "contracts": tuple(contract.digest for contract in contracts),
                "dataset_id": dataset_id,
                "sampling_config_digest": sampling_digest,
                "schema_version": self.schema_version,
                "solver_provenance_digest": (
                    None
                    if self.solver_provenance is None
                    else self.solver_provenance.digest
                ),
                "targets": tuple(_array_identity(target) for target in targets),
                "teacher_config_digest": teacher_digest,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("episode Oracle batch digest mismatch")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "teacher_config_digest", teacher_digest)
        object.__setattr__(self, "sampling_config_digest", sampling_digest)
        object.__setattr__(self, "contracts", contracts)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "digest", expected)

    @property
    def episode_count(self) -> int:
        return len(self.contracts)

    @property
    def decision_count(self) -> int:
        return sum(target.shape[0] for target in self.targets)


def sample_oracle_episode_contracts(
    dataset: MarketDataset,
    *,
    minimum_start_index: int,
    maximum_stop_index: int | None = None,
    config: OracleEpisodeSamplingConfig,
    initial_weight_provider: InitialWeightProvider | None = None,
) -> tuple[OracleEpisodeContract, ...]:
    """Sample deterministic fixed-horizon episode contracts from one train view."""

    if (
        isinstance(minimum_start_index, bool)
        or not isinstance(minimum_start_index, int)
        or minimum_start_index < 0
        or minimum_start_index >= dataset.n_bars
    ):
        raise ValueError("minimum_start_index is outside the dataset")
    if maximum_stop_index is None:
        resolved_stop = dataset.n_bars
    else:
        if (
            isinstance(maximum_stop_index, bool)
            or not isinstance(maximum_stop_index, int)
            or maximum_stop_index <= minimum_start_index
            or maximum_stop_index > dataset.n_bars
        ):
            raise ValueError("maximum_stop_index is outside the dataset train range")
        resolved_stop = maximum_stop_index
    maximum_start = resolved_stop - config.episode_bars - 1
    if maximum_start < minimum_start_index:
        raise ValueError(
            "dataset train range does not contain a complete episode horizon"
        )
    if "baseline" in config.initial_state_modes and initial_weight_provider is None:
        raise ValueError("non-cash episodes require an initial weight provider")

    valid_starts = np.arange(
        minimum_start_index,
        maximum_start + 1,
        dtype=np.int64,
    )
    rng = np.random.default_rng(config.seed)
    starts = rng.choice(valid_starts, size=config.episode_count, replace=True)
    mode_indices = rng.integers(
        0,
        len(config.initial_state_modes),
        size=config.episode_count,
    )
    contracts: list[OracleEpisodeContract] = []
    for episode_index, (raw_start, raw_mode_index) in enumerate(
        zip(starts, mode_indices, strict=True)
    ):
        start = int(raw_start)
        mode = config.initial_state_modes[int(raw_mode_index)]
        if mode == "cash":
            weights = np.zeros(dataset.n_symbols, dtype=np.float64)
        else:
            if initial_weight_provider is None:  # pragma: no cover - checked above
                raise RuntimeError("initial weight provider disappeared")
            weights = initial_weight_provider(mode, start)
        initial = _readonly_weight_vector(
            weights,
            n_symbols=dataset.n_symbols,
            field="episode initial weights",
        )
        contracts.append(
            OracleEpisodeContract(
                dataset_id=dataset.dataset_id,
                episode_index=episode_index,
                start=start,
                stop=start + config.episode_bars + 1,
                initial_state_mode=mode,
                initial_weights=initial,
            )
        )
    return tuple(contracts)


def build_episode_oracle_batch(
    dataset: MarketDataset,
    *,
    minimum_start_index: int,
    maximum_stop_index: int | None = None,
    sampling_config: OracleEpisodeSamplingConfig,
    teacher_config: OracleTeacherConfig,
    initial_weight_provider: InitialWeightProvider | None = None,
    max_workers: int = 1,
    solver_config: OracleSolverConfig | None = None,
    accelerator_backend: OracleBatchBackend | None = None,
) -> EpisodeOracleBatch:
    """Build bounded Oracle targets for independently sampled PPO-like episodes."""

    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise ValueError("Oracle episode worker count must be an integer")
    if max_workers <= 0:
        raise ValueError("Oracle episode worker count must be positive")
    contracts = sample_oracle_episode_contracts(
        dataset,
        minimum_start_index=minimum_start_index,
        maximum_stop_index=maximum_stop_index,
        config=sampling_config,
        initial_weight_provider=initial_weight_provider,
    )
    result = solve_oracle_episodes(
        dataset,
        states=portfolio_states(dataset, teacher_config),
        episode_inputs=OracleEpisodeInputs(
            episode_indices=np.asarray(
                [contract.episode_index for contract in contracts],
                dtype=np.int64,
            ),
            starts=np.asarray(
                [contract.start for contract in contracts], dtype=np.int64
            ),
            stops=np.asarray([contract.stop for contract in contracts], dtype=np.int64),
            initial_weights=np.stack(
                [contract.initial_weights for contract in contracts],
                axis=0,
            ),
        ),
        parameters=teacher_config.bellman_parameters,
        solver_config=solver_config or OracleSolverConfig(),
        accelerator_backend=accelerator_backend,
    )
    return EpisodeOracleBatch(
        dataset_id=dataset.dataset_id,
        teacher_config_digest=teacher_config.digest,
        sampling_config_digest=sampling_config.digest,
        contracts=contracts,
        targets=result.targets,
        solver_provenance=result.provenance,
    )


__all__ = [
    "EPISODE_ORACLE_BATCH_SCHEMA",
    "EPISODE_ORACLE_CONTRACT_SCHEMA",
    "EPISODE_ORACLE_TEACHER_SCHEMA",
    "EpisodeOracleBatch",
    "InitialWeightProvider",
    "OracleEpisodeContract",
    "OracleEpisodeSamplingConfig",
    "build_episode_oracle_batch",
    "episode_oracle_target_path",
    "sample_oracle_episode_contracts",
]
