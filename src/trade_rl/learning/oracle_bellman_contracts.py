"""Backend-neutral contracts for bounded Oracle Bellman solvers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np
from numpy.typing import DTypeLike

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.risk.portfolio import PortfolioRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig

ORACLE_SOLVER_CONFIG_SCHEMA: Final = "oracle_solver_config_v1"
ORACLE_BELLMAN_PARAMETERS_SCHEMA: Final = "oracle_bellman_parameters_v1"
ORACLE_EPISODE_INPUTS_SCHEMA: Final = "oracle_episode_inputs_v1"
ORACLE_SOLVER_PROVENANCE_SCHEMA: Final = "oracle_solver_provenance_v1"
ORACLE_SOLVE_RESULT_SCHEMA: Final = "oracle_solve_result_v1"
SOLVER_CONTRACT: Final = "batched_bellman_v1"
TIE_BREAK_CONTRACT: Final = "lowest_prior_within_tolerance_v1"

SolverSelection = Literal["numpy", "cuda", "cuda_or_numpy"]
CompileMode = Literal["disabled", "reduce_overhead"]
SolverBackend = Literal["numpy", "torch_cuda"]


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _payload_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _payload_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _payload_string(value, field=field)


def _payload_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _payload_optional_integer(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _payload_integer(value, field=field)


def _payload_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _payload_optional_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _payload_float(value, field=field)


def _readonly_array(
    value: object,
    *,
    dtype: DTypeLike,
    ndim: int,
    field: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{field} must be numeric")
    array = np.asarray(raw, dtype=dtype).copy(order="C")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must contain finite values")
    array.setflags(write=False)
    return array


def _readonly_integer_array(
    value: object,
    *,
    ndim: int,
    field: str,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != ndim:
        raise ValueError(f"{field} must be {ndim}-dimensional")
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{field} must contain integers")
    array = np.asarray(raw, dtype=np.int64).copy(order="C")
    array.setflags(write=False)
    return array


def _array_identity(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    return {
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "shape": tuple(int(size) for size in contiguous.shape),
    }


@dataclass(frozen=True, slots=True)
class OracleSolverConfig:
    """Explicit numerical and resource contract for one Oracle solver run."""

    selection: SolverSelection = "numpy"
    numeric_dtype: str = "float64"
    tie_tolerance: float = 1e-12
    episode_batch_size: int = 8
    target_state_block_size: int | None = None
    cuda_memory_fraction: float = 0.65
    compile_mode: CompileMode = "disabled"
    compile_chunk_size: int = 16
    schema_version: str = ORACLE_SOLVER_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.selection not in {"numpy", "cuda", "cuda_or_numpy"}:
            raise ValueError("selection is unsupported")
        if self.numeric_dtype != "float64":
            raise ValueError("numeric_dtype must be float64")
        if not math.isfinite(self.tie_tolerance) or self.tie_tolerance <= 0.0:
            raise ValueError("tie_tolerance must be finite and positive")
        _positive_integer(self.episode_batch_size, field="episode_batch_size")
        if self.target_state_block_size is not None:
            _positive_integer(
                self.target_state_block_size,
                field="target_state_block_size",
            )
        if (
            not math.isfinite(self.cuda_memory_fraction)
            or not 0.0 < self.cuda_memory_fraction <= 1.0
        ):
            raise ValueError("cuda_memory_fraction must be within (0, 1]")
        if self.compile_mode not in {"disabled", "reduce_overhead"}:
            raise ValueError("compile_mode is unsupported")
        _positive_integer(self.compile_chunk_size, field="compile_chunk_size")
        if self.schema_version != ORACLE_SOLVER_CONFIG_SCHEMA:
            raise ValueError("unsupported Oracle solver config schema")

    @property
    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class OracleBellmanParameters:
    """Teacher parameters copied into a solver-owned immutable boundary."""

    execution_cost: ExecutionCostConfig
    portfolio_risk: PortfolioRiskConfig
    positions: tuple[float, ...]
    max_gross: float
    max_abs_weight: float
    entry_threshold: float
    exit_threshold: float
    no_trade_band: float
    reference_portfolio_value: float
    maximum_states: int
    signal_delay_decisions: int
    approximation_contract: str
    control_tie_break_penalty: float
    schema_version: str = ORACLE_BELLMAN_PARAMETERS_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.execution_cost, ExecutionCostConfig):
            raise ValueError("execution_cost must be ExecutionCostConfig")
        if not isinstance(self.portfolio_risk, PortfolioRiskConfig):
            raise ValueError("portfolio_risk must be PortfolioRiskConfig")
        positions = tuple(float(value) for value in self.positions)
        if not positions or not np.isfinite(positions).all():
            raise ValueError("positions must contain finite values")
        for field in (
            "max_gross",
            "max_abs_weight",
            "reference_portfolio_value",
            "control_tie_break_penalty",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        for field in ("entry_threshold", "exit_threshold", "no_trade_band"):
            value = float(getattr(self, field))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
        _positive_integer(self.maximum_states, field="maximum_states")
        if self.signal_delay_decisions not in {0, 1}:
            raise ValueError("signal_delay_decisions must be zero or one")
        if not self.approximation_contract:
            raise ValueError("approximation_contract must be non-empty")
        if self.schema_version != ORACLE_BELLMAN_PARAMETERS_SCHEMA:
            raise ValueError("unsupported Oracle Bellman parameter schema")
        object.__setattr__(self, "positions", positions)

    @property
    def digest(self) -> str:
        return content_digest(self)


@dataclass(frozen=True, slots=True)
class OracleEpisodeInputs:
    """One same-symbol batched collection of independent episode bounds."""

    episode_indices: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    initial_weights: np.ndarray
    schema_version: str = ORACLE_EPISODE_INPUTS_SCHEMA

    def __post_init__(self) -> None:
        indices = _readonly_integer_array(
            self.episode_indices,
            ndim=1,
            field="episode_indices",
        )
        starts = _readonly_integer_array(
            self.starts,
            ndim=1,
            field="starts",
        )
        stops = _readonly_integer_array(
            self.stops,
            ndim=1,
            field="stops",
        )
        initial = _readonly_array(
            self.initial_weights,
            dtype=np.dtype(np.float64),
            ndim=2,
            field="initial_weights",
        )
        count = indices.size
        if count == 0 or starts.size != count or stops.size != count:
            raise ValueError("episode input arrays must have equal non-zero length")
        if initial.shape[0] != count or initial.shape[1] == 0:
            raise ValueError("initial_weights must match episode count and symbols")
        if np.any(indices < 0) or len(set(indices.tolist())) != count:
            raise ValueError("episode_indices must be unique and non-negative")
        if np.any(starts < 0) or np.any(stops <= starts + 1):
            raise ValueError("episode bounds must contain at least one decision")
        if self.schema_version != ORACLE_EPISODE_INPUTS_SCHEMA:
            raise ValueError("unsupported Oracle episode inputs schema")
        object.__setattr__(self, "episode_indices", indices)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "stops", stops)
        object.__setattr__(self, "initial_weights", initial)

    @property
    def episode_count(self) -> int:
        return int(self.starts.size)


@dataclass(frozen=True, slots=True)
class OracleSolverProvenance:
    """Numerical identity plus separately serialized runtime evidence."""

    backend: SolverBackend
    solver_config_digest: str
    market_tape_digest: str
    numeric_dtype: str
    tie_tolerance: float
    episode_batch_size: int
    target_state_block_size: int | None
    compile_mode: CompileMode
    compile_chunk_size: int
    fallback_reason: str | None = None
    oom_retry_performed: bool = False
    solver_wall_time_seconds: float | None = None
    peak_host_memory_bytes: int | None = None
    peak_device_memory_bytes: int | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    device_name: str | None = None
    compute_capability: str | None = None
    solver_contract: str = SOLVER_CONTRACT
    tie_break_contract: str = TIE_BREAK_CONTRACT
    digest: str = ""
    schema_version: str = ORACLE_SOLVER_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.backend not in {"numpy", "torch_cuda"}:
            raise ValueError("backend is unsupported")
        config_digest = require_sha256(
            self.solver_config_digest,
            field="solver_config_digest",
        )
        tape_digest = require_sha256(
            self.market_tape_digest,
            field="market_tape_digest",
        )
        if self.numeric_dtype != "float64":
            raise ValueError("numeric_dtype must be float64")
        if not math.isfinite(self.tie_tolerance) or self.tie_tolerance <= 0.0:
            raise ValueError("tie_tolerance must be finite and positive")
        _positive_integer(self.episode_batch_size, field="episode_batch_size")
        if self.target_state_block_size is not None:
            _positive_integer(
                self.target_state_block_size,
                field="target_state_block_size",
            )
        if self.compile_mode not in {"disabled", "reduce_overhead"}:
            raise ValueError("compile_mode is unsupported")
        _positive_integer(self.compile_chunk_size, field="compile_chunk_size")
        if self.solver_contract != SOLVER_CONTRACT:
            raise ValueError("unsupported solver contract")
        if self.tie_break_contract != TIE_BREAK_CONTRACT:
            raise ValueError("unsupported tie-break contract")
        if not isinstance(self.oom_retry_performed, bool):
            raise ValueError("oom_retry_performed must be a boolean")
        if self.solver_wall_time_seconds is not None and (
            not math.isfinite(self.solver_wall_time_seconds)
            or self.solver_wall_time_seconds < 0.0
        ):
            raise ValueError("solver_wall_time_seconds must be finite and non-negative")
        for field, memory_value in (
            ("peak_host_memory_bytes", self.peak_host_memory_bytes),
            ("peak_device_memory_bytes", self.peak_device_memory_bytes),
        ):
            if memory_value is not None and (
                isinstance(memory_value, bool)
                or not isinstance(memory_value, int)
                or memory_value < 0
            ):
                raise ValueError(f"{field} must be a non-negative integer")
        for field, text_value in (
            ("fallback_reason", self.fallback_reason),
            ("torch_version", self.torch_version),
            ("cuda_version", self.cuda_version),
            ("device_name", self.device_name),
            ("compute_capability", self.compute_capability),
        ):
            if text_value is not None and (
                not isinstance(text_value, str) or not text_value.strip()
            ):
                raise ValueError(f"{field} must be a non-empty string when present")
        if self.schema_version != ORACLE_SOLVER_PROVENANCE_SCHEMA:
            raise ValueError("unsupported Oracle solver provenance schema")
        expected = content_digest(self.identity_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Oracle solver provenance digest mismatch")
        object.__setattr__(self, "solver_config_digest", config_digest)
        object.__setattr__(self, "market_tape_digest", tape_digest)
        object.__setattr__(self, "digest", expected)

    def identity_payload(self) -> dict[str, object]:
        """Return only values that can change the numerical solve semantics."""

        return {
            "market_tape_digest": self.market_tape_digest,
            "numeric_dtype": self.numeric_dtype,
            "schema_version": self.schema_version,
            "solver_contract": self.solver_contract,
            "tie_break_contract": self.tie_break_contract,
            "tie_tolerance": self.tie_tolerance,
        }

    def runtime_payload(self) -> dict[str, object]:
        """Return execution-policy and runtime evidence excluded from identity."""

        return {
            "backend": self.backend,
            "compile_chunk_size": self.compile_chunk_size,
            "compile_mode": self.compile_mode,
            "compute_capability": self.compute_capability,
            "cuda_version": self.cuda_version,
            "device_name": self.device_name,
            "episode_batch_size": self.episode_batch_size,
            "fallback_reason": self.fallback_reason,
            "oom_retry_performed": self.oom_retry_performed,
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
            "peak_host_memory_bytes": self.peak_host_memory_bytes,
            "solver_config_digest": self.solver_config_digest,
            "solver_wall_time_seconds": self.solver_wall_time_seconds,
            "target_state_block_size": self.target_state_block_size,
            "torch_version": self.torch_version,
        }

    def serialized_payload(self) -> dict[str, object]:
        """Return complete provenance while keeping a stable numerical digest."""

        return {
            **self.identity_payload(),
            **self.runtime_payload(),
            "digest": self.digest,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> OracleSolverProvenance:
        """Parse fail-closed serialized provenance from an artifact manifest."""

        try:
            raw_retry = value["oom_retry_performed"]
            if not isinstance(raw_retry, bool):
                raise ValueError("oom_retry_performed must be a boolean")
            backend = cast(
                SolverBackend,
                _payload_string(value["backend"], field="backend"),
            )
            compile_mode = cast(
                CompileMode,
                _payload_string(value["compile_mode"], field="compile_mode"),
            )
            return cls(
                backend=backend,
                solver_config_digest=_payload_string(
                    value["solver_config_digest"],
                    field="solver_config_digest",
                ),
                market_tape_digest=_payload_string(
                    value["market_tape_digest"],
                    field="market_tape_digest",
                ),
                numeric_dtype=_payload_string(
                    value["numeric_dtype"],
                    field="numeric_dtype",
                ),
                tie_tolerance=_payload_float(
                    value["tie_tolerance"],
                    field="tie_tolerance",
                ),
                episode_batch_size=_payload_integer(
                    value["episode_batch_size"],
                    field="episode_batch_size",
                ),
                target_state_block_size=_payload_optional_integer(
                    value.get("target_state_block_size"),
                    field="target_state_block_size",
                ),
                compile_mode=compile_mode,
                compile_chunk_size=_payload_integer(
                    value["compile_chunk_size"],
                    field="compile_chunk_size",
                ),
                fallback_reason=_payload_optional_string(
                    value.get("fallback_reason"),
                    field="fallback_reason",
                ),
                oom_retry_performed=raw_retry,
                solver_wall_time_seconds=_payload_optional_float(
                    value.get("solver_wall_time_seconds"),
                    field="solver_wall_time_seconds",
                ),
                peak_host_memory_bytes=_payload_optional_integer(
                    value.get("peak_host_memory_bytes"),
                    field="peak_host_memory_bytes",
                ),
                peak_device_memory_bytes=_payload_optional_integer(
                    value.get("peak_device_memory_bytes"),
                    field="peak_device_memory_bytes",
                ),
                torch_version=_payload_optional_string(
                    value.get("torch_version"),
                    field="torch_version",
                ),
                cuda_version=_payload_optional_string(
                    value.get("cuda_version"),
                    field="cuda_version",
                ),
                device_name=_payload_optional_string(
                    value.get("device_name"),
                    field="device_name",
                ),
                compute_capability=_payload_optional_string(
                    value.get("compute_capability"),
                    field="compute_capability",
                ),
                solver_contract=_payload_string(
                    value["solver_contract"],
                    field="solver_contract",
                ),
                tie_break_contract=_payload_string(
                    value["tie_break_contract"],
                    field="tie_break_contract",
                ),
                digest=_payload_string(value["digest"], field="digest"),
                schema_version=_payload_string(
                    value["schema_version"],
                    field="schema_version",
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Oracle solver provenance payload is invalid") from error

    @classmethod
    def numpy_reference(
        cls,
        *,
        config: OracleSolverConfig,
        market_tape_digest: str,
    ) -> OracleSolverProvenance:
        return cls(
            backend="numpy",
            solver_config_digest=config.digest,
            market_tape_digest=market_tape_digest,
            numeric_dtype=config.numeric_dtype,
            tie_tolerance=config.tie_tolerance,
            episode_batch_size=config.episode_batch_size,
            target_state_block_size=config.target_state_block_size,
            compile_mode=config.compile_mode,
            compile_chunk_size=config.compile_chunk_size,
        )


@dataclass(frozen=True, slots=True)
class OracleSolveResult:
    """Submitted target paths and final objective values for one batch."""

    targets: tuple[np.ndarray, ...]
    final_scores: np.ndarray
    provenance: OracleSolverProvenance
    digest: str = ""
    schema_version: str = ORACLE_SOLVE_RESULT_SCHEMA

    def __post_init__(self) -> None:
        scores = _readonly_array(
            self.final_scores,
            dtype=np.dtype(np.float64),
            ndim=1,
            field="final_scores",
        )
        raw_targets = tuple(self.targets)
        if not raw_targets or len(raw_targets) != scores.size:
            raise ValueError("solve result must contain one target path per episode")
        targets: list[np.ndarray] = []
        for value in raw_targets:
            target = _readonly_array(
                value,
                dtype=np.dtype(np.float32),
                ndim=2,
                field="targets",
            )
            if target.shape[0] == 0 or target.shape[1] == 0:
                raise ValueError("target paths must be non-empty")
            targets.append(target)
        if not isinstance(self.provenance, OracleSolverProvenance):
            raise ValueError("provenance must be OracleSolverProvenance")
        if self.schema_version != ORACLE_SOLVE_RESULT_SCHEMA:
            raise ValueError("unsupported Oracle solve result schema")
        resolved_targets = tuple(targets)
        expected = content_digest(
            {
                "final_scores": _array_identity(scores),
                "provenance_digest": self.provenance.digest,
                "schema_version": self.schema_version,
                "targets": tuple(
                    _array_identity(target) for target in resolved_targets
                ),
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("Oracle solve result digest mismatch")
        object.__setattr__(self, "targets", resolved_targets)
        object.__setattr__(self, "final_scores", scores)
        object.__setattr__(self, "digest", expected)


class OracleBackendFailure(RuntimeError):
    """Typed backend failure eligible for explicit orchestration policy."""

    def __init__(self, backend: str, reason: str) -> None:
        if not backend:
            raise ValueError("backend must be non-empty")
        if not reason:
            raise ValueError("reason must be non-empty")
        self.backend = backend
        self.reason = reason
        super().__init__(f"Oracle backend {backend} failed: {reason}")


__all__ = [
    "CompileMode",
    "ORACLE_BELLMAN_PARAMETERS_SCHEMA",
    "ORACLE_EPISODE_INPUTS_SCHEMA",
    "ORACLE_SOLVE_RESULT_SCHEMA",
    "ORACLE_SOLVER_CONFIG_SCHEMA",
    "ORACLE_SOLVER_PROVENANCE_SCHEMA",
    "OracleBackendFailure",
    "OracleBellmanParameters",
    "OracleEpisodeInputs",
    "OracleSolveResult",
    "OracleSolverConfig",
    "OracleSolverProvenance",
    "SOLVER_CONTRACT",
    "SolverBackend",
    "SolverSelection",
    "TIE_BREAK_CONTRACT",
]
