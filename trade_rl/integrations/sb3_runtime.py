"""Runtime and resource policy for Stable-Baselines3 training."""

from __future__ import annotations

import os
from typing import Any, cast

import numpy as np

from trade_rl.learning.episode_oracle_bc import oracle_episode_sampling_config
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeSamplingConfig
from trade_rl.learning.oracle_bellman_contracts import (
    CompileMode,
    OracleSolverConfig,
    SolverSelection,
)
from trade_rl.learning.oracle_solver import OracleBatchBackend
from trade_rl.learning.oracle_teacher import OracleTeacherConfig
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.training_modes import CudaRuntimeMode


def _lagrangian_probe_worker_count(n_envs: int) -> int:
    raw = os.environ.get("TRADE_RL_LAGRANGIAN_PROBE_WORKERS", "1").strip()
    try:
        configured = int(raw)
    except ValueError as error:
        raise ValueError(
            "TRADE_RL_LAGRANGIAN_PROBE_WORKERS must be an integer"
        ) from error
    if configured <= 0:
        raise ValueError("TRADE_RL_LAGRANGIAN_PROBE_WORKERS must be positive")
    return min(n_envs, configured)


def oracle_teacher_config_for_environment(environment: Any) -> OracleTeacherConfig:
    """Derive the exact deterministic Oracle contract from a training environment."""

    risk_service = getattr(environment, "pre_trade_risk", None)
    portfolio_service = getattr(environment, "portfolio_risk", None)
    environment_config = getattr(environment, "config", None)
    risk_config = getattr(risk_service, "config", None)
    portfolio_config = getattr(portfolio_service, "config", None)
    execution_cost = getattr(environment_config, "execution_cost", None)
    signal_delay = getattr(environment_config, "signal_delay_decisions", None)
    initial_capital = getattr(environment, "initial_capital", None)
    if risk_config is None or portfolio_config is None or execution_cost is None:
        raise TypeError(
            "Oracle teacher environment is missing risk or execution config"
        )
    if (
        isinstance(initial_capital, bool)
        or not isinstance(initial_capital, (int, float))
        or not np.isfinite(initial_capital)
        or initial_capital <= 0.0
    ):
        raise ValueError("Oracle teacher environment initial_capital must be positive")
    if isinstance(signal_delay, bool) or not isinstance(signal_delay, int):
        raise TypeError("Oracle teacher signal_delay_decisions must be an integer")
    return OracleTeacherConfig(
        execution_cost=execution_cost,
        portfolio_risk=portfolio_config,
        max_gross=risk_config.max_gross,
        max_abs_weight=risk_config.max_abs_weight,
        entry_threshold=risk_config.entry_threshold,
        exit_threshold=risk_config.exit_threshold,
        no_trade_band=risk_config.no_trade_band,
        reference_portfolio_value=float(initial_capital),
        signal_delay_decisions=signal_delay,
    )


def _oracle_solver_config() -> OracleSolverConfig:
    """Parse the explicit Oracle backend/resource contract before generation."""

    selection_name = "TRADE_RL_ORACLE_SOLVER"
    raw_selection = os.environ.get(selection_name, "numpy").strip()
    if raw_selection not in {"numpy", "cuda", "cuda_or_numpy"}:
        raise ValueError(
            f"{selection_name} must be one of numpy, cuda, or cuda_or_numpy"
        )

    def positive_integer(name: str, default: int) -> int:
        raw = os.environ.get(name, str(default)).strip()
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer") from error
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    batch_size = positive_integer("TRADE_RL_ORACLE_EPISODE_BATCH_SIZE", 8)
    block_name = "TRADE_RL_ORACLE_TARGET_STATE_BLOCK_SIZE"
    raw_block = os.environ.get(block_name, "").strip()
    block_size: int | None = None
    if raw_block:
        try:
            block_size = int(raw_block)
        except ValueError as error:
            raise ValueError(f"{block_name} must be an integer when set") from error
        if block_size <= 0:
            raise ValueError(f"{block_name} must be positive when set")

    memory_name = "TRADE_RL_ORACLE_CUDA_MEMORY_FRACTION"
    raw_memory = os.environ.get(memory_name, "0.65").strip()
    try:
        memory_fraction = float(raw_memory)
    except ValueError as error:
        raise ValueError(f"{memory_name} must be numeric") from error
    if not np.isfinite(memory_fraction) or not 0.0 < memory_fraction <= 1.0:
        raise ValueError(f"{memory_name} must be within (0, 1]")

    compile_name = "TRADE_RL_ORACLE_COMPILE_MODE"
    raw_compile = os.environ.get(compile_name, "disabled").strip()
    if raw_compile not in {"disabled", "reduce_overhead"}:
        raise ValueError(f"{compile_name} must be disabled or reduce_overhead")
    chunk_name = "TRADE_RL_ORACLE_COMPILE_CHUNK_SIZE"
    compile_chunk_size = positive_integer(chunk_name, 16)
    if compile_chunk_size not in {8, 16, 32, 64}:
        raise ValueError(f"{chunk_name} must be one of 8, 16, 32, or 64")

    return OracleSolverConfig(
        selection=cast(SolverSelection, raw_selection),
        episode_batch_size=batch_size,
        target_state_block_size=block_size,
        cuda_memory_fraction=memory_fraction,
        compile_mode=cast(CompileMode, raw_compile),
        compile_chunk_size=compile_chunk_size,
    )


def _oracle_accelerator_backend(
    solver_config: OracleSolverConfig,
) -> OracleBatchBackend | None:
    """Resolve the concrete optional backend only for explicit CUDA selection."""

    if solver_config.selection == "numpy":
        return None
    from trade_rl.integrations.oracle_solver import solve_torch_cuda_oracle_batch

    return solve_torch_cuda_oracle_batch


def _teacher_worker_count(
    n_envs: int,
    *,
    solver_config: OracleSolverConfig | None = None,
) -> int:
    raw = os.environ.get("TRADE_RL_TEACHER_WORKERS", "").strip()
    try:
        if raw:
            configured = int(raw)
        else:
            configured = n_envs if solver_config is None else 1
    except ValueError as error:
        raise ValueError("TRADE_RL_TEACHER_WORKERS must be an integer") from error
    if configured <= 0:
        raise ValueError("TRADE_RL_TEACHER_WORKERS must be positive")
    if solver_config is not None and solver_config.selection != "numpy":
        if configured != 1:
            raise ValueError("CUDA Oracle solving requires TRADE_RL_TEACHER_WORKERS=1")
    return min(n_envs, configured)


def _oracle_episode_sampling_config(
    environment: Any,
    *,
    train_range: tuple[int, int],
    seed: int,
) -> OracleEpisodeSamplingConfig:
    return oracle_episode_sampling_config(
        environment,
        train_range=train_range,
        seed=seed,
    )


def _configure_torch_cuda_runtime(
    torch: Any,
    device: object,
    mode: CudaRuntimeMode | str,
) -> dict[str, object]:
    """Apply one explicit CUDA speed/reproducibility contract."""

    resolved_mode = CudaRuntimeMode(mode)
    requested = str(device).strip().lower()
    uses_cuda = requested == "auto" and bool(torch.cuda.is_available())
    if requested != "auto":
        try:
            uses_cuda = torch.device(device).type == "cuda"
        except (RuntimeError, TypeError, ValueError):
            uses_cuda = False

    deterministic = resolved_mode is CudaRuntimeMode.DETERMINISTIC
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if uses_cuda and deterministic:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif uses_cuda:
        # Performance mode intentionally permits nondeterministic kernel selection.
        # Parameters, optimizer state, losses, and checkpoints remain float32.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    bf16_supported = bool(
        uses_cuda
        and callable(getattr(torch.cuda, "is_bf16_supported", None))
        and torch.cuda.is_bf16_supported()
    )
    return {
        "mode": str(resolved_mode),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "sequence_encoder_autocast": ("bfloat16" if bf16_supported else "disabled"),
    }


def _configure_sequence_runtime(
    torch: Any,
    model: Any,
    config: ResidualTrainingConfig,
) -> dict[str, object]:
    # Apply the identity-bound sequence runtime after construction or load.

    compile_enabled = bool(config.sequence_compile)
    compile_target: str | None = None
    if compile_enabled:
        resolved_device = torch.device(model.device)
        if resolved_device.type != "cuda":
            raise RuntimeError("sequence_compile requires a resolved CUDA device")
        extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
        compile_module = getattr(extractor, "compile", None)
        if not callable(compile_module):
            raise RuntimeError(
                "sequence feature extractor does not support in-place compile"
            )
        compile_module(
            mode=config.sequence_compile_mode,
            fullgraph=False,
            dynamic=False,
        )
        compile_target = type(extractor).__name__
    return {
        "compile_enabled": compile_enabled,
        "compile_mode": config.sequence_compile_mode,
        "compile_target": compile_target,
        "fullgraph": False,
        "dynamic": False,
        "inductor_compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        "sequence_transfer_mode": config.sequence_transfer_mode,
        "torch_version": str(torch.__version__),
        "schema_version": "sequence_runtime_v2",
    }
