"""Same-segment economic attribution across Universal policy stages."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.binance_universal_runtime import build_runtime
from trade_rl.learning.rollout_evaluation import evaluate_action_path
from trade_rl.rl.policies import SharedPerAssetActorCriticPolicy
from trade_rl.rl.training_run_config import TrainingRunConfig
from trade_rl.workflows.universal_full_research_entrypoint import (
    UniversalRuntimeFactoryContext,
)
from trade_rl.workflows.universal_research import FullResearchAlgorithm
from trade_rl.workflows.universal_teacher_runtime import (
    build_universal_symbol_teacher_environment,
)


@dataclass(frozen=True, slots=True)
class UniversalPolicyArtifact:
    """One immutable policy state used by the causal stage audit."""

    label: str
    path: Path
    policy_only: bool
    training_step: int | None
    file_digest: str


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover_universal_policy_artifacts(
    member_root: Path,
) -> tuple[UniversalPolicyArtifact, ...]:
    """Discover the pretraining snapshots and ordered rollout checkpoints."""

    root = Path(member_root)
    artifacts: list[UniversalPolicyArtifact] = []
    for stage in ("random", "behavior_cloning", "behavior_cloning_critic"):
        path = root / "policy-stages" / stage / "policy.zip"
        if not path.is_file():
            raise FileNotFoundError(f"missing policy stage: {path}")
        artifacts.append(
            UniversalPolicyArtifact(
                label=stage,
                path=path,
                policy_only=True,
                training_step=None,
                file_digest=_file_digest(path),
            )
        )
    checkpoint_root = root / "checkpoints"
    checkpoint_paths = sorted(checkpoint_root.glob("step-*/policy.zip"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"no rollout checkpoints found under {checkpoint_root}")
    for path in checkpoint_paths:
        raw_step = path.parent.name.removeprefix("step-")
        try:
            step = int(raw_step)
        except ValueError as error:
            raise ValueError(f"invalid checkpoint step directory: {path.parent}") from error
        if step <= 0:
            raise ValueError(f"checkpoint step must be positive: {path.parent}")
        artifacts.append(
            UniversalPolicyArtifact(
                label=f"rollout_{step}",
                path=path,
                policy_only=False,
                training_step=step,
                file_digest=_file_digest(path),
            )
        )
    return tuple(artifacts)


def _load_policy(
    artifact: UniversalPolicyArtifact,
    *,
    algorithm_identifier: str,
    device: str,
) -> Any:
    if artifact.policy_only:
        return SharedPerAssetActorCriticPolicy.load(str(artifact.path), device=device)
    if algorithm_identifier == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(artifact.path), device=device)
    if algorithm_identifier == "lagrangian_ppo":
        from trade_rl.integrations.lagrangian_ppo import LagrangianPPO

        return LagrangianPPO.load(str(artifact.path), device=device)
    raise ValueError(f"unsupported checkpoint algorithm: {algorithm_identifier}")


def _action_diagnostics(actions: np.ndarray) -> dict[str, object]:
    resolved = np.asarray(actions, dtype=np.float64)
    if resolved.ndim != 2 or resolved.shape[1] != 1 or not np.isfinite(resolved).all():
        raise ValueError("Universal policy actions must be a finite (steps, 1) matrix")
    previous = np.vstack((np.zeros((1, 1), dtype=np.float64), resolved[:-1]))
    delta = np.abs(resolved - previous)
    sign_flips = int(np.count_nonzero(resolved[1:] * resolved[:-1] < 0.0))
    return {
        "absolute_target_delta_total": float(delta.sum()),
        "absolute_target_delta_mean": float(delta.mean()),
        "action_mean": float(resolved.mean()),
        "action_std": float(resolved.std()),
        "buy_sell_sign_flip_count": sign_flips,
    }


def _evaluate_symbol(
    *,
    model: Any,
    environment: Any,
    symbol: str,
    start: int,
    decisions: int,
) -> dict[str, object]:
    stop = start + decisions + 1
    evaluated = evaluate_action_path(
        environment,
        evaluation_range=(start, stop),
        model=model,
    )
    initial_capital = float(environment.initial_capital)
    policy_final_equity = float(environment.hybrid.portfolio_value)
    baseline_final_equity = float(environment.shadow.portfolio_value)
    net_return = policy_final_equity / initial_capital - 1.0
    baseline_return = baseline_final_equity / initial_capital - 1.0
    days = decisions * float(environment.decision_hours) / 24.0
    performance = asdict(evaluated.performance)
    return {
        "symbol": symbol,
        "evaluation_range": [start, stop],
        "initial_capital": initial_capital,
        "policy_final_equity": policy_final_equity,
        "baseline_final_equity": baseline_final_equity,
        "baseline_return": baseline_return,
        "baseline_excess_return": net_return - baseline_return,
        "gross_pnl": initial_capital * float(evaluated.performance.gross_return),
        "net_pnl": policy_final_equity - initial_capital,
        "turnover_multiple_per_day": (
            float(evaluated.performance.turnover_total) / days
        ),
        "performance": performance,
        "collapse_evidence": asdict(evaluated.collapse_evidence),
        "actions": _action_diagnostics(evaluated.actions),
    }


def _aggregate_stage(results: Sequence[dict[str, object]]) -> dict[str, object]:
    if not results:
        raise ValueError("stage evaluation requires at least one symbol")

    def mean(name: str) -> float:
        return float(
            np.mean([float(cast(int | float, result[name])) for result in results])
        )

    performance_values = [result["performance"] for result in results]
    action_values = [result["actions"] for result in results]
    if not all(isinstance(value, dict) for value in performance_values):
        raise TypeError("stage performance payload is invalid")
    if not all(isinstance(value, dict) for value in action_values):
        raise TypeError("stage action payload is invalid")
    performance = [value for value in performance_values if isinstance(value, dict)]
    actions = [value for value in action_values if isinstance(value, dict)]
    return {
        "symbol_count": len(results),
        "mean_gross_return": float(
            np.mean([float(value["gross_return"]) for value in performance])
        ),
        "mean_net_return": float(
            np.mean([float(value["net_return"]) for value in performance])
        ),
        "mean_baseline_return": mean("baseline_return"),
        "mean_baseline_excess_return": mean("baseline_excess_return"),
        "mean_turnover_multiple_per_day": mean("turnover_multiple_per_day"),
        "total_execution_cost": float(
            sum(float(value["cost_total"]) for value in performance)
        ),
        "total_closed_trade_count": int(
            sum(int(value["trade_count"]) for value in performance)
        ),
        "total_traded_step_count": int(
            sum(int(value["traded_step_count"]) for value in performance)
        ),
        "total_absolute_target_delta": float(
            sum(float(value["absolute_target_delta_total"]) for value in actions)
        ),
        "total_buy_sell_sign_flip_count": int(
            sum(int(value["buy_sell_sign_flip_count"]) for value in actions)
        ),
    }


def evaluate_universal_policy_stages(
    *,
    config_path: Path,
    runtime_manifest_path: Path,
    frozen_metadata_root: Path,
    member_root: Path,
    algorithm: FullResearchAlgorithm | str,
    member_seed: int,
    decisions: int,
    output_path: Path,
    symbols: Sequence[str] | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Evaluate every saved policy stage on identical real-data ranges."""

    if isinstance(member_seed, bool) or not isinstance(member_seed, int) or member_seed < 0:
        raise ValueError("member_seed must be a non-negative integer")
    if isinstance(decisions, bool) or not isinstance(decisions, int) or decisions <= 1:
        raise ValueError("decisions must be an integer greater than one")
    resolved_algorithm = FullResearchAlgorithm(algorithm)
    config = TrainingRunConfig.from_json(Path(config_path))
    context = UniversalRuntimeFactoryContext(
        runtime_manifest_path=Path(runtime_manifest_path),
        frozen_metadata_root=Path(frozen_metadata_root),
    )
    runtime = build_runtime(
        algorithm=resolved_algorithm,
        run_config=config,
        context=context,
    )
    routed = runtime.routed_environment_factory
    selected_symbols = tuple(routed.train_symbols if symbols is None else symbols)
    if not selected_symbols or not set(selected_symbols).issubset(routed.train_symbols):
        raise ValueError("symbols must be a non-empty subset of train_symbols")
    artifacts = discover_universal_policy_artifacts(Path(member_root))
    binding_by_symbol = {
        binding.concrete_symbol: binding for binding in routed.bindings
    }
    context_provider = routed.instrument_context_provider
    if context_provider is None:
        raise RuntimeError("Universal stage evaluation requires instrument context")
    stage_results: list[dict[str, object]] = []
    for artifact in artifacts:
        model = _load_policy(
            artifact,
            algorithm_identifier=config.training.algorithm,
            device=device,
        )
        symbol_results: list[dict[str, object]] = []
        for symbol in selected_symbols:
            environment = build_universal_symbol_teacher_environment(
                symbol=symbol,
                binding=binding_by_symbol[symbol],
                concrete_environment_factory=routed.concrete_environment_factory,
                instrument_context_provider=context_provider,
                partition_digest=routed.partition_digest,
                training_contract_digest=routed.training_contract_digest,
                run_seed=member_seed,
            )
            start = max(
                int(environment.minimum_start_index),
                int(context.manifest.fold_train_range[0]),
            )
            if start + decisions + 1 > int(context.manifest.fold_train_range[1]):
                raise ValueError("evaluation range exceeds the training fold")
            try:
                symbol_results.append(
                    _evaluate_symbol(
                        model=model,
                        environment=environment,
                        symbol=symbol,
                        start=start,
                        decisions=decisions,
                    )
                )
            finally:
                environment.close()
        stage_results.append(
            {
                "label": artifact.label,
                "policy_file": str(artifact.path),
                "policy_file_digest": artifact.file_digest,
                "training_step": artifact.training_step,
                "aggregate": _aggregate_stage(symbol_results),
                "symbols": symbol_results,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "universal_policy_stage_evaluation_v1",
        "algorithm": resolved_algorithm.value,
        "algorithm_identifier": config.training.algorithm,
        "member_seed": member_seed,
        "decision_count_per_symbol": decisions,
        "symbols": list(selected_symbols),
        "runtime_manifest_digest": context.manifest.manifest_digest,
        "training_config_digest": content_digest(config.training.digest_payload()),
        "reward_interpretation": "pure_net_log_growth_no_additional_cost_penalty",
        "stages": stage_results,
    }
    resolved = {**payload, "artifact_digest": content_digest(payload)}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, canonical_json_bytes(resolved) + b"\n")
    return resolved


__all__ = [
    "UniversalPolicyArtifact",
    "discover_universal_policy_artifacts",
    "evaluate_universal_policy_stages",
]
