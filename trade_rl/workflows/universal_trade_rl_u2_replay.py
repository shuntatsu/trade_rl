"""Deterministic synthetic Development replay boundary for Universal Trade RL U2."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from weakref import WeakValueDictionary

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_environment import (
    UniversalTradeEnvironment,
    UniversalTradeMarketEnv,
)
from trade_rl.workflows.universal_trade_rl_u1_contract import (
    UniversalTradeRLU1Contract,
    require_universal_trade_rl_u1_environment_contract,
)
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    U2_TRAINING_SEEDS,
    UniversalTradeRLU2Contract,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation import (
    UniversalTradeRLU2DevelopmentScopeClosure,
    UniversalTradeRLU2EvaluationScope,
    build_universal_trade_rl_u2_development_scope_closure,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation_dataset import (
    U2EvaluationSourceArtifactLoader,
    U2EvaluationSourceArtifactLocator,
    load_universal_trade_rl_u2_development_evaluation_datasets,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    UniversalTradeRLU2TimePartition,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

U2ReplayEnvironmentFactory = Callable[[MarketDataset], UniversalTradeEnvironment]


class UniversalTradeRLU2ReplayVariant(str, Enum):
    """The four preregistered deterministic Development replay variants."""

    CANDIDATE = "candidate"
    CASH = "cash"
    CONSTANT_LONG = "constant_long"
    CONSTANT_SHORT = "constant_short"


_BASELINE_ACTIONS = {
    UniversalTradeRLU2ReplayVariant.CASH: np.asarray([0.0], dtype=np.float32),
    UniversalTradeRLU2ReplayVariant.CONSTANT_LONG: np.asarray([1.0], dtype=np.float32),
    UniversalTradeRLU2ReplayVariant.CONSTANT_SHORT: np.asarray([-1.0], dtype=np.float32),
}


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2ReplayRequest:
    """One paired candidate/baseline replay identity before numeric execution."""

    scope_digest: str
    policy_variant: UniversalTradeRLU2ReplayVariant
    evaluation_seed: int
    paired_candidate_checkpoint_digest: str

    def __post_init__(self) -> None:
        require_sha256(self.scope_digest, field="U2 replay scope digest")
        if not isinstance(self.policy_variant, UniversalTradeRLU2ReplayVariant):
            raise TypeError("U2 replay policy variant is invalid")
        if (
            isinstance(self.evaluation_seed, bool)
            or not isinstance(self.evaluation_seed, int)
            or self.evaluation_seed not in U2_TRAINING_SEEDS
        ):
            raise ValueError("U2 replay evaluation seed must be preregistered")
        require_sha256(
            self.paired_candidate_checkpoint_digest,
            field="U2 replay paired candidate checkpoint digest",
        )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2ReplayEvidence:
    """Content-addressed raw net-economic evidence for one U2 replay scope."""

    scope_closure_digest: str
    scope_digest: str
    universe_manifest_digest: str
    u1_contract_digest: str
    u2_contract_digest: str
    source_dataset_digest: str
    evaluation_dataset_digest: str
    concrete_symbol: str
    symbol_role: str
    cell: str
    source_window: str
    tile_index: int
    policy_variant: str
    evaluation_seed: int
    paired_candidate_checkpoint_digest: str
    runtime_start_bar_index: int
    runtime_end_bar_index: int
    final_current_bar_index: int
    observed_decision_count: int
    normal_completion: bool
    terminated: bool
    truncated: bool
    termination_reason: str | None
    terminal_accounting_mode: str
    terminal_liquidation_cost: float
    initial_capital: float
    final_net_portfolio_value: float
    net_wealth_ratio: float
    net_simple_returns: tuple[float, ...]
    maximum_drawdown: float
    turnover_total: float
    total_execution_cost: float
    funding_pnl: float
    borrow_cost: float
    trade_count: int
    rebalance_count: int
    normalized_action_trace: tuple[float, ...]
    realized_exposure_trace: tuple[float, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        for field_name, value in (
            ("scope_closure_digest", self.scope_closure_digest),
            ("scope_digest", self.scope_digest),
            ("universe_manifest_digest", self.universe_manifest_digest),
            ("u1_contract_digest", self.u1_contract_digest),
            ("u2_contract_digest", self.u2_contract_digest),
            ("source_dataset_digest", self.source_dataset_digest),
            ("evaluation_dataset_digest", self.evaluation_dataset_digest),
            (
                "paired_candidate_checkpoint_digest",
                self.paired_candidate_checkpoint_digest,
            ),
        ):
            require_sha256(value, field=f"U2 replay evidence {field_name}")
        for field_name, value in (
            ("concrete_symbol", self.concrete_symbol),
            ("symbol_role", self.symbol_role),
            ("cell", self.cell),
            ("source_window", self.source_window),
            ("policy_variant", self.policy_variant),
            ("terminal_accounting_mode", self.terminal_accounting_mode),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"U2 replay evidence {field_name} must be non-empty")
        if self.policy_variant not in {
            variant.value for variant in UniversalTradeRLU2ReplayVariant
        }:
            raise ValueError("U2 replay evidence policy variant is invalid")
        if self.termination_reason is not None and (
            not isinstance(self.termination_reason, str) or not self.termination_reason
        ):
            raise ValueError("U2 replay evidence termination reason is invalid")
        for field_name, value in (
            ("tile_index", self.tile_index),
            ("runtime_start_bar_index", self.runtime_start_bar_index),
            ("runtime_end_bar_index", self.runtime_end_bar_index),
            ("final_current_bar_index", self.final_current_bar_index),
            ("observed_decision_count", self.observed_decision_count),
            ("trade_count", self.trade_count),
            ("rebalance_count", self.rebalance_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"U2 replay evidence {field_name} must be a non-negative integer"
                )
        if (
            isinstance(self.evaluation_seed, bool)
            or not isinstance(self.evaluation_seed, int)
            or self.evaluation_seed not in U2_TRAINING_SEEDS
        ):
            raise ValueError("U2 replay evidence evaluation seed must be preregistered")
        for field_name, value in (
            ("terminal_liquidation_cost", self.terminal_liquidation_cost),
            ("initial_capital", self.initial_capital),
            ("final_net_portfolio_value", self.final_net_portfolio_value),
            ("net_wealth_ratio", self.net_wealth_ratio),
            ("maximum_drawdown", self.maximum_drawdown),
            ("turnover_total", self.turnover_total),
            ("total_execution_cost", self.total_execution_cost),
            ("funding_pnl", self.funding_pnl),
            ("borrow_cost", self.borrow_cost),
        ):
            if not math.isfinite(value):
                raise ValueError(f"U2 replay evidence {field_name} must be finite")
        if self.initial_capital <= 0.0:
            raise ValueError("U2 replay evidence initial capital must be positive")
        if self.terminal_liquidation_cost < 0.0:
            raise ValueError("U2 replay terminal liquidation cost cannot be negative")
        if not 0.0 <= self.maximum_drawdown <= 1.0:
            raise ValueError("U2 replay maximum drawdown must be within [0, 1]")
        if self.turnover_total < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("U2 replay turnover and execution cost cannot be negative")
        if self.borrow_cost < 0.0:
            raise ValueError("U2 replay borrow cost cannot be negative")
        if not all(
            isinstance(value, bool)
            for value in (self.normal_completion, self.terminated, self.truncated)
        ):
            raise TypeError("U2 replay completion flags must be booleans")
        if self.terminated and self.truncated:
            raise ValueError("U2 replay cannot be both terminated and truncated")

        for field_name, values in (
            ("net_simple_returns", self.net_simple_returns),
            ("normalized_action_trace", self.normalized_action_trace),
            ("realized_exposure_trace", self.realized_exposure_trace),
        ):
            if not isinstance(values, tuple) or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError(f"U2 replay evidence {field_name} must be finite tuple")
            if len(values) != self.observed_decision_count:
                raise ValueError(
                    f"U2 replay evidence {field_name} length must match decisions"
                )
        if any(abs(value) > 1.0 for value in self.normalized_action_trace):
            raise ValueError("U2 replay normalized action trace is outside [-1, 1]")

        expected_ratio = self.final_net_portfolio_value / self.initial_capital
        if not math.isclose(
            expected_ratio,
            self.net_wealth_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("U2 replay final wealth ratio is inconsistent")
        wealth_from_returns = math.prod(
            1.0 + value for value in self.net_simple_returns
        )
        if not math.isclose(
            wealth_from_returns,
            self.net_wealth_ratio,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("U2 replay simple returns do not reconcile to net wealth")
        if self.normal_completion:
            if self.terminated or not self.truncated:
                raise ValueError("U2 normal replay completion flags are invalid")
            if self.termination_reason is not None:
                raise ValueError("U2 normal replay cannot have termination reason")
            if self.final_current_bar_index != self.runtime_end_bar_index:
                raise ValueError("U2 normal replay did not finish on runtime end")
            if self.terminal_accounting_mode != "mark_to_market":
                raise ValueError("U2 normal replay must use mark-to-market terminal accounting")
            if self.terminal_liquidation_cost != 0.0:
                raise ValueError("U2 normal replay cannot charge terminal liquidation")

        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 replay evidence digest")
            if self.digest != expected_digest:
                raise ValueError("U2 replay evidence digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "scope_closure_digest": self.scope_closure_digest,
            "scope_digest": self.scope_digest,
            "universe_manifest_digest": self.universe_manifest_digest,
            "u1_contract_digest": self.u1_contract_digest,
            "u2_contract_digest": self.u2_contract_digest,
            "source_dataset_digest": self.source_dataset_digest,
            "evaluation_dataset_digest": self.evaluation_dataset_digest,
            "concrete_symbol": self.concrete_symbol,
            "symbol_role": self.symbol_role,
            "cell": self.cell,
            "source_window": self.source_window,
            "tile_index": self.tile_index,
            "policy_variant": self.policy_variant,
            "evaluation_seed": self.evaluation_seed,
            "paired_candidate_checkpoint_digest": (
                self.paired_candidate_checkpoint_digest
            ),
            "runtime_start_bar_index": self.runtime_start_bar_index,
            "runtime_end_bar_index": self.runtime_end_bar_index,
            "final_current_bar_index": self.final_current_bar_index,
            "observed_decision_count": self.observed_decision_count,
            "normal_completion": self.normal_completion,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "terminal_accounting_mode": self.terminal_accounting_mode,
            "terminal_liquidation_cost": self.terminal_liquidation_cost,
            "initial_capital": self.initial_capital,
            "final_net_portfolio_value": self.final_net_portfolio_value,
            "net_wealth_ratio": self.net_wealth_ratio,
            "net_simple_returns": self.net_simple_returns,
            "maximum_drawdown": self.maximum_drawdown,
            "turnover_total": self.turnover_total,
            "total_execution_cost": self.total_execution_cost,
            "funding_pnl": self.funding_pnl,
            "borrow_cost": self.borrow_cost,
            "trade_count": self.trade_count,
            "rebalance_count": self.rebalance_count,
            "normalized_action_trace": self.normalized_action_trace,
            "realized_exposure_trace": self.realized_exposure_trace,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(slots=True)
class UniversalTradeRLU2DevelopmentReplaySession:
    """Own one canonical synthetic Development dataset generation for replay."""

    manifest: UniversalTradeRLUniverseManifest
    time_partition: UniversalTradeRLU2TimePartition
    u2_contract: UniversalTradeRLU2Contract
    u1_contract: UniversalTradeRLU1Contract
    policy_contract: UniversalTradePolicyContract
    normalizer: UniversalTradeSequenceNormalizer
    scope_closure: UniversalTradeRLU2DevelopmentScopeClosure
    datasets: dict[str, MarketDataset]
    environment_factory: U2ReplayEnvironmentFactory
    _issued_environments: WeakValueDictionary[int, UniversalTradeEnvironment] = field(
        default_factory=WeakValueDictionary,
        init=False,
        repr=False,
    )
    _issued_base_environments: WeakValueDictionary[int, UniversalTradeMarketEnv] = (
        field(
            default_factory=WeakValueDictionary,
            init=False,
            repr=False,
        )
    )

    @property
    def scope_closure_digest(self) -> str:
        return self.scope_closure.digest

    @property
    def evaluation_dataset_ids(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (symbol, dataset.dataset_id) for symbol, dataset in self.datasets.items()
        )

    def scope(self, scope_digest: str) -> UniversalTradeRLU2EvaluationScope:
        require_sha256(scope_digest, field="U2 replay scope digest")
        matches = tuple(
            scope for scope in self.scope_closure.scopes if scope.digest == scope_digest
        )
        if len(matches) != 1:
            raise ValueError("U2 replay scope is not present exactly once")
        return matches[0]

    def _require_canonical_scope(
        self,
        scope: UniversalTradeRLU2EvaluationScope,
    ) -> UniversalTradeRLU2EvaluationScope:
        if not isinstance(scope, UniversalTradeRLU2EvaluationScope):
            raise TypeError("U2 replay requires an evaluation scope")
        canonical = self.scope(scope.digest)
        if canonical != scope:
            raise ValueError(
                "U2 replay scope identity drifted from the canonical closure"
            )
        return canonical

    def _create_verified_environment(
        self,
        scope: UniversalTradeRLU2EvaluationScope,
    ) -> UniversalTradeEnvironment:
        canonical = self._require_canonical_scope(scope)
        dataset = self.datasets.get(canonical.concrete_symbol)
        if not isinstance(dataset, MarketDataset):
            raise TypeError("U2 replay scope dataset is unavailable")
        if dataset.symbols != (canonical.concrete_symbol,):
            raise ValueError("U2 replay dataset symbol mismatch")
        if dataset.dataset_id != canonical.evaluation_dataset_digest:
            raise ValueError("U2 replay dataset identity mismatch")

        environment = self.environment_factory(dataset)
        if not isinstance(environment, UniversalTradeEnvironment):
            raise TypeError(
                "U2 replay environment factory must return UniversalTradeEnvironment"
            )

        object_id = id(environment)
        existing = self._issued_environments.get(object_id)
        if existing is environment:
            raise ValueError(
                "U2 replay requires a fresh mutable U1 environment per variant"
            )
        if existing is not None:
            raise RuntimeError(
                "U2 replay observed a live U1 environment object-id collision"
            )

        base_environment = environment.base_env
        base_id = id(base_environment)
        existing_base = self._issued_base_environments.get(base_id)
        if existing_base is base_environment:
            raise ValueError(
                "U2 replay requires a fresh mutable U1 base environment per variant"
            )
        if existing_base is not None:
            raise RuntimeError("U2 replay observed a live U1 base object-id collision")

        try:
            require_universal_trade_rl_u1_environment_contract(
                contract=self.u1_contract,
                environment=environment,
            )
            if environment.contract.digest != self.policy_contract.digest:
                raise ValueError("U2 replay policy contract mismatch")
            child_normalizer = environment.sequence_normalizer
            if not isinstance(child_normalizer, UniversalTradeSequenceNormalizer):
                raise ValueError("U2 replay requires the frozen sequence normalizer")
            if child_normalizer.digest != self.normalizer.digest:
                raise ValueError("U2 replay normalizer generation mismatch")
            if child_normalizer.provenance_digest != self.normalizer.provenance_digest:
                raise ValueError("U2 replay normalizer provenance mismatch")
            if environment.dataset.dataset_id != canonical.evaluation_dataset_digest:
                raise ValueError("U2 replay environment dataset identity mismatch")
            if environment.dataset.symbols != (canonical.concrete_symbol,):
                raise ValueError("U2 replay environment dataset symbol mismatch")
        except Exception:
            environment.close()
            raise

        self._issued_environments[object_id] = environment
        self._issued_base_environments[base_id] = base_environment
        return environment

    def _reset_scope_environment(
        self,
        environment: UniversalTradeEnvironment,
        scope: UniversalTradeRLU2EvaluationScope,
        *,
        evaluation_seed: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        canonical = self._require_canonical_scope(scope)
        issued = self._issued_environments.get(id(environment))
        if issued is not environment:
            raise ValueError("U2 replay environment was not issued by this session")
        if (
            isinstance(evaluation_seed, bool)
            or not isinstance(evaluation_seed, int)
            or evaluation_seed not in self.u2_contract.training_seeds
        ):
            raise ValueError("U2 replay evaluation seed must be preregistered")

        observation, info = environment.reset(
            seed=evaluation_seed,
            options={"start_idx": canonical.evaluation_start_bar_index},
        )
        expected_start = canonical.evaluation_start_bar_index
        expected_end = canonical.outcome_stop_bar_index_exclusive - 1
        if info.get("start_index") != expected_start:
            raise RuntimeError("U2 replay reset start index drifted")
        if info.get("end_index") != expected_end:
            raise RuntimeError("U2 replay runtime end index drifted")
        base = environment.base_env
        if base.start_index != expected_start or base.current_index != expected_start:
            raise RuntimeError("U2 replay environment start state drifted")
        if base.end_index != expected_end:
            raise RuntimeError("U2 replay environment end state drifted")
        return observation, info

    @staticmethod
    def _candidate_action(model: Any, observation: dict[str, Any]) -> np.ndarray:
        predictor = getattr(model, "predict", None)
        if not callable(predictor):
            raise TypeError("U2 candidate replay model must expose predict()")
        raw_action, _state = predictor(observation, deterministic=True)
        action = np.asarray(raw_action, dtype=np.float32)
        if action.shape != (1,):
            raise ValueError("U2 candidate replay action must have shape (1,)")
        return action

    def replay(
        self,
        request: UniversalTradeRLU2ReplayRequest,
        *,
        model: Any | None = None,
    ) -> UniversalTradeRLU2ReplayEvidence:
        """Replay one canonical U2 Development scope through the frozen U1 runtime."""

        if not isinstance(request, UniversalTradeRLU2ReplayRequest):
            raise TypeError("U2 replay request is invalid")
        if request.evaluation_seed not in self.u2_contract.training_seeds:
            raise ValueError("U2 replay evaluation seed is outside the U2 contract")
        candidate = request.policy_variant is UniversalTradeRLU2ReplayVariant.CANDIDATE
        if candidate and model is None:
            raise ValueError("U2 candidate replay requires a model")
        if not candidate and model is not None:
            raise ValueError("U2 baseline replay must not receive a candidate model")

        scope = self.scope(request.scope_digest)
        environment = self._create_verified_environment(scope)
        try:
            observation, _reset_info = self._reset_scope_environment(
                environment,
                scope,
                evaluation_seed=request.evaluation_seed,
            )
            base = environment.base_env
            initial_capital = float(base.hybrid.portfolio_value)
            if base.hybrid.returns_history:
                raise RuntimeError("U2 replay reset produced non-empty return history")

            normalized_actions: list[float] = []
            realized_exposures: list[float] = []
            observed_decision_count = 0
            terminated = False
            truncated = False
            final_info: dict[str, Any] = {}

            while not terminated and not truncated:
                if observed_decision_count >= scope.decision_count:
                    raise RuntimeError("U2 replay exceeded the preregistered decision count")
                if candidate:
                    assert model is not None
                    action = self._candidate_action(model, observation)
                else:
                    action = _BASELINE_ACTIONS[request.policy_variant].copy()
                observation, _reward, terminated, truncated, info = environment.step(action)
                observed_decision_count += 1
                normalized_actions.append(float(action[0]))
                realized_exposures.append(
                    float(base.universal_trade_runtime_snapshot().current_weight)
                )
                final_info = info

            final_current_bar_index = base.current_index
            runtime_start_bar_index = base.start_index
            runtime_end_bar_index = base.end_index
            book = base.hybrid
            net_simple_returns = tuple(float(value) for value in book.returns_history)
            if len(net_simple_returns) != observed_decision_count:
                raise RuntimeError(
                    "U2 replay return history length does not match decisions"
                )

            final_net_portfolio_value = float(book.portfolio_value)
            net_wealth_ratio = final_net_portfolio_value / initial_capital
            wealth_from_returns = math.prod(1.0 + value for value in net_simple_returns)
            if not math.isclose(
                wealth_from_returns,
                net_wealth_ratio,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise RuntimeError("U2 replay simple returns do not reconcile to wealth")

            raw_reason = final_info.get("termination_reason")
            termination_reason = (
                None
                if raw_reason is None
                else str(getattr(raw_reason, "value", raw_reason))
            )
            terminal_accounting_mode = str(
                final_info.get("terminal_accounting_mode", "")
            )
            terminal_liquidation_cost = float(
                final_info.get("terminal_liquidation_cost", float("nan"))
            )
            expected_runtime_end = scope.outcome_stop_bar_index_exclusive - 1
            normal_completion = bool(
                observed_decision_count == scope.decision_count
                and not terminated
                and truncated
                and runtime_start_bar_index == scope.evaluation_start_bar_index
                and runtime_end_bar_index == expected_runtime_end
                and final_current_bar_index == expected_runtime_end
                and termination_reason is None
                and terminal_accounting_mode == "mark_to_market"
                and terminal_liquidation_cost == 0.0
            )
            if truncated and not normal_completion:
                raise RuntimeError("U2 replay time-limit completion drifted from contract")

            return UniversalTradeRLU2ReplayEvidence(
                scope_closure_digest=self.scope_closure.digest,
                scope_digest=scope.digest,
                universe_manifest_digest=scope.universe_manifest_digest,
                u1_contract_digest=scope.u1_contract_digest,
                u2_contract_digest=scope.u2_contract_digest,
                source_dataset_digest=scope.source_dataset_digest,
                evaluation_dataset_digest=scope.evaluation_dataset_digest,
                concrete_symbol=scope.concrete_symbol,
                symbol_role=scope.symbol_role.value,
                cell=scope.cell,
                source_window=scope.source_window,
                tile_index=scope.tile_index,
                policy_variant=request.policy_variant.value,
                evaluation_seed=request.evaluation_seed,
                paired_candidate_checkpoint_digest=(
                    request.paired_candidate_checkpoint_digest
                ),
                runtime_start_bar_index=runtime_start_bar_index,
                runtime_end_bar_index=runtime_end_bar_index,
                final_current_bar_index=final_current_bar_index,
                observed_decision_count=observed_decision_count,
                normal_completion=normal_completion,
                terminated=bool(terminated),
                truncated=bool(truncated),
                termination_reason=termination_reason,
                terminal_accounting_mode=terminal_accounting_mode,
                terminal_liquidation_cost=terminal_liquidation_cost,
                initial_capital=initial_capital,
                final_net_portfolio_value=final_net_portfolio_value,
                net_wealth_ratio=net_wealth_ratio,
                net_simple_returns=net_simple_returns,
                maximum_drawdown=float(book.max_drawdown),
                turnover_total=float(book.turnover_total),
                total_execution_cost=float(book.total_cost),
                funding_pnl=float(book.funding_pnl),
                borrow_cost=float(book.borrow_cost),
                trade_count=int(book.n_trades),
                rebalance_count=int(book.rebalance_events),
                normalized_action_trace=tuple(normalized_actions),
                realized_exposure_trace=tuple(realized_exposures),
            )
        finally:
            environment.close()


def _require_runtime_generation(
    *,
    u2_contract: UniversalTradeRLU2Contract,
    u1_contract: UniversalTradeRLU1Contract,
    policy_contract: UniversalTradePolicyContract,
    normalizer: UniversalTradeSequenceNormalizer,
    environment_factory: U2ReplayEnvironmentFactory,
) -> None:
    if not isinstance(u1_contract, UniversalTradeRLU1Contract):
        raise TypeError("U2 replay requires a U1 contract")
    if not isinstance(policy_contract, UniversalTradePolicyContract):
        raise TypeError("U2 replay requires a policy contract")
    if not isinstance(normalizer, UniversalTradeSequenceNormalizer):
        raise TypeError("U2 replay requires the frozen sequence normalizer")
    if not callable(environment_factory):
        raise TypeError("U2 replay environment factory must be callable")
    if u2_contract.u1_contract_digest != u1_contract.digest:
        raise ValueError("U2 replay U1 contract identity mismatch")
    if u1_contract.policy_contract_digest != policy_contract.digest:
        raise ValueError("U2 replay policy contract identity mismatch")
    if u2_contract.u1_normalizer_digest != normalizer.digest:
        raise ValueError("U2 replay normalizer generation mismatch")
    if normalizer.contract_digest != policy_contract.digest:
        raise ValueError("U2 replay normalizer policy contract mismatch")


def build_universal_trade_rl_u2_development_replay_session(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    time_partition: UniversalTradeRLU2TimePartition,
    u2_contract: UniversalTradeRLU2Contract,
    u1_contract: UniversalTradeRLU1Contract,
    policy_contract: UniversalTradePolicyContract,
    normalizer: UniversalTradeSequenceNormalizer,
    supplied_scope_closure: UniversalTradeRLU2DevelopmentScopeClosure,
    artifact_locators: Mapping[str, U2EvaluationSourceArtifactLocator],
    source_loader: U2EvaluationSourceArtifactLoader,
    environment_factory: U2ReplayEnvironmentFactory,
) -> UniversalTradeRLU2DevelopmentReplaySession:
    """Validate canonical metadata before crossing the Development numeric boundary."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U2 replay requires a U0 manifest")
    if not isinstance(time_partition, UniversalTradeRLU2TimePartition):
        raise TypeError("U2 replay requires a time partition")
    if not isinstance(u2_contract, UniversalTradeRLU2Contract):
        raise TypeError("U2 replay requires a U2 contract")
    if not isinstance(
        supplied_scope_closure,
        UniversalTradeRLU2DevelopmentScopeClosure,
    ):
        raise TypeError("U2 replay requires a Development scope closure")

    canonical_scope_closure = build_universal_trade_rl_u2_development_scope_closure(
        manifest=manifest,
        time_partition=time_partition,
        u2_contract=u2_contract,
    )
    if supplied_scope_closure != canonical_scope_closure:
        raise ValueError("U2 Development replay supplied closure is not canonical")

    _require_runtime_generation(
        u2_contract=u2_contract,
        u1_contract=u1_contract,
        policy_contract=policy_contract,
        normalizer=normalizer,
        environment_factory=environment_factory,
    )
    if not callable(source_loader):
        raise TypeError("U2 replay source loader must be callable")

    datasets = load_universal_trade_rl_u2_development_evaluation_datasets(
        manifest=manifest,
        scope_closure=canonical_scope_closure,
        artifact_locators=artifact_locators,
        loader=source_loader,
    )
    return UniversalTradeRLU2DevelopmentReplaySession(
        manifest=manifest,
        time_partition=time_partition,
        u2_contract=u2_contract,
        u1_contract=u1_contract,
        policy_contract=policy_contract,
        normalizer=normalizer,
        scope_closure=canonical_scope_closure,
        datasets=datasets,
        environment_factory=environment_factory,
    )


__all__ = [
    "U2ReplayEnvironmentFactory",
    "UniversalTradeRLU2DevelopmentReplaySession",
    "UniversalTradeRLU2ReplayEvidence",
    "UniversalTradeRLU2ReplayRequest",
    "UniversalTradeRLU2ReplayVariant",
    "build_universal_trade_rl_u2_development_replay_session",
]
