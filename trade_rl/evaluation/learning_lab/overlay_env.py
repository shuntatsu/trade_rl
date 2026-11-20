"""Gymnasium execution-overlay environment over the shared-portfolio engine.

One *decision* every ``decision_every_bars`` native bars (6 = daily). The base policy (the
anchor after the cash-account scale, built by the caller) proposes a target; the agent
chooses the trade fraction ``f`` of the gap between the current weights and that target.
``f = 1`` executes the base proposal untouched, ``f = 0`` holds the book exactly, and an
intermediate ``f`` moves the *effective* weights ``f`` of the way. Between decisions the
book is held by the base policy's own held path, exactly as a fixed stride would hold it.

Engine and reward oracle: every native bar goes through ``SharedPortfolioSession.advance``
(the incremental replay seam that ``SharedPortfolioGymEnv`` and
``run_shared_portfolio_replay`` also use); terminal intervals are reserved and settled with
``settle_next_interval``. The reward of a decision is the engine's net equity increment
(``net_equity_increment``, gamma 1) over the bars the decision covers. There is no second PnL
calculator here.

Causality: the observation at a decision bar reads only dataset rows ``<= index`` and the
engine book; the base target is peeked on a deep copy of the base policy stack (the large
shared arrays are not copied) so the real stack advances only through real decisions.

Domain randomization and an adversary are the same hook: a ``PerturbationSource`` called at
``reset`` returns an :class:`OverlayPerturbation` (cost multiplier, extra order latency,
return shock). The engine fixes its cost configuration per session, so a perturbation holds
for an episode.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Final, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.objectives.contract import net_equity_increment
from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_observation import (
    build_portfolio_decision_snapshot,
)

ACTION_DISCRETE: Final = "discrete"
ACTION_CONTINUOUS: Final = "continuous"
DISCRETE_FRACTIONS: Final[tuple[float, ...]] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)

OBSERVATION_FIELDS: Final[tuple[str, ...]] = (
    "gap_l1",
    "conviction_mean_abs_target",
    "portfolio_vol_60d",
    "cross_section_dispersion_3d",
    "log_quote_volume_vs_60d",
    "mean_funding",
    "drawdown",
    "days_since_last_trade",
)
OBSERVATION_CLIP: Final = 10.0
_BARS_PER_DAY_DEFAULT = 6
_VOL_WINDOW_DAYS = 60
_DISPERSION_DAYS = 3
_BAR_RETURN_CLIP = 0.5
_FULL = 1.0 - 1e-12
_NONE = 1e-12


class OverlayError(ValueError):
    """The overlay environment was configured or used inconsistently."""


class RebalanceGate:
    """The flag the caller flips to say whether the next base-policy call may rebalance."""

    __slots__ = ("rebalance_now",)

    def __init__(self) -> None:
        self.rebalance_now = True


@dataclass(frozen=True, slots=True)
class BaseStack:
    """A base-policy stack whose rebalance bars are chosen by the overlay.

    ``policy`` must consult ``gate.rebalance_now`` on every call (rebalance bar iff true).
    ``shared`` lists large read-only objects (signals) that a peek must not deep-copy.
    """

    policy: PortfolioPolicy
    gate: RebalanceGate
    shared: tuple[object, ...] = ()


class BaseFactory(Protocol):
    def __call__(self) -> BaseStack: ...


@dataclass(frozen=True, slots=True)
class OverlayPerturbation:
    """Bounded per-episode stress: cost multiplier, extra latency, return shock."""

    cost_multiplier: float = 1.0
    extra_latency_bars: int = 0
    return_shock_sigma: float = 0.0
    execution_seed: int | None = None
    shock_seed: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.cost_multiplier) or not (
            0.0 <= self.cost_multiplier <= MAX_COST_MULTIPLIER
        ):
            raise OverlayError(
                f"cost_multiplier must lie within [0, {MAX_COST_MULTIPLIER}]"
            )
        if (
            type(self.extra_latency_bars) is not int
            or not 0 <= self.extra_latency_bars <= MAX_EXTRA_LATENCY_BARS
        ):
            raise OverlayError(
                f"extra_latency_bars must be an int within [0, {MAX_EXTRA_LATENCY_BARS}]"
            )
        if not math.isfinite(self.return_shock_sigma) or not (
            0.0 <= self.return_shock_sigma <= MAX_RETURN_SHOCK_SIGMA
        ):
            raise OverlayError(
                f"return_shock_sigma must lie within [0, {MAX_RETURN_SHOCK_SIGMA}]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_multiplier": self.cost_multiplier,
            "extra_latency_bars": self.extra_latency_bars,
            "return_shock_sigma": self.return_shock_sigma,
            "execution_seed": self.execution_seed,
            "shock_seed": self.shock_seed,
        }


#: The adversary's bounds: nothing outside them reaches the engine.
MAX_COST_MULTIPLIER: Final = 10.0
MAX_EXTRA_LATENCY_BARS: Final = 3
MAX_RETURN_SHOCK_SIGMA: Final = 0.05

#: ``rng -> perturbation``: domain randomization or an adversary policy (RARL hook).
PerturbationSource = Callable[[np.random.Generator], OverlayPerturbation]


@dataclass(frozen=True, slots=True)
class UniformRandomization:
    """Domain randomization: every episode draws each stress uniformly from its range."""

    cost_multiplier: tuple[float, float] = (1.0, 1.0)
    extra_latency_bars: tuple[int, int] = (0, 0)
    return_shock_sigma: tuple[float, float] = (0.0, 0.0)

    def __call__(self, rng: np.random.Generator) -> OverlayPerturbation:
        return OverlayPerturbation(
            cost_multiplier=float(rng.uniform(*self.cost_multiplier)),
            extra_latency_bars=int(
                rng.integers(self.extra_latency_bars[0], self.extra_latency_bars[1] + 1)
            ),
            return_shock_sigma=float(rng.uniform(*self.return_shock_sigma)),
            execution_seed=int(rng.integers(0, 2**31 - 1)),
            shock_seed=int(rng.integers(0, 2**31 - 1)),
        )


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    """Environment controls that are not market data (all part of the config digest)."""

    action_mode: str = ACTION_DISCRETE
    decision_every_bars: int = _BARS_PER_DAY_DEFAULT
    initial_capital: float = 100_000.0
    fractions: tuple[float, ...] = DISCRETE_FRACTIONS

    def __post_init__(self) -> None:
        if self.action_mode not in (ACTION_DISCRETE, ACTION_CONTINUOUS):
            raise OverlayError(f"unknown action_mode: {self.action_mode!r}")
        if type(self.decision_every_bars) is not int or self.decision_every_bars < 1:
            raise OverlayError("decision_every_bars must be a positive int")
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0.0:
            raise OverlayError("initial_capital must be finite and positive")
        if (
            not self.fractions
            or any(not 0.0 <= value <= 1.0 for value in self.fractions)
            or list(self.fractions) != sorted(set(self.fractions))
            or self.fractions[0] != 0.0
            or self.fractions[-1] != 1.0
        ):
            raise OverlayError("fractions must be increasing, within [0, 1], 0 .. 1")

    def payload(self) -> dict[str, Any]:
        return {
            "action_mode": self.action_mode,
            "decision_every_bars": self.decision_every_bars,
            "initial_capital": self.initial_capital,
            "fractions": list(self.fractions),
            "observation_fields": list(OBSERVATION_FIELDS),
        }


def risk_scale(snapshot: PortfolioDecisionSnapshot) -> float:
    """``PreTradeRisk``'s linear drawdown ramp, from the snapshot alone."""
    peak = max(float(snapshot.account.peak_value), 1e-12)
    current = max(0.0, 1.0 - float(snapshot.account.portfolio_value) / peak)
    drawdown = min(1.0, max(current, float(snapshot.account.max_drawdown)))
    start = float(snapshot.risk.drawdown_start)
    stop = float(snapshot.risk.drawdown_stop)
    if drawdown <= start:
        return 1.0
    if drawdown >= stop or stop == start:
        return 0.0
    return 1.0 - (drawdown - start) / (stop - start)


def effective_weights(
    proposal: PortfolioTargetProposal, snapshot: PortfolioDecisionSnapshot
) -> np.ndarray:
    """The weights the engine will aim at: held legs stay, every other leg is scaled once."""
    current = np.asarray(snapshot.account.weights, dtype=np.float64)
    target = np.asarray(proposal.target_weights, dtype=np.float64)
    held = np.zeros(target.size, dtype=bool)
    if proposal.hold_current_quantity_symbols:
        held[list(proposal.hold_current_quantity_symbols)] = True
    return np.where(held, current, risk_scale(snapshot) * target)


def blend_proposal(
    proposal: PortfolioTargetProposal,
    snapshot: PortfolioDecisionSnapshot,
    fraction: float,
) -> PortfolioTargetProposal:
    """Move the effective weights ``fraction`` of the way from current to the proposal's."""
    if fraction >= _FULL:
        return proposal
    scale = risk_scale(snapshot)
    if scale <= 0.0:
        return proposal
    current = np.asarray(snapshot.account.weights, dtype=np.float64)
    held = np.zeros(current.size, dtype=bool)
    if proposal.hold_current_quantity_symbols:
        held[list(proposal.hold_current_quantity_symbols)] = True
    desired = current + fraction * (effective_weights(proposal, snapshot) - current)
    weights = np.where(held, current, desired / scale)
    return PortfolioTargetProposal(
        target_weights=tuple(float(w) for w in weights),
        hold_current_quantity_symbols=proposal.hold_current_quantity_symbols,
    )


def hold_all(snapshot: PortfolioDecisionSnapshot) -> PortfolioTargetProposal:
    n = len(snapshot.account.weights)
    return PortfolioTargetProposal(
        target_weights=tuple(float(w) for w in snapshot.account.weights),
        hold_current_quantity_symbols=tuple(range(n)),
    )


def shock_returns(
    dataset: MarketDataset,
    *,
    sigma: float,
    seed: int,
    from_index: int,
) -> MarketDataset:
    """Scale every price row after ``from_index`` by a shared random-walk factor.

    Rows ``<= from_index`` are unchanged; each later bar's return gets an independent
    ``N(0, sigma)`` log-shock per symbol, applied consistently to open, high, low, close and
    the mark and index prices, so bar shapes stay valid. ``sigma == 0`` returns the dataset.
    """
    if sigma <= 0.0:
        return dataset
    rng = np.random.default_rng(seed)
    shape = (dataset.n_bars, dataset.n_symbols)
    log_shock = rng.normal(0.0, sigma, size=shape)
    log_shock[: from_index + 1] = 0.0
    factor = np.exp(np.cumsum(log_shock, axis=0))
    updates: dict[str, Any] = {}
    for name in ("open", "high", "low", "close", "mark_price", "index_price"):
        array = getattr(dataset, name)
        if array is not None:
            updates[name] = np.asarray(array, dtype=np.float64) * factor
    return replace(dataset, **updates)


def _scaled_cost(
    base: ExecutionCostConfig, perturbation: OverlayPerturbation
) -> ExecutionCostConfig:
    mult = perturbation.cost_multiplier
    return replace(
        base,
        fee_rate=base.fee_rate * mult,
        maker_fee_rate=base.maker_fee_rate * mult,
        taker_fee_rate=base.taker_fee_rate * mult,
        spread_rate=base.spread_rate * mult,
        impact_rate=base.impact_rate * mult,
        order_latency_bars=base.order_latency_bars + perturbation.extra_latency_bars,
        random_seed=(
            base.random_seed
            if perturbation.execution_seed is None
            else perturbation.execution_seed
        ),
    )


@dataclass(slots=True)
class _Episode:
    session: SharedPortfolioSession
    stack: BaseStack
    dataset: MarketDataset
    snapshot: PortfolioDecisionSnapshot
    perturbation: OverlayPerturbation
    decisions: int = 0
    days_since_trade: float = 0.0
    bar_returns: list[float] = field(default_factory=list)
    done: bool = False


class ExecutionOverlayEnv(gym.Env):
    """Trade-fraction overlay on a base target (see the module docstring)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        start_index: int,
        stop_index: int,
        base_factory: BaseFactory,
        execution_cost: ExecutionCostConfig,
        risk_config: PreTradeRiskConfig,
        config: OverlayConfig | None = None,
        perturbation: PerturbationSource | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(dataset, MarketDataset):
            raise TypeError("dataset must be a MarketDataset")
        if not isinstance(execution_cost, ExecutionCostConfig):
            raise TypeError("execution_cost must be an ExecutionCostConfig")
        if not isinstance(risk_config, PreTradeRiskConfig):
            raise TypeError("risk_config must be a PreTradeRiskConfig")
        if not callable(base_factory):
            raise TypeError("base_factory must be callable")
        if perturbation is not None and not callable(perturbation):
            raise TypeError("perturbation must be callable")
        if not 0 <= start_index < stop_index < dataset.n_bars:
            raise OverlayError("range must satisfy 0 <= start < stop < n_bars")
        self.config = OverlayConfig() if config is None else config
        self.dataset = dataset
        self.start_index = int(start_index)
        self.stop_index = int(stop_index)
        self.base_factory = base_factory
        self.execution_cost = execution_cost
        self.risk_config = risk_config
        self.perturbation_source = perturbation
        if self.config.action_mode == ACTION_DISCRETE:
            self.action_space: spaces.Space[Any] = spaces.Discrete(
                len(self.config.fractions)
            )
        else:
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            )
        self.observation_space = spaces.Box(
            low=-OBSERVATION_CLIP,
            high=OBSERVATION_CLIP,
            shape=(len(OBSERVATION_FIELDS),),
            dtype=np.float32,
        )
        self._episode: _Episode | None = None
        self._bars_per_day = self.config.decision_every_bars

    # ------------------------------------------------------------------ gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if options:
            raise OverlayError("reset options are not supported")
        super().reset(seed=seed)
        perturbation = (
            OverlayPerturbation()
            if self.perturbation_source is None
            else self.perturbation_source(self.np_random)
        )
        if not isinstance(perturbation, OverlayPerturbation):
            raise TypeError("a perturbation source must return an OverlayPerturbation")
        dataset = shock_returns(
            self.dataset,
            sigma=perturbation.return_shock_sigma,
            seed=perturbation.shock_seed,
            from_index=self.start_index,
        )
        session = SharedPortfolioSession(
            dataset,
            start_index=self.start_index,
            stop_index=self.stop_index,
            initial_capital=self.config.initial_capital,
            execution_cost=_scaled_cost(self.execution_cost, perturbation),
            risk=PreTradeRisk(self.risk_config),
            settle_terminal_position=True,
        )
        stack = self.base_factory()
        episode = _Episode(
            session=session,
            stack=stack,
            dataset=dataset,
            snapshot=self._snapshot(session, dataset, self.start_index),
            perturbation=perturbation,
        )
        self._episode = episode
        observation = self._observation(episode)
        return observation, {
            "current_index": session.current_index,
            "perturbation": perturbation.to_dict(),
        }

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        episode = self._episode
        if episode is None or episode.done:
            raise OverlayError("call reset() before step()")
        session = episode.session
        fraction = self._decode(action)
        equity_before = float(session.book.portfolio_value)
        turnover_before = float(session.book.turnover_total)
        returns: list[float] = []
        try:
            for bar in range(self.config.decision_every_bars):
                if not session.can_decide:
                    break
                snapshot = (
                    episode.snapshot
                    if bar == 0
                    else self._snapshot(session, episode.dataset, session.current_index)
                )
                proposal = self._proposal(
                    episode, snapshot, fraction if bar == 0 else None
                )
                _, execution, _ = session.advance(
                    proposal.target_weights,
                    hold_current_quantity_symbols=proposal.hold_current_quantity_symbols,
                )
                returns.append(float(execution.interval_net_return))
            while session.can_settle_terminal:
                _, execution, _ = session.settle_next_interval()
                returns.append(float(execution.interval_net_return))
        except BaseException:
            episode.done = True
            raise
        episode.decisions += 1
        book = session.book
        reward = net_equity_increment(
            equity_before,
            float(book.portfolio_value),
            initial_capital=self.config.initial_capital,
        )
        traded = float(book.turnover_total) - turnover_before > 1e-12
        episode.days_since_trade = (
            0.0
            if traded
            else episode.days_since_trade
            + self.config.decision_every_bars / self._bars_per_day
        )
        episode.bar_returns.extend(returns)
        terminated = session.finished
        info: dict[str, Any] = {
            "current_index": session.current_index,
            "trade_fraction": fraction,
            "equity": float(book.portfolio_value),
            "equity_delta": float(book.portfolio_value) - equity_before,
            "bar_returns": tuple(returns),
            "total_cost": float(book.total_cost),
            "turnover_total": float(book.turnover_total),
        }
        if terminated:
            episode.done = True
            observation = np.zeros(len(OBSERVATION_FIELDS), dtype=np.float32)
        else:
            episode.snapshot = self._snapshot(
                session, episode.dataset, session.current_index
            )
            observation = self._observation(episode)
        return observation, reward, terminated, False, info

    # --------------------------------------------------------------- internals

    def _decode(self, action: Any) -> float:
        if self.config.action_mode == ACTION_DISCRETE:
            if isinstance(action, np.ndarray):
                action = action.item()
            if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
                raise OverlayError("a discrete action must be an integer index")
            if not 0 <= int(action) < len(self.config.fractions):
                raise OverlayError("discrete action index out of range")
            return float(self.config.fractions[int(action)])
        value = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        if not math.isfinite(value):
            raise OverlayError("a continuous action must be finite")
        return min(1.0, max(0.0, value))

    @staticmethod
    def _snapshot(
        session: SharedPortfolioSession, dataset: MarketDataset, index: int
    ) -> PortfolioDecisionSnapshot:
        return build_portfolio_decision_snapshot(
            dataset,
            index=index,
            agent_stop_index=session.agent_stop_index,
            book=session.book,
            active_orders=session.active_orders,
            risk_config=session.risk_controller.config,
        )

    @staticmethod
    def _validated(
        proposal: PortfolioTargetProposal,
        snapshot: PortfolioDecisionSnapshot,
    ) -> PortfolioTargetProposal:
        if not isinstance(proposal, PortfolioTargetProposal):
            raise TypeError("the base policy must return a PortfolioTargetProposal")
        n = len(snapshot.account.weights)
        if len(proposal.target_weights) != n:
            raise OverlayError("proposal vector must match the dataset symbols")
        for symbol in proposal.hold_current_quantity_symbols:
            if symbol >= n:
                raise OverlayError("HOLD symbol outside the roster")
            if proposal.target_weights[symbol] != snapshot.account.weights[symbol]:
                raise OverlayError("HOLD target must equal the current book weight")
        return proposal

    def _proposal(
        self,
        episode: _Episode,
        snapshot: PortfolioDecisionSnapshot,
        fraction: float | None,
    ) -> PortfolioTargetProposal:
        gate = episode.stack.gate
        policy = episode.stack.policy
        if fraction is None or fraction <= _NONE:
            gate.rebalance_now = False
            proposal = self._validated(policy.decide_portfolio(snapshot), snapshot)
            if fraction is not None and episode.decisions == 0:
                # the very first call has no intended book yet; nothing is executed
                proposal = hold_all(snapshot)
            return proposal
        gate.rebalance_now = True
        proposal = self._validated(policy.decide_portfolio(snapshot), snapshot)
        return self._validated(blend_proposal(proposal, snapshot, fraction), snapshot)

    def _peek(
        self, episode: _Episode, snapshot: PortfolioDecisionSnapshot
    ) -> np.ndarray:
        """Effective base target at this bar on a copy of the stack (no state advance)."""
        memo = {id(obj): obj for obj in episode.stack.shared}
        clone = copy.deepcopy(episode.stack, memo)
        clone.gate.rebalance_now = True
        proposal = self._validated(clone.policy.decide_portfolio(snapshot), snapshot)
        return effective_weights(proposal, snapshot)

    def _observation(self, episode: _Episode) -> np.ndarray:
        snapshot = episode.snapshot
        dataset = episode.dataset
        index = snapshot.index
        current = np.asarray(snapshot.account.weights, dtype=np.float64)
        target = self._peek(episode, snapshot)
        close = np.asarray(dataset.close, dtype=np.float64)
        per_day = self._bars_per_day

        gap = float(np.abs(target - current).sum())
        conviction = float(np.abs(target).mean())

        window = per_day * _VOL_WINDOW_DAYS
        lo = max(0, index - window)
        if index - lo >= 3:
            rets = close[lo + 1 : index + 1] / close[lo:index] - 1.0
            rets = np.clip(np.nan_to_num(rets), -_BAR_RETURN_CLIP, _BAR_RETURN_CLIP)
            vol = float((rets @ target).std() * math.sqrt(per_day))
        else:
            vol = 0.0

        lag = per_day * _DISPERSION_DAYS
        if index - lag >= 0:
            past = close[index - lag]
            ok = past > 0.0
            r3 = np.clip(close[index][ok] / past[ok] - 1.0, -1.0, 1.0)
            dispersion = float(r3.std()) if r3.size > 1 else 0.0
        else:
            dispersion = 0.0

        volume = np.asarray(dataset.volume, dtype=np.float64)
        recent = np.log1p(volume[max(0, index - per_day + 1) : index + 1]).mean()
        history = np.log1p(volume[lo : index + 1]).mean()
        log_volume = float(recent - history)

        funding = np.asarray(dataset.funding_rate, dtype=np.float64)
        mean_funding = float(
            funding[max(0, index - per_day + 1) : index + 1].mean() * per_day
        )

        peak = max(float(snapshot.account.peak_value), 1e-12)
        drawdown = max(0.0, 1.0 - float(snapshot.account.portfolio_value) / peak)
        values = np.array(
            [
                gap,
                conviction,
                vol,
                dispersion,
                log_volume,
                mean_funding,
                drawdown,
                episode.days_since_trade,
            ],
            dtype=np.float64,
        )
        values = np.nan_to_num(values, nan=0.0, posinf=OBSERVATION_CLIP, neginf=0.0)
        return np.clip(values, -OBSERVATION_CLIP, OBSERVATION_CLIP).astype(np.float32)

    # ---------------------------------------------------------------- read-only

    @property
    def episode_bar_returns(self) -> tuple[float, ...]:
        """Per-native-bar engine net returns of the current/last episode."""
        return () if self._episode is None else tuple(self._episode.bar_returns)

    @property
    def final_equity(self) -> float:
        if self._episode is None:
            raise OverlayError("no episode")
        return float(self._episode.session.book.portfolio_value)

    @property
    def session(self) -> SharedPortfolioSession:
        if self._episode is None:
            raise OverlayError("no episode")
        return self._episode.session


__all__ = [
    "ACTION_CONTINUOUS",
    "ACTION_DISCRETE",
    "DISCRETE_FRACTIONS",
    "MAX_COST_MULTIPLIER",
    "MAX_EXTRA_LATENCY_BARS",
    "MAX_RETURN_SHOCK_SIGMA",
    "OBSERVATION_FIELDS",
    "BaseFactory",
    "BaseStack",
    "ExecutionOverlayEnv",
    "OverlayConfig",
    "OverlayError",
    "OverlayPerturbation",
    "PerturbationSource",
    "RebalanceGate",
    "UniformRandomization",
    "blend_proposal",
    "effective_weights",
    "hold_all",
    "risk_scale",
    "shock_returns",
]
