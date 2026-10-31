"""Frozen result-blind protocol of the synthetic residual-PPO study (Issue #810).

Every constant that can influence the outcome is fixed here, before any PPO run,
and bound by ``FROZEN_PROTOCOL_SHA256``. The runner refuses to execute when the
computed digest differs, so a constant cannot be changed after the freeze without
visibly changing the committed digest as well.

G0 research question: on synthetic multi-asset markets with a known planted
after-cost alpha and explicit fees and spread, can a PPO agent that controls only a
low-dimensional trade-throttle residual on top of the cost-aware allocator beat the
allocator alone, net of fees, on held-out synthetic seeds?

Falsifiers (any one gives ``SYNTHETIC_FAIL``):

* F1 primary: the held-out median net log-return advantage over the allocator
  alone (kappa = 1) is below ``margin``, is not positive for every PPO seed, or its
  paired-bootstrap lower bound is not above zero.
* F2 learnability control: on the zero-alpha fee-only market the agent trained
  there does not cut turnover to at most ``control_turnover_ratio`` of the
  allocator's, or does not lose less than the allocator alone.

Residual action, chosen before any run: ``Discrete(4)`` over the throttle
multiplier set ``{0, 0.5, 1, 1.5}`` applied to the allocator's trade
(``w_exec = w_cur + m (w_target - w_cur)``, projected into the allocator's own
box). Rationale: ``m = 1`` is exactly the allocator (bit-for-bit parity oracle),
``m = 0`` is exactly no trade (so the fee-only control has an exactly attainable
optimum), the projection removes any leverage advantage, and a categorical head has
no squashing or clipping ambiguity.

A SYNTHETIC_PASS is synthetic software/learnability evidence only. It is never a
real-data profitability claim, and real-data RL stays blocked behind the fee-only
learnability check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trade_rl.artifacts.canonical import to_json_value
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.residual_rl.env import (
    AllocatorSettings,
    CostSettings,
    ObservationScales,
)
from trade_rl.evaluation.residual_rl.market import MarketSpec

PROTOCOL_ID = "issue810_residual_ppo_synthetic_v1"

# Bound after the freeze. ``test_protocol_digest_is_frozen`` recomputes it.
FROZEN_PROTOCOL_SHA256 = (
    "96c11b5c3f14b5b29a8557a3d250ce16ac34594a43cd8c5ff0336ef9f2f03904"
)


@dataclass(frozen=True, slots=True)
class PpoSettings:
    """Fixed PPO hyper-parameters (stable-baselines3 ``PPO`` / ``MlpPolicy``)."""

    learning_rate: float = 3e-4
    n_steps: int = 1008  # four 252-day episodes per rollout
    batch_size: int = 252
    n_epochs: int = 8
    gamma: float = 1.0  # undiscounted: the reward telescopes to terminal log equity
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    net_arch: tuple[int, ...] = (64, 64)
    reward_scale: float = 100.0  # training-time scaling only; evaluation is unscaled
    total_timesteps: int = 50_400  # 200 episodes; fixed for every fit
    seeds: tuple[int, ...] = (0, 1, 2)
    deterministic_evaluation: bool = True


@dataclass(frozen=True, slots=True)
class SeedPlan:
    """Disjoint seed pools; the same market seeds are used for both regimes."""

    train: tuple[int, ...] = tuple(range(1000, 1100))
    held_out: tuple[int, ...] = tuple(range(2000, 2040))
    calibration: tuple[int, ...] = tuple(range(9000, 9012))

    def __post_init__(self) -> None:
        pools = (set(self.train), set(self.held_out), set(self.calibration))
        if sum(len(p) for p in pools) != len(set().union(*pools)):
            raise ValueError("seed pools must be disjoint")


@dataclass(frozen=True, slots=True)
class DecisionRules:
    """Pre-declared pass/fail thresholds."""

    margin: float = 0.01  # primary: median net log-return advantage over allocator
    bootstrap_resamples: int = 2_000
    bootstrap_seed: int = 810
    bootstrap_quantile: float = 0.05
    control_turnover_ratio: float = 0.25
    label_pass: str = "SYNTHETIC_PASS"
    label_fail: str = "SYNTHETIC_FAIL"


@dataclass(frozen=True, slots=True)
class ResidualProtocol:
    protocol_id: str = PROTOCOL_ID
    market: MarketSpec = field(default_factory=MarketSpec)
    allocator: AllocatorSettings = field(default_factory=AllocatorSettings)
    costs: CostSettings = field(default_factory=CostSettings)
    scales: ObservationScales = field(default_factory=ObservationScales)
    multipliers: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
    ppo: PpoSettings = field(default_factory=PpoSettings)
    seeds: SeedPlan = field(default_factory=SeedPlan)
    rules: DecisionRules = field(default_factory=DecisionRules)
    baselines: tuple[str, ...] = (
        "CASH",
        "ALLOCATOR_KAPPA_1",
        "ALLOCATOR_KAPPA_3",
        "STATIC_M_0.5",
        "STATIC_M_1.5",
        "RESIDUAL_PPO",
    )
    kappa_primary: float = 1.0
    kappa_comparison: float = 3.0
    synthetic_only: bool = True
    real_data_rl_gate: str = (
        "Real-data RL remains blocked behind the fee-only learnability check; "
        "this synthetic study does not clear it and is not a profitability claim."
    )


PROTOCOL_V1 = ResidualProtocol()


def protocol_payload(protocol: ResidualProtocol = PROTOCOL_V1) -> object:
    """Canonical JSON value tree of the protocol."""

    return to_json_value(protocol)


def protocol_digest(protocol: ResidualProtocol = PROTOCOL_V1) -> str:
    return content_digest(protocol)


def require_frozen_protocol(protocol: ResidualProtocol = PROTOCOL_V1) -> str:
    """Return the digest, raising unless it equals the committed frozen digest."""

    digest = protocol_digest(protocol)
    if digest != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "residual-RL protocol differs from the frozen digest: "
            f"{digest} != {FROZEN_PROTOCOL_SHA256}"
        )
    return digest


__all__ = [
    "FROZEN_PROTOCOL_SHA256",
    "PROTOCOL_ID",
    "PROTOCOL_V1",
    "DecisionRules",
    "PpoSettings",
    "ResidualProtocol",
    "SeedPlan",
    "protocol_digest",
    "protocol_payload",
    "require_frozen_protocol",
]
