"""Pure-NumPy Shadow-Anchor MTF PPO and Non-RL Regime-Gated Portfolio Policies.

This module provides deterministic CPU/NumPy inference and target synthesis over
alternating ``pair:spot`` / ``pair:perp`` pairs consumed from ``Mtf6SignalPack``:

- ``anc_bidir_75_25``: causal All-Weather Bidirectional 75/25 + VolCushionTail anchor
- ``anc_regime_gate_nonrl``: anchor scaled by fixed causal Non-RL regime gates
- ``core7_sparse_ppo``: residual CORE7-pillar LinearClipMlpActorCritic + anchor
- ``orth6_cush_ppo``: residual ORTH6-pillar LinearClipMlpActorCritic + anchor/cushion

It carries zero runtime dependency on PyTorch, Stable-Baselines3, Gymnasium,
evaluation, simulation, risk, or integrations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from trade_rl.data.features.mtf6_signals import (
    ORTH6_INDICES,
    Mtf6SignalPack,
    normalize_balanced_neutral_row,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)

ShadowAnchorMode = Literal[
    "anc_bidir_75_25",
    "anc_regime_gate_nonrl",
    "core7_sparse_ppo",
    "orth6_cush_ppo",
]

VALID_MODES: tuple[ShadowAnchorMode, ...] = (
    "anc_bidir_75_25",
    "anc_regime_gate_nonrl",
    "core7_sparse_ppo",
    "orth6_cush_ppo",
)
RL_MODES: frozenset[str] = frozenset({"core7_sparse_ppo", "orth6_cush_ppo"})
WEIGHTS_SCHEMA_VERSION = "shadow_anchor_mlp_weights_v1"


def _readonly_f64(
    arr: np.ndarray, *, expected_shape: tuple[int, ...], name: str
) -> np.ndarray:
    out = np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).view()
    if out.shape != expected_shape:
        raise ValueError(f"{name} expected shape {expected_shape}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    out.setflags(write=False)
    return out


def _array_sha256(arr: np.ndarray) -> str:
    canonical = np.ascontiguousarray(arr, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float
) -> np.ndarray:
    mean = float(np.mean(x))
    diff = x - mean
    var = float(np.mean(diff * diff))
    inv_std = 1.0 / math.sqrt(var + eps)
    return (diff * inv_std) * gamma + beta


def _silu(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -60.0, 60.0)
    return x / (1.0 + np.exp(-clipped))


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(weight @ x + bias, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ShadowAnchorMlpWeights:
    """Typed pure-NumPy weight container matching ``LinearClipMlpActorCritic``."""

    obs_mean_41: np.ndarray
    obs_std_41: np.ndarray
    in_weight: np.ndarray
    in_bias: np.ndarray
    in_ln_gamma: np.ndarray
    in_ln_beta: np.ndarray
    res1_a_weight: np.ndarray
    res1_a_bias: np.ndarray
    res1_a_ln_gamma: np.ndarray
    res1_a_ln_beta: np.ndarray
    res1_b_weight: np.ndarray
    res1_b_bias: np.ndarray
    res2_a_weight: np.ndarray
    res2_a_bias: np.ndarray
    res2_a_ln_gamma: np.ndarray
    res2_a_ln_beta: np.ndarray
    res2_b_weight: np.ndarray
    res2_b_bias: np.ndarray
    mu_weight: np.ndarray
    mu_bias: np.ndarray
    obs_dim: int = 52
    act_dim: int = 7
    hidden: int = 128
    fit_stop_index: int = 0
    layer_norm_eps: float = 1e-5
    schema_version: str = WEIGHTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WEIGHTS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported weights schema_version: {self.schema_version!r}"
            )
        if self.obs_dim != 52 or self.act_dim != 7 or self.hidden < 1:
            raise ValueError(
                "ShadowAnchorMlpWeights requires obs_dim=52, act_dim=7, hidden>=1"
            )
        if self.fit_stop_index < 0:
            raise ValueError("fit_stop_index must be non-negative")
        if not math.isfinite(self.layer_norm_eps) or self.layer_norm_eps <= 0.0:
            raise ValueError("layer_norm_eps must be finite and positive")

        obs_std = _readonly_f64(
            self.obs_std_41, expected_shape=(41,), name="obs_std_41"
        )
        if not np.all(obs_std > 0.0):
            raise ValueError("obs_std_41 entries must be strictly positive")

        h = self.hidden
        o = self.obs_dim
        a = self.act_dim
        object.__setattr__(
            self,
            "obs_mean_41",
            _readonly_f64(self.obs_mean_41, expected_shape=(41,), name="obs_mean_41"),
        )
        object.__setattr__(self, "obs_std_41", obs_std)
        object.__setattr__(
            self,
            "in_weight",
            _readonly_f64(self.in_weight, expected_shape=(h, o), name="in_weight"),
        )
        object.__setattr__(
            self,
            "in_bias",
            _readonly_f64(self.in_bias, expected_shape=(h,), name="in_bias"),
        )
        object.__setattr__(
            self,
            "in_ln_gamma",
            _readonly_f64(self.in_ln_gamma, expected_shape=(h,), name="in_ln_gamma"),
        )
        object.__setattr__(
            self,
            "in_ln_beta",
            _readonly_f64(self.in_ln_beta, expected_shape=(h,), name="in_ln_beta"),
        )
        object.__setattr__(
            self,
            "res1_a_weight",
            _readonly_f64(
                self.res1_a_weight, expected_shape=(h, h), name="res1_a_weight"
            ),
        )
        object.__setattr__(
            self,
            "res1_a_bias",
            _readonly_f64(self.res1_a_bias, expected_shape=(h,), name="res1_a_bias"),
        )
        object.__setattr__(
            self,
            "res1_a_ln_gamma",
            _readonly_f64(
                self.res1_a_ln_gamma, expected_shape=(h,), name="res1_a_ln_gamma"
            ),
        )
        object.__setattr__(
            self,
            "res1_a_ln_beta",
            _readonly_f64(
                self.res1_a_ln_beta, expected_shape=(h,), name="res1_a_ln_beta"
            ),
        )
        object.__setattr__(
            self,
            "res1_b_weight",
            _readonly_f64(
                self.res1_b_weight, expected_shape=(h, h), name="res1_b_weight"
            ),
        )
        object.__setattr__(
            self,
            "res1_b_bias",
            _readonly_f64(self.res1_b_bias, expected_shape=(h,), name="res1_b_bias"),
        )
        object.__setattr__(
            self,
            "res2_a_weight",
            _readonly_f64(
                self.res2_a_weight, expected_shape=(h, h), name="res2_a_weight"
            ),
        )
        object.__setattr__(
            self,
            "res2_a_bias",
            _readonly_f64(self.res2_a_bias, expected_shape=(h,), name="res2_a_bias"),
        )
        object.__setattr__(
            self,
            "res2_a_ln_gamma",
            _readonly_f64(
                self.res2_a_ln_gamma, expected_shape=(h,), name="res2_a_ln_gamma"
            ),
        )
        object.__setattr__(
            self,
            "res2_a_ln_beta",
            _readonly_f64(
                self.res2_a_ln_beta, expected_shape=(h,), name="res2_a_ln_beta"
            ),
        )
        object.__setattr__(
            self,
            "res2_b_weight",
            _readonly_f64(
                self.res2_b_weight, expected_shape=(h, h), name="res2_b_weight"
            ),
        )
        object.__setattr__(
            self,
            "res2_b_bias",
            _readonly_f64(self.res2_b_bias, expected_shape=(h,), name="res2_b_bias"),
        )
        object.__setattr__(
            self,
            "mu_weight",
            _readonly_f64(self.mu_weight, expected_shape=(a, h), name="mu_weight"),
        )
        object.__setattr__(
            self,
            "mu_bias",
            _readonly_f64(self.mu_bias, expected_shape=(a,), name="mu_bias"),
        )

    @classmethod
    def zero_residual(
        cls,
        *,
        fit_stop_index: int = 0,
        hidden: int = 128,
    ) -> ShadowAnchorMlpWeights:
        """Create exact zero-output weights for baseline parity testing (FM1 oracle)."""
        return cls(
            obs_mean_41=np.zeros(41, dtype=np.float64),
            obs_std_41=np.ones(41, dtype=np.float64),
            in_weight=np.zeros((hidden, 52), dtype=np.float64),
            in_bias=np.zeros(hidden, dtype=np.float64),
            in_ln_gamma=np.ones(hidden, dtype=np.float64),
            in_ln_beta=np.zeros(hidden, dtype=np.float64),
            res1_a_weight=np.zeros((hidden, hidden), dtype=np.float64),
            res1_a_bias=np.zeros(hidden, dtype=np.float64),
            res1_a_ln_gamma=np.ones(hidden, dtype=np.float64),
            res1_a_ln_beta=np.zeros(hidden, dtype=np.float64),
            res1_b_weight=np.zeros((hidden, hidden), dtype=np.float64),
            res1_b_bias=np.zeros(hidden, dtype=np.float64),
            res2_a_weight=np.zeros((hidden, hidden), dtype=np.float64),
            res2_a_bias=np.zeros(hidden, dtype=np.float64),
            res2_a_ln_gamma=np.ones(hidden, dtype=np.float64),
            res2_a_ln_beta=np.zeros(hidden, dtype=np.float64),
            res2_b_weight=np.zeros((hidden, hidden), dtype=np.float64),
            res2_b_bias=np.zeros(hidden, dtype=np.float64),
            mu_weight=np.zeros((7, hidden), dtype=np.float64),
            mu_bias=np.zeros(7, dtype=np.float64),
            obs_dim=52,
            act_dim=7,
            hidden=hidden,
            fit_stop_index=fit_stop_index,
        )

    def normalize_regime41(self, raw41: np.ndarray) -> np.ndarray:
        """Standardize the 41 causal regime features with training-fold moments."""
        arr = np.asarray(raw41, dtype=np.float64)
        if arr.shape != (41,) or not np.all(np.isfinite(arr)):
            raise ValueError("raw41 must be finite with shape (41,)")
        return np.clip((arr - self.obs_mean_41) / self.obs_std_41, -5.0, 5.0)

    def forward_deterministic(self, obs_52: np.ndarray) -> np.ndarray:
        """Compute deterministic clipped action vector ``mu in [-1, 1]^7``."""
        x = np.asarray(obs_52, dtype=np.float64)
        if x.shape != (self.obs_dim,) or not np.all(np.isfinite(x)):
            raise ValueError(f"obs_52 must be finite with shape ({self.obs_dim},)")

        eps = self.layer_norm_eps
        h = _silu(
            _layer_norm(
                _linear(x, self.in_weight, self.in_bias),
                self.in_ln_gamma,
                self.in_ln_beta,
                eps,
            )
        )
        r1 = _silu(
            _layer_norm(
                _linear(h, self.res1_a_weight, self.res1_a_bias),
                self.res1_a_ln_gamma,
                self.res1_a_ln_beta,
                eps,
            )
        )
        h = h + _linear(r1, self.res1_b_weight, self.res1_b_bias)
        r2 = _silu(
            _layer_norm(
                _linear(h, self.res2_a_weight, self.res2_a_bias),
                self.res2_a_ln_gamma,
                self.res2_a_ln_beta,
                eps,
            )
        )
        h = h + _linear(r2, self.res2_b_weight, self.res2_b_bias)
        return np.clip(_linear(h, self.mu_weight, self.mu_bias), -1.0, 1.0)

    def weights_digest(self) -> str:
        """Compute deterministic SHA-256 digest over metadata and all tensors."""
        payload = {
            "schema_version": self.schema_version,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden": self.hidden,
            "fit_stop_index": self.fit_stop_index,
            "layer_norm_eps": self.layer_norm_eps,
            "tensors": {
                "obs_mean_41": _array_sha256(self.obs_mean_41),
                "obs_std_41": _array_sha256(self.obs_std_41),
                "in_weight": _array_sha256(self.in_weight),
                "in_bias": _array_sha256(self.in_bias),
                "in_ln_gamma": _array_sha256(self.in_ln_gamma),
                "in_ln_beta": _array_sha256(self.in_ln_beta),
                "res1_a_weight": _array_sha256(self.res1_a_weight),
                "res1_a_bias": _array_sha256(self.res1_a_bias),
                "res1_a_ln_gamma": _array_sha256(self.res1_a_ln_gamma),
                "res1_a_ln_beta": _array_sha256(self.res1_a_ln_beta),
                "res1_b_weight": _array_sha256(self.res1_b_weight),
                "res1_b_bias": _array_sha256(self.res1_b_bias),
                "res2_a_weight": _array_sha256(self.res2_a_weight),
                "res2_a_bias": _array_sha256(self.res2_a_bias),
                "res2_a_ln_gamma": _array_sha256(self.res2_a_ln_gamma),
                "res2_a_ln_beta": _array_sha256(self.res2_a_ln_beta),
                "res2_b_weight": _array_sha256(self.res2_b_weight),
                "res2_b_bias": _array_sha256(self.res2_b_bias),
                "mu_weight": _array_sha256(self.mu_weight),
                "mu_bias": _array_sha256(self.mu_bias),
            },
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ShadowAnchorMtfConfig:
    """Configuration for Shadow-Anchor MTF policies on alternating spot/perp legs."""

    mode: ShadowAnchorMode
    start_index: int
    fit_stop_index: int
    rebalance_every_bars: int = 6
    deadband: float = 0.04
    soft_threshold: float = 0.05
    residual_scale: float = 0.26
    alpha_scale: float = 1.5
    carry_scale: float = 1.0
    ema_decay: float = 0.70
    w_max_leg: float = 0.55
    gross_cap: float = 2.5
    min_pairs: int = 2
    signal_lag_bars: int = 0

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"unsupported ShadowAnchorMtfConfig mode: {self.mode!r}")
        if self.fit_stop_index < 0 or self.start_index < 0:
            raise ValueError("fit_stop_index and start_index must be non-negative")
        if self.fit_stop_index > self.start_index:
            raise ValueError(
                f"lookahead forbidden: fit_stop_index ({self.fit_stop_index}) exceeds start_index ({self.start_index})"
            )
        if self.rebalance_every_bars < 1:
            raise ValueError("rebalance_every_bars must be >= 1")
        if not math.isfinite(self.deadband) or self.deadband < 0.0:
            raise ValueError("deadband must be finite and non-negative")
        if (
            not math.isfinite(self.soft_threshold)
            or not 0.0 <= self.soft_threshold < 1.0
        ):
            raise ValueError("soft_threshold must lie in [0, 1)")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")
        if not math.isfinite(self.alpha_scale) or self.alpha_scale < 0.0:
            raise ValueError("alpha_scale must be non-negative")
        if not math.isfinite(self.carry_scale) or self.carry_scale < 0.0:
            raise ValueError("carry_scale must be non-negative")
        if not math.isfinite(self.ema_decay) or not 0.0 <= self.ema_decay <= 1.0:
            raise ValueError("ema_decay must lie in [0, 1]")
        if not math.isfinite(self.w_max_leg) or self.w_max_leg <= 0.0:
            raise ValueError("w_max_leg must be positive")
        if not math.isfinite(self.gross_cap) or self.gross_cap < self.w_max_leg:
            raise ValueError("gross_cap must be >= w_max_leg")
        if self.min_pairs < 2:
            raise ValueError("min_pairs must be >= 2")
        if self.signal_lag_bars < 0:
            raise ValueError("signal_lag_bars must be non-negative")


def synthesize_alternating_leg_weights(
    c_w: np.ndarray,
    alpha_w: np.ndarray,
    *,
    c_scale: float,
    a_scale: float,
    carry_mask: np.ndarray,
    perp_mask: np.ndarray,
    w_max_leg: float = 0.55,
    gross_cap: float = 2.5,
) -> tuple[float, ...]:
    """Synthesize constrained alternating ``(spot_0, perp_0, ...)`` leg weights."""
    cw = np.asarray(c_w, dtype=np.float64)
    aw = np.asarray(alpha_w, dtype=np.float64)
    cm = np.asarray(carry_mask, dtype=bool)
    pm = np.asarray(perp_mask, dtype=bool)
    if (
        cw.ndim != 1
        or aw.shape != cw.shape
        or cm.shape != cw.shape
        or pm.shape != cw.shape
    ):
        raise ValueError(
            "c_w, alpha_w, carry_mask, and perp_mask must have identical 1D shapes"
        )
    if cw.size < 2:
        raise ValueError("at least 2 pairs are required for leg weight synthesis")
    if not (np.all(np.isfinite(cw)) and np.all(np.isfinite(aw))):
        raise ValueError("c_w and alpha_w must be finite")

    c_clean = np.where(cm, np.maximum(cw, 0.0), 0.0)
    a_clean = np.where(pm, aw, 0.0)
    ws_raw = 0.5 * float(c_scale) * c_clean
    wp_raw = -0.5 * float(c_scale) * c_clean + float(a_scale) * a_clean
    ws = np.clip(ws_raw, 0.0, float(w_max_leg))
    wp = np.clip(wp_raw, -float(w_max_leg), float(w_max_leg))
    gross = float(np.sum(np.abs(ws)) + np.sum(np.abs(wp)))
    if gross > float(gross_cap) + 1e-12:
        scale = float(gross_cap) / gross
        ws = ws * scale
        wp = wp * scale

    legs = np.empty(2 * cw.size, dtype=np.float64)
    legs[0::2] = ws
    legs[1::2] = wp
    return tuple(float(v) for v in legs)


class CashHoldPortfolioPolicy(PortfolioPolicy):
    """Benchmark policy holding zero risky weight on all legs."""

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(target_weights=(0.0,) * len(snapshot.symbols))


class ShadowAnchorMtfPolicy(PortfolioPolicy):
    """Deterministic Shadow-Anchor MTF policy implementing ``PortfolioPolicy``."""

    def __init__(
        self,
        *,
        signal_pack: Mtf6SignalPack,
        config: ShadowAnchorMtfConfig,
        weights: ShadowAnchorMlpWeights | None = None,
    ) -> None:
        if not isinstance(signal_pack, Mtf6SignalPack):
            raise TypeError("signal_pack must be an Mtf6SignalPack")
        if config.fit_stop_index > config.start_index:
            raise ValueError("fit_stop_index must not exceed start_index")
        if config.mode in RL_MODES:
            if weights is None:
                raise ValueError(
                    f"mode {config.mode!r} requires explicit ShadowAnchorMlpWeights"
                )
            if weights.fit_stop_index > config.start_index:
                raise ValueError(
                    f"weights fit_stop_index ({weights.fit_stop_index}) exceeds start_index ({config.start_index})"
                )
        elif weights is not None and weights.fit_stop_index > config.start_index:
            raise ValueError("weights fit_stop_index must not exceed start_index")

        self._pack = signal_pack
        self._config = config
        self._weights = weights
        self._expected_symbols = tuple(
            sym
            for pair in signal_pack.pair_names
            for sym in (f"{pair}:spot", f"{pair}:perp")
        )
        self._last_index: int | None = None
        self._a_smooth = np.zeros(7, dtype=np.float64)
        self._ema_ex = 0.0
        self._ema_sq = 1e-6
        self._eq0: float | None = None
        self._anc_eq = 1.0
        self._anc_eq0 = 1.0
        self._anc_prev_legs = np.zeros(len(self._expected_symbols), dtype=np.float64)
        # Pre-risk target of the most recent rebalance bar (sent or deadband-held).
        self._intended_target: tuple[float, ...] | None = None

    @property
    def config(self) -> ShadowAnchorMtfConfig:
        return self._config

    @property
    def weights(self) -> ShadowAnchorMlpWeights | None:
        return self._weights

    def weights_digest(self) -> str | None:
        return None if self._weights is None else self._weights.weights_digest()

    def _validate_snapshot(self, snapshot: PortfolioDecisionSnapshot) -> int:
        if snapshot.symbols != self._expected_symbols:
            raise ValueError(
                "snapshot symbols do not match alternating spot/perp signal_pack pairs"
            )
        t_len = int(self._pack.timestamps.shape[0])
        if not 0 <= snapshot.index < t_len:
            raise ValueError(
                f"snapshot.index {snapshot.index} outside signal_pack range [0, {t_len})"
            )
        if snapshot.index < self._config.start_index:
            raise ValueError(
                f"snapshot.index ({snapshot.index}) precedes configured start_index ({self._config.start_index})"
            )
        if self._last_index is not None and snapshot.index <= self._last_index:
            raise ValueError(
                f"snapshot.index must advance monotonically: {snapshot.index} <= {self._last_index}"
            )
        self._last_index = snapshot.index
        return max(0, snapshot.index - self._config.signal_lag_bars)

    def _advance_causal_anchor_proxy(self, k: int) -> None:
        if k > 0:
            r_spot = self._pack.spot_ret_causal[k]
            r_perp = self._pack.perp_ret_causal[k]
            pnl = float(
                np.dot(self._anc_prev_legs[0::2], r_spot)
                + np.dot(self._anc_prev_legs[1::2], r_perp)
            )
            self._anc_eq = max(1e-8, self._anc_eq * math.exp(max(-0.5, min(0.5, pnl))))

    def _build_obs_52(self, k: int, snapshot: PortfolioDecisionSnapshot) -> np.ndarray:
        assert self._weights is not None
        f41_norm = self._weights.normalize_regime41(self._pack.regime_feats_41[k])
        eq = max(float(snapshot.account.portfolio_value), 1e-8)
        peak_eq = max(float(snapshot.account.peak_value), eq)
        dd_cur = max(0.0, 1.0 - eq / peak_eq)
        vc_rel = (float(self._pack.vol_cushion_tail[k]) - 0.92) * 8.0
        gross_u = (
            float(sum(abs(float(w)) for w in snapshot.account.weights))
            / self._config.gross_cap
        )
        sd = math.sqrt(max(self._ema_sq - self._ema_ex * self._ema_ex, 1e-8))
        diff_sr = float(np.clip(self._ema_ex / sd, -3.0, 3.0))
        extra = np.array(
            [
                min(max(dd_cur * 5.0, 0.0), 4.0),
                min(max(vc_rel, -3.0), 3.0),
                min(max(gross_u, 0.0), 2.0),
                diff_sr,
            ],
            dtype=np.float64,
        )
        return np.nan_to_num(
            np.concatenate([f41_norm, self._a_smooth, extra]),
            nan=0.0,
            posinf=5.0,
            neginf=-5.0,
        )

    def _drawdown_breached(self, snapshot: PortfolioDecisionSnapshot) -> bool:
        peak = max(float(snapshot.account.peak_value), 1e-12)
        current_dd = max(0.0, 1.0 - float(snapshot.account.portfolio_value) / peak)
        max_dd = float(snapshot.account.max_drawdown)
        threshold = float(snapshot.risk.drawdown_start) - 1e-12
        return current_dd >= threshold or max_dd >= threshold

    def _compute_rebalance_weights(
        self,
        k: int,
        snapshot: PortfolioDecisionSnapshot,
    ) -> tuple[float, ...]:
        cfg = self._config
        eq = max(float(snapshot.account.portfolio_value), 1e-8)
        if self._eq0 is not None:
            ex_lr = math.log(eq / max(self._eq0, 1e-8)) - math.log(
                max(self._anc_eq, 1e-8) / max(self._anc_eq0, 1e-8)
            )
            self._ema_ex = 0.96 * self._ema_ex + 0.04 * ex_lr
            self._ema_sq = 0.96 * self._ema_sq + 0.04 * (ex_lr * ex_lr)
        self._eq0 = eq
        self._anc_eq0 = self._anc_eq

        pm = self._pack.perp_mask[k]
        cm = self._pack.carry_mask[k]
        vc = float(self._pack.vol_cushion_tail[k])
        c_w = self._pack.c_weekly_iv[k]

        anc_legs = synthesize_alternating_leg_weights(
            c_w,
            self._pack.w_aw_bidir_75_25[k],
            c_scale=cfg.carry_scale,
            a_scale=cfg.alpha_scale * vc,
            carry_mask=cm,
            perp_mask=pm,
            w_max_leg=cfg.w_max_leg,
            gross_cap=cfg.gross_cap,
        )
        self._anc_prev_legs = np.asarray(anc_legs, dtype=np.float64)

        if cfg.mode == "anc_bidir_75_25":
            return anc_legs

        if cfg.mode == "anc_regime_gate_nonrl":
            return synthesize_alternating_leg_weights(
                c_w,
                self._pack.w_aw_bidir_75_25[k],
                c_scale=cfg.carry_scale * float(self._pack.gate_carry_nonrl[k]),
                a_scale=cfg.alpha_scale * vc * float(self._pack.gate_alpha_nonrl[k]),
                carry_mask=cm,
                perp_mask=pm,
                w_max_leg=cfg.w_max_leg,
                gross_cap=cfg.gross_cap,
            )

        assert self._weights is not None
        obs_52 = self._build_obs_52(k, snapshot)
        a_raw = self._weights.forward_deterministic(obs_52)
        self._a_smooth = cfg.ema_decay * self._a_smooth + (1.0 - cfg.ema_decay) * a_raw

        if cfg.mode == "core7_sparse_ppo":
            ae = np.sign(self._a_smooth) * np.maximum(
                np.abs(self._a_smooth) - cfg.soft_threshold, 0.0
            )
            res = cfg.residual_scale * np.tensordot(
                self._pack.core7_signals[k], ae, axes=([1], [0])
            )
            cm_adj = 1.0
        else:
            ae = np.sign(self._a_smooth[:6]) * np.maximum(
                np.abs(self._a_smooth[:6]) - cfg.soft_threshold, 0.0
            )
            res = cfg.residual_scale * np.tensordot(
                self._pack.orth6_signals[k], ae, axes=([1], [0])
            )
            cm_adj = float(np.clip(1.0 + 0.08 * float(self._a_smooth[6]), 0.92, 1.06))
            assert len(ORTH6_INDICES) == 6

        alpha_raw = self._pack.w_aw_bidir_75_25[k] + res
        alpha_row = (
            normalize_balanced_neutral_row(
                np.where(pm, alpha_raw, 0.0), pm, w_max=0.25, gross_max=1.0
            )
            if int(np.sum(pm)) >= cfg.min_pairs
            else np.zeros(len(self._pack.pair_names), dtype=np.float64)
        )
        return synthesize_alternating_leg_weights(
            c_w,
            alpha_row,
            c_scale=cfg.carry_scale,
            a_scale=cfg.alpha_scale * vc * cm_adj,
            carry_mask=cm,
            perp_mask=pm,
            w_max_leg=cfg.w_max_leg,
            gross_cap=cfg.gross_cap,
        )

    @staticmethod
    def _risk_scale(snapshot: PortfolioDecisionSnapshot) -> float:
        """``PreTradeRisk.risk_scale`` of the snapshot's drawdown (same linear ramp)."""
        peak = max(float(snapshot.account.peak_value), 1e-12)
        current_dd = max(0.0, 1.0 - float(snapshot.account.portfolio_value) / peak)
        drawdown = min(1.0, max(current_dd, float(snapshot.account.max_drawdown)))
        start = float(snapshot.risk.drawdown_start)
        stop = float(snapshot.risk.drawdown_stop)
        if drawdown <= start:
            return 1.0
        if drawdown >= stop or stop == start:
            return 0.0
        return 1.0 - (drawdown - start) / (stop - start)

    def _breach_proposal(
        self,
        snapshot: PortfolioDecisionSnapshot,
        intended: tuple[float, ...],
    ) -> PortfolioTargetProposal:
        """Non-rebalance proposal while the drawdown stage is active.

        The cached pre-risk target is re-proposed so ``PreTradeRisk`` applies
        ``risk_scale`` exactly once. An exact-quantity HOLD cannot be used here: risk
        would scale the held (already scaled) weights again. When the risk-scaled target
        is within the deadband of the current weights the proposal is instead the
        pre-image ``current / risk_scale`` that risk maps back onto the current weights,
        so drift-sized re-trades are not churned every bar, while a real change of
        ``risk_scale`` or a drift beyond the deadband still re-targets the intended book.
        """
        scale = self._risk_scale(snapshot)
        current = np.asarray(snapshot.account.weights, dtype=np.float64)
        scaled = np.asarray(intended, dtype=np.float64) * scale
        if scale > 0.0 and float(np.sum(np.abs(scaled - current))) < (
            self._config.deadband
        ):
            pre_image = current / scale
            if float(np.max(np.abs(pre_image), initial=0.0)) <= float(
                snapshot.risk.max_abs_weight
            ) and float(np.sum(np.abs(pre_image))) <= float(snapshot.risk.max_gross):
                return PortfolioTargetProposal(
                    target_weights=tuple(float(w) for w in pre_image)
                )
        return PortfolioTargetProposal(target_weights=intended)

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        k = self._validate_snapshot(snapshot)
        self._advance_causal_anchor_proxy(k)

        n_legs = len(self._expected_symbols)
        all_holds = tuple(range(n_legs))
        drawdown_breached = self._drawdown_breached(snapshot)
        step_offset = snapshot.index - self._config.start_index

        if step_offset % self._config.rebalance_every_bars != 0:
            if drawdown_breached and self._intended_target is not None:
                return self._breach_proposal(snapshot, self._intended_target)
            return PortfolioTargetProposal(
                target_weights=snapshot.account.weights,
                hold_current_quantity_symbols=all_holds,
            )

        proposed = self._compute_rebalance_weights(k, snapshot)
        self._intended_target = tuple(float(w) for w in proposed)
        l1_diff = sum(
            abs(target - current)
            for target, current in zip(proposed, snapshot.account.weights, strict=True)
        )
        if l1_diff < self._config.deadband and not drawdown_breached:
            return PortfolioTargetProposal(
                target_weights=snapshot.account.weights,
                hold_current_quantity_symbols=all_holds,
            )
        return PortfolioTargetProposal(target_weights=proposed)


__all__ = [
    "RL_MODES",
    "VALID_MODES",
    "WEIGHTS_SCHEMA_VERSION",
    "CashHoldPortfolioPolicy",
    "ShadowAnchorMlpWeights",
    "ShadowAnchorMode",
    "ShadowAnchorMtfConfig",
    "ShadowAnchorMtfPolicy",
    "synthesize_alternating_leg_weights",
]
