"""Shared synthetic fixtures for the learning-lab tests (no market data is read)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.evaluation.agy_anchor_oos import protocol
from trade_rl.evaluation.learning_lab import baselines, runner, synthetic
from trade_rl.evaluation.learning_lab import overlay_env as oe

CAPITAL = 100_000.0
START = 20
STOP = 190


def stage(**kwargs: object) -> synthetic.SyntheticStage:
    params: dict[str, object] = {"n_bars": 200, "seed": 1}
    params.update(kwargs)
    return synthetic.synthetic_stage(**params)  # type: ignore[arg-type]


def make_env(
    st: synthetic.SyntheticStage,
    *,
    start: int = START,
    stop: int = STOP,
    mode: str = oe.ACTION_DISCRETE,
    perturbation: oe.PerturbationSource | None = None,
) -> oe.ExecutionOverlayEnv:
    return oe.ExecutionOverlayEnv(
        st.dataset,
        start_index=start,
        stop_index=stop,
        base_factory=baselines.anchor_stack_factory(st.signals, start_index=start),
        execution_cost=protocol.scenario_cost("BASE"),
        risk_config=protocol.risk_config(),
        config=oe.OverlayConfig(action_mode=mode, initial_capital=CAPITAL),
        perturbation=perturbation,
    )


def run_episode(
    env: oe.ExecutionOverlayEnv,
    policy: object,
    *,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[float], list[object], dict[str, object]]:
    """Closed-loop episode: ``policy(step, observation) -> action``."""
    observation, _ = env.reset(seed=seed)
    observations = [observation]
    rewards: list[float] = []
    actions: list[object] = []
    info: dict[str, object] = {}
    done = False
    step = 0
    while not done:
        action = policy(step, observation)  # type: ignore[operator]
        observation, reward, done, _, info = env.step(action)
        observations.append(observation)
        rewards.append(reward)
        actions.append(action)
        step += 1
    return observations, rewards, actions, info


def synthetic_setup(n_bars: int = 120, seed: int = 3) -> runner.LabSetup:
    st = synthetic.synthetic_stage(n_bars=n_bars, seed=seed)
    return runner.LabSetup(
        dataset=st.dataset,
        signals=st.signals,
        execution_cost=protocol.scenario_cost("BASE"),
        risk_config=protocol.risk_config(),
        splits=runner.synthetic_splits(n_bars),
    )


def perturb_future(
    st: synthetic.SyntheticStage, *, cut: int, seed: int
) -> synthetic.SyntheticStage:
    """The stage with every market row and signal row after ``cut`` replaced."""
    rng = np.random.default_rng(seed)
    ds = st.dataset
    n_future = ds.n_bars - cut - 1
    factor = np.ones((ds.n_bars, ds.n_symbols))
    factor[cut + 1 :] = rng.uniform(0.6, 1.6, size=(n_future, ds.n_symbols))
    volume = np.array(ds.volume, copy=True)
    volume[cut + 1 :] = rng.uniform(1.0e12, 3.0e12, size=(n_future, ds.n_symbols))
    new_ds = replace(
        ds,
        open=ds.open * factor,
        high=ds.high * factor,
        low=ds.low * factor,
        close=ds.close * factor,
        volume=volume,
    )
    sig = st.signals
    weights = np.array(sig.w_aw_bidir_75_25, copy=True)
    weights[cut + 1 :] = rng.choice([-0.25, 0.25], size=weights[cut + 1 :].shape)
    new_sig = replace(
        sig,
        w_aw_bidir_75_25=weights,
        w_aw_regime=weights,
        w_perp_bidir_carry=weights,
    )
    return synthetic.SyntheticStage(dataset=new_ds, signals=new_sig)
