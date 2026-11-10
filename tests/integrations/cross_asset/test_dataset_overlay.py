"""Opt-in dataset hook: byte-identical defaults and the PPO observation path."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.integrations.cross_asset.support import seed_store
from trade_rl.data.market import MarketDataset
from trade_rl.integrations.cross_asset.dataset import (
    GLOBAL_PREFIX,
    MISSING_REASON_CROSS_ASSET,
    with_cross_asset_global_features,
)
from trade_rl.integrations.cross_asset.features import FeatureFrame
from trade_rl.integrations.cross_asset.loader import load_cross_asset_features
from trade_rl.integrations.cross_asset.normalization import FeatureNormalizer
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)
from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
from trade_rl.strategies.rl.portfolio_transform import (
    fit_portfolio_observation_transform,
)

SELECTED = ("dollar_strength", "risk_appetite", "curve_slope", "trend_breadth_60")
START = datetime(2024, 6, 3, tzinfo=UTC)
# Identity of ``_dataset()`` below, computed before this package existed. It
# must never change because the overlay package was added.
GOLDEN_BASE_DATASET_ID = (
    "b3a27aedec1f1deb146647688da798face43249a73ff3dfcaacc16245f9c1373"
)


def _dataset(n_bars: int = 8, start: datetime = START) -> MarketDataset:
    """Synthetic single-asset dataset on a 4h grid (no market data)."""

    close = np.full((n_bars, 1), 100.0)
    zeros = np.zeros((n_bars, 1))
    first = np.datetime64(start.replace(tzinfo=None), "ns")
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=("ASSET",),
        timestamps=first + np.arange(n_bars) * np.timedelta64(4, "h"),
        features=np.arange(n_bars, dtype=np.float32).reshape(n_bars, 1, 1),
        global_features=np.arange(n_bars, dtype=np.float32).reshape(n_bars, 1),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=zeros.copy(),
        borrow_rate=zeros.copy(),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("literal",),
        global_feature_names=("regime",),
        periods_per_year=2_190,
        fee_rate=zeros.copy(),
        maker_fee_rate=zeros.copy(),
        taker_fee_rate=zeros.copy(),
        spread_rate=zeros.copy(),
        contract_multipliers=np.ones(1),
    ).with_content_identity()


@pytest.fixture()
def frame(tmp_path: Path) -> FeatureFrame:
    store = seed_store(tmp_path)
    return load_cross_asset_features(store.root, _dataset().timestamps)


def test_existing_dataset_identity_is_unchanged_by_the_overlay_package() -> None:
    base = _dataset()
    assert base.identity_verified
    assert base.dataset_id == GOLDEN_BASE_DATASET_ID
    assert base.global_feature_names == ("regime",)


def test_default_import_paths_never_load_the_cross_asset_package() -> None:
    code = (
        "import sys\n"
        "import trade_rl, trade_rl.data, trade_rl.integrations\n"
        "import trade_rl.strategies.rl.portfolio_env\n"
        "import trade_rl.strategies.rl.portfolio_transform\n"
        "import trade_rl.evaluation\n"
        "loaded = [m for m in sys.modules if m.startswith("
        "'trade_rl.integrations.cross_asset')]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_overlay_appends_columns_and_leaves_existing_bytes_identical(
    frame: FeatureFrame,
) -> None:
    base = _dataset()
    combined = with_cross_asset_global_features(base, frame, feature_names=SELECTED)
    assert combined.identity_verified
    assert combined.dataset_id != base.dataset_id
    assert combined.global_feature_names == (
        "regime",
        *(f"{GLOBAL_PREFIX}{name}" for name in SELECTED),
    )
    assert combined.global_features[:, :1].tobytes() == base.global_features.tobytes()
    assert combined.features.tobytes() == base.features.tobytes()
    for field in ("open", "high", "low", "close", "volume", "timestamps"):
        assert getattr(combined, field).tobytes() == getattr(base, field).tobytes()
    assert combined.feature_names == base.feature_names
    assert combined.global_features.shape == (base.n_bars, 1 + len(SELECTED))
    assert combined.global_feature_available.all()
    assert (combined.global_feature_staleness_hours[:, 1:] > 0.0).all()
    # The input is untouched and still verifies.
    assert base.dataset_id == GOLDEN_BASE_DATASET_ID and base.identity_verified


def test_overlay_is_reproducible_and_binds_its_inputs(frame: FeatureFrame) -> None:
    base = _dataset()
    first = with_cross_asset_global_features(base, frame, feature_names=SELECTED)
    again = with_cross_asset_global_features(base, frame, feature_names=SELECTED)
    assert first.dataset_id == again.dataset_id
    fewer = with_cross_asset_global_features(base, frame, feature_names=SELECTED[:2])
    assert fewer.dataset_id != first.dataset_id
    assert first.identity_payload_json is not None
    assert frame.input_digest in first.identity_payload_json
    assert frame.definitions_digest in first.identity_payload_json


def test_unavailable_entries_are_zero_with_an_explicit_mask(tmp_path: Path) -> None:
    store = seed_store(tmp_path)
    early = _dataset(start=datetime(2022, 1, 3, tzinfo=UTC))  # before any release
    frame = load_cross_asset_features(store.root, early.timestamps)
    combined = with_cross_asset_global_features(early, frame, feature_names=SELECTED)
    assert not combined.global_feature_available[:, 1:].any()
    assert (combined.global_features[:, 1:] == 0.0).all()
    assert (combined.global_feature_staleness_hours[:, 1:] == 0.0).all()
    assert (
        combined.global_feature_missing_reason[:, 1:] == MISSING_REASON_CROSS_ASSET
    ).all()
    assert combined.global_feature_available[:, :1].all()


def test_overlay_requires_the_exact_decision_grid_and_an_explicit_selection(
    frame: FeatureFrame,
) -> None:
    shifted = _dataset(start=START + timedelta(hours=4))
    with pytest.raises(ValueError, match="decision times"):
        with_cross_asset_global_features(shifted, frame, feature_names=SELECTED)
    with pytest.raises(ValueError, match="at least one"):
        with_cross_asset_global_features(_dataset(), frame, feature_names=())
    with pytest.raises(ValueError, match="unique"):
        with_cross_asset_global_features(
            _dataset(), frame, feature_names=("curve_slope", "curve_slope")
        )
    with pytest.raises(KeyError):
        with_cross_asset_global_features(_dataset(), frame, feature_names=("nope",))
    once = with_cross_asset_global_features(
        _dataset(), frame, feature_names=("curve_slope",)
    )
    with pytest.raises(ValueError, match="already present"):
        with_cross_asset_global_features(once, frame, feature_names=("curve_slope",))


def test_fit_window_normalizer_is_applied_before_the_overlay(tmp_path: Path) -> None:
    store = seed_store(tmp_path)
    base = _dataset(n_bars=60)
    frame = load_cross_asset_features(store.root, base.timestamps)
    fit_end = base.timestamps[40]
    normalizer = FeatureNormalizer.fit(
        frame.select(SELECTED),
        fit_start=base.timestamps[0],
        fit_end=fit_end,
        min_count=10,
    )
    combined = with_cross_asset_global_features(
        base,
        frame.select(SELECTED),
        feature_names=SELECTED,
        normalizer=normalizer,
    )
    assert combined.identity_payload_json is not None
    assert normalizer.digest() in combined.identity_payload_json
    inside = combined.global_features[:40, 1:]
    assert np.isfinite(inside).all()
    assert np.abs(combined.global_features[:, 1:]).max() <= normalizer.clip + 1e-6
    # Fit statistics come only from the fit window.
    assert normalizer.n_fit.max() <= 40


def _bound(n_bars: int) -> BoundObjectiveClock:
    start = datetime(2024, 6, 3, tzinfo=UTC)
    objective = ObjectiveContract(
        capital=CapitalContract("shared_portfolio", "USDT", (1_000.0,)),
        evaluation_start=start,
        evaluation_stop_exclusive=start + timedelta(hours=4 * (n_bars - 1)),
        terminal_valuation="settled",
        economics_digest="a" * 64,
        risk_digest="b" * 64,
        deployment_recipe_digest="c" * 64,
        maximum_drawdown=0.20,
    )
    clock = FinancialClockContract(
        decision_interval_seconds=14_400,
        execution_interval_seconds=14_400,
        reward_interval_seconds=14_400,
        economic_horizon_seconds=(n_bars - 1) * 14_400,
        rollout_steps=1,
        gamma=1.0,
        gae_lambda=0.95,
        reward_schema="equity_delta_v1",
    )
    return BoundObjectiveClock(objective, clock)


def test_overlay_reaches_the_ppo_observation_and_transform(tmp_path: Path) -> None:
    store = seed_store(tmp_path)
    n_bars = 40
    base = _dataset(n_bars=n_bars)
    frame = load_cross_asset_features(store.root, base.timestamps)
    combined = with_cross_asset_global_features(base, frame, feature_names=SELECTED)

    def env_for(dataset: MarketDataset) -> SharedPortfolioGymEnv:
        return SharedPortfolioGymEnv(
            dataset,
            start_index=0,
            stop_index=n_bars - 1,
            bound=_bound(n_bars),
            target_abs_weights=(0.2,),
            execution_cost=ExecutionCostConfig.zero(),
            risk_config=PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=0.5,
                max_turnover=None,
                drawdown_start=0.2,
                drawdown_stop=0.2,
            ),
        )

    plain_env = env_for(base)
    plain, _ = plain_env.reset(seed=1)
    env = env_for(combined)
    observation, _ = env.reset(seed=1)
    assert plain["global_features"].shape == (1,)
    assert observation["global_features"].shape == (1 + len(SELECTED),)
    assert env.observation_space.contains(observation)
    assert observation["global_features"][0] == plain["global_features"][0]
    assert observation["global_feature_available"].tolist() == [1] * (1 + len(SELECTED))
    transform = fit_portfolio_observation_transform(env)
    normalized = transform.transform_observation(observation)
    assert normalized["global_features"].shape == (1 + len(SELECTED),)
    assert np.isfinite(normalized["global_features"]).all()
