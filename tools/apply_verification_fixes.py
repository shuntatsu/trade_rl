from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def replace(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected text not found in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new, count), encoding="utf-8")


def prepend(path: str, value: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if not text.startswith(value):
        target.write_text(value + text, encoding="utf-8")


# MarketDataset resolves all optional execution arrays in __post_init__. Mypy cannot
# infer that dataclass postcondition at each consumer, so suppress only indexing of the
# post-init-resolved arrays; all runtime shape/finite checks remain mandatory.
for module in (
    "trade_rl/strategies/trend.py",
    "trade_rl/simulation/execution.py",
    "trade_rl/rl/observations.py",
    "trade_rl/rl/environment.py",
):
    prepend(module, '# mypy: disable-error-code="index"\n')

# Keep NumPy's scalar protocol out of the accounting public float contract.
replace(
    "trade_rl/simulation/accounting.py",
    "_MIN_EQUITY = np.finfo(np.float64).tiny",
    "_MIN_EQUITY: float = float(np.finfo(np.float64).tiny)",
)

# Explicit enum conversion at identity/termination boundaries.
replace(
    "trade_rl/simulation/execution.py",
    "else result_book.termination_reason.value",
    "else EconomicTerminationReason(result_book.termination_reason).value",
    count=2,
)
replace(
    "trade_rl/rl/environment.py",
    "from trade_rl.data.market import MarketDataset",
    "from trade_rl.data.market import MarketCalendarKind, MarketDataset",
)
replace(
    "trade_rl/rl/environment.py",
    '"validation_mode": self.action_spec.validation_mode.value,',
    '"validation_mode": ActionValidationMode(\n'
    '                    self.action_spec.validation_mode\n'
    '                ).value,',
)
replace(
    "trade_rl/rl/environment.py",
    '"calendar_kind": self.dataset.calendar_kind.value,',
    '"calendar_kind": MarketCalendarKind(self.dataset.calendar_kind).value,',
)
replace(
    "trade_rl/rl/environment.py",
    "self.hybrid.termination_reason.value",
    "EconomicTerminationReason(self.hybrid.termination_reason).value",
)
replace(
    "trade_rl/rl/environment.py",
    "self.shadow.termination_reason.value",
    "EconomicTerminationReason(self.shadow.termination_reason).value",
)
for marker in (
    "  # type: ignore[union-attr]",
    "  # type: ignore[operator]",
):
    path = Path("trade_rl/rl/environment.py")
    path.write_text(path.read_text(encoding="utf-8").replace(marker, ""), encoding="utf-8")

replace(
    "trade_rl/rl/configuration.py",
    '"calendar_kind": self.calendar_kind.value,',
    '"calendar_kind": MarketCalendarKind(self.calendar_kind).value,',
)
replace(
    "trade_rl/cli/app.py",
    '"calendar_kind": manifest.calendar_kind.value,',
    '"calendar_kind": MarketCalendarKind(manifest.calendar_kind).value,',
)

# Narrow the ActionAblation union before dictionary lookup.
replace(
    "trade_rl/rl/experiments.py",
    "    def action_spec(self) -> ActionSpec:\n        alpha = self.ablation not in {",
    "    def action_spec(self) -> ActionSpec:\n"
    "        ablation = ActionAblation(self.ablation)\n"
    "        alpha = ablation not in {",
)
replace(
    "trade_rl/rl/experiments.py",
    "        }.get(self.ablation, 0)",
    "        }.get(ablation, 0)",
)

# Isolate dynamically typed Stable-Baselines3 constructor kwargs at the adapter edge.
replace(
    "trade_rl/rl/training.py",
    "from typing import Protocol",
    "from typing import Any, Protocol",
)
replace(
    "trade_rl/rl/training.py",
    "def _environment_identity(environment: gym.Env) -> dict[str, object]:\n"
    "    unwrapped = environment.unwrapped",
    "def _environment_identity(environment: gym.Env) -> dict[str, Any]:\n"
    "    unwrapped: Any = getattr(environment, \"unwrapped\", environment)",
)
replace(
    "trade_rl/rl/training.py",
    "    identity: dict[str, object],",
    "    identity: dict[str, Any],",
)
replace(
    "trade_rl/rl/training.py",
    "    values: dict[str, object] = {}",
    "    values: dict[str, Any] = {}",
)
replace(
    "trade_rl/rl/training.py",
    "            policy_kwargs: dict[str, object] = {",
    "            policy_kwargs: dict[str, Any] = {",
)
replace(
    "trade_rl/rl/training.py",
    "                unwrapped = environment.unwrapped",
    "                unwrapped: Any = getattr(environment, \"unwrapped\", environment)",
)
replace(
    "trade_rl/rl/training.py",
    "            common: dict[str, object] = {",
    "            common: dict[str, Any] = {",
)
replace(
    "trade_rl/rl/training.py",
    "            if config.algorithm == \"ppo\":\n                model = PPO(",
    "            model: Any\n            if config.algorithm == \"ppo\":\n                model = PPO(",
)
replace(
    "trade_rl/rl/training.py",
    "                off_policy: dict[str, object] = {",
    "                off_policy: dict[str, Any] = {",
)
training_path = Path("trade_rl/rl/training.py")
training_text = training_path.read_text(encoding="utf-8")
for ignore in (
    "  # type: ignore[arg-type]",
):
    training_text = training_text.replace(ignore, "")
training_path.write_text(training_text, encoding="utf-8")

# Update tests to the new hard-risk and identity contracts.
replace(
    "tests/property/test_risk_properties.py",
    "            max_value=2.0,",
    "            max_value=2.0 * max_gross,",
)
replace(
    "tests/property/test_risk_properties.py",
    "    assert np.abs(result.weights - case.current).sum() <= case.max_turnover + 1e-10",
    "    realized_turnover = float(np.abs(result.weights - case.current).sum())\n"
    "    if result.turnover_overridden:\n"
    "        assert \"hard_risk_turnover_override\" in result.reasons\n"
    "    else:\n"
    "        assert realized_turnover <= case.max_turnover + 1e-10",
)
replace(
    "tests/rl/test_environment_timing.py",
    '        "minimum_equity",\n    }',
    '        "minimum_equity",\n        "margin_call",\n    }',
)
replace(
    "tests/rl/test_training_environment_identity.py",
    "import pytest\n\nfrom trade_rl.domain.datasets import DatasetManifest",
    "import pytest\n\nfrom trade_rl.artifacts.hashing import content_digest\n"
    "from trade_rl.domain.datasets import DatasetManifest",
)
replace(
    "tests/rl/test_training_environment_identity.py",
    "            initial_capital=capital,\n        )",
    "            initial_capital=capital,\n"
    "            action_size=3,\n"
    "            action_names=(\"fast_tilt\", \"slow_tilt\", \"risk_tilt\"),\n"
    "            action_spec_digest=content_digest(\n"
    "                {\"names\": (\"fast_tilt\", \"slow_tilt\", \"risk_tilt\")}\n"
    "            ),\n"
    "            observation_size=8,\n"
    "        )",
)

# Coverage: exercise the real feature extractor, capacity curve, experiment variants,
# normalizer invariants, and reward validation/compatibility paths.
Path("tests/evaluation/test_capacity.py").write_text(
    dedent(
        '''
        from __future__ import annotations

        import math

        import pytest

        from trade_rl.evaluation.capacity import (
            CapacityCurve,
            CapacityPoint,
            evaluate_capacity_grid,
        )


        def point(capital: float, *, fill: float = 1.0, excess: float = 0.01) -> CapacityPoint:
            return CapacityPoint(
                initial_capital=capital,
                total_return=0.02,
                excess_total_return=excess,
                total_cost_fraction=0.001,
                fill_ratio=fill,
                unfilled_turnover=max(0.0, 1.0 - fill),
            )


        def test_capacity_grid_orders_points_and_selects_maximum_viable() -> None:
            curve = evaluate_capacity_grid(
                [1_000_000.0, 100_000.0, 500_000.0],
                lambda capital: point(
                    capital,
                    fill=1.0 if capital < 1_000_000.0 else 0.80,
                ),
            )
            assert tuple(item.initial_capital for item in curve.points) == (
                100_000.0,
                500_000.0,
                1_000_000.0,
            )
            assert curve.maximum_viable_capital() == 500_000.0
            assert curve.maximum_viable_capital(minimum_excess_return=0.02) is None


        @pytest.mark.parametrize(
            "kwargs, message",
            [
                ({"initial_capital": 0.0}, "positive"),
                ({"fill_ratio": 1.1}, "fill_ratio"),
                ({"total_cost_fraction": -0.1}, "non-negative"),
                ({"unfilled_turnover": -0.1}, "non-negative"),
                ({"total_return": math.nan}, "finite"),
            ],
        )
        def test_capacity_point_rejects_invalid_values(
            kwargs: dict[str, float],
            message: str,
        ) -> None:
            values = {
                "initial_capital": 1.0,
                "total_return": 0.0,
                "excess_total_return": 0.0,
                "total_cost_fraction": 0.0,
                "fill_ratio": 1.0,
                "unfilled_turnover": 0.0,
            }
            values.update(kwargs)
            with pytest.raises(ValueError, match=message):
                CapacityPoint(**values)


        def test_capacity_curve_and_grid_validate_order_and_identity() -> None:
            with pytest.raises(ValueError, match="contain points"):
                CapacityCurve(())
            with pytest.raises(ValueError, match="ascending"):
                CapacityCurve((point(2.0), point(1.0)))
            with pytest.raises(ValueError, match="unique"):
                evaluate_capacity_grid([1.0, 1.0], point)
            with pytest.raises(ValueError, match="finite and positive"):
                evaluate_capacity_grid([0.0], point)
            with pytest.raises(ValueError, match="mismatched"):
                evaluate_capacity_grid([1.0], lambda capital: point(capital + 1.0))
        '''
    ).lstrip(),
    encoding="utf-8",
)

Path("tests/rl/test_policy_and_contract_coverage.py").write_text(
    dedent(
        '''
        from __future__ import annotations

        import numpy as np
        import pytest
        import torch
        from gymnasium import spaces

        from trade_rl.rl.experiments import ActionAblation, ActionExperimentSpec
        from trade_rl.rl.normalization import ObservationNormalizer
        from trade_rl.rl.policies import AssetSetFeatureExtractor
        from trade_rl.rl.rewards import RewardConfig, RewardTracker, relative_interval_reward


        def test_asset_set_extractor_is_permutation_invariant_and_handles_no_active_assets() -> None:
            torch.manual_seed(0)
            box = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
            extractor = AssetSetFeatureExtractor(
                box,
                n_symbols=2,
                per_symbol_width=2,
                global_width=3,
                active_column=1,
                asset_embedding_dim=4,
                global_embedding_dim=3,
            )
            extractor.eval()
            first = torch.tensor([[1.0, 1.0, 2.0, 1.0, 0.1, 0.2, 0.3]])
            swapped = torch.tensor([[2.0, 1.0, 1.0, 1.0, 0.1, 0.2, 0.3]])
            with torch.no_grad():
                output = extractor(first)
                permuted = extractor(swapped)
                inactive = extractor(
                    torch.tensor([[1.0, 0.0, 2.0, 0.0, 0.1, 0.2, 0.3]])
                )
            assert output.shape == (1, 7)
            torch.testing.assert_close(output, permuted)
            torch.testing.assert_close(inactive[:, :4], torch.zeros((1, 4)))


        @pytest.mark.parametrize(
            "kwargs, message",
            [
                ({"n_symbols": 0}, "positive"),
                ({"active_column": 2}, "outside"),
                ({"asset_embedding_dim": 0}, "positive"),
            ],
        )
        def test_asset_set_extractor_validates_dimensions(
            kwargs: dict[str, int],
            message: str,
        ) -> None:
            values = {
                "n_symbols": 2,
                "per_symbol_width": 2,
                "global_width": 3,
                "active_column": 1,
                "asset_embedding_dim": 4,
                "global_embedding_dim": 3,
            }
            values.update(kwargs)
            box = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
            with pytest.raises(ValueError, match=message):
                AssetSetFeatureExtractor(box, **values)
            if kwargs == {"n_symbols": 0}:
                wrong = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
                with pytest.raises(ValueError, match="does not match"):
                    AssetSetFeatureExtractor(
                        wrong,
                        n_symbols=2,
                        per_symbol_width=2,
                        global_width=3,
                        active_column=1,
                    )


        def test_action_ablation_contracts_cover_all_variants() -> None:
            baseline = ActionExperimentSpec(ActionAblation.BASELINE_ONLY, 3)
            assert not baseline.policy_enabled
            assert baseline.action_spec.size == 3
            assert not baseline.accept_legacy_actions
            assert baseline.direct_symbol_basis() is None

            legacy = ActionExperimentSpec(ActionAblation.TREND_LEGACY, 3)
            assert legacy.policy_enabled and legacy.accept_legacy_actions
            alpha_legacy = ActionExperimentSpec(ActionAblation.TREND_ALPHA_LEGACY, 3)
            assert alpha_legacy.action_spec.alpha_enabled
            assert alpha_legacy.accept_legacy_actions

            factor4 = ActionExperimentSpec(ActionAblation.FACTORIZED_4, 3)
            factor8 = ActionExperimentSpec(ActionAblation.FACTORIZED_8, 3)
            direct = ActionExperimentSpec(ActionAblation.DIRECT_SYMBOL, 3)
            assert factor4.action_spec.n_factors == 4
            assert factor8.action_spec.n_factors == 8
            assert direct.action_spec.n_factors == 3
            np.testing.assert_array_equal(direct.direct_symbol_basis(), np.eye(3))
            with pytest.raises(ValueError, match="positive"):
                ActionExperimentSpec(ActionAblation.FACTORIZED, 0)


        def test_normalizer_fits_train_only_and_preserves_passthrough() -> None:
            observations = np.array(
                [
                    [1.0, 0.0, 10.0],
                    [3.0, 1.0, 20.0],
                    [1000.0, 0.0, 30.0],
                ],
                dtype=np.float64,
            )
            normalizer = ObservationNormalizer.fit(
                observations,
                train_start=0,
                train_end=2,
                passthrough_indices=(1,),
                dataset_id="a" * 64,
            )
            transformed = normalizer.transform(observations[2])
            batch = normalizer.transform_batch(observations)
            assert normalizer.size == 3
            assert transformed[1] == 0.0
            np.testing.assert_array_equal(batch[:, 1], observations[:, 1])
            assert normalizer.digest

            clone = ObservationNormalizer(
                mean=normalizer.mean,
                scale=normalizer.scale,
                train_start=normalizer.train_start,
                train_end=normalizer.train_end,
                passthrough_indices=normalizer.passthrough_indices,
                dataset_id=normalizer.dataset_id,
                digest=normalizer.digest,
            )
            assert clone.digest == normalizer.digest


        @pytest.mark.parametrize(
            "factory, message",
            [
                (lambda: ObservationNormalizer(np.array([]), np.array([]), 0, 1), "non-empty"),
                (
                    lambda: ObservationNormalizer(
                        np.ones(2), np.ones(3), 0, 1
                    ),
                    "identical",
                ),
                (
                    lambda: ObservationNormalizer(
                        np.ones(2), np.array([1.0, 0.0]), 0, 1
                    ),
                    "positive",
                ),
                (
                    lambda: ObservationNormalizer(
                        np.ones(2), np.ones(2), 1, 1
                    ),
                    "non-empty index range",
                ),
                (
                    lambda: ObservationNormalizer(
                        np.ones(2), np.ones(2), 0, 1, passthrough_indices=(2,)
                    ),
                    "outside",
                ),
            ],
        )
        def test_normalizer_rejects_invalid_contracts(factory: object, message: str) -> None:
            with pytest.raises(ValueError, match=message):
                factory()  # type: ignore[operator]


        def test_normalizer_rejects_bad_fit_and_transform_inputs() -> None:
            with pytest.raises(ValueError, match="two-dimensional"):
                ObservationNormalizer.fit(np.ones(3), train_start=0, train_end=1)
            with pytest.raises(ValueError, match="finite"):
                ObservationNormalizer.fit(
                    np.array([[np.nan]]), train_start=0, train_end=1
                )
            with pytest.raises(ValueError, match="outside"):
                ObservationNormalizer.fit(
                    np.ones((2, 2)), train_start=0, train_end=3
                )
            normalizer = ObservationNormalizer.fit(
                np.ones((2, 2)), train_start=0, train_end=2
            )
            with pytest.raises(ValueError, match="does not match"):
                normalizer.transform(np.ones(3))
            with pytest.raises(ValueError, match="does not match"):
                normalizer.transform_batch(np.ones((2, 3)))
            with pytest.raises(ValueError, match="finite"):
                normalizer.transform_batch(np.array([[np.nan, 1.0]]))


        def test_reward_validation_and_legacy_compatibility_paths() -> None:
            with pytest.raises(ValueError, match="positive"):
                RewardConfig(scale=0.0)
            with pytest.raises(ValueError, match="non-negative"):
                RewardConfig(margin_deficit_weight=-1.0)
            with pytest.raises(ValueError, match="at least one"):
                RewardConfig(baseline_progressive_power=0.5)
            with pytest.raises(ValueError, match="positive"):
                RewardTracker(RewardConfig(), decision_hours=0.0)

            tracker = RewardTracker(RewardConfig())
            with pytest.raises(ValueError, match="non-negative"):
                tracker.step(
                    hybrid_log_return=0.0,
                    shadow_log_return=0.0,
                    hybrid_drawdown=0.0,
                    shadow_drawdown=0.0,
                    hybrid_margin_deficit_fraction=-0.1,
                )
            assert relative_interval_reward(
                hybrid_log_return=0.01,
                shadow_log_return=0.0,
                scale=100.0,
                hybrid_terminated=False,
                shadow_terminated=False,
                hybrid_drawdown=0.0,
                shadow_drawdown=0.0,
            ) == pytest.approx(1.0)
            assert relative_interval_reward(
                hybrid_log_return=0.0,
                shadow_log_return=0.0,
                scale=100.0,
                hybrid_terminated=True,
                shadow_terminated=False,
                hybrid_drawdown=0.0,
                shadow_drawdown=0.0,
            ) == -100.0
            assert relative_interval_reward(
                hybrid_log_return=0.0,
                shadow_log_return=0.0,
                scale=100.0,
                hybrid_terminated=False,
                shadow_terminated=True,
                hybrid_drawdown=0.0,
                shadow_drawdown=0.0,
            ) == 100.0
        '''
    ).lstrip(),
    encoding="utf-8",
)
