from __future__ import annotations

from pathlib import Path
from textwrap import dedent

path = Path("tests/rl/test_training_environment_identity.py")
text = path.read_text(encoding="utf-8")
old = 'with pytest.raises(ValueError, match="initial capital"):'
if old not in text:
    raise RuntimeError("inconsistent AUM expectation was not found")
path.write_text(
    text.replace(
        old,
        'with pytest.raises(ValueError, match=r"initial[_ ]capital"):',
        1,
    ),
    encoding="utf-8",
)

coverage = Path("tests/rl/test_action_diagnostics_coverage.py")
coverage.write_text(
    dedent(
        '''
        from __future__ import annotations

        import numpy as np
        import pytest

        from trade_rl.rl.actions import (
            ActionSpec,
            ActionValidationMode,
            AlphaContract,
            AlphaSignalKind,
            ResidualAction,
        )
        from trade_rl.rl.diagnostics import ActionDiagnosticsAccumulator


        def test_alpha_contract_converts_all_supported_semantics() -> None:
            direction = AlphaContract(kind=AlphaSignalKind.DIRECTION)
            np.testing.assert_array_equal(
                direction.prepare(np.array([-2.0, 0.0, 3.0]), n_symbols=3),
                np.array([-0.5, 0.0, 0.5]),
            )
            confidence = AlphaContract(kind=AlphaSignalKind.DIRECTION_CONFIDENCE)
            prepared = confidence.prepare(
                np.array([-2.0, 0.25, 2.0]),
                n_symbols=3,
            )
            assert np.abs(prepared).sum() == pytest.approx(1.0)
            expected = AlphaContract(
                kind=AlphaSignalKind.EXPECTED_RETURN,
                expected_return_scale=0.02,
            )
            expected_values = expected.prepare(
                np.array([-0.02, 0.0, 0.02]),
                n_symbols=3,
            )
            assert expected_values[0] < 0.0 < expected_values[2]
            target = AlphaContract(kind=AlphaSignalKind.TARGET_WEIGHT, max_gross=0.5)
            assert np.abs(
                target.prepare(np.array([2.0, -1.0]), n_symbols=2)
            ).sum() == pytest.approx(0.5)


        @pytest.mark.parametrize(
            "factory, message",
            [
                (lambda: AlphaContract(kind="unknown"), "not supported"),
                (
                    lambda: AlphaContract(expected_return_scale=0.0),
                    "finite and positive",
                ),
                (lambda: AlphaContract(max_gross=2.0), "within"),
                (
                    lambda: ActionSpec(alpha_enabled=1),
                    "boolean",
                ),
                (lambda: ActionSpec(n_factors=-1), "non-negative"),
                (
                    lambda: ActionSpec(validation_mode="unknown"),
                    "not supported",
                ),
            ],
        )
        def test_action_contracts_reject_invalid_configuration(
            factory: object,
            message: str,
        ) -> None:
            with pytest.raises(ValueError, match=message):
                factory()  # type: ignore[operator]


        def test_alpha_and_action_parsers_reject_bad_vectors_and_report_clipping() -> None:
            contract = AlphaContract()
            with pytest.raises(ValueError, match="does not match"):
                contract.prepare(np.array([1.0]), n_symbols=2)
            with pytest.raises(ValueError, match="does not match"):
                contract.prepare(np.array([1.0, np.nan]), n_symbols=2)

            spec = ActionSpec(alpha_enabled=True, n_factors=1)
            assert spec.names == (
                "fast_tilt",
                "slow_tilt",
                "risk_tilt",
                "alpha_scale",
                "factor_0",
            )
            with pytest.raises(ValueError, match="exactly"):
                spec.parse(np.zeros(2))
            with pytest.raises(ValueError, match="finite"):
                spec.parse(np.array([0.0, 0.0, 0.0, 0.0, np.nan]))
            parsed = spec.parse(np.array([2.0, -2.0, 0.0, 0.5, 1.5]))
            assert parsed.saturated_count == 3
            assert parsed.raw_max_abs == 2.0
            np.testing.assert_array_equal(
                parsed.as_array(alpha_enabled=True),
                np.array([1.0, -1.0, 0.0, 0.5, 1.0], dtype=np.float32),
            )
            with pytest.raises(ValueError, match="outside"):
                spec.parse(
                    np.array([2.0, 0.0, 0.0, 0.0, 0.0]),
                    mode=ActionValidationMode.STRICT,
                )


        def test_legacy_residual_action_validation_and_round_trip() -> None:
            action = ResidualAction.from_array(np.array([2.0, -2.0]))
            np.testing.assert_array_equal(
                action.as_array(),
                np.array([1.0, -1.0], dtype=np.float32),
            )
            with pytest.raises(ValueError, match="exactly"):
                ResidualAction.from_array(np.array([0.0]))
            with pytest.raises(ValueError, match="finite"):
                ResidualAction.from_array(np.array([0.0, np.nan]))
            with pytest.raises(ValueError, match="within"):
                ResidualAction(2.0, 0.0)


        def test_action_diagnostics_accumulates_rates_and_validates_inputs() -> None:
            accumulator = ActionDiagnosticsAccumulator()
            empty = accumulator.snapshot()
            assert empty.saturation_rate == 0.0
            assert empty.constraint_activation_rate == 0.0
            accumulator.update(
                action=np.array([1.0, -0.5, 0.0]),
                saturated_count=1,
                action_delta_l1=0.5,
                projection_l1=0.25,
                constrained=True,
                turnover_overridden=True,
            )
            snapshot = accumulator.snapshot()
            assert snapshot.n_steps == 1
            assert snapshot.n_values == 3
            assert snapshot.saturation_rate == pytest.approx(1.0 / 3.0)
            assert snapshot.constraint_activation_rate == 1.0
            assert snapshot.turnover_override_steps == 1
            assert snapshot.mean_abs_action == pytest.approx(0.5)
            assert snapshot.mean_action_delta_l1 == pytest.approx(0.5)
            assert snapshot.mean_projection_l1 == pytest.approx(0.25)
            assert snapshot.maximum_abs_action == 1.0

            with pytest.raises(ValueError, match="finite non-empty"):
                accumulator.update(
                    action=np.array([]),
                    saturated_count=0,
                    action_delta_l1=0.0,
                    projection_l1=0.0,
                    constrained=False,
                    turnover_overridden=False,
                )
            with pytest.raises(ValueError, match="non-negative"):
                accumulator.update(
                    action=np.array([0.0]),
                    saturated_count=0,
                    action_delta_l1=-1.0,
                    projection_l1=0.0,
                    constrained=False,
                    turnover_overridden=False,
                )
            with pytest.raises(ValueError, match="outside"):
                accumulator.update(
                    action=np.array([0.0]),
                    saturated_count=2,
                    action_delta_l1=0.0,
                    projection_l1=0.0,
                    constrained=False,
                    turnover_overridden=False,
                )
            accumulator.reset()
            assert accumulator.snapshot().n_steps == 0
        '''
    ).lstrip(),
    encoding="utf-8",
)
