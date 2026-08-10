#!/usr/bin/env python3
"""Apply the execution-environment robustness contract with RED/GREEN verification."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/apply-execution-robustness.yml"
SCRIPT = Path(__file__).resolve()


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one marker, found {count}")
    return text.replace(old, new, 1)


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def add_red_tests() -> None:
    write(
        "tests/simulation/test_execution_environment_stress.py",
        '''
        from __future__ import annotations

        import pytest

        from trade_rl.simulation.execution import ExecutionCostConfig
        from trade_rl.simulation.execution_stress import ExecutionEnvironmentStress


        def _base_cost() -> ExecutionCostConfig:
            return ExecutionCostConfig(
                fee_rate=0.001,
                maker_fee_rate=0.0002,
                taker_fee_rate=0.0003,
                spread_rate=0.0004,
                impact_rate=0.0005,
                max_participation_rate=0.2,
                slippage_std=0.0006,
                tail_slippage_probability=0.01,
                tail_slippage_multiplier=4.0,
                borrow_rate_multiplier=1.5,
                order_latency_bars=1,
            )


        def test_environment_stress_applies_all_dimensions_without_mutating_base() -> None:
            base = _base_cost()
            stress = ExecutionEnvironmentStress(
                name="joint-adverse",
                fee_multiplier=2.0,
                spread_multiplier=3.0,
                impact_multiplier=4.0,
                slippage_std_multiplier=5.0,
                participation_fraction=0.25,
                minimum_order_latency_bars=3,
                tail_slippage_probability_floor=0.05,
                tail_slippage_multiplier_floor=8.0,
                borrow_rate_multiplier=2.0,
            )

            stressed = stress.apply(base)

            assert stressed is not base
            assert base.fee_rate == pytest.approx(0.001)
            assert stressed.fee_rate == pytest.approx(0.002)
            assert stressed.maker_fee_rate == pytest.approx(0.0004)
            assert stressed.taker_fee_rate == pytest.approx(0.0006)
            assert stressed.spread_rate == pytest.approx(0.0012)
            assert stressed.impact_rate == pytest.approx(0.002)
            assert stressed.slippage_std == pytest.approx(0.003)
            assert stressed.max_participation_rate == pytest.approx(0.05)
            assert stressed.order_latency_bars == 3
            assert stressed.tail_slippage_probability == pytest.approx(0.05)
            assert stressed.tail_slippage_multiplier == pytest.approx(8.0)
            assert stressed.borrow_rate_multiplier == pytest.approx(3.0)
            assert stress.digest_payload()["schema_version"] == (
                "execution_environment_stress_v1"
            )


        def test_neutral_environment_stress_preserves_cost_identity() -> None:
            base = _base_cost()

            assert ExecutionEnvironmentStress(name="neutral").apply(base) is base


        @pytest.mark.parametrize(
            ("field", "value"),
            [
                ("fee_multiplier", 0.5),
                ("spread_multiplier", float("inf")),
                ("impact_multiplier", -1.0),
                ("slippage_std_multiplier", 0.0),
                ("participation_fraction", 0.0),
                ("participation_fraction", 1.1),
                ("minimum_order_latency_bars", -1),
                ("tail_slippage_probability_floor", 1.1),
                ("tail_slippage_multiplier_floor", -1.0),
                ("borrow_rate_multiplier", 0.9),
            ],
        )
        def test_environment_stress_rejects_non_adverse_or_invalid_values(
            field: str,
            value: object,
        ) -> None:
            with pytest.raises(ValueError):
                ExecutionEnvironmentStress(name="invalid", **{field: value})
        ''',
    )

    write(
        "tests/workflows/test_execution_robustness_config.py",
        '''
        from __future__ import annotations

        from dataclasses import fields

        import pytest

        from trade_rl.simulation.execution_stress import ExecutionEnvironmentStress
        from trade_rl.workflows.market_walk_forward_config import (
            ExecutionSensitivityConfig,
            ExecutionSensitivityScenario,
        )


        def test_execution_sensitivity_scenario_declares_environment_cost_stress_fields() -> (
            None
        ):
            field_names = {field.name for field in fields(ExecutionSensitivityScenario)}

            assert {
                "fee_multiplier",
                "spread_multiplier",
                "impact_multiplier",
                "slippage_std_multiplier",
                "participation_fraction",
                "minimum_order_latency_bars",
                "tail_slippage_probability_floor",
                "tail_slippage_multiplier_floor",
                "borrow_rate_multiplier",
            } <= field_names


        def test_scenario_stress_and_digest_include_environment_dimensions() -> None:
            scenario = ExecutionSensitivityScenario(
                name="fee-spread-2x",
                adverse_tick_rounding=False,
                report_only=True,
                fee_multiplier=2.0,
                spread_multiplier=2.0,
            )

            stress = scenario.stress()
            payload = scenario.digest_payload()

            assert isinstance(stress, ExecutionEnvironmentStress)
            assert stress.fee_multiplier == 2.0
            assert stress.spread_multiplier == 2.0
            assert payload["fee_multiplier"] == 2.0
            assert payload["spread_multiplier"] == 2.0
            assert payload["report_only"] is True
            assert payload["schema_version"] == "execution_environment_stress_v1"


        def test_standard_gate_scenarios_cannot_hide_environment_cost_stress() -> None:
            scenarios = (
                ExecutionSensitivityScenario(name="nominal", adverse_tick_rounding=False),
                ExecutionSensitivityScenario(
                    name="tick_2x",
                    tick_size_factor=2.0,
                    adverse_tick_rounding=True,
                ),
                ExecutionSensitivityScenario(
                    name="lot_2x",
                    lot_size_factor=2.0,
                    adverse_tick_rounding=True,
                ),
                ExecutionSensitivityScenario(
                    name="minimum_notional_2x",
                    minimum_notional_factor=2.0,
                    adverse_tick_rounding=True,
                ),
                ExecutionSensitivityScenario(
                    name="joint_2x",
                    tick_size_factor=2.0,
                    lot_size_factor=2.0,
                    minimum_notional_factor=2.0,
                    adverse_tick_rounding=True,
                    fee_multiplier=2.0,
                ),
                ExecutionSensitivityScenario(
                    name="joint_5x",
                    tick_size_factor=5.0,
                    lot_size_factor=5.0,
                    minimum_notional_factor=5.0,
                    adverse_tick_rounding=True,
                    report_only=True,
                ),
            )

            with pytest.raises(
                ValueError,
                match="standard execution sensitivity scenarios",
            ):
                ExecutionSensitivityConfig(
                    scenarios=scenarios,
                    required_scenario="joint_2x",
                )
        ''',
    )

    write(
        "tests/examples/test_execution_robustness_profile.py",
        '''
        from __future__ import annotations

        from pathlib import Path

        from trade_rl.workflows.market_walk_forward_config import MarketWalkForwardConfig


        ROOT = Path(__file__).resolve().parents[2]
        PROFILE = (
            ROOT
            / "examples"
            / "binance-multitimeframe"
            / "walk-forward-target-weight-execution-robustness.json"
        )


        def test_execution_robustness_profile_is_report_only_extension() -> None:
            config = MarketWalkForwardConfig.from_json(PROFILE, n_bars=40_000)
            scenarios = {
                scenario.name: scenario
                for scenario in config.execution_sensitivity.scenarios
            }

            assert config.execution_sensitivity.required_scenario == "joint_2x"
            assert {
                "fee_spread_2x",
                "impact_2x",
                "slippage_2x",
                "capacity_50pct",
                "latency_1bar",
                "tail_slippage",
                "borrow_2x",
                "joint_adverse",
            } <= scenarios.keys()
            for name, scenario in scenarios.items():
                if name not in {
                    "nominal",
                    "tick_2x",
                    "lot_2x",
                    "minimum_notional_2x",
                    "joint_2x",
                    "joint_5x",
                }:
                    assert scenario.report_only is True

            joint = scenarios["joint_adverse"]
            stress = joint.stress()
            assert stress.fee_multiplier == 2.0
            assert stress.spread_multiplier == 2.0
            assert stress.impact_multiplier == 2.0
            assert stress.participation_fraction == 0.5
            assert stress.minimum_order_latency_bars == 1
            assert stress.tail_slippage_probability_floor == 0.01
            assert stress.tail_slippage_multiplier_floor == 10.0
            assert stress.borrow_rate_multiplier == 2.0

            payloads = {
                item["name"]: item
                for item in config.execution_sensitivity.digest_payload()["scenarios"]
            }
            assert payloads["joint_adverse"]["fee_multiplier"] == 2.0
            assert payloads["joint_adverse"]["report_only"] is True
        ''',
    )

    resource_test = ROOT / "tests/rl/test_environment_reward_execution_resources.py"
    text = resource_test.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from __future__ import annotations\n\n",
        "from __future__ import annotations\n\nfrom dataclasses import replace\n\n",
        label="resource test dataclasses import",
    )
    text = replace_once(
        text,
        "from trade_rl.simulation.execution import (\n"
        "    ExecutionCostConfig,\n"
        "    ExecutionRuleStress,\n"
        ")\n",
        "from trade_rl.simulation.execution import (\n"
        "    ExecutionCostConfig,\n"
        "    ExecutionRuleStress,\n"
        ")\n"
        "from trade_rl.simulation.execution_stress import ExecutionEnvironmentStress\n",
        label="resource test stress import",
    )
    text += textwrap.dedent(
        '''


        def test_builder_applies_identical_environment_stress_to_both_books() -> None:
            dataset = market()
            resolved_reward = reward_config(baseline_weight=0.0)
            base_cost = ExecutionCostConfig(
                fee_rate=0.001,
                spread_rate=0.002,
                impact_rate=0.003,
                max_participation_rate=0.2,
                slippage_std=0.004,
                tail_slippage_probability=0.0,
                tail_slippage_multiplier=5.0,
                borrow_rate_multiplier=1.5,
                order_latency_bars=0,
            )
            resolved_config = replace(
                config(reward=resolved_reward),
                execution_cost=base_cost,
            )
            stress = ExecutionEnvironmentStress(
                name="joint-adverse",
                fee_multiplier=2.0,
                spread_multiplier=2.0,
                impact_multiplier=2.0,
                slippage_std_multiplier=2.0,
                participation_fraction=0.5,
                minimum_order_latency_bars=2,
                tail_slippage_probability_floor=0.01,
                tail_slippage_multiplier_floor=10.0,
                borrow_rate_multiplier=2.0,
            )

            resources = EnvironmentRewardExecutionResourcesBuilder(
                dataset,
                config=resolved_config,
                reward_config=resolved_reward,
                resolved_decision_hours=2.0,
                minimum_start_index=8,
                execution_rule_stress=stress,
            ).build()

            assert resources.hybrid_executor.cost is resources.shadow_executor.cost
            assert resources.hybrid_executor.cost is not base_cost
            assert resources.hybrid_executor.cost.fee_rate == pytest.approx(0.002)
            assert resources.hybrid_executor.cost.spread_rate == pytest.approx(0.004)
            assert resources.hybrid_executor.cost.impact_rate == pytest.approx(0.006)
            assert resources.hybrid_executor.cost.max_participation_rate == pytest.approx(
                0.1
            )
            assert resources.hybrid_executor.cost.order_latency_bars == 2
            assert resources.hybrid_executor.cost.tail_slippage_probability == (
                pytest.approx(0.01)
            )
            assert resources.hybrid_executor.cost.tail_slippage_multiplier == (
                pytest.approx(10.0)
            )
            assert resources.hybrid_executor.rule_stress is stress
            assert resources.shadow_executor.rule_stress is stress
            assert resources.hybrid_executor.execution_policy_digest == (
                resources.shadow_executor.execution_policy_digest
            )
        '''
    )
    resource_test.write_text(text, encoding="utf-8")


def verify_red() -> None:
    command = [
        "uv",
        "run",
        "pytest",
        "-q",
        "tests/simulation/test_execution_environment_stress.py",
    ]
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout)
    if completed.returncode == 0:
        raise RuntimeError("execution robustness RED contract unexpectedly passed")
    if "trade_rl.simulation.execution_stress" not in completed.stdout:
        raise RuntimeError("execution robustness RED contract failed unexpectedly")


def implement_stress_contract() -> None:
    write(
        "trade_rl/simulation/execution_stress.py",
        '''
        """Immutable evaluation-only overlays for adverse execution environments."""

        from __future__ import annotations

        import math
        from dataclasses import dataclass, replace

        from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionRuleStress


        @dataclass(frozen=True, slots=True)
        class ExecutionEnvironmentStress(ExecutionRuleStress):
            """Rule stress plus deterministic adverse execution-cost assumptions."""

            fee_multiplier: float = 1.0
            spread_multiplier: float = 1.0
            impact_multiplier: float = 1.0
            slippage_std_multiplier: float = 1.0
            participation_fraction: float = 1.0
            minimum_order_latency_bars: int = 0
            tail_slippage_probability_floor: float = 0.0
            tail_slippage_multiplier_floor: float = 0.0
            borrow_rate_multiplier: float = 1.0

            def __post_init__(self) -> None:
                ExecutionRuleStress.__post_init__(self)
                for field_name, value in (
                    ("fee_multiplier", self.fee_multiplier),
                    ("spread_multiplier", self.spread_multiplier),
                    ("impact_multiplier", self.impact_multiplier),
                    ("slippage_std_multiplier", self.slippage_std_multiplier),
                    ("borrow_rate_multiplier", self.borrow_rate_multiplier),
                ):
                    if not math.isfinite(value) or value < 1.0:
                        raise ValueError(
                            f"{field_name} must be finite and at least 1.0"
                        )
                if (
                    not math.isfinite(self.participation_fraction)
                    or not 0.0 < self.participation_fraction <= 1.0
                ):
                    raise ValueError(
                        "participation_fraction must be finite and within (0, 1]"
                    )
                if (
                    isinstance(self.minimum_order_latency_bars, bool)
                    or not isinstance(self.minimum_order_latency_bars, int)
                    or self.minimum_order_latency_bars < 0
                ):
                    raise ValueError(
                        "minimum_order_latency_bars must be a non-negative integer"
                    )
                if (
                    not math.isfinite(self.tail_slippage_probability_floor)
                    or not 0.0 <= self.tail_slippage_probability_floor <= 1.0
                ):
                    raise ValueError(
                        "tail_slippage_probability_floor must be within [0, 1]"
                    )
                if (
                    not math.isfinite(self.tail_slippage_multiplier_floor)
                    or self.tail_slippage_multiplier_floor < 0.0
                ):
                    raise ValueError(
                        "tail_slippage_multiplier_floor must be finite and non-negative"
                    )

            @property
            def environment_enabled(self) -> bool:
                return (
                    self.fee_multiplier > 1.0
                    or self.spread_multiplier > 1.0
                    or self.impact_multiplier > 1.0
                    or self.slippage_std_multiplier > 1.0
                    or self.participation_fraction < 1.0
                    or self.minimum_order_latency_bars > 0
                    or self.tail_slippage_probability_floor > 0.0
                    or self.tail_slippage_multiplier_floor > 0.0
                    or self.borrow_rate_multiplier > 1.0
                )

            @property
            def enabled(self) -> bool:
                return ExecutionRuleStress.enabled.fget(self) or self.environment_enabled

            def apply(self, base: ExecutionCostConfig) -> ExecutionCostConfig:
                """Return one validated stressed cost while leaving ``base`` immutable."""

                if not self.environment_enabled:
                    return base
                tail_probability = max(
                    base.tail_slippage_probability,
                    self.tail_slippage_probability_floor,
                )
                tail_multiplier = max(
                    base.tail_slippage_multiplier,
                    self.tail_slippage_multiplier_floor,
                    1.0 if tail_probability > 0.0 else 0.0,
                )
                return replace(
                    base,
                    fee_rate=base.fee_rate * self.fee_multiplier,
                    maker_fee_rate=base.maker_fee_rate * self.fee_multiplier,
                    taker_fee_rate=base.taker_fee_rate * self.fee_multiplier,
                    spread_rate=base.spread_rate * self.spread_multiplier,
                    impact_rate=base.impact_rate * self.impact_multiplier,
                    slippage_std=(
                        base.slippage_std * self.slippage_std_multiplier
                    ),
                    max_participation_rate=(
                        base.max_participation_rate * self.participation_fraction
                    ),
                    order_latency_bars=max(
                        base.order_latency_bars,
                        self.minimum_order_latency_bars,
                    ),
                    tail_slippage_probability=tail_probability,
                    tail_slippage_multiplier=tail_multiplier,
                    borrow_rate_multiplier=(
                        base.borrow_rate_multiplier * self.borrow_rate_multiplier
                    ),
                )

            def digest_payload(self) -> dict[str, object]:
                return {
                    **ExecutionRuleStress.digest_payload(self),
                    "borrow_rate_multiplier": self.borrow_rate_multiplier,
                    "fee_multiplier": self.fee_multiplier,
                    "impact_multiplier": self.impact_multiplier,
                    "minimum_order_latency_bars": self.minimum_order_latency_bars,
                    "participation_fraction": self.participation_fraction,
                    "schema_version": "execution_environment_stress_v1",
                    "slippage_std_multiplier": self.slippage_std_multiplier,
                    "spread_multiplier": self.spread_multiplier,
                    "tail_slippage_multiplier_floor": (
                        self.tail_slippage_multiplier_floor
                    ),
                    "tail_slippage_probability_floor": (
                        self.tail_slippage_probability_floor
                    ),
                }


        __all__ = ["ExecutionEnvironmentStress"]
        ''',
    )

    init_path = ROOT / "trade_rl/simulation/__init__.py"
    init_text = init_path.read_text(encoding="utf-8")
    init_text = replace_once(
        init_text,
        "from trade_rl.simulation.execution import (\n"
        "    ExecutionCostConfig,\n"
        "    ExecutionResult,\n"
        "    MarketExecutor,\n"
        ")\n",
        "from trade_rl.simulation.execution import (\n"
        "    ExecutionCostConfig,\n"
        "    ExecutionResult,\n"
        "    MarketExecutor,\n"
        ")\n"
        "from trade_rl.simulation.execution_stress import ExecutionEnvironmentStress\n",
        label="simulation stress export import",
    )
    init_text = replace_once(
        init_text,
        '    "ExecutionCostConfig",\n',
        '    "ExecutionCostConfig",\n    "ExecutionEnvironmentStress",\n',
        label="simulation stress export",
    )
    init_path.write_text(init_text, encoding="utf-8")


def patch_walk_forward_config() -> None:
    path = ROOT / "trade_rl/workflows/market_walk_forward_config.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "import trade_rl.workflows._market_walk_forward_config_base as _base\n",
        "import trade_rl.workflows._market_walk_forward_config_base as _base\n"
        "from trade_rl.simulation.execution_stress import ExecutionEnvironmentStress\n",
        label="walk-forward stress import",
    )
    old_class_start = text.index("@dataclass(frozen=True, slots=True)\nclass ExecutionSensitivityScenario")
    old_class_end = text.index("\n\nNamedCandidateRun =", old_class_start)
    new_class = textwrap.dedent(
        '''
        @dataclass(frozen=True, slots=True)
        class ExecutionSensitivityScenario(ExecutionEnvironmentStress):
            """One reportable rule and execution-environment stress scenario."""

            report_only: bool = False

            def __post_init__(self) -> None:
                ExecutionEnvironmentStress.__post_init__(self)
                if not isinstance(self.report_only, bool):
                    raise ValueError(
                        "execution sensitivity report_only must be a boolean"
                    )

            def stress(self) -> ExecutionEnvironmentStress:
                return ExecutionEnvironmentStress(
                    name=self.name,
                    tick_size_factor=self.tick_size_factor,
                    lot_size_factor=self.lot_size_factor,
                    minimum_notional_factor=self.minimum_notional_factor,
                    adverse_tick_rounding=self.adverse_tick_rounding,
                    fee_multiplier=self.fee_multiplier,
                    spread_multiplier=self.spread_multiplier,
                    impact_multiplier=self.impact_multiplier,
                    slippage_std_multiplier=self.slippage_std_multiplier,
                    participation_fraction=self.participation_fraction,
                    minimum_order_latency_bars=self.minimum_order_latency_bars,
                    tail_slippage_probability_floor=(
                        self.tail_slippage_probability_floor
                    ),
                    tail_slippage_multiplier_floor=(
                        self.tail_slippage_multiplier_floor
                    ),
                    borrow_rate_multiplier=self.borrow_rate_multiplier,
                )

            def digest_payload(self) -> dict[str, object]:
                return {
                    **self.stress().digest_payload(),
                    "report_only": self.report_only,
                }
        '''
    ).strip()
    text = text[:old_class_start] + new_class + text[old_class_end:]

    config_class_start = text.index(
        "@dataclass(frozen=True, slots=True)\nclass ExecutionSensitivityConfig"
    )
    config_class_end = text.index("\n\ndef _absolute_config_path", config_class_start)
    config_class = textwrap.dedent(
        '''
        @dataclass(frozen=True, slots=True)
        class ExecutionSensitivityConfig(_base.ExecutionSensitivityConfig):
            """Canonical rule pack plus optional report-only environment stress."""

            def __post_init__(self) -> None:
                names = tuple(item.name for item in self.scenarios)
                if len(set(names)) != len(names):
                    raise ValueError(
                        "execution sensitivity scenario names must be unique"
                    )
                standard = tuple(
                    item
                    for item in self.scenarios
                    if item.name in _STANDARD_EXECUTION_SCENARIOS
                )
                if self.scenarios and not standard:
                    raise ValueError(
                        "execution sensitivity extensions require the standard scenario pack"
                    )
                _base.ExecutionSensitivityConfig(
                    scenarios=standard,
                    required_scenario=self.required_scenario,
                    minimum_selected_return=self.minimum_selected_return,
                    minimum_baseline_uplift=self.minimum_baseline_uplift,
                    maximum_drawdown=self.maximum_drawdown,
                    schema_version=self.schema_version,
                )
                for scenario in standard:
                    if (
                        isinstance(scenario, ExecutionEnvironmentStress)
                        and scenario.environment_enabled
                    ):
                        raise ValueError(
                            "standard execution sensitivity scenarios cannot change "
                            "execution-environment costs"
                        )
                for scenario in self.scenarios:
                    if (
                        scenario.name not in _STANDARD_EXECUTION_SCENARIOS
                        and not scenario.report_only
                    ):
                        raise ValueError(
                            "additional execution sensitivity scenarios must be report-only"
                        )
                if self.scenarios:
                    if self.required_scenario not in names:
                        raise ValueError(
                            "required execution sensitivity scenario is missing"
                        )
                    required = next(
                        item
                        for item in self.scenarios
                        if item.name == self.required_scenario
                    )
                    if required.report_only:
                        raise ValueError(
                            "required execution sensitivity scenario cannot be report-only"
                        )
        '''
    ).strip()
    text = text[:config_class_start] + config_class + text[config_class_end:]

    function_start = text.index("def _extended_execution_sensitivity(")
    function_end = text.index("\n\nclass SealedTestLedgerMode", function_start)
    function = textwrap.dedent(
        '''
        def _non_negative_integer(
            value: object,
            *,
            field: str,
            default: int,
        ) -> int:
            if value is None:
                return default
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
            return value


        def _scenario_from_base(
            scenario: _base.ExecutionSensitivityScenario,
        ) -> ExecutionSensitivityScenario:
            return ExecutionSensitivityScenario(
                name=scenario.name,
                tick_size_factor=scenario.tick_size_factor,
                lot_size_factor=scenario.lot_size_factor,
                minimum_notional_factor=scenario.minimum_notional_factor,
                adverse_tick_rounding=scenario.adverse_tick_rounding,
                report_only=scenario.report_only,
            )


        def _extended_execution_sensitivity(
            expanded: dict[str, Any],
            base: _base.ExecutionSensitivityConfig,
        ) -> ExecutionSensitivityConfig:
            sensitivity = expanded.get("execution_sensitivity")
            if not isinstance(sensitivity, dict):
                return ExecutionSensitivityConfig(
                    scenarios=tuple(
                        _scenario_from_base(scenario)
                        for scenario in base.scenarios
                    ),
                    required_scenario=base.required_scenario,
                    minimum_selected_return=base.minimum_selected_return,
                    minimum_baseline_uplift=base.minimum_baseline_uplift,
                    maximum_drawdown=base.maximum_drawdown,
                    schema_version=base.schema_version,
                )
            raw_scenarios = sensitivity.get("scenarios")
            if not isinstance(raw_scenarios, list):
                raise ValueError("execution_sensitivity.scenarios must be a list")
            base_by_name = {scenario.name: scenario for scenario in base.scenarios}
            scenarios: list[ExecutionSensitivityScenario] = []
            for index, raw_scenario in enumerate(raw_scenarios):
                if not isinstance(raw_scenario, dict):
                    raise ValueError(
                        f"execution_sensitivity.scenarios[{index}] must be a JSON object"
                    )
                name = raw_scenario.get("name")
                if not isinstance(name, str):
                    raise ValueError(
                        f"execution_sensitivity.scenarios[{index}].name must be a string"
                    )
                field = f"execution_sensitivity.scenarios[{index}]"
                canonical = base_by_name.get(name)
                scenarios.append(
                    ExecutionSensitivityScenario(
                        name=name,
                        tick_size_factor=(
                            canonical.tick_size_factor
                            if canonical is not None
                            else _finite_number(
                                raw_scenario.get("tick_size_factor"),
                                field=f"{field}.tick_size_factor",
                                default=1.0,
                            )
                        ),
                        lot_size_factor=(
                            canonical.lot_size_factor
                            if canonical is not None
                            else _finite_number(
                                raw_scenario.get("lot_size_factor"),
                                field=f"{field}.lot_size_factor",
                                default=1.0,
                            )
                        ),
                        minimum_notional_factor=(
                            canonical.minimum_notional_factor
                            if canonical is not None
                            else _finite_number(
                                raw_scenario.get("minimum_notional_factor"),
                                field=f"{field}.minimum_notional_factor",
                                default=1.0,
                            )
                        ),
                        adverse_tick_rounding=(
                            canonical.adverse_tick_rounding
                            if canonical is not None
                            else _boolean(
                                raw_scenario.get("adverse_tick_rounding"),
                                field=f"{field}.adverse_tick_rounding",
                                default=True,
                            )
                        ),
                        report_only=(
                            canonical.report_only
                            if canonical is not None
                            else _boolean(
                                raw_scenario.get("report_only"),
                                field=f"{field}.report_only",
                                default=False,
                            )
                        ),
                        fee_multiplier=_finite_number(
                            raw_scenario.get("fee_multiplier"),
                            field=f"{field}.fee_multiplier",
                            default=1.0,
                        ),
                        spread_multiplier=_finite_number(
                            raw_scenario.get("spread_multiplier"),
                            field=f"{field}.spread_multiplier",
                            default=1.0,
                        ),
                        impact_multiplier=_finite_number(
                            raw_scenario.get("impact_multiplier"),
                            field=f"{field}.impact_multiplier",
                            default=1.0,
                        ),
                        slippage_std_multiplier=_finite_number(
                            raw_scenario.get("slippage_std_multiplier"),
                            field=f"{field}.slippage_std_multiplier",
                            default=1.0,
                        ),
                        participation_fraction=_finite_number(
                            raw_scenario.get("participation_fraction"),
                            field=f"{field}.participation_fraction",
                            default=1.0,
                        ),
                        minimum_order_latency_bars=_non_negative_integer(
                            raw_scenario.get("minimum_order_latency_bars"),
                            field=f"{field}.minimum_order_latency_bars",
                            default=0,
                        ),
                        tail_slippage_probability_floor=_finite_number(
                            raw_scenario.get("tail_slippage_probability_floor"),
                            field=f"{field}.tail_slippage_probability_floor",
                            default=0.0,
                        ),
                        tail_slippage_multiplier_floor=_finite_number(
                            raw_scenario.get("tail_slippage_multiplier_floor"),
                            field=f"{field}.tail_slippage_multiplier_floor",
                            default=0.0,
                        ),
                        borrow_rate_multiplier=_finite_number(
                            raw_scenario.get("borrow_rate_multiplier"),
                            field=f"{field}.borrow_rate_multiplier",
                            default=1.0,
                        ),
                    )
                )
            return ExecutionSensitivityConfig(
                scenarios=tuple(scenarios),
                required_scenario=base.required_scenario,
                minimum_selected_return=base.minimum_selected_return,
                minimum_baseline_uplift=base.minimum_baseline_uplift,
                maximum_drawdown=base.maximum_drawdown,
                schema_version=base.schema_version,
            )
        '''
    ).strip()
    text = text[:function_start] + function + text[function_end:]
    path.write_text(text, encoding="utf-8")


def patch_environment_resources() -> None:
    path = ROOT / "trade_rl/rl/environment_reward_execution_resources.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from trade_rl.simulation.execution import ExecutionRuleStress\n",
        "from trade_rl.simulation.execution import ExecutionRuleStress\n"
        "from trade_rl.simulation.execution_stress import ExecutionEnvironmentStress\n",
        label="environment resource stress import",
    )
    marker = (
        "        hybrid_executor = MarketExecutor(\n"
        "            self.dataset,\n"
        "            self.config.execution_cost,\n"
        "            rule_stress=self.execution_rule_stress,\n"
        "        )\n"
        "        shadow_executor = MarketExecutor(\n"
        "            self.dataset,\n"
        "            self.config.execution_cost,\n"
        "            rule_stress=self.execution_rule_stress,\n"
        "        )\n"
    )
    replacement = (
        "        execution_cost = self.config.execution_cost\n"
        "        if isinstance(self.execution_rule_stress, ExecutionEnvironmentStress):\n"
        "            execution_cost = self.execution_rule_stress.apply(execution_cost)\n"
        "        hybrid_executor = MarketExecutor(\n"
        "            self.dataset,\n"
        "            execution_cost,\n"
        "            rule_stress=self.execution_rule_stress,\n"
        "        )\n"
        "        shadow_executor = MarketExecutor(\n"
        "            self.dataset,\n"
        "            execution_cost,\n"
        "            rule_stress=self.execution_rule_stress,\n"
        "        )\n"
    )
    text = replace_once(text, marker, replacement, label="environment resource executors")
    path.write_text(text, encoding="utf-8")


def create_profile_and_docs() -> None:
    base_path = (
        ROOT
        / "examples/binance-multitimeframe/"
        "walk-forward-target-weight-constrained-growth.json"
    )
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    sensitivity = payload["execution_sensitivity"]
    sensitivity["scenarios"].extend(
        [
            {
                "name": "fee_spread_2x",
                "adverse_tick_rounding": False,
                "report_only": True,
                "fee_multiplier": 2.0,
                "spread_multiplier": 2.0,
            },
            {
                "name": "impact_2x",
                "adverse_tick_rounding": False,
                "report_only": True,
                "impact_multiplier": 2.0,
            },
            {
                "name": "slippage_2x",
                "adverse_tick_rounding": False,
                "report_only": True,
                "slippage_std_multiplier": 2.0,
            },
            {
                "name": "capacity_50pct",
                "adverse_tick_rounding": False,
                "report_only": True,
                "participation_fraction": 0.5,
            },
            {
                "name": "latency_1bar",
                "adverse_tick_rounding": False,
                "report_only": True,
                "minimum_order_latency_bars": 1,
            },
            {
                "name": "tail_slippage",
                "adverse_tick_rounding": False,
                "report_only": True,
                "tail_slippage_probability_floor": 0.01,
                "tail_slippage_multiplier_floor": 10.0,
            },
            {
                "name": "borrow_2x",
                "adverse_tick_rounding": False,
                "report_only": True,
                "borrow_rate_multiplier": 2.0,
            },
            {
                "name": "joint_adverse",
                "adverse_tick_rounding": False,
                "report_only": True,
                "fee_multiplier": 2.0,
                "spread_multiplier": 2.0,
                "impact_multiplier": 2.0,
                "slippage_std_multiplier": 2.0,
                "participation_fraction": 0.5,
                "minimum_order_latency_bars": 1,
                "tail_slippage_probability_floor": 0.01,
                "tail_slippage_multiplier_floor": 10.0,
                "borrow_rate_multiplier": 2.0,
            },
        ]
    )
    target = (
        ROOT
        / "examples/binance-multitimeframe/"
        "walk-forward-target-weight-execution-robustness.json"
    )
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    write(
        "docs/EXECUTION_ROBUSTNESS.md",
        '''
        # Execution robustness evidence

        The maintained execution-robustness workflow extends the existing sealed
        walk-forward rule-rounding matrix with deterministic adverse assumptions for
        fees, spread, market impact, stochastic slippage scale, participation
        capacity, order latency, tail slippage, and borrow cost.

        ## Contract

        `ExecutionEnvironmentStress` is an immutable simulation-layer overlay. It
        transforms one immutable `ExecutionCostConfig` before either evaluation book
        is constructed. The hybrid selected-policy executor and independent shadow
        baseline executor receive the same transformed cost object and the same rule
        stress, preventing asymmetric evidence.

        The overlay is bound into each scenario digest and therefore into the
        walk-forward experiment-plan identity. Unknown or invalid values fail closed.
        Multipliers are adverse-only (`>= 1`), participation can only decrease, and
        latency/tail fields are floors rather than replacements.

        ## Maintained profile

        `walk-forward-target-weight-execution-robustness.json` retains the current
        target-weight constrained-growth candidate set and the existing mandatory
        `joint_2x` exchange-rule gate. It adds report-only scenarios for:

        - two-times fees and spread;
        - two-times impact and slippage standard deviation;
        - fifty-percent participation capacity;
        - at least one bar of order latency;
        - one-percent, ten-times tail-slippage floors;
        - two-times borrow cost;
        - one joint adverse environment combining all dimensions.

        Additional execution-environment scenarios remain report-only under
        `execution_sensitivity_config_v1`. They do not replace or weaken the required
        `joint_2x` gate and are not selected as the default full-research workflow.

        ## Safety status

        This evidence is evaluation-only. It does not change training rewards, PPO or
        BC objectives, checkpoint formats, serving, risk limits, sealed-test access,
        or live-order behavior. Production remains **NO-GO**.
        ''',
    )


def verify_green() -> None:
    modified = [
        "trade_rl/simulation/execution_stress.py",
        "trade_rl/simulation/__init__.py",
        "trade_rl/workflows/market_walk_forward_config.py",
        "trade_rl/rl/environment_reward_execution_resources.py",
        "tests/simulation/test_execution_environment_stress.py",
        "tests/workflows/test_execution_robustness_config.py",
        "tests/rl/test_environment_reward_execution_resources.py",
        "tests/examples/test_execution_robustness_profile.py",
    ]
    run("uv", "run", "ruff", "format", *modified)
    run(
        "uv",
        "run",
        "pytest",
        "-q",
        "tests/simulation/test_execution_environment_stress.py",
        "tests/workflows/test_execution_robustness_config.py",
        "tests/rl/test_environment_reward_execution_resources.py",
        "tests/examples/test_execution_robustness_profile.py",
        "tests/simulation/test_execution_sensitivity.py",
        "tests/evaluation/test_execution_sensitivity_matrix.py",
        "tests/workflows/test_market_walk_forward.py",
        "tests/examples/test_target_weight_constrained_growth_profiles.py",
    )
    run("uv", "run", "ruff", "check", *modified)
    run("uv", "run", "ruff", "format", "--check", *modified)
    run(
        "uv",
        "run",
        "mypy",
        "trade_rl/simulation/execution_stress.py",
        "trade_rl/workflows/market_walk_forward_config.py",
        "trade_rl/rl/environment_reward_execution_resources.py",
    )


def main() -> None:
    add_red_tests()
    verify_red()
    implement_stress_contract()
    patch_walk_forward_config()
    patch_environment_resources()
    create_profile_and_docs()
    verify_green()
    WORKFLOW.unlink()
    SCRIPT.unlink()


if __name__ == "__main__":
    main()
