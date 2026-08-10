#!/usr/bin/env python3
"""Apply the final pinned execution-stress contract with RED/GREEN checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRESS_PATH = ROOT / "trade_rl/simulation/execution_stress.py"
CONFIG_PATH = ROOT / "trade_rl/workflows/market_walk_forward_config.py"
DOC_PATH = ROOT / "docs/EXECUTION_ROBUSTNESS.md"
SOURCE_TEST = ROOT / "tests/workflows/test_execution_robustness_source_contract.py"
WORKFLOW = ROOT / ".github/workflows/apply-final-branch-contracts.yml"
SCRIPT = Path(__file__).resolve()
EXECUTION_SOURCE_TIP = "0ad2440eeaa5895ed55501a9c77d4a822cd07659"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one marker, found {count}")
    return text.replace(old, new, 1)


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def stage_source_execution_contract() -> None:
    completed = subprocess.run(
        [
            "git",
            "show",
            f"{EXECUTION_SOURCE_TIP}:tests/workflows/"
            "test_execution_robustness_config.py",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    SOURCE_TEST.write_bytes(completed.stdout)


def verify_red() -> None:
    test = "tests/workflows/test_execution_robustness_source_contract.py"
    print(f"+ uv run pytest -q {test}", flush=True)
    completed = subprocess.run(
        ["uv", "run", "pytest", "-q", test],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout)
    if completed.returncode == 0:
        raise RuntimeError("execution slippage-floor RED contract unexpectedly passed")
    if "slippage_std_floor" not in completed.stdout:
        raise RuntimeError("execution slippage-floor RED contract failed unexpectedly")


def patch_execution_stress() -> None:
    text = STRESS_PATH.read_text()
    import_marker = (
        "from trade_rl.simulation.execution import ExecutionCostConfig, "
        "ExecutionRuleStress\n\n\n"
    )
    helper = '''from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionRuleStress


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be a finite number")
    return resolved


'''
    text = replace_once(text, import_marker, helper, label="stress finite helper")
    text = replace_once(
        text,
        "    slippage_std_multiplier: float = 1.0\n",
        "    slippage_std_multiplier: float = 1.0\n"
        "    slippage_std_floor: float = 0.0\n",
        label="stress slippage floor field",
    )
    validation_start = text.index("        for field_name, value in (\n")
    validation_end = text.index(
        "\n    @property\n    def environment_enabled", validation_start
    )
    validation = '''        for field_name in (
            "fee_multiplier",
            "spread_multiplier",
            "impact_multiplier",
            "slippage_std_multiplier",
            "borrow_rate_multiplier",
        ):
            value = _finite_number(getattr(self, field_name), field=field_name)
            if value < 1.0:
                raise ValueError(f"{field_name} must be at least one")
            object.__setattr__(self, field_name, value)
        slippage_floor = _finite_number(
            self.slippage_std_floor,
            field="slippage_std_floor",
        )
        if slippage_floor < 0.0:
            raise ValueError("slippage_std_floor must be non-negative")
        object.__setattr__(self, "slippage_std_floor", slippage_floor)
        participation = _finite_number(
            self.participation_fraction,
            field="participation_fraction",
        )
        if not 0.0 < participation <= 1.0:
            raise ValueError("participation_fraction must be within (0, 1]")
        object.__setattr__(self, "participation_fraction", participation)
        if (
            isinstance(self.minimum_order_latency_bars, bool)
            or not isinstance(self.minimum_order_latency_bars, int)
            or self.minimum_order_latency_bars < 0
        ):
            raise ValueError(
                "minimum_order_latency_bars must be a non-negative integer"
            )
        probability = _finite_number(
            self.tail_slippage_probability_floor,
            field="tail_slippage_probability_floor",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("tail_slippage_probability_floor must be within [0, 1]")
        object.__setattr__(self, "tail_slippage_probability_floor", probability)
        multiplier_floor = _finite_number(
            self.tail_slippage_multiplier_floor,
            field="tail_slippage_multiplier_floor",
        )
        if multiplier_floor != 0.0 and multiplier_floor < 1.0:
            raise ValueError(
                "tail_slippage_multiplier_floor must be zero or at least one"
            )
        if probability > 0.0 and multiplier_floor < 1.0:
            raise ValueError(
                "tail_slippage_multiplier_floor must be at least one when the "
                "tail probability floor is positive"
            )
        object.__setattr__(
            self,
            "tail_slippage_multiplier_floor",
            multiplier_floor,
        )
'''
    text = text[:validation_start] + validation + text[validation_end:]
    text = replace_once(
        text,
        "            or self.slippage_std_multiplier > 1.0\n",
        "            or self.slippage_std_multiplier > 1.0\n"
        "            or self.slippage_std_floor > 0.0\n",
        label="stress floor enabled",
    )
    text = replace_once(
        text,
        "            slippage_std=(base.slippage_std * self.slippage_std_multiplier),\n",
        "            slippage_std=max(\n"
        "                base.slippage_std * self.slippage_std_multiplier,\n"
        "                self.slippage_std_floor,\n"
        "            ),\n",
        label="stress floor application",
    )
    text = replace_once(
        text,
        '            "slippage_std_multiplier": self.slippage_std_multiplier,\n',
        '            "slippage_std_floor": self.slippage_std_floor,\n'
        '            "slippage_std_multiplier": self.slippage_std_multiplier,\n',
        label="stress floor digest",
    )
    STRESS_PATH.write_text(text)


def patch_execution_config() -> None:
    text = CONFIG_PATH.read_text()
    text = replace_once(
        text,
        "    slippage_std_multiplier: float = 1.0\n",
        "    slippage_std_multiplier: float = 1.0\n"
        "    slippage_std_floor: float = 0.0\n",
        label="scenario floor field",
    )
    text = replace_once(
        text,
        "            slippage_std_multiplier=self.slippage_std_multiplier,\n",
        "            slippage_std_multiplier=self.slippage_std_multiplier,\n"
        "            slippage_std_floor=self.slippage_std_floor,\n",
        label="scenario floor conversion",
    )
    parser_marker = '''                slippage_std_multiplier=_finite_number(
                    raw_scenario.get("slippage_std_multiplier"),
                    field=f"{field}.slippage_std_multiplier",
                    default=1.0,
                ),
'''
    parser_replacement = parser_marker + '''                slippage_std_floor=_finite_number(
                    raw_scenario.get("slippage_std_floor"),
                    field=f"{field}.slippage_std_floor",
                    default=0.0,
                ),
'''
    text = replace_once(
        text,
        parser_marker,
        parser_replacement,
        label="scenario floor parser",
    )
    CONFIG_PATH.write_text(text)


def patch_docs() -> None:
    text = DOC_PATH.read_text()
    text = replace_once(
        text,
        "fees, spread, market impact, stochastic slippage scale, participation\n",
        "fees, spread, market impact, stochastic slippage scale and absolute floor, participation\n",
        label="execution robustness documentation",
    )
    DOC_PATH.write_text(text)


def verify_green() -> None:
    python_files = [
        "trade_rl/simulation/execution_stress.py",
        "trade_rl/workflows/market_walk_forward_config.py",
        "tests/workflows/test_execution_robustness_source_contract.py",
    ]
    run("uv", "run", "ruff", "check", "--fix", *python_files)
    run("uv", "run", "ruff", "format", *python_files)
    run(
        "uv",
        "run",
        "pytest",
        "-q",
        "tests/learning/test_hierarchical_behavior_cloning_evaluation.py",
        "tests/learning/test_behavior_cloning.py",
        "tests/learning/test_behavior_cloning_episode_training.py",
        "tests/learning/test_behavior_cloning_temporal_split.py",
        "tests/integrations/test_sb3_training.py",
        "tests/workflows/test_execution_robustness_source_contract.py",
        "tests/workflows/test_execution_robustness_config.py",
        "tests/simulation/test_execution_environment_stress.py",
        "tests/rl/test_environment_reward_execution_resources.py",
        "tests/examples/test_execution_robustness_profile.py",
        "tests/simulation/test_execution_sensitivity.py",
        "tests/evaluation/test_execution_sensitivity_matrix.py",
    )
    run("uv", "run", "ruff", "check", *python_files)
    run("uv", "run", "ruff", "format", "--check", *python_files)
    run(
        "uv",
        "run",
        "mypy",
        "trade_rl/integrations/behavior_cloning.py",
        "trade_rl/learning/episode_behavior_cloning.py",
        "trade_rl/simulation/execution_stress.py",
        "trade_rl/workflows/market_walk_forward_config.py",
    )


def main() -> None:
    stage_source_execution_contract()
    verify_red()
    patch_execution_stress()
    patch_execution_config()
    patch_docs()
    verify_green()
    WORKFLOW.unlink()
    SCRIPT.unlink()


if __name__ == "__main__":
    main()
