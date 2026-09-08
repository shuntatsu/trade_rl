from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ROOT / "trade_rl" / "strategies"

MOVES = {
    "trend.py": "rules/trend.py",
    "mean_reversion.py": "rules/mean_reversion.py",
    "forecast.py": "forecasts/controller.py",
    "supervised.py": "forecasts/supervised.py",
    "ridge.py": "forecasts/ridge.py",
    "lightgbm.py": "forecasts/lightgbm.py",
    "ppo.py": "rl/ppo.py",
}

IMPORT_REPLACEMENTS = {
    "trade_rl.strategies.trend": "trade_rl.strategies.rules.trend",
    "trade_rl.strategies.mean_reversion": "trade_rl.strategies.rules.mean_reversion",
    "trade_rl.strategies.forecast": "trade_rl.strategies.forecasts.controller",
    "trade_rl.strategies.supervised": "trade_rl.strategies.forecasts.supervised",
    "trade_rl.strategies.ridge": "trade_rl.strategies.forecasts.ridge",
    "trade_rl.strategies.lightgbm": "trade_rl.strategies.forecasts.lightgbm",
    "trade_rl.strategies.ppo": "trade_rl.strategies.rl.ppo",
}

EXPECTED_PUBLIC = {
    "CausalForecastTrainingSet",
    "ConstantIntentStrategy",
    "ForecastIntentConfig",
    "ForecastIntentController",
    "LightGBMForecastModel",
    "LightGBMForecastStrategy",
    "MeanReversionIntentConfig",
    "MeanReversionIntentStrategy",
    "PPOIntentStrategy",
    "PPOTradingEnv",
    "PositionIntent",
    "RidgeForecastModel",
    "RidgeForecastStrategy",
    "SingleSymbolStrategy",
    "StrategyObservation",
    "TrendIntentConfig",
    "TrendIntentStrategy",
    "build_causal_forecast_training_set",
    "fit_lightgbm_forecast",
    "fit_ppo_strategy",
    "fit_ridge_forecast",
    "target_weight_for_intent",
}


def _assert_preconditions() -> None:
    for old, new in MOVES.items():
        old_path = STRATEGIES / old
        new_path = STRATEGIES / new
        if not old_path.is_file():
            raise RuntimeError(f"missing old strategy owner: {old_path}")
        if new_path.exists():
            raise RuntimeError(f"new strategy owner already exists: {new_path}")
    for family in ("rules", "forecasts", "rl"):
        if (STRATEGIES / family).exists():
            raise RuntimeError(f"strategy family already exists: {family}")


def _move_files() -> None:
    for old, new in MOVES.items():
        old_path = STRATEGIES / old
        new_path = STRATEGIES / new
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)


def _rewrite_imports() -> None:
    for root_name in ("trade_rl", "tests", "scripts"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            updated = text
            for old, new in IMPORT_REPLACEMENTS.items():
                updated = updated.replace(old, new)
            if updated != text:
                path.write_text(updated, encoding="utf-8")


def _write_family_facades() -> None:
    (STRATEGIES / "rules" / "__init__.py").write_text(
        '''"""Rule-based maintained strategy family."""

from trade_rl.strategies.rules.mean_reversion import (
    MeanReversionIntentConfig,
    MeanReversionIntentStrategy,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig, TrendIntentStrategy

__all__ = [
    "MeanReversionIntentConfig",
    "MeanReversionIntentStrategy",
    "TrendIntentConfig",
    "TrendIntentStrategy",
]
''',
        encoding="utf-8",
    )
    (STRATEGIES / "forecasts" / "__init__.py").write_text(
        '''"""Supervised forecast maintained strategy family."""

from trade_rl.strategies.forecasts.controller import (
    ForecastIntentConfig,
    ForecastIntentController,
)
from trade_rl.strategies.forecasts.lightgbm import (
    LightGBMForecastModel,
    LightGBMForecastStrategy,
    fit_lightgbm_forecast,
)
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
    fit_ridge_forecast,
)
from trade_rl.strategies.forecasts.supervised import (
    CausalForecastTrainingSet,
    build_causal_forecast_training_set,
)

__all__ = [
    "CausalForecastTrainingSet",
    "ForecastIntentConfig",
    "ForecastIntentController",
    "LightGBMForecastModel",
    "LightGBMForecastStrategy",
    "RidgeForecastModel",
    "RidgeForecastStrategy",
    "build_causal_forecast_training_set",
    "fit_lightgbm_forecast",
    "fit_ridge_forecast",
]
''',
        encoding="utf-8",
    )
    (STRATEGIES / "rl" / "__init__.py").write_text(
        '''"""Reinforcement-learning maintained strategy family."""

from trade_rl.strategies.rl.ppo import PPOIntentStrategy, PPOTradingEnv, fit_ppo_strategy

__all__ = ["PPOIntentStrategy", "PPOTradingEnv", "fit_ppo_strategy"]
''',
        encoding="utf-8",
    )


def _verify_public_facade_source() -> None:
    tree = ast.parse((STRATEGIES / "__init__.py").read_text(encoding="utf-8"))
    exported: set[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            exported = set(ast.literal_eval(node.value))
    if exported != EXPECTED_PUBLIC:
        raise RuntimeError(f"strategy public facade changed: {exported}")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _assert_no_stale_private_imports() -> None:
    stale = set(IMPORT_REPLACEMENTS)
    offenders: list[tuple[str, list[str]]] = []
    for root_name in ("trade_rl", "tests"):
        for path in (ROOT / root_name).rglob("*.py"):
            found = sorted(_imported_modules(path) & stale)
            if found:
                offenders.append((str(path.relative_to(ROOT)), found))
    if offenders:
        raise RuntimeError(f"stale flat strategy imports remain: {offenders}")


def _assert_new_tree() -> None:
    for old, new in MOVES.items():
        if (STRATEGIES / old).exists():
            raise RuntimeError(f"old strategy path survived: {old}")
        path = STRATEGIES / new
        if not path.is_file():
            raise RuntimeError(f"new strategy owner missing: {new}")
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for family in ("rules", "forecasts", "rl"):
        init = STRATEGIES / family / "__init__.py"
        if not init.is_file():
            raise RuntimeError(f"missing family facade: {init}")
        ast.parse(init.read_text(encoding="utf-8"), filename=str(init))


def main() -> None:
    _assert_preconditions()
    _move_files()
    _rewrite_imports()
    _write_family_facades()
    _verify_public_facade_source()
    _assert_no_stale_private_imports()
    _assert_new_tree()
    print("Phase 3A strategy family migration created successfully")


if __name__ == "__main__":
    main()
