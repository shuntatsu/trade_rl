from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from mars_lite.eval import residual_walk_forward


def test_eval_residual_walk_forward_does_not_import_pipeline() -> None:
    source = Path("mars_lite/eval/residual_walk_forward.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("mars_lite.pipeline"):
                forbidden.append(module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mars_lite.pipeline"):
                    forbidden.append(alias.name)
    assert forbidden == []


def test_fold_specs_have_separate_checkpoint_and_configuration_windows() -> None:
    specs, skipped = residual_walk_forward.build_residual_fold_specs(
        n_bars=4_000,
        n_folds=3,
        purge_bars=24,
        horizon=12,
    )

    assert skipped == []
    assert len(specs) == 3
    for spec in specs:
        assert spec.policy_train_start == 0
        assert spec.policy_train_end < spec.checkpoint_validation_start
        assert spec.checkpoint_validation_end < spec.configuration_selection_start
        assert spec.configuration_selection_end < spec.outer_test_start
        assert spec.purge_bars == 24


def test_stitched_oos_uses_chronological_base_bar_returns() -> None:
    series_type = getattr(residual_walk_forward, "RelativeFoldSeries", None)
    stitch = getattr(residual_walk_forward, "stitch_relative_fold_results", None)
    assert series_type is not None
    assert callable(stitch)

    folds = [
        series_type(
            fold=0,
            hybrid_returns=np.array([0.01, -0.02], dtype=np.float64),
            shadow_returns=np.array([0.0, -0.01], dtype=np.float64),
            hybrid_trades=2,
            shadow_trades=1,
            hybrid_turnover=0.4,
            shadow_turnover=0.2,
            hybrid_cost=0.003,
            shadow_cost=0.001,
        ),
        series_type(
            fold=1,
            hybrid_returns=np.array([0.03], dtype=np.float64),
            shadow_returns=np.array([0.01], dtype=np.float64),
            hybrid_trades=1,
            shadow_trades=1,
            hybrid_turnover=0.2,
            shadow_turnover=0.1,
            hybrid_cost=0.002,
            shadow_cost=0.001,
        ),
    ]

    result = stitch(folds, bars_per_year=365, bootstrap_seed=7)
    hybrid = np.array([0.01, -0.02, 0.03], dtype=np.float64)
    shadow = np.array([0.0, -0.01, 0.01], dtype=np.float64)

    assert result["n_base_bars"] == 3
    assert result["hybrid"]["total_return"] == pytest.approx(
        float(np.prod(1.0 + hybrid) - 1.0)
    )
    assert result["shadow"]["total_return"] == pytest.approx(
        float(np.prod(1.0 + shadow) - 1.0)
    )
    assert result["paired"]["excess_log_return"] == pytest.approx(
        float(np.log1p(hybrid).sum() - np.log1p(shadow).sum())
    )
    assert result["hybrid"]["n_trades"] == 3
    assert result["shadow"]["n_trades"] == 2
