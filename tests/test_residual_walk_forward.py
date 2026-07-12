from __future__ import annotations

import json
from pathlib import Path

import pytest

from mars_lite.eval.residual_walk_forward import (
    build_residual_fold_specs,
    save_residual_walk_forward_report,
    summarize_residual_folds,
)


def _relative(
    *,
    hybrid_return: float,
    shadow_return: float,
    excess: float,
    hybrid_trades: int,
    shadow_trades: int,
) -> dict[str, object]:
    return {
        "hybrid": {
            "total_return": hybrid_return,
            "n_trades": hybrid_trades,
        },
        "shadow": {
            "total_return": shadow_return,
            "n_trades": shadow_trades,
        },
        "paired": {"excess_log_return": excess},
    }


def test_fold_specs_are_expanding_and_oos_ranges_do_not_overlap() -> None:
    specs, skipped = build_residual_fold_specs(
        n_bars=2_000,
        n_folds=3,
        purge_bars=24,
        horizon=12,
    )

    assert skipped == []
    assert [spec.outer_train_start for spec in specs] == [0, 0, 0]
    assert all(spec.outer_test_start >= spec.outer_train_end + 24 for spec in specs)
    assert all(
        left.outer_test_end <= right.outer_test_start
        for left, right in zip(specs, specs[1:])
    )
    assert all(spec.inner_train_end < spec.inner_validation_start for spec in specs)
    assert all(spec.inner_validation_end == spec.outer_train_end for spec in specs)


def test_fold_specs_skip_only_declared_minimum_size_failures() -> None:
    specs, skipped = build_residual_fold_specs(
        n_bars=500,
        n_folds=8,
        purge_bars=24,
        horizon=12,
    )

    assert len(specs) + len(skipped) == 8
    assert skipped
    assert {entry["reason"] for entry in skipped} <= {
        "inner_train_too_short",
        "inner_validation_too_short",
        "outer_test_too_short",
    }
    assert all("fold" in entry for entry in skipped)


@pytest.mark.parametrize(
    ("n_bars", "n_folds", "purge", "horizon"),
    [
        (0, 3, 24, 12),
        (1000, 0, 24, 12),
        (1000, 3, 0, 12),
        (1000, 3, 24, 0),
    ],
)
def test_fold_specs_reject_non_positive_inputs(
    n_bars: int,
    n_folds: int,
    purge: int,
    horizon: int,
) -> None:
    with pytest.raises(ValueError):
        build_residual_fold_specs(
            n_bars=n_bars,
            n_folds=n_folds,
            purge_bars=purge,
            horizon=horizon,
        )


def test_summary_counts_selection_activity_and_zero_trade_warnings() -> None:
    folds = [
        {
            "selected_configuration": "A",
            "alpha_enabled": False,
            "selected_seed_fallbacks": [],
            "outer_oos": {
                "relative_1x": _relative(
                    hybrid_return=0.10,
                    shadow_return=0.10,
                    excess=0.0,
                    hybrid_trades=4,
                    shadow_trades=4,
                ),
                "relative_2x": _relative(
                    hybrid_return=0.08,
                    shadow_return=0.08,
                    excess=0.0,
                    hybrid_trades=4,
                    shadow_trades=4,
                ),
            },
            "split": {"outer_test_scored_bars": 100},
        },
        {
            "selected_configuration": "B",
            "alpha_enabled": False,
            "selected_seed_fallbacks": [True, False],
            "outer_oos": {
                "relative_1x": _relative(
                    hybrid_return=0.02,
                    shadow_return=0.01,
                    excess=0.01,
                    hybrid_trades=0,
                    shadow_trades=0,
                ),
                "relative_2x": _relative(
                    hybrid_return=0.0,
                    shadow_return=0.0,
                    excess=0.0,
                    hybrid_trades=0,
                    shadow_trades=0,
                ),
            },
            "split": {"outer_test_scored_bars": 120},
        },
    ]

    summary = summarize_residual_folds(
        folds,
        requested_folds=2,
        skipped_folds=[],
    )

    assert summary["selection_counts"] == {"A": 1, "B": 1, "D": 0}
    assert summary["shadow_zero_trade_folds"] == 1
    assert summary["hybrid_zero_trade_folds"] == 1
    assert summary["selected_member_fallback_count"] == 1
    assert summary["selected_member_count"] == 2
    assert summary["total_scored_oos_bars"] == 220
    assert summary["completed_folds"] == 2
    assert summary["skipped_folds"] == 0
    assert summary["hybrid_beats_shadow_fraction_1x"] == pytest.approx(0.5)
    assert summary["survives_cost2x_fraction"] == pytest.approx(1.0)


def test_report_save_is_deterministic_and_rejects_non_finite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "residual_walk_forward.json"
    payload = {"b": 2, "a": {"value": 1.0}}

    save_residual_walk_forward_report(path, payload)
    first = path.read_text(encoding="utf-8")
    save_residual_walk_forward_report(path, payload)
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert first.endswith("\n")
    assert list(json.loads(first)) == ["a", "b"]

    with pytest.raises(ValueError):
        save_residual_walk_forward_report(path, {"bad": float("nan")})
