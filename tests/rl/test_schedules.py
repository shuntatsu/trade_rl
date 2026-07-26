from __future__ import annotations

import pytest

from trade_rl.rl.schedules import build_learning_rate_schedule


def test_linear_schedule_uses_progress_remaining() -> None:
    schedule = build_learning_rate_schedule(
        initial_rate=1.2e-4,
        final_ratio=0.1,
        kind="linear",
    )
    assert callable(schedule)
    assert schedule(1.0) == pytest.approx(1.2e-4)
    assert schedule(0.0) == pytest.approx(1.2e-5)


def test_cosine_schedule_uses_exact_endpoints() -> None:
    schedule = build_learning_rate_schedule(
        initial_rate=1.2e-4,
        final_ratio=0.1,
        kind="cosine",
    )
    assert callable(schedule)
    assert schedule(1.0) == pytest.approx(1.2e-4)
    assert schedule(0.0) == pytest.approx(1.2e-5)


def test_constant_schedule_returns_float() -> None:
    assert build_learning_rate_schedule(
        initial_rate=1.2e-4,
        final_ratio=0.1,
        kind="constant",
    ) == pytest.approx(1.2e-4)


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedule_rejects_progress_outside_unit_interval(kind: str) -> None:
    schedule = build_learning_rate_schedule(
        initial_rate=1.2e-4,
        final_ratio=0.1,
        kind=kind,
    )
    assert callable(schedule)
    with pytest.raises(ValueError, match="progress_remaining"):
        schedule(-0.01)
    with pytest.raises(ValueError, match="progress_remaining"):
        schedule(1.01)


@pytest.mark.parametrize(
    ("initial_rate", "final_ratio", "kind"),
    [
        (0.0, 0.1, "linear"),
        (1e-4, 0.0, "linear"),
        (1e-4, 1.1, "linear"),
        (1e-4, 0.1, "step"),
    ],
)
def test_schedule_rejects_invalid_configuration(
    initial_rate: float,
    final_ratio: float,
    kind: str,
) -> None:
    with pytest.raises(ValueError):
        build_learning_rate_schedule(
            initial_rate=initial_rate,
            final_ratio=final_ratio,
            kind=kind,
        )
