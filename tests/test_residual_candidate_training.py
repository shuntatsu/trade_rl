from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mars_lite.pipeline import residual_pipeline
from mars_lite.pipeline.residual_pipeline import train_select_residual_candidates


class _Alpha:
    def __init__(self, enabled: bool):
        self.enabled = enabled


class _Agent:
    def __init__(self, name: str):
        self.name = name


def _relative(excess: float) -> dict:
    return {
        "hybrid": {"max_drawdown": 0.1},
        "shadow": {"max_drawdown": 0.1},
        "paired": {"excess_log_return": excess},
    }


def test_candidate_training_builds_abcd_and_resolves_selected_agent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    b_agent = _Agent("B")
    d_agent = _Agent("D")
    calls: list[tuple[str, bool]] = []

    def fake_train(*, label: str, alpha_enabled: bool, output: Path, **kwargs):
        calls.append((label, alpha_enabled))
        if label == "B_trend_mix":
            return b_agent, [b_agent], output / "b.zip"
        return d_agent, [d_agent], output / "d.zip"

    evaluations = iter(
        [
            _relative(0.0),
            _relative(0.0),
            _relative(0.02),
            _relative(0.01),
            _relative(0.01),
            _relative(0.01),
            _relative(0.015),
            _relative(0.01),
        ]
    )
    monkeypatch.setattr(residual_pipeline, "_train_residual_ensemble", fake_train)
    monkeypatch.setattr(
        residual_pipeline,
        "evaluate_relative_agent",
        lambda *args, **kwargs: next(evaluations),
    )

    result = train_select_residual_candidates(
        args=SimpleNamespace(seed=7),
        train_fs=object(),
        val_fs=object(),
        trend_family=object(),
        alpha=_Alpha(enabled=True),
        env_kwargs={},
        output=tmp_path,
    )

    assert set(result.development_results) == {"A", "B", "C", "D"}
    assert set(result.development_cost2x_results) == {"A", "B", "C", "D"}
    assert result.selected_configuration == "B"
    assert result.selected_alpha_enabled is False
    assert result.selected_agent is b_agent
    assert result.selected_policies == (b_agent,)
    assert calls == [("B_trend_mix", False), ("D_combined", True)]


def test_candidate_training_omits_cd_when_alpha_gate_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    b_agent = _Agent("B")

    def fake_train(*, output: Path, **kwargs):
        return b_agent, [b_agent], output / "b.zip"

    evaluations = iter(
        [
            _relative(0.0),
            _relative(0.0),
            _relative(0.01),
            _relative(0.0),
        ]
    )
    monkeypatch.setattr(residual_pipeline, "_train_residual_ensemble", fake_train)
    monkeypatch.setattr(
        residual_pipeline,
        "evaluate_relative_agent",
        lambda *args, **kwargs: next(evaluations),
    )

    result = train_select_residual_candidates(
        args=SimpleNamespace(seed=3),
        train_fs=object(),
        val_fs=object(),
        trend_family=object(),
        alpha=_Alpha(enabled=False),
        env_kwargs={},
        output=tmp_path,
    )

    assert set(result.development_results) == {"A", "B"}
    assert result.selected_configuration == "B"
    assert result.selected_alpha_enabled is False
