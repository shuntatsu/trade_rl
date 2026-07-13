from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from mars_lite.pipeline import residual_candidates
from mars_lite.pipeline.residual_candidates import train_select_residual_candidates


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


def test_candidate_api_separates_checkpoint_and_configuration_validation() -> None:
    parameters = inspect.signature(train_select_residual_candidates).parameters

    assert "checkpoint_val_fs" in parameters
    assert "selection_fs" in parameters
    assert "val_fs" not in parameters


def test_candidate_training_builds_abcd_and_resolves_selected_agent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    b_agent = _Agent("B")
    d_agent = _Agent("D")
    checkpoint_fs = object()
    selection_fs = object()
    train_calls: list[tuple[str, bool, object]] = []
    evaluated_on: list[object] = []

    def fake_train(
        *,
        label: str,
        alpha_enabled: bool,
        checkpoint_val_fs: object,
        output: Path,
        **kwargs,
    ):
        train_calls.append((label, alpha_enabled, checkpoint_val_fs))
        if label == "B_trend_mix":
            path = output / "b.zip"
            path.write_bytes(b"b-model")
            return b_agent, [b_agent], path
        path = output / "d.zip"
        path.write_bytes(b"d-model")
        return d_agent, [d_agent], path

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
    monkeypatch.setattr(residual_candidates, "_train_residual_ensemble", fake_train)

    def fake_evaluate(agent, fs, **kwargs):
        evaluated_on.append(fs)
        return next(evaluations)

    monkeypatch.setattr(residual_candidates, "evaluate_relative_agent", fake_evaluate)

    result = train_select_residual_candidates(
        args=SimpleNamespace(seed=7),
        train_fs=object(),
        checkpoint_val_fs=checkpoint_fs,
        selection_fs=selection_fs,
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
    assert len(result.selected_model_digest) == 64
    assert train_calls == [
        ("B_trend_mix", False, checkpoint_fs),
        ("D_combined", True, checkpoint_fs),
    ]
    assert evaluated_on == [selection_fs] * 8


def test_candidate_training_omits_cd_when_alpha_gate_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    b_agent = _Agent("B")
    checkpoint_fs = object()
    selection_fs = object()

    def fake_train(*, output: Path, **kwargs):
        path = output / "b.zip"
        path.write_bytes(b"b-model")
        return b_agent, [b_agent], path

    evaluations = iter(
        [
            _relative(0.0),
            _relative(0.0),
            _relative(0.01),
            _relative(0.0),
        ]
    )
    monkeypatch.setattr(residual_candidates, "_train_residual_ensemble", fake_train)
    monkeypatch.setattr(
        residual_candidates,
        "evaluate_relative_agent",
        lambda *args, **kwargs: next(evaluations),
    )

    result = train_select_residual_candidates(
        args=SimpleNamespace(seed=3),
        train_fs=object(),
        checkpoint_val_fs=checkpoint_fs,
        selection_fs=selection_fs,
        trend_family=object(),
        alpha=_Alpha(enabled=False),
        env_kwargs={},
        output=tmp_path,
    )

    assert set(result.development_results) == {"A", "B"}
    assert result.selected_configuration == "B"
    assert result.selected_alpha_enabled is False
    assert len(result.selected_model_digest) == 64
