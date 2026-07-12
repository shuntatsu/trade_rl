import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mars_lite.pipeline import production_pipeline
from mars_lite.pipeline.release_eligibility import derive_release_eligibility
from mars_lite.pipeline.release_risk import ReleaseRiskPolicy
from mars_lite.serving.bundle import load_bundle
from mars_lite.serving.registry import ModelRegistry


class _PostConfig:
    def to_dict(self):
        return {"vol_lookback": 60}


class _PostProcessor:
    cfg = _PostConfig()


class _ObservationSpace:
    shape = (9,)


class _Environment:
    observation_space = _ObservationSpace()


class _Features:
    def __init__(self, n_bars: int = 500) -> None:
        self.n_bars = n_bars
        self.symbols = ["BTCUSDT"]
        self.feature_names = ["ret"]
        self.global_feature_names: list[str] = []

    def slice(self, start: int, stop: int):
        return _Features(stop - start)


def _eligibility(**overrides: object):
    values: dict[str, object] = {
        "forced": False,
        "skip_p0": False,
        "skip_pbt": True,
        "skip_wf": False,
        "skip_gate": False,
        "sealed_holdout_used": True,
        "p0_passed": True,
        "walk_forward_passed": True,
        "gate2_passed": True,
        "significance_passed": None,
    }
    values.update(overrides)
    return derive_release_eligibility(**values)  # type: ignore[arg-type]


def _risk() -> ReleaseRiskPolicy:
    return ReleaseRiskPolicy(
        max_leverage=1.0,
        max_single_weight=0.5,
        max_net_exposure=1.0,
        max_worst_case_notional=100_000.0,
        min_order_notional=10.0,
        symbol_liquidity_caps={"BTCUSDT": 50_000.0},
        forbidden_symbols=(),
    )


def _write_valid_risk(tmp_path: Path) -> Path:
    path = tmp_path / "release-risk.json"
    path.write_text(json.dumps(_risk().to_dict()), encoding="utf-8")
    return path


def _run_args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "output": str(tmp_path / "output"),
        "no_register": False,
        "force": False,
        "skip_p0": False,
        "skip_pbt": True,
        "skip_wf": False,
        "skip_gate": False,
        "risk_config": _write_valid_risk(tmp_path),
        "horizon": 4,
        "decision_every": 1,
        "days": 240,
        "holdout_frac": 0.15,
        "wf_cost_gate": 0.0,
        "require_significant": False,
        "seed": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _stub_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    features: _Features,
    registered: list[object],
) -> None:
    def phase_p0(args, output_dir: Path) -> None:
        (output_dir / "p0_report.json").write_text(
            '{"gate":{"P0_PASSED":true}}', encoding="utf-8"
        )

    def phase_wf(args, output_dir: Path, fs=None) -> None:
        (output_dir / "walk_forward_cost2x.json").write_text(
            '{"summary":{"agent_total_return":{"median":0.1}}}',
            encoding="utf-8",
        )

    def phase_train(args, output_dir: Path, dev_fs=None, holdout_fs=None):
        return {
            "agent_res": {"total_return": 0.1, "equity_curve": [1.0, 1.1]},
            "baselines": {},
            "gate2": {"passed": True},
        }

    monkeypatch.setattr(production_pipeline, "phase_p0", phase_p0)
    monkeypatch.setattr(production_pipeline, "phase_wf", phase_wf)
    monkeypatch.setattr(production_pipeline, "phase_train", phase_train)
    monkeypatch.setattr(
        production_pipeline, "build_feature_set", lambda args, output_dir: features
    )
    monkeypatch.setattr(
        production_pipeline,
        "build_and_register_candidate",
        lambda **kwargs: registered.append(kwargs),
    )


def test_pipeline_registers_complete_candidate_without_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "portfolio_model.zip").write_bytes(b"model")
    (output / "train_report.json").write_text(
        '{"feature_mask":[true],"signal_gate":{"passed":true},"lockbox":null}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        production_pipeline,
        "build_post_processor",
        lambda args, horizon: _PostProcessor(),
    )
    monkeypatch.setattr(
        production_pipeline,
        "build_env_kwargs",
        lambda args, processor, horizon: {
            "post_processor": processor,
            "obs_risk_state": True,
            "disagreement_dr_max": 0.3,
        },
    )
    monkeypatch.setattr(
        production_pipeline,
        "PortfolioTradingEnv",
        lambda feature_set, **kwargs: _Environment(),
    )
    args = SimpleNamespace(
        git_sha="a" * 40,
        model_version="v1",
        horizon=4,
        ensemble=1,
        feature_norm="none",
        base_timeframe="4h",
        signal_layer="off",
        registry_dir=str(tmp_path / "registry"),
    )
    features = _Features()
    result = {
        "agent_res": {"total_return": 0.1, "equity_curve": [1.0, 1.1]},
        "gate2": {"passed": True},
    }

    candidate = production_pipeline.build_and_register_candidate(
        args=args,
        output_dir=output,
        feature_set=features,
        train_result=result,
        risk_policy=_risk(),
        release_eligibility=_eligibility(),
    )

    registry = ModelRegistry(tmp_path / "registry")
    registered = load_bundle(registry.version_dir("v1"))
    run_config = registered.metadata["run_config"]
    assert candidate.is_dir()
    assert registry.list_versions() == ["v1"]
    assert not registry.active_path.exists()
    assert registered.metadata["observation_dim"] == 9
    assert registered.metadata["release_eligibility"]["eligible"] is True
    assert registered.risk["pre_trade"]["max_single_weight"] == 0.5
    assert run_config["obs_risk_state"] is True
    assert run_config["disagreement_dr_max"] == 0.3
    assert run_config["base_timeframe"] == "4h"
    assert run_config["observation_progress_mode"] == "zero"


def test_production_candidate_rejects_unreproducible_signal_layer(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(signal_layer="append")

    with pytest.raises(ValueError, match="signal_layer=off"):
        production_pipeline.build_and_register_candidate(
            args=args,
            output_dir=tmp_path,
            feature_set=SimpleNamespace(),
            train_result={},
            risk_policy=_risk(),
            release_eligibility=_eligibility(),
        )


@pytest.mark.parametrize("flag", ["force", "skip_p0", "skip_wf", "skip_gate"])
def test_release_disqualifying_override_never_registers_candidate(
    tmp_path: Path,
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[object] = []
    _stub_pipeline(monkeypatch, features=_Features(), registered=registered)
    args = _run_args(tmp_path, **{flag: True})

    assert production_pipeline.run(args) == 0
    assert registered == []


def test_release_run_rejects_missing_sealed_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered: list[object] = []
    _stub_pipeline(monkeypatch, features=_Features(n_bars=80), registered=registered)

    with pytest.raises(RuntimeError, match="sealed holdout"):
        production_pipeline.run(_run_args(tmp_path))

    assert registered == []


def test_research_run_may_continue_without_holdout_when_not_registering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered: list[object] = []
    _stub_pipeline(monkeypatch, features=_Features(n_bars=80), registered=registered)

    args = _run_args(tmp_path, no_register=True, risk_config=None)
    assert production_pipeline.run(args) == 0
    assert registered == []
