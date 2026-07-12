from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mars_lite.pipeline.release_eligibility import derive_release_eligibility
from mars_lite.pipeline.release_risk import ReleaseRiskPolicy
from mars_lite.serving.bundle import load_bundle
from mars_lite.serving.candidate import create_candidate_bundle
from mars_lite.serving.runtime_defaults import IdentityResidualPolicy, _load_policy


def _eligibility():
    return derive_release_eligibility(
        forced=False,
        skip_p0=False,
        skip_pbt=False,
        skip_wf=False,
        skip_gate=False,
        sealed_holdout_used=True,
        p0_passed=True,
        signal_gate_passed=True,
        walk_forward_passed=True,
        gate2_passed=True,
        significance_passed=None,
    )


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


def _alpha_artifact(path: Path) -> Path:
    path.write_text(
        '{"model":"ridge","horizon":4,"target":"cs_demean",'
        '"feature_names":["ret"],"symbols":["BTCUSDT"],'
        '"fit_cutoff_index":100,"fit_cutoff_timestamp":"2026-01-01T00:00:00",'
        '"prediction_mean":0.0,"prediction_std":1.0,'
        '"gate_result":{"passed":false},"dataset_identity":"abc",'
        '"ridge_weights":null,"gbm_model_string":null}\n',
        encoding="utf-8",
    )
    return path


def test_baseline_only_bundle_has_no_policy_model(tmp_path: Path) -> None:
    alpha = _alpha_artifact(tmp_path / "alpha.json")
    candidate = create_candidate_bundle(
        destination=tmp_path / "candidate",
        model_source=None,
        version="v1",
        git_sha="a" * 40,
        symbols=("BTCUSDT",),
        feature_names=("ret",),
        global_feature_names=(),
        feature_norm="none",
        feature_mask=None,
        observation_dim=9,
        observation_schema_version=2,
        post_processor={"vol_lookback": 48},
        run_config={"observation_progress_mode": "zero", "base_timeframe": "1h"},
        metrics={"gate": {"passed": True}},
        guardrails={},
        risk_policy=_risk(),
        release_eligibility=_eligibility(),
        action_schema="baseline_residual_v1",
        policy_mode="baseline_only",
        residual_alpha_source=alpha,
        trend_family_config={"fast_lookback": 24, "base_lookback": 48, "slow_lookback": 96},
        composer_config={"alpha_budget_max": 0.30, "max_gross": 1.0},
    )

    bundle = load_bundle(candidate)
    assert bundle.metadata["model_kind"] == "baseline_only"
    assert bundle.metadata["action_schema"] == "baseline_residual_v1"
    assert (candidate / "residual_alpha.json").is_file()
    assert not (candidate / "model.zip").exists()


def test_load_policy_returns_identity_for_baseline_only() -> None:
    bundle = SimpleNamespace(metadata={"policy_mode": "baseline_only"})
    policy = _load_policy(bundle)

    assert isinstance(policy, IdentityResidualPolicy)
    action, _ = policy.predict(np.zeros(10, dtype=np.float32), deterministic=True)
    np.testing.assert_array_equal(action, np.zeros(2, dtype=np.float32))


def test_identity_policy_preserves_batch_shape() -> None:
    action, _ = IdentityResidualPolicy().predict(
        np.zeros((3, 10), dtype=np.float32), deterministic=True
    )
    assert action.shape == (3, 2)
