from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.evaluation.experiments.test_evidence import _config
from tests.evaluation.experiments.test_workflow import _with_baseline
from trade_rl.evaluation.experiments import ControlledFactor, define_experiment
from trade_rl.evaluation.experiments.errors import ContractViolationError
from trade_rl.strategies.rl.ppo import PPO_GLOBAL_BTC_REGIME_CONTEXT


def test_feature_set_can_define_exact_global_btc_observation_extension(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None

    definition = define_experiment(
        root,
        dataset_root=dataset_root,
        hypothesis="Add only the sealed PPO global-BTC regime observation context.",
        factor=ControlledFactor.FEATURE_SET,
        candidate_config=replace(
            _config(),
            ppo_global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT,
        ),
        baseline_evidence_digest=snapshot.baseline.fingerprint,
    )

    assert definition.candidate_config.schema_version == "resolved_run_config_v3"
    assert (
        definition.candidate_config.ppo_observation_schema
        == "ppo_observation_v3_global_btc_regime"
    )
    assert definition.candidate_config.ppo_global_context == (
        PPO_GLOBAL_BTC_REGIME_CONTEXT
    )


def test_non_feature_factor_cannot_use_global_btc_observation_extension(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None

    with pytest.raises(ContractViolationError, match="Study-fixed resolved field"):
        define_experiment(
            root,
            dataset_root=dataset_root,
            hypothesis="A training-budget experiment cannot smuggle schema drift.",
            factor=ControlledFactor.PPO_TRAINING_BUDGET,
            candidate_config=replace(
                _config(),
                ppo_global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT,
            ),
            baseline_evidence_digest=snapshot.baseline.fingerprint,
        )


def test_feature_set_still_rejects_unrelated_study_fixed_window_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, dataset_root, snapshot = _with_baseline(tmp_path, monkeypatch)
    assert snapshot.baseline is not None

    with pytest.raises(ContractViolationError, match="evaluation_start"):
        define_experiment(
            root,
            dataset_root=dataset_root,
            hypothesis="FEATURE_SET cannot alter the frozen development window.",
            factor=ControlledFactor.FEATURE_SET,
            candidate_config=replace(
                _config(),
                evaluation_start=np.datetime64("2026-01-01T13:00:00", "ns"),
                ppo_global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT,
            ),
            baseline_evidence_digest=snapshot.baseline.fingerprint,
        )
