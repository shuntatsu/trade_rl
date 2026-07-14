from __future__ import annotations

from dataclasses import dataclass

from trade_rl.workflows.fold_signals import FoldSignalProviderFactory, SignalStage


@dataclass(frozen=True)
class DummyRun:
    alpha_artifact: None = None
    factor_artifact: None = None


def test_trend_only_factory_is_explicit_and_stage_scoped() -> None:
    factory = FoldSignalProviderFactory.trend_only(dataset_id="a" * 64)

    training = factory.build(stage=SignalStage.TRAIN, run=DummyRun())
    test = factory.build(stage=SignalStage.TEST, run=DummyRun(), evaluation_start=10)

    assert training.alpha is None and training.factor is None
    assert test.alpha is None and test.factor is None
    assert training.stage is SignalStage.TRAIN
    assert test.stage is SignalStage.TEST
