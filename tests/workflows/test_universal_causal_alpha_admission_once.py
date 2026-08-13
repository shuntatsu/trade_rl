from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)


def _batch(symbol: str) -> EpisodeOracleBatch:
    dataset_id = content_digest((symbol, "dataset"))
    contract = OracleEpisodeContract(
        dataset_id=dataset_id,
        episode_index=1,
        start=10,
        stop=13,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )
    return EpisodeOracleBatch(
        dataset_id=dataset_id,
        teacher_config_digest=content_digest("teacher"),
        sampling_config_digest=content_digest((symbol, "sampling")),
        contracts=(contract,),
        targets=(np.asarray([[0.2], [0.2]], dtype=np.float32),),
    )


def test_teacher_holdouts_are_replayed_exactly_once_per_symbol(monkeypatch) -> None:
    import trade_rl.workflows.universal_causal_alpha_teacher as module

    symbols = ("AAAUSDT", "BBBUSDT")
    calls: list[str] = []

    def replay(factory, contract, *, actions):
        del factory
        calls.append(contract.dataset_id)
        assert actions.shape == (2, 1)
        return SimpleNamespace(
            performance=SimpleNamespace(
                gross_return=0.02,
                net_return=0.01,
                turnover_total=3.0,
                cost_total=0.25,
                trade_count=4,
                maximum_drawdown=0.03,
            )
        )

    monkeypatch.setattr(module, "evaluate_episode_action_path", replay)
    evidence = module.evaluate_causal_alpha_teacher_holdouts(
        train_symbols=symbols,
        batches={symbol: _batch(symbol) for symbol in symbols},
        environment_factories={symbol: lambda: object() for symbol in symbols},
        episode_hours=720.0,
    )

    assert calls == [content_digest((symbol, "dataset")) for symbol in symbols]
    assert tuple(metric.symbol for metric in evidence.metrics) == symbols
    assert evidence.passed is True
