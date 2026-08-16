from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaEpisodePartition,
)
from trade_rl.workflows.universal_causal_alpha_v3_runtime import (
    validate_causal_alpha_v3_shared_chronology,
)


def _partition(dataset_id: str) -> CausalAlphaEpisodePartition:
    contracts = tuple(
        OracleEpisodeContract(
            dataset_id=dataset_id,
            episode_index=index,
            start=10 + index * 10,
            stop=16 + index * 10,
            initial_state_mode="cash",
            initial_weights=np.zeros(1, dtype=np.float64),
        )
        for index in range(3)
    )
    return CausalAlphaEpisodePartition(
        contracts=contracts,
        selection_contracts=contracts[:-1],
        holdout_contract=contracts[-1],
        train_start=0,
        train_stop=contracts[-1].stop,
    )


def _clock() -> np.ndarray:
    origin = np.datetime64("2026-01-01T00:00", "ns")
    return origin + np.arange(64) * np.timedelta64(15, "m")


def test_pooled_v3_rejects_cross_symbol_signal_delay_drift() -> None:
    symbols = ("BTCUSDT", "ETHUSDT")

    with pytest.raises(ValueError, match="signal delay"):
        validate_causal_alpha_v3_shared_chronology(
            train_symbols=symbols,
            timestamps_by_symbol={symbol: _clock() for symbol in symbols},
            partitions={
                "BTCUSDT": _partition("a" * 64),
                "ETHUSDT": _partition("b" * 64),
            },
            decision_bars={symbol: 1 for symbol in symbols},
            signal_delays={"BTCUSDT": 0, "ETHUSDT": 1},
        )
