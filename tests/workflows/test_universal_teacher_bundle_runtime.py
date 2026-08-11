from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding


def _digest(label: str) -> str:
    return content_digest(label)


def _binding(symbol: str) -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=_digest(f"dataset:{symbol}"),
        symbol_dataset_digest=_digest(f"dataset:{symbol}"),
        execution_metadata_digest=_digest(f"metadata:{symbol}"),
        instrument_descriptor_digest=_digest(f"descriptor:{symbol}"),
        split="train",
    )


def _batch(symbol: str) -> EpisodeOracleBatch:
    dataset_id = _digest(f"dataset:{symbol}")
    contract = OracleEpisodeContract(
        dataset_id=dataset_id,
        episode_index=0,
        start=10,
        stop=12,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )
    return EpisodeOracleBatch(
        dataset_id=dataset_id,
        teacher_config_digest=_digest("teacher-config"),
        sampling_config_digest=_digest(f"sampling:{symbol}"),
        contracts=(contract,),
        targets=(np.zeros((1, 1), dtype=np.float32),),
    )


def test_build_universal_pretraining_bundle_from_batches_closes_train_scope(
    monkeypatch,
) -> None:
    import trade_rl.workflows.universal_teacher_runtime as module
    from trade_rl.workflows.universal_teacher_runtime import (
        build_universal_pretraining_bundle_from_batches,
    )

    symbols = ("AAAUSDT", "BBBUSDT")
    opened: list[tuple[str, int]] = []
    closed: list[str] = []
    collected: list[str] = []
    combined: dict[str, object] = {}

    class _Environment:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def close(self) -> None:
            closed.append(self.symbol)

    def build_environment(**kwargs):
        symbol = kwargs["symbol"]
        opened.append((symbol, kwargs["run_seed"]))
        return _Environment(symbol)

    def collect(environment, batch, *, teacher_config_digest, gamma):
        assert teacher_config_digest == batch.teacher_config_digest
        assert gamma == 1.0
        collected.append(environment.symbol)
        dataset = SimpleNamespace(sample_count=1)
        return SimpleNamespace(
            dataset=dataset,
            critic_targets=np.zeros(1, dtype=np.float32),
        )

    split = BehaviorCloningSplit(
        train_indices=np.asarray([0], dtype=np.int64),
        validation_indices=np.asarray([], dtype=np.int64),
        train_episode_ids=np.asarray([0], dtype=np.int64),
        validation_episode_ids=np.asarray([], dtype=np.int64),
    )

    monkeypatch.setattr(
        module, "build_universal_symbol_teacher_environment", build_environment
    )
    monkeypatch.setattr(
        module,
        "oracle_teacher_config_for_environment",
        lambda _environment: SimpleNamespace(digest=_digest("teacher-config")),
    )
    monkeypatch.setattr(module, "collect_universal_episode_teacher", collect)
    monkeypatch.setattr(
        module, "behavior_cloning_split", lambda *_args, **_kwargs: split
    )

    sentinel = object()

    def combine(
        symbol_teachers, *, train_symbols, normalizer_digest, feature_schema_digest
    ):
        combined.update(symbol_teachers)
        assert tuple(train_symbols) == symbols
        assert normalizer_digest == _digest("normalizer")
        assert feature_schema_digest == _digest("features")
        return sentinel

    monkeypatch.setattr(module, "combine_symbol_teachers", combine)
    result = build_universal_pretraining_bundle_from_batches(
        train_symbols=symbols,
        bindings=tuple(_binding(symbol) for symbol in symbols),
        batches={symbol: _batch(symbol) for symbol in symbols},
        concrete_environment_factory=lambda _binding: SimpleNamespace(
            close=lambda: None
        ),
        instrument_context_provider=lambda _environment, _binding: np.zeros((1, 9)),
        partition_digest=_digest("partition"),
        training_contract_digest=_digest("training"),
        run_seed=100,
        gamma=1.0,
        validation_fraction=0.0,
        normalizer_digest=_digest("normalizer"),
        feature_schema_digest=_digest("features"),
    )

    assert result is sentinel
    assert opened == [("AAAUSDT", 100), ("BBBUSDT", 101)]
    assert collected == list(symbols)
    assert closed == list(symbols)
    assert set(combined) == set(symbols)
