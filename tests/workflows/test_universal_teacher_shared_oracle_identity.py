from __future__ import annotations

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding


def _digest(label: str) -> str:
    return content_digest(label)


def _binding(symbol: str) -> InstrumentDatasetBinding:
    dataset_id = _digest(f"dataset:{symbol}")
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=dataset_id,
        symbol_dataset_digest=dataset_id,
        execution_metadata_digest=_digest(f"metadata:{symbol}"),
        instrument_descriptor_digest=_digest(f"descriptor:{symbol}"),
        split="train",
    )


def _batch(symbol: str) -> EpisodeOracleBatch:
    dataset_id = _digest(f"dataset:{symbol}")
    contract = OracleEpisodeContract(
        dataset_id=dataset_id,
        episode_index=0,
        start=5,
        stop=7,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )
    return EpisodeOracleBatch(
        dataset_id=dataset_id,
        teacher_config_digest=_digest("shared-teacher-config"),
        sampling_config_digest=_digest("sampling"),
        contracts=(contract,),
        targets=(np.zeros((1, 1), dtype=np.float32),),
    )


def test_shared_oracle_batches_reject_candidate_execution_identity_drift(
    monkeypatch,
) -> None:
    import trade_rl.workflows.universal_teacher_runtime as module
    from trade_rl.workflows.universal_teacher_runtime import (
        build_universal_pretraining_bundle_from_batches,
    )

    binding = _binding("AAAUSDT")
    closed: list[bool] = []

    class ConcreteEnvironment:
        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(
        module,
        "oracle_teacher_config_for_environment",
        lambda _environment: type(
            "TeacherConfig",
            (),
            {"digest": _digest("different-teacher-config")},
        )(),
    )
    monkeypatch.setattr(
        module,
        "build_universal_symbol_teacher_environment",
        lambda **_kwargs: pytest.fail(
            "generic teacher must not be built after Oracle identity drift"
        ),
    )

    with pytest.raises(ValueError, match="Oracle teacher config identity"):
        build_universal_pretraining_bundle_from_batches(
            train_symbols=("AAAUSDT",),
            bindings=(binding,),
            batches={"AAAUSDT": _batch("AAAUSDT")},
            concrete_environment_factory=lambda _binding: ConcreteEnvironment(),
            instrument_context_provider=lambda *_args, **_kwargs: np.zeros((1, 9)),
            partition_digest=_digest("partition"),
            training_contract_digest=_digest("training"),
            run_seed=11,
            gamma=1.0,
            validation_fraction=0.0,
            normalizer_digest=_digest("normalizer"),
            feature_schema_digest=_digest("features"),
        )

    assert closed == [True]
