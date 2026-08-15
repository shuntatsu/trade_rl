from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaEpisodePartition,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3ExecutionIdentity,
)
import trade_rl.workflows.universal_causal_alpha_v3_runtime as runtime_module


def _sha(token: str) -> str:
    return token * 64


def _clock(*, offset_minutes: int = 0) -> np.ndarray:
    origin = np.datetime64("2026-01-01T00:00", "ns") + np.timedelta64(
        offset_minutes, "m"
    )
    return origin + np.arange(64) * np.timedelta64(15, "m")


def _partition(dataset_id: str, *, start_offset: int = 0) -> CausalAlphaEpisodePartition:
    contracts = tuple(
        OracleEpisodeContract(
            dataset_id=dataset_id,
            episode_index=index,
            start=start_offset + index * 10,
            stop=start_offset + index * 10 + 6,
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


def _validate_shared_chronology(**kwargs: object) -> str:
    validate = getattr(runtime_module, "validate_causal_alpha_v3_shared_chronology")
    return validate(**kwargs)


def test_shared_chronology_binds_one_identical_train_clock_and_schedule() -> None:
    symbols = ("BTCUSDT", "ETHUSDT")
    digest = _validate_shared_chronology(
        train_symbols=symbols,
        timestamps_by_symbol={symbol: _clock() for symbol in symbols},
        partitions={
            "BTCUSDT": _partition(_sha("a")),
            "ETHUSDT": _partition(_sha("b")),
        },
        decision_bars={symbol: 1 for symbol in symbols},
    )

    assert len(digest) == 64


def test_shared_chronology_rejects_cross_symbol_clock_drift() -> None:
    symbols = ("BTCUSDT", "ETHUSDT")
    with pytest.raises(ValueError, match="clock"):
        _validate_shared_chronology(
            train_symbols=symbols,
            timestamps_by_symbol={
                "BTCUSDT": _clock(),
                "ETHUSDT": _clock(offset_minutes=15),
            },
            partitions={
                "BTCUSDT": _partition(_sha("a")),
                "ETHUSDT": _partition(_sha("b")),
            },
            decision_bars={symbol: 1 for symbol in symbols},
        )


def test_shared_chronology_rejects_cross_symbol_episode_schedule_drift() -> None:
    symbols = ("BTCUSDT", "ETHUSDT")
    with pytest.raises(ValueError, match="episode schedule"):
        _validate_shared_chronology(
            train_symbols=symbols,
            timestamps_by_symbol={symbol: _clock() for symbol in symbols},
            partitions={
                "BTCUSDT": _partition(_sha("a")),
                "ETHUSDT": _partition(_sha("b"), start_offset=1),
            },
            decision_bars={symbol: 1 for symbol in symbols},
        )


def test_shared_chronology_rejects_cross_symbol_decision_cadence_drift() -> None:
    symbols = ("BTCUSDT", "ETHUSDT")
    with pytest.raises(ValueError, match="decision cadence"):
        _validate_shared_chronology(
            train_symbols=symbols,
            timestamps_by_symbol={symbol: _clock() for symbol in symbols},
            partitions={
                "BTCUSDT": _partition(_sha("a")),
                "ETHUSDT": _partition(_sha("b")),
            },
            decision_bars={"BTCUSDT": 1, "ETHUSDT": 2},
        )


def test_dependency_lock_digest_changes_when_lockfile_changes(tmp_path: Path) -> None:
    digest_lock = getattr(runtime_module, "causal_alpha_v3_dependency_lock_digest")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text("version = 1\n", encoding="utf-8")
    first = digest_lock(tmp_path)
    lock_path.write_text("version = 2\n", encoding="utf-8")
    second = digest_lock(tmp_path)

    assert len(first) == 64
    assert len(second) == 64
    assert first != second


def test_python_runtime_digest_is_bound_to_execution_identity() -> None:
    runtime_digest = getattr(runtime_module, "causal_alpha_v3_python_runtime_digest")()
    identity = CausalAlphaV3ExecutionIdentity(
        train_symbols=("BTCUSDT",),
        training_contract_digest=_sha("1"),
        instrument_context_schema_digest=_sha("2"),
        source_tree_digest=_sha("3"),
        shared_clock_digest=_sha("4"),
        dependency_lock_digest=_sha("5"),
        python_runtime_digest=runtime_digest,
        symbol_runtime_digests=(("BTCUSDT", _sha("6")),),
    )

    assert identity.schema_version == "causal_alpha_v3_execution_identity_v2"
    assert identity.shared_clock_digest == _sha("4")
    assert identity.dependency_lock_digest == _sha("5")
    assert identity.python_runtime_digest == runtime_digest
    assert CausalAlphaV3ExecutionIdentity.from_payload(identity.to_payload()) == identity


def test_execution_identity_v2_rejects_legacy_v1_payload() -> None:
    identity = CausalAlphaV3ExecutionIdentity(
        train_symbols=("BTCUSDT",),
        training_contract_digest=_sha("1"),
        instrument_context_schema_digest=_sha("2"),
        source_tree_digest=_sha("3"),
        shared_clock_digest=_sha("4"),
        dependency_lock_digest=_sha("5"),
        python_runtime_digest=_sha("6"),
        symbol_runtime_digests=(("BTCUSDT", _sha("7")),),
    )
    legacy = identity.to_payload()
    legacy["schema_version"] = "causal_alpha_v3_execution_identity_v1"
    legacy.pop("shared_clock_digest")
    legacy.pop("dependency_lock_digest")
    legacy.pop("python_runtime_digest")

    with pytest.raises(ValueError, match="fields mismatch|schema"):
        CausalAlphaV3ExecutionIdentity.from_payload(legacy)
