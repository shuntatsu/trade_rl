from __future__ import annotations

from types import SimpleNamespace

from trade_rl.artifacts.hashing import content_digest
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


def test_build_universal_oracle_batches_is_train_scoped_and_closes_children(
    monkeypatch,
) -> None:
    import trade_rl.workflows.universal_teacher_runtime as module
    from trade_rl.workflows.universal_teacher_runtime import (
        build_universal_oracle_batches,
    )

    bindings = (_binding("AAAUSDT"), _binding("BBBUSDT"))
    opened: list[str] = []
    closed: list[str] = []
    calls: list[tuple[str, tuple[int, int], int, int]] = []

    class Child:
        def __init__(self, binding: InstrumentDatasetBinding) -> None:
            self.binding = binding
            self.dataset = SimpleNamespace(dataset_id=binding.source_dataset_id)
            opened.append(binding.concrete_symbol)

        def close(self) -> None:
            closed.append(self.binding.concrete_symbol)

    def build_batch(environment, *, train_range, seed, n_envs):
        symbol = environment.binding.concrete_symbol
        calls.append((symbol, train_range, seed, n_envs))
        return SimpleNamespace(dataset_id=environment.binding.source_dataset_id)

    monkeypatch.setattr(
        module,
        "build_episode_oracle_batch_for_environment",
        build_batch,
    )

    batches = build_universal_oracle_batches(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        bindings=bindings,
        concrete_environment_factory=Child,
        fold_train_range=(11, 73),
        behavior_cloning_seed=29,
        n_envs=4,
    )

    assert tuple(batches) == ("AAAUSDT", "BBBUSDT")
    assert opened == ["AAAUSDT", "BBBUSDT"]
    assert closed == opened
    assert calls == [
        ("AAAUSDT", (11, 73), 29, 4),
        ("BBBUSDT", (11, 73), 29, 4),
    ]


def test_build_universal_oracle_batches_rejects_batch_dataset_identity_mismatch(
    monkeypatch,
) -> None:
    import pytest
    import trade_rl.workflows.universal_teacher_runtime as module
    from trade_rl.workflows.universal_teacher_runtime import (
        build_universal_oracle_batches,
    )

    binding = _binding("AAAUSDT")
    closed: list[bool] = []

    class Child:
        dataset = SimpleNamespace(dataset_id=binding.source_dataset_id)

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(
        module,
        "build_episode_oracle_batch_for_environment",
        lambda *_args, **_kwargs: SimpleNamespace(dataset_id=_digest("wrong-dataset")),
    )

    with pytest.raises(ValueError, match="dataset identity"):
        build_universal_oracle_batches(
            train_symbols=("AAAUSDT",),
            bindings=(binding,),
            concrete_environment_factory=lambda _binding: Child(),
            fold_train_range=(5, 30),
            behavior_cloning_seed=7,
            n_envs=2,
        )

    assert closed == [True]
