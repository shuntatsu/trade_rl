from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.workflows.universal_causal_alpha_teacher import (
    UniversalCausalAlphaTeacherPackage,
    build_universal_causal_alpha_teacher_package,
)


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


def test_package_builds_one_shared_teacher_identity(monkeypatch) -> None:
    import trade_rl.workflows.universal_causal_alpha_teacher as module

    symbols = ("AAAUSDT", "BBBUSDT")
    bindings = tuple(_binding(symbol) for symbol in symbols)
    opened: list[str] = []
    closed: list[str] = []

    class Environment:
        def __init__(self, binding: InstrumentDatasetBinding) -> None:
            self.binding = binding
            self.dataset = SimpleNamespace(
                dataset_id=binding.source_dataset_id,
                symbols=(binding.concrete_symbol,),
                n_symbols=1,
                n_bars=100,
            )
            self.minimum_start_index = 1
            self.episode_bars = 4
            self.decision_bars = 1
            self.config = SimpleNamespace(
                initial_state_modes=("cash",),
                episode_hours=720.0,
            )
            self.pre_trade_risk = SimpleNamespace(
                config=PreTradeRiskConfig(
                    max_gross=1.0,
                    max_abs_weight=1.0,
                    max_turnover=1.0,
                    entry_threshold=0.10,
                    exit_threshold=0.03,
                    no_trade_band=0.05,
                )
            )
            opened.append(binding.concrete_symbol)

        def initial_weights_for_reset(self, mode: str, start: int) -> np.ndarray:
            assert mode == "cash"
            return np.zeros(1, dtype=np.float64)

        def close(self) -> None:
            closed.append(self.binding.concrete_symbol)

    partitions: dict[str, object] = {}
    samples: dict[str, object] = {}

    def build_partition(environment, *, train_range):
        value = SimpleNamespace(
            contracts=(
                SimpleNamespace(
                    digest=_digest(
                        f"contract:{environment.binding.concrete_symbol}:0"
                    )
                ),
            ),
            selection_contracts=(
                SimpleNamespace(
                    digest=_digest(f"selection:{environment.binding.concrete_symbol}")
                ),
            ),
            holdout_contract=SimpleNamespace(
                digest=_digest(f"holdout:{environment.binding.concrete_symbol}")
            ),
            digest=_digest(f"partition:{environment.binding.concrete_symbol}"),
        )
        partitions[environment.binding.concrete_symbol] = value
        return value

    def build_samples(**kwargs):
        binding = kwargs["binding"]
        value = SimpleNamespace(digest=_digest(f"samples:{binding.concrete_symbol}"))
        samples[binding.concrete_symbol] = value
        return value

    candidate = SimpleNamespace(
        digest=_digest("candidate"),
        ridge=SimpleNamespace(digest=_digest("ridge")),
        controller=SimpleNamespace(digest=_digest("controller")),
    )
    selection = SimpleNamespace(
        selected_candidate_digest=candidate.digest,
        digest=_digest("selection"),
        candidates=(SimpleNamespace(candidate=candidate),),
    )
    monkeypatch.setattr(module, "build_chronological_episode_partition", build_partition)
    monkeypatch.setattr(module, "build_causal_alpha_symbol_samples", build_samples)
    monkeypatch.setattr(
        module,
        "validate_universal_causal_alpha_partitions",
        lambda **kwargs: dict(kwargs["partitions"]),
    )
    monkeypatch.setattr(
        module,
        "default_causal_alpha_candidate_grid",
        lambda _risk: (candidate,),
    )
    monkeypatch.setattr(
        module,
        "evaluate_causal_alpha_selection",
        lambda **_kwargs: selection,
    )

    batch_calls: list[tuple[str, str]] = []

    def build_batch(**kwargs):
        shared_digest = kwargs["teacher_config_digest"]
        symbol = kwargs["symbol"]
        batch_calls.append((symbol, shared_digest))
        return (
            SimpleNamespace(
                dataset_id=_digest(f"dataset:{symbol}"),
                teacher_config_digest=shared_digest,
                digest=_digest(f"batch:{symbol}"),
            ),
            SimpleNamespace(digest=_digest(f"evidence:{symbol}")),
        )

    monkeypatch.setattr(module, "build_causal_alpha_episode_batch", build_batch)

    package = build_universal_causal_alpha_teacher_package(
        train_symbols=symbols,
        bindings=bindings,
        concrete_environment_factory=Environment,
        instrument_context_provider=lambda *_args, **_kwargs: np.zeros((1, 9)),
        fold_train_range=(1, 80),
        feature_schema_digest=_digest("feature-schema"),
        episode_hours=720.0,
    )

    assert isinstance(package, UniversalCausalAlphaTeacherPackage)
    assert opened == list(symbols)
    assert closed == list(symbols)
    assert set(package.batches) == set(symbols)
    assert set(package.partitions) == set(symbols)
    assert set(package.samples) == set(symbols)
    assert package.selection is selection
    assert package.selected_candidate_digest == candidate.digest
    assert len({digest for _, digest in batch_calls}) == 1
    assert batch_calls[0][1] == package.teacher_config_digest
    assert all(
        batch.teacher_config_digest == package.teacher_config_digest
        for batch in package.batches.values()
    )
