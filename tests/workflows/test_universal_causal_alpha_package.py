from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaCostAwareConfig,
    CausalAlphaTeacherHoldoutMetric,
    evaluate_causal_alpha_teacher_admission,
)
from trade_rl.risk.portfolio import PortfolioRiskConfig
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.simulation.execution import ExecutionCostConfig
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


def test_package_builds_one_shared_teacher_identity(
    monkeypatch, tmp_path: Path
) -> None:
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
                execution_cost=ExecutionCostConfig(),
                signal_delay_decisions=1,
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
            self.portfolio_risk = SimpleNamespace(
                config=PortfolioRiskConfig(max_position_to_market_notional=0.02)
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
        del train_range
        value = SimpleNamespace(
            contracts=(
                SimpleNamespace(
                    digest=_digest(f"contract:{environment.binding.concrete_symbol}:0")
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
        value = SimpleNamespace(
            digest=_digest(f"samples:{binding.concrete_symbol}"),
            context_digest=_digest(f"context:{binding.concrete_symbol}"),
            feature_schema_digest=_digest("feature-schema"),
            reference_equity=100_000.0,
            reference_equity_mode="fixed",
        )
        samples[binding.concrete_symbol] = value
        return value

    candidate = SimpleNamespace(
        digest=_digest("candidate"),
        ridge=SimpleNamespace(digest=_digest("ridge")),
        controller=SimpleNamespace(digest=_digest("controller")),
        economic_controller=CausalAlphaCostAwareConfig(
            execution_cost_multiplier=1.5,
            edge_margin=0.001,
            confirmation_count=2,
            strong_reversal_threshold=0.02,
            max_abs_target=0.5,
            max_position_to_market_notional=0.02,
        ),
    )
    selection_payload = {
        "schema_version": "causal_alpha_selection_evidence_v2",
        "artifact_digest": _digest("selection"),
        "selected_candidate_digest": candidate.digest,
    }
    selection = SimpleNamespace(
        selected_candidate_digest=candidate.digest,
        digest=_digest("selection"),
        candidates=(SimpleNamespace(candidate=candidate),),
        to_payload=lambda: dict(selection_payload),
    )
    monkeypatch.setattr(
        module, "build_chronological_episode_partition", build_partition
    )
    monkeypatch.setattr(module, "build_causal_alpha_symbol_samples", build_samples)
    monkeypatch.setattr(
        module,
        "validate_universal_causal_alpha_partitions",
        lambda **kwargs: dict(kwargs["partitions"]),
    )
    grid_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "default_cost_aware_causal_alpha_candidate_grid",
        lambda **kwargs: grid_calls.append(dict(kwargs)) or (candidate,),
    )
    progress_payload = {
        "completed_replays": 1,
        "phase": "causal_teacher_selection_v2",
        "total_replays": 1,
    }

    def evaluate_selection(**kwargs):
        kwargs["progress_callback"](progress_payload)
        return selection

    monkeypatch.setattr(
        module, "evaluate_cost_aware_causal_alpha_selection", evaluate_selection
    )
    monkeypatch.setattr(
        module,
        "evaluate_causal_alpha_selection",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("canonical package must not use v1 selection")
        ),
    )
    monkeypatch.setattr(
        module,
        "CausalAlphaExpandingFitCache",
        lambda **_kwargs: object(),
    )

    batch_calls: list[tuple[str, str]] = []

    def build_batch(**kwargs):
        shared_digest = kwargs["teacher_config_digest"]
        symbol = kwargs["symbol"]
        batch_calls.append((symbol, shared_digest))
        assert kwargs["economic_controller_config"] is candidate.economic_controller
        assert isinstance(kwargs["execution_cost"], ExecutionCostConfig)
        assert kwargs["signal_delay_decisions"] == 1
        assert kwargs["decision_bars"] == 1
        evidence_payload = {
            "artifact_digest": _digest(f"evidence:{symbol}"),
            "schema_version": "test_batch_evidence_v1",
        }
        return (
            SimpleNamespace(
                dataset_id=_digest(f"dataset:{symbol}"),
                teacher_config_digest=shared_digest,
                digest=_digest(f"batch:{symbol}"),
            ),
            SimpleNamespace(
                digest=evidence_payload["artifact_digest"],
                to_payload=lambda payload=evidence_payload: dict(payload),
            ),
        )

    monkeypatch.setattr(module, "build_causal_alpha_episode_batch", build_batch)

    admission = evaluate_causal_alpha_teacher_admission(
        tuple(
            CausalAlphaTeacherHoldoutMetric(
                symbol=symbol,
                gross_return=0.02,
                net_return=0.01,
                turnover_per_day=0.1,
                total_execution_cost=0.2,
                trade_count=4,
                maximum_drawdown=0.03,
            )
            for symbol in symbols
        )
    )
    admission_calls: list[tuple[str, ...]] = []
    selection_path = tmp_path / "causal-teacher-selection.json"

    def evaluate_holdouts(**kwargs):
        assert selection_path.is_file(), (
            "selection must be persisted before holdout replay"
        )
        persisted = json.loads(selection_path.read_text(encoding="utf-8"))
        assert persisted == selection_payload
        admission_calls.append(tuple(kwargs["train_symbols"]))
        return admission

    monkeypatch.setattr(
        module,
        "evaluate_causal_alpha_teacher_holdouts",
        evaluate_holdouts,
        raising=False,
    )

    package = build_universal_causal_alpha_teacher_package(
        train_symbols=symbols,
        bindings=bindings,
        concrete_environment_factory=Environment,
        instrument_context_provider=lambda *_args, **_kwargs: np.zeros((1, 9)),
        fold_train_range=(1, 80),
        feature_schema_digest=_digest("feature-schema"),
        episode_hours=720.0,
        selection_evidence_path=selection_path,
    )

    assert grid_calls[0]["max_position_to_market_notional"] == 0.02

    assert isinstance(package, UniversalCausalAlphaTeacherPackage)
    assert opened == list(symbols)
    assert closed == list(symbols)
    assert set(package.batches) == set(symbols)
    assert set(package.partitions) == set(symbols)
    assert set(package.samples) == set(symbols)
    assert package.selection is selection
    assert package.teacher_admission is admission
    assert admission_calls == [symbols]
    assert package.selected_candidate_digest == candidate.digest
    assert len({digest for _, digest in batch_calls}) == 1
    assert batch_calls[0][1] == package.teacher_config_digest
    assert all(
        batch.teacher_config_digest == package.teacher_config_digest
        for batch in package.batches.values()
    )
    admission_path = tmp_path / "causal-teacher-admission.json"
    package_path = tmp_path / "causal-teacher-package.json"
    progress_path = tmp_path / "causal-teacher-progress.json"
    assert json.loads(progress_path.read_text(encoding="utf-8")) == {
        **progress_payload,
        "package_digest": package.digest,
        "phase": "causal_teacher_package_completed",
        "teacher_admission_digest": admission.digest,
        "teacher_admission_passed": True,
    }
    assert json.loads(admission_path.read_text(encoding="utf-8")) == (
        admission.to_payload()
    )
    assert json.loads(package_path.read_text(encoding="utf-8")) == (
        package.to_payload()
    )
