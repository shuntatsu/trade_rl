from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.universal_features import (
    UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
    universal_feature_schema_digest_from_names,
)
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.workflows.universal_causal_alpha_teacher import (
    build_causal_alpha_symbol_samples,
)


class _Dataset:
    def __init__(self) -> None:
        self.dataset_id = content_digest("sample-dataset")
        self.symbols = ("AAAUSDT",)
        self.n_symbols = 1
        self.n_bars = 14
        self.regular_cadence = True
        self.feature_names = ("momentum", "volatility")
        base = np.arange(self.n_bars, dtype=np.float64)
        self.features = np.column_stack((base, base * 0.1)).reshape(self.n_bars, 1, 2)
        self.feature_available = np.ones_like(self.features, dtype=np.bool_)
        self.feature_available[5, 0, 1] = False
        self.open = (100.0 + base).reshape(-1, 1)
        self.close = (101.0 + base).reshape(-1, 1)
        self.tradable = np.ones((self.n_bars, 1), dtype=np.bool_)
        self.asset_active = np.ones((self.n_bars, 1), dtype=np.bool_)
        self.tradable[6, 0] = False

    def bars_for_hours(self, hours: float) -> int:
        if hours == 24.0:
            return 1
        if hours == 72.0:
            return 3
        raise ValueError(hours)


def _binding(dataset_id: str) -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol="AAAUSDT",
        source_dataset_id=dataset_id,
        symbol_dataset_digest=dataset_id,
        execution_metadata_digest=content_digest("metadata"),
        instrument_descriptor_digest=content_digest("descriptor"),
        split="train",
    )


class _ContextProvider:
    schema_digest = content_digest("context-schema")
    digest = content_digest("context-provider")

    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def __call__(self, environment, binding) -> np.ndarray:
        assert binding.concrete_symbol == "AAAUSDT"
        equity = float(environment.hybrid.portfolio_value)
        self.calls.append((environment.current_index, equity))
        return np.full(
            (1, len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES)),
            0.01 * environment.current_index,
            dtype=np.float32,
        )


def _environment(dataset: _Dataset) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset,
        minimum_start_index=2,
        initial_capital=1_000.0,
        decision_bars=1,
        config=SimpleNamespace(signal_delay_decisions=1, execution_cost=object()),
    )


def test_symbol_sample_extraction_is_train_scoped_causal_and_reference_equity_anchored() -> (
    None
):
    dataset = _Dataset()
    provider = _ContextProvider()
    environment = _environment(dataset)
    schema_digest = universal_feature_schema_digest_from_names(dataset.feature_names)

    samples = build_causal_alpha_symbol_samples(
        environment=environment,
        binding=_binding(dataset.dataset_id),
        instrument_context_provider=provider,
        train_range=(2, 12),
        feature_schema_digest=schema_digest,
    )

    assert samples.decision_indices.tolist() == [2, 3, 4, 5, 7, 8, 9, 10, 11]
    assert samples.feature_names == (
        *dataset.feature_names,
        *UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
    )
    assert samples.features.shape == (9, 11)
    assert samples.feature_available[3, 1] == np.bool_(False)
    assert all(equity == pytest.approx(1_000.0) for _, equity in provider.calls)
    assert [index for index, _ in provider.calls] == samples.decision_indices.tolist()
    assert samples.reference_equity_mode == "initial_capital"
    assert samples.reference_equity == pytest.approx(1_000.0)

    # 24h/72h labels use the first executable bar after the maintained one-decision delay.
    first = 0
    assert samples.label_end_indices_24h[first] == 4
    assert samples.labels_24h[first] == pytest.approx(
        np.log(dataset.close[4, 0] / dataset.open[4, 0])
    )
    assert samples.label_end_indices_72h[first] == 6
    assert samples.labels_72h[first] == pytest.approx(
        np.log(dataset.close[6, 0] / dataset.open[4, 0])
    )

    # Labels whose horizon leaves the train prefix are explicitly unavailable.
    last = -1
    assert samples.label_end_indices_24h[last] == -1
    assert np.isnan(samples.labels_24h[last])
    assert samples.label_end_indices_72h[last] == -1
    assert np.isnan(samples.labels_72h[last])


def test_symbol_sample_extraction_rejects_non_train_binding_and_schema_drift() -> None:
    dataset = _Dataset()
    environment = _environment(dataset)
    provider = _ContextProvider()
    binding = _binding(dataset.dataset_id)
    bad_binding = InstrumentDatasetBinding(
        concrete_symbol=binding.concrete_symbol,
        source_dataset_id=binding.source_dataset_id,
        symbol_dataset_digest=binding.symbol_dataset_digest,
        execution_metadata_digest=binding.execution_metadata_digest,
        instrument_descriptor_digest=binding.instrument_descriptor_digest,
        split="validation",
    )
    with pytest.raises(ValueError, match="train binding"):
        build_causal_alpha_symbol_samples(
            environment=environment,
            binding=bad_binding,
            instrument_context_provider=provider,
            train_range=(2, 12),
            feature_schema_digest=universal_feature_schema_digest_from_names(
                dataset.feature_names
            ),
        )
    with pytest.raises(ValueError, match="feature schema"):
        build_causal_alpha_symbol_samples(
            environment=environment,
            binding=binding,
            instrument_context_provider=provider,
            train_range=(2, 12),
            feature_schema_digest=content_digest("wrong-schema"),
        )
