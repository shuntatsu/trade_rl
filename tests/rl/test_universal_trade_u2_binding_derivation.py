from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any, Callable, cast

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import make_u1_market
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.data.market import MarketDataset
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding
from trade_rl.workflows.universal_trade_rl_u1_contract import UniversalTradeRLU1Contract
from trade_rl.workflows.universal_trade_rl_u2_preflight import (
    U2TrainingSource,
    U2TrainingSourceClosure,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import U2_DECISION_STEP_NS

_U2_ENVIRONMENT_MODULE = "trade_rl.workflows.universal_trade_rl_u2_environment"


def _binding_builder() -> Callable[..., tuple[InstrumentDatasetBinding, ...]]:
    module = importlib.import_module(_U2_ENVIRONMENT_MODULE)
    builder = getattr(module, "build_universal_trade_rl_u2_instrument_bindings", None)
    if not callable(builder):
        pytest.fail("U2 internal instrument binding derivation is not implemented")
    return cast(Callable[..., tuple[InstrumentDatasetBinding, ...]], builder)


def _timestamp_ns(value: np.datetime64) -> int:
    return int(value.astype("datetime64[ns]").astype(np.int64))


def _u1_contract() -> UniversalTradeRLU1Contract:
    return UniversalTradeRLU1Contract(
        universe_manifest_digest="1" * 64,
        u0_identity_digest="2" * 64,
        policy_contract_digest="3" * 64,
        normalizer_digest="4" * 64,
        normalizer_provenance_digest="5" * 64,
        normalizer_knowledge_cutoff_ns=1,
        normalizer_clip_value=10.0,
        observation_schema_digest="6" * 64,
        state_layout_digest="7" * 64,
        policy_state_fields=("current_weight",),
        runtime_config_digest="8" * 64,
        execution_policy_digest="9" * 64,
        pretrade_risk_digest="a" * 64,
        portfolio_risk_digest="b" * 64,
    )


def _source_and_fit(
    *,
    symbol: str = "BTCUSDT",
    fit_start: int = 10,
    fit_stop: int = 100,
) -> tuple[U2TrainingSource, MarketDataset]:
    source_dataset = make_u1_market(symbol=symbol, n_bars=256)
    fit_dataset = MarketDatasetView(source_dataset, fit_start, fit_stop).materialize()
    fit_first = _timestamp_ns(fit_dataset.timestamps[0])
    fit_last = _timestamp_ns(fit_dataset.timestamps[-1])
    source = U2TrainingSource(
        symbol=symbol,
        dataset_digest=source_dataset.dataset_id,
        source_first_timestamp_ns=_timestamp_ns(source_dataset.timestamps[0]),
        source_last_timestamp_ns=_timestamp_ns(source_dataset.timestamps[-1]),
        source_row_count=source_dataset.n_bars,
        fit_first_timestamp_ns=fit_first,
        fit_last_timestamp_ns=fit_last,
        fit_stop_timestamp_ns_exclusive=fit_last + U2_DECISION_STEP_NS,
        fit_bar_count=fit_dataset.n_bars,
    )
    return source, fit_dataset


def _closure(
    *,
    sources: tuple[U2TrainingSource, ...],
    u1_contract: UniversalTradeRLU1Contract,
) -> U2TrainingSourceClosure:
    first = sources[0]
    return U2TrainingSourceClosure(
        u2_contract_digest="c" * 64,
        universe_manifest_digest=u1_contract.universe_manifest_digest,
        u1_contract_digest=u1_contract.digest,
        normalizer_digest=u1_contract.normalizer_digest,
        normalizer_provenance_digest=u1_contract.normalizer_provenance_digest,
        time_partition_digest="d" * 64,
        fit_first_timestamp_ns=first.fit_first_timestamp_ns,
        fit_last_timestamp_ns=first.fit_last_timestamp_ns,
        fit_stop_timestamp_ns_exclusive=first.fit_stop_timestamp_ns_exclusive,
        fit_bar_count=first.fit_bar_count,
        sources=sources,
    )


def test_u2_derives_binding_from_fit_view_and_frozen_u1_identities() -> None:
    u1_contract = _u1_contract()
    source, fit_dataset = _source_and_fit()
    closure = _closure(sources=(source,), u1_contract=u1_contract)

    bindings = _binding_builder()(
        closure=closure,
        fit_datasets={source.symbol: fit_dataset},
        u1_contract=u1_contract,
    )

    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.concrete_symbol == source.symbol
    assert binding.source_dataset_id == fit_dataset.dataset_id
    assert binding.source_dataset_id != source.dataset_digest
    assert binding.symbol_dataset_digest == source.dataset_digest
    assert binding.split == "train"
    assert binding.execution_metadata_digest == content_digest(
        {
            "schema_version": "universal_trade_rl_u2_execution_binding_v1",
            "fit_dataset_id": fit_dataset.dataset_id,
            "u1_execution_policy_digest": u1_contract.execution_policy_digest,
            "u1_pretrade_risk_digest": u1_contract.pretrade_risk_digest,
            "u1_portfolio_risk_digest": u1_contract.portfolio_risk_digest,
        }
    )
    assert binding.instrument_descriptor_digest == content_digest(
        {
            "schema_version": "universal_trade_rl_u2_instrument_descriptor_disabled_v1",
            "instrument_context_enabled": False,
            "v4_context_enabled": False,
        }
    )


@pytest.mark.parametrize(
    "fit_datasets",
    (
        {},
        {"BTCUSDT": None, "ETHUSDT": None},
    ),
)
def test_u2_binding_derivation_requires_exact_fit_symbol_closure(
    fit_datasets: dict[str, Any],
) -> None:
    u1_contract = _u1_contract()
    source, fit_dataset = _source_and_fit()
    closure = _closure(sources=(source,), u1_contract=u1_contract)
    resolved = {
        symbol: (fit_dataset if dataset is None else dataset)
        for symbol, dataset in fit_datasets.items()
    }

    with pytest.raises(ValueError, match="FIT|fit|symbol|closure"):
        _binding_builder()(
            closure=closure,
            fit_datasets=resolved,
            u1_contract=u1_contract,
        )


@pytest.mark.parametrize(
    "replacement_view",
    (
        (11, 101),
        (10, 99),
    ),
)
def test_u2_binding_derivation_rejects_fit_bounds_drift(
    replacement_view: tuple[int, int],
) -> None:
    u1_contract = _u1_contract()
    source, _fit_dataset = _source_and_fit()
    closure = _closure(sources=(source,), u1_contract=u1_contract)
    source_dataset = make_u1_market(symbol=source.symbol, n_bars=source.source_row_count)
    drifted = MarketDatasetView(source_dataset, *replacement_view).materialize()

    with pytest.raises(ValueError, match="FIT|fit|timestamp|bar|bound"):
        _binding_builder()(
            closure=closure,
            fit_datasets={source.symbol: drifted},
            u1_contract=u1_contract,
        )


def test_u2_binding_derivation_requires_matching_u1_contract_identity() -> None:
    u1_contract = _u1_contract()
    source, fit_dataset = _source_and_fit()
    closure = _closure(sources=(source,), u1_contract=u1_contract)
    drifted_contract = replace(u1_contract, runtime_config_digest="e" * 64, digest="")

    with pytest.raises(ValueError, match="U1|contract|identity"):
        _binding_builder()(
            closure=closure,
            fit_datasets={source.symbol: fit_dataset},
            u1_contract=drifted_contract,
        )
