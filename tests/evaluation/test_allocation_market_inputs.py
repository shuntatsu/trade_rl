"""Synthetic canonical preparation with independent allocation and refusal oracles."""

from __future__ import annotations

import importlib
import io
import json
import zipfile
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from hashlib import sha256

import numpy as np
import pytest

from tests.strategies.test_simple_return_stream import blocks, time
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data import write_market_dataset_files
from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.identity import (
    _LEGACY_MARKET_DATASET_IDENTITY_SCHEMA,
    canonical_identity_json,
    compute_market_dataset_id,
    parse_identity_json,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.evaluation.forecast_allocation import (
    HorizonCostEstimates,
    propose_forecast_target,
)
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import OrderBookState
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.forecasts.simple_stream_io import (
    load_simple_return_stream_artifact,
)


def api():
    name = "trade_rl.evaluation.allocation_market_inputs"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        pytest.fail(
            "Missing canonical allocation market input preparation", pytrace=False
        )


def market(*, future=False):
    times = time(0) + np.arange(18) * np.timedelta64(1, "h")
    series = {}
    for symbol in ("ALPHA", "BETA"):
        close = np.array([100, 200, 100, 200, 100] + [100] * 13, dtype=float)
        if symbol == "BETA":
            close[:] = 50
        if future:
            close[14:] *= 7
        high = 1.25 * close
        if future:
            high[14:] = 1.5 * close[14:]
        series[symbol] = RawMarketSeries(
            timestamps=times,
            open=close,
            high=high,
            low=0.75 * close,
            close=close,
            volume=np.full(18, 100_000.0),
            funding_rate=np.zeros(18),
            tradable=np.ones(18, dtype=np.bool_),
        )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(name="body", kind=FeatureKind.BODY_RETURN),
            FeatureSpec(name="range", kind=FeatureKind.HIGH_LOW_RANGE),
        ),
    )
    return MarketDatasetBuilder(config).build(
        InMemoryMarketDataSource(series),
        tuple(
            InstrumentContract(s, datetime(2025, 1, 1, tzinfo=timezone.utc))
            for s in series
        ),
    )


def arguments(tmp_path, dataset=None, *, saved_format="v4_v7"):
    dataset = market() if dataset is None else dataset
    if saved_format.endswith("_v6"):
        identity = parse_identity_json(dataset.identity_payload_json)
        identity["schema"] = _LEGACY_MARKET_DATASET_IDENTITY_SCHEMA
        dataset = replace(
            dataset,
            dataset_id=compute_market_dataset_id(identity, dataset.identity_arrays()),
            identity_payload_json=canonical_identity_json(identity),
        )
    root = tmp_path / "dataset"
    write_market_dataset_files(root, dataset)
    if saved_format == "v3_v6":
        # Encode the old field roster without changing any retained .npy bytes.
        arrays_path = root / "arrays.npz"
        encoded = io.BytesIO()
        with (
            zipfile.ZipFile(io.BytesIO(arrays_path.read_bytes())) as original,
            zipfile.ZipFile(encoded, "w") as legacy,
        ):
            for info in original.infolist():
                if info.filename != "funding_price_rate.npy":
                    legacy.writestr(info, original.read(info.filename))
        payload = encoded.getvalue()
        arrays_path.write_bytes(payload)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("artifact_digest")
        manifest["arrays"].pop("funding_price_rate")
        manifest["schema_version"] = "market_dataset_artifact_v3"
        manifest["arrays_digest"] = sha256(payload).hexdigest()
        manifest["artifact_digest"] = sha256(canonical_json_bytes(manifest)).hexdigest()
        manifest_path.write_bytes(canonical_json_bytes(manifest))
    costs = tuple(
        HorizonCostEstimates(
            symbol=s,
            decision_time=t,
            available_at=t,
            horizon_end=t + np.timedelta64(1, "h"),
            source_identity=f"synthetic-reviewed-{s}",
            buy_cost=0.025 if s == "ALPHA" else 0.0,
        ).payload()
        for t in dataset.timestamps
        if time(6) <= t < time(8) or time(10) <= t < time(14)
        for s in dataset.symbols
    )
    return dict(
        dataset_root=root,
        stream_path=tmp_path / "stream.json",
        expected_dataset_id=dataset.dataset_id,
        development_start=time(0),
        development_stop=time(18),
        feature_names=("range", "body"),
        fit_symbols=("ALPHA",),
        blocks=blocks(),
        horizon_hours=1,
        alpha=1.0,
        cost_rows=tuple(reversed(costs)),
    )


def forbid_fit_and_publication(module, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Invalid preparation reached fitting or publication")

    monkeypatch.setattr(module, "fit_prequential_simple_ridge", forbidden)
    monkeypatch.setattr(module, "publish_simple_return_stream_artifact", forbidden)


@pytest.mark.parametrize("saved_format", ("v3_v6", "v4_v6", "v4_v7"))
def test_saved_canonical_inputs_reach_literal_allocation_without_lane_refitting(
    tmp_path, monkeypatch, saved_format
):
    module = api()
    loader, producer, publisher = (
        module.load_market_dataset_artifact,
        module.fit_prequential_simple_ridge,
        module.publish_simple_return_stream_artifact,
    )
    calls = []

    def load(*args, **kwargs):
        calls.append("load")
        return loader(*args, **kwargs)

    def fit(*args, **kwargs):
        calls.append("fit")
        return producer(*args, **kwargs)

    def publish(*args, **kwargs):
        calls.append("publish")
        return publisher(*args, **kwargs)

    monkeypatch.setattr(module, "load_market_dataset_artifact", load)
    monkeypatch.setattr(module, "fit_prequential_simple_ridge", fit)
    monkeypatch.setattr(module, "publish_simple_return_stream_artifact", publish)
    supplied = arguments(tmp_path, saved_format=saved_format)
    prepared = module.prepare_allocation_market_inputs(**supplied)
    assert calls == ["load", "fit", "publish"]
    assert prepared.dataset.dataset_id == supplied["expected_dataset_id"]
    assert prepared.stream.vintages[0].model.feature_indices == (1, 0)
    assert {p.symbol for p in prepared.stream.packets} == {"ALPHA", "BETA"}
    assert prepared.stream.packets[0].feature_values == (0.5, 0.0)
    assert tuple(c[0].symbol for c in prepared.costs_by_symbol) == ("ALPHA", "BETA")
    assert all(
        tuple(c.decision_time for c in group)
        == (time(6), time(7), time(10), time(11), time(12), time(13))
        for group in prepared.costs_by_symbol
    )
    assert prepared.costs_by_symbol[1][0].buy_cost == 0.0
    assert (
        sha256(prepared.artifact.path.read_bytes()).hexdigest()
        == prepared.stream.digest
    )
    stream = load_simple_return_stream_artifact(
        prepared.artifact.path,
        expected_digest=prepared.artifact.digest,
        expected_dataset_id=prepared.artifact.dataset_id,
    )
    executor = MarketExecutor(prepared.dataset, ExecutionCostConfig.zero())
    book = BookState(
        quantities=np.zeros(2),
        cash=1000.0,
        mark_prices=prepared.dataset.close[6],
        peak_value=1000.0,
        as_of_index=6,
        as_of_dataset_id=prepared.dataset.dataset_id,
    )
    proposal = propose_forecast_target(
        executor,
        book,
        OrderBookState.empty(),
        account_id="independent-ALPHA",
        stream=stream,
        estimates=prepared.costs_by_symbol[0][0],
        allocator=AfterCostTargetAllocator(
            lower_weight=0.0, upper_weight=1.0, risk_aversion=1.0
        ),
        pretrade_risk=PreTradeRisk(
            PreTradeRiskConfig(max_abs_weight=1.0, max_turnover=None)
        ),
        symbol_index=0,
        start_index=6,
        expected_horizon_seconds=3600,
    )
    # Four equally weighted mature returns: 1,-.5,1,-.5. Max (.25-.025)w-.5625w².
    assert proposal.inputs.expected_simple_return == pytest.approx(0.25)
    assert proposal.inputs.return_variance == pytest.approx(0.5625)
    assert proposal.target_weight == pytest.approx(0.2)
    with pytest.raises(FrozenInstanceError):
        prepared.dataset = None


@pytest.mark.parametrize(
    "failure",
    (
        "pin",
        "lineage",
        "normalization",
        "period",
        "mark",
        "subset",
        "feature",
        "symbol",
        "delay",
        "availability",
    ),
)
def test_invalid_dataset_and_scope_are_rejected_before_fit(
    tmp_path, monkeypatch, failure
):
    module = api()
    dataset = market()
    identity = parse_identity_json(dataset.identity_payload_json)
    if failure == "lineage":
        identity = {"source_dataset": {"identity_payload": identity}}
    elif failure == "normalization":
        identity["config"]["features"][0]["normalization"] = "full_sample_zscore"
    elif failure == "subset":
        identity["config"]["features"][1]["kind"] = "cross_sectional_momentum_rank"
    elif failure == "mark":
        marks = dataset.close.copy()
        marks[6, 1] *= 1.01
        dataset = replace(dataset, mark_price=marks, identity_payload_json=None)
    elif failure == "availability":
        available = dataset.feature_available.copy()
        available[6, 1, 0] = False
        dataset = replace(
            dataset,
            feature_available=available,
            feature_staleness=None,
            identity_payload_json=None,
        )
    dataset = dataset.with_content_identity(identity)
    supplied = arguments(tmp_path, dataset)
    if failure == "pin":
        supplied["expected_dataset_id"] = "0" * 64
    elif failure == "period":
        supplied["development_start"] = time(1)
    elif failure == "feature":
        supplied["feature_names"] = ("missing",)
    elif failure == "symbol":
        supplied["fit_symbols"] = ("missing",)
    elif failure == "delay":
        supplied["blocks"] = blocks(delay_seconds=1)
    forbid_fit_and_publication(module, monkeypatch)
    with pytest.raises(ValueError):
        module.prepare_allocation_market_inputs(**supplied)
    assert not supplied["stream_path"].exists()


@pytest.mark.parametrize(
    "rate",
    (
        "buy_cost",
        "sell_cost",
        "exit_cost",
        "funding_return",
        "borrow_return",
        "cash_return",
    ),
)
def test_omitted_rate_is_not_an_explicit_zero(tmp_path, monkeypatch, rate):
    module = api()
    supplied = arguments(tmp_path)
    rows = list(supplied["cost_rows"])
    rows[0] = {k: v for k, v in rows[0].items() if k != rate}
    supplied["cost_rows"] = tuple(rows)
    forbid_fit_and_publication(module, monkeypatch)
    with pytest.raises(ValueError):
        module.prepare_allocation_market_inputs(**supplied)


@pytest.mark.parametrize(
    "failure",
    (
        "missing",
        "extra",
        "duplicate",
        "future",
        "horizon",
        "clock_bool",
        "clock_float",
        "clock_nat",
        "basis",
        "schema",
        "source",
    ),
)
def test_bad_cost_roster_and_clocks_fail_before_fit(tmp_path, monkeypatch, failure):
    module = api()
    supplied = arguments(tmp_path)
    rows = list(supplied["cost_rows"])
    if failure == "missing":
        rows.pop()
    elif failure == "extra":
        rows += [rows[0] | {"symbol": "UNDECLARED"}]
    elif failure == "duplicate":
        rows += [dict(rows[0])]
    else:
        changes = {
            "future": {"available_at": int(time(17).astype(np.int64))},
            "horizon": {"horizon_end": int(time(17).astype(np.int64))},
            "clock_bool": {"decision_time": True},
            "clock_float": {"decision_time": float(time(13).astype(np.int64))},
            "clock_nat": {"decision_time": -(2**63)},
            "basis": {"cost_basis": "per_bar_rates"},
            "schema": {"schema": "other"},
            "source": {"source_identity": ""},
        }
        rows[0] = rows[0] | changes[failure]
    supplied["cost_rows"] = tuple(rows)
    forbid_fit_and_publication(module, monkeypatch)
    with pytest.raises(ValueError):
        module.prepare_allocation_market_inputs(**supplied)


def test_future_suffix_preserves_earlier_predictions_and_changes_whole_content(
    tmp_path,
):
    module = api()
    a = module.prepare_allocation_market_inputs(**arguments(tmp_path / "a"))
    b = module.prepare_allocation_market_inputs(
        **arguments(tmp_path / "b", market(future=True))
    )
    assert a.dataset.dataset_id != b.dataset.dataset_id
    assert a.dataset.normalization_digest != b.dataset.normalization_digest
    assert a.stream.digest != b.stream.digest
    assert a.stream.causal_scope_digest == b.stream.causal_scope_digest
    assert [p.digest for p in a.stream.packets] == [p.digest for p in b.stream.packets]


def test_preexisting_stream_survives_preparation(tmp_path):
    module = api()
    supplied = arguments(tmp_path)
    supplied["stream_path"].write_bytes(b"prior owner")
    with pytest.raises(FileExistsError):
        module.prepare_allocation_market_inputs(**supplied)
    assert supplied["stream_path"].read_bytes() == b"prior owner"


def generated_arguments(tmp_path, dataset=None, *, saved_format="v4_v7"):
    from tests.evaluation.test_allocation_costs import market as cost_market
    from tests.evaluation.test_allocation_costs import recipe
    from trade_rl.strategies.forecasts.stream import ForecastBlock

    supplied = arguments(
        tmp_path,
        cost_market() if dataset is None else dataset,
        saved_format=saved_format,
    )
    supplied.pop("cost_rows")
    supplied.update(
        blocks=(ForecastBlock(time(20), time(20.25), time(24), time(26)),),
        development_stop=time(72),
        feature_names=("range",),
        cost_recipe=recipe(),
    )
    return supplied


@pytest.mark.parametrize("saved_format", ("v3_v6", "v4_v6", "v4_v7"))
def test_generated_costs_share_exact_recipe_and_single_loader_fit_publication(
    tmp_path, monkeypatch, saved_format
):
    from tests.evaluation.test_allocation_costs import market as cost_market

    module = api()
    dataset = cost_market()
    if saved_format == "v4_v7":
        products = np.zeros((72, 2))
        # Settlement products differ from the unchanged bar-close rate proxies.
        products[8], products[16] = (0.0095, 0.0045), (0.051, 0.026)
        identity = parse_identity_json(dataset.identity_payload_json)
        dataset = replace(
            dataset, funding_price_rate=products, identity_payload_json=None
        ).with_content_identity(identity)
    supplied = generated_arguments(tmp_path, dataset, saved_format=saved_format)
    calls = []
    for name in (
        "load_market_dataset_artifact",
        "estimate_declared_horizon_costs",
        "fit_prequential_simple_ridge",
        "publish_simple_return_stream_artifact",
    ):
        owner = getattr(module, name, None)
        assert owner is not None, f"Missing concrete preparation owner {name}"

        def tracked(*args, _owner=owner, _name=name, **kwargs):
            calls.append(_name)
            return _owner(*args, **kwargs)

        monkeypatch.setattr(module, name, tracked)
    prepared = module.prepare_allocation_market_inputs(**supplied)
    assert calls == [
        "load_market_dataset_artifact",
        "estimate_declared_horizon_costs",
        "fit_prequential_simple_ridge",
        "publish_simple_return_stream_artifact",
    ]
    assert prepared.cost_recipe is supplied["cost_recipe"]
    assert prepared.dataset.dataset_id == supplied["expected_dataset_id"]
    np.testing.assert_array_equal(
        prepared.dataset.funding_price_rate, dataset.funding_price_rate
    )
    assert prepared.cost_recipe.execution_cost is supplied["cost_recipe"].execution_cost
    assert tuple(
        tuple(c.decision_time for c in group) for group in prepared.costs_by_symbol
    ) == (
        (time(24), time(25)),
        (time(24), time(25)),
    )
    for group in prepared.costs_by_symbol:
        assert group[0].buy_cost == pytest.approx(0.0434)
        assert group[0].funding_return == pytest.approx(0.000025)
        assert group[0].borrow_return == pytest.approx(0.00001875)
        assert group[0].cash_return == 0


@pytest.mark.parametrize("failure", ("build_v2", "derived", "mark"))
def test_legacy_compatibility_preserves_before_fit_source_and_price_gates(
    tmp_path, monkeypatch, failure
):
    module = api()
    dataset = market()
    identity = parse_identity_json(dataset.identity_payload_json)
    if failure == "build_v2":
        identity["config"]["schema_version"] = "market_build_v2"
    elif failure == "derived":
        identity = {"source_dataset": {"identity_payload": identity}}
    else:
        marks = dataset.close.copy()
        marks[6, 1] *= 1.01
        dataset = replace(dataset, mark_price=marks, identity_payload_json=None)
    dataset = dataset.with_content_identity(identity)
    supplied = arguments(tmp_path, dataset, saved_format="v3_v6")
    # The real loader must succeed; only preparation may reject these sources.
    loaded = module.load_market_dataset_artifact(supplied["dataset_root"])
    assert loaded.dataset_id == supplied["expected_dataset_id"]
    forbid_fit_and_publication(module, monkeypatch)
    reason = "same-close" if failure == "mark" else "direct canonical"
    with pytest.raises(ValueError, match=reason):
        module.prepare_allocation_market_inputs(**supplied)
    assert not supplied["stream_path"].exists()


def test_v7_cancelled_rates_keep_nonzero_settlement_products_out_of_rate_projection(
    tmp_path,
):
    from tests.evaluation.test_allocation_costs import market as cost_market

    module = api()
    original = cost_market()
    rates, counts = original.funding_rate.copy(), original.funding_event_count.copy()
    rates[[8, 16], 0], counts[[8, 16], 0] = 0, 2
    identity = parse_identity_json(original.identity_payload_json)
    ids = []
    for factor in (1, 2):
        products = original.funding_price_rate.copy()
        # .0001*(95-105), .0002*(90-110): zero rates, nonzero cashflow products.
        products[[8, 16], 0] = (-0.001 * factor, -0.004 * factor)
        dataset = replace(
            original,
            funding_rate=rates,
            funding_event_count=counts,
            funding_price_rate=products,
            identity_payload_json=None,
        ).with_content_identity(identity)
        supplied = generated_arguments(tmp_path / str(factor), dataset)
        prepared = module.prepare_allocation_market_inputs(**supplied)
        assert prepared.dataset.dataset_id == dataset.dataset_id
        np.testing.assert_array_equal(prepared.dataset.funding_price_rate, products)
        assert prepared.costs_by_symbol[0][0].funding_return == 0
        assert prepared.costs_by_symbol[1][0].funding_return == pytest.approx(0.000025)
        ids.append(prepared.dataset.dataset_id)
    assert ids[0] != ids[1]


@pytest.mark.parametrize("paths", ("neither", "both", "wrong_type", "future"))
def test_invalid_cost_paths_and_declarations_fail_before_load(
    tmp_path, monkeypatch, paths
):
    module = api()
    supplied = arguments(tmp_path)
    supplied.pop("cost_rows")
    if paths != "neither":
        from tests.evaluation.test_allocation_costs import recipe

        supplied["cost_recipe"] = recipe()
        if paths == "both":
            supplied["cost_rows"] = ()
        elif paths == "wrong_type":
            supplied["cost_recipe"] = object()
        elif paths == "future":
            supplied["cost_recipe"] = recipe(assumptions_available_at=time(7))

    def forbidden(*_args, **_kwargs):
        pytest.fail("Invalid cost declaration reached Dataset loading")

    monkeypatch.setattr(module, "load_market_dataset_artifact", forbidden)
    with pytest.raises(ValueError):
        module.prepare_allocation_market_inputs(**supplied)


def test_generated_rejection_and_incomplete_payload_fail_before_fit(
    tmp_path, monkeypatch
):
    from tests.evaluation.test_allocation_costs import recipe

    module = api()
    supplied = generated_arguments(tmp_path / "source")
    supplied["cost_recipe"] = recipe(reference_notional=1501)
    forbid_fit_and_publication(module, monkeypatch)
    with pytest.raises(ValueError, match="capacity"):
        module.prepare_allocation_market_inputs(**supplied)
    supplied = generated_arguments(tmp_path / "payload")
    producer = getattr(module, "estimate_declared_horizon_costs", None)
    assert producer is not None, (
        "Missing concrete estimator for complete-cost admission"
    )

    def incomplete(*args, **kwargs):
        return producer(*args, **kwargs)[:-1]

    monkeypatch.setattr(module, "estimate_declared_horizon_costs", incomplete)
    with pytest.raises(ValueError, match="cover every"):
        module.prepare_allocation_market_inputs(**supplied)


def test_explicit_rows_and_old_four_field_positional_result_remain_compatible(tmp_path):
    module = api()
    prepared = module.prepare_allocation_market_inputs(**arguments(tmp_path))
    assert getattr(prepared, "cost_recipe", "missing") is None
    old = module.PreparedAllocationMarketInputs(
        prepared.dataset, prepared.stream, prepared.artifact, prepared.costs_by_symbol
    )
    assert old.cost_recipe is None


def signature_arguments(tmp_path, parent=None, *, include_volume=False):
    from trade_rl.data.features.signature import with_path_signatures

    parent = market() if parent is None else parent
    augmented = with_path_signatures(
        parent, window_bars=3, depth=2, include_volume=include_volume
    )
    supplied = arguments(tmp_path, augmented)
    parent_root = tmp_path / "parent"
    write_market_dataset_files(parent_root, parent)
    supplied.update(
        signature_parent_root=parent_root,
        expected_signature_parent_dataset_id=parent.dataset_id,
        feature_names=(augmented.feature_names[parent.n_features],),
        fit_symbols=parent.symbols,
    )
    return supplied


@pytest.mark.parametrize("include_volume", (False, True))
def test_pinned_signature_reaches_literal_fit_allocation_and_canonical_cash(
    tmp_path, monkeypatch, include_volume
):
    from trade_rl.evaluation.forecast_allocation import execute_forecast_proposal

    module = api()
    supplied = signature_arguments(tmp_path, include_volume=include_volume)
    calls = []
    for name in (
        "fit_prequential_simple_ridge",
        "publish_simple_return_stream_artifact",
    ):
        owner = getattr(module, name)

        def tracked(*args, _owner=owner, _name=name, **kwargs):
            calls.append(_name)
            return _owner(*args, **kwargs)

        monkeypatch.setattr(module, name, tracked)
    prepared = module.prepare_allocation_market_inputs(**supplied)
    assert calls == [
        "fit_prequential_simple_ridge",
        "publish_simple_return_stream_artifact",
    ]
    assert prepared.stream.dataset_id == supplied["expected_dataset_id"]
    assert (
        prepared.stream.dataset_id != supplied["expected_signature_parent_dataset_id"]
    )
    first, second = prepared.stream.vintages
    # Warmup excludes starts0,1; mature endpoint must be strictly BEFORE cutoff5.
    assert first.training.n_samples == 4
    assert first.training.trace.row_symbols == ("ALPHA", "ALPHA", "BETA", "BETA")
    np.testing.assert_array_equal(
        first.training.trace.start_times, [time(2), time(3), time(2), time(3)]
    )
    np.testing.assert_array_equal(
        first.training.trace.end_times, [time(3), time(4), time(3), time(4)]
    )
    np.testing.assert_allclose(first.training.labels, [1, -0.5, 0, 0])
    np.testing.assert_array_equal(first.model.coefficients, [0])
    assert first.model.intercept == 1 / 8
    assert first.model.fit_prefix_marginal_variance == 19 / 64
    assert second.training.n_samples == 12
    assert second.model.intercept == pytest.approx(1 / 24)
    assert second.model.fit_prefix_marginal_variance == pytest.approx(59 / 576)
    for symbol_index, weight, quantity, fee, cash, equity in (
        (0, 16 / 95, 32 / 19, 32 / 95, 78968 / 95, 94968 / 95),
        (1, 4 / 19, 80 / 19, 8 / 19, 14992 / 19, 18992 / 19),
    ):
        executor = MarketExecutor(
            prepared.dataset,
            replace(ExecutionCostConfig.zero(), fee_rate=0.002),
        )
        book = BookState(
            quantities=np.zeros(2),
            cash=1000,
            mark_prices=prepared.dataset.close[6],
            peak_value=1000,
            as_of_index=6,
            as_of_dataset_id=prepared.dataset.dataset_id,
        )
        orders = OrderBookState.empty()
        kwargs = dict(
            account_id=f"independent-{prepared.dataset.symbols[symbol_index]}",
            stream=prepared.stream,
            estimates=prepared.costs_by_symbol[symbol_index][0],
            allocator=AfterCostTargetAllocator(
                lower_weight=0, upper_weight=1, risk_aversion=1
            ),
            pretrade_risk=PreTradeRisk(
                PreTradeRiskConfig(max_abs_weight=1, max_turnover=None)
            ),
            symbol_index=symbol_index,
            start_index=6,
            expected_horizon_seconds=3600,
        )
        proposal = propose_forecast_target(executor, book, orders, **kwargs)
        assert proposal.inputs.expected_simple_return == 1 / 8
        assert proposal.inputs.return_variance == 19 / 64
        assert proposal.target_weight == pytest.approx(weight)
        result = execute_forecast_proposal(executor, book, orders, proposal, **kwargs)
        realized = result.execution.book
        assert realized.quantities[symbol_index] == pytest.approx(quantity)
        assert realized.cash == pytest.approx(cash)
        assert realized.total_cost == pytest.approx(fee)
        assert realized.portfolio_value == pytest.approx(equity)
        assert realized.fill_count == 1
        assert book.cash == 1000 and not np.any(book.quantities)
        # At zero target, canonical cash has no fill, no fee and no lost equity.
        kwargs["allocator"] = AfterCostTargetAllocator(lower_weight=0, upper_weight=0)
        cash_proposal = propose_forecast_target(executor, book, orders, **kwargs)
        cash_result = execute_forecast_proposal(
            executor, book, orders, cash_proposal, **kwargs
        )
        assert cash_proposal.is_hold and cash_result.execution.book.cash == 1000
        assert cash_result.execution.book.total_cost == 0
        assert cash_result.execution.book.fill_count == 0


@pytest.mark.parametrize(
    "failure",
    (
        "default",
        "parent_pin",
        "parent_ref",
        "appended_value",
        "prefix_value",
        "fee",
        "noncanonical_parent",
        "bad_window",
        "bad_channels",
        "mt",
        "subset",
        "reversed_symbols",
        "no_signature",
        "late_prediction",
    ),
)
def test_signature_refusals_precede_fit_and_publication(tmp_path, monkeypatch, failure):
    module = api()
    parent = market()
    if failure == "late_prediction":
        arrivals = parent.available_at.copy()
        arrivals[6, 1] = time(7)
        parent = replace(
            parent,
            available_at=arrivals,
            information_available=None,
            identity_payload_json=None,
        ).with_content_identity(parse_identity_json(parent.identity_payload_json))
    supplied = signature_arguments(tmp_path, parent)
    augmented = module.load_market_dataset_artifact(supplied["dataset_root"])
    identity = parse_identity_json(augmented.identity_payload_json)
    changed = None
    if failure == "default":
        supplied.pop("signature_parent_root")
        supplied.pop("expected_signature_parent_dataset_id")
    elif failure == "parent_pin":
        supplied["expected_signature_parent_dataset_id"] = "0" * 64
    elif failure == "parent_ref":
        identity["source_dataset_id"] = "0" * 64
        changed = augmented.with_content_identity(identity)
    elif failure in ("appended_value", "prefix_value"):
        features = augmented.features.copy()
        features[6, 0, parent.n_features if failure == "appended_value" else 0] += 1
        changed = replace(
            augmented, features=features, identity_payload_json=None
        ).with_content_identity(identity)
    elif failure == "fee":
        fees = augmented.fee_rate.copy()
        fees[6, 0] += 0.001
        changed = replace(
            augmented, fee_rate=fees, identity_payload_json=None
        ).with_content_identity(identity)
    elif failure == "noncanonical_parent":
        parent_identity = parse_identity_json(parent.identity_payload_json)
        parent_identity["config"]["schema_version"] = "market_build_v2"
        parent = parent.with_content_identity(parent_identity)
        root = tmp_path / "invalid-parent"
        write_market_dataset_files(root, parent)
        supplied["signature_parent_root"] = root
        supplied["expected_signature_parent_dataset_id"] = parent.dataset_id
    elif failure in ("bad_window", "bad_channels"):
        identity["window_bars" if failure == "bad_window" else "channels"] = (
            True if failure == "bad_window" else ["t", "v", "p"]
        )
        changed = augmented.with_content_identity(identity)
    elif failure == "mt":
        from trade_rl.data.features.signature_multitimeframe import (
            with_multitimeframe_path_signatures,
        )

        changed = with_multitimeframe_path_signatures(
            parent, base_timeframe="1h", windows_by_timeframe={"1h": 3}, depth=2
        )
        supplied["feature_names"] = (changed.feature_names[parent.n_features],)
    elif failure == "subset":
        supplied["fit_symbols"] = ("ALPHA",)
    elif failure == "reversed_symbols":
        supplied["fit_symbols"] = tuple(reversed(parent.symbols))
    elif failure == "no_signature":
        supplied["feature_names"] = parent.feature_names
    if changed is not None:
        root = tmp_path / "invalid-augmentation"
        write_market_dataset_files(root, changed)
        assert changed.identity_verified
        supplied["dataset_root"] = root
        supplied["expected_dataset_id"] = changed.dataset_id
    forbid_fit_and_publication(module, monkeypatch)
    with pytest.raises(ValueError):
        module.prepare_allocation_market_inputs(**supplied)
    assert not supplied["stream_path"].exists()


@pytest.mark.parametrize(
    "missing", ("signature_parent_root", "expected_signature_parent_dataset_id")
)
def test_incomplete_signature_parent_is_rejected_before_any_load(
    tmp_path, monkeypatch, missing
):
    module = api()
    supplied = signature_arguments(tmp_path)
    supplied.pop(missing)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Incomplete parent declaration reached Dataset loading")

    monkeypatch.setattr(module, "load_market_dataset_artifact", forbidden)
    forbid_fit_and_publication(module, monkeypatch)
    with pytest.raises(ValueError, match="both Signature"):
        module.prepare_allocation_market_inputs(**supplied)


def generated_signature_arguments(tmp_path, *, future=False):
    from tests.evaluation.test_allocation_costs import market as cost_market
    from trade_rl.data.features.signature import with_path_signatures

    parent = cost_market(future=future)
    supplied = generated_arguments(tmp_path, parent)
    augmented = with_path_signatures(parent, window_bars=3, depth=2)
    root = tmp_path / "augmentation"
    write_market_dataset_files(root, augmented)
    supplied.update(
        signature_parent_root=supplied["dataset_root"],
        expected_signature_parent_dataset_id=parent.dataset_id,
        dataset_root=root,
        expected_dataset_id=augmented.dataset_id,
        feature_names=augmented.feature_names[parent.n_features :],
        fit_symbols=parent.symbols,
    )
    return supplied


def test_signature_cost_recipe_uses_verified_parent_without_changing_stream(
    tmp_path, monkeypatch
):
    module = api()
    supplied = generated_signature_arguments(tmp_path)
    owner = module.estimate_declared_horizon_costs
    seen = []

    def tracked(dataset, **kwargs):
        seen.append((dataset.dataset_id, kwargs["recipe"]))
        return owner(dataset, **kwargs)

    monkeypatch.setattr(module, "estimate_declared_horizon_costs", tracked)
    prepared = module.prepare_allocation_market_inputs(**supplied)
    assert seen == [
        (supplied["expected_signature_parent_dataset_id"], supplied["cost_recipe"])
    ]
    assert prepared.cost_recipe is supplied["cost_recipe"]
    assert prepared.stream.dataset_id == supplied["expected_dataset_id"]
    for group in prepared.costs_by_symbol:
        cost = group[0]
        # Existing declared-notional proxy: fee+spread+impact+absolute slippage.
        assert (cost.buy_cost, cost.sell_cost, cost.exit_cost) == pytest.approx(
            (0.0434, 0.0434, 0.0434)
        )
        assert cost.funding_return == pytest.approx(0.000025)
        assert cost.borrow_return == pytest.approx(0.00001875)
        assert cost.cash_return == 0


def test_signature_future_changes_keep_economics_and_update_provenance(tmp_path):
    module = api()
    before, changed = (
        module.prepare_allocation_market_inputs(
            **generated_signature_arguments(tmp_path / name, future=future)
        )
        for name, future in (("before", False), ("changed", True))
    )
    assert before.dataset.dataset_id != changed.dataset.dataset_id
    assert before.stream.digest != changed.stream.digest
    left, right = before.stream.packets[0], changed.stream.packets[0]
    assert left.as_of == right.as_of == time(24)
    assert left.feature_values == right.feature_values == (1, 0, 0.5, 0, 0, 0)
    assert left.expected_simple_return == right.expected_simple_return == 0
    assert left.fit_prefix_marginal_variance == right.fit_prefix_marginal_variance == 0
    for a, b in zip(before.stream.vintages, changed.stream.vintages, strict=True):
        np.testing.assert_array_equal(a.model.coefficients, b.model.coefficients)
        assert a.model.intercept == b.model.intercept
        np.testing.assert_array_equal(
            a.training.trace.start_times, b.training.trace.start_times
        )
    costs = before.costs_by_symbol[0][0], changed.costs_by_symbol[0][0]
    assert costs[0].source_identity != costs[1].source_identity
    assert costs[0].buy_cost == costs[1].buy_cost
    assert costs[0].funding_return == costs[1].funding_return
    decisions = []
    for inputs, cost in zip((before, changed), costs, strict=True):
        executor = MarketExecutor(inputs.dataset, inputs.cost_recipe.execution_cost)
        book = BookState(
            quantities=np.zeros(2),
            cash=1000,
            mark_prices=inputs.dataset.close[24],
            peak_value=1000,
            as_of_index=24,
            as_of_dataset_id=inputs.dataset.dataset_id,
        )
        from trade_rl.evaluation.forecast_allocation import execute_forecast_proposal

        kwargs = dict(
            account_id="same-account",
            stream=inputs.stream,
            estimates=cost,
            allocator=AfterCostTargetAllocator(lower_weight=0, upper_weight=1),
            pretrade_risk=PreTradeRisk(PreTradeRiskConfig(max_abs_weight=1)),
            symbol_index=0,
            start_index=24,
            expected_horizon_seconds=3600,
        )
        orders = OrderBookState.empty()
        proposal = propose_forecast_target(executor, book, orders, **kwargs)
        decisions.append(proposal)
        result = execute_forecast_proposal(executor, book, orders, proposal, **kwargs)
        assert result.execution.book.cash == 1000
        assert result.execution.book.total_cost == 0
        assert not np.any(result.execution.book.quantities)
    assert decisions[0].target_weight == decisions[1].target_weight == 0
    assert decisions[0].decision_digest != decisions[1].decision_digest


def test_finer_native_markers_survive_direct_allocation_preparation(tmp_path):
    from tests.data.test_native_cross_asset_alignment import _dataset, _Source, _source

    raw_source = _source()
    shift = time(0) - np.datetime64("2026-01-01T00:00", "ns")
    source = _Source(
        {
            key: replace(
                raw,
                timestamps=raw.timestamps + shift,
                available_at=raw.available_at + shift,
            )
            for key, raw in raw_source.values.items()
        }
    )
    parent = _dataset(source)
    supplied = arguments(tmp_path, parent)
    supplied.update(
        blocks=(blocks()[0],),
        fit_symbols=parent.symbols,
        feature_names=("15m__rolling_beta_to_btc_4bar",),
    )
    prepared = api().prepare_allocation_market_inputs(**supplied)
    config = parse_identity_json(prepared.dataset.identity_payload_json)["config"]
    assert config["native_cross_asset_alignment"] == "native_before_base_sync_v1"
    assert config["native_cross_asset_history"] == "last_n_eligible_pair_events_v1"
    assert prepared.dataset.dataset_id == parent.dataset_id
    assert prepared.stream.packets[0].as_of == time(6)
