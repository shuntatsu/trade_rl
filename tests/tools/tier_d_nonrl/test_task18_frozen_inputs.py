"""Full-size artificial Task18 freeze/reopen falsifiers; no market artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset

P = build_task18_protocol()
INDEX_PLAN = P.index_plan
E = P.economics
N = P.dataset.row_count
HOUR = 3_600_000_000_000


def _identity(dataset):
    return dataset.with_content_identity({"fixture": "task18-frozen-inputs-v1"})


def _market():
    t = np.arange(N, dtype=float)
    prices = np.column_stack(
        (100 + 4 * np.sin(t / 37) + t / 3000, 200 + 6 * np.cos(t / 53) - t / 6000)
    )
    features = np.stack(
        [
            np.column_stack(
                (
                    np.sin(t / 7 + s),
                    np.cos(t / 11 + s),
                    np.sin(t / 33 + s),
                    np.cos(t / 26 + s),
                    t / N,
                    np.ones(N),
                )
            )
            for s in (0, 1)
        ],
        axis=1,
    ).astype(np.float32)
    counts = np.zeros((N, 2), dtype=np.int32)
    counts[::8, 0] = 1
    counts[3::8, 1] = 1
    funding = np.where(counts > 0, 0.0001 * np.sin(t[:, None] / 120), 0.0)
    funding[::16, 0] = 0.0  # A real counted event can have zero rate.
    volume = np.column_stack((1e7 + 1e6 * np.sin(t / 31), 2e7 + 2e6 * np.cos(t / 41)))
    return _identity(
        MarketDataset(
            dataset_id="0" * 64,
            symbols=P.dataset.symbols,
            timestamps=np.datetime64(P.dataset.development_start, "ns")
            + np.arange(N) * np.timedelta64(1, "h"),
            features=features,
            global_features=np.zeros((N, 1), dtype=np.float32),
            open=prices,
            high=prices,
            low=prices,
            close=prices,
            mark_price=prices,
            volume=volume,
            funding_rate=funding,
            funding_price_rate=funding * prices,
            funding_event_count=counts,
            funding_due=counts > 0,
            tradable=np.ones_like(prices, dtype=bool),
            feature_available=np.ones_like(features, dtype=bool),
            feature_names=P.dataset.feature_names,
            global_feature_names=("artificial",),
            periods_per_year=8760,
            volume_units=(VolumeUnit.QUOTE_NOTIONAL,) * 2,
            fee_rate=np.full_like(prices, E.base_fee_rate),
            maker_fee_rate=np.zeros_like(prices),
            taker_fee_rate=np.zeros_like(prices),
            spread_rate=np.full_like(prices, E.base_spread_rate),
            max_participation_rate=np.full_like(prices, E.maximum_participation_rate),
            borrow_available=np.ones_like(prices, dtype=bool),
            borrow_rate=np.zeros_like(prices),
            cash_rate=np.zeros(N),
        )
    )


def _binding(dataset):
    from tools.tier_d_nonrl.task18_frozen_inputs import FrozenInputBinding

    return FrozenInputBinding(
        scope="SYNTHETIC_ONLY",
        dataset_id=dataset.dataset_id,
        manifest_sha256="a" * 64,
        arrays_sha256="b" * 64,
        source_commit="c" * 40,
        source_manifest_sha256="d" * 64,
        runtime_sha256="e" * 64,
    )


@pytest.fixture(scope="module")
def prepared():
    from tools.tier_d_nonrl.task18_frozen_inputs import prepare_frozen_inputs

    market = _market()
    binding = _binding(market)
    return market, binding, prepare_frozen_inputs(market, binding=binding)


def test_full_frozen_roster_and_independent_cost_oracle(prepared):
    market, binding, frozen = prepared
    assert frozen.binding == binding
    assert not frozen.execution_authorized
    assert len(frozen.models) == 12
    assert frozen.forecasts.shape == (4, 2, 6527)
    assert frozen.cost_rates.shape == (2, 2, 2, 6527, 2)
    for arm in P.arms:
        if arm.role != "NONRL":
            continue
        config = next(c for c in P.configurations if c.name == arm.configuration)
        symbol = P.dataset.symbols.index(arm.symbol)
        for index in (2903, 5086, 5087, 7294, 7295, 9429):
            clock = int(market.timestamps[index].astype("int64"))
            row = frozen.lookup_nonrl(arm.arm_id, index, decision_time_ns=clock)
            assert row.horizon_end_ns == clock + config.horizon_hours * HOUR
            expected_fee = (
                E.base_fee_rate
                + E.base_spread_rate
                + E.impact_coefficient
                * (E.impact_reference_notional_usdt / market.volume[index, symbol])
                ** 0.5
            )
            expected_fee *= 2 if arm.scenario == "COST2L0" else 1
            assert (
                row.buy_cost
                == row.sell_cost
                == row.exit_cost
                == pytest.approx(expected_fee, rel=1e-14)
            )
            expected_funding = (
                sum(
                    float(x)
                    for x in market.funding_rate[index - 23 : index + 1, symbol]
                )
                * config.horizon_hours
                / 24
            )
            assert row.funding_return == pytest.approx(expected_funding, abs=1e-16)
            assert row.cash_return == row.borrow_return == 0.0
            assert row.vintage_start_index == max(
                v for v in INDEX_PLAN.vintage_start_indices if v <= index
            )
            assert row.feature_values == tuple(
                float(x) for x in market.features[index, symbol]
            )
            assert row.decision_close == market.close[index, symbol]
            model = frozen.models[
                P.configurations.index(config) * 3
                + INDEX_PLAN.vintage_start_indices.index(row.vintage_start_index)
            ]
            expected = model.predict_simple_return(
                market.features[index, symbol],
                decision_time=market.timestamps[index],
                feature_available=market.feature_available[index, symbol],
                feature_names=market.feature_names,
            )
            assert row.expected_simple_return == expected
            assert row.training_scope_sha256 == model.training_scope_sha256
    assert np.array_equal(
        frozen.cost_rates[:, 1, ..., 0], 2 * frozen.cost_rates[:, 0, ..., 0]
    )
    assert np.array_equal(
        frozen.cost_rates[:, 1, ..., 1], frozen.cost_rates[:, 0, ..., 1]
    )
    for value in (
        frozen.forecasts,
        frozen.cost_rates,
        frozen.snapshots,
        frozen.source_available_at_ns,
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_publish_reopen_no_refit_no_recost_no_repredict(prepared, tmp_path):
    from tools.tier_d_nonrl import task18_frozen_inputs as freeze

    market, binding, frozen = prepared
    dest = tmp_path.resolve() / "plan"
    pin = freeze.publish_frozen_inputs(dest, frozen)
    raw = (dest / "frozen-plan.bin").read_bytes()
    assert pin.sha256 == hashlib.sha256(raw).hexdigest()
    assert pin.size_bytes == len(raw)
    with (
        patch.object(
            freeze,
            "fit_simple_return_ridge",
            side_effect=AssertionError("must not fit"),
        ),
        patch.object(
            freeze, "build_cost_rates", side_effect=AssertionError("must not recost")
        ),
        patch(
            "trade_rl.strategies.forecasts.simple_return.SimpleReturnRidgeModel.predict_simple_return",
            side_effect=AssertionError("must not predict"),
        ),
    ):
        restored = freeze.load_frozen_inputs(
            dest / "frozen-plan.bin", expected_binding=binding, expected_pin=pin
        )
        for arm in P.arms[::2]:
            row = restored.lookup_nonrl(
                arm.arm_id,
                7295,
                decision_time_ns=int(market.timestamps[7295].astype("int64")),
            )
            assert row == frozen.lookup_nonrl(
                arm.arm_id,
                7295,
                decision_time_ns=int(market.timestamps[7295].astype("int64")),
            )
    assert freeze.encode_frozen_inputs(restored) == raw
    for left, right in zip(frozen.models, restored.models, strict=True):
        for name in ("feature_mean", "feature_scale", "coefficients"):
            assert getattr(left, name).tobytes() == getattr(right, name).tobytes()
    with pytest.raises(FileExistsError):
        freeze.publish_frozen_inputs(dest, frozen)
    assert (dest / "frozen-plan.bin").read_bytes() == raw


@pytest.mark.parametrize(
    "change",
    (
        "digest",
        "size",
        "binding",
        "truncated",
        "trailing",
        "wrong_schema",
        "wrong_unit",
        "swapped_model",
        "duplicate_json",
        "bad_array",
        "early_source_clock",
        "future_source_clock",
        "excess_cost",
        "reused_scope",
        "execution_authority",
        "oversize_header",
    ),
)
def test_restore_rejects_wrong_pin_and_resealed_structural_corruption(
    prepared, tmp_path, change
):
    from tools.tier_d_nonrl import task18_frozen_inputs as freeze

    _, binding, frozen = prepared
    raw = freeze.encode_frozen_inputs(frozen)
    header_len = int.from_bytes(raw[8:12], "big")
    header = json.loads(raw[12 : 12 + header_len])
    tail = raw[12 + header_len :]
    pin = freeze.FrozenInputsPin(hashlib.sha256(raw).hexdigest(), len(raw))
    if change == "digest":
        pin = replace(pin, sha256="f" * 64)
    elif change == "size":
        pin = replace(pin, size_bytes=pin.size_bytes - 1)
    elif change == "binding":
        binding = replace(binding, runtime_sha256="f" * 64)
    elif change == "truncated":
        raw = raw[:-1]
    elif change == "trailing":
        raw += b"x"
    elif change in (
        "wrong_schema",
        "wrong_unit",
        "swapped_model",
        "duplicate_json",
        "reused_scope",
        "execution_authority",
        "oversize_header",
    ):
        if change == "wrong_schema":
            header["schema"] = "old-log"
        elif change == "wrong_unit":
            header["return_unit"] = "log"
        elif change == "swapped_model":
            header["models"][0], header["models"][1] = (
                header["models"][1],
                header["models"][0],
            )
        elif change == "reused_scope":
            # Same forged digest for two distinct vintage scopes, including
            # both alpha variants, preserving the intra-vintage invariant.
            for model_index in (1, 4):
                header["models"][model_index]["training_scope_sha256"] = header[
                    "models"
                ][0]["training_scope_sha256"]
        elif change == "execution_authority":
            header["execution_authorized"] = True
        header_raw = canonical_json_bytes(header)
        if change == "duplicate_json":
            header_raw = header_raw.replace(
                b'"execution_authorized":false',
                b'"execution_authorized":true,"execution_authorized":false',
            )
        if change == "oversize_header":
            raw = raw[:8] + (2**31).to_bytes(4, "big") + raw[12:]
        else:
            raw = raw[:8] + len(header_raw).to_bytes(4, "big") + header_raw + tail
    elif change == "bad_array":
        # All dense arrays must be finite, even if a caller re-pins the bytes.
        raw = (
            raw[: 12 + header_len]
            + np.asarray([float("nan")], dtype="<f8").tobytes()
            + tail[8:]
        )
    elif change in ("early_source_clock", "future_source_clock"):
        last = int.from_bytes(raw[-8:], "little", signed=True)
        adjusted = last - 1 if change == "early_source_clock" else last + 1
        raw = raw[:-8] + adjusted.to_bytes(8, "little", signed=True)
    elif change == "excess_cost":
        mutable = bytearray(raw)
        table = np.frombuffer(
            mutable,
            dtype="<f8",
            count=frozen.cost_rates.size,
            offset=12 + header_len + frozen.forecasts.nbytes,
        ).reshape(frozen.cost_rates.shape)
        # Preserve cross-horizon, stress and carry relationships while
        # violating the maximum possible reference-participation cost.
        table[:, 0, ..., 0] = 0.01
        table[:, 1, ..., 0] = 0.02
        raw = bytes(mutable)
    if change not in ("digest", "size", "binding"):
        pin = freeze.FrozenInputsPin(hashlib.sha256(raw).hexdigest(), len(raw))
    path = tmp_path / "frozen-plan.bin"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        freeze.load_frozen_inputs(path, expected_binding=binding, expected_pin=pin)
    assert path.read_bytes() == raw


def test_lookup_refuses_cash_stale_or_reserved_decisions(prepared):
    market, _, frozen = prepared
    clock = int(market.timestamps[2903].astype("int64"))
    for arm, index, time in (
        (P.arms[1].arm_id, 2903, clock),
        ("foreign", 2903, clock),
        (P.arms[0].arm_id, 2903, clock - 1),
        (P.arms[0].arm_id, 9430, clock),
        (P.arms[0].arm_id, 2902, clock),
        (P.arms[0].arm_id, True, clock),
    ):
        with pytest.raises(ValueError):
            frozen.lookup_nonrl(arm, index, decision_time_ns=time)


@pytest.mark.parametrize(
    "change",
    (
        "fee",
        "capacity",
        "funding_gap",
        "funding_count",
        "unavailable",
        "mark",
        "cash",
        "volume_unit",
        "binding",
    ),
)
def test_preparation_refuses_unbound_or_inadmissible_cost_inputs_before_any_fit(
    prepared, change
):
    from tools.tier_d_nonrl import task18_frozen_inputs as freeze

    market, binding, _ = prepared
    kwargs = {}
    row = INDEX_PLAN.actor_start_inclusive
    if change == "fee":
        a = market.fee_rate.copy()
        a[row, 0] = 0.0
        kwargs["fee_rate"] = a
    elif change == "capacity":
        a = market.volume.copy()
        a[row, 0] = 100.0
        kwargs["volume"] = a
    elif change == "funding_gap":
        counts = market.funding_event_count.copy()
        counts[row - 23 : row + 1, 0] = 0
        funding = market.funding_rate.copy()
        funding[row - 23 : row + 1, 0] = 0.0
        kwargs.update(
            funding_event_count=counts, funding_due=counts > 0, funding_rate=funding
        )
    elif change == "funding_count":
        counts = market.funding_event_count.copy()
        counts[row, 0] = 10
        kwargs.update(funding_event_count=counts, funding_due=counts > 0)
    elif change == "unavailable":
        a = market.information_available.copy()
        a[row - 1, 0] = False
        kwargs["information_available"] = a
    elif change == "mark":
        a = market.mark_price.copy()
        a[row, 0] += 1.0
        kwargs["mark_price"] = a
    elif change == "cash":
        a = market.cash_rate.copy()
        a[row] = 0.001
        kwargs["cash_rate"] = a
    elif change == "volume_unit":
        kwargs["volume_units"] = (VolumeUnit.BASE_ASSET,) * 2
    if kwargs:
        if "funding_rate" in kwargs or "mark_price" in kwargs:
            kwargs["funding_price_rate"] = kwargs.get(
                "funding_rate", market.funding_rate
            ) * kwargs.get("mark_price", market.mark_price)
        market = _identity(
            replace(market, dataset_id="0" * 64, identity_payload_json=None, **kwargs)
        )
        binding = _binding(market)
    else:
        binding = replace(binding, dataset_id="f" * 64)
    with patch.object(
        freeze,
        "fit_simple_return_ridge",
        side_effect=AssertionError("fit before admission"),
    ):
        with pytest.raises(ValueError):
            freeze.prepare_frozen_inputs(market, binding=binding)


@pytest.mark.parametrize("change", ("missing", "different"))
def test_cost_builder_rejects_absent_or_different_v7_funding_witness(change):
    from tools.tier_d_nonrl.task18_costs import build_cost_rates

    market = _market()
    funding_price_rate = None
    if change == "different":
        funding_price_rate = market.funding_price_rate.copy()
        funding_price_rate[INDEX_PLAN.actor_start_inclusive, 0] += 1.0
    altered = _identity(
        replace(
            market,
            dataset_id="0" * 64,
            identity_payload_json=None,
            funding_price_rate=funding_price_rate,
        )
    )
    with pytest.raises(ValueError, match="exact v7 bar-close"):
        build_cost_rates(altered)


def test_every_frozen_cost_row_matches_independent_vectorized_formula(prepared):
    market, _, frozen = prepared
    first, stop = INDEX_PLAN.actor_start_inclusive, INDEX_PLAN.actor_stop_exclusive
    # Independent full-row calculation, not the producer's per-row loop.
    fee = (
        E.base_fee_rate
        + E.base_spread_rate
        + E.impact_coefficient
        * np.sqrt(E.impact_reference_notional_usdt / market.volume[first:stop].T)
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        market.funding_rate.T, window_shape=24, axis=1
    )
    sums = np.sum(windows[:, first - 23 : stop - 23], axis=-1)
    assert sums.shape == fee.shape == (2, 6527)
    for h, horizon in enumerate((24, 72)):
        for scenario, multiplier in enumerate((1.0, 2.0)):
            np.testing.assert_allclose(
                frozen.cost_rates[h, scenario, ..., 0],
                multiplier * fee,
                rtol=1e-14,
                atol=0.0,
            )
            np.testing.assert_allclose(
                frozen.cost_rates[h, scenario, ..., 1],
                sums * horizon / 24.0,
                rtol=1e-14,
                atol=1e-18,
            )


def test_external_digest_is_checked_before_any_json_decode(prepared, tmp_path):
    from tools.tier_d_nonrl import task18_frozen_inputs as freeze

    _, binding, frozen = prepared
    dest = tmp_path.resolve() / "plan"
    pin = freeze.publish_frozen_inputs(dest, frozen)
    with patch.object(
        freeze.json, "loads", side_effect=AssertionError("unverified decode")
    ):
        with pytest.raises(ValueError, match="digest mismatch"):
            freeze.load_frozen_inputs(
                dest / "frozen-plan.bin",
                expected_binding=binding,
                expected_pin=replace(pin, sha256="f" * 64),
            )


def test_scope_and_size_require_exact_external_binding(prepared):
    from tools.tier_d_nonrl.task18_frozen_inputs import FrozenInputsPin

    _, binding, _ = prepared
    historical = P.historical_reference
    with pytest.raises(ValueError, match="synthetic plan"):
        replace(binding, dataset_id=historical.dataset_id)
    with pytest.raises(ValueError, match="exact original"):
        replace(binding, scope="CONSUMED_DEVELOPMENT_ADAPTED")
    for size in (True, 0, -1, 2**31):
        with pytest.raises(ValueError, match="size"):
            FrozenInputsPin("a" * 64, size)
