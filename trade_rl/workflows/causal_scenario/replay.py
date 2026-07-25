"""Materialize one selected relative scenario as a query-anchored dataset."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.causal_scenario.library import (
    CausalScenarioSelection,
    FrozenCausalScenarioLibrary,
    RelativeScenarioBlock,
    _extract_block,
)

_REPLAY_IDENTITY_SCHEMA: Final = "causal_scenario_replay_identity_v1"


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class CausalScenarioReplayIdentity:
    """Digest-bound provenance for one query/block replay dataset."""

    query_dataset_id: str
    library_digest: str
    selection_digest: str
    block_digest: str
    scenario_id: str
    query_index: int
    selected_rank: int
    schema_version: str = _REPLAY_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "query_dataset_id",
            "library_digest",
            "selection_digest",
            "block_digest",
            "scenario_id",
        ):
            object.__setattr__(
                self, name, require_sha256(str(getattr(self, name)), field=name)
            )
        object.__setattr__(
            self, "query_index", _non_negative_int("query_index", self.query_index)
        )
        object.__setattr__(
            self,
            "selected_rank",
            _non_negative_int("selected_rank", self.selected_rank),
        )
        if self.schema_version != _REPLAY_IDENTITY_SCHEMA:
            raise ValueError("unsupported causal scenario replay identity schema")

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "block_digest": self.block_digest,
            "library_digest": self.library_digest,
            "query_dataset_id": self.query_dataset_id,
            "query_index": self.query_index,
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "selected_rank": self.selected_rank,
            "selection_digest": self.selection_digest,
        }


def _query_row(dataset: MarketDataset, field: str, index: int) -> np.ndarray:
    if field == "features":
        return dataset.features[index : index + 1]
    if field == "global_features":
        return dataset.global_features[index : index + 1]
    if field == "funding_rate":
        return dataset.funding_rate[index : index + 1]
    if field == "tradable":
        return dataset.tradable[index : index + 1]
    if field == "feature_available":
        return dataset.feature_available[index : index + 1]
    return dataset.resolved_array(field)[index : index + 1]


def _concat_future(
    dataset: MarketDataset,
    block: RelativeScenarioBlock,
    field: str,
    query_index: int,
) -> np.ndarray:
    return np.concatenate(
        (_query_row(dataset, field, query_index), block.future_arrays[field]), axis=0
    )


def _replay_prices(
    dataset: MarketDataset,
    block: RelativeScenarioBlock,
    query_index: int,
) -> dict[str, np.ndarray]:
    close_anchor = dataset.close[query_index]
    mark_anchor = dataset.resolved_array("mark_price")[query_index]
    index_anchor = dataset.resolved_array("index_price")[query_index]
    result: dict[str, np.ndarray] = {}
    for field in ("open", "high", "low", "close"):
        query = getattr(dataset, field)[query_index : query_index + 1]
        future = close_anchor * block.price_relatives[field]
        result[field] = np.concatenate((query, future), axis=0)
    result["mark_price"] = np.concatenate(
        (
            dataset.resolved_array("mark_price")[query_index : query_index + 1],
            mark_anchor * block.price_relatives["mark_price"],
        ),
        axis=0,
    )
    result["index_price"] = np.concatenate(
        (
            dataset.resolved_array("index_price")[query_index : query_index + 1],
            index_anchor * block.price_relatives["index_price"],
        ),
        axis=0,
    )
    result["dividend"] = np.concatenate(
        (
            dataset.resolved_array("dividend")[query_index : query_index + 1],
            close_anchor * block.price_relatives["dividend"],
        ),
        axis=0,
    )
    result["volume"] = np.concatenate(
        (
            dataset.volume[query_index : query_index + 1],
            dataset.volume[query_index] * block.volume_relative,
        ),
        axis=0,
    )
    return result


def _revalidate_selected_block(block: RelativeScenarioBlock) -> RelativeScenarioBlock:
    try:
        return RelativeScenarioBlock(
            anchor_index=block.anchor_index,
            source_start=block.source_start,
            source_stop=block.source_stop,
            elapsed_ns=block.elapsed_ns,
            raw_condition=block.raw_condition,
            normalized_condition=block.normalized_condition,
            price_relatives=dict(block.price_relatives),
            volume_relative=block.volume_relative,
            market_notional_relative=block.market_notional_relative,
            future_arrays=dict(block.future_arrays),
            block_digest=block.block_digest,
            schema_version=block.schema_version,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "selected block is not contained in scenario library"
        ) from error


def materialize_causal_scenario_dataset(
    library: FrozenCausalScenarioLibrary,
    selection: CausalScenarioSelection,
    query_view: MarketDatasetView,
    *,
    selected_rank: int,
) -> MarketDataset:
    """Create a deterministic replay without reading realized query-future rows."""

    rank = _non_negative_int("selected_rank", selected_rank)
    if rank >= len(selection.blocks):
        raise ValueError("selected scenario rank is outside the selection")
    if selection.library_digest != library.library_digest:
        raise ValueError("selection library identity does not match library")
    dataset = query_view.dataset
    query_index = selection.query_index
    if (
        query_view.stop != query_index + 1
        or not query_view.start <= query_index < query_view.stop
    ):
        raise ValueError("query view must be the causal prefix used by selection")
    if dataset.dataset_id != library.dataset_id or dataset.symbols != library.symbols:
        raise ValueError("query dataset does not match scenario library")
    block = _revalidate_selected_block(selection.blocks[rank])
    matching = np.flatnonzero(library.anchor_indices == block.anchor_index)
    if matching.size != 1:
        raise ValueError("selected block is not contained in scenario library")
    position = int(matching[0])
    expected_block = _extract_block(
        dataset,
        anchor=block.anchor_index,
        horizon=library.config.horizon_decisions,
        raw_condition=library.raw_conditions[position],
        normalized_condition=library.normalized_conditions[position],
        relative_floor=library.config.relative_floor,
    )
    if block.block_digest != expected_block.block_digest:
        raise ValueError("selected block is not contained in scenario library")
    identity = CausalScenarioReplayIdentity(
        query_dataset_id=dataset.dataset_id,
        library_digest=library.library_digest,
        selection_digest=selection.selection_digest,
        block_digest=block.block_digest,
        scenario_id=selection.scenario_set.scenario_ids[rank],
        query_index=query_index,
        selected_rank=rank,
    )
    query_timestamp_ns = int(
        dataset.timestamps[query_index].astype("datetime64[ns]").astype(np.int64)
    )
    future_timestamp_ns = query_timestamp_ns + block.elapsed_ns
    timestamps = np.concatenate(
        (
            np.asarray([query_timestamp_ns], dtype=np.int64),
            future_timestamp_ns,
        )
    ).astype("datetime64[ns]")
    prices = _replay_prices(dataset, block, query_index)
    future_delay = block.future_arrays["availability_delay_ns"].astype(np.int64)
    available_at_ns = np.concatenate(
        (
            dataset.resolved_array("available_at")[query_index : query_index + 1]
            .astype("datetime64[ns]")
            .astype(np.int64),
            future_timestamp_ns[:, None] + future_delay,
        ),
        axis=0,
    )
    replay = MarketDataset(
        dataset_id="0" * 64,
        symbols=dataset.symbols,
        timestamps=timestamps,
        features=_concat_future(dataset, block, "features", query_index),
        global_features=_concat_future(dataset, block, "global_features", query_index),
        open=prices["open"],
        high=prices["high"],
        low=prices["low"],
        close=prices["close"],
        volume=prices["volume"],
        funding_rate=_concat_future(dataset, block, "funding_rate", query_index),
        tradable=_concat_future(dataset, block, "tradable", query_index),
        feature_available=_concat_future(
            dataset, block, "feature_available", query_index
        ),
        feature_names=dataset.feature_names,
        global_feature_names=dataset.global_feature_names,
        periods_per_year=dataset.periods_per_year,
        calendar_kind=dataset.calendar_kind,
        nominal_bar_hours=dataset.nominal_bar_hours,
        funding_event_count=_concat_future(
            dataset, block, "funding_event_count", query_index
        ),
        feature_staleness_hours=_concat_future(
            dataset, block, "feature_staleness_hours", query_index
        ),
        feature_missing_reason=_concat_future(
            dataset, block, "feature_missing_reason", query_index
        ),
        global_feature_available=_concat_future(
            dataset, block, "global_feature_available", query_index
        ),
        global_feature_staleness_hours=_concat_future(
            dataset, block, "global_feature_staleness_hours", query_index
        ),
        global_feature_missing_reason=_concat_future(
            dataset, block, "global_feature_missing_reason", query_index
        ),
        fee_rate=_concat_future(dataset, block, "fee_rate", query_index),
        maker_fee_rate=_concat_future(dataset, block, "maker_fee_rate", query_index),
        taker_fee_rate=_concat_future(dataset, block, "taker_fee_rate", query_index),
        spread_rate=_concat_future(dataset, block, "spread_rate", query_index),
        max_participation_rate=_concat_future(
            dataset, block, "max_participation_rate", query_index
        ),
        minimum_notional=_concat_future(
            dataset, block, "minimum_notional", query_index
        ),
        lot_size=_concat_future(dataset, block, "lot_size", query_index),
        tick_size=_concat_future(dataset, block, "tick_size", query_index),
        borrow_available=_concat_future(
            dataset, block, "borrow_available", query_index
        ),
        borrow_rate=_concat_future(dataset, block, "borrow_rate", query_index),
        funding_due=_concat_future(dataset, block, "funding_due", query_index),
        asset_active=_concat_future(dataset, block, "asset_active", query_index),
        buy_allowed=_concat_future(dataset, block, "buy_allowed", query_index),
        sell_allowed=_concat_future(dataset, block, "sell_allowed", query_index),
        mark_price=prices["mark_price"],
        index_price=prices["index_price"],
        dividend=prices["dividend"],
        split_factor=_concat_future(dataset, block, "split_factor", query_index),
        delisting_recovery=_concat_future(
            dataset, block, "delisting_recovery", query_index
        ),
        cash_rate=_concat_future(dataset, block, "cash_rate", query_index),
        information_available=_concat_future(
            dataset, block, "information_available", query_index
        ),
        available_at=available_at_ns.astype("datetime64[ns]"),
        feature_staleness=_concat_future(
            dataset, block, "feature_staleness", query_index
        ),
        volume_units=dataset.volume_units,
        contract_multipliers=dataset.contract_multipliers,
        feature_config_digest=dataset.feature_config_digest,
        normalization_digest=dataset.normalization_digest,
    ).with_content_identity(
        {
            "causal_scenario_replay": identity.payload(),
            "causal_scenario_replay_digest": identity.digest,
        }
    )
    base_notional = replay.market_notional(0, prices=replay.close[0])
    replay_relatives = (
        np.vstack(
            [
                replay.market_notional(index, prices=replay.close[index])
                for index in range(1, replay.n_bars)
            ]
        )
        / base_notional
    )
    if not np.allclose(
        replay_relatives,
        block.market_notional_relative,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("replayed market notional does not match selected block")
    return replay


__all__ = [
    "CausalScenarioReplayIdentity",
    "materialize_causal_scenario_dataset",
]
