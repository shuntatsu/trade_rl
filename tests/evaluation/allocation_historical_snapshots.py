"""Opt-in pre-settlement-product snapshots of unbound synthetic fixtures only."""

import numpy as np

from trade_rl.data.market import MarketDataset

# Exact identity-array roster from immutable parent 1ae11644, not a generic
# legacy-schema adapter. Ordinary tests continue to use every current array.
_PARENT_ARRAYS = frozenset(
    "timestamps available_at information_available features feature_available "
    "feature_staleness feature_staleness_hours feature_missing_reason "
    "global_features global_feature_available global_feature_staleness_hours "
    "global_feature_missing_reason open high low close volume funding_rate "
    "funding_event_count tradable symbol_active fee_rate maker_fee_rate "
    "taker_fee_rate spread_rate max_participation_rate minimum_notional "
    "lot_size tick_size borrow_available borrow_rate funding_due buy_allowed "
    "sell_allowed mark_price index_price dividend split_factor "
    "delisting_recovery cash_rate contract_multipliers".split()
)


def use_historical_allocation_array_roster(monkeypatch):
    """Keep old literal goldens without changing the current production roster."""
    current_arrays = MarketDataset.identity_arrays

    def historical_arrays(dataset):
        if dataset.dataset_id != "d" * 64 or dataset.identity_payload_json is not None:
            raise ValueError("historical snapshots require an unbound d*64 fixture")
        arrays = current_arrays(dataset)
        if arrays.keys() != _PARENT_ARRAYS | {"funding_price_rate"}:
            raise ValueError("historical snapshot array roster changed")
        if not np.array_equal(
            arrays["funding_price_rate"],
            arrays["funding_rate"] * arrays["mark_price"],
        ):
            raise ValueError("historical snapshots require fallback funding products")
        return {
            name: values for name, values in arrays.items() if name in _PARENT_ARRAYS
        }

    monkeypatch.setattr(MarketDataset, "identity_arrays", historical_arrays)
