from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected snippet not found in {path}: {old[:120]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "trade_rl/data/builder.py",
    "from trade_rl.data.market import MarketDataset\n"
    "from trade_rl.data.source import MarketDataSource, RawMarketSeries\n",
    "from trade_rl.data.market import MarketDataset\n"
    "from trade_rl.data.multitimeframe import align_native_feature\n"
    "from trade_rl.data.source import (\n"
    "    MarketDataSource,\n"
    "    MultiTimeframeMarketDataSource,\n"
    "    RawMarketSeries,\n"
    ")\n",
)

replace_once(
    "trade_rl/data/builder.py",
    "        for symbol_index in range(n_symbols):\n"
    "            for feature_index, spec in enumerate(self.config.features):\n"
    "                event_values, event_valid = _feature_events(\n"
    "                    spec,\n"
    "                    close=close[:, symbol_index],\n"
    "                    volume=volume[:, symbol_index],\n"
    "                    funding_rate=funding_rate[:, symbol_index],\n"
    "                    funding_available=funding_available[:, symbol_index],\n"
    "                    row_present=causal_row_present[:, symbol_index],\n"
    "                    active=symbol_active[:, symbol_index],\n"
    "                )\n"
    "                values, available, age_hours, staleness = _carry_feature(\n"
    "                    event_values,\n"
    "                    event_valid,\n"
    "                    symbol_active[:, symbol_index],\n"
    "                    timestamps,\n"
    "                    max_staleness_hours=spec.max_staleness_hours,\n"
    "                )\n"
    "                features[:, symbol_index, feature_index] = values\n"
    "                feature_available[:, symbol_index, feature_index] = available\n"
    "                feature_age_hours[:, symbol_index, feature_index] = age_hours\n"
    "                feature_staleness[:, symbol_index, feature_index] = staleness\n",
    "        native_cache: dict[tuple[str, str], RawMarketSeries] = {}\n"
    "        for symbol_index, contract in enumerate(instruments):\n"
    "            for feature_index, spec in enumerate(self.config.features):\n"
    "                native_timeframe = spec.resolved_timeframe(\n"
    "                    self.config.base_timeframe\n"
    "                )\n"
    "                if native_timeframe == self.config.base_timeframe:\n"
    "                    event_values, event_valid = _feature_events(\n"
    "                        spec,\n"
    "                        close=close[:, symbol_index],\n"
    "                        volume=volume[:, symbol_index],\n"
    "                        funding_rate=funding_rate[:, symbol_index],\n"
    "                        funding_available=funding_available[:, symbol_index],\n"
    "                        row_present=causal_row_present[:, symbol_index],\n"
    "                        active=symbol_active[:, symbol_index],\n"
    "                    )\n"
    "                    values, available, age_hours, staleness = _carry_feature(\n"
    "                        event_values,\n"
    "                        event_valid,\n"
    "                        symbol_active[:, symbol_index],\n"
    "                        timestamps,\n"
    "                        max_staleness_hours=spec.max_staleness_hours,\n"
    "                    )\n"
    "                else:\n"
    "                    if not isinstance(source, MultiTimeframeMarketDataSource):\n"
    "                        raise ValueError(\n"
    "                            \"native multi-timeframe features require a \"\n"
    "                            \"MultiTimeframeMarketDataSource\"\n"
    "                        )\n"
    "                    key = (contract.symbol, native_timeframe)\n"
    "                    native = native_cache.get(key)\n"
    "                    if native is None:\n"
    "                        native = source.load_timeframe(\n"
    "                            contract.symbol, native_timeframe\n"
    "                        )\n"
    "                        native_cache[key] = native\n"
    "                    values, available, age_hours, staleness = (\n"
    "                        align_native_feature(\n"
    "                            spec,\n"
    "                            native,\n"
    "                            contract,\n"
    "                            timestamps,\n"
    "                            symbol_active[:, symbol_index],\n"
    "                            timeframe=native_timeframe,\n"
    "                        )\n"
    "                    )\n"
    "                features[:, symbol_index, feature_index] = values\n"
    "                feature_available[:, symbol_index, feature_index] = available\n"
    "                feature_age_hours[:, symbol_index, feature_index] = age_hours\n"
    "                feature_staleness[:, symbol_index, feature_index] = staleness\n",
)
