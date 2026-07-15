from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected snippet not found in {path}: {old[:160]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "trade_rl/integrations/binance.py",
    "class BinanceDatasetBuildResult:\n"
    "    dataset: MarketDataset\n"
    "    metadata: tuple[BinanceInstrumentMetadata, ...]\n"
    "    sources_used: tuple[str, ...]\n",
    "class BinanceDatasetBuildResult:\n"
    "    dataset: MarketDataset\n"
    "    metadata: tuple[BinanceInstrumentMetadata, ...]\n"
    "    sources_used: tuple[str, ...]\n"
    "    feature_timeframes: tuple[str, ...] = ()\n",
)

replace_once(
    "trade_rl/integrations/binance.py",
    "def vision_funding_url(\n",
    "def vision_monthly_kline_url(\n"
    "    market: BinanceMarket | str,\n"
    "    symbol: str,\n"
    "    interval: str,\n"
    "    month: datetime,\n"
    ") -> str:\n"
    "    resolved = _market(market)\n"
    "    _interval_ms(interval)\n"
    "    period = _aware_utc(month, field=\"month\").strftime(\"%Y-%m\")\n"
    "    if resolved is BinanceMarket.SPOT:\n"
    "        prefix = \"spot/monthly/klines\"\n"
    "    elif resolved is BinanceMarket.USDS_M:\n"
    "        prefix = \"futures/um/monthly/klines\"\n"
    "    else:\n"
    "        prefix = \"futures/cm/monthly/klines\"\n"
    "    return (\n"
    "        f\"{_VISION_ROOT}/{prefix}/{symbol}/{interval}/\"\n"
    "        f\"{symbol}-{interval}-{period}.zip\"\n"
    "    )\n\n\n"
    "def _next_month(value: datetime) -> datetime:\n"
    "    if value.month == 12:\n"
    "        return value.replace(year=value.year + 1, month=1, day=1)\n"
    "    return value.replace(month=value.month + 1, day=1)\n\n\n"
    "def plan_vision_kline_urls(\n"
    "    market: BinanceMarket | str,\n"
    "    symbol: str,\n"
    "    interval: str,\n"
    "    start_time: datetime,\n"
    "    end_time: datetime,\n"
    ") -> tuple[str, ...]:\n"
    "    start = _aware_utc(start_time, field=\"start_time\")\n"
    "    end = _aware_utc(end_time, field=\"end_time\")\n"
    "    if end <= start:\n"
    "        raise ValueError(\"end_time must be later than start_time\")\n"
    "    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)\n"
    "    urls: list[str] = []\n"
    "    while cursor < end:\n"
    "        month_start = cursor.replace(day=1)\n"
    "        next_month = _next_month(month_start)\n"
    "        if cursor == month_start and start <= cursor and next_month <= end:\n"
    "            urls.append(\n"
    "                vision_monthly_kline_url(\n"
    "                    market, symbol, interval, month_start\n"
    "                )\n"
    "            )\n"
    "            cursor = next_month\n"
    "        else:\n"
    "            urls.append(vision_kline_url(market, symbol, interval, cursor))\n"
    "            cursor += timedelta(days=1)\n"
    "    return tuple(urls)\n\n\n"
    "def vision_funding_url(\n",
)

replace_once(
    "trade_rl/integrations/binance.py",
    "    def _load_vision_klines(\n"
    "        self,\n"
    "        *,\n"
    "        market: BinanceMarket,\n"
    "        symbol: str,\n"
    "        interval: str,\n"
    "        start_ms: int,\n"
    "        end_ms: int,\n"
    "    ) -> list[list[object]]:\n"
    "        result: list[list[object]] = []\n"
    "        for day in _iter_days(start_ms, end_ms):\n"
    "            url = vision_kline_url(market, symbol, interval, day)\n"
    "            rows = _csv_rows_from_zip(self._request_bytes(url), source=url)\n"
    "            if rows and _looks_like_header(rows[0]):\n"
    "                rows = rows[1:]\n"
    "            for row in rows:\n"
    "                if len(row) < 8:\n"
    "                    raise BinanceTransportError(\n"
    "                        f\"Binance Vision kline row is short: {url}\"\n"
    "                    )\n"
    "                open_ms = _normalize_epoch_ms(row[0])\n"
    "                if start_ms <= open_ms < end_ms:\n"
    "                    result.append(list(row))\n"
    "        return result\n",
    "    def _load_vision_klines(\n"
    "        self,\n"
    "        *,\n"
    "        market: BinanceMarket,\n"
    "        symbol: str,\n"
    "        interval: str,\n"
    "        start_ms: int,\n"
    "        end_ms: int,\n"
    "    ) -> list[list[object]]:\n"
    "        result: list[list[object]] = []\n"
    "        start_time = datetime.fromtimestamp(start_ms / 1_000, tz=UTC)\n"
    "        end_time = datetime.fromtimestamp(end_ms / 1_000, tz=UTC)\n"
    "        for url in plan_vision_kline_urls(\n"
    "            market, symbol, interval, start_time, end_time\n"
    "        ):\n"
    "            rows = _csv_rows_from_zip(self._request_bytes(url), source=url)\n"
    "            if rows and _looks_like_header(rows[0]):\n"
    "                rows = rows[1:]\n"
    "            for row in rows:\n"
    "                if len(row) < 8:\n"
    "                    raise BinanceTransportError(\n"
    "                        f\"Binance Vision kline row is short: {url}\"\n"
    "                    )\n"
    "                open_ms = _normalize_epoch_ms(row[0])\n"
    "                if start_ms <= open_ms < end_ms:\n"
    "                    result.append(list(row))\n"
    "        return result\n",
)

replace_once(
    "trade_rl/integrations/binance.py",
    "    if len(parsed) < 3:\n"
    "        raise ValueError(\"Binance range must contain at least three closed bars\")\n",
    "    if len(parsed) < 2:\n"
    "        raise ValueError(\"Binance range must contain at least two closed bars\")\n",
)

source_start = "class BinanceMarketDataSource(MarketDataSource):\n"
source_end = "\n\ndef _filter_value(\n"
path = Path("trade_rl/integrations/binance.py")
text = path.read_text(encoding="utf-8")
if "def load_timeframe(self, symbol: str, timeframe: str)" not in text:
    start = text.index(source_start)
    end = text.index(source_end, start)
    replacement = '''class BinanceMarketDataSource(MarketDataSource):
    """Load one fixed Binance range on one or more causal native clocks."""

    def __init__(
        self,
        *,
        market: BinanceMarket | str,
        interval: str,
        start_time: datetime,
        end_time: datetime,
        transport_mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
        transport: Any | None = None,
    ) -> None:
        self.market = _market(market)
        self.interval = interval
        self.interval_ms = _interval_ms(interval)
        self.start_time = _aware_utc(start_time, field="start_time")
        self.end_time = _aware_utc(end_time, field="end_time")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time")
        start_ms = _epoch_ms(self.start_time)
        end_ms = _epoch_ms(self.end_time)
        if start_ms % self.interval_ms != 0 or end_ms % self.interval_ms != 0:
            raise ValueError("Binance range boundaries must align to the interval")
        self.transport_mode = _mode(transport_mode)
        self.transport = transport or BinancePublicTransport()
        self._sources_used: set[str] = set()
        self._series_cache: dict[tuple[str, str], RawMarketSeries] = {}
        self._funding_cache: dict[str, tuple[list[tuple[int, float]], object]] = {}

    @property
    def sources_used(self) -> tuple[str, ...]:
        return tuple(sorted(self._sources_used))

    def _record_source(self, source: object) -> None:
        if isinstance(source, str):
            self._sources_used.add(source)
            return
        if isinstance(source, Sequence):
            self._sources_used.update(str(item) for item in source)
            return
        self._sources_used.add(str(source))

    def _funding_events(self, symbol: str) -> list[tuple[int, float]]:
        cached = self._funding_cache.get(symbol)
        if cached is None:
            events, funding_source = self.transport.load_funding_rates(
                market=self.market,
                symbol=symbol,
                start_ms=_epoch_ms(self.start_time),
                end_ms=_epoch_ms(self.end_time),
                mode=self.transport_mode,
            )
            cached = (list(events), funding_source)
            self._funding_cache[symbol] = cached
            self._record_source(funding_source)
        return cached[0]

    def load(self, symbol: str) -> RawMarketSeries:
        return self.load_timeframe(symbol, self.interval)

    def load_timeframe(self, symbol: str, timeframe: str) -> RawMarketSeries:
        if not symbol:
            raise ValueError("Binance symbol must not be empty")
        interval_ms = _interval_ms(timeframe)
        start_ms = _epoch_ms(self.start_time)
        end_ms = _epoch_ms(self.end_time)
        if start_ms % interval_ms != 0 or end_ms % interval_ms != 0:
            raise ValueError(
                f"Binance range boundaries must align to native timeframe {timeframe}"
            )
        key = (symbol, timeframe)
        cached = self._series_cache.get(key)
        if cached is not None:
            return cached
        rows, kline_source = self.transport.load_klines(
            market=self.market,
            symbol=symbol,
            interval=timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
            mode=self.transport_mode,
        )
        self._record_source(kline_source)
        timestamps, open_price, high, low, close, volume = _parse_kline_rows(
            rows,
            interval_ms=interval_ms,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if timeframe == self.interval:
            funding, funding_available = _align_funding(
                timestamps,
                self._funding_events(symbol),
            )
        else:
            funding = np.zeros(len(timestamps), dtype=np.float64)
            funding_available = np.zeros(len(timestamps), dtype=np.bool_)
        series = RawMarketSeries(
            timestamps=timestamps,
            available_at=timestamps,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            funding_rate=funding,
            funding_available=funding_available,
            tradable=np.ones(len(timestamps), dtype=np.bool_),
        )
        self._series_cache[key] = series
        return series
'''
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")

replace_once(
    "trade_rl/integrations/binance.py",
    "def _default_features(interval: str) -> tuple[FeatureSpec, ...]:\n",
    "def binance_multitimeframe_feature_specs(\n"
    "    *,\n"
    "    base_timeframe: str,\n"
    "    feature_timeframes: Sequence[str],\n"
    ") -> tuple[FeatureSpec, ...]:\n"
    "    _interval_ms(base_timeframe)\n"
    "    resolved = tuple(feature_timeframes)\n"
    "    if len(set(resolved)) != len(resolved):\n"
    "        raise ValueError(\"duplicate Binance feature timeframes are not allowed\")\n"
    "    if base_timeframe in resolved:\n"
    "        raise ValueError(\"base timeframe must not be repeated as a feature timeframe\")\n"
    "    for timeframe in resolved:\n"
    "        _interval_ms(timeframe)\n"
    "    ordered = tuple(\n"
    "        sorted((*resolved, base_timeframe), key=_interval_ms)\n"
    "    )\n"
    "    features: list[FeatureSpec] = []\n"
    "    for timeframe in ordered:\n"
    "        native = None if timeframe == base_timeframe else timeframe\n"
    "        native_hours = _interval_ms(timeframe) / 3_600_000.0\n"
    "        staleness = max(native_hours * 2.0, base_timeframe == timeframe and 1.0 or native_hours)\n"
    "        features.append(\n"
    "            FeatureSpec(\n"
    "                name=f\"{timeframe}__log_return_1bar\",\n"
    "                kind=FeatureKind.LOG_RETURN,\n"
    "                timeframe=native,\n"
    "                lookback=1,\n"
    "                max_staleness_hours=staleness,\n"
    "            )\n"
    "        )\n"
    "        if timeframe == base_timeframe:\n"
    "            one_day = max(1, int(round(24.0 / native_hours)))\n"
    "            features.extend(\n"
    "                (\n"
    "                    FeatureSpec(\n"
    "                        name=f\"{timeframe}__log_return_1d\",\n"
    "                        kind=FeatureKind.LOG_RETURN,\n"
    "                        lookback=one_day,\n"
    "                        max_staleness_hours=staleness,\n"
    "                    ),\n"
    "                    FeatureSpec(\n"
    "                        name=f\"{timeframe}__volume_zscore_1d\",\n"
    "                        kind=FeatureKind.VOLUME_ZSCORE,\n"
    "                        lookback=one_day,\n"
    "                        min_periods=min(one_day, 2),\n"
    "                        max_staleness_hours=staleness,\n"
    "                    ),\n"
    "                    FeatureSpec(\n"
    "                        name=f\"{timeframe}__funding_bps\",\n"
    "                        kind=FeatureKind.FUNDING_BPS,\n"
    "                        max_staleness_hours=8.0,\n"
    "                    ),\n"
    "                )\n"
    "            )\n"
    "        elif timeframe == \"1d\":\n"
    "            features.append(\n"
    "                FeatureSpec(\n"
    "                    name=\"1d__log_return_7bar\",\n"
    "                    kind=FeatureKind.LOG_RETURN,\n"
    "                    timeframe=\"1d\",\n"
    "                    lookback=7,\n"
    "                    max_staleness_hours=48.0,\n"
    "                )\n"
    "            )\n"
    "        else:\n"
    "            one_day = max(2, int(round(24.0 / native_hours)))\n"
    "            features.append(\n"
    "                FeatureSpec(\n"
    "                    name=f\"{timeframe}__realized_volatility_{one_day}bar\",\n"
    "                    kind=FeatureKind.REALIZED_VOLATILITY,\n"
    "                    timeframe=timeframe,\n"
    "                    lookback=one_day,\n"
    "                    max_staleness_hours=staleness,\n"
    "                )\n"
    "            )\n"
    "    return tuple(features)\n\n\n"
    "def _default_features(interval: str) -> tuple[FeatureSpec, ...]:\n",
)

replace_once(
    "trade_rl/integrations/binance.py",
    "    listed_ats: Sequence[datetime] | None = None,\n"
    ") -> BinanceDatasetBuildResult:\n",
    "    listed_ats: Sequence[datetime] | None = None,\n"
    "    feature_timeframes: Sequence[str] | None = None,\n"
    ") -> BinanceDatasetBuildResult:\n",
)

replace_once(
    "trade_rl/integrations/binance.py",
    "    resolved_mode = _mode(transport_mode)\n"
    "    resolved_tick = _optional_values(\n",
    "    resolved_mode = _mode(transport_mode)\n"
    "    requested_feature_timeframes = tuple(feature_timeframes or ())\n"
    "    resolved_features = (\n"
    "        _default_features(interval)\n"
    "        if not requested_feature_timeframes\n"
    "        else binance_multitimeframe_feature_specs(\n"
    "            base_timeframe=interval,\n"
    "            feature_timeframes=requested_feature_timeframes,\n"
    "        )\n"
    "    )\n"
    "    resolved_tick = _optional_values(\n",
)

replace_once(
    "trade_rl/integrations/binance.py",
    "            base_timeframe=interval,\n"
    "            features=_default_features(interval),\n",
    "            base_timeframe=interval,\n"
    "            features=resolved_features,\n",
)

replace_once(
    "trade_rl/integrations/binance.py",
    "        sources_used=tuple(sorted(sources)),\n"
    "    )\n",
    "        sources_used=tuple(sorted(sources)),\n"
    "        feature_timeframes=tuple(\n"
    "            sorted(\n"
    "                {\n"
    "                    spec.resolved_timeframe(interval)\n"
    "                    for spec in resolved_features\n"
    "                },\n"
    "                key=_interval_ms,\n"
    "            )\n"
    "        ),\n"
    "    )\n",
)

replace_once(
    "trade_rl/integrations/binance.py",
    "    \"build_binance_market_dataset\",\n"
    "    \"vision_funding_url\",\n"
    "    \"vision_kline_url\",\n",
    "    \"binance_multitimeframe_feature_specs\",\n"
    "    \"build_binance_market_dataset\",\n"
    "    \"plan_vision_kline_urls\",\n"
    "    \"vision_funding_url\",\n"
    "    \"vision_kline_url\",\n"
    "    \"vision_monthly_kline_url\",\n",
)

replace_once(
    "trade_rl/cli/app.py",
    "        transport_mode=args.transport,\n"
    "        tick_sizes=_repeated_float_values(\n",
    "        transport_mode=args.transport,\n"
    "        feature_timeframes=tuple(args.feature_timeframe or ()),\n"
    "        tick_sizes=_repeated_float_values(\n",
)

replace_once(
    "trade_rl/cli/app.py",
    "    _write_json(\n"
    "        stdout,\n"
    "        {\n"
    "            \"artifact_digest\": artifact_digest,\n",
    "    payload: dict[str, object] = {\n"
    "        \"artifact_digest\": artifact_digest,\n"
    "        \"dataset_id\": result.dataset.dataset_id,\n"
    "        \"end_time\": end_time.isoformat(),\n"
    "        \"interval\": args.interval,\n"
    "        \"market\": args.market,\n"
    "        \"n_bars\": result.dataset.n_bars,\n"
    "        \"n_features\": result.dataset.n_features,\n"
    "        \"n_symbols\": result.dataset.n_symbols,\n"
    "        \"production_status\": \"NO-GO\",\n"
    "        \"schema\": \"binance_dataset_build_result_v1\",\n"
    "        \"sources_used\": list(result.sources_used),\n"
    "        \"start_time\": start_time.isoformat(),\n"
    "        \"symbols\": list(result.dataset.symbols),\n"
    "        \"transport\": args.transport,\n"
    "    }\n"
    "    if args.feature_timeframe:\n"
    "        payload[\"feature_timeframes\"] = list(result.feature_timeframes)\n"
    "    _write_json(stdout, payload)\n"
    "    return 0\n\n\n"
    "def _status_handler(area: str) -> Callable[[argparse.Namespace, TextIO], int]:\n"
    "    def handler(_: argparse.Namespace, stdout: TextIO) -> int:\n"
    "        _write_json(\n"
    "            stdout,\n"
    "            {\n"
    "                \"area\": area,\n"
    "                \"authoritative_package\": \"trade_rl\",\n"
    "                \"production_status\": \"NO-GO\",\n"
    "                \"schema\": \"component_status_v1\",\n"
    "            },\n"
    "        )\n"
    "        return 0\n\n\n"
    "def _removed_original_data_binance_payload_marker() -> None:\n"
    "    _write_json(\n"
    "        sys.stdout,\n"
    "        {\n"
    "            \"artifact_digest\": artifact_digest,\n",
)

# Remove the temporarily retained original tail inserted by the structural replacement.
path = Path("trade_rl/cli/app.py")
text = path.read_text(encoding="utf-8")
marker = "\n\ndef _removed_original_data_binance_payload_marker() -> None:\n"
if marker in text:
    start = text.index(marker)
    end = text.index("\n\ndef _status_handler", start)
    text = text[:start] + text[end:]
    path.write_text(text, encoding="utf-8")

replace_once(
    "trade_rl/cli/app.py",
    "    data_binance.add_argument(\"--start-time\", required=True)\n",
    "    data_binance.add_argument(\n"
    "        \"--feature-timeframe\",\n"
    "        choices=(\"15m\", \"30m\", \"1h\", \"2h\", \"4h\", \"6h\", \"8h\", \"12h\", \"1d\"),\n"
    "        action=\"append\",\n"
    "    )\n"
    "    data_binance.add_argument(\"--start-time\", required=True)\n",
)
