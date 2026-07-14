from __future__ import annotations

from pathlib import Path

path = Path("trade_rl/cli/app.py")
text = path.read_text(encoding="utf-8")
start = text.index("def _data_build(")
end = text.index("def _status_handler(", start)
replacement = '''def _data_build(args: argparse.Namespace, stdout: TextIO) -> int:
    request = load_market_build_request(args.config)
    dataset = MarketDatasetBuilder(request.config).build(
        CsvMarketDataSource(request.source_root),
        request.instruments,
    )
    artifact_digest = publish_market_dataset_artifact(
        args.output, dataset
    ).artifact_digest
    _write_json(
        stdout,
        {
            "artifact_digest": artifact_digest,
            "dataset_id": dataset.dataset_id,
            "n_bars": dataset.n_bars,
            "n_features": dataset.n_features,
            "n_symbols": dataset.n_symbols,
            "schema": "market_dataset_build_result_v1",
        },
    )
    return 0


def _parse_aware_datetime(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _repeated_float_values(
    values: list[float] | None,
    *,
    symbols: tuple[str, ...],
    field: str,
) -> tuple[float, ...] | None:
    if not values:
        return None
    if len(values) != len(symbols):
        raise ValueError(f"{field} must be supplied once per symbol")
    return tuple(values)


def _repeated_datetime_values(
    values: list[str] | None,
    *,
    symbols: tuple[str, ...],
) -> tuple[datetime, ...] | None:
    if not values:
        return None
    if len(values) != len(symbols):
        raise ValueError("listed-at must be supplied once per symbol")
    return tuple(_parse_aware_datetime(value, field="listed-at") for value in values)


def _data_binance(args: argparse.Namespace, stdout: TextIO) -> int:
    symbols = tuple(args.symbol)
    start_time = _parse_aware_datetime(args.start_time, field="start-time")
    end_time = _parse_aware_datetime(args.end_time, field="end-time")
    result = build_binance_market_dataset(
        market=args.market,
        symbols=symbols,
        interval=args.interval,
        start_time=start_time,
        end_time=end_time,
        transport_mode=args.transport,
        feature_timeframes=tuple(args.feature_timeframe or ()),
        tick_sizes=_repeated_float_values(
            args.tick_size, symbols=symbols, field="tick-size"
        ),
        lot_sizes=_repeated_float_values(
            args.lot_size, symbols=symbols, field="lot-size"
        ),
        minimum_notionals=_repeated_float_values(
            args.minimum_notional,
            symbols=symbols,
            field="minimum-notional",
        ),
        listed_ats=_repeated_datetime_values(args.listed_at, symbols=symbols),
    )
    artifact_digest = publish_market_dataset_artifact(
        args.output, result.dataset
    ).artifact_digest
    payload: dict[str, object] = {
        "artifact_digest": artifact_digest,
        "dataset_id": result.dataset.dataset_id,
        "end_time": end_time.isoformat(),
        "interval": args.interval,
        "market": args.market,
        "n_bars": result.dataset.n_bars,
        "n_features": result.dataset.n_features,
        "n_symbols": result.dataset.n_symbols,
        "production_status": "NO-GO",
        "schema": "binance_dataset_build_result_v1",
        "sources_used": list(result.sources_used),
        "start_time": start_time.isoformat(),
        "symbols": list(result.dataset.symbols),
        "transport": args.transport,
    }
    if args.feature_timeframe:
        payload["feature_timeframes"] = list(result.feature_timeframes)
    _write_json(stdout, payload)
    return 0


'''
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
