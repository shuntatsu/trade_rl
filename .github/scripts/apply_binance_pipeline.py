from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected snippet not found in {path}: {old[:100]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


# InstrumentContract carries static execution constraints into dataset identity.
replace_once(
    "trade_rl/data/contracts.py",
    "    volume_unit: VolumeUnit = VolumeUnit.BASE_ASSET\n    contract_multiplier: float = 1.0\n",
    "    volume_unit: VolumeUnit = VolumeUnit.BASE_ASSET\n"
    "    contract_multiplier: float = 1.0\n"
    "    tick_size: float = 0.0\n"
    "    lot_size: float = 0.0\n"
    "    minimum_notional: float = 0.0\n",
)
replace_once(
    "trade_rl/data/contracts.py",
    "        if (\n            not math.isfinite(self.contract_multiplier)\n            or self.contract_multiplier <= 0.0\n        ):\n            raise ValueError(\"contract_multiplier must be finite and positive\")\n",
    "        if (\n"
    "            not math.isfinite(self.contract_multiplier)\n"
    "            or self.contract_multiplier <= 0.0\n"
    "        ):\n"
    "            raise ValueError(\"contract_multiplier must be finite and positive\")\n"
    "        for field_name, value in (\n"
    "            (\"tick_size\", self.tick_size),\n"
    "            (\"lot_size\", self.lot_size),\n"
    "            (\"minimum_notional\", self.minimum_notional),\n"
    "        ):\n"
    "            if not math.isfinite(value) or value < 0.0:\n"
    "                raise ValueError(f\"{field_name} must be finite and non-negative\")\n",
)
replace_once(
    "trade_rl/data/contracts.py",
    '            "volume_unit": self.volume_unit.value,\n            "contract_multiplier": self.contract_multiplier,\n',
    '            "volume_unit": self.volume_unit.value,\n'
    '            "contract_multiplier": self.contract_multiplier,\n'
    '            "tick_size": self.tick_size,\n'
    '            "lot_size": self.lot_size,\n'
    '            "minimum_notional": self.minimum_notional,\n',
)

# Dataset builder broadcasts static constraints over the aligned clock.
replace_once(
    "trade_rl/data/builder.py",
    "            contract_multipliers=np.asarray(\n                [contract.contract_multiplier for contract in instruments],\n                dtype=np.float64,\n            ),\n",
    "            contract_multipliers=np.asarray(\n"
    "                [contract.contract_multiplier for contract in instruments],\n"
    "                dtype=np.float64,\n"
    "            ),\n"
    "            tick_size=np.broadcast_to(\n"
    "                np.asarray(\n"
    "                    [contract.tick_size for contract in instruments],\n"
    "                    dtype=np.float64,\n"
    "                )[None, :],\n"
    "                (n_bars, n_symbols),\n"
    "            ).copy(),\n"
    "            lot_size=np.broadcast_to(\n"
    "                np.asarray(\n"
    "                    [contract.lot_size for contract in instruments],\n"
    "                    dtype=np.float64,\n"
    "                )[None, :],\n"
    "                (n_bars, n_symbols),\n"
    "            ).copy(),\n"
    "            minimum_notional=np.broadcast_to(\n"
    "                np.asarray(\n"
    "                    [contract.minimum_notional for contract in instruments],\n"
    "                    dtype=np.float64,\n"
    "                )[None, :],\n"
    "                (n_bars, n_symbols),\n"
    "            ).copy(),\n",
)

# Authoritative CLI command.
replace_once(
    "trade_rl/cli/app.py",
    "from dataclasses import asdict\nfrom typing import TextIO\n",
    "from dataclasses import asdict\nfrom datetime import UTC, datetime\nfrom typing import TextIO\n",
)
replace_once(
    "trade_rl/cli/app.py",
    "from trade_rl.data.source import CsvMarketDataSource\n",
    "from trade_rl.data.source import CsvMarketDataSource\n"
    "from trade_rl.integrations.binance import (\n"
    "    BinanceMarket,\n"
    "    BinanceTransportMode,\n"
    "    build_binance_market_dataset,\n"
    ")\n",
)
replace_once(
    "trade_rl/cli/app.py",
    "\ndef _status_handler(area: str) -> Callable[[argparse.Namespace, TextIO], int]:\n",
    "\ndef _parse_aware_datetime(value: str, *, field: str) -> datetime:\n"
    "    try:\n"
    "        parsed = datetime.fromisoformat(value.replace(\"Z\", \"+00:00\"))\n"
    "    except ValueError as error:\n"
    "        raise ValueError(f\"{field} must be an ISO-8601 datetime\") from error\n"
    "    if parsed.tzinfo is None or parsed.utcoffset() is None:\n"
    "        raise ValueError(f\"{field} must include a timezone\")\n"
    "    return parsed.astimezone(UTC)\n\n\n"
    "def _repeated_float_values(\n"
    "    values: list[float] | None,\n"
    "    *,\n"
    "    symbols: tuple[str, ...],\n"
    "    field: str,\n"
    ") -> tuple[float, ...] | None:\n"
    "    if not values:\n"
    "        return None\n"
    "    if len(values) != len(symbols):\n"
    "        raise ValueError(f\"{field} must be supplied once per symbol\")\n"
    "    return tuple(values)\n\n\n"
    "def _repeated_datetime_values(\n"
    "    values: list[str] | None,\n"
    "    *,\n"
    "    symbols: tuple[str, ...],\n"
    ") -> tuple[datetime, ...] | None:\n"
    "    if not values:\n"
    "        return None\n"
    "    if len(values) != len(symbols):\n"
    "        raise ValueError(\"listed-at must be supplied once per symbol\")\n"
    "    return tuple(\n"
    "        _parse_aware_datetime(value, field=\"listed-at\") for value in values\n"
    "    )\n\n\n"
    "def _data_binance(args: argparse.Namespace, stdout: TextIO) -> int:\n"
    "    symbols = tuple(args.symbol)\n"
    "    start_time = _parse_aware_datetime(args.start_time, field=\"start-time\")\n"
    "    end_time = _parse_aware_datetime(args.end_time, field=\"end-time\")\n"
    "    result = build_binance_market_dataset(\n"
    "        market=args.market,\n"
    "        symbols=symbols,\n"
    "        interval=args.interval,\n"
    "        start_time=start_time,\n"
    "        end_time=end_time,\n"
    "        transport_mode=args.transport,\n"
    "        tick_sizes=_repeated_float_values(\n"
    "            args.tick_size, symbols=symbols, field=\"tick-size\"\n"
    "        ),\n"
    "        lot_sizes=_repeated_float_values(\n"
    "            args.lot_size, symbols=symbols, field=\"lot-size\"\n"
    "        ),\n"
    "        minimum_notionals=_repeated_float_values(\n"
    "            args.minimum_notional,\n"
    "            symbols=symbols,\n"
    "            field=\"minimum-notional\",\n"
    "        ),\n"
    "        listed_ats=_repeated_datetime_values(args.listed_at, symbols=symbols),\n"
    "    )\n"
    "    artifact_digest = publish_market_dataset_artifact(\n"
    "        args.output, result.dataset\n"
    "    ).artifact_digest\n"
    "    _write_json(\n"
    "        stdout,\n"
    "        {\n"
    "            \"artifact_digest\": artifact_digest,\n"
    "            \"dataset_id\": result.dataset.dataset_id,\n"
    "            \"end_time\": end_time.isoformat(),\n"
    "            \"interval\": args.interval,\n"
    "            \"market\": args.market,\n"
    "            \"n_bars\": result.dataset.n_bars,\n"
    "            \"n_features\": result.dataset.n_features,\n"
    "            \"n_symbols\": result.dataset.n_symbols,\n"
    "            \"production_status\": \"NO-GO\",\n"
    "            \"schema\": \"binance_dataset_build_result_v1\",\n"
    "            \"sources_used\": list(result.sources_used),\n"
    "            \"start_time\": start_time.isoformat(),\n"
    "            \"symbols\": list(result.dataset.symbols),\n"
    "            \"transport\": args.transport,\n"
    "        },\n"
    "    )\n"
    "    return 0\n\n\n"
    "def _status_handler(area: str) -> Callable[[argparse.Namespace, TextIO], int]:\n",
)
replace_once(
    "trade_rl/cli/app.py",
    "    data_build.add_argument(\"--output\", required=True)\n    data_build.set_defaults(handler=_data_build)\n\n",
    "    data_build.add_argument(\"--output\", required=True)\n"
    "    data_build.set_defaults(handler=_data_build)\n\n"
    "    data_binance = data_commands.add_parser(\n"
    "        \"binance\",\n"
    "        help=\"build a deterministic dataset from public Binance data\",\n"
    "    )\n"
    "    data_binance.add_argument(\n"
    "        \"--market\",\n"
    "        choices=tuple(item.value for item in BinanceMarket),\n"
    "        required=True,\n"
    "    )\n"
    "    data_binance.add_argument(\"--symbol\", action=\"append\", required=True)\n"
    "    data_binance.add_argument(\n"
    "        \"--interval\",\n"
    "        choices=(\"15m\", \"30m\", \"1h\", \"2h\", \"4h\", \"6h\", \"8h\", \"12h\", \"1d\"),\n"
    "        required=True,\n"
    "    )\n"
    "    data_binance.add_argument(\"--start-time\", required=True)\n"
    "    data_binance.add_argument(\"--end-time\", required=True)\n"
    "    data_binance.add_argument(\n"
    "        \"--transport\",\n"
    "        choices=tuple(item.value for item in BinanceTransportMode),\n"
    "        default=BinanceTransportMode.AUTO.value,\n"
    "    )\n"
    "    data_binance.add_argument(\"--tick-size\", type=float, action=\"append\")\n"
    "    data_binance.add_argument(\"--lot-size\", type=float, action=\"append\")\n"
    "    data_binance.add_argument(\n"
    "        \"--minimum-notional\", type=float, action=\"append\"\n"
    "    )\n"
    "    data_binance.add_argument(\"--listed-at\", action=\"append\")\n"
    "    data_binance.add_argument(\"--output\", required=True)\n"
    "    data_binance.set_defaults(handler=_data_binance)\n\n",
)
