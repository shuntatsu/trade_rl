import json
from pathlib import Path
from typing import Optional

from mars_lite.data.sources import SyntheticSource, create_source
from mars_lite.features.feature_pipeline import FeaturePipeline, FeatureSet

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "SUIUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "APTUSDT",
    "ARBUSDT",
    "OPUSDT",
]


def build_feature_set(args, output_dir: Optional[Path] = None) -> FeatureSet:
    if args.source == "synthetic":
        source = SyntheticSource(
            n_days=args.days,
            alpha=args.alpha,
            alpha_strength=args.alpha_strength,
            seed=args.seed,
        )
        symbols = source.symbols
    else:
        symbols = args.symbols or DEFAULT_SYMBOLS
        if args.source == "csv":
            kwargs = {"data_dir": args.data}
        elif args.source == "hyperliquid":
            kwargs = {"days": args.days}
        elif args.source == "postgres":
            kwargs = {
                "dsn": args.pg_dsn,
                "source": args.pg_source,
                "derivatives_source": args.pg_derivatives_source,
                "orderflow_source": args.pg_derivatives_source,
            }
        else:
            kwargs = {}
        source = create_source(args.source, symbols, **kwargs)
        from mars_lite.data.quality import run_quality_gate

        qrep = run_quality_gate(source, symbols, base_timeframe="1h")
        print(qrep.summary())
        if output_dir is not None:
            with open(
                output_dir / "data_quality_report.json", "w", encoding="utf-8"
            ) as f:
                json.dump(qrep.to_dict(), f, indent=2, ensure_ascii=False)
        symbols = qrep.passing_symbols
        if len(symbols) < 2:
            raise ValueError(
                "品質ゲート通過銘柄が2未満です。データを確認してください。"
            )
    return FeaturePipeline(symbols).build(source)
