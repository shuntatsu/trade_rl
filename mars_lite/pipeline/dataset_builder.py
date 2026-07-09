import json
from pathlib import Path
from typing import Optional

from mars_lite.data.data_utils import TF_TO_MINUTES
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
    base_timeframe = getattr(args, "base_timeframe", "1h")
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
        elif args.source in ("hyperliquid", "bitget", "okx"):
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

        qrep = run_quality_gate(source, symbols, base_timeframe=base_timeframe)
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
    fs = FeaturePipeline(symbols, base_timeframe=base_timeframe).build(source)

    # ウォームアップ切り捨て: 最長のローリング窓（1dTFのvol_ratio長期側=100日
    # ≈2400本@1h）が埋まるまで特徴が不完全（min_periods未満はゼロ埋め）。
    # 実効学習期間をNdays確保したい場合は取得を(N+warmup_days)日にして
    # 先頭warmup_days日を切り捨てる（例: 500日学習→600日取得+warmup 100）。
    warmup_days = getattr(args, "warmup_days", 0)
    if warmup_days > 0:
        bar_minutes = TF_TO_MINUTES[base_timeframe]
        warmup_bars = int(warmup_days * 24 * 60 / bar_minutes)
        if warmup_bars >= fs.n_bars:
            raise ValueError(
                f"--warmup-days {warmup_days} はデータ長({fs.n_bars}本)以上です。"
            )
        fs = fs.slice(warmup_bars, fs.n_bars)
        print(
            f"[Warmup] 先頭{warmup_days}日（{warmup_bars}本）を切り捨て。"
            f"残り{fs.n_bars}本を学習/検証に使用。"
        )
    return fs
