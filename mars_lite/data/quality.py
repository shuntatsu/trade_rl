"""
データ品質ゲート

実データソースを学習に使う前に、各銘柄が最低限の品質基準
（十分なバー数・欠損率・異常価格ジャンプなし）を満たすか検査する。
不合格銘柄は学習対象から除外する。
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from mars_lite.data.data_utils import TF_TO_MINUTES
from mars_lite.data.sources import DataSource


@dataclass
class SymbolQuality:
    symbol: str
    passed: bool
    n_bars: int
    missing_ratio: float
    max_jump: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "passed": self.passed,
            "n_bars": self.n_bars,
            "missing_ratio": self.missing_ratio,
            "max_jump": self.max_jump,
            "reasons": self.reasons,
        }


@dataclass
class QualityReport:
    results: List[SymbolQuality]

    @property
    def passing_symbols(self) -> List[str]:
        return [r.symbol for r in self.results if r.passed]

    def to_dict(self) -> Dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "passing_symbols": self.passing_symbols,
        }

    def summary(self) -> str:
        lines = ["[Data Quality Gate]"]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"  {r.symbol:<12} {status}  bars={r.n_bars} "
                f"missing={r.missing_ratio:.1%} max_jump={r.max_jump:.1%}"
                + (f"  reasons={r.reasons}" if r.reasons else "")
            )
        lines.append(f"  Passing: {len(self.passing_symbols)}/{len(self.results)}")
        return "\n".join(lines)


def check_symbol(
    source: DataSource,
    symbol: str,
    base_timeframe: str = "1h",
    min_bars: int = 400,
    max_missing_ratio: float = 0.1,
    max_jump: float = 0.5,
) -> SymbolQuality:
    """1銘柄の品質を検査"""
    reasons = []
    df = source.load_klines(symbol, base_timeframe)
    n_bars = len(df)

    if n_bars < min_bars:
        reasons.append(f"too_few_bars({n_bars}<{min_bars})")
        return SymbolQuality(symbol, False, n_bars, 1.0, 0.0, reasons)

    ts = df["timestamp"]
    step_minutes = TF_TO_MINUTES[base_timeframe]
    expected_bars = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 60.0 / step_minutes
    missing_ratio = float(max(0.0, 1.0 - n_bars / max(expected_bars, 1)))

    close = df["close"].to_numpy()
    rel_jump = np.abs(np.diff(np.log(np.clip(close, 1e-12, None))))
    jump = float(rel_jump.max()) if len(rel_jump) else 0.0

    if missing_ratio > max_missing_ratio:
        reasons.append(f"missing_ratio({missing_ratio:.2%}>{max_missing_ratio:.0%})")
    if jump > max_jump:
        reasons.append(f"price_jump({jump:.2%}>{max_jump:.0%})")
    if df[["open", "high", "low", "close"]].le(0).any().any():
        reasons.append("non_positive_price")

    return SymbolQuality(
        symbol=symbol,
        passed=len(reasons) == 0,
        n_bars=n_bars,
        missing_ratio=missing_ratio,
        max_jump=jump,
        reasons=reasons,
    )


def run_quality_gate(
    source: DataSource,
    symbols: List[str],
    base_timeframe: str = "1h",
    **kwargs,
) -> QualityReport:
    """全銘柄の品質を検査してレポートを返す"""
    results = [check_symbol(source, s, base_timeframe, **kwargs) for s in symbols]
    return QualityReport(results=results)
