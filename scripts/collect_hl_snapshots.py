"""
Hyperliquidネイティブ建玉残高・funding・プレミアムのスナップショット収集

Hyperliquidの公開info APIは OI/funding/premium の**現在値**しか返さない
（履歴APIが無い）。過去分はBinance代理で埋める（fetch_hl_derivatives.py）が、
前向き（これから先）の期間についてはHLネイティブの値を直接蓄積した方が
精度が高い。本スクリプトは metaAndAssetCtxs を1回叩いて全銘柄分を
CSVに追記する「単発実行」ツールで、定期実行（cron/タスクスケジューラ）
を前提に設計する。

出力: data/hyperliquid/snapshots/{COIN}_ctx.csv
    列: timestamp, open_interest, funding, premium, mark_px, oracle_px, day_ntl_vlm

使い方（単発）:
    python scripts/collect_hl_snapshots.py --symbols BTCUSDT ETHUSDT SOLUSDT

Windowsでの定期実行（タスクスケジューラ、1時間毎）:
    schtasks /create /tn "HL Snapshot Collector" /tr ^
        "\"C:\\path\\to\\python.exe\" \"C:\\dev\\trade_rl\\scripts\\collect_hl_snapshots.py\"" ^
        /sc hourly /mo 1

または cron（Linux/Mac）:
    0 * * * * cd /path/to/trade_rl && python scripts/collect_hl_snapshots.py
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

_HL_INFO_URL = "https://api.hyperliquid.xyz/info"

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "SUIUSDT",
    "DOGEUSDT",
]


def _coin(symbol: str) -> str:
    s = symbol.upper()
    for suf in ("USDT", "USDC", "PERP"):
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return s


def fetch_snapshot() -> pd.DataFrame:
    """metaAndAssetCtxsを1回取得し、コイン別の現在値DataFrameを返す"""
    resp = requests.post(_HL_INFO_URL, json={"type": "metaAndAssetCtxs"}, timeout=20)
    resp.raise_for_status()
    meta, ctxs = resp.json()
    universe = meta["universe"]
    now = pd.Timestamp.now().floor("min")

    rows = []
    for u, ctx in zip(universe, ctxs):
        rows.append(
            {
                "timestamp": now,
                "coin": u["name"],
                "open_interest": float(ctx.get("openInterest", 0.0)),
                "funding": float(ctx.get("funding", 0.0)),
                "premium": float(ctx.get("premium", 0.0) or 0.0),
                "mark_px": float(ctx.get("markPx", 0.0) or 0.0),
                "oracle_px": float(ctx.get("oraclePx", 0.0) or 0.0),
                "day_ntl_vlm": float(ctx.get("dayNtlVlm", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def append_snapshot(df: pd.DataFrame, coin: str, out_dir: Path) -> None:
    path = out_dir / f"{coin}_ctx.csv"
    row = df[df["coin"] == coin].drop(columns=["coin"])
    if row.empty:
        return
    header = not path.exists()
    row.to_csv(path, mode="a", index=False, header=header)


def main():
    ap = argparse.ArgumentParser(
        description="Hyperliquidネイティブスナップショット収集（単発実行）"
    )
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--out-dir", default="./data/hyperliquid/snapshots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    snap = fetch_snapshot()
    coins = [_coin(s) for s in args.symbols]
    n = 0
    for coin in coins:
        if coin in set(snap["coin"]):
            append_snapshot(snap, coin, out_dir)
            n += 1
        else:
            print(f"  [skip] {coin}: not found in universe")
    print(f"Snapshot appended for {n}/{len(coins)} coins -> {out_dir}")


if __name__ == "__main__":
    main()
