from __future__ import annotations

import json
from pathlib import Path

from guide.tools.code_symbols import build_symbol_index

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "guide/content/meta/implementation-ppo.json"
PAGE = ROOT / "guide/content/pages/implementation-ppo.md"
REVISION = "93d3f2a1b66e517c0ec265df3084e07e05e3893c"


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"expected exactly one replacement target: {old[:80]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    index = build_symbol_index(ROOT / "trade_rl", revision=REVISION)
    symbols = {
        item["qualified_name"]: item
        for item in index["symbols"]
        if isinstance(item, dict) and isinstance(item.get("qualified_name"), str)
    }

    meta = json.loads(META.read_text(encoding="utf-8"))
    meta["summary"] = (
        "PPO Observation v2の基準経路と、同一Dataset rowのBTC regime contextを追加する"
        "候補Observation v3、環境step内のhard riskと約定、学習後のPPOIntentStrategyまでを"
        "実装と対応付けて追います。"
    )
    keywords = list(meta["keywords"])
    for keyword in ("Observation v3", "global_reference_value", "BTCUSDT"):
        if keyword not in keywords:
            keywords.append(keyword)
    meta["keywords"] = keywords

    refs = meta["code_references"]
    if not isinstance(refs, list):
        raise SystemExit("implementation-ppo code_references is malformed")
    by_id = {ref.get("id"): ref for ref in refs if isinstance(ref, dict)}
    if "encode-global-btc-regime" not in by_id:
        insert_at = (
            next(i for i, ref in enumerate(refs) if ref.get("id") == "encode-observation")
            + 1
        )
        refs.insert(
            insert_at,
            {
                "id": "encode-global-btc-regime",
                "symbol": "trade_rl.strategies.rl.ppo._encode_with_global_context",
                "kind": "function",
                "source_sha256": "0" * 64,
                "label_ja": "候補BTCレジーム観測を追加",
                "description_ja": (
                    "既存Observation v2を壊さず、global_contextが明示された候補だけ同一Dataset rowの"
                    "BTCUSDT 1h__log_return_24bar値・利用可否・既存stalenessを状態直前へ追加します。"
                ),
                "variables": [],
                "tests": ["tests/strategies/test_ppo_global_btc_regime.py"],
            },
        )
    by_id = {ref.get("id"): ref for ref in refs if isinstance(ref, dict)}
    if "global-btc-regime-channels" not in by_id:
        insert_at = next(
            i for i, ref in enumerate(refs) if ref.get("id") == "encode-global-btc-regime"
        )
        refs.insert(
            insert_at,
            {
                "id": "global-btc-regime-channels",
                "symbol": "trade_rl.strategies.rl.ppo._global_btc_regime_channels",
                "kind": "function",
                "source_sha256": "0" * 64,
                "label_ja": "同一rowのBTC参照チャネルを読む",
                "description_ja": (
                    "canonical Datasetの同一indexからBTCUSDT / 1h__log_return_24barを読み、"
                    "unavailableまたはnon-finiteなら値とusableを0へfail-closedし、既存stalenessをそのまま使います。"
                ),
                "variables": [],
                "tests": ["tests/strategies/test_ppo_global_btc_regime.py"],
            },
        )

    for ref in refs:
        symbol_name = ref["symbol"]
        symbol = symbols.get(symbol_name)
        if symbol is None:
            raise SystemExit(f"missing guide code symbol: {symbol_name}")
        ref["source_sha256"] = symbol["source_sha256"]
        if ref.get("id") == "fit-ppo":
            ref["description_ja"] = (
                "既定はObservation v2のまま、global_contextを明示した候補だけObservation v3で"
                "stable-baselines3 PPOを学習し、同じcontext契約を持つPPOIntentStrategyを返します。"
            )
        elif ref.get("id") == "ppo-decide":
            ref["description_ja"] = (
                "実行時も学習時と同じglobal_context設定で観測を作り、deterministic predictのactionを"
                "SHORT・FLAT・LONGの売買意図へ戻します。"
            )
    META.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    page = PAGE.read_text(encoding="utf-8")
    page = _replace_once(
        page,
        "学習時だけ使える情報をpolicyへ混ぜず、実行時も同じObservation v2を使うことが中心契約です。",
        "既定の学習・実行経路は同じObservation v2を使います。#607のControlled Factorでは、明示指定した候補だけ同一Dataset rowのBTCレジーム情報を加えたObservation v3を使い、学習時と実行時のschemaを一致させます。",
    )
    page = _replace_once(
        page,
        "初回canonical M2のpolicy inputにはsymbol IDやdataset-global aggregateを入れません。",
        """初回canonical M2とglobal context未指定時のpolicy inputにはsymbol IDやdataset-global aggregateを入れません。既定のObservation v2はこの5区分のまま変わりません。

## 候補Observation v3は同一rowのBTCレジームを3チャネルだけ追加

#607のControlled Factorで`global_context=\"ppo_global_btc_regime_context\"`を明示した場合だけ、`local_staleness`と現在stateの間へ次の3チャネルを追加します。

- `global_reference_value`: 同じDataset rowの`BTCUSDT / 1h__log_return_24bar`。利用不能時は0。
- `global_reference_available_and_finite`: availabilityとfinite判定を満たすときだけ1。
- `global_reference_normalized_staleness`: Datasetに既に保存された同じ参照featureのstaleness。

```text
Observation v2の local 3区分
              +
同一row BTC 24h return value / usable / staleness
              +
 current_intent / current_weight
              ↓
   _encode_with_global_context
              ↓
ppo_observation_v3_global_btc_regime
```

参照値は**同じdecision row**からだけ読みます。前後rowへのas-of置換や再計算stalenessは使いません。BTC参照がunavailableまたはnon-finiteならvalueとusableは0へfail-closedし、Dataset identity自体は変更しません。symbol ID、symbol embedding、銘柄別係数、cross-sectional集約はこの候補に追加しません。""",
    )
    page = _replace_once(
        page,
        "            │ 1. Observation v2を符号化",
        "            │ 1. 選択されたObservation契約を符号化",
    )
    page = _replace_once(
        page,
        "      次のObservation v2",
        "      次の同一Observation契約",
    )
    page = _replace_once(
        page,
        "PPO policyはObservation v2から離散actionを返します。このactionはまだ注文ではありません。",
        "PPO policyは既定のObservation v2、または#607で明示した候補Observation v3から離散actionを返します。このactionはまだ注文ではありません。",
    )
    page = _replace_once(
        page,
        "`fit_ppo_strategy`が返す`PPOIntentStrategy`は、実行時の`decide`でも同じ`_encode_observation`を使用します。\n\n```text\n学習時: StrategyObservation → _encode_observation → PPO\n実行時: StrategyObservation → _encode_observation → deterministic predict\n```\n\n学習時と実行時で観測schemaを変えないことが重要です。",
        """`fit_ppo_strategy`が返す`PPOIntentStrategy`は、学習時と同じ`global_context`設定を保持します。既定経路はObservation v2、#607候補は`_encode_with_global_context`経由のObservation v3で、どちらも学習時と実行時のschemaを一致させます。

```text
既定: 学習 / 実行 → Observation v2
#607候補: 学習 / 実行 → 同一row BTC context付き Observation v3
```

学習時だけglobal contextを見せたり、実行時だけ別rowを参照したりしないことが重要です。""",
    )
    page = _replace_once(
        page,
        "- policy inputはObservation v2の固定5区分。\n- unavailable/non-finite valueはそのままpolicyへ渡さない。\n- fit-scope外symbol由来のdataset-global aggregateを初回policy inputへ入れない。",
        "- global context未指定時はObservation v2の固定5区分を維持する。\n- #607候補だけ、同一rowのBTC参照3チャネルをstate直前へ追加したObservation v3を使う。\n- unavailable/non-finiteなlocal/BTC参照値はそのままpolicyへ渡さない。\n- BTC参照はcanonical Datasetの同一rowと既存stalenessだけを使い、Dataset identityを変更しない。\n- symbol ID・symbol embedding・銘柄別係数・cross-sectional aggregateをこの候補へ混ぜない。",
    )
    PAGE.write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
