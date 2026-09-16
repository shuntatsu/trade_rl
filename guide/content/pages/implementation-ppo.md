## このページで答える問い

PPOは何を観測し、actionをどう売買意図へ変え、どの時点でhard risk・約定・reward計算を通るのか。

既定の学習・実行経路は同じObservation v2を使います。#607のControlled Factorでは、明示指定した候補だけ同一Dataset rowのBTCレジーム情報を加えたObservation v3を使い、学習時と実行時のschemaを一致させます。

## PPO Observation v2は5区分

policy inputは次の固定順序です。

| 順番 | 区分 | 意味 |
| ---: | --- | --- |
| 1 | `local_values` | 選択した現在時点のローカル特徴量 |
| 2 | `local_available` / finite mask | 値をpolicyへ渡してよいか |
| 3 | `local_staleness` | 選択featureと同じ順序の鮮度遅延 |
| 4 | `current_intent` | 現在のSHORT / FLAT / LONG状態 |
| 5 | `current_weight` | 現在のportfolio weight |

```text
local_values
    + local_available / finite
    + local_staleness
    + current_intent
    + current_weight
              ↓
       _encode_observation
              ↓
      PPO observation vector
```

初回canonical M2とglobal context未指定時のpolicy inputにはsymbol IDやdataset-global aggregateを入れません。既定のObservation v2はこの5区分のまま変わりません。

## 候補Observation v3は同一rowのBTCレジームを3チャネルだけ追加

#607のControlled Factorで`global_context="ppo_global_btc_regime_context"`を明示した場合だけ、`local_staleness`と現在stateの間へ次の3チャネルを追加します。

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

参照値は**同じdecision row**からだけ読みます。前後rowへのas-of置換や再計算stalenessは使いません。BTC参照がunavailableまたはnon-finiteならvalueとusableは0へfail-closedし、Dataset identity自体は変更しません。symbol ID、symbol embedding、銘柄別係数、cross-sectional集約はこの候補に追加しません。

## 学習stepの全体像

```text
現在時点の市場・portfolio状態
            │
            │ 1. 選択されたObservation契約を符号化
            ▼
      PPO observation
            │
            │ 2. policyが離散actionを選ぶ
            ▼
         PPO action
            │
            │ 3. SHORT / FLAT / LONGへ変換
            ▼
        proposal_weight
            │
            │ 4. hard risk
            ▼
        target_weight
            │
            │ 5. MarketExecutorで約定・会計
            ▼
 interval_net_return + BookState
            │
            │ 6. reward = log1p(interval_net_return)
            ▼
      次の同一Observation契約
```

actionが直接rewardになるわけではありません。必ずriskとexecution/accountingを通ります。

## 1. 観測を符号化する

利用不能またはnon-finiteなlocal valueはmaskし、stalenessを同じfeature順序で持ちます。現在のintentとweightも含め、固定順序の`float32` vectorへ連結します。

このcontractを固定することで、学習時にだけ便利な情報を後から追加してバックテストを有利にする余地を減らします。

## 2. policyがactionを選ぶ

PPO policyは既定のObservation v2、または#607で明示した候補Observation v3から離散actionを返します。このactionはまだ注文ではありません。

## 3. actionを売買意図へ変換する

離散actionを`SHORT` / `FLAT` / `LONG`へ写像し、現在のportfolio状態からproposal weightを作ります。

同じintentを維持する場合、価格driftだけを理由に毎decisionで機械的に元weightへ戻すのが標準ではありません。標準はquantity-preserving holdです。

## 4. hard riskを通す

PPOもrule strategyと同じ`PreTradeRisk`を通ります。PPOだけrisk上限を回避する経路はありません。

## 5. 約定・会計を通す

制約済みtargetを`MarketExecutor`へ渡し、fill、cost、funding、borrow、BookState、区間net returnを共通経路で更新します。

## 6. net returnからrewardを作る

rewardは約定・コスト反映後の区間returnから計算します。

```text
reward = log1p(interval_net_return)
```

したがって、取引コストを無視したpolicy scoreを別経路で最大化しているわけではありません。

## 学習後も同じ観測契約を使う

`fit_ppo_strategy`が返す`PPOIntentStrategy`は、学習時と同じ`global_context`設定を保持します。既定経路はObservation v2、#607候補は`_encode_with_global_context`経由のObservation v3で、どちらも学習時と実行時のschemaを一致させます。

```text
既定: 学習 / 実行 → Observation v2
#607候補: 学習 / 実行 → 同一row BTC context付き Observation v3
```

学習時だけglobal contextを見せたり、実行時だけ別rowを参照したりしないことが重要です。

## 不変条件

- global context未指定時はObservation v2の固定5区分を維持する。
- #607候補だけ、同一rowのBTC参照3チャネルをstate直前へ追加したObservation v3を使う。
- unavailable/non-finiteなlocal/BTC参照値はそのままpolicyへ渡さない。
- BTC参照はcanonical Datasetの同一rowと既存stalenessだけを使い、Dataset identityを変更しない。
- symbol ID・symbol embedding・銘柄別係数・cross-sectional aggregateをこの候補へ混ぜない。
- PPO actionを直接P&L/rewardへ変換しない。
- hard riskと共通execution/accountingを必ず通す。
- 実行時も学習時と同じencoderを使う。

## まだ保証していないこと

Observation contractがcausalであることと、PPOが儲かることは別です。現行研究はPPO superiorityやproduction profitabilityを証明していません。
