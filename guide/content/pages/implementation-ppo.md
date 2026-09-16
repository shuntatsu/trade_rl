## このページで答える問い

PPOは何を観測し、actionをどう売買意図へ変え、どの時点でhard risk・約定・reward計算を通るのか。

学習時だけ使える情報をpolicyへ混ぜず、実行時も同じObservation v2を使うことが中心契約です。

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

初回canonical M2のpolicy inputにはsymbol IDやdataset-global aggregateを入れません。

## 学習時の銘柄スケジュールは2方式

`fit_ppo_strategy`の既定は従来どおり`sequential`です。1つの`PPOTradingEnv`がfit対象銘柄をfull-window episodeごとにround-robinするため、既存のStudyやcandidateの意味は変わりません。

`interleaved`は明示的に選ぶ別layoutです。fit対象の各銘柄について1銘柄だけに固定した同じ`PPOTradingEnv`を1個ずつ作り、`DummyVecEnv`で同じPPO policyへ渡します。`rollout_steps_per_env`はcallerが明示し、全envを合わせたrollout sample数がminibatch size 64で割り切れなければfail closedにします。`total_timesteps`、Observation v2、reward、hard risk、約定・会計、network、entropy係数は変えません。

```text
sequential（既定）
  1つのenv: BTC full window → ETH full window → ...

interleaved（opt-in）
  BTC固定env ─┐
  ETH固定env ─┼─ DummyVecEnv → 同じPPO policy update
  ...         ─┘
```

これは学習データの並べ方を変える**未評価の実装能力**です。interleavedの方が儲かる、seed安定性が改善する、productionに適する、という結論はまだありません。developmentで比較するときはlayoutと`rollout_steps_per_env`を結果を見る前に別実験として固定します。 なお、`DummyVecEnv`はsub-envへ異なるreset seedを配るため、execution乱数まで同時に変えないよう`slippage_std > 0`の確率的slippageはinterleavedではfail closedです。

## 学習stepの全体像

```text
現在時点の市場・portfolio状態
            │
            │ 1. Observation v2を符号化
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
      次のObservation v2
```

actionが直接rewardになるわけではありません。必ずriskとexecution/accountingを通ります。

## 1. 観測を符号化する

利用不能またはnon-finiteなlocal valueはmaskし、stalenessを同じfeature順序で持ちます。現在のintentとweightも含め、固定順序の`float32` vectorへ連結します。

このcontractを固定することで、学習時にだけ便利な情報を後から追加してバックテストを有利にする余地を減らします。

## 2. policyがactionを選ぶ

PPO policyはObservation v2から離散actionを返します。このactionはまだ注文ではありません。

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

`fit_ppo_strategy`が返す`PPOIntentStrategy`は、実行時の`decide`でも同じ`_encode_observation`を使用します。

```text
学習時: StrategyObservation → _encode_observation → PPO
実行時: StrategyObservation → _encode_observation → deterministic predict
```

学習時と実行時で観測schemaを変えないことが重要です。

## 不変条件

- policy inputはObservation v2の固定5区分。
- unavailable/non-finite valueはそのままpolicyへ渡さない。
- fit-scope外symbol由来のdataset-global aggregateを初回policy inputへ入れない。
- PPO actionを直接P&L/rewardへ変換しない。
- hard riskと共通execution/accountingを必ず通す。
- 実行時も学習時と同じencoderを使う。

## まだ保証していないこと

Observation contractがcausalであることと、PPOが儲かることは別です。現行研究はPPO superiorityやproduction profitabilityを証明していません。
