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

## 学習時の銘柄スケジュールは3方式

`fit_ppo_strategy`の既定は従来どおり`sequential`です。1つの`PPOTradingEnv`がfit対象銘柄をfull-window episodeごとにround-robinするため、既存のStudyやcandidateの意味は変わりません。

`interleaved`は明示的に選ぶ別layoutです。fit対象の各銘柄について1銘柄だけに固定した同じ`PPOTradingEnv`を1個ずつ作り、`DummyVecEnv`で同じPPO policyへ渡します。`rollout_steps_per_env`はcallerが明示し、全envを合わせたrollout sample数がminibatch size 64で割り切れなければfail closedにします。

`shared_cash`はもう一つのopt-in layoutです。各symbol slotは従来どおりObservation v2と3値actionを使いますが、custom VecEnvが同じ時点の全actionを先に集め、portfolio proposalを一度だけriskへ通して、一つのshared cash / BookStateを一度だけexecutionします。全slotには同じportfolio log-return rewardを返します。

```text
sequential（既定）
  1つのenv: BTC full window → ETH full window → ...

interleaved（opt-in）
  BTC固定env ─┐
  ETH固定env ─┼─ DummyVecEnv → 同じPPO policy update
  ...         ─┘

shared_cash（opt-in）
  BTC Observation ─┐
  ETH Observation ─┼─ 同時action収集 → portfolio risk 1回 → shared execution 1回
  ...             ─┘                         ↓
                                  同じteam rewardを各slotへ
```

これらは**未評価の実装能力**です。shared-cashはcash accountだけでなく、portfolio-level risk couplingと全slot共通team rewardを同時に持ちます。またlocal Observation v2には他symbol weightやshared cashを追加しないため、各slotから見るとportfolio stateは部分観測です。したがって「accountだけを変えた純粋な比較」とは扱いません。developmentで比較するときはlayout、`rollout_steps_per_env`、team-reward semanticsを結果を見る前に固定し、per-symbol contribution rewardやjoint observationは別factorにします。学習deviceはCPUへ固定し、実行マシンのGPU有無だけでpolicy学習経路が変わらないようにします。なお、`DummyVecEnv`はsub-envへ異なるreset seedを配るため、execution乱数まで同時に変えないよう`slippage_std > 0`の確率的slippageはinterleavedではfail closedです。

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

入力値の標準化は明示的に選べる追加機能です。学習対象の期間・銘柄だけで平均と標準偏差を計算し、学習中も評価中も同じ係数を使います。欠損値は標準化後も0とし、利用可否・鮮度・保有状態の意味は変えません。既定では従来の値をそのまま使います。

標準化したモデルには係数も必要です。保存時にモデルと係数を一つの識別子で結び付け、読み込み時には識別子と入力項目の順序を照合します。係数が欠けたり、別のモデルの係数に置き換わった場合は推論を始めません。利益が改善するかは、別の実データ比較で判定します。

このcontractを固定することで、学習時にだけ便利な情報を後から追加してバックテストを有利にする余地を減らします。

## 2. policyがactionを選ぶ

PPO policyはObservation v2から離散actionを返します。このactionはまだ注文ではありません。

## 3. actionを売買意図へ変換する

離散actionを`SHORT` / `FLAT` / `LONG`へ写像し、現在のportfolio状態からproposal weightを作ります。

同じintentを維持する場合、価格driftだけを理由に毎decisionで機械的に元weightへ戻すのが標準ではありません。標準はquantity-preserving holdです。

## 4. hard riskを通す

PPOもrule strategyと同じ`PreTradeRisk`を通ります。PPOだけrisk上限を回避する経路はありません。

学習用のリスク設定は明示的に指定でき、episodeをresetしても同じ設定を使います。省略時は従来の設定を保ちます。学習と評価で保有上限や下落時の縮小条件が違うと、同じ売買意図でも約定やコストが変わります。両者を合わせる実験ではリスク設定だけを変更し、報酬・観測・学習データの並べ方は別の比較として扱います。

PPOでも `max_turnover` と `drawdown_deleveraging` はそのstepのtransient risk projectionです。`max_turnover` が静的hard capと同時に発火しても、そのstepのtargetを `desired_quantity` へ書き戻しません。これにより反転actionの元proposalを保持したまま次stepのturnover projectionを再計算できます。transient理由がない静的hard capだけはbounded proposalとして再束縛します。このproposal/risk state分離はcanonical replayと同じです。

## 5. 約定・会計を通す

制約済みtargetを`MarketExecutor`へ渡し、fill、cost、funding、borrow、BookState、区間net returnを共通経路で更新します。

stepの`info`では、risk後の`target_weight`と約定後の`realized_weight`を分け、risk理由、fill ratio、requested/filled turnover、cost・funding・borrow・dividend・cash-interestの金額、termination理由も返します。policy判断・risk制約・partial fill・carryを同じ値として扱わないための診断情報です。

Directional PPOでは学習とdevelopment評価が同じbase execution設定を共有し、Datasetにあるborrowも両方で課します。過去のPPO実験は当時の実装へ固定された証拠であり、この修正後の学習経済へ自動的に読み替えません。

## 6. net returnからrewardを作る

### Directional terminal settlement

通常stepのrewardは約定・コスト反映後の区間returnから計算します。Directional PPOでは評価時のterminal FLATと経済endpointを合わせるため、最後のagent decision後にsettlement専用区間を予約します。agentが選んだactionをFLATへ上書きするのではなく、環境が外生的な`FLAT` proposalを同じhard riskと`MarketExecutor`へ1 barずつ流します。latency・capacity・partial fill・turnover制約も同じ状態機械で処理し、完全flatにならない残余はそのまま残します。

terminal transitionのrewardは、最後のagent intervalのlog returnにsettlement各intervalの実現log wealth changeを加算します。settlement barはagent stepではなくepisode末尾のterminal economic costとして1つのterminal transitionへ畳み込むため、settlement内部へPPOのdiscount factorを別途掛けません。これによりforced settlementをagent actionとして記録せず、episodeを終える時点の実行可能な経済結果だけをterminal rewardへ含めます。標準化を使う場合も、統計fit範囲はagentが実際に観測するdecision rowsまでです。なおtrainingはper-symbol account、developmentはshared-cashであるため、cross-symbol accountingまで一致したという意味ではありません。

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
- shared-cashでは全symbol actionを同じpre-execution snapshotから集め、risk/executionをportfolioごとに1回だけ実行する。
- shared-cashのteam rewardをper-symbol独立rewardと解釈しない。

## まだ保証していないこと

Observation contractがcausalであることと、PPOが儲かることは別です。shared-cash capabilityもsynthetic/canonical/real-SB3 integrationを検証する実装能力であり、実データでの優位性はまだ評価していません。現行研究はPPO superiorityやproduction profitabilityを証明していません。
