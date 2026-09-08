# Trade RL：Lean Redesign 2026-09-08

作成日: 2026-09-08 (JST)  
状態: 現行設計・実装契約  
基準: `main`

## 0. 結論

Trade RLは、旧U系・Causal Alpha世代・teacher → admission → BC → RLの多段構造を現行研究基盤として維持しない。

現在の目的は一つだけである。

> **銘柄IDに依存しない一つの共通戦略を学習し、各銘柄を独立にlong / shortし、point-in-time情報と同一の約定・会計条件の下で、コスト控除後の資産成長が未使用データでも再現するかを検証する。**

現mainでは、この目的に必要なlean coreとM2比較基盤まで実装済みである。

- causal / point-in-time `MarketDataset`
- 一つの`MarketExecutor` / `BookState`会計経路
- hard risk / execution feasibility
- `SHORT / FLAT / LONG`の小さなstrategy interface
- quantity-preserving hold
- trend / mean reversion / Ridge24 / LightGBM24 / teacher-free PPO
- cash / constant long / constant short controls
- 銘柄IDを特徴へ入れないuniversal supervised fit
- 銘柄ごとの学習寄与を均等化したRidge / LightGBM
- 銘柄をround-robinするuniversal PPO training
- 同じ凍結strategyを全銘柄へ独立Replayする評価
- filesystem dataset artifactから比較を実行し、immutable result artifactを保存するrunner

**ただし、収益性はまだ証明されていない。** ここからの主作業は、新しいarchitectureをさらに増やすことではなく、実データdevelopment runを実行し、5候補+3 controlsの証拠を比較することである。

---

## 1. 現行の最小構造

```text
trade_rl/
  data/          point-in-time market dataset / artifact
  simulation/    execution + accounting ledger
  risk/          hard safety / feasibility
  strategies/    rule / forecast / teacher-free PPO
  evaluation/    replay / metrics / comparison / candidate run
```

研究DB、UI、teacher generations、世代別runner、複数RL algorithm群を必須経路に戻さない。

Git履歴は旧コードの保存場所として扱い、現mainを旧architectureの博物館にしない。

---

## 2. Strategy contract

各strategyはdecision時点で観測可能なstateだけを受け取り、次のlogical intentを返す。

```text
SHORT
FLAT
LONG
```

### 保有継続

既にLONGで次もLONG、または既にSHORTで次もSHORTなら、価格変動によるweight driftを毎decisionで元weightへ戻さない。

標準挙動は**quantity hold**である。

intentが変化した場合だけ新しいtarget quantityを決め、その後はhard riskが必要な場合だけde-riskする。

これにより、同じ売買判断なのに毎bar不要なrebalance costを発生させる構造を避ける。

---

## 3. StrategyとRiskの責任分離

### Strategy

- entry
- hold
- exit
- reversal
- signal / forecastに基づく経済判断

### Risk / Execution

- max gross exposure
- max absolute weight
- max leverage
- margin / insolvency
- drawdown emergency
- liquidity / participation capacity
- minimum notional
- tick / lot
- inactive / untradable
- invalid stateのfail closed

同じentry判断をstrategyとriskの双方へ重複実装しない。

---

## 4. 初回比較対象は5候補 + 3 controlsに固定

| 名前 | 系統 | 内容 |
|---|---|---|
| `trend` | rule | 単一signalのtrend + hysteresis |
| `mean_reversion` | rule | 単一signalのmean reversion + hysteresis |
| `ridge24` | forecast | universal Ridge 24h forecast + 共通controller |
| `lightgbm24` | forecast | universal shallow LightGBM 24h forecast + 共通controller |
| `ppo` | RL | universal teacher-free PPO、64×64 actor / critic |
| `cash` | control | 常時FLAT |
| `constant_long` | control | 常時LONG |
| `constant_short` | control | 常時SHORT |

初回からTransformer、SAC、TD3、TQC、複数horizon ensemble、大規模hyperparameter gridを追加しない。

### 採用原則

- ruleが同等以上ならruleを優先
- forecastが明確に優位でPPOが上乗せしないならforecastを優先
- PPOは未使用データでも安定して上乗せするときだけ残す
- 差が不明ならより単純な候補を残す
- 全候補が弱ければ**no winner**を正しい結論とする

---

## 5. Universal model contract

最終目的は「銘柄ごとに別モデルを作る」ことではない。

### 共通ルール

- symbol IDをmodel featureへ入れない
- symbol-specific coefficient / embeddingを初期modelへ入れない
- 同じfeature schemaを使う
- 同じfit cutoffを使う
- 同じ凍結model / policyを全銘柄へ適用する
- 評価時は各銘柄を独立portfolioとしてReplayする

### Ridge / LightGBM

supervised training rowは事前登録した`fit_symbol_names`の銘柄だけからpoolする。fit scope外の銘柄は学習行へ入れず、評価対象symbolはfit対象と独立に扱う。

各銘柄のeligible row数が異なっても、一つの銘柄が行数だけで学習を支配しないよう、**各銘柄の総sample weightを等しくする**。

Ridgeではweighted mean / variance / intercept / normal equationを使用する。LightGBMでは同じweightを`sample_weight`としてfitへ渡す。

### PPO

PPOの観測は次だけである。

```text
selected features
feature availability
current intent
current weight
```

symbol IDは入れない。

training episodeは`fit_symbol_names`で指定した銘柄だけをround-robinで巡回し、各episodeでは一つのactive symbolだけを売買する。fit scope外の銘柄は学習episodeへ入れず、同じ凍結policyを評価時に全対象銘柄へ適用する。

---

## 6. Causality / Data contract

必須条件:

- `feature_available_time <= decision_time`
- supervised labelは`label_end_time < fit_cutoff`
- scaler / normalization / imputation statisticsをfutureから計算しない
- development / final futureをfeature選択やthreshold調整へ戻さない
- 15分barの行数を独立標本数と解釈しない
- 同calendar shockを受けた複数銘柄を統計上の完全独立標本として扱わない

### 初期feature family

- multi-horizon returns
- realized volatility / range / candle geometry
- trend / moving-average distance
- volume activity
- point-in-timeで検証可能なfunding / basis / mark-index情報

モデルサイズより先に、情報の因果性と有効性を疑う。

---

## 7. Execution / Accounting contract

P&Lの正本は`MarketExecutor` + `BookState`の一経路だけにする。

必須:

- submissionとfillを分離
- feeはfill時に一度だけ計上
- spread / impactの二重控除を禁止
- partial fill後はrealized fill quantityでposition更新
- fundingは対象時刻・符号・quantityに対して一度だけ計上
- borrow / mark-to-market / liquidationを別channelで追跡
- terminal mark-to-marketとforced closeを混同しない

candidate runnerの初期execution overlayはzeroである。ただしこれは「取引コストゼロ」を意味しない。dataset内のpoint-in-time `fee_rate`、venue fee、spread等のexecution fieldは`MarketExecutor`でそのまま使用される。

OHLCVからqueue positionやhidden liquidityを復元したとは主張しない。

---

## 8. 銘柄別評価を正本にする

同じstrategy objectを各銘柄へ独立Replayする。

```text
universal frozen strategy
  ├─ BTCUSDT independent replay
  ├─ ETHUSDT independent replay
  ├─ ...
  └─ each symbol retained separately
```

**aggregate P&Lを正本にしない。**

BTCの利益でETHの損失を相殺し、「全銘柄で成功した」ように見せることを禁止する。

`UniversalStrategyComparison.by_symbol`が主要な比較結果である。

各銘柄×strategyについて最低限以下を保持する。

- total return
- Sharpe
- Sortino
- maximum drawdown
- turnover
- total execution cost
- funding P&L
- borrow cost
- trade count
- rebalance count
- termination
- raw interval returns

---

## 9. M1 / M2 / M3の現在地

### M1: Lean core — 完了

実装済み:

- filesystem `MarketDataset` artifact
- canonical execution / accounting
- hard-risk projection
- quantity-preserving replay
- single-symbol / selected-symbol independent replay
- DB / UI / teacher pipelineを必要としないcore CI

M1の完了条件:

> DB・UI・teacher pipelineなしで、同一ledgerによる正しい独立symbol replayを実行できること。

現mainはこの条件を満たす。

### M2: Common comparison — 基盤完了、実データ比較未実施

実装済み:

- 5候補 + 3 controls
- universal Ridge / LightGBM / PPO fit
- symbol-balanced supervised training
- symbol-ID-free PPO
- 全銘柄独立comparison
- filesystem candidate runner
- immutable result artifact

未完了:

1. 実データdevelopment artifactを選ぶ
2. feature名・fit cutoff・development windowをfreezeする
3. 8候補を実行する
4. 全銘柄結果を完全報告する
5. winnerをfreezeする、またはno-winnerと判断する

### M3: Finalize and delete — 未着手

M2で候補が決まった後だけ進む。

1. frozen candidateを未使用future / zero-shotで評価
2. pre-registered stressを実行
3. 支持されなかったstrategy familyを削除
4. README / config / CI / testsを採用構成へさらに縮約
5. Production認可は研究結果とは別に扱う

---

## 10. 実データdevelopment runの実行方法

必要なのは次の3つだけである。

1. canonical filesystem dataset artifact
2. 一つのJSON config
3. 新規のoutput directory名

### JSON config

```json
{
  "signal_name": "<rule signal feature name>",
  "feature_names": ["<feature A>", "<feature B>"],
  "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
  "fit_cutoff": "2026-01-01T00:00:00",
  "evaluation_start": "2026-01-01T00:00:00",
  "evaluation_stop_exclusive": "2026-02-01T00:00:00",
  "rule_entry_threshold": 0.10,
  "rule_exit_threshold": 0.02,
  "forecast_entry_threshold": 0.01,
  "forecast_exit_threshold": 0.002,
  "ppo_total_timesteps": 100000,
  "ppo_seed": 0,
  "gross_budget": 0.5,
  "initial_capital": 100000.0
}
```

時刻はdataset timestampへ**exact match**する必要がある。

### 実行

```bash
uv run --extra forecast-gbm --extra train-sb3 \
  python -m trade_rl.evaluation.candidate_run \
  --dataset <dataset-artifact-dir> \
  --config <run-config.json> \
  --output <new-result-dir>
```

既存output directoryへの上書きは拒否する。

### 出力

```text
<new-result-dir>/
  summary.json
  returns.npz
```

`summary.json`:

- dataset identity
- symbols
- fit / evaluation scope
- 各symbol × 各strategyのmetrics
- execution diagnostics
- final portfolio value
- fill count
- raw return arrayへのkey

`returns.npz`:

- 各symbol × 各strategyのinterval return series
- paired comparison / block bootstrap等の後段解析に再利用可能

このartifactをdevelopment evidenceの正本とする。

---

## 11. Development後の判断手順

結果を見たら、architectureを増やす前に次の順で判断する。

1. controlsより本当に上か
2. 全銘柄で符号・drawdown・costが許容可能か
3. 特定一銘柄だけが利益を作っていないか
4. turnover / costでgross edgeが消えていないか
5. ruleとforecastの差は十分大きいか
6. PPOが単純候補へ本当に上乗せしているか
7. raw returnsで時間block依存・regime依存が強すぎないか
8. 差が弱ければ単純側を残す
9. 全て弱ければno-winner

no-winnerの場合に最初に増やす候補はmodel sizeではない。

優先順位:

1. 情報源
2. feature causality / quality
3. prediction horizon
4. market regime仮説
5. 対象期間
6. その後にmodel complexity

---

## 12. Final evaluation / Stress contract

Developmentで何度も見た期間はfinal testではない。

候補、feature、threshold、seed policy、cost/riskをfreezeした後、未使用futureを一度だけ開く。

採用候補には事前固定したstressをかける。

- fee adverse
- spread adverse
- impact adverse
- +1 decision latency
- capacity reduction
- initial capital sensitivity
- funding / borrow coverage確認

stress結果を見てから合格thresholdを変更しない。

---

## 13. 削除ルール

Git履歴が残るため、現mainへ旧世代を残すことを「保存」と定義しない。

削除前に守るもの:

- source data / non-regenerable raw artifact
- 未記録の実験結果
- exact reproductionに必要なcommit / lock / config
- licensing notice

候補選定後は、支持されなかったstrategy family、その専用test、不要extra、dead adapterを一緒に削除する。

削除率やファイル数をKPIにしない。

KPIは、**同じ研究目的をより少ない責務・依存・分岐で、因果的かつ再現可能に検証できること**である。

---

## 14. 現時点の判断

2026-09-07の旧実データrunがteacher selectionでrejectされた事実は、旧teacher必須経路を捨てる理由にはなる。しかし、それ自体はtrend、mean reversion、Ridge、LightGBM、teacher-free PPOのどれが利益を出すかを示さない。

現在はarchitecture再設計の主要部分が実装された段階であり、次の本質的な問いはコードではなくデータで答える。

> **同じuniversal model / policyを全銘柄へ独立適用したとき、どの候補がcontrolsを超え、コスト控除後でも未使用データへ持続するのか。**

次に実施すべき作業は、V14/V15のような新世代追加ではない。

> **canonical実データartifactを一つ選び、development windowをfreezeし、5候補+3 controlsを一回通して完全比較する。**

そこでwinnerがなければ、no-winnerを結論として受け入れ、次の情報・feature・horizon仮説へ進む。
