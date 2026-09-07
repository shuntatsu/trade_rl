# Trade RL：Lean Redesign 2026-09-08

作成日: 2026-09-08 (JST)
状態: 現行の再設計方針。旧世代の研究構造を継承するための文書ではない。
基準: `main`。過去の実験結果そのものはGit履歴・report artifactの事実として保持するが、旧設計を現行契約として維持しない。

## 0. 結論

Trade RLは、旧U系・Causal Alpha世代・teacher→admission→BC→RLの多段研究構造を積み増す方針をやめる。

現行の目的は一つだけとする。

> 同じ市場情報と同じ実行条件の下で、各銘柄を独立にlong/shortし、現実的なコスト控除後の資産成長を、未使用データで再現可能に確認する。

この目的に直接必要でない機能は削除候補とする。残す核は次の5点だけである。

1. causal / point-in-time data
2. 一つの約定・会計ledger
3. hard safety / execution feasibility
4. 差し替え可能な小さなstrategy interface
5. walk-forward / unseen / stressを含む共通評価

戦略は最初から一つの学習方式へ固定しない。`rule`、`forecast + controller`、`teacher-free PPO`の3系統を同じ本体で比較し、支持されなかった系統は現行コードから削除する。

収益性、研究上の支持、Production認可は別である。ソフトウェアが動くこと、backtestが正になること、将来利益が保証されることを混同しない。

---

## 1. 何を捨てるか

旧構造は互換性維持を目的に残さない。削除はGit履歴からの復元可能性を確認したうえで行う。

### 削除・切離し対象

- `causal_alpha_v*`のような世代別アルゴリズム本体、世代別workflow、runner、configの積み増し
- teacher selection → admission → BC → critic warm-start → PPOを必須にする直列経路
- BC / DAgger / critic warm-startを「まず必要」とする契約
- PPO、Lagrangian、discounted、CostCritic、SAC、TD3、TQC等を同時に維持する初期研究面
- 大型sequence encoder、hierarchical gate、asset/timeframe attention等の未証明の複雑性
- 研究UIを学習・評価の必須依存にする構造
- PostgreSQL catalogを研究実行の必須依存にする構造
- offline研究に不要なserving、署名、release、live-order関連依存
- 古い世代だけを守るREADME、docs、Actions、coverage requirement、compatibility adapter

### 残すもの

- source data identity / timestamp / availability
- deterministic feature generation
- fillベースのfee、spread、impact、funding、borrow、mark-to-market
- partial fill、latency、capacity、minimum notional、tick、lot
- max exposure、leverage、margin、drawdown stop等のhard safety
- immutable run configと結果
- train / development / final evaluationの分離

削除した機能のためだけのtestは一緒に削除してよい。一方、因果性、会計、hard-risk、未来情報漏洩を検出するtestは残す。

---

## 2. 新しい最小アーキテクチャ

```text
trade_rl/
  data/          point-in-time market data / features
  simulation/    one execution + accounting ledger
  risk/          hard safety + feasibility only
  strategies/    rule / forecast / ppo adapters
  evaluation/    walk-forward / metrics / stress
  cli/           run / evaluate
```

新しい汎用plugin framework、研究DB、UI schedulerは作らない。

各strategyは同じinterfaceで、decision時点で観測可能なstateだけを受け取り、logical position intentを返す。

初回actionは次の3値を基本とする。

```text
SHORT
FLAT
LONG
```

既にLONGで次もLONGなら、原則として新規注文ではなく保有継続である。価格変動によるweight driftを毎decisionで元weightへ戻すことを標準挙動にしない。

注文intent、projected position、realized positionは分ける。未約定注文を約定済みpositionとしてstate更新しない。

---

## 3. StrategyとRiskの責任を分ける

旧構造ではstrategy/controllerと`PreTradeRisk`の双方にentry/exit/no-trade判断が入り得た。新構造では、経済的な売買判断をstrategy側へ一元化する。

### Strategy側

- entry / hold / exit / reversal
- signal confidence
- turnoverを払って変更する価値
- 通常のdecision cadence

### Risk / Execution側

- max exposure / max leverage
- margin / insolvency
- drawdown emergency
- liquidity / participation capacity
- tick / lot / minimum notional
- inactive / untradable状態
- invalid / non-finite stateのfail closed

取引所minimum notionalは削除しない。削除するのは、同じentry判断を複数層で重ねる不要なsoft ruleである。

---

## 4. 比較する戦略は最初の5候補だけ

同じtrain scope、features、cost、risk、reset、evaluationで以下を比較する。

| ID | 系統 | 初回内容 |
|---|---|---|
| T | rule | 共通パラメータのtrend |
| M | rule | 共通パラメータのmean reversion |
| R24 | forecast | 小さなRidgeによる24h予測 + 単一controller |
| G24 | forecast | 浅いLightGBMによる24h予測 + 同一controller |
| P | RL | teacher-freeの小型PPO |

対照として`cash`、`constant long`、`constant short`を必ず出す。

初回から24h/72h混合、複数LightGBM grid、Transformer、SAC/TQC等を並べない。多方面を検討することと、候補数を無制限に増やすことは分ける。

### 選択原則

- ruleが同等以上ならruleを優先
- forecastが優位でPPOが上乗せしないならforecastだけ残す
- teacher-free PPOが未使用データでも安定して優位ならPPOを残す
- 差が不明ならより単純なものを残す
- 全候補が弱い場合、無理にwinnerを作らない

---

## 5. Data設計

15分barの行数を独立経験数と見なさない。24h予測では隣接labelが大きく重複し、複数銘柄も同じmarket shockを共有する。

### 必須ルール

- `feature_available_time <= decision_time`
- training rowは`label_end_time < fit_cutoff`
- normalizer / scaler / imputation statisticsもfit prefixだけで計算
- future / admission / final testをfeature選択、threshold選択、early stoppingへ戻さない
- 同じcalendar blockの複数銘柄を独立標本として水増ししない

### 初回feature family

特徴量は意味のある少数familyへ縮約する。

- multi-horizon returns
- realized volatility / range / candle geometry
- trend distance / moving-average distance
- volume activity
- point-in-timeで検証できるfunding / basis / mark-index information

目安は20〜40列程度だが、列数自体を目標にしない。大量の相関指標を重複投入しない。

全候補が弱い場合、次に増やすのはmodel sizeではなく、情報源、期間、horizon、market-regime仮説を優先する。

---

## 6. Execution / Accountingは一つにする

P&Lの正本は一つのledgerだけにする。

必須照合:

- order submissionとfillを区別
- feeはfillで一度だけ計上
- spread / impactをfill priceへ反映した場合の二重控除を禁止
- partial fill後のpositionはfill quantityで更新
- fundingは対象時刻・符号・position quantityに対して一度だけ計上
- borrow、mark-to-market、liquidationを別channelで照合
- episode終端のmark-to-marketとforced closeを混同しない

OHLCVだけからqueue positionやhidden liquidityを復元したとは主張しない。初回はmarket orderを基本とし、根拠のないmaker optimizationを追加しない。

### golden fixtures

本体置換前に最低限次を手計算で照合する。

1. cash
2. constant long
3. constant short
4. one entry + one exit
5. partial fill
6. no fill
7. funding payment
8. funding receipt
9. drawdown emergency reduction
10. terminal mark-to-market

この10件が正しくない状態でstrategy比較へ進まない。

---

## 7. PPOは小さく、独立に評価する

teacherが黒字であることをteacher-free PPOの研究開始条件にしない。

初回は標準PPO + 小型MLPを使う。目安としてactor / critic各64x64程度から始める。巨大encoder、独自gate、DAgger、critic warm-startは入れない。

rewardは原則として実コスト控除後のinterval net log growthとする。

```text
reward_t = scale * log(E_after / E_before)
```

flat罰、trade bonus、既にledgerへ入ったfeeの二重penaltyを追加しない。

ただし、短期rolloutと長期保有利益の整合はsynthetic taskで先に検査する。即時costがあり、数時間〜数日holdすれば利益になるtoy marketを標準PPOが学習できない場合、実市場での敗因と混同せず、rollout、GAE、episode boundary、state/action semanticsを調べる。

---

## 8. 評価契約

評価は三段階に分ける。

### Development

仮説・実装の比較。ここで何度も見た期間は未使用testではない。

### Frozen final evaluation

候補、features、hyperparameters、seed policy、cost/riskをfreezeした後に一度だけ開く。

### Stress

採用候補に対し、事前固定した条件で以下を確認する。

- fee / spread / impact adverse
- +1 decision latency
- capacity reduction
- initial capital sensitivity
- funding / borrow coverageが必要な期間

stress結果を見てから合格thresholdを変更しない。

### 指標

最低限、次を同じscopeで報告する。

- net log growth
- gross P&L
- execution cost decomposition
- maximum drawdown
- realized exposure
- turnover
- trade/fill count
- worst time block
- symbol別結果
- seed dispersion
- runtime / memory

`positive episode fraction >= 0.5`のように資産成長から直接導けない条件は、原則diagnosticへ移す。hard risk、損失上限、turnover budget等、実際の制約として根拠を説明できるものだけをgateにする。

全銘柄での成立という最終目標は残す。ただし、開発途中の診断で一部銘柄の不振原因を調査することまで禁止しない。結果後に銘柄を削除して「全銘柄成功」とは呼ばない。

---

## 9. 実装順序

### M1: Lean core

1. 現mainと必要artifactの復元点を固定
2. data / execution / accounting / hard-riskの必要部分を抽出
3. golden fixturesを通す
4. DB / UI / teacher generations / old runnersへの依存を外す
5. 単一symbol replayをfilesystem artifactだけで実行可能にする

M1の完了条件は「旧世代が全部再実行できる」ではない。

> DB・UI・teacher pipelineなしで、正しい単一symbol replayとledgerを生成できること。

### M2: Common comparison

1. 5候補 + 3 controlsを同一条件で実行
2. development dataで全候補を完全報告
3. 不要な複雑性を追加せず、原因別に一変更だけ試す
4. 候補を一つにfreeze、またはno-winnerで終了

### M3: Finalize and delete

1. frozen candidateを未使用future / zero-shotで評価
2. stressを実行
3. 支持されなかったstrategy familyを削除
4. 現行README / config / CI / testsを採用構成だけに整理
5. Productionは別認可として維持

---

## 10. 削除の運用ルール

Git履歴が保存されているため、現mainに旧世代を残すことを「保存」と定義しない。

ただし次は削除前に退避・確認する。

- source data / non-regenerable raw artifacts
- 未記録の実験結果
- 正確な再現に必要なlock / config / commit
- 稼働中runの結果
- licensing notice等の法的に必要な情報

世代別コードを削除するときは、import graph、CLI entry point、config reader、CI、docs、testを同時に整理し、dead adapterだけを残さない。

削除率やファイル数をKPIにしない。KPIは、同じ研究目的をより少ない責務・依存・分岐で正しく実行できることである。

---

## 11. 現時点の判断

2026-09-07の実データrunはteacher selectionでrejectされ、BC/RLは実行されていない。この事実は旧teacher経路を見直す十分な理由だが、teacher-free PPO、単純rule、別forecast familyの収益性を確定する証拠ではない。

よって次の実装は、旧teacherを合格させるためのV14/V15追加ではない。

> 一つのlean replay/accounting coreを作り、rule / forecast / teacher-free PPOを同じ条件で比較する。

この比較の後、最も単純で未使用データ上の根拠が強い構成だけを残す。全候補が不合格なら、不合格を正しく結論とし、データ・情報・horizonの次仮説へ進む。

以上を、Trade RLの現行再設計方針とする。
