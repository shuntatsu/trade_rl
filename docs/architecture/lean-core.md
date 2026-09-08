# Lean core architecture

## 結論

Trade RLの現行coreは、**causalなmarket data、1つのexecution/accounting ledger、hard risk、small strategy interface、銘柄ごとの独立評価**に絞る。

旧U-series、旧Causal Alpha世代、mandatoryな `teacher -> admission -> BC -> RL`、研究DB、UI、世代別runnerは現行研究経路の必須条件ではない。Git historyに残る旧architectureをcurrent treeへ戻さない。

現在の研究目的は、銘柄IDに依存しない共通strategy/model/policyを複数銘柄へ適用し、point-in-time情報と同一の約定・会計条件で、コスト控除後の結果がunused dataでも維持されるかを検証することである。

## Core flow

```text
causal raw / venue evidence
        ↓
MarketDataset + immutable dataset artifact
        ↓
strategy logical intent (SHORT / FLAT / LONG)
        ↓
hard risk / feasibility
        ↓
MarketExecutor + BookState
        ↓
independent per-symbol replay
        ↓
metrics / comparison / robustness / immutable run artifact
```

## Data contract

`MarketDataset` はpoint-in-timeで観測可能な情報だけをdecisionへ渡す。

必須条件:

- featureのinformation/availability timeはdecision timeを越えない。
- supervised labelはfit cutoffの内側だけで完結する。
- scaler、normalization、imputation、feature selectionにfuture情報を混ぜない。
- developmentやfinal futureをthreshold/model選択へ戻さない。
- 15分barの行数を独立標本数と解釈しない。
- 同calendar shockを共有する複数銘柄を完全独立標本と仮定しない。

Dataset identityは内容にbindされ、canonical artifactはdeterministicでなければならない。publication先が既に存在する場合は上書きせずfailする。

## Strategy contract

Strategyが返すlogical intentは小さく保つ。

```text
SHORT
FLAT
LONG
```

Strategyの責任:

- entry
- hold
- exit
- reversal
- signal / forecast / policyに基づく経済判断

同じLONG→LONGまたはSHORT→SHORTなら、価格変動でweightがdriftしただけを理由に毎decisionでtarget weightへ戻さない。標準は**quantity-preserving hold**であり、intentが変わった場合かhard riskがde-riskを要求する場合にquantityを変える。

## StrategyとRiskの責任分離

Risk / executionが担当するもの:

- maximum gross exposure
- maximum absolute weight
- leverage / margin / insolvency
- drawdown emergency
- liquidity / participation capacity
- minimum notional
- tick / lot constraints
- inactive / untradable state
- malformed/invalid stateのfail closed

同じentry/exit判断をstrategyとriskへ二重実装しない。

## Execution / accounting authority

P&Lの正本は `MarketExecutor + BookState` の一経路である。

不変条件:

- order submissionとfillを区別する。
- feeはrealized fillに対して一度だけ計上する。
- spread / impactを複数channelで二重控除しない。
- partial fill後のpositionはrealized fill quantityで更新する。
- fundingは対象時刻・符号・quantityに対して一度だけ計上する。
- borrow、mark-to-market、liquidationを別channelで追跡する。
- terminal mark-to-marketとforced closeを混同しない。
- OHLCVだけからqueue positionやhidden liquidityを再現したとは主張しない。

Candidate runのexecution overlayがzeroでも、datasetに含まれるpoint-in-time fee/spread等までzeroになるわけではない。既存execution fieldはcanonical executorを通る。

## Independent per-symbol evaluation

同じfrozen strategy objectを各銘柄へ独立Replayする。

```text
universal frozen strategy
  ├─ symbol A independent replay
  ├─ symbol B independent replay
  └─ ...
```

Aggregate P&Lだけを成功判定の正本にしない。ある銘柄の利益で別銘柄の損失を隠さず、各symbol × strategyについてreturns、drawdown、turnover、execution cost、funding、borrow、trade/fill/rebalance diagnostics、terminationを保持する。

## Artifact and evidence rules

- Canonical JSON/digestのauthorityは `trade_rl.artifacts` に置く。
- Market dataset artifactのcodec/publicationは `trade_rl.data.artifacts` が持つ。
- Candidate/evaluation runはimmutable filesystem artifactとして残す。
- 同じ入力・設定・identityからはdeterministicなidentityを得る。
- resultを見た後にevidence条件やthresholdを都合よく変更しない。

## 非目標

Lean coreが保証しないもの:

- profitability
- Production/live order authorization
- OHLCVから観測不能なmicrostructureの完全再現
- model complexityがedgeを生むという前提
- DB/UI/teacher pipelineが研究成立に必須であること

利益やlive suitabilityはarchitectureではなく、凍結した研究条件とunused-data evidenceで別途判断する。
