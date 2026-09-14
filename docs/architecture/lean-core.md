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

- `feature_available_time <= decision_time` を満たす。featureのinformation/availability timeはdecision timeを越えない。
- supervised labelはfit cutoffの内側だけで完結する。
- scaler、normalization、imputation、feature selectionにfuture情報を混ぜない。
- developmentやfinal futureをthreshold/model選択へ戻さない。
- 15分barの行数を独立標本数と解釈しない。
- 同calendar shockを共有する複数銘柄を完全独立標本と仮定しない。

Dataset identityは内容にbindされ、canonical artifactはdeterministicでなければならない。publication先が既に存在する場合は上書きせずfailする。

Canonical Datasetのidentity-bound feature numericsは `trade_rl.data.features.numerics` を単一authorityとし、scalar `math.log` と固定順序の `math.fsum` を基礎にmean / variance / standard deviation / dot / covariance / correlationを定義する。identityを一致させるためのrounding、quantization、tolerance-based hash canonicalizationは行わない。現行buildは `market_build_v3` と `portable_feature_numerics_v1` をbuild identityへ明示bindし、保存feature dtypeは従来どおり`float32`とする。

このportable contractは、同一code・config・sealed sourceから構築した完全Datasetについて、現行のUbuntu x86_64 hosted runner上の複数AMD EPYC系と複数Intel Xeon系で `features`、`global_features`、normalization digest、Dataset IDのbit-exact一致を実証済みである。一方、任意のARM、任意libm、任意platformまでの普遍的なbit-identical保証は主張しない。historical `market_build_v2` artifactは書き換えず、current readerでそのidentityのまま読み取れる互換を維持する。

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

### PPO Observation v2

初回canonical real-data M2で使うteacher-free PPOのpolicy observationは、fit-scope leakageを避けるため最小のcausal contractへ固定する。tensor順序は次である。

1. selected local feature values
2. selected local availability / finite mask
3. selected normalized local feature staleness
4. current intent
5. current weight

unavailableまたはnon-finiteなlocal valueは0へmaskする。stalenessはselected featureと同じ順序で`[0, 1]`の正規化値を渡し、unavailable featureは最大stalenessでなければならない。PPO v2はstalenessが欠けたobservationをfail-closedにする。一方、`StrategyObservation`はPPO専用型ではないため、既存のrule/forecast/control caller向けconstructor互換を維持し、generic contractではstaleness省略を許す。canonical replayとPPO environmentは実datasetのstalenessを明示的に渡す。

初回M2のpolicy tensorにはsymbol IDを入れず、dataset-global featureも入れない。現行`MarketDatasetBuilder`の`active_fraction`、`tradable_fraction`、`market_return_mean`、`market_return_dispersion`はdataset全symbol universeから集計されるため、fit symbol subsetだけでPPOを学習するStudyで使うとfit-scope外symbolがtraining observationへ間接混入し得る。このためObservation v2のglobal rosterは明示的な空集合である。global regimeを将来試す場合は、fit-scope外evaluation symbolを含まないreference universeを事前固定した別Controlled Factorとして扱う。

Observation contractは暗黙のimplementation detailにしない。新規Candidate Runと新規Studyのresolved configはObservation schema、staleness利用、空のglobal rosterをsemantic identityへbindする。execution economics、reward、action、risk、PPO network architectureはObservation v2のpolicy inputへ追加しない。

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

`MarketExecutor` の標準executionはnext-openであり、decision row `t` の注文は最初にrow `t+1` のopenで約定可能になる。participation capacityのvolume authorityは `ExecutionCostConfig.processing_bar_volume_capacity` に明示bindする。

- `True` は既存互換のlegacy modeであり、processing bar全体のvolumeをcapacity poolへ使う。同一barのopen時点ではbar最終volumeは未確定なので、これはpoint-in-time liquidity forecastではない。既存canonical Dataset / Study / Runの意味を変えないためdefaultとして保持する。
- `False` はcausal stress modeであり、processing barの直前に完全終了したbarのvolumeをcapacity poolへ使う。base-volumeをmarket notionalへ換算するときも、その前barのcloseをreference priceに使う。current processing barの最終volumeはcapacityへ使わない。
- どちらのmodeも同じspread / impact / fee / accounting経路を使い、mode差だけで別のexecution-policy digestになる。`False` は将来volumeの予測モデルではなく、同一bar最終volumeへの依存を除くための保守的stressである。

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
- Candidate Runはresolved result `summary.json`、raw interval return `returns.npz`、implementation/runtime/research-context evidence `provenance.json` の3ファイルを一体としてpublishする。
- Candidate Runは実行前後でimplementation/runtime provenanceが一致する場合だけpublishする。実行中にsource/runtime provenanceが変化したRunを正当なevidenceとして残さない。
- Candidate artifact identityはNPZのZIP圧縮表現そのものではなく、summary/provenanceと検証済みreturn arrayのsemantic contentへbindする。一方、各fileのraw SHA-256/sizeもtamper検出用evidenceとして保持できる。
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
