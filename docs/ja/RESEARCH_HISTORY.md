# 研究履歴

[English](../RESEARCH_HISTORY.md) | **日本語**

この文書は、過去の実験の役割を記録するものです。現行システム要件や収益性の主張を定義する正典ではありません。

## 対象範囲

このrepositoryでは、次を検討してきました。

- synthetic positive／negative control
- feature predictive powerとleakage診断
- forecast horizonとtarget定義
- PPO hyperparameterとbehavioral-cloning warm start
- multi-timeframe feature encoder
- post-processing、turnover penalty、no-trade band、volatility targeting
- seed ensembleとdisagreement scaling
- rule-basedおよびRL risk overlay
- Walk-Forward、cost sensitivity、bootstrap比較、replay simulation
- 履歴制約の異なる複数の公開exchange data source

## 解釈ルール

Data、symbol、date range、target、horizon、decision frequency、fee、execution model、random seed、code revision、evaluation protocolが同一でない限り、過去結果を直接比較してはいけません。

Synthetic returnとSharpe比は、softwareまたはhypothesis診断です。実現可能なlive performanceの推定値ではありません。

Single split、single seed、繰り返し確認したholdout結果はexploratoryです。昇格証拠には、現行のgated Control Planeと正確なServingBundle identityが必要です。

## 重要な過去の教訓

1. 適切なwarmup除外、Walk-Forward評価、繰り返しhorizon／target探索の補正後に、見かけの予測力が消える場合がある。
2. 高いin-sampleまたはdiagnostic ICだけでは、cost控除後のtrading valueを証明できない。
3. 固定baselineは1つのsplitで強く見えてもfold全体で失敗する場合がある。RLと同じprotocolで評価しなければならない。
4. Turnover penalty、smoothing、no-trade bandは、学習action scaleとexecution behaviorの両方を変える。それぞれの効果を分離し、1回のrunから原因を断定してはいけない。
5. Train／serve parityには、同一model fileだけでなく、完全なobservation、preprocessing、portfolio state、decision pathが必要である。
6. 公開OHLCVのみのdataには、安定したedgeが不足している可能性がある。定義済みwithdrawal criterionを超えてparameter searchを続けるとselection biasが増える。

## 詳細数値の扱い

過去のMarkdownには、耐久性のあるmachine-readable provenance standardを伴わない、実験固有の数値が多数含まれていました。それらのfileは正典ドキュメントから削除されています。Forensic referenceとしてGit履歴に残っています。

新しい研究証拠は、次を含む不変Artifactとして保存してください。

- commandと完全なconfiguration
- datasetとtime range identity
- symbolとsample count
- 適用可能な場合のGit SHAとbundle digest
- foldとseedの詳細
- raw metricとconfidence interval
- limitation、およびexploratory resultかpromotion evidenceかの区分

## Production境界

この文書内の記録だけでProductionをNO-GOからGOへ変更することはできません。その判断を許可できるのは、[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md)の証拠checklistと、承認済みdeployment processだけです。