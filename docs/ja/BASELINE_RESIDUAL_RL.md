# Baseline-Anchored Residual RL

[English](../BASELINE_RESIDUAL_RL.md) | **日本語**

この文書は、baseline-anchored residual policyとして実装された研究経路を説明します。正規のシステムアーキテクチャは[`../ARCHITECTURE.md`](../ARCHITECTURE.md)です。

## 目的

従来のdirect-weight PPOは、方向、銘柄配分、回転抑制、risk挙動を同時に発見しなければなりません。弱いシグナルと現実的なcostの下では、安全解が完全flatになり得ます。Residual経路ではidentity actionをflatではなく実行可能なtrend baselineへ変更します。

```text
identity action [0, 0]
  -> base_trend_v2
  -> HTF proposal constraint
  -> 共通post-processing
  -> execution
```

RL policyが変更できるのは次の2つだけです。

- `action[0]`: fast／base／slow trendの連続混合
- `action[1]`: grossの30%以内に制限されたresidual-alpha budget

## 主要契約

- `TrendFamily`は価格履歴とUTC bar timestampだけに依存し、現在の口座weightやslice相対indexには依存しません。
- environmentの1 actionは`decision_every`本分を1区間として進め、集約rewardを1つ返します。
- 解決済み`decision_every`はすべてのResidual environmentへ明示的に渡し、run設定にも実効値を書き戻します。0や不正値はfail-closedで拒否します。
- Hybrid bookとshadow bookは独立したportfolio stateを持ちますが、価格、funding、cost、HTF制約、post-processing、hard risk checkは同一です。
- Rewardはhybrid区間log returnからshadow base-trend区間log returnを引いた値です。
- Identity actionは、テスト済み数値誤差内でshadowのweight、cost、PnLと一致しなければなりません。
- residual-alpha artifactは、label確定済みdevelopment dataだけでfitします。HoldoutとServingでは凍結済みartifactをloadし、再fitしません。
- residual-alphaのdataset identityはmetadataだけでなく、順序付きschema、timestamp、学習特徴値、target生成に使う価格履歴をhash化します。
- HTFはstateful post-processingより前にdesired proposalへ作用します。neutral signalが制約済みpositionを繰り返し半減させることはありません。
- 完全oracleとnoisy oracleは診断専用で、必須release gateには入りません。
- Sharpe／Sortinoの年率換算は固定1h値ではなく、稼働中post-processorのbase timeframe係数を使用します。

## Development matrix

研究runnerは、最終test区間を見る前にdevelopment validation区間で構成を固定します。

```text
A: pure base_trend_v2 identity policy
B: PPO fast／base／slow trend mixing、alpha無効
C: 固定+15% residual-alpha budget、診断専用
D: PPO trend mixing + PPO alpha budget
```

選択規則:

- Bは、development excess returnが正で、drawdown許容幅を超えず、developmentの2倍costでも非負の場合だけ候補になります。
- Cはrelease policyとして選択しません。凍結alpha sleeve単体の価値を測る診断です。Dでalphaを有効化するには、C自身がAを上回り、developmentの2倍costにも耐える必要があります。
- DはB／C／Dの適格条件を満たし、2倍costに耐え、通常costでBとCの両方を厳密に上回る場合だけ選択します。
- それ以外は、trend mixingが適格ならB、適格でなければA（`baseline_only`）へfallbackします。

固定済みの選択構成だけを最終test区間で1回評価し、cost 1x／2xを報告します。

## Checkpoint selection

Residual PPOのaction headはzero initializeされるため、初期deterministic policyはidentity actionと厳密に一致します。Validationはrollout境界に揃え、連続blockのexcess log return中央値を使います。Checkpointが適格になる条件は次です。

- block excess中央値が正
- 正のblockが半数以上

適格checkpointがなければ、そのseedは任意の学習済みpolicyではなくidentity snapshotへ戻ります。

## Run tier

`timesteps`はbase bar数ではなくdecision transition数です。

- `smoke`: PPO update 5回以上
- `research`: update 50回以上、3 seed以上
- `release`: update 100回以上、5 seed以上

8 environment、`n_steps=256`では、最低timestepsはそれぞれ10,240、102,400、204,800です。

## 研究経路の実行

専用runner:

```bash
uv run python scripts/run_baseline_residual.py \
  --source postgres \
  --pg-source binance \
  --base-timeframe 1h \
  --decision-every 4 \
  --timesteps 102400 \
  --ensemble 3 \
  --signal-model gbm \
  --run-tier research \
  --output output/baseline_residual_research
```

一般Control Plane entrypointからも実行できますが、明示的に研究専用でなければなりません。

```bash
uv run python scripts/run_pipeline.py \
  --action-mode baseline-residual \
  --no-register \
  --source postgres \
  --timesteps 102400 \
  --ensemble 3 \
  --output output/baseline_residual_research
```

## Residual Walk-Forward

従来のdirect-weight Walk-Forwardは、引き続き`walk_forward_cost1x.json`と`walk_forward_cost2x.json`を出力します。これらはResidualアーキテクチャの評価結果ではありません。

Residual専用のnested Walk-Forwardは明示的に実行します。

```bash
uv run python scripts/run_pipeline.py \
  --action-mode baseline-residual \
  --phase wf \
  --no-register \
  --source postgres \
  --pg-source binance \
  --base-timeframe 1h \
  --decision-every 4 \
  --horizon 12 \
  --timesteps 102400 \
  --folds 3 \
  --ensemble 3 \
  --n-seeds 3 \
  --signal-model gbm \
  --run-tier research \
  --output output/realdata_residual_wf
```

正規の結果ファイルは`residual_walk_forward.json`です。各outer foldの内部でtrain／validationを分離し、outer OOS結果を見る前にA／B／Dを選択します。その後、同じ選択済みpolicyをcost 1x／2xで評価します。Reportにはhybridとshadow双方の取引回数を記録するため、residual actionがゼロなのか、portfolio全体が取引ゼロなのかを区別できます。foldごとのartifactは`residual_wf/fold_<k>/`へ保存し、Registryには登録しません。

## Artifact

単一分割の研究runnerは次を出力します。

- `residual_alpha.json` — 凍結residual-alpha modelと内容に結び付いたdata identity
- `B_trend_mix_model.zip`または`B_trend_mix_ensemble/`
- alpha gate合格時の`D_combined_model.zip`または`D_combined_ensemble/`
- `residual_train_report.json` — cost 1x／2xのA/B/C/D development結果、固定済み選択、最終relative評価、gate、診断
- `residual_model_manifest.json` — dataset、学習、選択policy mode、選択済みalpha有効化状態のidentity

Residual Walk-Forward runnerは次を出力します。

- `residual_walk_forward.json` — fold横断の構成選択、activity、return、cost、fallback、warning集計
- `residual_wf/fold_<k>/residual_alpha.json` — fold内trainだけでfitした凍結alpha artifact
- `residual_wf/fold_<k>/fold_report.json` — fold境界、development matrix、固定済み選択、cost 1x／2x OOS診断
- 学習されたfold-local B／D model artifact

## Serving schema

Residual ServingBundleは次を使用します。

- action schema: `baseline_residual_v1`
- observation schema version 2
- policy mode: `ppo_residual_ensemble`または`baseline_only`
- 絶対時刻TrendFamily設定
- 凍結`residual_alpha.json`
- 選択構成に結び付いた明示的な`residual_alpha_enabled`
- composerとHTF proposal constraint設定

`baseline_only` BundleはPPO modelを持たず、residual alphaを常に無効化します。B Bundleも、凍結artifactの研究gateが合格していてもresidual alphaを無効化し、選択されたD Bundleだけが有効化します。`baseline_only`ではServingがidentity actionを供給し、同じtrend、composer、HTF、post-processing、guardrail、risk経路を実行します。Schema不一致、timestamp欠落、alpha artifact欠落、symbol／feature順序不一致、model layout不整合、またはgate不合格artifactを有効化する宣言はfail-closedで拒否します。

## Release境界

Residualの単一分割研究経路とnested Walk-Forwardは実装されていますが、top-level登録経路は意図的に無効です。`--no-register`なしでResidual経路を実行すると、sealed release evidence、promotion policy、運用検証が実装・確認されるまでerrorになります。

これは意図的な安全境界です。Walk-Forward完了、CI合格、正しいBundle形状だけでは、収益性やProduction readinessは証明されません。Productionは引き続き**NO-GO**です。
