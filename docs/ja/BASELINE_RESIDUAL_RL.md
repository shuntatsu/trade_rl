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
- Hybrid bookとshadow bookは独立したportfolio stateを持ちますが、価格、funding、cost、HTF制約、post-processing、hard risk checkは同一です。
- Rewardはhybrid区間log returnからshadow base-trend区間log returnを引いた値です。
- Identity actionは、テスト済み数値誤差内でshadowのweight、cost、PnLと一致しなければなりません。
- residual-alpha artifactは、label確定済みdevelopment dataだけでfitします。HoldoutとServingでは凍結済みartifactをloadし、再fitしません。
- HTFはstateful post-processingより前にdesired proposalへ作用します。neutral signalが制約済みpositionを繰り返し半減させることはありません。
- 完全oracleとnoisy oracleは診断専用で、必須release gateには入りません。

## Development matrix

研究runnerは、最終test区間を見る前にdevelopment validation区間で構成を固定します。

```text
A: pure base_trend_v2 identity policy
B: PPO fast／base／slow trend mixing、alpha無効
C: 固定+15% residual-alpha budget、診断専用
D: PPO trend mixing + PPO alpha budget
```

選択規則:

- Bは、development excess returnが正で、drawdown許容幅を超えない場合だけ候補になります。
- Cはrelease policyとして選択しません。凍結alpha sleeve単体の価値を測る診断です。
- Dは適格で、かつBとCの両方を厳密に上回る場合だけ選択します。
- それ以外はA（`baseline_only`）へfallbackします。

選択済み構成だけを最終test区間で1回評価し、cost 1x／2xを報告します。

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

## Artifact

研究runnerは次を出力します。

- `residual_alpha.json` — 凍結residual-alpha modelとdata identity
- `B_trend_mix_model.zip`または`B_trend_mix_ensemble/`
- alpha gate合格時の`D_combined_model.zip`または`D_combined_ensemble/`
- `residual_train_report.json` — A/B/C/D development結果、固定済み選択、最終relative評価、cost 2x、gate、診断
- `residual_model_manifest.json` — datasetと学習identity

## Serving schema

Residual ServingBundleは次を使用します。

- action schema: `baseline_residual_v1`
- observation schema version 2
- policy mode: `ppo_residual_ensemble`または`baseline_only`
- 絶対時刻TrendFamily設定
- 凍結`residual_alpha.json`
- composerとHTF proposal constraint設定

`baseline_only` BundleはPPO modelを持ちません。Servingがidentity actionを供給し、同じtrend、alpha、composer、HTF、post-processing、guardrail、risk経路を実行します。Schema不一致、timestamp欠落、alpha artifact欠落、feature順序不一致、model layout不整合はfail-closedで拒否します。

## Release境界

Residual candidate構築とServing検証は実装されていますが、top-level登録経路は意図的に無効です。`--no-register`なしでResidual経路を実行すると、sealed multi-fold residual Walk-Forwardとrelease evidence workflowが実装・検証されるまでerrorになります。

これは意図的な安全境界です。単一development／test実行、CI合格、正しいBundle形状だけでは、収益性やProduction readinessは証明されません。Productionは引き続き**NO-GO**です。
