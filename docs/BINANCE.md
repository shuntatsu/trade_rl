# Binance Public Data Workflow

このIntegrationは、BinanceのPublic market dataから決定論的Dataset artifactを作成します。Account認証、残高取得、注文送信は行いません。

> Production status: **NO-GO**

## 対応範囲

- Spot
- USDⓈ-M Linear futures
- Binance Vision archiveとPublic REST fallback
- Kline、Funding、Mark/Index price、Execution metadata
- Incremental cacheと不足期間だけの更新

現在のAccounting modelはLinear productを前提とするため、COIN-M inverse futuresはFail closedします。

## 単一Symbolの例

```bash
uv run trade-rl data binance \
  --market usds-m \
  --symbol BTCUSDT \
  --interval 1h \
  --start-time 2026-06-01T00:00:00Z \
  --end-time 2026-06-29T00:00:00Z \
  --transport vision \
  --tick-size 0.1 \
  --lot-size 0.001 \
  --minimum-notional 5 \
  --listed-at 2019-09-08T00:00:00Z \
  --output artifacts/datasets/binance-btcusdt
```

出力先にはCanonical `manifest.json`と`arrays.npz`を作成します。同じPathを異なる内容で上書きしません。

## Metadata mode

| Mode | Integrity | 用途 |
|---|---|---|
| `historical_signed` | 最高 | 期間を覆う署名済みPoint-in-time metadata |
| `frozen_snapshot` | 非Point-in-time | Current official payloadをByte単位で固定 |
| `conservative_static` | 近似 | Versioned static ruleを保守的に適用 |

`frozen_snapshot`と`conservative_static`は、過去時点の真実として偽装しません。Modeと制約をDataset identityとReportへ残します。

## Cache

Public archiveは、学習Artifactとは別の再利用Cacheに保存します。学習開始時は次の順で処理します。

1. Cache manifestとFile digestを検証
2. 対象期間の欠損を検出
3. 不足ArchiveだけをDownload
4. 最新の未確定区間だけをPublic endpointから補完
5. Datasetを再構築しIdentityを検証

Raw cacheを削除すると再Downloadが必要ですが、Published dataset artifactは別に保持されます。

## Multi-Timeframe

維持対象Full researchは15mをDecision clockとし、15m、1h、4h、1dのNative sequenceを使用します。

- Higher-timeframe Barは完成後にだけ利用可能
- As-of alignmentでFuture valueを参照しない
- Derived featureはSource ageをStalenessへ伝播
- Symbol順、Feature順、Clock windowをDatasetとPolicy identityへ固定

## Target-weight growth profiles

長期複利成長の比較では、報酬、action space、BC、encoder、執行条件を同時に変更しません。次の3設定はすべてdirect target-weight、同一Oracle BC、同一Transformer、同一hard-risk条件を使用します。報酬とEpisode境界の正本契約は[`REWARD_OBJECTIVE.md`](REWARD_OBJECTIVE.md)です。

| Profile | 役割 | Objective |
|---|---|---|
| [`training-target-weight-growth-ppo.json`](../examples/binance-multitimeframe/training-target-weight-growth-ppo.json) | 必須対照群 | `gamma=1.0`、有限720時間、実コスト控除後net log growth、通常PPO |
| [`training-target-weight-constrained-growth.json`](../examples/binance-multitimeframe/training-target-weight-constrained-growth.json) | 本命候補 | 同じ有限期間growth objective、Lagrangian PPOでsoft constraint予算を管理 |
| [`training-target-weight-constrained-growth-discounted.json`](../examples/binance-multitimeframe/training-target-weight-constrained-growth-discounted.json) | 時間選好アブレーション | 168時間half-lifeのdiscounted continuing objective |

Gamma-one PPOとLagrangianの6-fold比較は[`walk-forward-target-weight-constrained-growth.json`](../examples/binance-multitimeframe/walk-forward-target-weight-constrained-growth.json)を使用します。Discounted continuing ablationは、観測とEpisode境界が異なるため同じcandidate setへ混在させず、[`walk-forward-target-weight-constrained-growth-discounted.json`](../examples/binance-multitimeframe/walk-forward-target-weight-constrained-growth-discounted.json)で独立に評価します。両方とも`run_file`でstandalone profileを参照し、埋め込みコピーによる設定ドリフトを起こしません。

[`training-full.json`](../examples/binance-multitimeframe/training-full.json)は、baseline、drawdown、excess growth、時間割引を混合したlegacy shaping比較として維持します。Production defaultではありません。

Hard safetyは学習成功へ依存させません。`max_abs_weight`、`max_gross`、drawdown stop、minimum equity、証拠金、取引所ルールは環境とpre-trade riskが常に強制します。Lagrangianはdrawdown excess、turnover、execution costなどのsoft budgetだけを調整します。

Gamma-one profileでは、720時間をMDP内の明示的な有限期間として扱います。`episode_boundary_mode=finite_horizon_termination`、`finite_horizon_observation=true`、`liquidate_on_end=false`を固定し、時間上限では保有Positionをmark-to-marketしたまま`terminated=true`とします。人工的なforced closeやterminal-equity shapingは行いません。これにより区間net log rewardの和が720時間終端のnet log wealthへ一致します。

Discounted profileでは、720時間を外部の学習窓として扱います。`episode_boundary_mode=external_truncation`、`finite_horizon_observation=false`、`liquidate_on_end=false`を固定し、時間上限では`truncated=true`としてterminal observationから継続価値をbootstrapします。Gamma-one profileとdiscounted profileのEpisode returnを同じ尺度として直接比較せず、共通の実コスト控除後net log growth証拠で比較します。

## Target-weight production gate

`trade_rl.evaluation.target_weight_growth_gate`は、6 fold × 3 seedの共通経済証拠からGO/NO-GOを決定します。異なるgammaやreward shapingのepisode rewardは比較せず、実コスト控除後net log growthを使用します。

Production候補は次をすべて満たす必要があります。

- Nominal 18 cellのnet log growth中央値が正
- Baselineとのpaired difference中央値が正
- Fold-cluster bootstrap 95%下限が0を上回る
- 6 fold中4 fold以上でseed中央値のpaired differenceが正
- 全3 seedでfold中央値が非負、うち2 seed以上が正
- Forced liquidation、margin deficit、insolvency、hard-safety violationが全scenarioでゼロ
- 各soft constraintは全foldのpoint estimateとpooled one-sided 95%上限がbudget以下
- Joint 2xでpaired difference中央値が正
- Joint 3xでnet log growth中央値が非負
- Architecture、reward、constraint、checkpoint/serving identityが検証済み

G1-PPOとG1-Lagrangianが両方gateを満たした場合、fold-cluster bootstrapでLagrangian minus PPOのnet log growth下限が0を上回るときだけLagrangianを採用します。差が有意でなければ、運用が単純なG1-PPOを採用します。どちらもgateを満たさなければRLはProduction採用しません。

判定結果と入力証拠はcanonical payloadとSHA-256 digestを持ち、後続ArtifactStoreへ同一内容で保存できます。

## Metadata evidence

`historical_signed`では、Market、Symbol順、Coverage、Issued time、Source URI、Policy version、Payload digestを署名対象にします。Research interval全体を覆わないEvidenceは拒否します。

Private keyをRepository、Docker image、Actions secret、Runtimeへ配置しません。TrainerとServingへ渡すのはPublic keyと署名済みArtifactだけです。

## Dataset使用前の確認

- 時刻がUTCでClose時刻として解釈されている
- Missing barと非取引期間を区別している
- AvailabilityとStalenessが値と別に存在する
- Fee、Spread、Participation、Tick/Lot、Minimum notionalが対象市場と一致する
- Funding、Borrow、Mark/Index priceのCoverageが十分
- 上場前・廃止後をActiveとして扱っていない
- `historical_signed`以外の制約がReportに残っている

Dataset生成成功は、取引戦略の有効性やProduction readinessを意味しません。

### Raw archive content evidence

Vision archive cacheはpayloadだけを信用しません。各`.bin`に`binance_vision_raw_cache_v1` sidecarを併置し、URL、取得時刻、byte数、SHA-256、ETag、Last-Modified、downloader identityを固定します。再利用時はbyte列を再hashし、sidecar欠落、size不一致、digest不一致をfail closedします。

## Action-head ablation

The maintained Gate-versus-direct target-weight comparison is declared by
`examples/binance-multitimeframe/walk-forward-action-head-ablation.json`.
Its two `run_file` candidates are required by regression test to differ only in
`training.policy_actor_head`; folds, seeds, Oracle teacher, encoder, PPO, reward,
risk, execution costs, and stress scenarios remain identical.

Run the same market walk-forward command used for the other canonical profiles,
substituting `walk-forward-action-head-ablation.json` as the configuration file.
Compare net return, baseline uplift, maximum drawdown, turnover, cost, action-stage
L1 metrics, and seed dispersion. This experiment does not authorize live exchange
routing.
