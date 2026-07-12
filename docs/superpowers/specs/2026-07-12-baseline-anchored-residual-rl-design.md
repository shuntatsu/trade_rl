# Baseline-Anchored Residual RL 再設計仕様

日付: 2026-07-12
状態: 設計承認済み・実装前
対象: `shuntatsu/trade_rl`

## 1. 背景

現行のポートフォリオPPOは、15銘柄のターゲットウェイトを直接出力する。実データ検証では、弱い予測信号、実執行コスト、turnover罰則、no-trade band、検証モデル選択が組み合わさり、復元モデルが完全flatとなった。一方、同一OOS期間のルールベース `trend_following` は正の収益とSharpeを示した。

この結果は、PPOそのものが必ず無価値であることを意味しない。しかし、PPOに方向性シグナルの発見、銘柄配分、回転抑制、リスク制御、執行後処理を同時に学ばせる現行責務分担は過大であり、ゼロ行動がflatに対応する構造は弱いアルファ下で保守崩壊を誘発する。

本設計は、実証済みの因果的trend-followingを基準方策として固定し、RLの責務を「基準方策に対する追加価値の学習」に限定する。

## 2. 目的

1. ゼロ行動をflatではなく純trend-followingに対応させる。
2. RLが学習できない場合でも、少なくとも基準方策と同等の提案を生成する。
3. RLの追加価値を、同一期間・同一コスト・同一リスク制約のshadow baselineとの差分で直接測定する。
4. `decision_every > 1` でactionとrewardの対応を壊さない。
5. 4h階層ゲートを冪等にし、現在保有を反復縮小しない。
6. Gate 2から到達不能なoracleを除外し、事前登録した実行可能baselineとの比較に限定する。
7. 学習、評価、Serving Planeが同一のaction-to-target契約を使用する。
8. flat化、過剰売買、HTF抑制、後処理freeze、checkpoint巻き戻しをレポートから識別可能にする。

## 3. 非目的

本フェーズでは以下を行わない。

- 15次元の銘柄別自由残差をRLへ再導入しない。
- carry、crowding、複数戦略のメタアロケータを同時に実装しない。
- Gate閾値を今回の単一holdoutに合わせて緩和しない。
- maker約定率、板キュー、逆選択を新たにモデル化しない。
- 単一seedまたは単一splitの好結果でProduction GOにしない。
- 既存Control Plane / Serving Planeの境界を崩さない。

## 4. 検討案と採用判断

### 4.1 現行15次元PPOの調整

変更量は少ないが、過去のアブレーションでflatと過剰売買の二極化が確認されている。予測・配分・執行責務の過密も残るため不採用とする。

### 4.2 Baseline-Anchored Residual RL

trend-followingを構造的基準にし、RLは低次元の露出倍率と残差アルファ配分のみを学ぶ。ゼロ行動同値、比較可能性、診断容易性が高いため採用する。

### 4.3 複数スリーブのRLメタアロケータ

長期的な拡張候補だが、現時点では負の期待値を持つスリーブも混在する。まず本設計でRLが単一の有効baselineに追加価値を出せることを確認した後に検討する。

## 5. 主要不変条件

### 5.1 ゼロ行動同値

同一の市場状態、ポートフォリオ状態、コスト設定、リスク設定において、RL actionがゼロベクトルなら、hybridの提案ターゲット、最終執行ウェイト、コスト、損益はshadow trend baselineと数値許容誤差内で一致しなければならない。

許容誤差:

- ウェイト: `atol <= 1e-8`
- コスト、1step損益: `atol <= 1e-10`

### 5.2 Oracle非必須

`oracle_dp` と `oracle_ic*` は診断専用であり、Gate 2のPASS条件に含めない。

### 5.3 同一執行条件

hybridとshadow baselineは、同じ価格、funding、fee、spread、impact、decision interval、HTF制約、post-processing、pre-trade riskを使用する。

### 5.4 Train / Eval / Serve一致

基準方策生成、action composition、HTF proposal constraint、post-processingの順序は、学習環境、OOS評価、Serving Planeで共通実装を使用する。

### 5.5 非意思決定actionの不存在

`decision_every > 1` の場合、PPOは意思決定間隔ごとに1 transitionだけ受け取る。無視されるactionを持つtransitionを生成しない。

## 6. 全体アーキテクチャ

```text
FeatureSet + portfolio state
        |
        +--> TrendCorePolicy -------------------+
        |                                       |
        +--> ResidualAlphaProvider -- gate -----+--> BaselineResidualComposer
        |                                       |          |
        +--> PPO policy: [a_trend, a_alpha] -----+          v
        |                                             proposed weights
        |                                                    |
        +--------------------------------------> HTFProposalConstraint
                                                             |
                                                     PortfolioPostProcessor
                                                             |
                                                     PreTradeRiskVerifier
                                                             |
                                                        execution model
                                                             |
                                +----------------------------+----------------+
                                |                                             |
                           hybrid book                                  shadow trend book
                                |                                             |
                                +-------------- relative reward --------------+
```

## 7. コンポーネント設計

### 7.1 TrendCorePolicy

責務:

- 既存 `trend_following_strategy` と同じ因果的時系列モメンタムを生成する。
- 銘柄順、lookback、rebalance cadenceを明示的設定として持つ。
- 状態を隠さず、`weights(fs, t, current_weights) -> np.ndarray` 契約を維持する。

初期既定値:

- lookback: 48 base bars
- rebalance_every: 24 base bars
- gross cap: 1.0
- allow_short: true

既存baselineとの数値一致を回帰テストで保証する。

### 7.2 ResidualAlphaProvider

責務:

- 市場中立の相対アルファ候補を生成する。
- Gate不合格時はゼロベクトルを返す。
- 推論時に未来情報を使用しない。

初期方式:

- target: `cs_demean`
- model: `gbm`
- 因果ローリング再学習
- horizon分のembargo
- 出力はクロスセクショナル中心化後、gross 1以下へ射影

Gate条件は既存の研究値ではなくdevelopmentデータ上で計算する。最低条件:

- mean OOS rank IC >= 0.02
- positive fold ratio >= 0.6
- stability t-stat >= 1.0

Gate不合格時もtrend-only RLは実行可能である。

### 7.3 BaselineResidualComposer

RL action spaceは2次元連続値 `[-1, 1]^2` とする。

```text
action[0] = a_trend
action[1] = a_alpha
```

初期変換:

```text
trend_scale = 1.0 + 0.5 * a_trend
alpha_scale = 0.30 * max(a_alpha, 0.0)
proposal = trend_scale * w_trend + alpha_scale * w_alpha
proposal = project_gross(proposal, max_gross=1.0)
```

範囲:

- trend_scale: `[0.5, 1.5]`
- alpha_scale: `[0.0, 0.30]`
- 最終proposal gross: `<= 1.0`

`a_alpha < 0` を負のalpha反転に使わない。相対アルファの符号反転は別戦略であり、本フェーズのaction意味論を不安定にするためである。

zero actionでは `trend_scale=1.0`, `alpha_scale=0.0` となる。

### 7.4 HTFProposalConstraint

現行の4hゲートは、post-processing後の保有ターゲットへ適用されるため、neutral時に同じ保有を反復縮小し得る。新設計ではHTF制約をstateful post-processingの前へ移す。

順序:

```text
composed proposal
  -> HTF proposal constraint
  -> EMA / vol target / concentration / DD / disagreement / no-trade
  -> execution
```

初期意味論:

- 4h trend > threshold: short proposalを0にする。
- 4h trend < -threshold: long proposalを0にする。
- neutral: proposalを `htf_neutral_scale` 倍する。

HTFは現在保有を直接変換せず、今回のdesired proposalだけを変換する。

例:

```text
current=0.05, raw proposal=0.10, neutral_scale=0.5
constrained proposal=0.05
post-process output=0.05
```

次バーも同条件なら0.05を維持し、0.025へ再縮小しない。

### 7.5 PortfolioPostProcessor

本設計の初期実験では、恣意的なhard dead-zoneと追加turnover罰則を無効化する。

初期設定:

- `no_trade_band = 0.0`
- `lambda_turnover = 0.0`
- `ema_alpha` は有効
- 実fee / spread / impactは有効
- max gross / max symbol weightは有効
- target vol / DD derisk / disagreement scalingは設定どおり有効

回転抑制は次の構造で行う。

1. trend coreの低頻度更新
2. RL decision interval
3. EMA
4. max turnover hard constraint（後続フェーズで導入可能）
5. 実コスト

追加turnover penaltyは、実コストだけでは過剰売買が抑えられないことを複数seedで確認した場合のみ、developmentデータ上で再導入する。

## 8. Decision Interval Aggregation

### 8.1 現行問題

現行環境では非意思決定バーでもPPO transitionが生成され、actionだけが無視される。その間のrewardは無視されたactionへ紐づくため、action-credit assignmentが壊れる。

### 8.2 新契約

1回の `env.step(action)` は、最大 `decision_every` 本のbase barを内部で進める。

処理:

1. decision barでactionを受け取る。
2. trend / alpha / composer / HTF / post-processを1回実行する。
3. 決定したtargetをinterval内で保持する。
4. 各base barで価格損益、fundingを計算する。
5. 取引コストはtarget変更時に1回だけ計上する。
6. interval内のlog returnを集計する。
7. interval終了時のobservationと1つのrewardを返す。

エピソード末尾で残りbarが `decision_every` 未満の場合は、存在するbarだけを進める。

`info` には以下を含める。

- `bars_advanced`
- `interval_gross_return`
- `interval_cost`
- `interval_funding`
- `interval_net_return`
- `decision_step_index`

### 8.3 Annualization

Sharpe / Sortino / volatility annualizationは、base timeframeのbars/yearではなく、評価するreturn系列の実時間間隔に整合させる。

- base-bar系列を使う指標: base bars/year
- decision-interval系列を使う指標: base bars/year / decision_every

どちらの系列を用いたかをreportへ記録する。

## 9. Shadow Baselineと報酬

### 9.1 Shadow Trend Book

各hybrid環境は、同じ初期資本からshadow trend bookを並走させる。

shadow bookは以下を共有する。

- FeatureSet
- decision interval
- HTF constraint
- post-processing
- execution cost model
- funding
- pre-trade hard limits

異なるのはcomposerだけで、shadowは常にzero action相当を使う。

### 9.2 基本報酬

各decision intervalについて、複利整合のためlog return差を使う。

```text
r_hybrid = log(V_hybrid_after / V_hybrid_before)
r_shadow = log(V_shadow_after / V_shadow_before)
reward = reward_scale * (r_hybrid - r_shadow)
```

初期設計では追加turnover penaltyを加えない。実コスト差はすでに `r_hybrid - r_shadow` に含まれる。

### 9.3 リスク制約

hard risk limitsはrewardではなく実行前制約として扱う。

初期フェーズでは、drawdown差やtracking errorへのsoft penaltyを加えない。まず純粋な相対収益を学習できるかを確認する。複数seedで高収益・過大DDが一貫して観測された場合、次フェーズで次のいずれかを1つずつ試験する。

- downside excess penalty
- drawdown excess penalty
- volatility excess penalty

複数penaltyを同時導入しない。

### 9.4 破産・非有限値

以下の場合はエピソードを終了し、明示的失敗rewardを返す。

- portfolio value <= 1e-6 * initial capital
- NaN / Inf in action, proposal, target, cost, reward
- pre-trade hard constraint violation

shadow bookだけが失敗した場合も評価不能としてエピソードを失敗させる。hybridだけを有利に継続しない。

## 10. 観測設計

既存のper-symbol MTF特徴とportfolio stateを維持する。追加する明示的状態:

- current trend core weights
- current residual alpha weightsまたはalpha score
- current trend gross
- current alpha gross
- previous trend_scale
- previous alpha_scale
- shadow portfolio drawdown
- hybrid minus shadow cumulative log return

ただし、観測次元の急増を避けるため、初期実装ではtrend / alpha weightsをper-symbol既存ブロックへ各1列追加し、グローバル値は集約値だけを追加する。

ServingBundleのobservation schema versionを更新し、旧bundleとの誤読をfail closedにする。

## 11. PPO設定と学習ラン種別

8環境、`n_steps=256` の場合、1 updateは2,048 environment transitionsである。decision aggregation後は1 transitionが複数base barを含む点をreportする。

ラン種別:

### 11.1 Smoke

- 目的: 実行経路、shape、finite、保存復元の確認
- 最低updates: 5
- release evidence: 不可

### 11.2 Research

- 最低updates: 50
- 8 env時の最低timesteps: 102,400
- checkpoint評価回数: 10回以上
- 複数seed必須

### 11.3 Release candidate

- 最低updates: 100
- 8 env時の最低timesteps: 204,800
- development WF、cost sensitivity、sealed holdout必須
- 単一seed不可

CLIは `--run-tier {smoke,research,release}` を受け、総timestepsがtier最低値を下回る場合:

- smoke: 実行可
- research: error
- release: error

既存 `--timesteps` は維持するが、tier検証を追加する。

## 12. Checkpoint Selection

### 12.1 評価頻度

固定20,000 stepsを廃止し、次で決める。

```text
n_eval_targets = 10
eval_freq = max(one_rollout_steps, floor(total_timesteps / n_eval_targets))
eval_freq = rollout境界へ切り上げ
```

学習開始時、各評価点、学習終了時を候補にする。

### 12.2 選択スコア

validation期間を最低3個の連続blockに分け、各blockでhybridとshadowを同時評価する。

block score:

```text
excess_log_return = log(V_hybrid_end / V_hybrid_start)
                  - log(V_shadow_end / V_shadow_start)
```

checkpointの主要スコア:

```text
median_block_excess_log_return
```

タイブレーク:

1. positive block ratioが高い
2. hybrid maxDD - shadow maxDDが小さい
3. turnover excessが小さい
4. 早いcheckpoint

初期checkpointはzero-action baselineとしてスコア0を持つ。

checkpoint採用条件:

- median block excess > 0
- positive block ratio >= 0.5

条件を満たすcheckpointが無い場合は初期zero-action checkpointを復元し、結果を `baseline_fallback=true` と記録する。このfallbackは失敗ではなく、安全な研究結果である。ただしGate 2はPASSしない。

## 13. Gate設計

### 13.1 Trend Gate

trend coreの有効性をdevelopment WFで確認する。

最低条件:

- cost 1x median return > flat
- positive fold ratio >= 0.6
- 方向性trend gateが有意、またはrule baselineが事前登録WF基準を満たす

trend coreが不合格ならResidual RL学習を開始しない。

### 13.2 Residual Alpha Gate

Section 7.2のIC条件を満たす場合だけalpha sleeveを有効化する。不合格時はtrend-scale-onlyモードへ縮退する。

### 13.3 RL Development Gate

fold x seedで以下を満たす。

- median excess return > 0
- hybridがshadowを上回るrun比率 >= 0.6
- cost 2x median excess return >= 0
- hybrid median maxDD <= shadow median maxDD + 0.05

### 13.4 Gate 2: Sealed Holdout

必須比較対象:

- flat
- shadow trend_following

PASS条件:

- hybrid total return > flat total return
- hybrid total return > shadow trend total return
- hybrid maxDD <= shadow trend maxDD + 0.05
- hybridのpaired excess return bootstrap片側p-value < 0.05

診断専用:

- equal_weight_bh
- inverse_vol
- cross_momentum
- trend_v2
- carry
- crowding
- oracle_dp
- oracle_ic*

診断専用baselineの勝敗はGate 2を変更しない。

## 14. 実験マトリクス

同一development folds、seeds、費用条件で次を比較する。

```text
A: pure trend core
B: trend core + RL trend scale
C: trend core + fixed gated residual alpha
D: trend core + RL trend scale + RL alpha scale
```

必要seed数:

- research: 3以上
- release candidate: 5以上

費用条件:

- 1x
- 2x

decision interval候補:

- 1
- 4
- 8 base bars

候補選択はdevelopmentデータだけで行う。sealed holdoutで候補を変更しない。

採用規則:

- BがAに勝てなければRL trend scalingを採用しない。
- CがAに勝てなければresidual alphaを採用しない。
- Dがmax(B, C)に勝てなければ複合RLを採用しない。
- どのRL構成もAに勝てなければ、Production候補はpure trend coreとし、RL研究を停止する。

## 15. レポート契約

`train_report.json` または後継reportへ次を保存する。

### 15.1 Identity

- git SHA
- dataset identity
- symbols
- date range
- feature schema version
- action schema version
- run tier
- seed
- PPO update count

### 15.2 Gates

- trend gate
- residual alpha gate
- development RL gate
- Gate 2
- oracleはdiagnosticフラグ付き

### 15.3 Checkpoints

- initial score
- evaluation step
- block scores
- median excess
- positive block ratio
- selected step
- baseline fallback

### 15.4 Actions

- raw `a_trend` mean/std/min/max
- raw `a_alpha` mean/std/min/max
- trend_scale distribution
- alpha_scale distribution

### 15.5 Weight stages

- trend core gross
- residual alpha gross
- composed proposal gross
- HTF-constrained gross
- post-process gross
- executed gross
- desired turnover
- executed turnover
- tracking ratio
- freeze ratio
- HTF zeroed fraction
- HTF neutral-scaled fraction

### 15.6 Performance

hybridとshadowの両方について:

- total return
- annualized return
- Sharpe
- Sortino
- max drawdown
- turnover
- cost
- funding
- n trades

paired metrics:

- excess total return
- excess log return
- excess information ratio
- paired bootstrap confidence interval
- paired bootstrap p-value

## 16. ServingBundleとServing Plane

ServingBundleに以下を追加する。

- action schema: `baseline_residual_v1`
- trend core config
- residual alpha provider configとgate結果
- composer config
- HTF proposal constraint config
- decision aggregation config
- observation schema version
- shadow baseline identity

Serving Planeはbundleから同じ構成を復元する。modelの2次元actionを直接15銘柄ウェイトとして扱ってはならない。

旧action schema bundleとの互換変換は行わない。schema mismatchはfail closedとする。

`/api/signal/latest` の監査eventへ以下を含める。

- raw 2D action
- trend_scale
- alpha_scale
- trend weights digest
- alpha weights digest
- composed weights
- HTF-constrained weights
- final target weights

## 17. 後方互換性

旧 `direct_weights_v1` モデルは研究artifactとして読み込み可能なまま残してよいが、新しいrelease pathでは不適格とする。

CLIに明示的なaction modeを追加する。

```text
--action-mode {direct,baseline-residual}
```

既定値移行:

1. 実装直後は `direct` を既定に維持し、回帰を防ぐ。
2. P0、WF、Serving parityテストが通った後、研究CLIの既定を `baseline-residual` へ変更する。
3. release pathは `baseline-residual` のみ許可する。

## 18. テスト設計

### 18.1 Unit

- zero actionでcomposer出力がtrend coreに一致する。
- action boundsからscale boundsが守られる。
- alpha gate不合格時にalpha_scaleが常に0になる。
- proposal grossが1以下になる。
- HTF neutralが同一desired targetを反復縮小しない。
- HTF方向禁止が逆方向だけを0にする。
- decision aggregationが正しいbar数を進める。
- interval costがtarget変更時に1回だけ計上される。
- non-finite actionを拒否する。
- oracleがGate 2 mandatory setに入らない。

### 18.2 Integration

- zero-action agentとshadow trendのequity curveが一致する。
- train / evaluate / serveで同じ入力から同じtargetを生成する。
- save / load後もzero-action同値が維持される。
- action schema mismatch bundleをServingが拒否する。
- residual alpha gate不合格でtrend-onlyへ縮退する。
- checkpoint評価が総timesteps内で10回以上発生する。
- baseline fallback時にpure trend modelが保存される。

### 18.3 Synthetic controls

Positive control:

- trend coreに対し、観測可能なレジームで露出倍率を変えると改善できる合成市場
- BまたはDがAを上回る

Negative control:

- trend coreが最適で、追加actionが期待値を持たない市場
- checkpoint selectionがzero-action baselineへfallbackする
- turnoverが暴走しない

Adversarial:

- NaN feature
- all-zero alpha
- 4h trendが閾値近傍で振動
- episode末尾がdecision interval未満
- shadowとhybridの一方だけpre-trade violation

### 18.4 Regression

- 既存trend_following baselineの数値を維持する。
- direct modeの既存研究テストを壊さない。
- Gate 2の通常baseline表示とoracle診断表示を維持する。
- bundle digest、registry、release eligibilityの既存契約を維持する。

## 19. 実装単位

推奨ファイル境界:

```text
mars_lite/trading/trend_core.py
mars_lite/trading/residual_alpha.py
mars_lite/trading/baseline_residual.py
mars_lite/trading/htf_constraint.py
mars_lite/env/baseline_residual_env.py
mars_lite/learning/relative_val_selection.py
mars_lite/eval/relative_evaluation.py
mars_lite/pipeline/gates.py
```

既存 `PortfolioTradingEnv` を巨大な条件分岐で拡張せず、共通の市場進行・執行部分を抽出した上で、direct環境とbaseline-residual環境を薄い構成にする。

`DecisionPipeline` は以下の明確な段階を持つよう再編する。

```text
compose proposal
-> constrain proposal
-> stateful post-process
-> risk verify
```

各段階の入力・出力をdiagnosticsへ残す。

## 20. エラー処理

- action dimension不一致: 即時error
- schema version不一致: bundle拒否
- trend core生成失敗: 学習・serveを停止
- residual alpha生成失敗: release pathでは停止、research pathでは明示設定時のみtrend-onlyへ縮退
- shadow mismatch: テストまたはruntime assertionで停止
- report必須項目欠落: candidate construction拒否
- oracleがmandatory gate setへ混入: assertion failure

silent fallbackは行わない。許可されたfallbackは、明示的に設計した `baseline_fallback` と residual gate不合格時のtrend-onlyだけである。

## 21. 段階導入

### Phase 1: Correctness foundation

- Gate 2 oracle除外
- HTF適用順序修正
- decision aggregation
- relative diagnostics
- zero-action parity

### Phase 2: Baseline-residual policy

- TrendCorePolicy
- ResidualAlphaProvider
- 2D composer
- shadow book
- relative reward
- relative checkpoint selection

### Phase 3: Research validation

- A/B/C/D matrix
- 3+ seeds
- folds
- costs 1x/2x
- decision interval 1/4/8

### Phase 4: Release integration

- ServingBundle schema更新
- Serving parity
- release gate更新
- 5+ seeds
- sealed holdout

## 22. 受入基準

実装完了は次をすべて満たすこととする。

1. zero-action parityのunit/integrationテストが通る。
2. `decision_every > 1` で1actionにつき1集約rewardだけが返る。
3. HTF neutralが保有を再帰的に縮小しない。
4. Gate 2のmandatory setにoracleが存在しない。
5. total timestepsに対して10回以上のcheckpoint評価が行われる。
6. synthetic positive controlでRLがbaselineを改善する。
7. synthetic negative controlでbaseline fallbackが選択される。
8. train/eval/serve target parityが通る。
9. reportからaction、各weight stage、shadow差、checkpoint理由を追跡できる。
10. 全既存CI gateが通る。
11. 実データdevelopment WFでA/B/C/D比較が完了する。
12. RL構成がAを上回らない場合、RLをProduction候補にしない。

## 23. 設計上の最終判断

本システムの目的はRLを使用することではなく、同一条件の実行可能baselineより高いリスク調整後価値を再現可能に生むことである。

したがって、Baseline-Anchored Residual RLは「trend-followingを置き換えるモデル」ではなく、「trend-followingに追加価値を出せるかを厳密に検証するモデル」と位置づける。複数fold・複数seed・cost sensitivity・sealed holdoutで追加価値を示せない場合、pure trend coreを採用し、RL部分を棄却する。
