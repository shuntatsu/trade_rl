# Baseline-Anchored Residual RL 再設計仕様

日付: 2026-07-12
状態: 設計承認済み・実装前
対象: `shuntatsu/trade_rl`

## 1. 背景

現行のポートフォリオPPOは15銘柄のターゲットウェイトを直接出力する。実データ検証では、弱い予測信号、実執行コスト、turnover罰則、no-trade band、検証モデル選択が組み合わさり、復元モデルが完全flatとなった。一方、同一OOS期間のルールベース `trend_following` は正の収益とSharpeを示した。

これはPPOが必ず無価値であることを意味しない。しかし、方向性シグナルの発見、銘柄配分、回転抑制、リスク制御、執行後処理を1つのPPOへ同時に負わせ、ゼロ行動をflatへ対応させる現行構造は、弱いアルファ下で保守崩壊しやすい。

本設計は、因果的trend-followingを基準方策として固定し、RLの責務を「基準方策に対する追加価値の学習」に限定する。

## 2. 目的

1. identity actionを純trend-followingに対応させる。
2. RLが追加価値を学習できない場合は、基準方策へ厳密にfallbackする。
3. RLの追加価値を、同一期間・同一コスト・同一リスク制約のshadow baselineとの差分で測定する。
4. `decision_every > 1` でactionとrewardの対応を壊さない。
5. 4h階層ゲートを冪等にし、現在保有を反復縮小しない。
6. Gate 2から到達不能なoracleを除外し、実行可能baselineとの比較に限定する。
7. 学習、評価、Serving Planeで同一のaction-to-target契約を使う。
8. flat化、過剰売買、HTF抑制、post-process freeze、checkpoint巻き戻しをレポートから識別可能にする。

## 3. 非目的

本フェーズでは以下を行わない。

- 15次元の銘柄別自由残差をRLへ再導入しない。
- carry、crowding、複数戦略のメタアロケータを同時実装しない。
- Gate閾値を今回の単一holdoutに合わせて緩和しない。
- maker約定率、板キュー、逆選択を新たにモデル化しない。
- 単一seedまたは単一splitの好結果でProduction GOにしない。
- オンラインServing中にGBMを再学習しない。
- 既存Control Plane / Serving Planeの境界を崩さない。

## 4. 検討案と採用判断

### 4.1 現行15次元PPOの調整

変更量は少ないが、過去のアブレーションでflatと過剰売買の二極化が確認されている。予測・配分・執行責務の過密も残るため不採用とする。

### 4.2 Baseline-Anchored Residual RL

trend-followingを構造的基準にし、RLは低次元のtrend timing選択と残差アルファ配分だけを学ぶ。identity action同値、比較可能性、診断容易性が高いため採用する。

### 4.3 複数スリーブのRLメタアロケータ

長期的な拡張候補だが、現時点では負の期待値を持つスリーブも混在する。まず本設計でRLが単一の有効baselineに追加価値を出せることを確認した後に検討する。

## 5. 主要不変条件

### 5.1 Identity action同値

identity actionは `[0.0, 0.0]` とする。同一の市場状態、ポートフォリオ状態、コスト設定、リスク設定においてidentity actionを与えたhybridは、基準trendの提案ターゲット、最終執行ウェイト、コスト、損益と数値許容誤差内で一致しなければならない。

許容誤差:

- ウェイト: `atol <= 1e-8`
- コスト、1step損益: `atol <= 1e-10`

### 5.2 Oracle非必須

`oracle_dp` と `oracle_ic*` は診断専用であり、Gate 2のPASS条件に含めない。

### 5.3 同一執行条件

hybridとshadow baselineは、同じ価格、funding、fee、spread、impact、decision interval、HTF制約、post-processing、pre-trade riskを使用する。両bookは独立したportfolio stateとpost-process stateを持つが、設定値と実装は共有する。

### 5.4 Train / Eval / Serve一致

trend生成、action composition、HTF proposal constraint、post-processingの順序は、学習環境、OOS評価、Serving Planeで共通実装を使う。

### 5.5 非意思決定actionの不存在

`decision_every > 1` の場合、PPOは意思決定間隔ごとに1 transitionだけ受け取る。無視されるactionを持つtransitionを生成しない。

### 5.6 明示的fallback

許可されるfallbackは次の2つだけである。

- residual alpha gate不合格時のtrend-only mode
- 有効なRL checkpointが無い場合のidentity policy

silent fallbackは行わない。

## 6. 全体アーキテクチャ

```text
FeatureSet + actual portfolio state
        |
        +--> TrendFamily(base/fast/slow) --------+
        |                                        |
        +--> FrozenResidualAlphaProvider -- gate +--> BaselineResidualComposer
        |                                        |          |
        +--> PPO action [a_trend_mix, a_alpha] ---+          v
        |                                              proposed weights
        |                                                     |
        +---------------------------------------> HTFProposalConstraint
                                                              |
                                                      PortfolioPostProcessor
                                                              |
                                                      PreTradeRiskVerifier
                                                              |
                                                         execution model
                                                              |
                                +-----------------------------+----------------+
                                |                                              |
                           hybrid book                                   shadow base-trend book
                                |                                              |
                                +--------------- relative reward --------------+
```

## 7. コンポーネント設計

### 7.1 TrendFamily

責務:

- 既存 `trend_following_strategy` と同じ因果的時系列モメンタムを基準trendとして生成する。
- 同じ計算式でfast / slow候補を生成する。
- 各候補は銘柄順を維持し、gross 1以下とする。

初期設定:

- fast lookback: 24 base bars
- base lookback: 48 base bars
- slow lookback: 96 base bars
- rebalance_every: 24 base bars
- gross cap: 1.0
- allow_short: true

基準baselineはbase lookback 48であり、既存 `trend_following_strategy` と数値一致しなければならない。

### 7.2 FrozenResidualAlphaProvider

責務:

- 市場中立の相対アルファ候補を生成する。
- Gate不合格時はゼロベクトルを返す。
- 推論時に未来情報を使わない。
- Serving中に再学習しない。

初期方式:

- target: `cs_demean`
- model: `gbm`
- fold評価では各foldのtrain区間だけでfit
- final modelはdevelopment区間だけでfit
- fitted model、特徴順、前処理、学習終了時刻をServingBundleへ保存
- holdout評価とServingは同じfitted artifactでpredict
- 出力はクロスセクショナル中心化後、gross 1以下へ射影

Gate条件はdevelopmentデータで計算する。

- mean OOS rank IC >= 0.02
- positive fold ratio >= 0.6
- stability t-stat >= 1.0

Gate不合格時もtrend-only RLは実行可能である。

### 7.3 BaselineResidualComposer

RL action spaceは2次元連続値 `[-1, 1]^2` とする。

```text
action[0] = a_trend_mix
action[1] = a_alpha
```

#### Trend timing action

`a_trend_mix` はレバレッジ倍率ではなく、gross上限内でfast / base / slowを選ぶ連続混合係数である。

```text
if a_trend_mix >= 0:
    w_trend = (1 - a_trend_mix) * w_base + a_trend_mix * w_fast
else:
    m = abs(a_trend_mix)
    w_trend = (1 - m) * w_base + m * w_slow
```

これにより:

- `a_trend_mix = 0`: base trend
- `a_trend_mix = 1`: fast trend
- `a_trend_mix = -1`: slow trend

各成分の凸結合なのでgrossは1以下を保つ。gross上限1.0のまま「1.5倍してから射影で元へ戻る」という無効actionを作らない。

#### Residual alpha action

```text
alpha_budget = 0.30 * a_alpha
proposal = (1 - abs(alpha_budget)) * w_trend + alpha_budget * w_alpha
proposal = project_gross(proposal, max_gross=1.0)
```

意味:

- `a_alpha = 0`: alpha無し
- `a_alpha > 0`: 予測方向の相対alphaへ最大30%配分
- `a_alpha < 0`: 反対方向の相対alphaへ最大30%配分

負方向を許す理由は、identity actionを0に保ちつつaction領域の両側を意味のある選択にするためである。反転がOOSで有効でなければcheckpoint選択とGateで棄却される。alpha gate不合格時は `alpha_budget=0` を強制する。

identity action `[0, 0]` では `w_trend=w_base`, `alpha_budget=0`, `proposal=w_base` となる。

### 7.4 Residual policy初期化

baseline-residual modeではBC warm startを使用しない。PPOのaction headのweightとbiasを0で初期化し、初期deterministic actionを厳密に `[0, 0]` とする。value networkは通常初期化を維持する。

初期parameter snapshotを学習開始前に保存し、baseline fallbackはこのsnapshotへ復元する。

探索用log standard deviationは有限の明示値で初期化する。deterministic評価はidentity action、stochastic学習はidentity周辺を探索する。

### 7.5 HTFProposalConstraint

現行の4hゲートはpost-processing後の保有ターゲットへ適用されるため、neutral時に同じ保有を反復縮小し得る。新設計ではHTF制約をstateful post-processingの前へ移す。

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

```text
current=0.05, raw proposal=0.10, neutral_scale=0.5
constrained proposal=0.05
post-process output=0.05
```

次バーも同条件なら0.05を維持し、0.025へ再縮小しない。

### 7.6 PortfolioPostProcessor

初期実験ではhard dead-zoneと追加turnover罰則を無効化する。

- `no_trade_band = 0.0`
- `lambda_turnover = 0.0`
- EMAは有効
- 実fee / spread / impactは有効
- max gross / max symbol weightは有効
- target vol / DD derisk / disagreement scalingは設定どおり有効

回転抑制は次で行う。

1. trend familyの低頻度更新
2. RL decision interval
3. EMA
4. 実コスト

追加turnover penaltyは、実コストだけでは過剰売買が抑えられないことを複数seedで確認した場合のみ、developmentデータ上で単独変数として再導入する。

## 8. Decision Interval Aggregation

### 8.1 現行問題

現行環境では非意思決定バーでもPPO transitionが生成され、actionだけが無視される。その間のrewardは無視されたactionへ紐づき、credit assignmentが壊れる。

### 8.2 新契約

1回の `env.step(action)` は最大 `decision_every` 本のbase barを内部で進める。

1. decision barでactionを受け取る。
2. trend / alpha / composer / HTF / post-processを1回実行する。
3. 決定したtargetをinterval内で保持する。
4. 各base barで価格損益、fundingを計算する。
5. 取引コストはtarget変更時に1回だけ計上する。
6. interval内のlog returnを集計する。
7. interval終了時のobservationと1つのrewardを返す。

エピソード末尾で残りbarが `decision_every` 未満の場合は、存在するbarだけを進める。

`info`:

- `bars_advanced`
- `interval_gross_return`
- `interval_cost`
- `interval_funding`
- `interval_net_return`
- `decision_step_index`

### 8.3 Annualization

- base-bar return系列: base bars/year
- decision-interval return系列: base bars/year / 実効平均bars_advanced

使用系列とannualization factorをreportへ記録する。

### 8.4 学習予算

PPO timestepsはdecision transitionsを数える。異なるdecision intervalの比較ではPPO update数を固定し、追加で処理したbase-bar総数とdataset pass相当数をreportする。

## 9. Shadow Baselineと報酬

### 9.1 Shadow Base-Trend Book

各hybrid環境は、同じ初期資本からbase trendのshadow bookを並走させる。

共有条件:

- FeatureSet
- decision interval
- HTF constraint
- post-processing設定
- execution cost model
- funding
- pre-trade hard limits

shadowは常にidentity action相当を使い、hybridとは独立したweights、portfolio value、peak、drawdown、post-process stateを持つ。

### 9.2 基本報酬

各decision intervalについてlog return差を使う。

```text
r_hybrid = log(V_hybrid_after / V_hybrid_before)
r_shadow = log(V_shadow_after / V_shadow_before)
reward = reward_scale * (r_hybrid - r_shadow)
```

追加turnover penaltyは加えない。実コスト差はすでに相対returnへ含まれる。

### 9.3 リスク制約

hard risk limitsはrewardではなく実行前制約として扱う。初期フェーズではdrawdown差、tracking error、volatilityへのsoft penaltyを加えない。

複数seedで高収益・過大DDが一貫して観測された場合に限り、次のいずれか1つをdevelopmentデータで試験する。

- downside excess penalty
- drawdown excess penalty
- volatility excess penalty

複数penaltyを同時導入しない。

### 9.4 破産・非有限値

次の場合はエピソードを終了し、明示的失敗rewardを返す。

- portfolio value <= 1e-6 * initial capital
- NaN / Inf in action, proposal, target, cost, reward
- pre-trade hard constraint violation

shadowだけが失敗した場合も評価不能としてエピソードを失敗させる。hybridだけを有利に継続しない。

## 10. 観測設計

Servingで再現可能な状態だけをpolicy observationへ含める。

既存のper-symbol MTF特徴とactual portfolio stateを維持し、per-symbolブロックへ次を追加する。

- fast trend weight
- base trend weight
- slow trend weight
- frozen residual alpha weight

グローバルブロックへ次を追加する。

- fast / base / slow trend gross
- residual alpha gross

次はtraining-onlyでありpolicy observationへ含めない。

- shadow portfolio value / drawdown
- hybrid minus shadow cumulative return
- validation score

これによりServingがshadow stateを持つ必要をなくす。ServingBundleのobservation schema versionを更新し、旧bundleとの誤読をfail closedにする。

## 11. PPO設定とラン種別

baseline-residual modeでは `n_envs` と `n_steps` から1 updateのtransition数を計算する。

ラン種別:

### 11.1 Smoke

- 最低updates: 5
- 目的: shape、finite、保存復元、経路確認
- release evidence: 不可

### 11.2 Research

- 最低updates: 50
- checkpoint評価回数: 10回以上
- 3 seeds以上

### 11.3 Release candidate

- 最低updates: 100
- development WF、cost sensitivity、sealed holdout必須
- 5 seeds以上

CLI:

```text
--run-tier {smoke,research,release}
```

`required_timesteps = required_updates * n_envs * n_steps` とする。指定timestepsが不足するresearch / release runはerrorにする。

## 12. Checkpoint Selection

### 12.1 評価頻度

```text
one_rollout_steps = n_envs * n_steps
n_eval_targets = 10
eval_freq = ceil_to_multiple(total_timesteps / n_eval_targets, one_rollout_steps)
```

学習開始時、各評価点、学習終了時を候補にする。同じstepを重複評価しない。

### 12.2 Validation score

validation期間を最低3個の連続blockに分け、各blockでhybridとshadowを同時評価する。

```text
block_excess = log(V_hybrid_end / V_hybrid_start)
             - log(V_shadow_end / V_shadow_start)
primary_score = median(block_excess)
```

タイブレーク:

1. positive block ratioが高い
2. hybrid maxDD - shadow maxDDが小さい
3. turnover excessが小さい
4. 早いcheckpoint

checkpoint採用条件:

- median block excess > 0
- positive block ratio >= 0.5

条件を満たすcheckpointが無い場合、学習前に保存したidentity parameter snapshotへ復元し、`baseline_fallback=true` と記録する。Gate 2はPASSしない。

## 13. Gate設計

### 13.1 Trend Core Gate

trend coreはdevelopment WFで次をすべて満たす必要がある。

- cost 1x median total return > 0
- cost 1x positive fold ratio >= 0.6
- cost 2x median total return >= 0
- cost 2x positive fold ratio >= 0.5

方向性trend t-statは診断へ残すが、取引Gateの代替条件にはしない。trend core不合格ならResidual RL学習を開始しない。

### 13.2 Residual Alpha Gate

Section 7.2のIC条件を満たす場合だけalpha sleeveを有効化する。不合格時はtrend-mix-only modeへ縮退する。

### 13.3 RL Development Gate

fold x seedで以下をすべて満たす。

- median excess return > 0
- hybridがshadowを上回るrun比率 >= 0.6
- cost 2x median excess return >= 0
- hybrid median maxDD <= shadow median maxDD + 0.05

### 13.4 Gate 2: Sealed Holdout

必須比較:

- flat
- shadow base trend

PASS条件:

- hybrid total return > 0
- hybrid total return > shadow total return
- hybrid maxDD <= shadow maxDD + 0.05
- moving-block paired bootstrapによるexcess returnの片側p-value < 0.05

bootstrap block lengthとseedは事前設定し、holdout結果を見て変更しない。

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

同一development folds、seeds、費用条件で比較する。

```text
A: pure base trend
B: trend family + RL trend mix
C: base trend + fixed gated residual alpha
D: trend family + RL trend mix + RL alpha budget
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

候補選択はdevelopmentデータだけで行い、sealed holdoutで変更しない。

採用規則:

- BがAに勝てなければRL trend mixを採用しない。
- CがAに勝てなければresidual alphaを採用しない。
- Dがmax(B, C)に勝てなければ複合RLを採用しない。
- どのRL構成もAに勝てなければ、Production候補はpure base trendとし、RL部分を棄却する。

## 15. レポート契約

### 15.1 Identity

- git SHA
- dataset identity
- symbols / date range
- feature / observation / action schema version
- run tier
- seed
- PPO update count
- decision transitions
- base bars advanced

### 15.2 Gates

- trend core gate
- residual alpha gate
- RL development gate
- Gate 2
- diagnostic-only baseline一覧

### 15.3 Checkpoints

- identity score
- evaluation step
- block scores
- median excess
- positive block ratio
- selected step
- baseline fallback

### 15.4 Actions

- raw `a_trend_mix` mean/std/min/max
- raw `a_alpha` mean/std/min/max
- fast/base/slow mix比率
- alpha budget mean/std/min/max
- contrarian alpha利用率

### 15.5 Weight stages

- fast/base/slow trend gross
- residual alpha gross
- composed proposal gross
- HTF-constrained gross
- post-process gross
- executed gross
- desired / executed turnover
- tracking ratio
- freeze ratio
- HTF zeroed fraction
- HTF neutral-scaled fraction

### 15.6 Performance

hybridとshadowの両方:

- total / annualized return
- Sharpe / Sortino
- max drawdown
- turnover / cost / funding
- n trades

paired metrics:

- excess total return
- excess log return
- excess information ratio
- moving-block bootstrap confidence interval
- moving-block bootstrap p-value

## 16. ServingBundleとServing Plane

ServingBundleへ追加する。

- action schema: `baseline_residual_v1`
- policy mode: `ppo_residual` または `baseline_only`
- trend family config
- frozen residual alpha artifact、特徴順、fit cutoff、gate結果
- composer config
- HTF proposal constraint config
- decision aggregation config
- observation schema version
- shadow baseline identity

Serving Planeはbundleから同じ構成を復元する。2次元actionを直接15銘柄ウェイトとして扱ってはならない。

旧action schemaとの互換変換は行わず、schema mismatchはfail closedとする。

監査event:

- raw 2D action
- trend family mix
- alpha budget
- trend / alpha weights digest
- composed weights
- HTF-constrained weights
- final target weights

`baseline_only` bundleではPPO推論を呼ばず、identity actionをcomposerへ渡す。これによりRL不採用時にダミーモデルを必要としない。

## 17. 後方互換性

旧 `direct_weights_v1` モデルは研究artifactとして読み込み可能なまま残すが、新release pathでは不適格とする。

```text
--action-mode {direct,baseline-residual}
```

移行:

1. 実装直後はresearch CLIの既定を `direct` に維持する。
2. P0、WF、Serving parityが通った後、research CLI既定を `baseline-residual` へ変更する。
3. release pathは `baseline-residual` または `baseline_only` のみ許可する。

## 18. テスト設計

### 18.1 Unit

- identity actionでcomposer出力がbase trendに一致する。
- fast/base/slow混合がaction端点と一致する。
- trend混合とalpha合成後のgrossが1以下になる。
- alpha gate不合格時にalpha budgetが0になる。
- PPO action headの初期deterministic actionが厳密に0になる。
- HTF neutralが同一desired targetを反復縮小しない。
- HTF方向禁止が逆方向だけを0にする。
- decision aggregationが正しいbar数を進める。
- interval costがtarget変更時に1回だけ計上される。
- non-finite actionを拒否する。
- oracleがGate 2 mandatory setに入らない。

### 18.2 Integration

- identity policyとshadow base trendのequity curveが一致する。
- train / evaluate / serveで同じ入力から同じtargetを生成する。
- save / load後もidentity action同値が維持される。
- action schema mismatch bundleをServingが拒否する。
- frozen alpha artifactがholdout/serveで再fitされない。
- residual gate不合格でtrend-onlyへ縮退する。
- checkpoint評価が10回以上発生する。
- baseline fallback時に `baseline_only` bundleを構築できる。

### 18.3 Synthetic controls

Positive control:

- 観測可能なレジームに応じてfast/base/slowを選ぶとbase trendを改善できる市場
- BまたはDがAを上回る

Negative control:

- base trendが最適で追加actionが期待値を持たない市場
- identity snapshotへfallbackする
- turnoverが暴走しない

Adversarial:

- NaN feature
- all-zero alpha
- 4h trendが閾値近傍で振動
- episode末尾がdecision interval未満
- shadowとhybridの一方だけpre-trade violation
- frozen alpha artifactのfeature order不一致

### 18.4 Regression

- 既存trend_following baselineの数値を維持する。
- direct modeの研究テストを壊さない。
- oracle診断表示を維持する。
- bundle digest、Registry、release eligibility契約を維持する。

## 19. 実装境界

推奨ファイル:

```text
mars_lite/trading/trend_family.py
mars_lite/trading/residual_alpha.py
mars_lite/trading/baseline_residual.py
mars_lite/trading/htf_constraint.py
mars_lite/env/market_execution_core.py
mars_lite/env/baseline_residual_env.py
mars_lite/learning/relative_val_selection.py
mars_lite/eval/relative_evaluation.py
mars_lite/pipeline/gates.py
```

既存 `PortfolioTradingEnv` を巨大な条件分岐で拡張しない。市場進行、book state更新、executionを `market_execution_core` へ抽出し、direct環境とbaseline-residual環境を薄く構成する。

`DecisionPipeline` は次の段階を明示する。

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
- trend family生成失敗: 学習・serve停止
- residual alpha artifact不正: release停止
- researchで明示許可されたalpha無効化: trend-onlyへ縮退しreport記録
- shadow parity mismatch: runtime assertionまたはテスト失敗
- report必須項目欠落: candidate construction拒否
- oracleがmandatory gateへ混入: assertion failure

## 21. 段階導入

### Phase 1: Correctness foundation

- Gate 2 oracle除外
- HTF適用順序修正
- decision aggregation
- relative diagnostics
- identity parity

### Phase 2: Baseline-residual policy

- TrendFamily
- FrozenResidualAlphaProvider
- 2D composer
- zero-initialized residual head
- shadow book
- relative reward / checkpoint selection

### Phase 3: Research validation

- A/B/C/D matrix
- 3+ seeds
- multi-fold
- costs 1x/2x
- decision interval 1/4/8

### Phase 4: Release integration

- ServingBundle schema更新
- Serving parity
- release gate更新
- 5+ seeds
- sealed holdout

## 22. 受入基準

1. identity parityのunit/integrationテストが通る。
2. `decision_every > 1` で1actionにつき1集約rewardだけが返る。
3. HTF neutralが保有を再帰的に縮小しない。
4. Gate 2 mandatory setにoracleが存在しない。
5. 10回以上のcheckpoint評価が行われる。
6. synthetic positive controlでRLがbase trendを改善する。
7. synthetic negative controlでidentity fallbackが選択される。
8. train/eval/serve target parityが通る。
9. frozen alpha artifactがholdout/serveで再fitされない。
10. reportからaction、各weight stage、shadow差、checkpoint理由を追跡できる。
11. 全既存CI gateが通る。
12. 実データdevelopment WFでA/B/C/D比較が完了する。
13. RL構成がAを上回らない場合、RLをProduction候補にしない。

## 23. 最終判断原則

本システムの目的はRLを使用することではなく、同一条件の実行可能baselineより高いリスク調整後価値を再現可能に生むことである。

Baseline-Anchored Residual RLはtrend-followingを当然に置き換えるものではなく、trend-followingへ追加価値を出せるかを厳密に検証する仕組みである。複数fold・複数seed・cost sensitivity・sealed holdoutで追加価値を示せない場合、`baseline_only` のpure base trendを採用し、RL部分を棄却する。
