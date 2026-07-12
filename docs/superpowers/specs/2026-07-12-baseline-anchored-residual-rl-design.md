# Baseline-Anchored Residual RL 再設計仕様

日付: 2026-07-12
状態: 設計承認済み・実装前
対象: `shuntatsu/trade_rl`

## 1. 背景

現行のポートフォリオPPOは15銘柄のターゲットウェイトを直接出力する。実データ検証では、弱い予測信号、実執行コスト、turnover罰則、no-trade band、検証モデル選択が組み合わさり、復元モデルが完全flatとなった。一方、同一OOS期間のルールベース `trend_following` は正の収益とSharpeを示した。

問題は単純な学習step不足だけではない。方向性シグナルの発見、銘柄配分、回転抑制、リスク制御、執行後処理を1つのPPOへ同時に負わせ、ゼロ行動をflatへ対応させる現行構造は、弱いアルファ下で保守崩壊しやすい。

本設計は、因果的なtrend戦略を基準方策として固定し、RLの責務を「基準方策に対する追加価値の学習」に限定する。

## 2. 目的

1. identity actionを純base-trendに対応させる。
2. RLが追加価値を学習できない場合は、base-trendへ厳密にfallbackする。
3. RLの追加価値を、同一期間・同一コスト・同一リスク制約のshadow baselineとの差分で測定する。
4. `decision_every > 1` でactionとrewardの対応を壊さない。
5. 4h階層ゲートを冪等にし、現在保有を反復縮小しない。
6. Gate 2から到達不能なoracleを除外する。
7. 学習、評価、Serving Planeで同一のaction-to-target契約を使う。
8. action、HTF、post-process、checkpoint選択のどこで性能が変化したかを追跡可能にする。
9. RL不採用時にも、独立した基準戦略のrelease可否を正しく判定できるようにする。

## 3. 非目的

本フェーズでは以下を行わない。

- 15次元の銘柄別自由残差をRLへ再導入しない。
- carry、crowding、複数戦略のメタアロケータを同時実装しない。
- Gate閾値を今回の単一holdoutに合わせて緩和しない。
- maker約定率、板キュー、逆選択を新たにモデル化しない。
- 単一seedまたは単一splitの好結果でProduction GOにしない。
- Serving中に予測器を再学習しない。
- 既存Control Plane / Serving Planeの境界を崩さない。

## 4. 採用アプローチ

### 4.1 不採用: 現行15次元PPOの調整

変更量は少ないが、過去のアブレーションでflatと過剰売買の二極化が確認されている。予測・配分・執行責務の過密も残るため不採用とする。

### 4.2 採用: Baseline-Anchored Residual RL

base-trendを構造的基準にし、RLは低次元のtrend timing選択と残差アルファ配分だけを学ぶ。identity action同値、比較可能性、診断容易性が高いため採用する。

### 4.3 将来候補: 複数スリーブのメタアロケータ

本設計でRLが単一の有効baselineに追加価値を出せた後に検討する。

## 5. 主要不変条件

### 5.1 Identity action同値

identity actionは `[0.0, 0.0]` とする。同一の市場状態、ポートフォリオ状態、コスト設定、リスク設定においてidentity actionを与えたhybridは、base-trendの提案ターゲット、最終執行ウェイト、コスト、損益と一致しなければならない。

許容誤差:

- ウェイト: `atol <= 1e-8`
- コスト、1step損益: `atol <= 1e-10`

### 5.2 Oracle非必須

`oracle_dp` と `oracle_ic*` は診断専用であり、いかなるrelease PASS条件にも含めない。

### 5.3 同一執行条件

hybridとshadow baselineは、同じ価格、funding、fee、spread、impact、decision interval、HTF制約、post-processing、pre-trade riskを使用する。両bookは独立したportfolio stateを持つが、設定と実装は共有する。

### 5.4 Train / Eval / Serve一致

trend生成、alpha予測、action composition、HTF proposal constraint、post-processingの順序は、学習環境、OOS評価、Serving Planeで共通実装を使う。

### 5.5 非意思決定actionの不存在

`decision_every > 1` の場合、PPOは意思決定間隔ごとに1 transitionだけ受け取る。無視されるactionを持つtransitionを生成しない。

### 5.6 明示的fallback

許可されるfallbackは次の2つだけである。

- residual alpha gate不合格時のtrend-only mode
- 有効なRL checkpointが無い場合のidentity policy / `baseline_only`

silent fallbackは行わない。

## 6. 全体構造

```text
FeatureSet + actual portfolio state
        |
        +--> TrendFamily(fast/base/slow) --------+
        |                                        |
        +--> FrozenResidualAlphaProvider -- gate +--> BaselineResidualComposer
        |                                        |          |
        +--> PPO action [trend_mix, alpha] -------+          v
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

## 7. TrendFamily

### 7.1 口座状態からの独立

TrendFamilyは実口座またはhybridのcurrent weightsを入力に使わない。既存 `trend_following_strategy(fs, t, w)` のように、非リバランス時に渡された `w` を返す構造は、hybrid保有を基準trendへ混入させるため使用しない。

TrendFamilyは次の純粋契約を持つ。

```text
targets(feature_set, timestamp) -> {fast, base, slow}
```

各targetは価格履歴と絶対時刻だけで決まり、呼び出し元のportfolio stateに依存しない。

### 7.2 絶対時刻リバランス

リバランス判定はslice内indexの `t % N` ではなく、UTC bar close timestampを使う。これによりtrain split、holdout slice、Servingで同じ時刻のtargetが一致する。

```text
slot = floor(bar_close_unix / base_bar_seconds)
rebalance_slot = floor(slot / rebalance_every) * rebalance_every
```

時刻tのtargetは、直近rebalance slotで計算したsignalを保持したものとする。

### 7.3 初期設定

- fast lookback: 24 base bars
- base lookback: 48 base bars
- slow lookback: 96 base bars
- rebalance_every: 24 base bars
- gross cap: 1.0
- allow_short: true

各候補は時系列モメンタムの符号と強度から生成し、gross 1以下へ射影する。

既存index基準の `trend_following` は `trend_following_v1` として診断用に残せるが、release比較の基準は絶対時刻契約の `base_trend_v2` とする。過去の+26.34%はv1の結果であり、v2で再測定する。

## 8. FrozenResidualAlphaProvider

責務:

- 市場中立の相対アルファ候補を生成する。
- Gate不合格時はゼロベクトルを返す。
- 推論時に未来情報を使わない。
- Serving中に再学習しない。

初期方式:

- target: `cs_demean`
- model: `gbm`
- fold評価では各foldのtrain区間だけでfit
- final artifactはdevelopment区間のうちラベル確定済み領域だけでfit
- fit cutoffは `development_end - horizon`
- fitted model、特徴順、前処理、fit cutoff、学習データidentityをServingBundleへ保存
- holdout評価とServingは同じfitted artifactでpredict
- 出力はクロスセクショナル中心化後、gross 1以下へ射影

Gate条件:

- mean OOS rank IC >= 0.02
- positive fold ratio >= 0.6
- stability t-stat >= 1.0

Gate不合格時はalpha出力とalpha actionを無効化し、trend-mix-only modeへ縮退する。

## 9. BaselineResidualComposer

RL action spaceは2次元連続値 `[-1, 1]^2` とする。

```text
action[0] = a_trend_mix
action[1] = a_alpha
```

### 9.1 Trend timing action

`a_trend_mix` はレバレッジ倍率ではなく、gross上限内でfast / base / slowを選ぶ連続混合係数である。

```text
if a_trend_mix >= 0:
    w_trend = (1 - a_trend_mix) * w_base + a_trend_mix * w_fast
else:
    m = abs(a_trend_mix)
    w_trend = (1 - m) * w_base + m * w_slow
```

- `0`: base
- `1`: fast
- `-1`: slow

各成分の凸結合なのでgrossは1以下を保つ。gross上限1.0のまま倍率を増やして射影で打ち消す無効actionを作らない。

### 9.2 Residual alpha action

```text
alpha_budget = 0.30 * a_alpha
proposal = (1 - abs(alpha_budget)) * w_trend + alpha_budget * w_alpha
proposal = project_gross(proposal, max_gross=1.0)
```

- `a_alpha = 0`: alpha無し
- `a_alpha > 0`: 予測方向へ最大30%配分
- `a_alpha < 0`: 反対方向へ最大30%配分

負方向を許すのは、identity actionを0に保ちつつaction領域の両側を意味のある選択にするためである。OOSで有効でなければcheckpoint選択とGateで棄却する。alpha gate不合格時は `alpha_budget=0` を強制する。

identity action `[0, 0]` では `proposal=w_base` となる。

## 10. Residual policy初期化とensemble

### 10.1 初期化

baseline-residual modeではBC warm startを使用しない。PPOのaction headのweightとbiasを0で初期化し、初期deterministic actionを厳密に `[0, 0]` とする。value networkは通常初期化を維持する。

学習開始前にidentity parameter snapshotを保存する。deterministic評価はidentity action、stochastic学習はidentity周辺を探索する。

### 10.2 Release ensemble

researchではseed別分布を報告する。release候補のRL policyは5 seed以上のaction-level ensembleとする。

```text
ensemble_action = mean(deterministic_action_by_seed)
```

- 各seedはdevelopment上で独立学習・checkpoint選択する。
- action平均後にcomposerへ1回だけ渡す。
- seed間action分散をdisagreementとしてreportする。
- v1ではdisagreementによる追加縮小を行わず、診断だけに使う。
- ensemble全体をWF、cost 2x、sealed holdoutで評価する。

単一seedをholdout成績で選ばない。

## 11. HTFProposalConstraint

HTF制約をstateful post-processingの前へ移す。

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

## 12. PortfolioPostProcessor

初期実験:

- `no_trade_band = 0.0`
- `lambda_turnover = 0.0`
- EMAは有効
- 実fee / spread / impactは有効
- max gross / max symbol weightは有効
- target vol / DD deriskは設定どおり有効

回転抑制はtrendの低頻度更新、RL decision interval、EMA、実コストで行う。追加turnover penaltyは、実コストだけでは過剰売買が抑えられないことを複数seedで確認した場合のみ、単独変数として再導入する。

## 13. Decision Interval Aggregation

### 13.1 新契約

1回の `env.step(action)` は最大 `decision_every` 本のbase barを内部で進める。

1. decision barでactionを受け取る。
2. trend / alpha / composer / HTF / post-processを1回実行する。
3. targetをinterval内で保持する。
4. 各base barで価格損益、fundingを計算する。
5. 取引コストはtarget変更時に1回だけ計上する。
6. interval内のlog returnを集計する。
7. interval終了時のobservationと1つのrewardを返す。

エピソード末尾で残りbarが不足する場合は、存在するbarだけを進める。

`info`:

- `bars_advanced`
- `interval_gross_return`
- `interval_cost`
- `interval_funding`
- `interval_net_return`
- `decision_step_index`

### 13.2 Annualization

- base-bar return系列: base bars/year
- decision-interval系列: base bars/year / 実効平均bars_advanced

使用系列とannualization factorをreportへ記録する。

### 13.3 学習予算

PPO timestepsはdecision transitionsを数える。異なるdecision intervalの比較ではPPO update数を固定し、処理base-bar総数とdataset pass相当数もreportする。

## 14. Shadow baselineと報酬

### 14.1 Shadow book

shadowは常にbase trend targetを使用する。hybridとは独立したweights、portfolio value、peak、drawdown、post-process stateを持つ。

共有条件:

- FeatureSet
- decision interval
- HTF constraint
- post-processing設定
- execution cost model
- funding
- pre-trade hard limits

### 14.2 報酬

```text
r_hybrid = log(V_hybrid_after / V_hybrid_before)
r_shadow = log(V_shadow_after / V_shadow_before)
reward = reward_scale * (r_hybrid - r_shadow)
```

追加turnover penaltyは加えない。実コスト差は相対returnへ含まれる。

### 14.3 リスク

hard risk limitsは実行前制約として扱う。初期フェーズではsoft risk penaltyを加えない。

複数seedで高収益・過大DDが一貫して観測された場合に限り、downside excess、drawdown excess、volatility excessのいずれか1つをdevelopmentデータで試験する。

### 14.4 失敗条件

次の場合はエピソードを終了し、明示的失敗rewardを返す。

- portfolio value <= 1e-6 * initial capital
- NaN / Inf in action, proposal, target, cost, reward
- pre-trade hard constraint violation

shadowだけが失敗した場合も評価不能として終了する。

## 15. 観測設計

Servingで再現可能な状態だけをpolicy observationへ含める。

既存のper-symbol MTF特徴とactual portfolio stateを維持し、per-symbolブロックへ追加する。

- fast trend weight
- base trend weight
- slow trend weight
- frozen residual alpha weight

グローバルブロックへ追加する。

- fast / base / slow trend gross
- residual alpha gross

training-onlyでpolicy observationへ含めないもの:

- shadow portfolio value / drawdown
- hybrid minus shadow cumulative return
- validation score

ServingBundleのobservation schema versionを更新し、旧bundleとの誤読をfail closedにする。

## 16. PPOラン種別

`one_rollout_steps = n_envs * n_steps` とする。

### Smoke

- 最低updates: 5
- shape、finite、保存復元、経路確認のみ
- release evidence不可

### Research

- 最低updates: 50
- checkpoint評価10回以上
- 3 seeds以上

### Release candidate

- 最低updates: 100
- development WF、cost sensitivity、sealed holdout必須
- 5 seeds以上のensemble

```text
--run-tier {smoke,research,release}
required_timesteps = required_updates * n_envs * n_steps
```

不足するresearch / release runはerrorにする。

## 17. Checkpoint Selection

### 17.1 評価頻度

```text
n_eval_targets = 10
eval_freq = ceil_to_multiple(total_timesteps / n_eval_targets, one_rollout_steps)
```

学習開始時、各評価点、学習終了時を候補にし、同じstepを重複評価しない。

### 17.2 Validation score

validation期間を最低3個の連続blockに分ける。

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

採用条件:

- median block excess > 0
- positive block ratio >= 0.5

条件を満たすcheckpointが無いseedはidentity snapshotへ復元し、`baseline_fallback=true` と記録する。全seedがfallbackの場合はRL artifactではなく `baseline_only` を候補とする。

## 18. Gate設計

### 18.1 Trend Development Gate

release比較に使う正確な `base_trend_v2` 経路、すなわちHTF、post-process、decision interval、実コスト込みで判定する。

全条件:

- cost 1x median total return > 0
- cost 1x positive fold ratio >= 0.6
- cost 2x median total return >= 0
- cost 2x positive fold ratio >= 0.5

方向性trend t-statは診断へ残すが、取引Gateの代替条件にはしない。不合格ならResidual RL学習を開始しない。

### 18.2 Residual Alpha Gate

Section 8のIC条件を満たす場合だけalpha sleeveを有効化する。不合格時はtrend-mix-only modeへ縮退する。

### 18.3 RL Development Gate

fold x seedおよび最終ensembleで判定する。

- median excess return > 0
- hybridがshadowを上回るrun比率 >= 0.6
- cost 2x median excess return >= 0
- hybrid median maxDD <= shadow median maxDD + 0.05

### 18.4 Residual Gate 2: Sealed Holdout

RLを含む候補に適用する。

- hybrid total return > 0
- hybrid total return > shadow total return
- hybrid maxDD <= shadow maxDD + 0.05
- moving-block paired bootstrapによるexcess returnの片側p-value < 0.05

bootstrap block lengthとseedは事前設定し、holdout結果を見て変更しない。

### 18.5 Baseline-Only Release Gate

RLが棄却された場合、`baseline_only` を別契約で判定する。自分自身を上回ることは要求しない。

- Trend Development GateがPASS
- sealed holdout total return > 0
- sealed holdout maxDDがrelease risk policy内
- cost 2x sealed holdout total return >= 0
- moving-block bootstrapでbase-trend return > 0の片側p-value < 0.05

このGateをPASSしない場合は、RLもbaselineもrelease候補を作らない。

### 18.6 診断専用baseline

次はGate PASS条件を変更しない。

- flat以外のequal_weight_bh、inverse_vol、cross_momentum
- `trend_following_v1`
- trend_v2、carry、crowding
- oracle_dp、oracle_ic*

flatはbaseline-onlyのゼロ収益比較に使うが、oracleは使わない。

## 19. 実験マトリクス

```text
A: pure base_trend_v2
B: TrendFamily + RL trend mix
C: base_trend_v2 + fixed residual alpha budget +0.15
D: TrendFamily + RL trend mix + RL alpha budget
```

- research: 3 seeds以上
- release: 5 seeds以上のaction ensemble
- costs: 1x / 2x
- decision interval候補: 1 / 4 / 8 base bars

候補選択はdevelopmentデータだけで行い、sealed holdoutで変更しない。

ここで「勝つ」とは、比較対象に対するpaired median excess returnが正で、該当development risk条件も満たすことを意味する。

- BがAに勝てなければRL trend mixを採用しない。
- CがAに勝てなければresidual alphaを採用しない。
- DがBとCの良い方に勝てなければ複合RLを採用しない。
- どのRL構成もAに勝てなければ、Baseline-Only Release Gateだけを評価する。

## 20. レポート契約

### Identity

- git SHA、dataset identity、symbols、date range
- feature / observation / action schema version
- run tier、seed、ensemble members
- PPO update count、decision transitions、base bars advanced

### Gates

- trend development gate
- residual alpha gate
- RL development gate
- residual Gate 2またはbaseline-only gate
- diagnostic-only baseline一覧

### Checkpoints

- identity score、evaluation step、block scores
- median excess、positive block ratio
- selected step、baseline fallback

### Actions

- `a_trend_mix` mean/std/min/max
- `a_alpha` mean/std/min/max
- fast/base/slow mix比率
- alpha budget mean/std/min/max
- contrarian alpha利用率
- seed disagreement

### Weight stages

- fast/base/slow trend gross
- residual alpha gross
- composed proposal gross
- HTF-constrained gross
- post-process gross
- executed gross
- desired / executed turnover
- tracking ratio、freeze ratio
- HTF zeroed / neutral-scaled fraction

### Performance

hybridとshadowの両方:

- total / annualized return
- Sharpe / Sortino
- max drawdown
- turnover / cost / funding
- n trades

paired:

- excess total / log return
- excess information ratio
- moving-block bootstrap confidence interval / p-value

## 21. ServingBundleとServing Plane

ServingBundleへ追加する。

- action schema: `baseline_residual_v1`
- policy mode: `ppo_residual_ensemble` または `baseline_only`
- trend family configとabsolute rebalance contract
- frozen residual alpha artifact、特徴順、fit cutoff、gate結果
- composer config
- HTF proposal constraint config
- decision aggregation config
- observation schema version
- shadow baseline identity

Serving Planeは2次元actionを直接15銘柄ウェイトとして扱ってはならない。旧action schemaとの互換変換は行わず、schema mismatchはfail closedとする。

`baseline_only` はPPO推論を呼ばずidentity actionをcomposerへ渡す。

監査event:

- raw 2D actionまたはidentity marker
- trend family mix、alpha budget
- trend / alpha weights digest
- composed、HTF-constrained、final target weights

## 22. 後方互換性

旧 `direct_weights_v1` は研究artifactとして残すが、新release pathでは不適格とする。

```text
--action-mode {direct,baseline-residual}
```

移行:

1. 実装直後はresearch CLIの既定を `direct` に維持する。
2. P0、WF、Serving parity後にresearch既定を `baseline-residual` へ変更する。
3. release pathは `ppo_residual_ensemble` または `baseline_only` のみ許可する。

## 23. テスト設計

### Unit

- identity actionでcomposer出力がbase trendに一致する。
- TrendFamilyがcurrent portfolio weightsに依存しない。
- 同一timestampでslice位置に関係なく同じtrend targetを返す。
- fast/base/slow混合がaction端点と一致する。
- alpha合成後grossが1以下になる。
- alpha gate不合格時にalpha budgetが0になる。
- PPO action headの初期deterministic actionが厳密に0になる。
- HTF neutralが同一desired targetを反復縮小しない。
- decision aggregationが正しいbar数を進め、costを1回だけ計上する。
- non-finite actionを拒否する。
- oracleがmandatory gateへ入らない。

### Integration

- identity policyとshadow base trendのequity curveが一致する。
- train / evaluate / serveで同じ入力から同じtargetを生成する。
- save / load後もidentity同値が維持される。
- frozen alpha artifactがholdout/serveで再fitされない。
- action / observation schema mismatchをServingが拒否する。
- checkpoint評価が10回以上発生する。
- 全seed fallback時に `baseline_only` を構築できる。
- baseline-only gateとresidual Gate 2が混同されない。

### Synthetic controls

Positive:

- 観測可能なレジームに応じfast/base/slowを選ぶとbaseを改善できる市場
- BまたはDがAを上回る

Negative:

- baseが最適で追加actionが期待値を持たない市場
- identityへfallbackしturnoverが暴走しない

Adversarial:

- NaN feature、all-zero alpha
- HTF閾値近傍の振動
- episode末尾がdecision interval未満
- shadowとhybridの一方だけpre-trade violation
- frozen alpha feature order不一致
- timestamp alignment不一致

Regression:

- `trend_following_v1` は診断用として既存数値契約を維持する。
- direct mode研究テストを壊さない。
- bundle digest、Registry、release eligibility契約を維持する。

## 24. 実装境界

```text
mars_lite/trading/trend_family.py
mars_lite/trading/residual_alpha.py
mars_lite/trading/baseline_residual.py
mars_lite/trading/htf_constraint.py
mars_lite/env/market_execution_core.py
mars_lite/env/baseline_residual_env.py
mars_lite/learning/relative_val_selection.py
mars_lite/learning/residual_ensemble.py
mars_lite/eval/relative_evaluation.py
mars_lite/pipeline/gates.py
```

既存 `PortfolioTradingEnv` を巨大な条件分岐で拡張しない。市場進行、book state更新、executionを共通coreへ抽出し、direct環境とbaseline-residual環境を薄く構成する。

`DecisionPipeline` は次の段階を明示する。

```text
compose proposal
-> constrain proposal
-> stateful post-process
-> risk verify
```

各段階の入力・出力をdiagnosticsへ残す。

## 25. エラー処理

- action dimension不一致: 即時error
- timestamp / schema version不一致: bundle拒否
- trend family生成失敗: 学習・serve停止
- residual alpha artifact不正: release停止
- researchで明示許可されたalpha無効化: trend-onlyへ縮退しreport記録
- shadow parity mismatch: runtime assertionまたはテスト失敗
- report必須項目欠落: candidate construction拒否
- oracleがmandatory gateへ混入: assertion failure

## 26. 段階導入

### Phase 1: Correctness foundation

- Gate 2 oracle除外とbaseline-only gate分離
- absolute-time TrendFamily
- HTF適用順序修正
- decision aggregation
- identity parityとrelative diagnostics

### Phase 2: Baseline-residual policy

- FrozenResidualAlphaProvider
- 2D composer
- zero-initialized residual head
- shadow book
- relative reward / checkpoint selection
- seed ensemble

### Phase 3: Research validation

- A/B/C/D matrix
- 3+ seeds
- multi-fold
- costs 1x/2x
- decision interval 1/4/8

### Phase 4: Release integration

- ServingBundle schema更新
- Serving parity
- 5+ seed ensemble
- sealed holdout
- residualまたはbaseline-only release判定

## 27. 受入基準

1. identity parityのunit/integrationテストが通る。
2. TrendFamilyがportfolio stateとslice indexに依存せず絶対時刻で一致する。
3. `decision_every > 1` で1actionにつき1集約rewardだけが返る。
4. HTF neutralが保有を再帰的に縮小しない。
5. mandatory gateにoracleが存在しない。
6. baseline-only gateとresidual Gate 2が分離される。
7. 10回以上のcheckpoint評価が行われる。
8. synthetic positive controlでRLがbaseを改善する。
9. synthetic negative controlでidentity fallbackが選択される。
10. train/eval/serve target parityが通る。
11. frozen alpha artifactがholdout/serveで再fitされない。
12. reportからaction、weight stages、shadow差、checkpoint理由を追跡できる。
13. 全既存CI gateが通る。
14. 実データdevelopment WFでA/B/C/D比較が完了する。
15. RLがAを上回らない場合、RLをProduction候補にしない。
16. baseline-onlyも専用Gateを通らない場合、候補を作成しない。

## 28. 最終判断原則

目的はRLを使用することではなく、同一条件の実行可能baselineより高いリスク調整後価値を再現可能に生むことである。

Baseline-Anchored Residual RLはbase-trendを当然に置き換えるものではない。複数fold・複数seed・cost sensitivity・sealed holdoutで追加価値を示せない場合はRLを棄却する。base-trend自体も専用release gateを通らない場合は、Registry候補を作らない。
