# Target-Weight Constrained Growth 設計

日付: 2026-07-30

## 1. 背景

現行の `examples/binance-multitimeframe/training-full.json` は、direct target-weight 方策に対して、net log growth、baseline 超過成長、rolling baseline 劣後罰、ドローダウン悪化罰、terminal equity 罰、margin deficit 罰、168 時間 half-life を同時に適用している。

各要素には個別の合理性があるが、合成後の目的は「最終的な複利成長」から外れ、係数依存かつ経路依存になる。また、既存の `training-growth-optimal.json` と `training-constrained-growth.json` は residual action を使うため、target-weight の `training-full.json` と直接比較すると、報酬、gamma、行動空間、BC、アルゴリズムが同時に変わる。

本設計では、現在の本命である direct target-weight 方策を維持したまま、経済目的、安全性、比較実験を分離する。

## 2. 目的

1. 学習目的を、実コスト控除後の期待最終対数資産へ明確化する。
2. 破ってはならない安全条件を、学習成功に依存しない hard constraint として保証する。
3. 平均的に抑えたいリスクと運用コストを、独立した constraint cost として扱う。
4. baseline を報酬 shaping から外し、未見期間の採用基準として使う。
5. PPO と Lagrangian PPO を、同一 action space、encoder、BC、データ、seed、執行条件で比較する。
6. 720 時間の上限を経済的終端ではなく truncation として扱い、継続運用を学習対象にする。

## 3. 非目的

本変更では、CVaR、distributional RL、極値制約、PID-Lagrangian、action space 再設計、encoder 変更、oracle teacher 再設計、baseline 戦略変更、執行モデル全面改修は行わない。これらは本設計の比較基盤を確立した後に独立して評価する。

## 4. 中核設計

### 4.1 方策出力

本命と対照群はすべて `ActionMode.TARGET_WEIGHT` を使用する。方策は銘柄ごとの目標比率を直接出力し、既存の target-weight action contract、pre-trade risk、execution pipeline を通過する。

初期 hard-risk 値は現行 `training-full.json` と揃える。

```text
max_abs_weight = 0.45
max_gross = 1.0
drawdown_start = 0.10
drawdown_stop = 0.20
entry_threshold = 0.10
exit_threshold = 0.03
no_trade_band = 0.05
max_turnover = null
```

報酬比較の段階ではこれらを変更しない。

### 4.2 主報酬

主報酬は実コスト控除後の net log return のみにする。

```text
reward_t = scale * log(V_{t+1} / V_t)
```

`V_t` は手数料、spread、impact、funding、borrow cost、部分約定その他の執行結果を反映した hybrid portfolio value とする。

本命設定では全 weight を暗黙 default にせず明示する。

```json
{
  "scale": 100.0,
  "absolute_growth_weight": 1.0,
  "excess_growth_weight": 0.0,
  "incremental_drawdown_weight": 0.0,
  "baseline_underperformance_weight": 0.0,
  "projection_penalty_weight": 0.0,
  "terminal_equity_weight": 0.0,
  "margin_deficit_weight": 0.0
}
```

割引なしでは、scale 前の累積報酬が次へ一致しなければならない。

```text
sum_t log(V_{t+1} / V_t) = log(V_T / V_0)
```

### 4.3 時間割引と GAE

本命は `gamma = 1.0`、`gae_lambda = 0.95` とする。

早期利益の再投資効果は既に portfolio value の複利推移へ含まれる。`gamma < 1` は複利の表現ではなく、同じ最終資産でも早期利益を追加で優遇する時間選好である。

168 時間 half-life は削除せず、比較用アブレーションとして残す。`gamma = 1.0` の profile では `discount_half_life_hours` を指定してはならない。

`gae_lambda = 0.97` は本設計の勝敗確定後に独立して比較する。gamma と GAE を同時に変更しない。

## 5. 安全性の三層分離

### 5.1 Hard safety

次は学習アルゴリズムへ委ねず、環境が常に強制する。

- 1 銘柄当たり最大 absolute weight
- 最大 gross exposure
- 証拠金、利用可能資金、最大 leverage
- 非有限値と不正 action の fail-closed
- 取引所最小数量、刻み、価格制約
- emergency flatten
- drawdown stop
- minimum equity termination
- liquidation と insolvency accounting

既存の `PreTradeRisk` と execution/accounting を canonical enforcement point とする。Lagrangian multiplier をゼロへ固定しても hard safety は破れないことをテストする。

### 5.2 Soft constraints

次は完全禁止ではなく、Lagrangian が平均予算を管理する。

- `drawdown_excess`
- `daily_turnover`
- `execution_cost_fraction`
- `gross_exposure_request_excess`

cost は reward total へ加算せず、reward advantage と cost advantage を分離する。

### 5.3 Catastrophic event gates

次は平均予算を満たすだけでは production 合格にしない。

- `forced_liquidation_event`
- `margin_deficit_fraction > 0`
- insolvency
- unexpected drawdown stop
- hard safety violation

通常の未見評価では一件でも発生すれば NO-GO とする。fail-safe を意図的に発火させる fault-injection test は別集計とする。

## 6. Baseline の役割

baseline は学習報酬へ混ぜない。次にのみ使用する。

- paired unseen net log growth comparison
- fold ごとの相対成績
- deployment gate
- regression detection
- market regime 別診断

`shadow` path は比較・診断のため維持する。ただし baseline 関連 weight がゼロなら、shadow return や shadow history を変更しても reward total が変化してはならない。

## 7. Episode termination と truncation

720 時間は継続運用をサンプリングする訓練窓であり、経済的な投資終了ではない。

本命 profile は次を保証する。

- `liquidate_on_end = false`
- time limit は `truncated = true`
- insolvency、minimum equity、明示的 forced close は `terminated = true`
- truncation 時の final observation を保持する
- truncation では critic value を bootstrap する
- true termination では bootstrap しない
- time limit に人工的な清算 return を追加しない

SB3 adapter まで含めて統合テストを置き、time limit 直前だけポジションを閉じる人工的方策を誘発しない。

## 8. 比較 profile

すべて次を共通化する。

- target-weight action
- dataset、fold、episode sampling
- observation encoder と architecture parameters
- oracle BC teacher、epochs、validation gate
- rollout、batch、epochs、learning-rate schedule
- environment、risk、execution config
- seeds と timesteps

### G1-PPO

```text
target_weight
net log growth only
gamma = 1.0
algorithm = ppo
hard safety enabled
```

純粋成長の必須対照群。

### G1-Lagrangian

```text
target_weight
net log growth only
gamma = 1.0
algorithm = lagrangian_ppo
hard safety enabled
soft constraint budgets enabled
```

本命候補。

### D168-Lagrangian

```text
target_weight
net log growth only
168-hour discount half-life
algorithm = lagrangian_ppo
hard safety enabled
soft constraint budgets enabled
```

時間選好アブレーション。

### Legacy Full

既存 `training-full.json` は互換性と研究比較のため残すが、production default にはしない。

既存 residual growth profiles も削除せず、residual-policy research controls として明記する。

## 9. Config contract

追加または強化する validation:

1. pure-growth profile では shaping、terminal、margin weight が全てゼロでなければ拒否する。
2. `gamma = 1.0` と `discount_half_life_hours` の同時指定を拒否する。
3. baseline shaping が有効なら `require_full_reward_preroll = true` を要求する。
4. `baseline_tolerance = 0` かつ `baseline_progressive_power > 1` を拒否する。
5. target-weight profile で residual-only field が有効なら拒否する。
6. Lagrangian profile では cost names、budgets、learning rates、EMA betas、multiplier limits の長さを一致させる。
7. production profile は catastrophic event を平均 budget だけで許容できない。
8. reward/profile identity を checkpoint manifest、resume、serving loader で検証する。

## 10. 実装範囲

最低限、次を追加・変更する。

- target-weight G1-PPO training profile
- target-weight G1-Lagrangian training profile
- target-weight D168-Lagrangian training profile
- 対応する walk-forward profiles
- reward/profile validation
- reward telescoping tests
- target-weight constraint alignment tests
- truncation bootstrap integration tests
- comparison report fields
- automatic production gate
- docs と実行手順

## 11. Test contract

### Reward

- scale 前の累積 pure reward が `log(final_equity / initial_equity)` へ一致する。
- terminal loss を二重計上しない。
- baseline weight がゼロなら shadow path は reward に影響しない。
- drawdown と margin cost は reward total に混ざらない。
- execution cost、funding、borrow cost を含む net return を使う。

### Validation

- zero tolerance と progressive power greater than one を拒否する。
- gamma-one profile に half-life を指定すると拒否する。
- cost vector の長さ不一致を拒否する。
- target-weight profile に residual-only controls を指定すると拒否する。

### Safety

- max_abs_weight と max_gross を必ず守る。
- drawdown stop では turnover limit を上書きして flatten できる。
- invalid/non-finite target は fail-closed する。
- multiplier がゼロでも hard safety を破れない。

### Truncation

- time limit と insolvency の flag を区別する。
- `liquidate_on_end = false` では time limit に清算 return を加えない。
- final observation が rollout へ渡る。
- truncation では bootstrap し、termination では bootstrap しない。

### Lagrangian

- reward advantage と cost advantage を分離する。
- multiplier は rollout 中に固定する。
- completed episode aggregation と elapsed-time weighting を維持する。
- multiplier saturation、constraint residual sign flip、penalty-to-reward ratio を report する。

## 12. 実験手順

### Stage 0: Contract and smoke

各 profile で CPU と CUDA smoke を行う。

合格条件:

- finite reward、value、advantage、loss
- checkpoint 保存・resume
- architecture、reward、constraint digest 一致
- truncation semantics test 合格
- hard safety violation ゼロ

### Stage 1: Screening

```text
1 fold x 3 seeds x 131,072 timesteps
```

G1-PPO、G1-Lagrangian、D168-Lagrangian を同条件で実行する。

即時失格:

- non-finite value、advantage、loss
- checkpoint/resume failure
- hard safety violation
- catastrophic event
- critic divergence

no-trade policy は自動失格にしない。cash および baseline より劣り、かつ action variance と gross exposure が崩壊している場合に限り学習崩壊と判定する。

### Stage 2: Full evaluation

原則として次の二つを必ず評価する。

```text
G1-PPO
G1-Lagrangian
```

```text
6 folds x 3 seeds x 524,288 timesteps
nested walk-forward
```

D168-Lagrangian は screening で、G1-Lagrangian より高い paired baseline difference を全 3 seed で示し、catastrophic event がゼロの場合のみ full evaluation へ追加する。

## 13. 共通評価指標

異なる gamma や shaping の episode reward は直接比較しない。

主要指標:

- execution-adjusted unseen net log growth
- paired baseline net log growth difference

副指標:

- fold ごとの growth
- seed ごとの growth
- maximum drawdown
- drawdown excess area
- daily turnover
- execution cost fraction
- forced liquidation、margin deficit、insolvency rates
- multiplier saturation rate
- constraint residual stability

stress tests:

- 2x execution cost
- 3x execution cost
- spread widening
- slippage worsening
- latency/rejection worsening
- partial-fill degradation

## 14. 定量 Production Gate

候補は次を全て満たした場合だけ GO とする。

1. 18 個の fold-seed cell における unseen net log growth の中央値が正。
2. 18 cell の paired baseline difference の中央値が正。
3. paired bootstrap 95% confidence interval の下限が 0 を上回る。サンプル単位は fold を維持した cluster bootstrap とする。
4. 6 fold 中少なくとも 4 fold で、3 seed 中央値の paired baseline difference が正。
5. 3 seed 全てで fold 中央値が 0 以上、かつ少なくとも 2 seed で正。
6. forced liquidation、margin deficit、insolvency、hard safety violation が全未見 episode でゼロ。
7. 各 soft constraint は全 fold の point estimate で budget 以下、かつ pooled one-sided 95% upper confidence bound が budget 以下。
8. 2x cost stress でも paired baseline difference の中央値が正。
9. 3x cost stress では unseen net log growth の中央値が 0 以上で、catastrophic event がゼロ。
10. checkpoint resume と serving loader が architecture、reward、constraint digest を検証する。

cluster bootstrap の反復回数、seed、欠損処理は report contract へ固定し、再現可能にする。

## 15. PPO と Lagrangian の選択規則

安全 gate を満たした候補だけを比較する。

1. G1-Lagrangian が production gate を満たさず、G1-PPO が満たす場合は G1-PPO を採用する。
2. G1-PPO が満たさず、G1-Lagrangian が満たす場合は G1-Lagrangian を採用する。
3. 両方が満たす場合は、paired unseen net log growth difference の cluster-bootstrap 95% confidence interval を比較する。
4. Lagrangian 対 PPO の growth 差が統計的に正なら G1-Lagrangian を採用する。
5. 差の confidence interval が 0 を含む場合は、実装と運用が単純な G1-PPO を採用する。
6. Lagrangian の growth が統計的に低い場合は、soft-risk 改善だけを理由に production default にしない。必要なら別の低リスク運用 profile として保持する。
7. どの RL profile も baseline gate を満たさない場合は RL を production 採用しない。

## 16. 完了条件

- 3 つの target-weight training profile と対応する walk-forward profile が存在する。
- config contract が不正な目的混合を fail-closed する。
- reward、hard safety、soft constraints、catastrophic gates が分離される。
- time-limit truncation bootstrap が統合テストで保証される。
- comparison report と production gate が自動化される。
- CPU/CUDA smoke が成功する。
- full test、Ruff、MyPy、import-linter が成功する。
