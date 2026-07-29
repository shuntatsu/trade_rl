# Target-Weight Constrained Growth 設計

日付: 2026-07-30

## 1. 背景

現行の `examples/binance-multitimeframe/training-full.json` は、direct target-weight 方策に対して次の要素を同一報酬へ混合している。

- 実コスト控除後の毎ステップ対数資産成長
- shadow baseline に対する超過成長
- rolling baseline 劣後の拡大罰
- ドローダウン悪化罰
- terminal equity 罰
- margin deficit 罰
- 168 時間 half-life による時間割引

各要素は個別には合理性があるが、合成後の目的は「最終的な複利成長」から外れ、係数依存かつ経路依存になる。また、既存の `training-growth-optimal.json` と `training-constrained-growth.json` は residual action を使うため、target-weight の `training-full.json` と直接比較すると、報酬、gamma、行動空間、BC、アルゴリズムが同時に変わる。

この設計では、現在の本命である direct target-weight 方策を維持したまま、経済目的、安全性、比較実験を分離する。

## 2. 目的

本設計の目的は次のとおり。

1. 学習目的を、実コスト控除後の期待最終対数資産へ明確化する。
2. 破ってはならない安全条件を、学習成功に依存しない hard constraint として保証する。
3. 平均的に抑えたいリスクと運用コストを、独立した constraint cost として扱う。
4. baseline を報酬 shaping から外し、未見期間の採用基準として使う。
5. PPO と Lagrangian PPO を、同一 action space、encoder、BC、データ、seed、執行条件で比較できるようにする。
6. 720 時間のエピソード上限を経済的終端ではなく truncation として扱い、継続運用を学習対象にする。

## 3. 非目的

本変更では次を行わない。

- CVaR、distributional RL、極値制約の導入
- Lagrangian dual update の PID 化
- action space の再設計
- observation encoder の変更
- oracle teacher の再設計
- 執行モデルそのものの全面改修
- baseline 戦略の変更

これらは、本設計による比較基盤を確立した後に独立して評価する。

## 4. 中核となる設計判断

### 4.1 方策出力

本命と対照群は、どちらも `ActionMode.TARGET_WEIGHT` を使用する。

各意思決定時点で、方策は銘柄ごとの目標比率を直接出力する。既存の target-weight action contract、pre-trade risk、execution pipeline は維持する。

### 4.2 主報酬

主報酬は実コスト控除後の net log return のみにする。

```text
reward_t = scale * log(V_{t+1} / V_t)
```

`V_t` は手数料、スリッページ、funding、borrow cost、部分約定その他の執行結果を反映した hybrid portfolio value とする。

本命設定では次を明示的にゼロへ固定する。

```json
{
  "absolute_growth_weight": 1.0,
  "excess_growth_weight": 0.0,
  "incremental_drawdown_weight": 0.0,
  "baseline_underperformance_weight": 0.0,
  "projection_penalty_weight": 0.0,
  "terminal_equity_weight": 0.0,
  "margin_deficit_weight": 0.0
}
```

これにより、割引なしの場合は累積報酬が次へ一致する。

```text
sum_t log(V_{t+1} / V_t) = log(V_T / V_0)
```

### 4.3 時間割引

本命は `gamma = 1.0` とする。

理由は、早期に得た利益の再投資効果が既に portfolio value の複利推移へ含まれているためである。`gamma < 1` は複利の表現ではなく、同じ最終資産でも早期利益を追加で優遇する別の時間選好になる。

168 時間 half-life は削除せず、比較用アブレーションとして維持する。

`gamma = 1.0` の設定では `discount_half_life_hours` を指定してはならない。割引設定では既存の real-time half-life validation を使用する。

### 4.4 GAE

初期値は `gae_lambda = 0.95` を維持する。

`gamma` は目的関数を決め、`gae_lambda` は advantage 推定のバイアス・分散を調整する。目的関数変更と推定器変更を同時に行わない。

`gae_lambda = 0.97` は、本命設計確立後の独立アブレーションとする。

## 5. 安全性の三層分離

### 5.1 Hard safety: 環境が必ず強制する条件

次は学習アルゴリズムへ委ねない。

- 1 銘柄当たり最大 absolute weight
- 最大 gross exposure
- 有効証拠金と利用可能資金
- 非有限値と不正 action の fail-closed
- 取引所最小数量、刻み、価格制約
- emergency flatten
- drawdown stop
- minimum equity termination
- liquidation と insolvency accounting

既存の `PreTradeRisk` と execution/accounting を canonical enforcement point とする。Lagrangian が失敗しても hard safety は破れない。

### 5.2 Soft constraints: Lagrangian が予算管理する条件

次を Lagrangian cost として使用する。

- `drawdown_excess`
- `daily_turnover`
- `execution_cost_fraction`
- `gross_exposure_request_excess`

既存 cost vector の他要素も学習・診断へ残せるが、予算値は意味ごとに区別する。

### 5.3 Catastrophic event gates: 未見評価でゼロを要求する条件

次は平均予算を満たすだけでは採用しない。

- `forced_liquidation_event`
- `margin_deficit_fraction > 0`
- insolvency
- unexpected drawdown stop

production candidate の未見評価では、原則として発生ゼロを要求する。発生が一件でもあれば、平均 net growth に関係なく NO-GO とする。意図的な fail-safe 動作を検証する fault-injection test は別集計とする。

## 6. Baseline の役割

baseline は学習報酬へ混ぜない。

baseline は次に使用する。

- paired unseen net log growth comparison
- fold ごとの相対成績
- deployment gate
- regression detection
- market regime 別診断

baseline を報酬から外すことで、方策は baseline への追従ではなく、absolute growth を自由に探索できる。

`shadow` path は比較・診断のため維持できるが、baseline weight がゼロの設定では reward total へ影響してはならない。

## 7. エピソード終端と truncation

720 時間は継続運用をサンプリングする訓練窓であり、経済的な投資終了ではない。

本命設定では次を保証する。

- `liquidate_on_end = false`
- time limit は `truncated = true`
- insolvency、minimum equity、明示的な forced close のみ `terminated = true`
- truncation 時の final observation を保持する
- critic value を使って bootstrap する
- time limit による人工的な最終清算コストを reward へ入れない
- time limit 直前だけポジションを閉じる方策を誘発しない

SB3 adapter まで含め、Gymnasium の terminated/truncated semantics を統合テストで固定する。

## 8. 比較用 training profiles

すべて次を共通化する。

- action mode: target_weight
- dataset と fold
- observation encoder と全 architecture parameter
- BC teacher、epochs、validation gate
- PPO rollout、batch、epochs、learning-rate schedule
- environment、risk、execution config
- seeds
- training timesteps

### 8.1 G1-PPO

```text
target_weight
net log growth only
gamma = 1.0
algorithm = ppo
hard safety enabled
```

純粋成長の必須対照群。

### 8.2 G1-Lagrangian

```text
target_weight
net log growth only
gamma = 1.0
algorithm = lagrangian_ppo
hard safety enabled
soft constraint budgets enabled
```

本命候補。

### 8.3 D168-Lagrangian

```text
target_weight
net log growth only
168-hour discount half-life
algorithm = lagrangian_ppo
hard safety enabled
soft constraint budgets enabled
```

時間選好アブレーション。

### 8.4 Legacy Full

既存 `training-full.json` は互換性と研究比較のため残すが、production candidate の既定値にはしない。

## 9. 設定契約

新 profile は、暗黙 default に依存せず reward weight をすべて明示する。

追加する validation:

1. pure growth profile では shaping weight、terminal weight、margin weight がすべてゼロでなければ拒否する。
2. `gamma = 1.0` と `discount_half_life_hours` の同時指定を拒否する。
3. baseline shaping が有効な profile では `require_full_reward_preroll = true` を要求する。
4. `baseline_tolerance = 0` かつ `baseline_progressive_power > 1` を拒否する。
5. target-weight constrained profile で residual-only field が有効なら拒否する。
6. Lagrangian profile では cost name、budget、learning rate、EMA beta、multiplier limit の長さを一致させる。
7. catastrophic event budget を緩い平均値だけで production 合格へ使用する設定を拒否または明示的に research-only とする。

## 10. 実装範囲

最低限、次を変更する。

- target-weight pure-growth PPO profile
- target-weight pure-growth Lagrangian profile
- target-weight discounted Lagrangian profile
- 対応する walk-forward profile
- reward/profile validation
- truncation bootstrap integration tests
- reward telescoping tests
- target-weight constraint alignment tests
- experiment comparison report fields
- docs と実行手順

既存の residual growth profiles は削除しない。役割を residual-policy research control として明記する。

## 11. テスト設計

### 11.1 Reward contract

- scale 前の累積 pure reward が `log(final_equity / initial_equity)` へ一致する。
- terminal loss を二重計上しない。
- baseline weight がゼロなら shadow path を変更しても reward が変わらない。
- drawdown と margin cost が reward total に混ざらない。
- execution cost、funding、borrow cost を含む net return が reward source になる。

### 11.2 Validation

- zero tolerance と progressive power greater than one を拒否する。
- gamma one profile に half-life を指定すると拒否する。
- constrained profile の cost vector length mismatch を拒否する。
- target-weight profile に residual-only controls を指定すると拒否する。

### 11.3 Safety

- max_abs_weight を必ず守る。
- max_gross を必ず守る。
- drawdown stop では turnover limit を上書きして flatten できる。
- invalid/non-finite target は fail-closed する。
- hard constraint が有効な状態で Lagrangian multiplier をゼロへ固定しても安全境界を破れない。

### 11.4 Truncation

- time limit と insolvency の flag を区別する。
- `liquidate_on_end = false` なら time limit に清算 return を加えない。
- time-limit truncation の final observation が SB3 rollout へ渡る。
- time-limit truncation では value bootstrap が行われる。
- true termination では bootstrap しない。

### 11.5 Lagrangian

- reward advantage と cost advantage を別々に計算する。
- multiplier は rollout 中に固定する。
- completed episode aggregation と elapsed-time weighting を維持する。
- max multiplier saturation、constraint residual sign flip、penalty-to-reward ratio を report する。

## 12. 実験手順

### 12.1 Stage 0: smoke と契約検証

各 profile について CPU と CUDA smoke を行う。

合格条件:

- finite reward、value、advantage、loss
- checkpoint 保存・resume
- architecture digest 一致
- reward schema と constraint schema 一致
- truncation semantics test 合格

### 12.2 Stage 1: screening

```text
1 fold x 3 seeds x 131,072 timesteps
```

対象:

- G1-PPO
- G1-Lagrangian
- D168-Lagrangian

この段階は勝者決定ではなく、明白な失敗を除外する。

除外条件:

- no-trade collapse
- multiplier の恒常的上限張り付き
- critic divergence
- hard safety violation
- catastrophic event
- seed 間の極端な崩壊

### 12.3 Stage 2: full evaluation

screening を通過した上位 2 profile を次で評価する。

```text
6 folds x 3 seeds x 524,288 timesteps
nested walk-forward
```

## 13. モデル選抜

異なる gamma や shaping の episode reward を直接比較しない。

共通の未見経済指標で比較する。

主要指標:

- execution-adjusted unseen net log growth

副指標:

- paired baseline net log growth difference
- median and worst-fold growth
- median and worst-seed growth
- maximum drawdown
- drawdown excess area
- daily turnover
- execution cost fraction
- forced liquidation rate
- margin deficit rate
- multiplier saturation rate
- constraint residual stability

stress tests:

- 2x execution cost
- 3x execution cost
- spread widening
- slippage worsening
- latency and rejection worsening
- partial-fill degradation

## 14. Production gate

候補は次をすべて満たした場合だけ GO とする。

1. 未見 net log growth の中央値が正。
2. baseline に対する paired difference が複数 fold で正。
3. 成績が単一 seed の大勝ちに依存しない。
4. forced liquidation、margin deficit、insolvency がゼロ。
5. hard safety violation がゼロ。
6. soft constraint budget を未見期間でも満たす。
7. 2x cost stress で優位性を維持する。
8. 3x cost stress で破綻しない。
9. checkpoint resume と serving loader が architecture/reward/constraint digest を検証する。
10. G1-Lagrangian が G1-PPO に勝たない場合、Lagrangian を production default にしない。

## 15. 推奨結果の解釈

- G1-Lagrangian が G1-PPO より高い未見成長と低いリスクを示す場合、G1-Lagrangian を本命とする。
- G1-PPO が同等以上で hard safety と production gate を満たす場合、単純性を優先して G1-PPO を採用する。
- D168-Lagrangian が改善する場合も、改善が短期時間選好によることを明示し、長期複利目的とは別 profile として扱う。
- どの RL profile も baseline を安定して超えない場合、RL を production 採用しない。

## 16. 完了条件

本設計の実装完了は、コード追加だけではなく次を満たした時点とする。

- 3 つの target-weight profile と walk-forward profile が存在する。
- config contract が不正な目的混合を fail-closed する。
- reward、hard safety、soft constraints、catastrophic gates が分離されている。
- time-limit truncation の bootstrap が統合テストで保証される。
- CPU/CUDA smoke が成功する。
- full test、Ruff、MyPy、import-linter が成功する。
- screening report が profile 間を共通経済指標で比較できる。
- production gate が自動判定される。
