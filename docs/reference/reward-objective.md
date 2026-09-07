# 報酬目的とEpisode境界の契約

## 結論

維持対象の学習目的は、実コスト控除後のnet equityに対する区間log growthだけとする。

```text
reward_t = scale * log(net_equity_after / net_equity_before)
```

Drawdown、margin deficit、forced liquidation、gross exposure、turnover、execution costは、このscalar rewardへ混合せず、hard risk、独立constraint cost、walk-forward production gateで管理する。

`gamma = 1.0`を使う学習は、明示的な有限期間MDPとして扱う。人工的な学習窓を外部truncationとしてbootstrapする継続タスクでは、undiscounted returnの有限性を保証できないため、維持対象のgrowth objectiveには使用しない。

## 目的

- 経済的に解釈可能で、監査可能な学習目的を一つに固定する。
- 収益目的と安全・運用制約を分離する。
- `gamma`、Episode境界、Gymnasiumの`terminated`/`truncated`、value bootstrapの意味を一致させる。
- 既存のlegacy shaping、Artifact、Checkpointを暗黙に再解釈しない。

## 非目標

- average-reward RLアルゴリズムを今回導入しない。
- Lagrangian PPOのbudget値やdual updateを今回再調整しない。
- 実行シミュレータ、risk limit、BC teacher、policy architectureを変更しない。
- 既存Artifactを新しい目的へ自動移行しない。

## 経済報酬

区間報酬は、約定後のnet equityから計算済みの`hybrid_log_return`をそのままscaleする。

含まれる経済効果:

- mark-to-market損益
- fee
- spread
- market impact
- funding
- borrow cost
- partial fill
- terminal liquidationを明示的に行う評価経路では、その約定損益

純粋なgrowth profileでは次をゼロに固定する。

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

Terminal equity penaltyは使用しない。経済損失はnet log returnへ既に含まれるため、追加penaltyは二重計上になる。

## Constraint分離

維持対象のsoft constraintは既存の7 cost channelを使用する。

1. `drawdown_excess`
2. `drawdown_stop_event`
3. `margin_deficit_fraction`
4. `forced_liquidation_event`
5. `gross_exposure_request_excess`
6. `daily_turnover`
7. `execution_cost_fraction`

Hard exposure、minimum equity、margin、drawdown stop、exchange rule、execution capacityは環境とpre-trade riskが常に強制する。Lagrangian PPOはhard safetyの代替ではない。

## Episode境界

Environment configへ明示的な境界modeを追加する。

```text
external_truncation
finite_horizon_termination
```

### `external_truncation`

- Episode長は外部の計算上の切断。
- `terminated = false`、`truncated = true`。
- Value functionはterminal observationからbootstrapする。
- 維持対象では`gamma < 1.0`のdiscounted continuing objectiveに限る。

### `finite_horizon_termination`

- Episode長はMDPの有限期間そのもの。
- 時間上限で`terminated = true`、`truncated = false`。
- Time-to-goを観測へ含めるため、`finite_horizon_observation = true`を要求する。
- `gamma = 1.0`のnet log growthはEpisode終端net wealthへtelescopingする。
- `liquidate_on_end = false`なら終端はmark-to-marketであり、人工的なforced closeは行わない。
- 時間上限terminationは経済破綻ではないため、`hybrid_terminated`をtrueにせず、terminal-equity shapingを発火させない。

## ConfigとIdentity

`ResidualMarketEnvConfig`へ境界modeを追加し、次へ含める。

- validation
- environment digest
- persisted environment payload
- training config identity
- documentation and profile tests

Episode境界modeは環境力学の契約であり、観測schemaそのものではない。同じ観測特徴と`finite_horizon_observation`を使う2環境は、境界modeが異なっても同じ`observation_contract_digest`を持つ。一方、`environment_digest`は異なり、CheckpointやEvidenceが異なる遷移意味を暗黙に共有することを防ぐ。

Defaultは既存互換の`external_truncation`とする。既存ConfigとArtifactは従来意味のまま読み取るため、default modeは既存`environment_digest`へ新しいfieldを追加しない。非defaultの`finite_horizon_termination`だけを明示的にenvironment identityへ追加する。

維持対象のgamma-one target-weight growth profileは、明示的に次を設定する。

```json
{
  "episode_boundary_mode": "finite_horizon_termination",
  "finite_horizon_observation": true,
  "liquidate_on_end": false
}
```

168時間half-lifeのdiscounted ablationは、次を維持する。

```json
{
  "episode_boundary_mode": "external_truncation",
  "finite_horizon_observation": false,
  "liquidate_on_end": false
}
```

## Legacy shaping

`training-full.json`のbaseline、excess growth、drawdown、terminal、marginを混合した報酬は、比較・回帰用のlegacy profileとして保持する。

- Production候補として扱わない。
- 既存weightやschemaを変更しない。
- 新しい純粋growth contractへ自動変換しない。

## データフロー

```text
Execution result
  -> net interval log return
  -> RewardTracker
  -> pure scaled growth reward
  -> PPO actor/value objective

Environment transition
  -> hard risk and economic termination
  -> explicit episode boundary mode
  -> exclusive terminated/truncated flags
  -> SB3 bootstrap decision

Constraint diagnostics
  -> seven independent cost channels
  -> Cost Critic / Lagrangian update
  -> walk-forward evidence and production gate
```

## Fail-closed条件

- 未知のEpisode boundary modeを拒否する。
- `finite_horizon_termination`で`finite_horizon_observation=false`を拒否する。
- `gamma=1.0`の維持対象pure-growth profileが`external_truncation`を使う場合、profile contract testで拒否する。
- `terminated`と`truncated`が同時にtrueになる遷移を生成しない。
- 新しい境界modeをenvironment identityへ含めない実装を禁止する。

## テスト戦略

### Unit

- Configのmode normalizationと不正値拒否。
- 有限期間境界と外部truncationのpure transition classification。
- Hybrid insolvency、forced close、time limitの優先順位。
- Gamma-one pure growthが区間rewardの和としてterminal net log growthへ一致すること。
- 有限期間time limitがterminal-equity penaltyを発火させないこと。

### Contract

- Canonical target-weight PPO/Lagrangian profileが純粋growth、`gamma=1.0`、finite-horizon terminationを共有すること。
- Discounted profileとの差分がdiscountingと境界modeだけであること。
- Legacy full profileが明示的にlegacy comparisonとして残ること。
- Environment digestが境界modeで変化し、Observation digestはmode単独では変化しないこと。

### Integration

- Gymnasium stepがmodeに応じて排他的な`terminated`/`truncated`を返すこと。
- SB3 time-limit bootstrapが`external_truncation`だけで行われること。
- Cost Critic/Lagrangianのtransition alignmentを維持すること。

## Rollout方針

- RewardまたはEpisode境界の意味を変更する場合は、対応するConfig/Environment identityを更新し、既存実験を暗黙に再解釈しない。
- 意味の異なる学習結果は新しいConfig digestで再学習し、旧Artifact・Checkpointを新契約へ自動移行しない。
- Exact-head CI、型検査、format、architecture、full pytestが揃っても、それだけで収益性やProduction認可を意味しない。
- 実証Evidenceと運用認可が揃うまでProduction statusは`NO-GO`を維持する。
