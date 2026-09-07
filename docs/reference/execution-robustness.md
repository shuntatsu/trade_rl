# Execution robustness evidence

## 結論

`walk-forward-target-weight-execution-robustness.json` は、学習済みの gamma-one target-weight 方策を、取引所ルールだけでなく fee、spread、market impact、capacity、latency、tail slippage、borrow cost の不利な前提で再評価するための独立した evidence profile です。

この profile は既存の default workflow を変更せず、追加 scenario はすべて `report_only=true` とします。必須 production gate は従来どおり `joint_2x` であり、本 profile 単独では Production を許可しません。Production status は常に `NO-GO` です。

## 対象 family

他の研究作業で finite-horizon gamma-one と discounted continuing objective を分離しているため、本 profile は次の同一目的 family だけを比較します。

- `target-weight-growth-gamma-one-ppo`
- `target-weight-constrained-growth-gamma-one`

Discounted profile、BC、reward、Episode boundary、action head、risk budget、checkpoint selection は変更しません。

## 評価の流れ

各 sealed outer-OOS fold について、通常の configuration selection を完了した後に、選択済み policy と baseline を同じ scenario 前提で再実行します。各 scenario payload、selected/baseline metrics、scenario-result digest、scenario-pack digest は既存の sealed sensitivity artifact へ格納されます。

Hybrid book と shadow book は別 executor state を保持しますが、同じ immutable stressed `ExecutionCostConfig` を受け取ります。Selected と baseline の評価も同じ scenario definition を使用します。

## Scenario pack

| Scenario | 変更内容 | Gate |
|---|---|---|
| `nominal` | nominal execution | 既存 evidence |
| `tick_2x` | tick size 2x | 既存 standard |
| `lot_2x` | lot size 2x | 既存 standard |
| `minimum_notional_2x` | minimum notional 2x | 既存 standard |
| `joint_2x` | tick / lot / minimum notional 2x | **required** |
| `joint_5x` | rule burden 5x | report only |
| `joint_3x` | rule burden 3x | report only |
| `fee_spread_2x` | fee 2x、spread 2x | report only |
| `impact_2x` | market impact 2x | report only |
| `capacity_half` | participation capacity 0.5x | report only |
| `latency_1bar` | minimum order latency 1 bar | report only |
| `tail_slippage_adverse` | slippage floor 5 bps、tail probability 1%、tail multiplier floor 5x | report only |
| `borrow_2x` | borrow rate cost 2x | report only |
| `joint_execution_adverse` | rule 2x、fee 1.5x、spread / impact 2x、slippage 2x with 5 bps floor、capacity 0.5x、latency 1 bar、tail 1% / 5x、borrow 2x | report only |

## Backward compatibility

`execution_sensitivity_config_v1` の standard six-scenario pack と `required_scenario=joint_2x` は変更しません。

追加 execution fields がすべて identity の scenario は、従来の `execution_rule_stress_v1` digest を維持します。したがって既存の `joint_3x` などは意味も identity も変わりません。Standard scenario 名へ fee や latency などを混在させる設定は fail closed とし、required gate を暗黙に広げません。

## Randomness

Tail slippage を有効にする scenario は、run configuration の既存 `random_seed` を使用して再現可能にします。Selected と baseline は同じ scenario と seed を使いますが、注文発生回数や fill path は action に依存するため、乱数 draw の一対一対応までは仮定しません。比較証拠は実現した after-cost return、cost、turnover、drawdown、trade diagnostics を使用します。

## 既存 exploratory matrix との違い

`trade_rl/workflows/execution_sensitivity.py` は単一区間の exploratory parameter matrix です。本 profile は nested walk-forward の sealed outer-OOS、selected policy identity、baseline identity、fold access evidence、scenario digest に結合された post-selection robustness evaluation です。両者を代替関係として扱いません。

## Non-goals

- stress を学習時 domain randomization へ自動適用しない。
- default full-research workflow を変更しない。
- `joint_2x` gate threshold を変更しない。
- BC、reward、Episode boundary、PPO、Lagrangian update、serving、live order routing を変更しない。
- report-only scenario の結果だけで Production を許可しない。
