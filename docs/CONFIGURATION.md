# Training Configuration v4

維持対象のTop-level Schemaは`training_run_config_v4`です。設定は`trade-rl train run`と`trade-rl walk-forward run`で使用します。

```json
{
  "schema_version": "training_run_config_v4",
  "training": {},
  "environment": {"require_full_reward_preroll": true},
  "action": {},
  "risk": {},
  "execution": {
  "fee_rate": 0.0005,
  "maker_fee_rate": 0.0,
  "taker_fee_rate": 0.0,
  "spread_rate": 0.0002,
  "impact_rate": 0.0001,
  "multiplier": 1.0,
  "max_participation_rate": 0.05,
  "slippage_std": 0.0,
  "tail_slippage_probability": 0.0,
  "tail_slippage_multiplier": 5.0,
  "random_seed": 0,
  "minimum_notional": 0.0,
  "lot_size": 0.0,
  "tick_size": 0.0,
  "allow_short": true,
  "borrow_rate_multiplier": 1.0,
  "max_leverage": 1.0,
  "maintenance_margin_rate": 0.25,
  "collateral_haircut": 1.0,
  "margin_mode": "cross",
  "order_latency_bars": 0,
  "order_type": "market",
  "limit_offset_rate": 0.0005,
  "path_mode": "conservative",
  "processing_bar_volume_capacity": true,
  "partial_fill_carry": true,
  "trigger_volume_fractions": [1.0, 0.5, 0.25, 0.0]
},
  "reward": {},
  "trend": {},
  "exports": {}
}
```

完全な維持対象Example:

- [Quickstart](../examples/quickstart/training.json)
- [Binance Multi-Timeframe full training](../examples/binance-multitimeframe/training-full.json)
- [Binance Multi-Timeframe walk-forward](../examples/binance-multitimeframe/walk-forward-full.json)

## Legacy設定

`training_run_config_v1`、`training_run_config_v2`、`training_run_config_v3`は自動変換しません。明示的に拒否します。v4ではTop-level `execution`、Executionの全Field、`environment.require_full_reward_preroll: true`が必須です。

次の旧Booleanと曖昧な共通Attention設定も廃止済みです。

```text
sequence_encoder
asset_set_encoder
sequence_attention_heads
sequence_attention_layers
```

対応する現行設定は、単一の`observation_encoder`と、時間足・銘柄別のAttention設定です。

## `observation_encoder`

| 値 | Policy | 対応Algorithm | 説明 |
|---|---|---|---|
| `flat_mlp` | `MlpPolicy` | PPO-family、SAC、TD3、TQC | Flat observationをMLPへ入力 |
| `asset_set` | 通常は`MultiInputPolicy` | PPO-family、SAC、TD3、TQC | 銘柄集合を明示的に扱う非系列Encoder |
| `hierarchical_sequence_v2` | `MultiInputPolicy` | PPO、CostCriticPPO、LagrangianPPO | Native sequenceを階層的に融合 |

設定値は小文字へ正規化されます。無効な値や、EncoderとPolicy／Algorithmの不整合は初期化時に拒否されます。

Encoderで使わない設定を非既定値にするとFail closedします。これは、設定に書いた値が黙って無視されることを防ぐためです。

## Hierarchical sequence v2

入力Clockは順序固定です。

```text
15m -> 96 bars
1h  -> 168 bars
4h  -> 120 bars
1d  -> 60 bars
```

各Clockは`values`、`available`、`staleness`を別Tensorとして保持します。

Model flow:

```text
Clock別Causal TCN
  -> Clock latent + Context token
  -> Gated Cross-Timeframe Attention
  -> Asset token + Symbol embedding
  -> Gated Cross-Asset Attention
  -> Actor / Critic
```

### TCNと共通次元

| 設定 | 既定 | 制約 |
|---|---:|---|
| `sequence_tcn_capacity` | `standard` | `standard`または`compact` |
| `sequence_d_model` | `320` | 正整数、各Head数で割り切れること |
| `sequence_dropout` | `0.05` | `0.0`から`0.05` |
| `max_policy_parameters` | `12000000` | 正整数 |
| `max_rollout_buffer_bytes` | `805306368` | 正整数 |

維持対象のFull training Exampleは`sequence_d_model: 336`を使用します。既定値と維持対象Presetを混同しないでください。

### Cross-Timeframe Attention

| 設定 | 既定 |
|---|---:|
| `sequence_timeframe_attention_heads` | `8` |
| `sequence_timeframe_attention_layers` | `2` |
| `sequence_timeframe_ffn_multiplier` | `3` |
| `sequence_timeframe_gate_bias` | `-2.0` |

### Cross-Asset Attention

| 設定 | 既定 |
|---|---:|
| `sequence_asset_attention_heads` | `8` |
| `sequence_asset_attention_layers` | `2` |
| `sequence_asset_ffn_multiplier` | `3` |
| `sequence_asset_gate_bias` | `-2.0` |

時間足と銘柄の設定は独立しています。同じ値を使う場合でも、意味の異なる設定を1つのFieldへまとめません。

## Runtime acceleration

| 設定 | 既定 | 説明 |
|---|---|---|
| `sequence_compile` | `false` | PyTorch compileを使用 |
| `sequence_compile_mode` | `reduce-overhead` | `default`、`reduce-overhead`、`max-autotune` |
| `sequence_transfer_mode` | `synchronous` | `synchronous`または`pinned_non_blocking` |
| `vector_environment_mode` | `auto` | `auto`、`in_process`、`subprocess` |

`sequence_compile`が`false`の場合、Compile modeを別値にしても無視せず拒否します。Sequence専用設定は`hierarchical_sequence_v2`以外では既定値のままにします。

## PPO-familyとBehavior Cloning

対応するPPO-family:

```text
ppo
cost_critic_ppo
lagrangian_ppo
```

`behavior_cloning_epochs > 0`はPPO-familyでだけ有効です。Teacherは`oracle`または`trend_baseline`です。

```json
{
  "training": {
    "algorithm": "ppo",
    "behavior_cloning_epochs": 10,
    "behavior_cloning_learning_rate": 0.001,
    "behavior_cloning_batch_size": 256,
    "behavior_cloning_teacher": "oracle"
  }
}
```

BCと後続PPOは同じFeature extractorを使用します。Cost criticやLagrangian固有Identityとは別に、Policy architecture identityを合成して保存します。

## Architecture identityとCheckpoint

`sb3_policy_identity_v1`は、設定値の写しではなく、**実際に組み立てられたModel**から生成します。

`hierarchical_sequence_v2`では次を結合します。

- Clock順、Window、Input channel、Latent width、TCN widthとDilation
- Timeframe/Asset Attention構造
- Symbol順
- Action名とAction順
- Observation encoder

Checkpoint manifestはTraining config digestとPolicy architecture identityを別々に保存します。Resume時は両方を検証し、構造不一致を拒否します。Weight更新だけではArchitecture digestは変化しません。

## TensorBoard診断

```json
{
  "training": {
    "tensorboard_enabled": true,
    "tensorboard_log_interval": 1
  }
}
```

系列Policyでは次の診断を低頻度で記録します。

- Clock別Attention shareとMissing ratio
- Timeframe/Asset Attention entropyと最大Share
- Timeframe/Asset Gate平均と飽和率
- Timeframe encoder、Timeframe fusion、Cross-asset BlockのGradient norm

診断は最適化状態を観察するためのもので、Checkpoint選択、Sealed評価、収益性Evidenceではありません。

## ExportとServing

Flat policyでは`exports.onnx`と`exports.torchscript`を使用できます。

`hierarchical_sequence_v2`はFlat tensor Exportを使わず、`structured_policy_export_v2`を使用します。構造化Exportは次をManifestへ固定します。

- Canonical Dict input順
- Input shapeとdtype
- `sb3_policy_identity_v1`
- Sequence architecture digest
- Model file digestとSize
- Action size
- Parity corpusの最大誤差とTolerance

Model fileは`policy.structured.torchscript.pt`です。Servingでは`CanonicalStructuredPolicyLoader`が`serving_bundle_v5`、Sequence observation schema、Export manifest、Model、Architecture digestを照合します。

## TensorとParameterの上限

`n_steps * n_envs`はPPOの1 Rollout sizeです。`batch_size`は完全なRolloutを割り切る必要があります。系列ModelではParameter上限とRollout buffer byte上限を、Model構築・学習開始前に検証します。

Full trainingの値を変更するときは、Parameter数、GPU memory、Throughput、Seed間分散、Worst foldを一緒に評価してください。
