# RTX 4070 Ti SUPER Full Training Profile

この文書は、RTX 4070 Ti SUPERで維持対象Full configurationを実行するための候補Profileです。性能最適値や収益性を保証するものではありません。

## Model profile

正本は[training-full.json](../../examples/binance-multitimeframe/training-full.json)です。

```text
observation encoder: hierarchical_sequence_v2
action: target_weight (BTCUSDT, ETHUSDT, BNBUSDT order)
TCN capacity: standard
d_model: 336
Cross-Timeframe Attention: 8 heads / 2 layers / FFN x3
Cross-Asset Attention: 8 heads / 2 layers / FFN x3
Actor: [384, 256, 128]
Value: [512, 384, 256]
Parameter ceiling: 12,000,000
PPO rollout: 256 steps x 4 environments
PPO batch size: 256
PPO epochs: 10
Oracle behavior cloning: 15 epochs / 10% chronological validation / patience 5
Learning-rate schedule: linear 0.00012 -> 0.000012
Seeds: 0, 1, 2
TensorBoard sequence diagnostics: enabled
```

## Runtime settings

```text
sequence_compile: false
sequence_compile_mode: reduce-overhead
sequence_transfer_mode: pinned_non_blocking
vector_environment_mode: subprocess
TORCHINDUCTOR_COMPILE_THREADS: 4
```

Compileの初回Costを含む短いSmokeだけでThroughputを判断しません。Warm-up後のEnvironment step/s、Update time、GPU memory、CPU saturationを記録します。
Inductorの永続compile worker poolは4 processに制限し、compile完了後もDockerのCPU memory headroomを確保します。

## 速度改善の境界

許可する改善:

- Precomputed `SequencePolicyPlane`
- Fold-scoped teacher rollout cache
- Persistent pinned staging with bounded non-blocking transfer
- Compile済みSequence rollout path
- Independent vector environment
- Cost criticの共有Feature cache

Model capacity、PPO epoch、Seed、Foldを黙って減らして速くする変更は、同じExperimentとして扱いません。

## 測定項目

- Dataset、Config、Source、Lock digest
- Parameter数
- Peak GPU memory
- Environment steps/s
- PPO update time
- BC epoch time
- CPU utilizationとData wait
- Checkpoint save/resume time
- Seed間分散とWorst fold

GPU使用率が低いことだけを品質問題としません。小さいModelではEnvironment、Python、SerializationがBottleneckになるため、End-to-end throughputを評価します。

## Verification

1. CUDA preflight
2. Structured observation forward
3. BC warm start
4. PPO rollout/update
5. Checkpoint save/resume
6. TensorBoard sequence diagnostics
7. Structured export parity
8. Canonical serving loader smoke
9. Docker non-root artifact write

すべてのEvidenceは[GPU運用手順](../operations/docker-gpu-full-training.md)に従ってGeneration単位で保存します。
