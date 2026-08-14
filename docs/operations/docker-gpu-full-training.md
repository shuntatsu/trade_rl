# Docker GPU Full Training

Production status remains `NO-GO`. このRunbookは、維持対象のBinance Multi-Timeframe研究をDockerとCUDAで実行する手順です。成功はPipeline Evidenceであり、収益性やProduction認可を意味しません。

## 前提

- Linux x64
- NVIDIA DriverとContainer Toolkit
- Docker Compose
- Repositoryの`main`
- 書込み可能なArtifact volume
- 必要なPublic verification keyと署名済みEvidence

確認:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Local execution

```bash
docker compose -f docker/compose.training.yaml build trainer
docker compose -f docker/compose.training.yaml run --rm trainer
```

Containerは次をFail closedで確認します。

- CUDA deviceとPyTorch CUDA
- Source/lock provenance
- DatasetとMetadata evidence
- `training_run_config_v4`
- ParameterとRollout buffer上限
- BC、PPO-family、Checkpoint、Resume
- Evaluation、Execution sensitivity、Research gate

## 維持対象Full configuration

既定の正本は
[walk-forward-target-weight-constrained-growth.json](../../examples/binance-multitimeframe/walk-forward-target-weight-constrained-growth.json)
です。明示的な`--training-template`指定がない`develop` phaseは、次の3候補を同じwalk-forward契約で比較します。

1. `training-target-weight-growth-ppo.json`: gamma-one net-growth PPO control
2. `training-target-weight-constrained-growth.json`: gamma-one constrained-growth Lagrangian PPO candidate
3. `training-target-weight-constrained-growth-discounted.json`: 168-hour discounted constrained-growth ablation

3候補に共通する主な値:

```text
symbol / action: BTCUSDT / one direct target-weight action
observation_encoder: hierarchical_sequence_v2
policy: MultiInputPolicy
behavior cloning: oracle teacher / 45 epochs / 10% chronological validation
learning rate: linear 0.00012 -> 0.000012
n_envs / n_steps / batch_size / n_epochs: 8 / 128 / 256 / 10
seeds: 0, 1, 2
device: cuda
timesteps per seed: 524288
sequence_d_model: 336
timeframe attention: 8 heads / 2 layers
max_policy_parameters: 12,000,000
tensorboard diagnostics: enabled
```

`training-full.json`はlegacy mixed-shaping comparisonとして保持します。暗黙のdefault候補には含めず、必要な場合だけ`--training-template training-full.json`で明示選択します。`walk-forward-full.json`も過去の再現性のため削除しません。

既存generationは記録済みsource commit、image、configuration、artifact identityへ固定されます。default変更によって実行中または完了済みgenerationを移行、上書き、別objectiveでresumeしません。新しいdefaultは、新しいsource commitとimageから作成した新generationだけに適用されます。

これらは維持対象候補であり、最適値やProduction認可を意味しません。

## Oracle Bellman solver

Oracle teacher生成の既定backendは`numpy`です。CUDAは明示選択でのみ使用します。Teacher artifactとcache identityは、数値契約・market tape・実際のtarget/scoreだけで決まり、実行時間、peak memory、GPU名、fallback、OOM再試行などのruntime evidenceでは変化しません。runtime evidenceはmanifestとcatalog metadataへ別に保存します。

```text
TRADE_RL_ORACLE_SOLVER=numpy|cuda|cuda_or_numpy
TRADE_RL_ORACLE_EPISODE_BATCH_SIZE=8
TRADE_RL_ORACLE_TARGET_STATE_BLOCK_SIZE=
TRADE_RL_ORACLE_CUDA_MEMORY_FRACTION=0.65
TRADE_RL_ORACLE_COMPILE_MODE=disabled
TRADE_RL_TEACHER_WORKERS=1
```

- `numpy`: 維持対象の既定経路です。
- `cuda`: CUDA unavailable、OOM再試行失敗、backend errorをそのままFail closedします。
- `cuda_or_numpy`: CUDA solve全体がartifact promotion前に失敗した場合だけ、部分結果を破棄してNumPyで全episodeを最初から再実行します。fallback理由はruntime provenanceとcatalog metadataへ記録されますが、数値artifact identityには含めません。
- CUDA選択時は`TRADE_RL_TEACHER_WORKERS=1`が必須です。並列性は複数GPU-owner processではなく`TRADE_RL_ORACLE_EPISODE_BATCH_SIZE`で管理します。
- `TRADE_RL_ORACLE_TARGET_STATE_BLOCK_SIZE`を空にすると、backendがmemory予算からblock sizeを決定します。
- compiled Oracle executionは未検証のため使用できません。固定長chunkを実際にcompileする実装、NumPy parity、維持対象GPU上の同期benchmark evidenceが揃うまで`disabled`固定です。

CUDAを既定へ変更する判断は、同期済みcorrectness corpusと維持対象GPUのbenchmark evidenceを別途満たした後に行います。

## 3-update CUDA smoke

本学習を開始する前に、非常に小さいfloat64ネットワークでoptimizer更新を正確に3回だけ実行します。このスモークは、CUDA認識、forward/backward、勾配、optimizer update、device/dtype不一致、NaN/Inf、非同期CUDAエラー、peak allocated/reserved memoryを短時間で検査します。

PowerShell:

```powershell
$env:CUDA_LAUNCH_BLOCKING = "1"
uv run python -m trade_rl.operations.cuda_training_smoke `
  --device cuda:0 `
  --output artifacts/cuda-smoke/tiny-training.json
```

Bash:

```bash
CUDA_LAUNCH_BLOCKING=1 uv run python -m trade_rl.operations.cuda_training_smoke \
  --device cuda:0 \
  --output artifacts/cuda-smoke/tiny-training.json
```

成功条件:

- Exit code `0`
- `schema_version == "tiny_cuda_training_smoke_v1"`
- `updates == 3`
- lossとgradient normがすべてfinite
- `parameter_delta_l2 > 0`
- `device_type == "cuda"`
- GPU名、compute capability、CUDA version、peak memoryが記録される

CPUでコマンド構造だけを確認するときは`--device cpu --allow-cpu`を指定できます。ただし、その結果はGPU実機証拠として扱いません。

続いて、PRのOracle CUDA経路そのものを疑似市場・1 episode・3 barsでNumPyと照合します。

```powershell
uv run python -m trade_rl.operations.oracle_cuda_smoke `
  --backend all `
  --episode-count 1 `
  --episode-bars 3 `
  --repetitions 1 `
  --episode-batch-size 1 `
  --target-state-block-size 2 `
  --output artifacts/cuda-smoke/oracle-parity.json
```

この2段階は役割が異なります。3-update smokeの成功はPyTorch/CUDA学習ループの基本動作を示しますが、Oracle transition、Bellman backpointer、NumPy parity、OOM retry、provenance契約の代替にはなりません。Oracle benchmarkも成功して初めてPR固有の極小CUDAスモークが成立します。

## Shared market archive

Raw Binance archive cacheと学習Artifact volumeを分離します。学習開始時にCacheを検証し、不足期間だけを取得します。

Cache削除は次回Downloadを発生させます。Published dataset、Run、Checkpointを同時に削除しないでください。

Protected **Control Binance frozen 226 full generation** workflowは、Trainer起動前にMarket archiveの不足分だけを同期します。直接`trainer`を起動した場合も、学習開始前のRead-only completeness checkを通過しなければ非ゼロ終了します。

## GitHub Actionsの安全境界

Self-hosted GPU workflowは、Pull RequestのCodeを実行しません。次を要求します。

- Owner-authored `main`
- Labels: `self-hosted`, `linux`, `x64`, `gpu`, `nvidia`
- `gpu-full-training` Environment
- External ActionのCommit SHA固定
- Source checkoutはread-only

Jobが`queued`で`runner_id = 0`の場合、Environment承認ではなく、Labelに一致するRunnerが接続されていない可能性があります。Runner serviceとLabelを確認してください。

Private Ed25519 keys must never be stored in Actions secrets, Docker environment variables, images, volumes, or the repository. TrainerとRuntimeへ渡すのはpublic keys onlyと署名済みArtifactです。

## Phase control

**Control Binance frozen 226 full generation**を`main`から実行し、Operation、Phase、Generationを明示します。

- `develop`: Dataset、Walk-forward、Gateを実行し、selection proposal and authorizationの境界で停止可能
- `train-selected`: 外部Authorizationを検証してSelected-finalを学習し、`awaiting_fresh_confirmation`で停止可能
- `finalize`: Fresh confirmationとPaper reconciliationを検証してFinal gateを記録

Waiting stateは失敗ではありません。Research rejectionとInfrastructure failureは別のExit codeで記録します。

## Evidence

Generationごとに、少なくとも次を保持します。

```text
/workspace/var/runs/<generation>/cuda-preflight.json
/workspace/var/runs/<generation>/entrypoint-provenance.json
/workspace/var/runs/<generation>/heartbeat.json
/workspace/var/runs/<generation>/summary.json
/workspace/var/runs/<generation>/artifacts/
container inspect
complete logs
exit status
```

失敗Containerを削除する前に、Inspect、Log、Generation directoryを回収します。

## Monitoring

**Monitor Binance frozen 226 full generation**はRead-onlyです。external expectation file、Container label、Image ID、Source/lock identity、heartbeat、Maximum runtime、OOM、Docker healthを照合します。

Expectationが存在しない場合は`idle`です。Container消失、Identity変更、Stale/Future heartbeat、OOM、Abnormal stateを検出した場合はFail closedします。

## TensorBoardとStudio

維持対象Full configurationは`tensorboard_enabled=true`で、Seed/RunごとのEvent fileを生成します。StudioはLoopback API経由でAllowlist済みScalarだけを読みます。TensorBoard serverの公開は不要です。

Sequence diagnosticsにはAttention、Gate、Missing ratio、Gradient normが含まれます。診断はModel selection Evidenceではありません。

## Failure handling

- CUDA preflight失敗: Driver、Toolkit、Device mountを確認
- OOM: Modelを黙って縮小せず、Batch、Rollout、Compile、Environment数を明示変更して別Configとして記録
- Dataset mismatch: CacheではなくPublished artifactを再検証
- Checkpoint mismatch: Training config digestとArchitecture identityを確認
- Runner待ち: Self-hosted runner processとLabelsを確認
- Phase失敗: 同じGenerationをClean runとして再利用しない

selected-final training forbids injected resume checkpoints by contract. 失敗したGenerationをClean runとして再利用せず、許可されたWaiting stateからの継続か、新しいGenerationを使用します。

## Stop

Detached実行を停止するときは、Container停止、Evidence回収、Artifact upload、Container削除、Expectation削除の順に処理します。

Logs are never requested after container removal. Log回収前にContainerを削除しないでください。

Hardware別の候補値は[4070 Ti SUPER資料](../performance/4070ti-super-full-training.md)を参照してください。
