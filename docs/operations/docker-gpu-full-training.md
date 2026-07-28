# Docker GPU Full Training

このRunbookは、維持対象のBinance Multi-Timeframe研究をDockerとCUDAで実行する手順です。成功はPipeline Evidenceであり、収益性やProduction認可を意味しません。

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
docker compose -f compose.training.yaml build trainer
docker compose -f compose.training.yaml run --rm trainer
```

Containerは次をFail closedで確認します。

- CUDA deviceとPyTorch CUDA
- Source/lock provenance
- DatasetとMetadata evidence
- `training_run_config_v2`
- ParameterとRollout buffer上限
- BC、PPO-family、Checkpoint、Resume
- Evaluation、Execution sensitivity、Research gate

## 維持対象Full configuration

正本は[training-full.json](../../examples/binance-multitimeframe/training-full.json)です。

主な値:

```text
observation_encoder: hierarchical_sequence_v2
policy: MultiInputPolicy
algorithm: ppo
n_envs: 4
seeds: 0, 1, 2
device: cuda
sequence_d_model: 336
timeframe attention: 8 heads / 2 layers
asset attention: 8 heads / 2 layers
max_policy_parameters: 12,000,000
```

このPresetは維持対象候補であり、最適値を意味しません。

## Shared market archive

Raw Binance archive cacheと学習Artifact volumeを分離します。学習開始時にCacheを検証し、不足期間だけを取得します。

Cache削除は次回Downloadを発生させます。Published dataset、Run、Checkpointを同時に削除しないでください。

## GitHub Actionsの安全境界

Self-hosted GPU workflowは、Pull RequestのCodeを実行しません。次を要求します。

- Owner-authored `main`
- Labels: `self-hosted`, `linux`, `x64`, `gpu`, `nvidia`
- `gpu-full-training` Environment
- External ActionのCommit SHA固定
- Source checkoutはread-only

Jobが`queued`で`runner_id = 0`の場合、Environment承認ではなく、Labelに一致するRunnerが接続されていない可能性があります。Runner serviceとLabelを確認してください。

## Evidence

Generationごとに、少なくとも次を保持します。

```text
cuda-preflight.json
entrypoint-provenance.json
heartbeat.json
summary.json
artifacts/
container inspect
complete logs
exit status
```

失敗Containerを削除する前に、Inspect、Log、Generation directoryを回収します。

## TensorBoardとStudio

`tensorboard_enabled`を有効にすると、Seed/RunごとのEvent fileを生成します。StudioはLoopback API経由でAllowlist済みScalarだけを読みます。TensorBoard serverの公開は不要です。

Sequence diagnosticsにはAttention、Gate、Missing ratio、Gradient normが含まれます。診断はModel selection Evidenceではありません。

## Failure handling

- CUDA preflight失敗: Driver、Toolkit、Device mountを確認
- OOM: Modelを黙って縮小せず、Batch、Rollout、Compile、Environment数を明示変更して別Configとして記録
- Dataset mismatch: CacheではなくPublished artifactを再検証
- Checkpoint mismatch: Training config digestとArchitecture identityを確認
- Runner待ち: Self-hosted runner processとLabelsを確認
- Phase失敗: 同じGenerationをClean runとして再利用しない

## Stop

Detached実行を停止するときは、Container停止、Evidence回収、Artifact upload、Container削除の順に処理します。Log回収前にContainerを削除しないでください。

Hardware別の候補値は[4070 Ti SUPER資料](../performance/4070ti-super-full-training.md)を参照してください。
