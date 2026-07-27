# Docker GPU Full-Research Operations

Production status remains `NO-GO`. These workflows generate software and research evidence only; successful execution is not profitability evidence, release approval, or permission to deploy capital.

## Required repository and runner configuration

Configure the GitHub Environment `gpu-full-training` outside the repository with required reviewers, prevent self-review, restrict deployment branches to `main`, and disable administrator bypass where the plan supports it. Use an ephemeral/JIT self-hosted GPU runner or destroy the runner host after each privileged job. Docker access is equivalent to host-root access and must not be shared with untrusted workloads.

The hourly monitor intentionally does not use the protected Environment. It is read-only and validates an external expectation file, container labels, actual image ID, generation heartbeat, maximum runtime, OOM state and Docker health.

## Immutable build identity

The training image is tagged by the exact Git commit and binds:

- checked-out commit SHA;
- canonical source-tree SHA-256;
- `uv.lock` SHA-256;
- actual Docker image ID;
- pinned Python base-image digest.

The image build and runtime both recompute source and lock identities. A caller-supplied commit string alone is never accepted as provenance. The process also records generation identity, phase, runtime/container identity, heartbeat, terminal state, and retained evidence paths before cleanup.

## Shared market-data archives

Binance Vision ZIP archives are stored in the independent Docker volume `trade-rl-market-archives`. They are not stored inside a training generation or PostgreSQL. The one-shot `market-data-sync` service mounts this volume read-write; `trainer` mounts it read-only.

Use the canonical launcher for every phase start:

```bash
python scripts/run_docker_training.py
```

The launcher executes these operations in order:

1. plan the exact archive URLs required by the maintained symbols, native timeframes and fixed research interval;
2. inspect the shared cache and download only missing or empty archives;
3. stop immediately if synchronization fails;
4. start `trainer` without rerunning dependencies.

Manual synchronization is available when only the archive volume should be updated:

```bash
docker compose -f compose.training.yaml run --rm market-data-sync
```

A direct trainer invocation still declares `market-data-sync` as a completed dependency, but the canonical launcher is preferred because it explicitly runs synchronization on every invocation. The training image also performs a read-only completeness check before CUDA preflight. If the archive volume is incomplete, training fails closed and prints the synchronization command instead of downloading from the read-only mount.

The maintained research end time remains authoritative. Advancing it in the pipeline causes only newly required URL-addressed archives to be downloaded; unchanged archives are read from the shared cache. The trailing incomplete USDⓈ-M funding month remains a REST responsibility of the existing dataset path because Binance Vision publishes funding archives monthly.

The other persistent payload volumes are separate:

- `trade-rl-training-runs`: generation directories, logs, checkpoints and evaluation artifacts;
- `trade-rl-teacher-cache`: reusable teacher artifacts.

Removing `trade-rl-training-runs` does not remove the market archive cache. Remove `trade-rl-market-archives` only when a complete redownload is intentional.

## Evidence and public keys

Set `TRADE_RL_EVIDENCE_ROOT` as a host path readable by the runner. Mount only public verification material into the trainer:

```text
binance-rule-history.json
metadata-public-keys.json
selection-authorization.json
selection-public-keys.json
fresh-confirmation.json
confirmation-public-keys.json
```

Private Ed25519 keys must never be stored in Actions secrets, Docker environment variables, images, volumes or the repository. Trainer and runtime receive public keys only. Generate signed artifacts with the offline CLI commands documented in `README.md`. Numerical datasets, arrays, checkpoints, models, and evidence remain immutable filesystem artifacts; an optional PostgreSQL service may index metadata, provenance, dependency, lifecycle, cache, and sealed-reservation records but is not payload storage.

## Start a phase

Use **Control Binance frozen 226 full generation** from `main`. Supply:

- operation: `start`
- phase: `develop`, `train-selected`, or `finalize`
- generation: one stable identifier reused across the three phases

The workflow checks out `${{ github.sha }}`, verifies the source/lock labels of the commit-tagged image, requires a real CUDA device, then starts one supervised container. `develop` may finish in `awaiting_selection_authorization`; `train-selected` may finish in `awaiting_fresh_confirmation`. These are successful persisted states.

Every generation stores its own:

```text
/workspace/var/runs/<generation>/cuda-preflight.json
/workspace/var/runs/<generation>/entrypoint-provenance.json
/workspace/var/runs/<generation>/heartbeat.json
/workspace/var/runs/<generation>/summary.json
/workspace/var/runs/<generation>/artifacts/
```

## Monitor

**Monitor Binance frozen 226 full generation** runs hourly and can be dispatched manually. If no external expectation exists it reports `idle`. Otherwise it fails closed when the expected container disappears, the image/source/lock identity changes, Docker reports OOM or an abnormal state, health fails, the heartbeat is stale/future-dated, or the maximum runtime is exceeded.

When a phase exits, the monitor copies the complete generation directory to an Actions artifact. It does not remove the retained container.

## Stop and remove

Dispatch the control workflow with operation `stop`. It performs this order:

1. stop the container;
2. capture inspect evidence and complete Docker logs;
3. copy the complete generation directory;
4. upload retained evidence;
5. remove the container and external expectation.

Logs are never requested after container removal.

## Metadata modes

`frozen_snapshot` is the default and is disclosed as unauthenticated, non-point-in-time evidence. `historical_signed` requires the Ed25519 v4 signed history and public-key store. `conservative_static` requires an explicitly versioned approximation payload. No mode presents current exchange metadata as historical truth.

## Failure handling

Do not reuse a failed generation as if it were clean. Preserve its artifact and logs, diagnose the exact failing boundary, then start a new generation or continue the same phase only when its persisted state contract explicitly permits it. A fresh retry uses a new generation identity. A phase continuation reuses the same generation only from an allowed waiting state. selected-final training forbids injected resume checkpoints by contract.

## Studio 学習診断の確認

`training-full.json` と `training-growth-optimal.json` は linear learning-rate decay と TensorBoard scalar 出力を有効にしています。linear decay は候補設定であり、最適値とみなさないでください。

GPU training の開始後、Studio の `Live Training` で Run と Seed を選択し、`学習診断` を開きます。event file が作成されるまでは `未出力` と表示されます。resume 後は同じ seed/run identity の event file を統合し、global step 軸を継続します。市場リプレイの JSONL telemetry は独立して継続します。
