# Docker GPU Full-Training Operations

This runbook operates the maintained Binance multi-timeframe research pipeline
through `compose.training.yaml`. It requires a CUDA-capable Docker runtime. The
container fails closed when PyTorch cannot see CUDA; it does not fall back to
CPU. Runtime data is stored in the Docker named volume
`trade-rl-training-data`, mounted at `/workspace/var`.

The full run is research-only. Its artifacts and exit code do not guarantee
profitability, authorize live trading, or change the repository's `NO-GO`
production status.

Run every command from the repository root.

## Build and foreground run

In PowerShell, capture provenance from the checkout that supplies the Docker
build context before building:

```powershell
$env:TRADE_RL_GIT_COMMIT = (git rev-parse HEAD).Trim()
$env:TRADE_RL_GIT_DIRTY = if (git status --porcelain) { "true" } else { "false" }
```

The commit must be exactly 40 lowercase hexadecimal characters, and the dirty
state must be exactly `true` or `false`. The Docker build fails closed when
either value is missing or invalid. Compose itself remains parseable without
the variables so an already built image can still be run or stopped from a new
shell. The image exports both values so the maintained runners can preserve
source provenance even though `.git` is excluded from the image.

Build the locked Python 3.12 training image:

```bash
docker compose -f compose.training.yaml build trainer
```

Choose a unique run generation before every full invocation. A UTC timestamp
is a convenient default; add a suffix if another run could start in the same
second:

```powershell
$env:TRADE_RL_RUN_GENERATION = "full-$((Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss'))"
```

Start the maintained full workflow in the foreground and automatically remove
its one-off container when it exits:

```bash
docker compose -f compose.training.yaml run --rm trainer
```

The command runs
`examples/binance-multitimeframe/training_cuda_preflight.py` first and then
`examples/binance-multitimeframe/run_full_research.py`. Its generation root is
`/workspace/var/runs/$TRADE_RL_RUN_GENERATION`. The runner reuses the shared
cache at `/workspace/var/cache/binance-vision` and never deletes or overwrites
an existing generation: a duplicate name fails before cache or evidence is
changed. Exit code `0` means the workflow and its configured research gate
completed successfully; any preflight, training, evaluation, artifact, or gate
failure produces a nonzero exit code.

## Detached start, status, and logs

Use a named, detached one-off container when another terminal must inspect the
long-running job:

```bash
docker compose -f compose.training.yaml run --detach --name trade-rl-full-training trainer
```

Inspect its current state and exit code:

```bash
docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}}' trade-rl-full-training
```

Follow stdout and stderr until the container stops:

```bash
docker logs --follow --timestamps trade-rl-full-training
```

After recording the final status and extracting any required artifacts, remove
the stopped container:

```bash
docker rm trade-rl-full-training
```

If a shell disconnects while the detached container is still running, rerun
the `docker inspect` and `docker logs` commands. Do not start a second trainer
against the same named volume concurrently.

## Inspect the named volume and evidence

Inspect Docker's volume metadata:

```bash
docker volume inspect trade-rl-training-data
```

List persisted files without granting the helper container write access:

```bash
docker run --rm --mount type=volume,source=trade-rl-training-data,target=/workspace/var,readonly alpine:3.20 find /workspace/var -maxdepth 6 -type f -print
```

For a generation named `full-20260715-120000`, the principal evidence paths
are:

- `/workspace/var/cuda-preflight.json`
- `/workspace/var/cache/binance-vision/`
- `/workspace/var/runs/full-20260715-120000/training.log`
- `/workspace/var/runs/full-20260715-120000/walk-forward.log`
- `/workspace/var/runs/full-20260715-120000/research-gate.json`
- `/workspace/var/runs/full-20260715-120000/summary.json`
- `/workspace/var/runs/full-20260715-120000/artifacts/`

`summary.json` and `research-gate.json` are finalization evidence and may be
absent when an earlier stage fails. Inspect the container logs and any
persisted stage logs in that case.

The maintained sequence preset observes and recomputes target positions every
15 minutes, while no-trade and hysteresis controls suppress uneconomic orders.
It uses 226 ordered point-in-time feature channels across native clocks:
59 on 15m, 59 on 1h, 55 on 4h, and 53 on 1d. The policy receives completed
native sequences of 96, 168, 120, and 60 bars respectively, together with
feature availability/staleness masks, the current snapshot, execution state,
portfolio/risk state, and the finite-horizon coordinate when enabled. These
inputs remain a structured Dict observation and are not flattened into the old
1,241-value snapshot MLP contract.

The maintained policy uses `d_model=336`, two eight-head cross-asset attention
layers, separate actor and critic heads, and approximately 6.08 million
parameters. Four PPO environments use `n_steps=128` and batch size 128. Before
workers or the model are allocated, training estimates the exact SB3 Dict
rollout-buffer payload and rejects configurations above 768 MiB. The maintained
configuration estimates 473,122,816 bytes. Structured Oracle teacher artifacts
store compact decision indices and reconstruct only the requested sequence
mini-batch, avoiding duplication of overlapping 60-day windows.

The preset freezes two non-overlapping 360-hour outer
windows covering `2026-06-01T00:00:00Z` through
`2026-07-01T00:00:00Z`. Earlier outer windows were used during development and
must not be described as sealed evidence. Within each fold, checkpoint data
retains the top three policies per seed; configuration-selection data chooses
among those finalists before the outer window is opened. Each fold artifact
records the full experiment-plan digest and sealed-access digest needed to
audit that ordering.

## Copy artifacts to the host

In PowerShell, create the host destination and resolve it to an absolute path:

```powershell
$ExportDir = Join-Path (Get-Location) "training-export"
New-Item -ItemType Directory -Force -Path $ExportDir | Out-Null
$ExportDir = (Resolve-Path -LiteralPath $ExportDir).Path
```

Start an ephemeral Alpine container with the training volume mounted read-only
at `/volume` and the absolute host destination bind-mounted writable at
`/export`. Quote both `--mount` values so a Windows path containing spaces is
passed as one argument:

```powershell
docker run --rm `
  --mount "type=volume,source=trade-rl-training-data,target=/volume,readonly" `
  --mount "type=bind,source=$ExportDir,target=/export" `
  alpine:3.20 sh -c "cp -a /volume/. /export/"
```

The container must run for Docker to attach the user-created volume. The source
mount is read-only, while `/export` is the only writable data mount. Because
the container uses `--rm`, it is removed automatically after `cp` exits.

## Retry and recovery semantics

The maintained full runner does not resume an interrupted PPO or walk-forward
process. Each invocation requires a new generation name. Failed and interrupted
evidence stays in its original generation, while the next invocation reuses the
shared cache. The runner refuses an existing generation even if it is empty or
contains only partial evidence.

First stop and remove a detached trainer if it is still present:

```bash
docker rm --force trade-rl-full-training
```

Set a fresh name, then start a fresh attempt:

```powershell
$env:TRADE_RL_RUN_GENERATION = "full-$((Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss'))-retry1"
docker compose -f compose.training.yaml run --rm trainer
```

The prior generation and `/workspace/var/cache/binance-vision` remain intact.
If the new command reports that the generation already exists, choose another
name instead of moving or deleting volume contents.

The named volume survives container removal, but the new invocation is a fresh
run rather than checkpoint resume. Do not describe a retried run as resumed.

An already built image can be stopped or inspected without re-exporting the Git
provenance variables. A full `run` still requires
`TRADE_RL_RUN_GENERATION`; an image rebuild still requires valid
`TRADE_RL_GIT_COMMIT` and `TRADE_RL_GIT_DIRTY` values.

## Optional CUDA preflight and bounded smoke

Run only the maintained CUDA preflight:

```bash
docker compose -f compose.training.yaml run --rm --entrypoint uv trainer run python examples/binance-multitimeframe/training_cuda_preflight.py --output /workspace/var/cuda-preflight.json
```

Run the maintained bounded smoke with four rollout environments:

```bash
docker compose -f compose.training.yaml run --rm --entrypoint uv trainer run python examples/binance-multitimeframe/run_gpu_training_smoke.py --work-root /workspace/var/gpu-smoke --timesteps 8192
```

The smoke configuration itself fixes `n_envs` at `4`; the CLI accepts the work
root and timestep count only.

These tools live under `examples/binance-multitimeframe/`; there are no
maintained root-level CUDA or smoke scripts.

## Cleanup

Remove a retained one-off trainer container, if present:

```bash
docker rm --force trade-rl-full-training
```

Remove Compose service resources while retaining the named volume:

```bash
docker compose -f compose.training.yaml down
```

After exporting anything that must be retained, permanently delete all
downloaded data, logs, policies, checkpoints, and evidence in the named volume:

```bash
docker volume rm trade-rl-training-data
```

Volume deletion is irreversible. If Docker reports that the volume is in use,
remove the container named in the error and rerun the volume command.
