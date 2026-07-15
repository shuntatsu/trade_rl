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

Build the locked Python 3.12 training image:

```bash
docker compose -f compose.training.yaml build trainer
```

Start the maintained full workflow in the foreground and automatically remove
its one-off container when it exits:

```bash
docker compose -f compose.training.yaml run --rm trainer
```

The command runs
`examples/binance-multitimeframe/training_cuda_preflight.py` first and then
`examples/binance-multitimeframe/run_full_research.py`. The fixed work root is
`/workspace/var/binance-multitimeframe-full`. Exit code `0` means the workflow
and its configured research gate completed successfully; any preflight,
training, evaluation, artifact, or gate failure produces a nonzero exit code.

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

The principal evidence paths are:

- `/workspace/var/cuda-preflight.json`
- `/workspace/var/binance-multitimeframe-full/training.log`
- `/workspace/var/binance-multitimeframe-full/walk-forward.log`
- `/workspace/var/binance-multitimeframe-full/research-gate.json`
- `/workspace/var/binance-multitimeframe-full/summary.json`
- `/workspace/var/binance-multitimeframe-full/artifacts/`

`summary.json` and `research-gate.json` are finalization evidence and may be
absent when an earlier stage fails. Inspect the container logs and any
persisted stage logs in that case.

## Copy artifacts to the host

Create a host destination. In PowerShell:

```powershell
New-Item -ItemType Directory -Force training-export
```

In a POSIX shell:

```bash
mkdir -p training-export
```

Create a read-only helper container, copy the complete volume contents, and
remove the helper:

```bash
docker create --name trade-rl-artifact-export --mount type=volume,source=trade-rl-training-data,target=/workspace/var,readonly alpine:3.20
docker cp trade-rl-artifact-export:/workspace/var/. ./training-export
docker rm trade-rl-artifact-export
```

The helper never starts and therefore cannot modify the mounted volume.
Remove an old helper with `docker rm trade-rl-artifact-export` before repeating
these commands if a previous copy was interrupted.

## Retry and recovery semantics

The maintained full runner does not resume an interrupted PPO or walk-forward
process. At startup it deletes and recreates the fixed
`binance-multitimeframe-full` work root. Therefore, preserve evidence from a
failed or interrupted generation before retrying.

First stop and remove a detached trainer if it is still present:

```bash
docker rm --force trade-rl-full-training
```

Export the volume as described above, or rename the prior work root to a unique
generation name. This example reserves `failed-001`; choose a new suffix for
every retry:

```bash
docker run --rm --mount type=volume,source=trade-rl-training-data,target=/workspace/var alpine:3.20 sh -c 'if [ -e /workspace/var/binance-multitimeframe-full ]; then mv /workspace/var/binance-multitimeframe-full /workspace/var/binance-multitimeframe-full.failed-001; fi'
```

Then start a fresh attempt:

```bash
docker compose -f compose.training.yaml run --rm trainer
```

The named volume survives container removal, but the new invocation is a fresh
run rather than checkpoint resume. Do not describe a retried run as resumed.

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
