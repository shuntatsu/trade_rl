# Docker GPU Full Training Design

## Understanding summary

- Run the maintained `agent/full-training-source-export` research pipeline to completion.
- Keep downloaded market data, deterministic datasets, checkpoints, policies, and evaluation evidence inside a Docker named volume.
- Use the single RTX 4050 Laptop GPU for PPO optimization and parallel CPU environments for rollout collection.
- Treat the dataset's 96 causal indicators separately from the 1,252-element full policy observation observed in the authoritative CPU artifact; it also contains masks, staleness, missing reasons, execution state, factor loadings, and book state.
- Select a network that is materially larger than the existing preset while remaining safe for 6 GiB VRAM.
- Require positive cost-adjusted sealed-test return, non-negative uplift over the Trend baseline, and maximum drawdown no greater than 20%.
- Keep production status `NO-GO`; this is research training, not live order routing or a profitability guarantee.

## Assumptions

- The maintained BTCUSDT, ETHUSDT, and BNBUSDT universe and four native clocks (`15m`, `1h`, `4h`, `1d`) remain authoritative.
- Three seeds and two sealed outer folds remain required.
- The host has one CUDA-capable GPU, so seeds run sequentially. Parallelism is inside each seed through vectorized environments.
- Docker Desktop exposes the GPU through the Compose `gpus: all` contract.
- Named-volume data may be reused by later container invocations, but published dataset identities are always revalidated before use.
- The final evidence run freezes two 360-hour outer windows from 2026-06-01 through 2026-07-01. Pre-June outer windows are development validation and are not reused as sealed evidence.

## Architecture

The training configuration gains `n_envs`, which is included in its digest and validated as a positive integer. PPO's rollout batch becomes `n_steps * n_envs`; the batch size must divide that product. The SB3 backend probes one unwrapped environment for identity and observation layout, then uses a subprocess vector environment when `n_envs > 1`. Checkpoint scheduling continues to use global observed timesteps, so retained artifacts remain comparable across vector widths.

The full preset uses CUDA, four rollout environments, a `[256, 256]` policy/value network, and 128-dimensional asset and global embeddings. This is large enough for the 1,252-element observation while staying comfortably below the 6 GiB VRAM limit. Seeds remain sequential to avoid GPU memory contention and unstable throughput.

`Dockerfile.training` builds the Python 3.12 training image. `compose.training.yaml` grants GPU access and mounts the `trade-rl-training-data` named volume at `/workspace/var`. The maintained runner receives `/workspace/var/binance-multitimeframe-full` as its work root, so cache, datasets, configs, logs, checkpoints, policies, and evaluation artifacts never depend on a host data directory.

## Data flow

1. A CUDA preflight verifies that PyTorch can see a GPU and records device name and memory.
2. Binance Vision data is downloaded into the named-volume cache.
3. Two independent dataset artifacts are built and their IDs and digests are compared.
4. The training configuration is materialized next to the dataset and factor artifact.
5. Three PPO seeds train sequentially, each with four parallel rollout environments and GPU updates.
6. Two-fold walk-forward evaluation keeps the top three checkpoint-validation policies per seed, chooses the fold policy on configuration-selection data, and opens each post-June outer test only after selection.
7. A final gate writes machine-readable evidence and fails the process when any required threshold is missed.

## Failure and recovery

CUDA fallback is forbidden for the maintained Docker run. Dataset identity mismatch, non-finite observations, missing checkpoints, incomplete folds, or missing evaluation evidence fail closed. Named-volume inputs and caches remain available after failure. Published run directories stay atomic; a rerun uses a new work-root generation rather than silently overwriting verified evidence.

## Verification strategy

Configuration and backend changes follow RED-GREEN TDD. Docker assets receive contract tests that parse the Compose YAML as text and verify the volume, GPU, work-root, and command. A CUDA smoke run proves container GPU visibility and a small vectorized PPO update before full execution. Repository lint, formatting, typing, import boundaries, and the full test suite must remain green. Completion requires the final `summary.json`, three member policies with checkpoints, two sealed folds, and a passing return/drawdown gate.

## Decision log

- Chose one GPU trainer plus four CPU rollout environments over concurrent GPU seeds because the host has one 6 GiB GPU.
- Chose `[256, 256]` with 128-dimensional embeddings over the existing `[128, 128]`/64 preset to match the true observation width without unnecessary VRAM risk.
- Kept 96 causal indicators rather than inventing extra indicators; the larger quantity the user recalled is the expanded policy observation.
- Chose a Docker named volume as the sole persistent data exchange boundary.
- Chose fail-closed CUDA and profitability gates so CPU fallback or negative sealed results cannot be reported as completion.
