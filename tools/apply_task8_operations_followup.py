from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(
            f"missing Task 8 operations follow-up anchor in {path}: {old[:160]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        "tests/examples/test_docker_training_assets.py",
        '    combined = "\n".join((architecture, research, runbook))\n',
        '    combined = "\\n".join((architecture, research, runbook))\n',
    )
    replace_once(
        "examples/binance-multitimeframe/run_gpu_training_smoke.py",
        '''    resolved = dict(result)
    actual_timesteps = int(resolved.get("actual_timesteps", 0))
    return resolved, {
''',
        '''    resolved = dict(result)
    artifact_path = Path(str(resolved["artifact_path"]))
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    ensemble_payload = json.loads(
        (artifact_path / "ensemble.json").read_text(encoding="utf-8")
    )
    actual_timesteps = int(ensemble_payload["actual_timesteps"])
    resolved["actual_timesteps"] = actual_timesteps
    return resolved, {
''',
    )
    replace_once(
        "docs/ARCHITECTURE.md",
        "The BC teacher is an approximate portfolio teacher, not an exact optimal Oracle.",
        "The BC teacher is an approximate portfolio teacher rather than a globally optimal oracle.",
    )

    (ROOT / "docs" / "MULTITIMEFRAME_RESEARCH.md").write_text(
        '''# Native Multi-Timeframe Research

Trade RL treats multi-timeframe market context as a causal dataset contract. The maintained Binance example makes decisions every 15 minutes while computing each feature only on its own completed native clock (`15m`, `1h`, `4h`, and `1d`). Availability-aware as-of alignment prevents backward filling, incomplete-bar use, and future Ichimoku shifts.

## Maintained research contract

The maintained example uses:

- decision clock: `15m`;
- instruments: `BTCUSDT`, `ETHUSDT`, and `BNBUSDT` USDⓈ-M futures;
- development range ending `2026-06-01T00:00:00Z`, with later data reserved for sealed evaluation and fresh confirmation;
- 226 ordered point-in-time channels: 59 on 15m, 59 on 1h, 55 on 4h, and 53 on 1d;
- completed native sequence windows of 96, 168, 120, and 60 bars;
- direct target-weight actions, one shared per-asset actor head, and a portfolio-level critic;
- a one-decision signal delay and hard pre-trade, liquidity, portfolio, and emergency-risk projection;
- three fixed seeds retained as the final ensemble rather than selecting a lucky representative seed;
- six sealed walk-forward folds covering at least 180 OOS days, followed by a separate fresh confirmation interval.

The sequence channels include causal returns, volatility, volume, funding, momentum, trend, range, and Ichimoku-derived distances. Ichimoku values use only information available at the native bar close; forward-shifted Senkou and backward-shifted Chikou chart conventions are not fed to the policy. Every value is accompanied by availability and staleness state, and undefined cross-asset correlation or beta remains missing rather than becoming a valid zero.

## Model and learning evidence

The policy consumes structured Dict observations. Timeframe-specific encoders preserve temporal order, the shared per-asset actor is permutation-equivariant over the maintained symbol set, and the critic pools portfolio context. PPO uses an index-backed rollout: persistent storage contains decision indices and current state, while overlapping sequence tensors are reconstructed from the immutable dataset only for sampled minibatches.

Behavior cloning uses an approximate portfolio teacher. It is explicitly not treated as a proof of global optimality; its bounded beam applies executor-compatible minimum-notional, capacity, partial-fill, fee, spread, and signal-delay semantics. BC, network, feature, and sequence-window ablations remain required research evidence.

## Statistical protocol

A positive average return is insufficient. The maintained gate requires at least six folds and 180 OOS days, a positive circular block-bootstrap lower confidence bound on mean daily log growth, acceptable drawdown, turnover and cost fractions, and stable recipe selection. Final training retains the predetermined three-seed ensemble.

A fresh confirmation interval is opened only after development and walk-forward choices are frozen. It requires at least 30 sealed days, positive return, acceptable drawdown, and exact policy identity. Structured sequence serving then rebuilds the same normalized observation from a bounded rolling dataset while rejecting symbol, feature-order, cadence, layout, or incomplete-bar drift.

## Running the complete example

```bash
uv sync --extra dev --extra train-sb3
uv run python examples/binance-multitimeframe/run_full_research.py \
  --work-root var/binance-multitimeframe-full
```

The runner reuses immutable Binance Vision archives, records exchange-metadata provenance, rebuilds and identity-checks the dataset, executes the maintained walk-forward protocol, and publishes content-addressed artifacts. Dataset artifacts from older feature contracts are rejected rather than silently reused.

Production remains NO-GO until the OOS gate, fresh confirmation, CUDA verification, checkpoint recovery, structured serving parity, and live paper-trading reconciliation are complete. The research workflow does not authenticate an account or place live orders.
''',
        encoding="utf-8",
    )

    replace_once(
        "docs/operations/docker-gpu-full-training.md",
        '''A
compact rollout buffer preserves float16 sequence/staleness and uint8 availability
dtypes, reducing the maintained estimate from 473,122,816 bytes to about
200,495,104 bytes; configurations above 768 MiB still fail closed. Structured
Oracle teacher artifacts store compact decision indices and reconstruct only the
requested normalized sequence mini-batch, avoiding duplication of overlapping
60-day windows.
'''.replace("A\ncompact", "A compact"),
        '''An index-backed rollout stores decision indices and non-overlapping current
state only. Native sequence tensors are reconstructed from the immutable dataset
for each sampled PPO minibatch, reducing the maintained persistent rollout
estimate from roughly 200.5 MiB to roughly 5.77 MiB. Configurations above the
configured memory ceiling still fail closed. Approximate portfolio teacher
artifacts likewise store compact decision indices and reconstruct only requested
normalized sequence minibatches.
''',
    )
    replace_once(
        "docs/operations/docker-gpu-full-training.md",
        '''The preset freezes two non-overlapping 360-hour outer
windows covering `2026-06-01T00:00:00Z` through
`2026-07-01T00:00:00Z`. Earlier outer windows were used during development and
must not be described as sealed evidence. Within each fold, checkpoint data retains one predeclared finalist per seed.
Configuration selection evaluates the full seed distribution and rejects a
candidate when its median return is non-positive, its worst seed/dispersion is
unstable, or turnover, cost fraction or drawdown exceeds the configured limit.
The sealed outer window is opened only for the deterministic median
representative. Final full-data training is blocked unless both folds agree on
the same eligible recipe and representative seed. Each fold artifact records the
full experiment-plan digest and sealed-access digest needed to audit that
ordering.
''',
        '''The maintained gate requires six sealed walk-forward folds covering at
least 180 OOS days. It rejects a candidate when the circular block-bootstrap
lower confidence bound on mean daily log growth is non-positive, or when drawdown,
turnover, cost fraction, or selection stability violates the configured limits.
The final full-data run preserves the predetermined three-seed ensemble rather
than selecting a representative seed. A separate fresh confirmation interval is
opened only after the recipe and ensemble are frozen, and every fold records the
experiment-plan and sealed-access digests needed to audit that ordering.
''',
    )
    replace_once(
        "docs/operations/docker-gpu-full-training.md",
        '''The maintained full runner does not resume an interrupted PPO or walk-forward
process. Each invocation requires a new generation name. Failed and interrupted
evidence stays in its original generation, while the next invocation reuses the
shared cache. The runner refuses an existing generation even if it is empty or
contains only partial evidence.
''',
        '''Each full invocation still requires a new immutable generation name. Failed
and interrupted evidence stays in its original generation, while the next
generation reuses the shared market-data cache. A single-seed PPO member may be
continued by mapping that seed to a validated checkpoint directory through
`resume_checkpoints`; the loader verifies seed, algorithm, environment identity,
training-config identity, and observed timestep. Walk-forward orchestration is
restarted rather than resumed in place.
''',
    )
    replace_once(
        "docs/operations/docker-gpu-full-training.md",
        '''The named volume survives container removal, but the new invocation is a fresh
run rather than checkpoint resume. Do not describe a retried run as resumed.
''',
        '''The named volume survives container removal. A new generation is a fresh
workflow invocation; only members explicitly bound through `resume_checkpoints`
continue from prior policy state. Do not describe cache reuse alone as checkpoint
resume.
''',
    )


if __name__ == "__main__":
    main()
