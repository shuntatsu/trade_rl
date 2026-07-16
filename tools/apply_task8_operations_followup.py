from __future__ import annotations

import re
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


def replace_pattern(path: str, pattern: str, replacement: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"missing Task 8 operations pattern in {path}: {pattern[:160]!r}")
    target.write_text(updated, encoding="utf-8")


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

## Maintained contract

The maintained dataset contains 226 ordered point-in-time channels: 59 on 15m, 59 on 1h, 55 on 4h, and 53 on 1d. The policy receives completed native windows of 96, 168, 120, and 60 bars, availability and staleness state, the current market snapshot, execution state, portfolio state, and finite-horizon state. Actions are direct target weights with a one-decision signal delay and hard pre-trade, liquidity, portfolio, and emergency-risk projection.

The structured policy uses timeframe-specific sequence encoders, a shared per-asset actor, and a portfolio-level critic. PPO uses an index-backed rollout: persistent storage contains decision indices and current state, while overlapping histories are reconstructed only for sampled minibatches. Behavior cloning uses an approximate portfolio teacher with executor-compatible minimum-notional, capacity, partial-fill, fee, spread, and delay semantics.

## Evidence protocol

The maintained gate requires six sealed folds covering at least 180 OOS days, a positive circular block-bootstrap lower confidence bound, acceptable drawdown, turnover and costs, and stable recipe selection. Final training retains the fixed three-seed ensemble rather than selecting a lucky seed. A separate fresh confirmation interval is opened only after development choices are frozen and must contain at least 30 sealed days with matching policy identity.

Structured sequence serving restores both normalizers and rebuilds the same Dict observation from a bounded rolling dataset. It rejects symbol order, feature order, cadence, sequence layout, and incomplete-bar drift. Live order routing remains outside the policy artifact.

## Running the complete example

```bash
uv sync --extra dev --extra train-sb3
uv run python examples/binance-multitimeframe/run_full_research.py \
  --work-root var/binance-multitimeframe-full
```

Production remains NO-GO until the OOS gate, fresh confirmation, CUDA verification, checkpoint recovery, structured serving parity, and paper-trading reconciliation are complete. The research workflow does not authenticate an account or place live orders.
''',
        encoding="utf-8",
    )

    replace_pattern(
        "docs/operations/docker-gpu-full-training.md",
        r"A\ncompact rollout buffer preserves float16 sequence/staleness.*?60-day windows\.\n",
        '''An index-backed rollout stores decision indices and non-overlapping current
state only. Native sequence tensors are reconstructed from the immutable dataset
for each sampled PPO minibatch, reducing the maintained persistent rollout
estimate from roughly 200.5 MiB to roughly 5.77 MiB. Configurations above the
configured memory ceiling still fail closed. Approximate portfolio teacher
artifacts likewise reconstruct only requested normalized sequence minibatches.
''',
    )
    replace_pattern(
        "docs/operations/docker-gpu-full-training.md",
        r"The preset freezes two non-overlapping 360-hour outer.*?audit that\nordering\.\n",
        '''The maintained gate requires six sealed walk-forward folds covering at
least 180 OOS days. It rejects a candidate when the circular block-bootstrap
lower confidence bound on mean daily log growth is non-positive, or when drawdown,
turnover, cost fraction, or selection stability violates configured limits. The
final run preserves the predetermined three-seed ensemble. A separate fresh
confirmation interval is opened only after the recipe and ensemble are frozen.
''',
    )
    replace_pattern(
        "docs/operations/docker-gpu-full-training.md",
        r"The maintained full runner does not resume an interrupted PPO or walk-forward.*?partial evidence\.\n",
        '''Each full invocation requires a new immutable generation name. Failed evidence
stays in its original generation while the next generation reuses the shared
market-data cache. A single-seed PPO member may continue through a validated
`resume_checkpoints` mapping; walk-forward orchestration restarts rather than
resuming in place.
''',
    )
    replace_once(
        "docs/operations/docker-gpu-full-training.md",
        '''The named volume survives container removal, but the new invocation is a fresh
run rather than checkpoint resume. Do not describe a retried run as resumed.
''',
        '''The named volume survives container removal. A new generation is a fresh
workflow invocation; only members explicitly bound through `resume_checkpoints`
continue prior policy state. Cache reuse alone is not checkpoint resume.
''',
    )


if __name__ == "__main__":
    main()
