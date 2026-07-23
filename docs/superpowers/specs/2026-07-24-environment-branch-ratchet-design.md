# Environment Branch Coverage Ratchet Design

Date: 2026-07-24

## Goal

Protect the mutable Gymnasium facade against regressions by testing currently unprotected decision branches and enforcing a per-file branch-coverage floor for `trade_rl/rl/environment.py`.

## Scope

The environment remains the owner of mutable episode state. This change does not split the facade again or change action, reward, execution, observation, reset, or termination semantics.

## Test focus

Focused tests exercise:

- legacy object and callable alpha providers;
- static, object, and callable factor-basis providers;
- invalid factor shape and non-finite output rejection;
- cash, baseline, random, stress, partial-fill, and restore initial states;
- restore-state identity and value rejection;
- regular-cadence bar resolution and invalid pre-roll intervals;
- terminal-step rejection.

## Ratchet

The existing full-suite branch coverage is 56.25%. New tests must raise it to at least 75.0%, after which `pyproject.toml` records:

```toml
"trade_rl/rl/environment.py" = 75.0
```

The threshold may be raised to the observed stable value but must not be set below 75.0%.

## Safety

Tests use deterministic synthetic data and zero execution costs. No production behavior, schema identity, or public API is changed. Production remains `NO-GO`.