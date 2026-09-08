# Lean Strategy Family Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make the maintained strategy families visible in the filesystem as `rules`, `forecasts`, and `rl` while preserving every strategy's fit/inference behavior and the existing `trade_rl.strategies` public API.

**Architecture:** Keep `controls.py`, `interface.py`, and `position_intent.py` at strategy root. Move trend/mean-reversion to `rules/`, common forecast controller/supervised fitting/Ridge/LightGBM to `forecasts/`, and PPO to `rl/`. Internal module paths may break; package-level exports remain unchanged. This phase is a semantic-zero relocation and should appear as renames plus import rewrites, not algorithm edits.

**Tech Stack:** Python 3.12, NumPy, optional LightGBM/SB3/Torch, pytest, Ruff, Mypy, GitHub Actions.

**Spec:** `docs/specs/2026-09-08-lean-package-boundaries-design.md`.

## Global Constraints

- Base exact verified Phase 2B head: `2c4ba3fcbe7d7d0601495eba2815eb3c1b6a036f`.
- Preserve the exact current `trade_rl.strategies.__all__` set.
- Preserve PR #436 fit-symbol scope and zero-eligible-row fail-closed behavior for Ridge, LightGBM, and PPO.
- Preserve equal-symbol supervised weighting and PPO round-robin fit scope.
- Preserve strategy thresholds, controller semantics, model hyperparameters, seeds, observation/action semantics, and candidate-suite behavior.
- Do not keep forwarding files at old private paths.
- Production changes outside `trade_rl/strategies` must be import-only.
- Same-head full CI is required.

---

## Task 1: Define RED layout and public API contracts

**Files:**
- Create: `tests/architecture/test_lean_strategy_layout.py`

- [ ] Require final files:

```text
strategies/rules/{__init__.py,trend.py,mean_reversion.py}
strategies/forecasts/{__init__.py,controller.py,supervised.py,ridge.py,lightgbm.py}
strategies/rl/{__init__.py,ppo.py}
```

- [ ] Require old flat private files absent:

```text
forecast.py
supervised.py
ridge.py
lightgbm.py
trend.py
mean_reversion.py
ppo.py
```

- [ ] Freeze exact package-level public API from current `trade_rl.strategies.__all__`.
- [ ] Assert `rules`, `forecasts`, and `rl` do not import `trade_rl.evaluation`.
- [ ] Run `uv run pytest -q tests/architecture/test_lean_strategy_layout.py`; expected RED is structural only.
- [ ] Commit RED test only.

---

## Task 2: Move rule strategies

**Files:**
- Move: `trend.py` -> `rules/trend.py`
- Move: `mean_reversion.py` -> `rules/mean_reversion.py`
- Create: `rules/__init__.py`
- Rewrite internal/test imports.

- [ ] Move files without function/class body edits.
- [ ] Export `TrendIntentConfig`, `TrendIntentStrategy`, `MeanReversionIntentConfig`, `MeanReversionIntentStrategy` from `rules/__init__.py`.
- [ ] Run rule strategy tests and candidate-suite regressions.

---

## Task 3: Move forecast family

**Files:**
- Move: `forecast.py` -> `forecasts/controller.py`
- Move: `supervised.py` -> `forecasts/supervised.py`
- Move: `ridge.py` -> `forecasts/ridge.py`
- Move: `lightgbm.py` -> `forecasts/lightgbm.py`
- Create: `forecasts/__init__.py`

- [ ] Rewrite intra-family imports to new owners.
- [ ] Preserve exact Ridge/LightGBM signatures including `fit_symbol_indices` / scope contracts introduced by #436.
- [ ] Run supervised weighting, Ridge, LightGBM and candidate-suite tests.

---

## Task 4: Move PPO family

**Files:**
- Move: `ppo.py` -> `rl/ppo.py`
- Create: `rl/__init__.py`

- [ ] Rewrite imports only.
- [ ] Preserve PPO environment, action/reward/training semantics, seed and fit-symbol scope behavior.
- [ ] Run PPO strategy tests and candidate-suite scope tests.

---

## Task 5: Rebuild package facade and delete old private paths

- [ ] Update `trade_rl/strategies/__init__.py` to import from new family packages while preserving exact `__all__`.
- [ ] Rewrite every repository import of old flat strategy modules.
- [ ] Require no executable old import/path remains and no forwarding shim exists.
- [ ] Run all `tests/strategies`, candidate suite/run artifact tests, and architecture tests.

---

## Task 6: Final falsification and exact-head gate

- [ ] `uv run ruff check trade_rl tests`
- [ ] `uv run ruff format --check trade_rl tests`
- [ ] `uv run mypy trade_rl`
- [ ] `uv run pytest -q tests`
- [ ] package identity
- [ ] Compare exact Phase 2B base to final head; moved strategy implementation files should be 100% renames except import-owner lines required by intra-family relocation.
- [ ] Explicitly verify no threshold/hyperparameter/model-body diff.
- [ ] Recheck fit-symbol/evaluation-symbol separation, supervised equal-symbol weighting, PPO round-robin scope, and package-level public symbols.
- [ ] Require normal same-head GitHub Actions success; keep Draft and do not merge.
