# Environment Construction Decomposition — Current Main Design

Date: 2026-07-23

## Context

PR #114 is already merged and is the canonical owner of observation-contract construction. It extracted `EnvironmentObservationContractBuilder`, preserved flat and structured observation behavior, reduced `ResidualMarketEnv.__init__` from 540 to 321 lines, and established a 100.0% branch-coverage ratchet for `trade_rl/rl/environment_observation_contract.py`.

The remaining `AUD-RL-001` risk is therefore narrower than the original plan. The constructor still owns three unrelated construction responsibilities:

- provider, artifact, risk, action, timing, and reward dependency resolution;
- assembly of executors, sampler, and maintained step-time services;
- creation of initial mutable Gymnasium state.

This design starts from current main `cc1aac077c73c1ec5304236c0a9471b5bea9b106` and does not replace or duplicate PR #114.

## Goals

- Preserve the complete public `ResidualMarketEnv(...)` signature.
- Preserve environment, action, observation, and execution-policy identities.
- Preserve validation order/messages, reset/step results, and mutable-state ownership.
- Retain `EnvironmentObservationContractBuilder` unchanged as the observation owner.
- Reduce `ResidualMarketEnv.__init__` to at most 180 source lines.
- Introduce three stateless, typed construction owners.

## Non-goals

- No modification to `environment_observation_contract.py` or its 100.0% coverage threshold.
- No action, risk, reward, execution, episode, observation, or termination policy change.
- No public builder API.
- No direct exchange routing, production authorization, or profitability claim.

## Architecture

### Dependency resolution

`trade_rl.rl.environment_dependencies.EnvironmentDependencyResolver` receives an immutable `EnvironmentDependencyRequest` and returns immutable `EnvironmentDependencies`.

It owns:

- trend strategy and `MarketInputResolver` reconciliation;
- alpha enablement, provider requirements, and artifact identity;
- static/provider factor-basis validation and minimum-index contribution;
- pre-trade, portfolio-risk, and advanced risk-input-provider resolution;
- action-spec creation and validation;
- nominal episode/decision timing and reward preroll;
- reward-tracker creation.

It retains no environment instance and no cross-construction state.

### Service assembly

`trade_rl.rl.environment_assembly.EnvironmentServiceAssembler` receives an immutable assembly request and returns immutable `EnvironmentServiceAssembly`.

It owns construction of:

- emergency risk monitoring;
- separate hybrid and shadow executors;
- episode sampling;
- execution, observation, decision, risk, reward, information, and termination services.

It only composes existing maintained services; step-time policy remains in those services.

### Initial state

`trade_rl.rl.environment_state.EnvironmentInitialStateFactory` creates a fresh immutable result containing:

- initial indices and episode metadata;
- hybrid/shadow books;
- previous action, pending targets, and order books;
- position age, observation execution state, diagnostics, reward-history cache, and reset flag.

The factory retains no mutable state. `ResidualMarketEnv` explicitly assigns and subsequently owns every mutable field.

## Facade flow

The constructor performs only:

1. store dataset and raw optional normalizer/stress inputs;
2. call `EnvironmentDependencyResolver.resolve()` and assign resolved dependencies;
3. call the existing `EnvironmentObservationContractBuilder(...).build()` and assign its result;
4. call `EnvironmentServiceAssembler.assemble()` and assign maintained services;
5. compute the unchanged environment digest;
6. call `EnvironmentInitialStateFactory.create()` and explicitly assign mutable state.

Low-level provider, executor, sampler, book, and step-service construction is forbidden in the facade.

## Compatibility invariants

- `ResidualMarketEnv`, `AlphaProvider`, and `FactorBasisProvider` remain declared in `trade_rl.rl.environment`.
- Constructor parameter names, order, positional/keyword-only boundary, and defaults remain unchanged.
- `EnvironmentObservationContractBuilder` and its current public/result contracts remain unchanged.
- Hybrid and shadow executors remain distinct instances with equal execution-policy identity.
- The environment remains the only owner of episode-varying mutable state.
- No reflection-based state assignment is permitted.

## Test strategy

### Current-main RED

Before production modules exist, require:

- all three owner modules and classes;
- frozen request/result dataclasses;
- constructor span at most 180 lines;
- explicit dependency, observation, assembly, and state delegation;
- absence of low-level construction symbols in the constructor.

The expected current-main failure is missing three modules and a 321-line constructor. Existing unrelated tests must continue to pass.

### Characterization

Reuse the independently captured pre-refactor canonical payload SHA-256:

`9d6540b3e3d3616bbb41caff036c6ef37228af56506adb030229aead86b11de1`

The payload covers all public digests, spaces, timing, initial state, seeded reset, one seeded direct-target step, and post-step mutable state. PR #114 already preserves this observation behavior; the current refactor must preserve the complete payload again.

### Coverage

Keep the PR #114 100.0% per-file observation-builder ratchet. Add a separate measured aggregate branch-coverage ratchet for only:

- `environment_dependencies.py`;
- `environment_assembly.py`;
- `environment_state.py`.

No existing threshold may be lowered.

## Safety

Production remains `NO-GO`. Direct exchange routing and paper/live equivalence remain outside this change.