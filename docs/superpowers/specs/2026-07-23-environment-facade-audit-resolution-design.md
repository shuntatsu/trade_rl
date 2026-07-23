# Environment Facade Audit Resolution Design

## Purpose

Resolve `AUD-RL-001` only if the original risk conditions are no longer present, without performing another mechanical split of `ResidualMarketEnv`.

The original audit did not reproduce a behavioral defect. It recorded a P2 maintainability risk because `environment.py` was 1,620 lines, `ResidualMarketEnv.__init__()` spanned 479 lines, `step()` spanned 223 lines, and future action/risk/reward work could re-concentrate policy in the facade. It explicitly protected mutable Gymnasium state and the stable reset/step/info APIs, and recommended waiting for concrete feature seams rather than splitting mechanically.

## Resolution criteria

The finding can be marked `RESOLVED` when all of the following are true:

- constructor and runtime wiring are delegated through typed owners;
- action planning, risk projection, execution coordination, termination, reward, observation, and information assembly are delegated to independently tested services;
- `reset()` continues to own application of mutable Gymnasium episode state;
- `step()` remains the stable orchestration API while policy and information construction stay outside the facade;
- architecture tests prohibit extracted policy and construction responsibilities from returning inline;
- the constructor has a maintained 150-line limit;
- extracted contracts and wiring modules have permanent coverage ratchets;
- exact-head CI, Ubuntu, Windows, training-image/non-root, CLI, and PostgreSQL checks pass;
- production remains `NO-GO` and no profitability or exchange-realism claim is introduced.

## Implementation

Add a documentation contract that requires the closeout summary and finding section to classify `AUD-RL-001` as `RESOLVED`, records the protected reset/step facade responsibilities, requires the 150-line limit and typed ownership controls, and preserves production `NO-GO`.

Update the architecture closeout and add a finding-specific resolution verification document. Do not modify Python production code, existing behavior, public APIs, coverage thresholds, or workflow permissions.

## Non-goals

- reducing `reset()` or `step()` line counts through mechanical extraction;
- moving mutable state ownership out of the Gymnasium facade;
- claiming exchange-equivalent execution, profitability, authorization, or production readiness;
- rewriting historical finding-specific verification documents that accurately recorded intermediate states.
