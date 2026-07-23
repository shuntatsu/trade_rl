# Environment Facade Audit Resolution — 2026-07-23

## 1. Decision

`AUD-RL-001` is resolved as a software-architecture finding.

The original audit recorded a P2 maintainability risk rather than a behavioral defect.
It protected mutable Gymnasium state and the stable reset/step/info APIs, rejected a
mechanical split, and deferred remediation until concrete action/risk/reward work
supplied testable typed seams.

PRs #92, #114, #120, #122, #125, #126, #140, and #152 supplied those seams. The
maintained facade now orchestrates typed owners while retaining the mutable state and
stable APIs the original audit explicitly required it to own.

Production remains `NO-GO`.

## 2. RED documentation-contract evidence

The RED head was:

- commit: `e9be30ce1a10d7695d8a5a00d682c5a0a655bc23`;
- CI run: `30017876462`;
- pytest artifact: `8568032136`;
- artifact digest:
  `sha256:6663569694cdc228f727a1e70f5aef15f9a05149344c22175a96b25408c5700c`.

Static checks, Mypy, Import Linter, Serving smoke, Ubuntu, Windows, and the complete
training image/non-root probe passed. Full pytest produced exactly one intended
failure:

- 1 failed;
- 1,343 passed;
- 2 skipped;
- 11 warnings;
- 84.25% total coverage.

The failure was
`test_environment_facade_audit_is_resolved_with_protected_state_boundary`, because the
closeout summary still classified `AUD-RL-001` as `OPEN RISK, FURTHER REDUCED`.

## 3. Resolution criteria

The finding is closed because:

- the constructor is bounded by a maintained 150-line architecture limit, down from
  the audited 479 lines;
- typed builders own observation, provider, runtime-service, portfolio-risk,
  policy/schedule, initial-state, and reward/execution construction;
- typed services own decision planning, risk projection, stateful target execution,
  termination, reward coordination, observation assembly, and information building;
- `step()` remains an orchestration facade and no longer owns extracted policy;
- `reset()` retains mutable Gymnasium state and stable reset semantics as required by
  the protected invariant;
- architecture tests prohibit extracted responsibilities from returning inline;
- extracted construction and wiring modules have permanent 100.0% critical coverage
  ratchets;
- no further mechanical split is justified without a new concrete behavior seam.

## 4. Capability boundary

Resolving this maintainability finding does not establish:

- profitability;
- exchange-equivalent fills or L2 realism;
- authenticated exchange/broker access;
- direct order submit/cancel/replace/reconciliation;
- production secret distribution, venue kill switches, or operational alerting;
- paper/live reconciliation evidence;
- required out-of-sample duration or statistical uplift.

Production status remains `NO-GO` until the independent research, evidence,
operational, authorization, and profitability gates pass.

## 5. Integration gate

This PR changes documentation and one executable documentation contract only. Before
merge, its exact final head must pass normal CI, complete pytest and coverage, critical
coverage checks, CLI smoke, Ubuntu and Windows compatibility, the complete training
image/non-root probe, and PostgreSQL Catalog validation.
