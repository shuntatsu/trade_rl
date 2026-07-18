# Complete Trust-Boundary Remediation Design

## Status and scope

This specification replaces the incomplete trust-boundary hardening currently proposed by PR #67. It covers the maintained research path from Binance metadata ingestion through walk-forward selection, final training, fresh confirmation, serving bundle packaging, release approval, runtime activation, and privileged GPU operations. Production remains `NO-GO`; the goal is to make the research and release evidence truthful, reproducible, and fail-closed.

The user has approved remediation of every confirmed audit finding. No compatibility behavior may label unverified evidence as authenticated, selected-final, fresh-confirmed, or released.

## Security model

Maintained evidence uses Ed25519 signatures. Private keys exist only in offline approval commands or an external KMS-equivalent process. Trainers, CI verification jobs, serving runtimes, and Docker containers receive public keys only. HMAC artifacts remain loadable only as explicitly legacy, unauthenticated `NO-GO` evidence and cannot authorize selected-final training or released serving.

Every signed envelope binds the schema version, purpose, key ID, payload digest, signature algorithm, signature, and signing time. Public keys are purpose-bound and carry a validity interval. Verification rejects unknown keys, wrong purpose, invalid signature, expired/not-yet-valid keys, malformed base64, noncanonical payloads, and timestamps beyond a small configured clock skew.

## Phase-separated research state machine

The maintained full-research workflow is split into irreversible evidence stages:

1. `development_evaluation`: build immutable datasets, run sealed walk-forward, validate the entire walk-forward directory, evaluate the strict research gate and execution sensitivity, then produce a content-addressed selection proposal.
2. `awaiting_selection_authorization`: stop normally without training. An offline approver verifies the proposal and signs a selection authorization.
3. `selected_final_training`: consume the signed authorization and matching proposal, canonicalize the training config through the same function used by selection, reject all resume checkpoints, train the fixed seed ensemble, and publish `training_run_v3` as `research_selected_final`.
4. `awaiting_fresh_confirmation`: stop normally after selected-final training. The required confirmation start is the later of the frozen development end and final training completion time.
5. `fresh_confirmation_verified`: accept only a signed confirmation interval whose start is at or after the required boundary, whose end is no later than trusted current time plus clock skew, and whose identity matches the selected-final run.
6. `awaiting_release_approval`: package a serving bundle carrying the training run, selection authorization, and confirmation identities.
7. `released`: an offline approver signs a release attestation. Serving activation requires the complete signed chain.

Expected waiting states return success with a machine-readable status. Research rejection and infrastructure failure remain distinct nonzero outcomes.

## Selection proposal and authorization

`SelectionProposal` is generated only after `validate_walk_forward_run_directory()` succeeds. It binds the walk-forward manifest digest, exact gate artifact digest, exact execution-sensitivity digest, dataset ID, selected configuration, canonical candidate config digest, fixed seed tuple, source Git commit, dependency digest, and an empty resume-checkpoint set.

`SelectionAuthorization` version 2 is an Ed25519-signed approval of exactly one proposal digest. It additionally records approver, approval time, key ID, and expiry. The final-training entry point accepts a proposal and authorization together and verifies both before fitting normalizers or constructing a model. Selected-final training rejects any resume checkpoint. Exploratory training remains available but cannot produce a selected-final manifest.

Canonical config normalization is a single public function applied before candidate identity generation in walk-forward selection, proposal generation, and final training. It includes the maintained terminal liquidation contract and full-reward preroll, eliminating pre/post-normalization digest drift.

## Manifest and release identity

`TrainingRunManifest` advances to `training_run_v3` with first-class fields for `run_kind`, `selection_proposal_digest`, `selection_authorization_digest`, `walk_forward_run_digest`, `gate_evidence_digest`, and `completed_at`. The validator requires the corresponding sidecars for selected-final runs and recomputes their identities.

`ServingBundleManifest` advances to version 5 and binds the training run digest, run kind, selection authorization digest, and fresh-confirmation evidence digest. A released residual-policy bundle must be `research_selected_final` and contain all four identities.

`ReleaseAttestation` advances to an Ed25519-signed schema binding the bundle, training run, selection authorization, confirmation, dataset, environment, policy, source commit, dependency digest, approver, and approval time. Runtime activation rejects legacy release manifests and any incomplete identity chain.

A maintained packaging command performs:

`selected training directory -> validated serving bundle v5 -> offline release approval -> registry installation -> runtime smoke activation`.

## Fresh confirmation

Fresh confirmation evidence uses Ed25519 and includes `created_at`, interval start/end, return cadence, returns, recomputed performance metrics, policy/dataset/environment/training-run identities, order/fill/reconciliation digests, and the required-after boundary. Verification receives a trusted current time and maximum clock skew. It rejects overlap with development or training, future intervals, mismatched cadence, identity mismatches, nonpositive return, excessive drawdown, insufficient duration, and wrong key purpose.

The confirmation recheck command loads the required boundary from the selected-final summary and never accepts a caller-provided weaker boundary.

## Binance metadata evidence

The signed-history payload includes an explicit ordered `symbol_order` array separate from the symbol mapping. Core code exposes a verified-history factory that accepts the complete signed document and public-key trust store, verifies the Ed25519 signature, validates the semantic scope, and returns a `VerifiedBinanceRuleHistory`. `resolution_from_historical_signed()` accepts only that verified type.

Validation requires exact USD-M market, exact ordered symbols, exact research coverage, `issued_at >= coverage_end`, `issued_at <= trusted_now + clock_skew`, allowed source URI scheme, listing dates within coverage, strictly positive finite rules, strictly increasing effective times, initial rule coverage, and equality between the summary execution values and final effective rule.

The complete signed document, envelope, key ID, signature, and rule history are copied immutably into the run generation. Dataset identity references the signed document digest. No public constructor can assign `authentication=ed25519` without a verified document.

Frozen snapshot and conservative static modes remain explicitly unauthenticated and non-point-in-time.

## GPU and Docker operations

The training image pins the reviewed official Python 3.12 slim multi-platform digest. The build records source-tree, lockfile, and image identities. GitHub workflows check out the immutable event SHA and verify `HEAD == github.sha`; they never use mutable `ref: main` after approval.

Full CUDA training runs only on the protected self-hosted GPU runner. GitHub-hosted workflows perform contract validation and bounded CPU smoke only. Every CUDA workflow verifies the resolved PyTorch device and fails if CUDA is unavailable.

The supervisor uses a repository/project-specific label plus expected generation and commit. Status rejects missing, dead, paused, restarting, OOM-killed, unhealthy, wrong-generation, wrong-commit, and stale-heartbeat containers. It records image ID/digest, command, mounts, state error, OOM status, health, timestamps, and exit code. Stop captures inspect evidence and complete logs before removal.

A dedicated entry point maintains a heartbeat and writes CUDA preflight evidence beneath the immutable generation root. Expected waiting states do not look like infrastructure failures. Scheduled read-only status checks are separate from start/stop jobs that use the protected Environment.

Private signing keys are never passed into the trainer. Public-key files are mounted read-only. Named volumes and container labels are repository-specific.

## Workflow policy and supply chain

The workflow policy checker parses YAML structurally. It validates every workflow and job, including trigger types, self-hosted runner labels, job conditions, environments, permissions, immutable action SHAs, checkout ref, credential persistence, and prohibited write permissions. Comments and unreachable strings cannot satisfy policy.

All external Actions use immutable commit SHAs. The Python base image is digest-pinned. Dependency locking includes `cryptography==49.0.0` and `PyYAML` for the repository-owned workflow checker. CI retains diagnostics and adds critical branch-coverage thresholds for signing, authorization, metadata verification, run manifests, supervisor behavior, and release/runtime activation.

## Legacy-path removal

The old `run_full_research.py` ceases to be an independent maintained implementation. It becomes a thin compatibility launcher to the hardened state-machine entry point or is removed after all docs/tests are migrated. No maintained code loads another script through `runpy` or accesses private globals by string.

Legacy HMAC release and metadata APIs are removed from public exports and cannot produce authenticated identities. Legacy serving release manifests remain readable only for explicit `allow_legacy_no_go` inspection and are never activatable.

## Test strategy

Tests are written before implementation and cover:

- Ed25519 key-purpose, validity, tamper, wrong-key, malformed-signature, and private-key absence cases.
- Fake selection proposal, altered gate artifact, incomplete walk-forward directory, resume injection, config canonicalization drift, and expired authorization.
- Confirmation overlap, future interval, wrong identity, wrong purpose, insufficient duration, and valid post-boundary confirmation.
- Reordered symbol mappings with fixed `symbol_order`, forged verified-history construction, issued-at anomalies, rule-summary mismatch, and signed-document retention.
- Training manifest v3 and serving bundle v5 chain validation, exploratory-run release rejection, and complete package/approve/activate smoke.
- YAML policy bypass attempts using comments, unrelated jobs, mutable actions, mutable checkout refs, and hidden self-hosted runners.
- Supervisor missing/dead/OOM/unhealthy/stale/wrong-generation states and stop-log-before-remove ordering.
- A bounded end-to-end state-machine test from sealed walk-forward evidence through `awaiting_selection_authorization`, selected-final training stub, `awaiting_fresh_confirmation`, confirmation verification, release packaging, approval, and runtime activation.

The full existing test suite, Ruff, formatter, MyPy, import-linter, workflow policy, critical branch coverage, compatibility matrix, training-image build, non-root probe, and a real CUDA smoke must pass before PR #67 returns to Ready status.
