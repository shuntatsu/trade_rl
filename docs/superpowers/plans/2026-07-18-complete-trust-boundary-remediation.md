# Complete Trust-Boundary Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incomplete PR #67 trust boundaries with an externally approved, Ed25519-signed, phase-separated research and release chain that is reproducible and fail-closed.

**Architecture:** Introduce common asymmetric evidence primitives, then rebuild metadata, selection, confirmation, manifests, release packaging, and GPU supervision around verified typed evidence. The maintained full workflow becomes a state machine that pauses normally for external approvals instead of generating and approving its own evidence.

**Tech Stack:** Python 3.12, dataclasses, `cryptography==49.0.0`, PyYAML, pytest, Hypothesis, GitHub Actions, Docker Compose, Stable-Baselines3, import-linter.

## Global Constraints

- Production status remains exactly `NO-GO`.
- Trainers and serving runtimes receive public keys only; private keys are restricted to offline approval commands.
- Selected-final training rejects resume checkpoints.
- No maintained workflow checks out mutable `main`; it checks out and verifies the event SHA.
- No maintained API may label unsigned evidence as authenticated, selected-final, fresh-confirmed, or released.
- Existing exploratory research remains usable but cannot cross the release boundary.
- Every behavior change follows red-green-refactor and receives adversarial tests.

---

### Task 1: Add asymmetric evidence primitives and locked dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `trade_rl/release/asymmetric.py`
- Test: `tests/release/test_asymmetric_evidence.py`

**Interfaces:**
- Produces: `PublicVerificationKey`, `SignedEnvelope`, `sign_payload_ed25519()`, `verify_payload_ed25519()`, `load_public_key_store()`.

- [ ] **Step 1: Write failing tests** for key purpose, validity intervals, wrong keys, tampering, malformed base64, and verification with public material only.
- [ ] **Step 2: Run** `uv run pytest tests/release/test_asymmetric_evidence.py -q` and confirm imports/functions are missing.
- [ ] **Step 3: Implement** canonical Ed25519 envelopes. Sign `canonical_json_bytes({"key_id": ..., "payload_digest": ..., "purpose": ..., "schema_version": ..., "signed_at": ...})`; serialize public keys and signatures with strict base64; enforce aware UTC timestamps and key validity.
- [ ] **Step 4: Add** `cryptography==49.0.0` to runtime dependencies and `PyYAML>=6.0.2,<7` to dev dependencies; regenerate `uv.lock` with `uv lock`.
- [ ] **Step 5: Run** the targeted tests, Ruff, and MyPy.
- [ ] **Step 6: Commit** `feat: add asymmetric evidence primitives`.

### Task 2: Build verified Binance signed-history evidence

**Files:**
- Modify: `trade_rl/workflows/binance_metadata_modes.py`
- Modify: `examples/binance-multitimeframe/run_full_research_hardened.py`
- Test: `tests/workflows/test_binance_metadata_modes.py`
- Test: `tests/examples/test_binance_metadata_mode_runner.py`

**Interfaces:**
- Produces: `VerifiedBinanceRuleHistory`, `load_verified_binance_rule_history(path, trusted_keys, trusted_now, max_clock_skew)`, and `resolution_from_historical_signed(verified_history, start_time, end_time)`.

- [ ] **Step 1: Write failing tests** proving a raw scope cannot create authenticated evidence, `symbol_order` is signed explicitly, issued-at bounds are enforced, summary values equal the final rule, and the complete signed document is retained.
- [ ] **Step 2: Run** the two targeted test modules and confirm the current scope-only constructor passes forged input or lacks the new API.
- [ ] **Step 3: Implement** the verified factory and change the resolution function to accept only `VerifiedBinanceRuleHistory`.
- [ ] **Step 4: Persist** `exchange-info.signed.json` immutably and reference its digest from dataset identity.
- [ ] **Step 5: Remove** raw HMAC secret loading from the maintained runner; load purpose-bound public keys from a read-only file.
- [ ] **Step 6: Run** targeted tests, MyPy, and metadata critical coverage.
- [ ] **Step 7: Commit** `fix: require verified Binance history evidence`.

### Task 3: Canonicalize final-training identity and create signed selection approvals

**Files:**
- Create: `trade_rl/workflows/selection_proposal.py`
- Replace: `trade_rl/workflows/selection_authorization.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/cli/extended.py`
- Test: `tests/workflows/test_selection_authorization.py`
- Test: `tests/workflows/test_training_run_selection.py`

**Interfaces:**
- Produces: `canonicalize_selected_training_config()`, `SelectionProposal`, `SelectionAuthorization`, `authorize_selection_proposal()`, `verify_selection_authorization()`.

- [ ] **Step 1: Write failing tests** for the `liquidate_on_end` digest drift, incomplete walk-forward artifact, altered gate evidence, expired approval, wrong purpose/key, and resume-checkpoint injection.
- [ ] **Step 2: Run** targeted tests and confirm current code fails the desired behavior.
- [ ] **Step 3: Implement** one canonicalizer used before every candidate digest calculation.
- [ ] **Step 4: Implement** proposal generation only from a validated walk-forward directory and explicit gate/sensitivity artifacts.
- [ ] **Step 5: Implement** Ed25519 authorization of the proposal digest with approver, approval time, expiry, and empty resume set.
- [ ] **Step 6: Change** selected-final training to require proposal + signed authorization + public trust store, verify before normalizer fitting, and reject any resume checkpoint.
- [ ] **Step 7: Add CLI commands** to create a proposal and authorize it offline; never expose private-key parameters in the training command.
- [ ] **Step 8: Run** targeted tests, Ruff, MyPy, and existing training tests.
- [ ] **Step 9: Commit** `fix: externally authorize selected-final training`.

### Task 4: Upgrade training manifests to schema v3

**Files:**
- Modify: `trade_rl/artifacts/run_manifest.py`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/artifacts/test_run_manifests_v2.py`
- Test: `tests/artifacts/test_run_manifest_critical_coverage.py`

**Interfaces:**
- Produces: `TrainingRunManifest` schema `training_run_v3` with `run_kind`, proposal/authorization/walk-forward/gate identities, and `completed_at`.

- [ ] **Step 1: Write failing tests** that exploratory manifests reject selected-final fields, selected-final manifests require all sidecars, and sidecar digests are recomputed.
- [ ] **Step 2: Run** manifest tests and confirm v2 lacks the required fields.
- [ ] **Step 3: Implement** schema v3 build/load/validation while allowing explicit legacy v2 inspection only.
- [ ] **Step 4: Update** training publication to populate the typed fields and validate sidecars.
- [ ] **Step 5: Run** manifest, store, training, and critical coverage tests.
- [ ] **Step 6: Commit** `feat: bind selected-final identity in training manifests`.

### Task 5: Enforce genuinely fresh confirmation

**Files:**
- Modify: `trade_rl/evaluation/confirmation.py`
- Modify: `examples/binance-multitimeframe/recheck_confirmation.py`
- Modify: maintained full-run state machine file from Task 8
- Test: `tests/evaluation/test_confirmation.py`
- Test: `tests/examples/test_binance_multitimeframe_full_assets.py`

**Interfaces:**
- Produces: Ed25519 `FreshConfirmationEvidence` v4 and `verify_confirmation(expected_required_after, trusted_now, max_clock_skew, ...)`.

- [ ] **Step 1: Write failing tests** for development overlap, training overlap, future interval, insufficient days, wrong identity, wrong purpose/key, and a valid post-boundary interval.
- [ ] **Step 2: Run** targeted tests and observe current acceptance of overlapping evidence.
- [ ] **Step 3: Implement** signed v4 evidence with created time and required-after boundary.
- [ ] **Step 4: Make** recheck load the immutable required boundary from the selected-final summary; reject caller weakening.
- [ ] **Step 5: Replace** raw confirmation secrets with read-only public-key stores.
- [ ] **Step 6: Run** targeted tests and gate tests.
- [ ] **Step 7: Commit** `fix: require post-training fresh confirmation`.

### Task 6: Carry authorization and confirmation into serving and release

**Files:**
- Modify: `trade_rl/serving/bundle.py`
- Modify: `trade_rl/release/attestation.py`
- Modify: `trade_rl/release/offline_approval.py`
- Modify: `trade_rl/serving/registry.py`
- Modify: `trade_rl/serving/runtime.py`
- Create: `trade_rl/workflows/release_package.py`
- Modify: `trade_rl/cli/extended.py`
- Test: `tests/release/test_attestation.py`
- Test: `tests/serving/test_bundle_manifest_critical_coverage.py`
- Test: `tests/serving/test_registry_runtime_critical_coverage.py`
- Test: `tests/e2e/test_research_to_serving_v2.py`

**Interfaces:**
- Produces: serving bundle v5, Ed25519 release attestation v3, `package_selected_release()`, and CLI package/approve/activate smoke commands.

- [ ] **Step 1: Write failing tests** showing exploratory runs, missing authorization, missing confirmation, and legacy release manifests cannot activate.
- [ ] **Step 2: Run** targeted tests and confirm current release boundary accepts an incomplete chain or lacks packaging.
- [ ] **Step 3: Add** training run, run kind, selection authorization, and confirmation identities to bundle v5.
- [ ] **Step 4: Replace** HMAC release approval with offline Ed25519 approval and public-key runtime verification.
- [ ] **Step 5: Add** maintained package command that validates the complete training directory and copies every policy/normalizer/sidecar declared by the bundle.
- [ ] **Step 6: Add** registry install and activation smoke command; legacy bundles remain inspectable only with an explicit no-activation flag.
- [ ] **Step 7: Run** release, serving, registry, and end-to-end tests.
- [ ] **Step 8: Commit** `fix: bind release activation to the full evidence chain`.

### Task 7: Replace the text-matching workflow checker with structural policy

**Files:**
- Replace: `.github/check_workflow_security.py`
- Test: `tests/architecture/test_workflow_security_policy.py`
- Modify: `pyproject.toml` critical coverage configuration

**Interfaces:**
- Produces: `validate_workflow_security(root) -> tuple[str, ...]` using parsed YAML and job-level policy.

- [ ] **Step 1: Write failing tests** for comment bypass, unrelated-job bypass, hidden self-hosted labels, mutable actions, mutable checkout refs, persisted credentials, and contents write.
- [ ] **Step 2: Run** the policy tests and confirm the existing string checker is bypassed.
- [ ] **Step 3: Implement** structural parsing with PyYAML, accounting for YAML 1.1 `on` key normalization, and validate every job.
- [ ] **Step 4: Enforce** immutable action SHA references across all workflows, not only self-hosted jobs.
- [ ] **Step 5: Add** the checker and new trust-boundary modules to critical branch coverage.
- [ ] **Step 6: Run** policy tests and the checker against the repository.
- [ ] **Step 7: Commit** `fix: structurally validate workflow security`.

### Task 8: Introduce a phase-separated maintained full-run state machine

**Files:**
- Create: `trade_rl/workflows/full_research_state.py`
- Create: `examples/binance-multitimeframe/run_full_research_state.py`
- Replace: `examples/binance-multitimeframe/run_full_research.py` with a thin launcher
- Replace: `examples/binance-multitimeframe/run_full_research_hardened.py` with a thin launcher
- Modify: `examples/binance-multitimeframe/recheck_confirmation.py`
- Test: `tests/examples/test_full_research_state.py`

**Interfaces:**
- Produces statuses `awaiting_selection_authorization`, `awaiting_fresh_confirmation`, `awaiting_release_approval`, `complete_no_go`, and explicit rejection/failure outcomes.

- [ ] **Step 1: Write failing state-machine tests** using injected stage runners and artifact validators.
- [ ] **Step 2: Run** the tests and confirm the monolithic runner cannot express waiting states.
- [ ] **Step 3: Extract** maintained logic from script globals into typed workflow functions; do not use `runpy` or private-global lookup.
- [ ] **Step 4: End** after proposal generation when authorization is absent, with status `awaiting_selection_authorization` and exit 0.
- [ ] **Step 5: End** after selected-final training when confirmation is absent, with status `awaiting_fresh_confirmation` and exit 0.
- [ ] **Step 6: Keep** research rejection and infrastructure error nonzero and distinguish them in summary schema.
- [ ] **Step 7: Convert** old runners to thin argument-compatible launchers.
- [ ] **Step 8: Run** state, runner, and CLI smoke tests.
- [ ] **Step 9: Commit** `refactor: phase the maintained full research workflow`.

### Task 9: Harden Docker image, provenance, heartbeat, and supervisor

**Files:**
- Modify: `Dockerfile.training`
- Modify: `compose.training.yaml`
- Create: `examples/binance-multitimeframe/full_run_entrypoint.py`
- Replace: `examples/binance-multitimeframe/full_run_supervisor.py`
- Test: `tests/examples/test_docker_training_assets.py`
- Test: `tests/examples/test_full_run_supervisor.py`

**Interfaces:**
- Produces: generation-scoped preflight/heartbeat, repository-specific labels, strict status result, log-before-remove stop, image/source/lock evidence.

- [ ] **Step 1: Write failing tests** for absent/dead/paused/restarting/OOM/unhealthy/stale/wrong-generation states and stop ordering.
- [ ] **Step 2: Run** tests and confirm current status returns success for absent/dead and removes before external log capture.
- [ ] **Step 3: Pin** both Docker stages to `python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf`.
- [ ] **Step 4: Add** source-tree and lockfile digests to build/runtime provenance and verify the workflow-computed values.
- [ ] **Step 5: Implement** an entry point with a background heartbeat and generation-scoped CUDA preflight.
- [ ] **Step 6: Implement** strict supervisor identity and state checks, image/mount/health evidence, and capture logs before removal.
- [ ] **Step 7: Replace** secret environment variables with read-only public-key file mounts.
- [ ] **Step 8: Run** supervisor tests, Compose config, image build, and non-root probe.
- [ ] **Step 9: Commit** `fix: harden full-run container supervision`.

### Task 10: Correct and separate GitHub workflows

**Files:**
- Modify: `.github/workflows/launch-binance-frozen-226.yml`
- Modify: `.github/workflows/gpu-nightly.yml`
- Modify: `.github/workflows/multitimeframe-live-full.yml`
- Modify: `.github/workflows/binance-live-smoke.yml`
- Modify: `.github/workflows/full-training-capability-audit.yml`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: protected start/stop, unprivileged scheduled status, event-SHA checkout, CUDA gate, hosted contract validation, complete artifact retention.

- [ ] **Step 1: Extend workflow-policy tests** with the desired job structures and verify current YAML fails.
- [ ] **Step 2: Change** every checkout to `${{ github.sha }}` and verify `git rev-parse HEAD` equals it.
- [ ] **Step 3: Separate** protected start/stop jobs from scheduled read-only status so status does not require Environment approval.
- [ ] **Step 4: Require** actual CUDA in every GPU job and record device evidence.
- [ ] **Step 5: Replace** the hosted six-hour full-training workflow with contract validation and bounded CPU smoke; full model training remains on the GPU runner.
- [ ] **Step 6: Upload** complete immutable run directories or a validated archive, never manifests without policies/normalizers.
- [ ] **Step 7: Pin** every external action to a 40-character SHA.
- [ ] **Step 8: Run** structural checker and YAML tests.
- [ ] **Step 9: Commit** `fix: make privileged workflows immutable and observable`.

### Task 11: Restore architecture boundaries and documentation

**Files:**
- Modify: `.importlinter`
- Move framework-specific behavior cloning implementation into `trade_rl/integrations/`
- Modify: `docs/MULTITIMEFRAME_RESEARCH.md`
- Modify: `docs/operations/docker-gpu-full-training.md`
- Modify: `docs/ARCHITECTURE.md`
- Test: `tests/architecture/test_architecture_audit_fixes.py`

**Interfaces:**
- Produces: framework-neutral `learning` layer and current operational instructions.

- [ ] **Step 1: Write failing architecture tests** that forbid Torch in `trade_rl.learning` and forbid serving from importing offline approval modules.
- [ ] **Step 2: Run** import-linter/tests and confirm current Torch/offline-approval boundaries are permitted.
- [ ] **Step 3: Move** Torch/SB3 policy optimization into an integration adapter while retaining framework-neutral BC datasets/config/results in learning.
- [ ] **Step 4: Add** import-linter contracts for offline signing isolation.
- [ ] **Step 5: Update** all docs to the state-machine commands, Ed25519 public-key files, schema versions, waiting states, and `NO-GO` release policy.
- [ ] **Step 6: Run** import-linter, MyPy, and documentation asset tests.
- [ ] **Step 7: Commit** `refactor: isolate learning and offline approval boundaries`.

### Task 12: Full adversarial and platform verification

**Files:**
- Modify: `pyproject.toml` critical coverage list if measured gaps remain
- Modify: PR #67 description after verification

**Interfaces:**
- Consumes every interface from Tasks 1-11.

- [ ] **Step 1: Run targeted trust tests** for signing, metadata, authorization, confirmation, manifests, release, workflow policy, supervisor, and state machine.
- [ ] **Step 2: Run** `uv run ruff check .` and `uv run ruff format --check .`.
- [ ] **Step 3: Run** `uv run mypy trade_rl` and `uv run lint-imports`.
- [ ] **Step 4: Run** `uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json` and the critical coverage checker.
- [ ] **Step 5: Run** Ubuntu and Windows compatibility jobs.
- [ ] **Step 6: Build** the digest-pinned training image and run the non-root probe.
- [ ] **Step 7: Run** a bounded real CUDA smoke on the protected self-hosted runner and verify the resolved device, checkpoint recovery, generation-scoped evidence, and supervisor lifecycle.
- [ ] **Step 8: Re-read** the design and verify every requirement against code/tests; document any external repository-setting requirement that code cannot enforce.
- [ ] **Step 9: Update** PR #67 with exact evidence and only then mark Ready for review. Do not merge without an explicit user instruction.
