# Causal Alpha V3 Manual Launcher Design

## Status and objective

This specification adds an owner-only GitHub Actions control surface for fresh, artifact-bound Causal Alpha V3 research runs. The launcher must start a new V2 research generation from an exact `main` commit, keep the long-running computation outside the lifetime of the Actions job, expose status/collection/stop operations, and retain immutable research evidence without changing Causal Alpha V3 numerical semantics.

The feature is operational infrastructure only. It does not change Signal Gate thresholds, candidate ranking, Teacher admission, BC, critic warm start, PPO/Lagrangian, reward, risk, execution, or production-promotion semantics.

Current integration context at design time:

- `main`: `7b02d92b0234a84c7c4e240d5422be276efdfe73`;
- PR #410 owns deterministic machine run reporting and is not merged;
- PR #411 owns Signal Forensics and is not merged;
- this launcher is based directly on `main` and must not depend on either open PR.

## Approaches considered

### A. Run the V3 CLI synchronously inside one Actions job

This is the smallest implementation, but the V3 signal/selection path can take much longer than a control-plane job should remain attached to a self-hosted runner. An Actions timeout or runner interruption would conflate infrastructure lifetime with research lifetime and make partial evidence harder to retain.

Rejected.

### B. Start an immutable detached container and control it with separate operations

A short owner-only Actions job resolves immutable provenance, validates the runtime manifest and storage roots, builds the digest-bound training image, and starts a named detached container. Later `status`, `collect`, and `stop` dispatches inspect the same durable generation state. The container writes V3 evidence to the existing persistent `trade-rl-training-data` volume, so Actions checkout cleanup does not delete the active run.

Selected. This matches the repository's existing long-running Universal/full-training operational pattern while keeping V3-specific result semantics explicit.

### C. Stack the launcher on PR #410 and generate Markdown/JSON reports inside the workflow

This would provide richer collection output immediately, but it would bind V3 execution infrastructure to an unmerged reporting branch and make the execution identity depend on unrelated reporting code.

Rejected. Collection will retain the raw V3 output root and terminal evidence. Once #410 is merged, its read-only reporter can consume the retained run without changing this launch contract.

## Quality contract

### Objective

Provide a repeatable, owner-only manual launch/control path for a fresh Causal Alpha V3 Signal Contract V2 real-data generation, with exact software/runtime provenance and durable evidence collection.

### Non-goals

- no automatic experiment tuning or retry with changed thresholds/configuration;
- no use of historical V1/V2-incompatible output roots for resume;
- no modification of the V3 research config or U6 run config from workflow inputs;
- no arbitrary user-supplied host paths on the privileged runner;
- no dependency on PR #410 or PR #411;
- no profitability, alpha, RL-uplift, or Production GO claim;
- no automatic merge or branch cleanup.

### Acceptance criteria

1. The workflow is `workflow_dispatch` only and can run on the privileged self-hosted GPU runner only when `github.actor == github.repository_owner` and `github.ref == 'refs/heads/main'`.
2. Workflow and job permissions remain read-only; all external Actions are pinned by immutable SHA; checkout uses exact `${{ github.sha }}` with credentials disabled.
3. Supported operations are exactly `start`, `status`, `collect`, and `stop`.
4. `generation` is validated against a restricted identifier grammar and cannot become a path traversal or shell-injection vector.
5. Research config and run config are repository-authored fixed paths: `examples/binance/universal-causal-alpha-v3-research.json` and `examples/binance-multitimeframe/universal-u6-ppo.json`.
6. Runtime artifact root is a trusted repository/environment variable, not a dispatch path input. The expected runtime manifest is `<trusted-root>/runtime-manifest.json` and is mounted read-only.
7. Frozen metadata and run output use the persistent `trade-rl-training-data` Docker volume. The output root is `/workspace/var/runs/<generation>` and `start` rejects an existing output root.
8. `start` requires a clean exact checkout, records commit/source-tree/lockfile/runtime-manifest identities, builds or validates the immutable training image, preflights the mounted runtime manifest, and starts one named detached V3 container.
9. No stale or foreign container/state may be silently reused. Existing container, generation state, output root, or identity mismatch fails closed.
10. `status` never mutates or stops the research container. It reports Docker running/exited state, OOM state, exit code when available, and the immutable launch identity.
11. `collect` requires a terminal container, copies the complete generation output plus container logs into a retained directory, and classifies V3 exit codes without turning research rejection into infrastructure success/failure ambiguity: `0=admitted`, `2=signal_rejected`, `3=selection_rejected`, `4=admission_rejected`; other terminal codes are execution failures.
12. `stop` explicitly stops an active matching container, retains partial output/logs, and marks the result as operator-stopped rather than a scientific rejection.
13. Collection does not delete the durable source run by default. Container removal, when performed after successful retention, must not remove the named training-data volume.
14. All generated control-plane JSON is deterministic except explicitly observational fields such as current Docker state; immutable identities remain stable.
15. Existing V3 fit/gate/selection/admission code and existing U6 training behavior remain unchanged.

### Invariants

- Scalar reward remains pure net log growth.
- Hard `max_position_to_market_notional=0.02` remains authoritative.
- The maintained one-decision execution delay remains unchanged.
- Signal, economic selection, and Teacher admission chronological separation remains owned by the existing V3 runner.
- V3 artifacts remain `research_only` and non-promotable.
- A fresh generation never copies old V3 output records into the new output root.
- The active research process does not depend on the transient Actions checkout after `start` returns.

## Architecture

### 1. Dedicated V3 Compose service

Add `docker/compose.causal-alpha-v3-research.yaml` with one long-running `research` service. It uses the same digest-bound `training-runtime` image contract as maintained Universal training and invokes:

```text
python scripts/run_universal_causal_alpha_v3_research.py
  --config examples/binance/universal-causal-alpha-v3-research.json
  --run-config examples/binance-multitimeframe/universal-u6-ppo.json
  --runtime-factory trade_rl.workflows.binance_universal_runtime:build_runtime
  --runtime-manifest /workspace/var/universal/runtime-manifest.json
  --frozen-metadata-root /workspace/var/cache/frozen-metadata/usds-m
  --output-root /workspace/var/runs/<generation>
```

The service mounts:

- external `trade-rl-training-data` at `/workspace/var` for frozen metadata and durable run output;
- a trusted host Universal artifact root at `/workspace/var/universal:ro`;
- the existing external `trade_rl_default` network when the runtime requires PostgreSQL-backed artifacts.

The service has `restart: "no"`. No source checkout bind is used by the active container.

### 2. V3 generation controller

Add `scripts/control_causal_alpha_v3_research_generation.py` as the single owner of generation lifecycle operations.

The controller stores durable control-plane state outside the checkout under a trusted state root (defaulting to `$HOME/.local/state/trade-rl/causal-alpha-v3`, overridable only by an environment/repository variable). Each generation has one immutable `launch-manifest.json` containing at least:

- generation;
- container name;
- exact git commit;
- source-tree digest;
- dependency lock digest;
- runtime-manifest digest;
- research-config digest;
- run-config digest;
- image tag and image ID/digest;
- persistent container output path;
- launch schema version.

The controller verifies state identity before every non-start operation.

#### `start`

`start` validates the generation name, exact clean Git checkout, trusted runtime artifact root, runtime manifest, Compose file, authored configs, external Docker volume/network, and absence of existing state/container/output. It builds the training image with exact provenance build arguments, validates image labels, preflights the mounted runtime manifest, writes the launch manifest atomically, then starts the named container detached.

If container start fails after state creation, state remains as failed launch evidence rather than being silently reused.

#### `status`

`status` reads the launch manifest, validates the referenced container identity/labels, and reports `running`, `exited`, or invalid/missing state. It includes Docker exit code and OOM status when terminal. It does not copy, stop, restart, or delete anything.

#### `collect`

`collect` requires the matching container to be terminal. It copies `/workspace/var/runs/<generation>` and Docker logs into a caller-provided retained directory inside the current Actions workspace, writes `research-result.json`, and returns a control-plane exit code representing collection validity rather than the V3 scientific decision.

`research-result.json` records both layers separately:

- `execution_status`: completed / failed / operator_stopped;
- `research_outcome`: admitted / signal_rejected / selection_rejected / admission_rejected / unavailable;
- original container exit code;
- OOM state;
- immutable launch identity.

Scientific rejection exit codes 2/3/4 are therefore retained as valid terminal research outcomes, not mislabeled as a launcher failure.

#### `stop`

`stop` validates identity, stops only the matching active container, then uses the same retention path as collection with `execution_status=operator_stopped`. It must not treat the resulting Docker exit code as Signal/Selection/Admission evidence.

### 3. Owner-only GitHub Actions workflow

Add `.github/workflows/causal-alpha-v3-research.yml` with inputs:

- `operation`: choice `start|status|collect|stop`;
- `generation`: required string.

The job uses `[self-hosted, linux, x64, gpu, nvidia]`, environment `gpu-full-training`, read-only contents permission, owner/main guards, and a concurrency group that prevents overlapping control mutations.

Trusted environment/repository variables provide only infrastructure roots, for example:

- `TRADE_RL_UNIVERSAL_ARTIFACT_ROOT`;
- optional `TRADE_RL_CAUSAL_ALPHA_V3_STATE_ROOT`.

The workflow never accepts an arbitrary runtime manifest, frozen metadata root, output root, Compose file, config file, or shell fragment from dispatch input.

For `collect` and `stop`, the workflow uploads retained run output, logs, launch manifest, and result JSON as a GitHub Actions artifact. For `start`/`status`, it uploads the small control evidence available in the checkout.

### 4. Reporting boundary

This PR does not call `scripts/build_run_report.py` because #410 is unmerged. The retained V3 root is intentionally compatible with the reporter once #410 lands. This keeps generator/source-tree provenance for the fresh experiment independent of reporting changes.

## Failure modes

- Dispatch from non-owner or non-main ref: job does not run.
- Mutable external Action or privileged PR trigger: workflow-security test fails.
- Malformed generation: fail before Docker mutation.
- Missing trusted artifact/state root: fail before build/start.
- Dirty checkout or HEAD mismatch: fail before build/start.
- Runtime-manifest/config digest drift after launch: later operations fail closed.
- Existing generation state/container/output: `start` rejects rather than resumes implicitly.
- Missing external training volume/network: fail before start.
- Image provenance mismatch: fail before container start.
- Preflight cannot read exact mounted runtime manifest: fail before start.
- Active container during `collect`: reject; do not snapshot a moving run as final evidence.
- Container OOM/unexpected exit: retain output/logs and classify as execution failure.
- V3 exits 2/3/4: retain as valid scientific rejection outcome.
- Operator stop: retain partial evidence but do not infer a scientific outcome.
- `docker cp` or artifact retention failure: do not remove the source container/volume.

## Test oracle

Correctness is observed through:

- parsed workflow structure and existing workflow-security validator;
- exact immutable checkout/action references;
- ordered mocked Docker/Git subprocess calls;
- launch-manifest file contents and atomic state behavior;
- refusal to launch over existing state/container/output;
- container labels and exact image identity;
- status read-only call history;
- collect/stop retained filesystem contents;
- explicit two-layer execution/research outcome classification;
- unchanged hashes/diffs for existing V3 numerical modules and U6 configs.

## Required test layers

- Unit: generation validation, exit-code classification, launch/status/collect/stop state transitions.
- Contract: Compose service command, persistent volume, trusted read-only runtime bind, external network, fixed config paths.
- Workflow/architecture: owner/main guard, privileged environment, read-only permissions, immutable actions, no PR trigger, restricted inputs.
- Integration-style subprocess tests: fake Docker/Git command recorder and filesystem fixtures.
- Static: Ruff, format, Mypy, vulture/import architecture as applicable.
- Full regression: repository pytest/coverage and exact-head GitHub Actions before Ready status.

Real self-hosted GPU/data execution is a separate empirical verification layer. Software CI cannot prove that the private runner currently has the expected runtime manifest, frozen metadata, volume contents, PostgreSQL catalog, or real market data.

## Quality gate

The feature is not complete unless:

- acceptance criteria are represented by tests or independent structural evidence;
- targeted tests pass after valid RED evidence;
- workflow security passes;
- Ruff/format/Mypy/import architecture pass;
- full repository CI is Green on the exact final head;
- final diff contains no changes to V3 numerical/gating/selection/admission or U6 objective semantics;
- independent/falsification review explicitly attempts path injection, stale identity reuse, premature collection, research-rejection misclassification, source-run deletion, and operator-stop misclassification;
- unverified real-run prerequisites are documented rather than claimed successful.
