# Production Architecture Hardening Design

## Status

Approved direction from the architecture review. This specification defines the first hardening increment required before any Production GO decision.

Production remains **NO-GO** after this change. The change only closes code-verifiable architecture gaps; operational, exchange, security, legal, and GameDay evidence remain separate owner responsibilities.

## Goal

Make it structurally impossible for a research-only, forced, skipped-gate, incomplete-risk, or unverified candidate to become the model served as Production.

The release path must prove all of the following for one exact identity:

1. the candidate passed the mandatory Control Plane gates without overrides;
2. the candidate contains a complete, validated risk policy;
3. the candidate was evaluated on a sealed holdout;
4. the approved immutable bundle was activated in persistent stage storage;
5. the live Serving Plane reports the same model version and bundle digest after activation.

## Scope

This increment changes:

- Production candidate eligibility and metadata;
- Control Plane fail-closed behavior;
- ServingBundle validation;
- deployment workflow activation and post-activation verification;
- tests and normative documentation.

This increment does not implement:

- a real exchange execution adapter;
- multi-node Serving or distributed idempotency;
- TLS, proxy, secret rotation, or GitHub Environment administration;
- a new feature store;
- a full container deployment platform.

## Considered approaches

### Approach A: Documentation-only operational rules

Keep `--force` and skip flags unrestricted, rely on operators not to promote those runs, and document required risk limits and holdout evidence.

This is rejected because the repository already contains this style of rule and the deployment path cannot prove compliance mechanically.

### Approach B: Add eligibility metadata but keep activation loosely coupled

Record forced/skipped status, risk configuration, and holdout status in the bundle, then let the deployment workflow activate the registry without querying the running service.

This improves auditability but still permits a successful deployment job that never changes the model actually served.

### Approach C: Fail-closed release eligibility plus served-identity verification

Separate research execution from release eligibility, persist an explicit eligibility record in the bundle, validate mandatory risk and holdout requirements, activate only after evidence validation, then query the live Serving Plane and require its version and digest to match the approved bundle.

This is the selected approach. It closes the identified gaps with limited changes to existing boundaries and avoids introducing a new deployment platform.

## Architecture

### 1. Release eligibility is an explicit immutable contract

Every Production-capable ServingBundle must include a `release_eligibility` object in `metadata.json`:

```json
{
  "release_eligibility": {
    "eligible": true,
    "forced": false,
    "skipped_gates": [],
    "sealed_holdout_used": true,
    "required_gates": {
      "p0": "passed",
      "walk_forward": "passed",
      "gate2": "passed",
      "significance": "passed_or_not_required"
    }
  }
}
```

The bundle validator rejects malformed eligibility metadata. The deployment gate rejects any bundle where:

- `eligible` is not exactly `true`;
- `forced` is not exactly `false`;
- `skipped_gates` is non-empty;
- `sealed_holdout_used` is not exactly `true`;
- any mandatory gate is not in an accepted state.

Eligibility is derived by the Control Plane. It is not accepted from a user-supplied JSON file or CLI argument.

### 2. Research overrides cannot produce a registrable release candidate

The existing research controls remain available for diagnosis:

- `--force`;
- `--skip-p0`;
- `--skip-pbt`;
- `--skip-wf`;
- `--skip-gate`.

However, any run using a release-disqualifying override must not construct or register a candidate.

The Control Plane uses a pure eligibility evaluator to classify the resolved run. A disqualified run may still train and write research reports, but candidate construction fails closed with a clear message. `--no-register` remains valid for intentionally non-release runs.

`--skip-pbt` alone is not automatically disqualifying because PBT is an optimization step rather than a safety gate. It is still recorded in run metadata. `--skip-p0`, `--skip-wf`, `--skip-gate`, and `--force` are disqualifying.

### 3. Sealed holdout is mandatory for release candidates

A release-capable run must create a non-empty sealed holdout that is untouched by PBT, walk-forward selection, feature-mask selection, and final training.

If the data cannot produce both a development partition and a holdout partition after purge, the Production pipeline stops before training a release candidate. Research-only runs may continue only when explicitly non-registering.

The eligibility record stores `sealed_holdout_used=true` only when the final Gate 2 evaluation used the supplied sealed holdout.

### 4. Risk policy is complete and environment supplied

Production candidate construction must receive an explicit risk policy instead of silently serializing `PreTradeRiskConfig()` with all limits unset.

A new typed release-risk configuration contains at least:

- `max_leverage`;
- `max_single_weight`;
- `max_net_exposure`;
- `max_worst_case_notional`;
- `min_order_notional`;
- `symbol_liquidity_caps`;
- `forbidden_symbols`.

Required scalar limits must be finite and positive. Percentage or leverage values must also satisfy their semantic bounds. Liquidity caps must cover every bundle symbol and be finite and positive. Forbidden symbols may be empty but must be explicitly present.

The Control Plane loads this policy from a JSON file supplied through a release-only CLI option such as `--risk-config`. Absence or invalid content blocks candidate construction. The resolved risk policy is copied into `risk.json` and protected by the bundle digest.

Guardrail weight-cap behavior remains a warning in this increment because the authoritative hard rejection is provided by `PreTradeRiskVerifier.max_single_weight`. A later cleanup may remove the duplicate guardrail warning.

### 5. Deployment activation targets persistent stage storage

The workflow keeps the existing Registry activation command but makes the deployment target explicit through environment configuration:

- `TRADE_RL_REGISTRY_DIR` must be an absolute path;
- `TRADE_RL_SERVING_READY_URL` must identify the stage Serving Plane `/ready` endpoint;
- the stage runner must have access to the same persistent Registry storage used by Serving.

The workflow cannot prove storage topology from YAML alone, so the readiness handshake is the final authority.

### 6. Post-activation served-identity verification

After registry activation, the workflow polls the configured `/ready` endpoint with a bounded retry loop. Success requires:

- HTTP 200;
- readiness status `ready` or `degraded`;
- `active_version` equals the approved bundle version;
- `bundle_digest` equals the approved bundle digest.

`degraded` is accepted only when it reports the newly approved identity. A degraded response still serving the previous bundle fails deployment.

If verification fails, the workflow fails and records the expected and observed identities. Automatic rollback is intentionally out of scope for this increment because an unverified deployment must not mutate Production a second time without an explicit operator decision. The runbook instructs operators to invoke the existing rollback command.

### 7. Running code identity is exposed and checked

The Serving Plane receives `TRADE_RL_RELEASE_GIT_SHA` at startup. It must be a 40-character hexadecimal commit SHA.

Readiness reports this running release SHA. Bundle refresh rejects an active bundle whose `manifest.git_sha` does not match the running release SHA when strict release binding is enabled.

Strict binding is enabled by default in the Production server entrypoint. Unit tests may construct a runtime without strict binding for isolated component testing.

This prevents a model trained under one code revision from being served by silently different feature, observation, decision, or risk code.

## Data flow

```text
resolved CLI and risk config
  -> data quality
  -> mandatory sealed holdout split
  -> P0
  -> optional PBT
  -> mandatory walk-forward
  -> final training on development data
  -> Gate 2 on sealed holdout
  -> derive release eligibility
  -> create immutable ServingBundle
  -> bundle schema and digest validation
  -> immutable Registry registration
  -> evidence production
  -> deployment gate validation
  -> persistent Registry activation
  -> Serving refresh
  -> /ready identity handshake
```

## Error handling

All release-path failures are fail-closed.

- Missing sealed holdout: stop before candidate construction.
- Forced or skipped mandatory gate: research output allowed, release candidate denied.
- Missing or invalid risk policy: candidate creation denied.
- Invalid release eligibility metadata: bundle load denied.
- Running Git SHA mismatch: bundle refresh denied; previous healthy in-memory bundle remains active and readiness becomes degraded.
- Post-activation identity mismatch: deployment workflow fails and provides rollback instructions.
- Unreachable readiness endpoint: deployment workflow fails after bounded retries.

No failure path returns actionable target weights from an unvalidated new bundle.

## Testing strategy

### Unit tests

Add tests for:

- eligibility derivation for normal, forced, and skipped-gate runs;
- candidate rejection when risk configuration is missing or incomplete;
- bundle validation of release eligibility fields;
- sealed holdout requirement;
- runtime rejection of a bundle whose Git SHA differs from the running release SHA;
- readiness serialization of running release SHA.

### Integration tests

Add tests proving:

- a fully eligible candidate registers but is not activated by training;
- a forced or skipped-gate run cannot register a candidate;
- the deployment workflow contains activation followed by served-identity verification;
- the expected bundle version and digest are compared with `/ready` output;
- a mismatched active identity fails the verification script.

### Regression expectations

Existing research commands remain usable when `--no-register` is supplied. Existing serving hot-swap behavior remains unchanged for a matching release SHA. Existing bundle digest and registry immutability tests must continue to pass.

## Documentation changes

Update the normative documents to state:

- research overrides can never produce a promotable candidate;
- sealed holdout and explicit risk configuration are mandatory for release candidates;
- deployment is complete only after the Serving Plane reports the approved version and digest;
- the running release Git SHA must equal the bundle Git SHA;
- Production remains NO-GO until operational checklist evidence is complete.

## Acceptance criteria

This increment is complete when all of the following are true:

1. `--force`, `--skip-p0`, `--skip-wf`, or `--skip-gate` cannot produce a registered release candidate.
2. A release candidate cannot be created without a sealed holdout.
3. A release candidate cannot be created with unset mandatory pre-trade limits.
4. Bundle validation rejects ineligible release metadata.
5. Serving rejects a bundle built from a different Git SHA than the running release.
6. Deployment fails unless `/ready` reports the exact approved version and bundle digest.
7. Ruff, format, mypy, and the complete pytest suite pass.
8. Production documentation continues to report **NO-GO**.
