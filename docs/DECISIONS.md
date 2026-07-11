# Architecture Decisions

## ADR-001: Separate Control and Serving planes

**Decision:** Training, evidence, Registry mutation, activation, and rollback run offline. Online Serving is authenticated and read-only.

**Reason:** Combining privileged management and live signal delivery created an unnecessary trust and failure boundary.

## ADR-002: Use one immutable bundle Registry

**Decision:** `mars_lite.serving.registry.ModelRegistry` is the only model lifecycle authority. Registered version directories are immutable and `active.json` is the only active pointer.

**Reason:** Multiple registries and fixed filenames allowed promotion and serving identity to diverge.

## ADR-003: Separate registration from activation

**Decision:** Training may construct and register a candidate but may not activate it. Activation requires deployment evidence and environment approval.

**Reason:** A successful training process is not deployment authorization.

## ADR-004: Bundle every inference dependency

**Decision:** Model, symbols, feature schemas, preprocessing, observation contract, post-processing, risk configuration, Git SHA, metrics identity, and digests travel in one ServingBundle.

**Reason:** Partial metadata caused train/serve distribution and shape mismatches.

## ADR-005: Use real positions before policy inference

**Decision:** The policy observation is built from authenticated current account state before `predict()`.

**Reason:** Applying previous weights only after inference violated the training observation contract.

## ADR-006: Use deterministic online progress

**Decision:** Production-compatible models use observation progress mode `zero`. Episode-relative progress is research-only unless an online-equivalent state is designed and validated.

**Reason:** An arbitrary training episode position cannot be reproduced in stateless online inference.

## ADR-007: Trade Platform is the execution authority

**Decision:** Serving computes guardrail and pre-trade risk verdicts; the Trade Platform performs final enforcement and order execution.

**Reason:** Serving has no authoritative exchange connection or account store and must not claim orders were executed.

## ADR-008: Keep serving state minimal

**Decision:** Account state arrives with each authenticated request. SQLite stores audit events and replay claims only.

**Reason:** Duplicating portfolio truth inside Serving would create reconciliation and stale-state risk.

## ADR-009: Fail closed while preserving known-good service

**Decision:** Invalid requests return no actionable signal. Invalid new bundles do not replace a healthy in-memory bundle and set readiness to degraded.

**Reason:** Availability must not override integrity, but a bad candidate should not unnecessarily remove a verified existing service.

## ADR-010: One normative documentation set

**Decision:** `docs/ARCHITECTURE.md` is the only architecture authority. Research results live separately and cannot authorize Production.

**Reason:** Previous documents mixed historical experiments, aspirational design, and executable behavior.
