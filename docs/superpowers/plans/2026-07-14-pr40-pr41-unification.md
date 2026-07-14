# PR #40 and PR #41 unification

This branch combines the two July 14 architecture-hardening lines into one integration history.

- PR #40 is the structural base because it contains the broader provenance, session-market, portfolio-risk, algorithm configuration, replay/resume, and end-to-end work.
- PR #41's verified causal-research, artifact, release, and serving contracts are retained.
- Signal artifacts combine generator/prediction lineage with row-wise point-in-time knowledge cutoffs.
- Walk-forward evaluation combines sealed-test access evidence, execution diagnostics, and independent-fold metric summaries.
- Serving validates both legacy release manifests and adjacent non-circular release attestations against immutable bundle identity.
- Temporary remediation payloads, patch helpers, and one-shot workflows are excluded from the final tree.
- Dataset closure, timestamp feature age, recomputable identity, multiplier accounting, session carry, elapsed-time metrics, and serving tamper regressions pass.
- The standard Linux and Windows repository CI is the final merge gate.
