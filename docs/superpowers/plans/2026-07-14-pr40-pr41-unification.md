# PR #40 and PR #41 unification

This branch combines the two July 14 architecture-hardening lines into one integration history.

- PR #40 is the structural base because it contains the broader provenance, session-market, portfolio-risk, algorithm configuration, replay/resume, and end-to-end work.
- Every path changed by PR #41 is taken from PR #41's verified head to avoid mixed-file conflict artifacts.
- Temporary remediation payloads and one-shot workflows are intentionally excluded from the final tree.
- PR #40-only additions remain in the unified branch and are validated against PR #41's causal-research and serving implementation.
- The unified branch must pass the standard repository CI before it replaces the two source PRs.
