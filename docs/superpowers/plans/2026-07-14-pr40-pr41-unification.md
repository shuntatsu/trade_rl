# PR #40 and PR #41 unification

This branch combines the two July 14 architecture-hardening lines into one integration history.

- PR #40 is the structural base because it contains the broader provenance, session-market, portfolio-risk, algorithm configuration, replay/resume, and end-to-end work.
- PR #41 wins overlapping conflict hunks because its causal-research and serving hardening passed the complete repository CI before integration.
- Temporary remediation payloads and one-shot workflows are intentionally excluded from the final tree.
- The unified branch must pass the standard repository CI before it replaces the two source PRs.
