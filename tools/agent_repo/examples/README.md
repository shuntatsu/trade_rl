# Research assurance v1 (generic example)

research_assurance.json is a neutral authored packet example, not a passed
experiment. It is intentionally UNREVIEWED and includes no actual model result,
P&L, reviewed outcome or production/live authorization.

- Replace the example identities with the full exact protocol and implementation
  Git commit SHAs from the actual proposed research change.
- Specify a falsifiable research question, plausible counter-hypothesis, cheapest
  falsifier and stop rule before looking at outcomes.
- List every material changed mechanism boundary under changed_boundaries and
  provide an independent statement/counterexample/oracle/limitations for each.
- Separate discovery, development, unused, and prospective evidence. Do not
  claim absolute profitability from relative or software-only evidence.
- Author packets must NOT contain a review. Digest the entire packet with
  python -m tools.agent_repo assurance digest <author-packet.json>.
- The reviewer must separately generate a research_assurance_review_v1 record
  bound to the exact packet digest and both commit identities. An author-provided
  JSON record, even with PASS fields, cannot establish reviewer independence.
- The external verifier requires an authenticated GitHub PR review from a
  distinct principal at the exact implementation HEAD (no author-self-review).
  Its body must be canonical JSON, with a real review ID supplied to
  python -m tools.research_assurance_github.
- The reviewed PASS state is G0–G2 assurance only; economic execution,
  unused/final data access and live trading are separately unauthorized.

Both JSON Schemas are in ../schemas/. Runtime integrity is checked by
tools.agent_repo.research_assurance, not merely by JSON Schema validation.
