---
name: research-challenge
disable-model-invocation: true
description: "Adversarial, result-blind challenge of a trade_rl research hypothesis, protocol, or implementation claim before economic execution."
---

# Research challenge (author-side only)

Use this skill when proposing or materially changing a trading research study,
selecting what experiment to run next, or testing whether a research claim is
warranted. It is an **adversarial preparation tool**, not an independent review
receipt and not G0–G5 approval.

1. Read `AGENTS.md`, `docs/AGENTS.md`,
   `docs/architecture/research-assurance.md`, the closest architecture/research
   contract, and the actual code/tests; re-check current GitHub `main`, related
   open PRs and Issues before recommending edits. Source/test behavior wins over
   stale documentation; report and correct inconsistencies in the change.
2. State the proposed decision, operational objective, exact account mode,
   hypothesis, mechanism and the specific economic value it is supposed to add.
   Identify the strongest **alternative explanation** that would make a good
   result spurious (beta exposure, data leakage, costs, turnover, funding,
   capacity, cherry-picked periods, accounting, or accidental test reuse).
3. Build the strongest case **against** the hypothesis. Give three concrete,
   realistic failure modes and the cheapest decisive falsifier for each.
   Separate implementation failure from economic failure. Do not assume RL or
   extra features create an edge.
4. Identify claim boundaries using the existing G0–G5 contract: G0 question,
   G1 mechanism, G2 independent code oracle, G3 evidence validity, G4 development
   improvement, G5 unused-data/deployment eligibility. Explicitly separate
   relative improvement from positive after-cost profitability, independent
   accounts from one shared-cash ledger, and repeated development from unused
   evidence.
5. Propose a result-blind stop rule, fixed comparison/control, hard-risk and
   cost assumptions, multi-symbol/period checks, and evidence needed to change
   the current decision. Preserve all historical experiment outcomes and
   sealed datasets. Never change a frozen gate after seeing its results.
6. Finish with **reject / revise / proceed to separate independent review**,
   the reason, open uncertainties, and the exact next non-economic verification.
   A passing synthetic unit test, self-review, draft packet, or CI run does not
   authorize market-value computation, unused/final evaluation, or live orders.

Do not run new market-data diagnostics, fits, backtests, replay, funding/carry
paper evaluation, or sealed-data reads merely because this skill is invoked.
Actual economic execution needs the separately applicable reviewed packet,
protocol, scope and authorization. Never impersonate an independent reviewer.
