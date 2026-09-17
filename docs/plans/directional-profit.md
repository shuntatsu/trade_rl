# Directional profit implementation

Status: Active

Specification: [directional-profit.md](../specs/directional-profit.md).

- [x] Confirm the user-selected 20% drawdown tolerance and competing work.
- [x] Freeze candidate roster, risk, data/time scope, budget and stop rule.
- [x] Test and implement causal channels and the single breakout strategy.
- [x] Compose reproducible shared-account comparison and immutable evidence.
- [ ] Execute all fixed arms, report failures, and apply the frozen screen.
- [ ] Run stress and individual-symbol diagnostics for eligible arms only.
- [ ] Update current docs/Guide; full verification and exact-head PR review.

Use isolated worktree codex/directional-profit-bot. Do not modify or merge the
pending interleaved PPO, ridge economic-gate, or sealed final-test PRs. All
generated data/results stay in ignored output directories. Run focused tests
after each implementation step and every permanent quality gate before merge.
