# Funding carry implementation plan

Goal: implement the fixed contract in ../specs/funding-carry.md and execute its
development experiment with evidence, without claiming a production winner.

- [x] Check current main, competing PRs and latest actual research decisions.
- [x] Isolate work on codex/funding-carry-bot; freeze the experiment before data.
- [ ] Implement and test paired quantity sizing and irreversible risk stop.
- [ ] Implement/test two-market dataset assembly using maintained transport.
- [ ] Implement/test replay and immutable evidence on the canonical ledger.
- [ ] Acquire declared development source, execute all fixed comparisons/stress.
- [ ] Interpret the frozen gate; retain unfavorable results and limitations.
- [ ] Update current architecture/research docs and Guide if referenced.
- [ ] Run full quality gates, review, create PR and verify exact final head.

Run `uv run pytest -q tests/strategies/test_carry.py
tests/integrations/test_binance_carry.py tests/evaluation/test_carry.py` during
development. Final checks are those in .github/workflows/ci.yml. No new runtime
dependency or parallel research worker is required.
