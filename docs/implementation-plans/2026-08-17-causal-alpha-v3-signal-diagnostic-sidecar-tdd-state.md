# Causal Alpha V3 Signal Diagnostic Sidecar TDD State

Current phase: **RED**.

The branch intentionally contains the approved design, implementation plan, and Task 1 tests before any production implementation.

Expected failure reason: `tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py` imports the not-yet-implemented `universal_causal_alpha_v3_signal_diagnostic` module and the not-yet-public `causal_alpha_v3_weight_digest` helper.

No Signal Gate, model, label, selection, Teacher, BC, RL, reward, risk, or execution production code has been changed at this checkpoint.
