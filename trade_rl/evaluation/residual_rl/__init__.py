"""Synthetic-only residual-PPO study over the cost-aware allocator (Issue #810).

Import from the concrete modules; this package intentionally re-exports nothing so
the ``trade_rl.evaluation`` public API stays unchanged. Everything here runs on
generated markets only: there is no dataset, network, live or paper access.
"""
