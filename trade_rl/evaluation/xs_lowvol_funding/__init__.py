"""Out-of-sample test of the weekly cross-sectional low-volatility / funding-contrarian perp
book (``xs_lowvol_funding_v1``).

The normative text is ``docs/specs/xs-lowvol-funding.md``. This package owns the frozen
constants and digest, the monthly point-in-time universe, the weekly policy, the stage
dataset assembly, the pure decision rule and the runner that drives
``run_shared_portfolio_replay``. It owns no network, market-store or ML dependency; only
``tools/xs_lowvol_funding.py`` wires the data sources and the owner authorization. Passing the
tests authorizes no stage, no label and no order.
"""

from trade_rl.evaluation.xs_lowvol_funding import (
    data,
    decision,
    policy,
    protocol,
    runner,
    universe,
)

__all__ = ["data", "decision", "policy", "protocol", "runner", "universe"]
