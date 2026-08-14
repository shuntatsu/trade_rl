from __future__ import annotations

from pathlib import Path


def _replace(path: str, old: str, new: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one patch location in {path}")
    source.write_text(text.replace(old, new, 1), encoding="utf-8")


_replace(
    "trade_rl/workflows/universal_causal_alpha_v3_signal.py",
    "zip(values, values[1:], strict=True)",
    "zip(values, values[1:])",
)

_replace(
    "trade_rl/workflows/universal_causal_alpha_v3_contracts.py",
    """    @property
    def unexplained_execution_rejection_count(self) -> int:
        return sum(count for _, count in self.execution_rejection_reason_counts)
""",
    """    @property
    def unexplained_execution_rejection_count(self) -> int:
        explained = frozenset(
            {"below_minimum_notional", "zero_quantity_after_rounding"}
        )
        return sum(
            count
            for reason, count in self.execution_rejection_reason_counts
            if reason not in explained
        )
""",
)
