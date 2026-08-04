from pathlib import Path

path = Path("trade_rl/learning/oracle_transition_torch.py")
text = path.read_text(encoding="utf-8")
start = text.index("class TorchMarketTapeLike(Protocol):")
end = text.index("\n\n@dataclass", start)
fields = (
    "raw_position_factor",
    "equity_position_factor",
    "mark_open_ratio",
    "active",
    "tradable",
    "buy_allowed",
    "sell_allowed",
    "borrow_available",
    "market_notional",
    "participation_capacity",
    "minimum_notional",
    "base_unit_cost",
    "funding_due_rate",
    "borrow_rate",
    "dividend_open_ratio",
    "cash_rate",
    "elapsed_year_fraction",
)
lines = ["class TorchMarketTapeLike(Protocol):"]
for field in fields:
    lines.extend(("    @property", f"    def {field}(self) -> torch.Tensor: ...", ""))
replacement = "\n".join(lines).rstrip()
path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")

path = Path("trade_rl/learning/oracle_bellman_torch.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "    OracleBackendFailure,\n",
    "    CompileMode,\n    OracleBackendFailure,\n",
    1,
)
text = text.replace(
    ") -> tuple[_ResultT, str, str | None]:\n",
    ") -> tuple[_ResultT, CompileMode, str | None]:\n",
    1,
)
text = text.replace(
    "    ) -> tuple[TorchBellmanResult, str, str | None]:\n",
    "    ) -> tuple[TorchBellmanResult, CompileMode, str | None]:\n",
    1,
)
path.write_text(text, encoding="utf-8")
