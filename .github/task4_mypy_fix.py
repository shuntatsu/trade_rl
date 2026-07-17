from pathlib import Path

path = Path("trade_rl/workflows/market_walk_forward.py")
content = path.read_text(encoding="utf-8")

old_import = "from typing import Any\n"
new_import = "from typing import Any, TypedDict\n"
if old_import not in content:
    raise SystemExit("typing import anchor missing")
content = content.replace(old_import, new_import, 1)

old_header = '''def _sensitivity_metrics(
    evidence: RangeEvaluation,
    *,
    initial_capital: float,
    duration_days: float,
) -> dict[str, object]:
'''
new_header = '''class _SensitivityMetrics(TypedDict):
    cost_fraction: float
    diagnostics: dict[str, object]
    maximum_drawdown: float
    n_trades: int
    returns: tuple[float, ...]
    rule_burden_percentiles: dict[str, object] | None
    total_return: float
    turnover_per_day: float


def _sensitivity_metrics(
    evidence: RangeEvaluation,
    *,
    initial_capital: float,
    duration_days: float,
) -> _SensitivityMetrics:
'''
if old_header not in content:
    raise SystemExit("sensitivity metrics anchor missing")
content = content.replace(old_header, new_header, 1)
path.write_text(content, encoding="utf-8")
