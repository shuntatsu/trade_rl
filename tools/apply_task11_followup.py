from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "tests/risk/test_portfolio_risk_inputs.py"


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    text = text.replace("from dataclasses import replace\n\n", "", 1)
    old = '''    future = replace(dataset, close=_dataset(future_shift=0.5).close)
    provider = env.portfolio_risk_inputs_provider
    assert provider is not None
    baseline = provider.inputs(dataset, index=120)
    changed = provider.inputs(future, index=120)
'''
    new = '''    provider = env.portfolio_risk_inputs_provider
    assert provider is not None
    baseline = provider.inputs(dataset, index=120)
    changed = provider.inputs(_dataset(future_shift=0.5), index=120)
'''
    if old not in text:
        raise RuntimeError("Task 11 future-mutation fixture anchor is missing")
    TARGET.write_text(text.replace(old, new, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
