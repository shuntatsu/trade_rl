from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_TARGET = ROOT / "tests/risk/test_portfolio_risk_inputs.py"
INPUT_TARGET = ROOT / "trade_rl/risk/inputs.py"


def main() -> None:
    test_text = TEST_TARGET.read_text(encoding="utf-8")
    test_text = test_text.replace("from dataclasses import replace\n\n", "", 1)
    old_fixture = '''    future = replace(dataset, close=_dataset(future_shift=0.5).close)
    provider = env.portfolio_risk_inputs_provider
    assert provider is not None
    baseline = provider.inputs(dataset, index=120)
    changed = provider.inputs(future, index=120)
'''
    new_fixture = '''    provider = env.portfolio_risk_inputs_provider
    assert provider is not None
    baseline = provider.inputs(dataset, index=120)
    changed = provider.inputs(_dataset(future_shift=0.5), index=120)
'''
    if old_fixture not in test_text:
        raise RuntimeError("Task 11 future-mutation fixture anchor is missing")
    TEST_TARGET.write_text(
        test_text.replace(old_fixture, new_fixture, 1),
        encoding="utf-8",
    )

    input_text = INPUT_TARGET.read_text(encoding="utf-8")
    old_protocol = '''class PortfolioRiskDataset(Protocol):
    n_bars: int
    n_symbols: int
    periods_per_year: int
    close: np.ndarray
'''
    new_protocol = '''class PortfolioRiskDataset(Protocol):
    @property
    def n_bars(self) -> int: ...

    @property
    def n_symbols(self) -> int: ...

    @property
    def periods_per_year(self) -> int: ...

    @property
    def close(self) -> np.ndarray: ...
'''
    if old_protocol not in input_text:
        raise RuntimeError("Task 11 read-only dataset protocol anchor is missing")
    INPUT_TARGET.write_text(
        input_text.replace(old_protocol, new_protocol, 1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
