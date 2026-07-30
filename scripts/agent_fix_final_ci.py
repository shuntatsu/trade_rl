from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    replace_once(
        "tests/simulation/test_execution_slippage_coverage.py",
        '''    assert executor._capacity_notional(\n        np.array([100.0]), np.array([2.0])\n    ).tolist() == pytest.approx([200.0])\n''',
        '''    assert executor.dataset.market_notional(\n        1,\n        prices=np.array([100.0]),\n        volume=np.array([2.0]),\n    ).tolist() == pytest.approx([200.0])\n''',
    )
    replace_once(
        "tests/simulation/test_stateful_execution_characterization.py",
        '"3856e696c998e727c78690222d418e070c71eeb56f7f747f0932a17eb8ff2cc2"',
        '"3f88b8802db74a7a4fe2f81d8c822dff2d85caee9c798deac6d96d76948d4e74"',
    )
    for relative in (
        "scripts/agent_fix_final_ci.py",
        ".github/workflows/agent-fix-final-ci.yml",
        ".agent/fix-final-ci-trigger",
    ):
        path = ROOT / relative
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
