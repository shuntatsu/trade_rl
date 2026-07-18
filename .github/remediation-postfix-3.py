from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    source = target.read_text(encoding="utf-8")
    if old not in source:
        raise SystemExit(f"postfix-3 marker is missing in {path}: {old[:80]!r}")
    target.write_text(source.replace(old, new, 1), encoding="utf-8")


replace_once(
    "examples/binance-multitimeframe/full_research_pipeline.py",
    '        "raw_feature_count": result.dataset.n_features,\n',
    '        "raw_feature_count": result.dataset.n_features,\n        "policy_observation_count": _EXPECTED_POLICY_OBSERVATIONS,\n',
)
replace_once(
    "tests/examples/test_binance_metadata_mode_runner.py",
    "def test_runner_frozen_mode_does_not_require_signed_history() -> None:\n",
    "def test_runner_frozen_mode_does_not_require_signed_history(\n    monkeypatch: pytest.MonkeyPatch,\n) -> None:\n",
)
replace_once(
    "tests/examples/test_binance_metadata_mode_runner.py",
    '''    resolve.__globals__["_load_rule_history"] = lambda: pytest.fail(\n        "frozen mode must not read signed history"\n    )\n''',
    '''    monkeypatch.setitem(\n        resolve.__globals__,\n        "_load_rule_history",\n        lambda: pytest.fail("frozen mode must not read signed history"),\n    )\n''',
)
replace_once(
    "tests/examples/test_binance_metadata_mode_runner.py",
    "def test_runner_historical_mode_accepts_only_verified_history() -> None:\n",
    "def test_runner_historical_mode_accepts_only_verified_history(\n    monkeypatch: pytest.MonkeyPatch,\n) -> None:\n",
)
replace_once(
    "tests/examples/test_binance_metadata_mode_runner.py",
    '    resolve.__globals__["_load_rule_history"] = load\n',
    '    monkeypatch.setitem(resolve.__globals__, "_load_rule_history", load)\n',
)
