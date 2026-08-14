from __future__ import annotations

from pathlib import Path


def _replace(path: str, old: str, new: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one patch location in {path}: {old!r}")
    source.write_text(text.replace(old, new, 1), encoding="utf-8")


path = "tests/workflows/test_universal_causal_alpha_v3_runner_orchestration.py"
_replace(path, "from types import SimpleNamespace\n", "from dataclasses import replace\nfrom types import SimpleNamespace\n")
_replace(
    path,
    """from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
""",
    """from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.simulation.execution import ExecutionCostConfig
""",
)
_replace(
    path,
    '        execution_costs={"BTCUSDT": SimpleNamespace()},\n',
    '        execution_costs={"BTCUSDT": ExecutionCostConfig()},\n',
)
_replace(
    path,
    """    return CausalAlphaV3SignalGateEvidence(
        metrics=(metric,),
        expected_scope_count=1,
        scope_coverage=1.0,
        rank_ic=boot,
        top_bottom_spread=boot,
        direction_accuracy_excess=boot,
        gate_digest=_config().signal_gate.digest,
        passed=passed,
        rejection_reasons=() if passed else ("rank_ic_lower_ci",),
    )


def _selection(freeze_digest: str) -> CausalAlphaV3SelectionEvidence:
""",
    """    return CausalAlphaV3SignalGateEvidence(
        metrics=(metric,),
        expected_scope_count=1,
        scope_coverage=1.0,
        rank_ic=boot,
        top_bottom_spread=boot,
        direction_accuracy_excess=boot,
        gate_digest=_config().signal_gate.digest,
        passed=passed,
        rejection_reasons=() if passed else ("rank_ic_lower_ci",),
    )


def _scope_metric(*, passed: bool, **kwargs) -> CausalAlphaV3SignalScopeMetric:
    base = _signal_evidence(passed=passed).metrics[0]
    contract = kwargs["contract"]
    return replace(
        base,
        symbol=kwargs["symbol"],
        episode_index=contract.episode_index,
        contract_digest=contract.digest,
        digest="",
    )


def _selection(freeze_digest: str) -> CausalAlphaV3SelectionEvidence:
""",
)
_replace(
    path,
    "lambda **kwargs: _signal_evidence(passed=False).metrics[0],",
    "lambda **kwargs: _scope_metric(passed=False, **kwargs),",
)
_replace(
    path,
    "lambda **kwargs: _signal_evidence(passed=True).metrics[0],",
    "lambda **kwargs: _scope_metric(passed=True, **kwargs),",
)
