from __future__ import annotations

from pathlib import Path

from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3ResearchConfig,
)


_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _ROOT / "examples" / "binance" / "universal-causal-alpha-v3-research.json"


def test_v3_research_example_is_bounded_strict_and_non_duplicate() -> None:
    config = CausalAlphaV3ResearchConfig.from_json(_EXAMPLE)

    assert len(config.candidates) == 6
    assert len({candidate.name for candidate in config.candidates}) == 6
    assert len({candidate.semantic_digest for candidate in config.candidates}) == 6
    assert config.nested_selection.signal_contract_count == 8
    assert config.nested_selection.minimum_economic_contract_count == 4
    assert config.signal_gate.minimum_scope_coverage == 1.0
    assert config.selection_gate.minimum_symbol_episode_net_return == -0.05
    assert config.selection_gate.maximum_unexplained_execution_rejections == 0
    assert all(
        max(candidate.target.target_magnitudes) <= 0.25
        for candidate in config.candidates
    )
