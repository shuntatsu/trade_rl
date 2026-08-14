import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "universal-u6-ppo.json",
    "universal-u6-lagrangian.json",
    "universal-u6-discounted.json",
)


def test_v3_research_lane_does_not_change_canonical_u6_contracts() -> None:
    for name in CONFIGS:
        payload = json.loads(
            (ROOT / "examples/binance-multitimeframe" / name).read_text(encoding="utf-8")
        )

        assert payload["action"]["mode"] == "target_weight"
        assert payload["action"]["target_weight_count"] == 1
        assert payload["training"]["behavior_cloning_teacher"] == "causal_alpha_ridge"
        assert payload["portfolio_risk"]["max_position_to_market_notional"] == 0.02
        assert payload["reward"]["absolute_growth_weight"] == 1.0
        assert payload["reward"]["baseline_underperformance_weight"] == 0.0
        assert payload["reward"]["excess_growth_weight"] == 0.0
        assert payload["reward"]["incremental_drawdown_weight"] == 0.0
        assert payload["reward"]["margin_deficit_weight"] == 0.0
        assert payload["reward"]["projection_penalty_weight"] == 0.0
        assert payload["reward"]["terminal_equity_weight"] == 0.0
        assert payload["environment"]["signal_delay_decisions"] == 1
