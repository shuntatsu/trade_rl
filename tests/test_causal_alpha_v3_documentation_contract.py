from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_causal_alpha_v3_is_documented_as_non_promotable_research_lane() -> None:
    universal = _text("docs/UNIVERSAL_TRAINING.md")
    research = _text("docs/RESEARCH_STATUS.md")
    combined = f"{universal}\n{research}".lower()

    for phrase in (
        "causal alpha v3",
        "research-only",
        "promotion_eligible=false",
        "overlap",
        "uncertainty",
        "anchored_target_residual",
        "dagger",
        "teacher admission",
    ):
        assert phrase in combined

    assert 'behavior_cloning_teacher: "causal_alpha_ridge"' in universal
    assert "canonical u6" in universal.lower()
    assert "target_weight" in universal
    assert "production statusは**no-go**" in universal.lower()


def test_v3_docs_preserve_reward_and_holdout_invariants() -> None:
    universal = _text("docs/UNIVERSAL_TRAINING.md").lower()

    for phrase in (
        "pure net-log-growth",
        "teacher-admission holdout",
        "reward unchanged",
        "does not bypass teacher admission",
    ):
        assert phrase in universal
