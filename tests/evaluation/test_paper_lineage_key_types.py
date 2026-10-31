"""Application-level lineage gates must not trust equality/hash on JSON key subclasses.

Only offline, in-memory fake inputs and new pytest roots; no network or launch.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.evaluation.test_paper_attempt2 import START, clock, offline  # noqa: F401
from trade_rl.evaluation.paper import operations, predecessor
from trade_rl.evaluation.paper.account import PaperSettings
from trade_rl.evaluation.paper.attempt2 import attempt2_lineage
from trade_rl.evaluation.paper.engine import PaperEngine
from trade_rl.evaluation.paper.supervisor import seal_collection


class BlindKey(str):
    """Its serialized text is ordinary, but in-memory dict.get cannot find it."""

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return False


class StringSubclass(str):
    """Even equality-compatible string subclasses must not become JSON keys."""


def _settings() -> PaperSettings:
    return PaperSettings(start_at=START, close_at=START + timedelta(days=90))


def test_public_collection_seal_refuses_hidden_attempt_lineage_without_token(tmp_path):
    lineage = attempt2_lineage("unavailable")
    plan = {BlindKey("attempt_lineage"): lineage, "schema": "carry_paper_screen_v2"}
    assert plan.get("attempt_lineage") is None
    with pytest.raises(ValueError, match="key|lineage|plain"):
        seal_collection(
            tmp_path / "seal", settings=_settings(), research_plan=plan, clock=clock
        )
    assert not (tmp_path / "seal").exists()
    assert predecessor.collection_lineage_problems(
        plan, root=tmp_path, start_at=START, now=clock
    )


def test_engine_initialize_refuses_hidden_study_research_plan_before_writing(tmp_path):
    lineage = attempt2_lineage("unavailable")
    plan = {"schema": "carry_paper_screen_v2", "attempt_lineage": lineage}
    study = {
        "schema": "paper_public_collection_v1",
        BlindKey("research_plan"): plan,
    }
    assert study.get("research_plan") is None
    with pytest.raises(ValueError, match="key|lineage|plain"):
        PaperEngine.initialize(tmp_path / "seal", settings=_settings(), study=study)
    assert not (tmp_path / "seal").exists()


def test_recursive_lineage_key_type_is_rejected_even_if_text_matches(tmp_path):
    lineage = attempt2_lineage("unavailable")
    original_root = lineage["predecessor_root"]
    lineage["predecessor_root"] = {
        (StringSubclass(key) if key == "status" else key): value
        for key, value in original_root.items()
    }
    assert predecessor.lineage_structure_problems(lineage)
    with pytest.raises(ValueError, match="key|lineage|canonical|plain"):
        operations.validate_attempt_lineage(lineage)
    with pytest.raises(ValueError, match="key|lineage|canonical|plain"):
        seal_collection(
            tmp_path / "seal",
            settings=_settings(),
            research_plan={"attempt_lineage": lineage},
            attempt2_seal_token=operations._ATTEMPT2_SEAL_TOKEN,
            clock=clock,
        )
    assert not (tmp_path / "seal").exists()


def test_plain_native_keys_remain_accepted_for_canonical_attempt_one():
    lineage = {
        "schema": "carry_paper_attempt_lineage_v2",
        "attempt_number": 1,
        "predecessor": None,
    }
    assert predecessor.lineage_structure_problems(lineage) == []
