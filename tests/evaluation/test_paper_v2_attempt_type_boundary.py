"""A v2-labelled successor cannot bypass the v3 attempt-2 lineage policy."""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.evaluation.test_paper_attempt2 import START, clock, offline  # noqa: F401
from trade_rl.evaluation.paper import cli, predecessor
from trade_rl.evaluation.paper.account import PaperSettings
from trade_rl.evaluation.paper.engine import PaperEngine
from trade_rl.evaluation.paper.store import PaperJournal
from trade_rl.evaluation.paper.supervisor import PaperCollector, seal_collection
from trade_rl.evaluation.runs import build_candidate_run_provenance


class IntegerSubclass(int):
    pass


@pytest.mark.parametrize(
    "number", [2.0, 1e30, True, 0, -1, "2", None, IntegerSubclass(2)]
)
def test_v2_invalid_number_is_refused_before_public_seal_or_collector(tmp_path, number):
    lineage = {
        "schema": "carry_paper_attempt_lineage_v2",
        "attempt_number": number,
        "predecessor": None,
    }
    plan = {"schema": "carry_paper_screen_v2", "attempt_lineage": lineage}
    settings = PaperSettings(start_at=START, close_at=START + timedelta(days=90))
    assert predecessor.lineage_structure_problems(lineage)
    with pytest.raises(ValueError, match="lineage|attempt|canonical"):
        seal_collection(
            tmp_path / "sealed", settings=settings, research_plan=plan, clock=clock
        )
    assert not (tmp_path / "sealed").exists()
    study = {
        "schema": "paper_public_collection_v1",
        "created_at": clock().isoformat(),
        "interval_seconds": 60,
        "terminal_grace_seconds": 180,
        "provenance": build_candidate_run_provenance(),
        "research_plan": plan,
    }
    with pytest.raises(ValueError, match="lineage|attempt|canonical"):
        PaperEngine.initialize(tmp_path / "engine", settings=settings, study=study)
    assert not (tmp_path / "engine").exists()
    forged = tmp_path / "forged"
    digest = PaperJournal.initialize(
        forged,
        {
            "engine": "funding_carry_paper_engine_v1",
            "settings": settings.to_dict(),
            "study": study,
        },
    )
    before = {p.name: p.read_bytes() for p in forged.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="lineage|attempt|canonical"):
        with PaperCollector(forged, expected_protocol_sha256=digest, clock=clock):
            pytest.fail("collector opened invalid v2 attempt")
    with pytest.raises(ValueError, match="lineage|attempt|canonical"):
        cli._refuse_non_screen_root(forged, digest)
    assert before == {p.name: p.read_bytes() for p in forged.iterdir() if p.is_file()}


def test_v2_attempt_one_requires_no_predecessor_and_passes_when_canonical():
    valid = {
        "schema": "carry_paper_attempt_lineage_v2",
        "attempt_number": 1,
        "predecessor": None,
    }
    assert predecessor.lineage_structure_problems(valid) == []
    invalid = {**valid, "predecessor": {"attempt_number": 0}}
    assert predecessor.lineage_structure_problems(invalid)


def test_cli_seal_rejects_duplicate_keys_in_lineage_json(tmp_path, capsys):
    # Last-key-wins would turn this forged attempt 2 into a valid attempt 1.
    path = tmp_path / "ambiguous-lineage.json"
    path.write_text(
        '{"schema":"carry_paper_attempt_lineage_v2",'
        '"attempt_number":2,"attempt_number":1,"predecessor":null}',
        encoding="utf-8",
    )
    assert (
        cli.main(
            [
                "seal",
                "--root",
                str(tmp_path / "attempt"),
                "--start-at",
                START.isoformat(),
                "--attempt-lineage-json",
                str(path),
            ]
        )
        == 2
    )
    assert not (tmp_path / "attempt").exists()
    reported = capsys.readouterr().err
    assert "DuplicateKeyError" in reported and "repeats" in reported
