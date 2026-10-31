"""Pre-seal follow-ups for attempt 2 (review 5719733380): N1 collector re-verification,
N2 a discriminating hot-journal fixture, the private seal token and N5 wording.

Offline only: sockets are blocked and no collector cycle ever runs.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from tests.evaluation.test_paper_attempt2 import (  # noqa: F401  (autouse fixture)
    AUTH,
    START,
    Recorder,
    clock,
    launch,
    offline,
    refusal_of,
)
from tests.evaluation.test_paper_attempt2_lineage import (
    FakeReader,
    forged_available,
    v1_report,
)
from trade_rl.evaluation.paper import attempt2, cli, operations, predecessor
from trade_rl.evaluation.paper.attempt2 import attempt2_lineage
from trade_rl.evaluation.paper.store import PaperJournal
from trade_rl.evaluation.paper.supervisor import PaperCollector


def listing(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(root.iterdir())}


# ----------------------------------------------- N1: the generic collector path


def test_a_forged_available_protocol_is_refused_by_run_supervise_collector_and_launch(
    tmp_path, capsys
):
    forged = tmp_path / "attempt2"
    digest = forged_available(tmp_path, tmp_path / "never-existed")
    before = listing(forged)
    log = tmp_path / "child.log"

    for arguments in (
        ["run", "--root", str(forged), "--protocol-sha256", digest],
        [
            "supervise",
            "--root",
            str(forged),
            "--protocol-sha256",
            digest,
            "--log-file",
            str(log),
        ],
    ):
        assert cli.main(arguments) == 2
        error = json.loads(capsys.readouterr().err)["error"]
        assert "cannot be re-verified" in error
        assert "predecessor_root_not_a_directory" in error
    assert listing(forged) == before  # no marker, heartbeat, control file or lock
    assert not log.exists()

    with pytest.raises(ValueError, match="cannot be re-verified"):
        with PaperCollector(forged, expected_protocol_sha256=digest):
            pytest.fail("the collector must not open")
    assert listing(forged) == before

    recorder = Recorder()
    reasons = refusal_of(lambda **kw: launch(tmp_path, digest, recorder, **kw))
    assert (
        "predecessor_root_not_reverifiable:predecessor_root_not_a_directory" in reasons
    )
    assert recorder.calls == []


def test_the_collector_gate_reads_before_the_start_and_for_a_fresh_root(tmp_path):
    root = tmp_path / "attempt1-root"
    root.mkdir()
    forged_available(tmp_path, root)
    sealed = tmp_path / "attempt2"
    protocol = json.loads((sealed / "protocol.json").read_bytes())["protocol"]
    plan = protocol["study"]["research_plan"]
    reader = FakeReader()

    def problems(now_offset, read=reader):
        return predecessor.collection_lineage_problems(
            plan,
            root=sealed,
            start_at=START,
            now=lambda: START + now_offset,
            read=read,
        )

    assert problems(timedelta(minutes=-10)) == [] and len(reader.calls) == 1
    # After the start a root with no journal event yet is still re-read (a late first
    # start is not a restart).
    assert problems(timedelta(minutes=5)) == [] and len(reader.calls) == 2
    mismatch = FakeReader(v1_report(tip="0" * 64))
    assert problems(timedelta(minutes=-10), mismatch) == [
        "predecessor_root_not_reverifiable:predecessor_root_tip_differs"
    ]
    # A recorded event is not proof that a lost predecessor was verified before
    # that event: a public PaperEngine.gap can write it without root validation.
    journal = PaperJournal(sealed, expected_protocol_sha256=_digest(sealed))
    journal.append(
        "k0", "gap", {"request": {}, "result": {}}, expected_parent=_digest(sealed)
    )
    gone = FakeReader(error=ValueError("root gone"))
    assert problems(timedelta(minutes=5), gone) == [
        "predecessor_root_not_reverifiable:predecessor_root_unreadable:ValueError"
    ]
    # A restart before the start also refuses the unavailable predecessor.
    assert problems(timedelta(minutes=-10), gone) == [
        "predecessor_root_not_reverifiable:predecessor_root_unreadable:ValueError"
    ]


def _digest(root: Path) -> str:
    import hashlib

    return hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest()


def test_unavailable_and_non_v3_lineages_are_not_touched_by_the_collector_gate(
    tmp_path,
):
    def never(*_a, **_k):
        raise AssertionError("nothing to re-read")

    def now():
        raise AssertionError("the clock must not be read")

    for plan in (
        None,
        {},
        {"attempt_lineage": None},
        {"attempt_lineage": attempt2_lineage("unavailable")},
        {
            "attempt_lineage": {
                "schema": "carry_paper_attempt_lineage_v2",
                "attempt_number": 1,
                "predecessor": None,
            }
        },
    ):
        assert (
            predecessor.collection_lineage_problems(
                plan, root=tmp_path, start_at=START, now=now, read=never
            )
            == []
        )


def test_unavailable_still_works_end_to_end_through_attempt2(tmp_path, capsys):
    from tests.evaluation.test_paper_attempt2 import seal

    result = seal(tmp_path)  # predecessor_root_status="unavailable"
    root = tmp_path / "attempt2"
    recorder = Recorder()
    outcome, _ = launch(tmp_path, result["protocol_sha256"], recorder)
    assert outcome["exit_code"] == 0
    # The generic collector gate also accepts it (nothing to re-read).
    assert (
        cli._refuse_non_screen_root(root, result["protocol_sha256"]) is None
    )  # no exception


# ------------------------------------------------ the private seal token (N1)


def test_the_seal_token_is_private_and_a_truthy_flag_does_not_unlock_v3(tmp_path):
    lineage = attempt2_lineage("unavailable")
    with pytest.raises(TypeError):
        operations.seal_paper_study(  # the old public flag no longer exists
            tmp_path / "a", start_at=START, attempt_lineage=lineage, via_attempt2=True
        )
    for token in (True, 1, "attempt2", object()):
        with pytest.raises(ValueError, match="only be sealed through attempt2"):
            operations.seal_paper_study(
                tmp_path / "b",
                start_at=START,
                attempt_lineage=lineage,
                attempt2_seal_token=token,
                clock=clock,
            )
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    assert not hasattr(operations, "via_attempt2")


# ------------------------------------------------ N2: a hot rollback journal


def make_hot_root(tmp_path: Path) -> tuple[Path, str, str]:
    """A copy of a journal taken mid-transaction after pages spilled to the file."""
    root = tmp_path / "source"
    digest = PaperJournal.initialize(root, {"engine": "synthetic"})
    journal = PaperJournal(root, expected_protocol_sha256=digest)
    tip = journal.append(
        "k0", "decide", {"request": {}, "result": {}}, expected_parent=digest
    )["sha256"]
    database = sqlite3.connect(root / "journal.sqlite", isolation_level=None)
    database.execute("PRAGMA cache_size=1")
    database.execute("BEGIN IMMEDIATE")
    for index in range(1, 60):
        database.execute(
            "INSERT INTO events(sequence,event_key,sha256,body) VALUES(?,?,?,?)",
            (index, f"junk{index}", f"{index:064x}", b"x" * 20_000),
        )
    hot = tmp_path / "hot"
    shutil.copytree(root, hot)  # the journal file exists only during the transaction
    database.rollback()
    database.close()
    assert (hot / "journal.sqlite-journal").is_file()
    return hot, digest, tip


def test_read_only_refuses_a_hot_journal_and_leaves_it_intact(tmp_path):
    hot, digest, _tip = make_hot_root(tmp_path)
    before = listing(hot)
    with pytest.raises(sqlite3.OperationalError):
        PaperJournal(hot, expected_protocol_sha256=digest, read_only=True)
    assert listing(hot) == before  # journal still present, database bytes unchanged


def test_writable_open_recovers_the_hot_journal_which_read_only_must_not(tmp_path):
    hot, digest, tip = make_hot_root(tmp_path)
    journal = PaperJournal(hot, expected_protocol_sha256=digest, read_only=False)
    assert [record["sha256"] for record in journal.events()] == [tip]
    assert not (hot / "journal.sqlite-journal").exists()  # recovery deleted it


def test_predecessor_verification_fails_closed_on_a_hot_journal(tmp_path, monkeypatch):
    hot, digest, tip = make_hot_root(tmp_path)
    monkeypatch.setattr(predecessor, "ATTEMPT_ONE_PROTOCOL_SHA256", digest)
    monkeypatch.setattr(predecessor, "ATTEMPT_ONE_FINAL_TIP_SHA256", tip)
    monkeypatch.setattr(predecessor, "ATTEMPT_ONE_EVENT_COUNT", 1)
    before = listing(hot)
    assert predecessor.verify_predecessor_root(hot) == [
        "predecessor_root_unreadable:OperationalError"
    ]
    assert listing(hot) == before


# --------------------------- M1 (review 5720105406): the lowest sealing layer


@pytest.mark.parametrize("status", ["available", "unavailable"])
def test_seal_collection_refuses_a_v3_lineage_without_the_private_token(
    tmp_path, status
):
    from trade_rl.evaluation.paper.account import PaperSettings
    from trade_rl.evaluation.paper.supervisor import seal_collection

    lineage = attempt2_lineage(status, "D:/old/root" if status == "available" else None)
    plan = operations.screen_plan(attempt_lineage=lineage)
    settings = PaperSettings(start_at=START, close_at=START + timedelta(days=90))
    # No private token at all.
    with pytest.raises(ValueError, match="only be sealed through attempt2"):
        seal_collection(
            tmp_path / "plain", settings=settings, research_plan=plan, clock=clock
        )
    for token in (None, True, object(), "attempt2"):
        with pytest.raises(ValueError, match="only be sealed through attempt2"):
            seal_collection(
                tmp_path / "wrong",
                settings=settings,
                research_plan=plan,
                attempt2_seal_token=token,
                clock=clock,
            )
    assert not (tmp_path / "plain").exists() and not (tmp_path / "wrong").exists()


def tampered_unavailable_plan():
    lineage = attempt2_lineage("unavailable")
    lineage["predecessor_root"]["user_statement"]["quote"] = "different words"
    return dict(
        operations.screen_plan(attempt_lineage=attempt2_lineage("unavailable")),
        attempt_lineage=lineage,
    )


def seal_raw(tmp_path, plan):
    """Seal without any attempt2 precondition and without the private token."""
    from trade_rl.evaluation.paper.account import PaperSettings
    from trade_rl.evaluation.paper.engine import PaperEngine
    from trade_rl.evaluation.runs import build_candidate_run_provenance

    return PaperEngine.initialize(
        tmp_path / "attempt2",
        settings=PaperSettings(start_at=START, close_at=START + timedelta(days=90)),
        study=dict(
            schema="paper_public_collection_v1",
            created_at=(START - timedelta(hours=2)).isoformat(),
            interval_seconds=60,
            terminal_grace_seconds=180,
            provenance=build_candidate_run_provenance(),
            research_plan=plan,
        ),
    )


def test_a_non_canonical_unavailable_protocol_is_refused_everywhere(tmp_path, capsys):
    # Sealed through the lowest layer (PaperEngine.initialize), with no token.
    digest = seal_raw(tmp_path, tampered_unavailable_plan())
    forged = tmp_path / "attempt2"
    before = listing(forged)
    log = tmp_path / "child.log"
    for arguments in (
        ["run", "--root", str(forged), "--protocol-sha256", digest],
        [
            "supervise",
            "--root",
            str(forged),
            "--protocol-sha256",
            digest,
            "--log-file",
            str(log),
        ],
    ):
        assert cli.main(arguments) == 2
        error = json.loads(capsys.readouterr().err)["error"]
        assert "lineage_v3_not_canonical" in error
    with pytest.raises(ValueError, match="lineage_v3_not_canonical"):
        with PaperCollector(forged, expected_protocol_sha256=digest):
            pytest.fail("the collector must not open")
    assert listing(forged) == before and not log.exists()
    recorder = Recorder()
    reasons = refusal_of(lambda **kw: launch(tmp_path, digest, recorder, **kw))
    assert "protocol_not_attempt2_lineage" in reasons
    assert recorder.calls == []


def test_an_available_record_that_is_not_canonical_for_its_path_is_refused(tmp_path):
    root = tmp_path / "attempt1-root"
    root.mkdir()
    lineage = attempt2_lineage("available", root)
    lineage["predecessor"]["reason_codes"] = ["manual_abort"]
    plan = dict(
        operations.screen_plan(attempt_lineage=attempt2_lineage("available", root)),
        attempt_lineage=lineage,
    )
    seal_raw(tmp_path, plan)
    problems = predecessor.collection_lineage_problems(
        plan,
        root=tmp_path / "attempt2",
        start_at=START,
        now=lambda: START - timedelta(minutes=10),
        read=FakeReader(),
    )
    assert problems == ["lineage_v3_not_canonical"]


def test_a_canonical_unavailable_protocol_passes_the_collector_gate(tmp_path):
    plan = operations.screen_plan(attempt_lineage=attempt2_lineage("unavailable"))
    seal_raw(tmp_path, plan)
    assert (
        predecessor.collection_lineage_problems(
            plan, root=tmp_path / "attempt2", start_at=START, now=lambda: START
        )
        == []
    )


@pytest.mark.parametrize(
    "mutation",
    ["attempt_float", "predecessor_bool", "attestation_integer"],
)
def test_collector_rejects_canonical_looking_type_mutations_from_raw_engine(
    tmp_path, mutation
):
    """JSON values must retain their exact types, even when Python == coerces them."""
    lineage = attempt2_lineage("unavailable")
    if mutation == "attempt_float":
        lineage["attempt_number"] = 2.0
    elif mutation == "predecessor_bool":
        lineage["predecessor"]["attempt_number"] = True
    else:
        lineage["predecessor_root"]["re_verified_at_seal"] = 0
    with pytest.raises(ValueError):
        operations.validate_attempt_lineage(lineage)
    valid_plan = operations.screen_plan(attempt_lineage=attempt2_lineage("unavailable"))
    tampered_plan = dict(valid_plan, attempt_lineage=lineage)
    digest = seal_raw(tmp_path, tampered_plan)
    root = tmp_path / "attempt2"
    before = listing(root)
    with pytest.raises(ValueError, match="lineage_v3_not_canonical"):
        with PaperCollector(root, expected_protocol_sha256=digest):
            pytest.fail("not a canonical lineage")
    assert listing(root) == before


@pytest.mark.parametrize(
    "wrong_schema",
    [
        "carry_paper_attempt_lineage_v3 ",
        "carry_paper_attempt_lineage_v3\u200b",
    ],
)
def test_forged_v3_schema_is_refused_before_collection_or_seal(tmp_path, wrong_schema):
    from trade_rl.evaluation.paper.account import PaperSettings
    from trade_rl.evaluation.paper.supervisor import seal_collection

    lineage = attempt2_lineage("available", tmp_path / "root-missing")
    lineage["schema"] = wrong_schema
    plan = dict(
        operations.screen_plan(attempt_lineage=attempt2_lineage("unavailable")),
        attempt_lineage=lineage,
    )
    with pytest.raises(ValueError, match="lineage|schema"):
        seal_collection(
            tmp_path / "ordinary-seal",
            settings=PaperSettings(start_at=START, close_at=START + timedelta(days=90)),
            research_plan=plan,
            clock=clock,
        )
    assert not (tmp_path / "ordinary-seal").exists()
    digest = seal_raw(tmp_path, plan)
    root = tmp_path / "attempt2"
    before = listing(root)
    with pytest.raises(ValueError, match="lineage|schema"):
        with PaperCollector(root, expected_protocol_sha256=digest):
            pytest.fail("forged schema must not start acquisition")
    assert listing(root) == before


@pytest.mark.parametrize(
    "mutation", ["attempt_float", "predecessor_bool", "attestation_integer"]
)
def test_attempt2_launch_refuses_canonical_looking_numeric_mutations(
    tmp_path, mutation
):
    lineage = attempt2_lineage("unavailable")
    if mutation == "attempt_float":
        lineage["attempt_number"] = 2.0
    elif mutation == "predecessor_bool":
        lineage["predecessor"]["attempt_number"] = True
    else:
        lineage["predecessor_root"]["re_verified_at_seal"] = 0
    plan = dict(
        operations.screen_plan(attempt_lineage=attempt2_lineage("unavailable")),
        attempt_lineage=lineage,
    )
    digest = seal_raw(tmp_path, plan)
    reasons = attempt2._protocol_reasons(tmp_path / "attempt2", digest, START)
    assert "protocol_not_attempt2_lineage" in reasons
    recorder = Recorder()
    launch_reasons = refusal_of(
        lambda **kwargs: launch(tmp_path, digest, recorder, **kwargs)
    )
    assert "protocol_not_attempt2_lineage" in launch_reasons
    assert recorder.calls == []


def test_seal_refuses_dict_subclass_that_forges_get_method(tmp_path):
    from trade_rl.evaluation.paper.account import PaperSettings
    from trade_rl.evaluation.paper.supervisor import seal_collection

    class HiddenLineage(dict):
        def get(self, key, default=None):
            if key == "attempt_lineage":
                return None
            return super().get(key, default)

    plan = HiddenLineage(
        operations.screen_plan(attempt_lineage=attempt2_lineage("unavailable"))
    )
    with pytest.raises(ValueError, match="lineage|plan"):
        seal_collection(
            tmp_path / "forged",
            settings=PaperSettings(start_at=START, close_at=START + timedelta(days=90)),
            research_plan=plan,
            clock=clock,
        )
    assert not (tmp_path / "forged").exists()


def test_forged_native_gap_event_cannot_skip_predecessor_root_read(tmp_path):
    """An event from public PaperEngine.gap is not lineage verification evidence."""
    from trade_rl.evaluation.paper.engine import PaperEngine

    root = tmp_path / "attempt2"
    digest = forged_available(tmp_path, tmp_path / "predecessor-never-existed")
    engine = PaperEngine(root, expected_protocol_sha256=digest)
    engine.gap("forged-gap", at=START + timedelta(seconds=2), reason="observation_gap")
    before = listing(root)
    with pytest.raises(ValueError, match="predecessor_root_not_a_directory"):
        with PaperCollector(
            root,
            expected_protocol_sha256=digest,
            clock=lambda: START + timedelta(minutes=5),
        ):
            pytest.fail("forged journal evidence must not authorize acquisition")
    assert listing(root) == before


def test_v3_lineage_dict_subclass_cannot_hide_schema_from_sealer(tmp_path):
    from trade_rl.evaluation.paper.account import PaperSettings
    from trade_rl.evaluation.paper.supervisor import seal_collection

    class HiddenSchema(dict):
        def get(self, key, default=None):
            if key == "schema":
                return "carry_paper_attempt_lineage_v2"
            return super().get(key, default)

    lineage = HiddenSchema(attempt2_lineage("unavailable"))
    plan = dict(
        operations.screen_plan(attempt_lineage=attempt2_lineage("unavailable")),
        attempt_lineage=lineage,
    )
    with pytest.raises(ValueError, match="schema|lineage"):
        seal_collection(
            tmp_path / "forged-nested",
            settings=PaperSettings(start_at=START, close_at=START + timedelta(days=90)),
            research_plan=plan,
            clock=clock,
        )
    assert not (tmp_path / "forged-nested").exists()
