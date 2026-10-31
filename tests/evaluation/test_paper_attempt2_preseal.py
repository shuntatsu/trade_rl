"""Pre-seal follow-ups for attempt 2 (review 5719733380): N1 collector re-verification,
N2 a discriminating hot-journal fixture, the private seal token and N5 wording.

Offline only: sockets are blocked and no collector cycle ever runs.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.evaluation.test_paper_attempt2 import (  # noqa: F401  (autouse fixture)
    AUTH,
    NOW,
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
from trade_rl.evaluation.paper.engine import PaperEngine
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


def seal_raw(tmp_path, plan, name="attempt2"):
    """Write a protocol at the lowest layer (PaperJournal): no token, no gate."""
    from trade_rl.evaluation.paper.account import PaperSettings
    from trade_rl.evaluation.runs import build_candidate_run_provenance

    settings = PaperSettings(start_at=START, close_at=START + timedelta(days=90))
    return PaperJournal.initialize(
        tmp_path / name,
        dict(
            engine="funding_carry_paper_engine_v1",
            settings=settings.to_dict(),
            study=dict(
                schema="paper_public_collection_v1",
                created_at=(START - timedelta(hours=2)).isoformat(),
                interval_seconds=60,
                terminal_grace_seconds=180,
                provenance=build_candidate_run_provenance(),
                research_plan=plan,
            ),
        ),
    )


def test_a_non_canonical_unavailable_protocol_is_refused_everywhere(tmp_path, capsys):
    # Written through the lowest layer (PaperJournal), with no token.
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


# ------------- review 5721122305: M-A (v2 label) and M-B (public PaperEngine)


def v2_label_with_v3_root(status):
    lineage = attempt2_lineage(status, "D:/old/root" if status == "available" else None)
    lineage["schema"] = "carry_paper_attempt_lineage_v2"
    return lineage


def v2_forgeries():
    unavailable = v2_label_with_v3_root("unavailable")
    unavailable["predecessor_root"]["user_statement"]["quote"] = "different words"
    extra_key = attempt2_lineage("unavailable")
    extra_key["schema"] = "carry_paper_attempt_lineage_v2"
    extra_key["unexpected"] = 1
    return {
        "available_under_v2_label": v2_label_with_v3_root("available"),
        "unavailable_under_v2_label": v2_label_with_v3_root("unavailable"),
        "altered_unavailable_under_v2_label": unavailable,
        "v2_with_unexpected_key": extra_key,
    }


@pytest.mark.parametrize("name", sorted(v2_forgeries()))
def test_a_v2_label_cannot_carry_v3_fields_through_any_gate(tmp_path, capsys, name):
    from trade_rl.evaluation.paper.account import PaperSettings
    from trade_rl.evaluation.paper.supervisor import seal_collection

    forged = v2_forgeries()[name]
    plan = {"schema": "carry_paper_screen_v2", "attempt_lineage": forged}
    settings = PaperSettings(start_at=START, close_at=START + timedelta(days=90))
    # The shared classifier, directly and through the validator.
    assert predecessor.lineage_structure_problems(forged) == [
        "lineage_v2_unexpected_fields"
    ]
    with pytest.raises(ValueError):
        operations.validate_attempt_lineage(forged, sealed_at=NOW)
    # seal_collection with and without the token, and the engine's own initialize.
    for token in (None, operations._ATTEMPT2_SEAL_TOKEN):
        with pytest.raises(ValueError, match="non-canonical|unsupported"):
            seal_collection(
                tmp_path / "sealed",
                settings=settings,
                research_plan=plan,
                attempt2_seal_token=token,
                clock=clock,
            )
    assert not (tmp_path / "sealed").exists()
    # A protocol written at the lowest layer is refused by every entry point.
    digest = seal_raw(tmp_path, plan)
    root = tmp_path / "attempt2"
    before = listing(root)
    for arguments in (
        ["run", "--root", str(root), "--protocol-sha256", digest],
        [
            "supervise",
            "--root",
            str(root),
            "--protocol-sha256",
            digest,
            "--log-file",
            str(tmp_path / "child.log"),
        ],
    ):
        assert cli.main(arguments) == 2
        assert (
            "lineage_v2_unexpected_fields"
            in json.loads(capsys.readouterr().err)["error"]
        )
    with pytest.raises(ValueError, match="lineage_v2_unexpected_fields"):
        with PaperCollector(root, expected_protocol_sha256=digest):
            pytest.fail("the collector must not open")
    with pytest.raises(ValueError, match="not canonical"):
        PaperEngine(root, expected_protocol_sha256=digest)
    assert listing(root) == before
    assert not (tmp_path / "child.log").exists()


def test_a_plain_v2_lineage_is_still_sealed_and_collected(tmp_path):
    from trade_rl.evaluation.paper.account import PaperSettings
    from trade_rl.evaluation.paper.supervisor import seal_collection

    lineage = {
        "schema": "carry_paper_attempt_lineage_v2",
        "attempt_number": 1,
        "predecessor": None,
    }
    plan = operations.screen_plan(attempt_lineage=lineage)
    assert predecessor.lineage_structure_problems(lineage) == []
    digest = seal_collection(
        tmp_path / "first",
        settings=PaperSettings(start_at=START, close_at=START + timedelta(days=90)),
        research_plan=plan,
        clock=clock,
    )
    PaperEngine(tmp_path / "first", expected_protocol_sha256=digest)  # opens


def test_public_engine_refuses_to_initialize_or_open_a_noncanonical_v3_root(tmp_path):
    from trade_rl.evaluation.paper.account import PaperSettings

    settings = PaperSettings(start_at=START, close_at=START + timedelta(days=90))
    mutated = attempt2_lineage("unavailable")
    mutated["attempt_number"] = 2.0
    study = dict(
        schema="paper_public_collection_v1",
        created_at=(START - timedelta(hours=2)).isoformat(),
        interval_seconds=60,
        terminal_grace_seconds=180,
        research_plan={"attempt_lineage": mutated},
    )
    # initialize: refused for a non-canonical record, with and without the token ...
    for token in (None, operations._ATTEMPT2_SEAL_TOKEN):
        with pytest.raises(
            ValueError, match="only be sealed through attempt2|non-canonical"
        ):
            PaperEngine.initialize(
                tmp_path / "e",
                settings=settings,
                study=study,
                attempt2_seal_token=token,
            )
    # ... and a canonical v3 is refused without the token, accepted with it.
    canonical = dict(
        study, research_plan={"attempt_lineage": attempt2_lineage("unavailable")}
    )
    with pytest.raises(ValueError, match="only be sealed through attempt2"):
        PaperEngine.initialize(tmp_path / "e", settings=settings, study=canonical)
    assert not (tmp_path / "e").exists()
    digest = PaperEngine.initialize(
        tmp_path / "e",
        settings=settings,
        study=canonical,
        attempt2_seal_token=operations._ATTEMPT2_SEAL_TOKEN,
    )
    engine = PaperEngine(tmp_path / "e", expected_protocol_sha256=digest)
    assert engine.status()["events"] == 0
    # A root written below the engine with a mutated lineage cannot be opened, so
    # gap/decide/execute cannot append to it.
    bad = seal_raw(tmp_path, {"attempt_lineage": mutated}, name="bad")
    before = listing(tmp_path / "bad")
    with pytest.raises(ValueError, match="not canonical"):
        PaperEngine(tmp_path / "bad", expected_protocol_sha256=bad)
    assert listing(tmp_path / "bad") == before  # nothing was appended


def test_duplicate_json_keys_are_refused_before_the_lock_with_no_marker(
    tmp_path, capsys
):
    digest = seal_raw(
        tmp_path,
        operations.screen_plan(attempt_lineage=attempt2_lineage("unavailable"))
        | {"purpose": "x"},
    )
    root = tmp_path / "attempt2"
    raw = (root / "protocol.json").read_bytes()
    # Re-write the protocol with the lineage schema key repeated (last one wins in
    # json.loads); the digest is the new file's own digest.
    doubled = raw.replace(
        b'"schema":"carry_paper_attempt_lineage_v3"',
        b'"schema":"carry_paper_attempt_lineage_v2","schema":"carry_paper_attempt_lineage_v3"',
        1,
    )
    assert doubled != raw
    import hashlib

    (root / "protocol.json").write_bytes(doubled)
    new_digest = hashlib.sha256(doubled).hexdigest()
    before = listing(root)
    with pytest.raises(ValueError, match="repeats an object key"):
        with PaperCollector(root, expected_protocol_sha256=new_digest):
            pytest.fail("the collector must not open")
    assert cli.main(["run", "--root", str(root), "--protocol-sha256", new_digest]) == 2
    assert "repeats an object key" in json.loads(capsys.readouterr().err)["error"]
    assert listing(root) == before  # no failure marker, lock file or heartbeat
    assert digest != new_digest


def test_collector_helper_requires_exact_plain_types(tmp_path):
    class Plan(dict):
        def get(self, key, default=None):
            return None  # hides the lineage from a naive reader

    forged = {"attempt_lineage": v2_forgeries()["available_under_v2_label"]}
    assert predecessor.collection_lineage_problems(
        Plan(forged), root=tmp_path, start_at=START, now=lambda: START
    ) == ["research_plan_not_a_plain_dict"]
    assert predecessor.collection_lineage_problems(
        ["not", "a", "mapping"], root=tmp_path, start_at=START, now=lambda: START
    ) == ["research_plan_not_a_plain_dict"]
    assert (
        predecessor.collection_lineage_problems(
            None, root=tmp_path, start_at=START, now=lambda: START
        )
        == []
    )


def test_the_comparator_is_public_and_type_strict():
    assert not hasattr(predecessor, "_same_canonical_record")
    assert predecessor.same_canonical_record({"a": [1]}, {"a": [1]})
    assert not predecessor.same_canonical_record({"a": 1.0}, {"a": 1})
    assert not predecessor.same_canonical_record({"a": True}, {"a": 1})


# ------------------------- review 5721346805: N1 (initialize), N2 (v2 attempt 2)


def _settings():
    from trade_rl.evaluation.paper.account import PaperSettings

    return PaperSettings(start_at=START, close_at=START + timedelta(days=90))


def test_engine_initialize_requires_plain_dicts_so_a_get_override_cannot_hide_v3(
    tmp_path,
):
    class HidingStudy(dict):
        def get(self, key, default=None):  # hides the plan from a naive lookup
            return None

    class HidingPlan(dict):
        def get(self, key, default=None):
            return None

    canonical_plan = {"attempt_lineage": attempt2_lineage("unavailable")}
    for study in (
        HidingStudy(schema="paper_public_collection_v1", research_plan=canonical_plan),
        dict(
            schema="paper_public_collection_v1",
            research_plan=HidingPlan(canonical_plan),
        ),
    ):
        with pytest.raises(ValueError, match="plain dict"):
            PaperEngine.initialize(
                tmp_path / "hidden", settings=_settings(), study=study
            )
        assert not (tmp_path / "hidden").exists()
    # Q1b: the plain-dict canonical v3 is still refused without the token.
    with pytest.raises(ValueError, match="only be sealed through attempt2"):
        PaperEngine.initialize(
            tmp_path / "plain",
            settings=_settings(),
            study=dict(
                schema="paper_public_collection_v1", research_plan=canonical_plan
            ),
        )
    assert not (tmp_path / "plain").exists()


def test_engine_initialize_refuses_a_non_dict_research_plan(tmp_path):
    for plan in ("text", ["list"], 1):
        with pytest.raises(ValueError, match="plain dict"):
            PaperEngine.initialize(
                tmp_path / "plan",
                settings=_settings(),
                study=dict(schema="paper_public_collection_v1", research_plan=plan),
            )
    assert not (tmp_path / "plan").exists()
    # A root written below the engine with a non-dict plan cannot be opened either.
    digest = seal_raw(tmp_path, "just a string", name="odd")
    with pytest.raises(ValueError, match="plain dict"):
        PaperEngine(tmp_path / "odd", expected_protocol_sha256=digest)


def v2_attempt(number):
    return {
        "schema": "carry_paper_attempt_lineage_v2",
        "attempt_number": number,
        "predecessor": {
            "attempt_number": number - 1,
            "protocol_sha256": "a" * 64,
            "final_tip_sha256": "b" * 64,
            "disposition": "invalidated",
            "reason_codes": ["observation_gap"],
            "last_observed_at": "2026-09-17T23:11:01+00:00",
        },
    }


@pytest.mark.parametrize("number", [2, 3])
def test_v2_label_is_for_attempt_one_only_at_every_gate(tmp_path, capsys, number):
    from trade_rl.evaluation.paper.supervisor import seal_collection

    lineage = v2_attempt(number)
    assert predecessor.lineage_structure_problems(lineage) == [
        "lineage_v2_attempt_requires_v3"
    ]
    # validate_attempt_lineage and seal_paper_study
    with pytest.raises(ValueError, match="lineage_v2_attempt_requires_v3"):
        operations.validate_attempt_lineage(lineage, sealed_at=NOW)
    with pytest.raises(ValueError, match="lineage_v2_attempt_requires_v3"):
        operations.seal_paper_study(
            tmp_path / "sealed",
            start_at=NOW + timedelta(hours=2),
            attempt_lineage=lineage,
            clock=clock,
        )
    # seal_collection (with the token, too)
    plan = {"schema": "carry_paper_screen_v2", "attempt_lineage": lineage}
    for token in (None, operations._ATTEMPT2_SEAL_TOKEN):
        with pytest.raises(ValueError, match="lineage_v2_attempt_requires_v3"):
            seal_collection(
                tmp_path / "sealed",
                settings=_settings(),
                research_plan=plan,
                attempt2_seal_token=token,
                clock=clock,
            )
    # cli seal
    path = tmp_path / "lineage.json"
    path.write_text(json.dumps(lineage), encoding="utf-8")
    code = cli.main(
        [
            "seal",
            "--root",
            str(tmp_path / "clisealed"),
            "--start-at",
            (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            "--attempt-lineage-json",
            str(path),
        ]
    )
    assert code == 2
    assert (
        "lineage_v2_attempt_requires_v3" in json.loads(capsys.readouterr().err)["error"]
    )
    assert not (tmp_path / "sealed").exists()
    assert not (tmp_path / "clisealed").exists()
    # A protocol written below the gates is refused by PaperCollector, cli run and
    # supervise, and the engine will not open it.
    digest = seal_raw(tmp_path, plan)
    root = tmp_path / "attempt2"
    before = listing(root)
    with pytest.raises(ValueError, match="lineage_v2_attempt_requires_v3"):
        with PaperCollector(root, expected_protocol_sha256=digest):
            pytest.fail("the collector must not open")
    for arguments in (
        ["run", "--root", str(root), "--protocol-sha256", digest],
        [
            "supervise",
            "--root",
            str(root),
            "--protocol-sha256",
            digest,
            "--log-file",
            str(tmp_path / "child.log"),
        ],
    ):
        assert cli.main(arguments) == 2
        assert (
            "lineage_v2_attempt_requires_v3"
            in json.loads(capsys.readouterr().err)["error"]
        )
    with pytest.raises(ValueError, match="not canonical"):
        PaperEngine(root, expected_protocol_sha256=digest)
    assert listing(root) == before and not (tmp_path / "child.log").exists()


def test_attempt_one_under_the_v2_label_still_works_end_to_end(tmp_path):
    first = {
        "schema": "carry_paper_attempt_lineage_v2",
        "attempt_number": 1,
        "predecessor": None,
    }
    assert predecessor.lineage_structure_problems(first) == []
    digest = operations.seal_paper_study(
        tmp_path / "first",
        start_at=NOW + timedelta(hours=2),
        attempt_lineage=first,
        clock=clock,
    )
    PaperEngine(tmp_path / "first", expected_protocol_sha256=digest)  # opens
    assert (
        predecessor.collection_lineage_problems(
            operations.screen_plan(attempt_lineage=first),
            root=tmp_path / "first",
            start_at=START,
            now=lambda: START,
        )
        == []
    )


@pytest.mark.parametrize("variant", ["reordered", "extra_key"])
def test_non_canonical_protocol_bytes_are_refused_before_the_lock_with_no_marker(
    tmp_path, capsys, variant
):
    import hashlib

    plan = operations.screen_plan(attempt_lineage=attempt2_lineage("unavailable"))
    seal_raw(tmp_path, plan)
    root = tmp_path / "attempt2"
    value = json.loads((root / "protocol.json").read_bytes())
    if variant == "reordered":
        value = {"protocol": value["protocol"], "schema": value["schema"]}
        edited = json.dumps(value, separators=(",", ":")).encode("utf-8")
    else:
        value["extra"] = 1
        edited = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    (root / "protocol.json").write_bytes(edited)
    digest = hashlib.sha256(edited).hexdigest()
    before = listing(root)
    with pytest.raises(ValueError, match="not canonical"):
        with PaperCollector(root, expected_protocol_sha256=digest):
            pytest.fail("the collector must not open")
    assert cli.main(["run", "--root", str(root), "--protocol-sha256", digest]) == 2
    assert "not canonical" in json.loads(capsys.readouterr().err)["error"]
    assert listing(root) == before  # no failure marker, no lock file


def test_available_root_problems_uses_exact_types():
    class Lineage(dict):
        pass

    lineage = Lineage(attempt2_lineage("available", "D:/old/root"))
    assert predecessor.available_root_problems(lineage, read=FakeReader()) == []
