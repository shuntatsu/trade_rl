"""The FS1 seal-record generator: exact content, consistency refusals, print-only."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.evaluation.forward_acquisition_support import (
    AUTH,
    CODE_SHA,
    Scenario,
    make_scenario,
)
from trade_rl.evaluation.forward_acquisition import cli, plan, seal_record
from trade_rl.evaluation.forward_acquisition.acquire import AcquisitionRefusal
from trade_rl.evaluation.forward_study import seal
from trade_rl.evaluation.forward_study.registry import (
    FROZEN_PARAMS_DIGEST,
    FS1_REGISTRY,
)

SEALED = "d" * 40


def write_real_root(root: Path, **overrides: Any) -> tuple[bytes, bytes]:
    """A hand-made acquisition root consistent with the REAL frozen registry."""

    root.mkdir()
    receipt: dict[str, Any] = {
        "schema_version": plan.RECEIPT_SCHEMA,
        "command_line": (
            "python -m trade_rl.evaluation.forward_acquisition acquire "
            f"--authorization {AUTH} --output-root {root} --allow-network"
        ),
        "output_root": str(root),
        "builder": {"git_head": SEALED, "tracked_tree_clean": True},
        "authorization_reference": AUTH,
        "registry_params_digest": FS1_REGISTRY.digest(),
        "build_policy_digest": FS1_REGISTRY.build_policy_digest(),
        "fetched_utc": "2027-05-02T01:00:00Z",
        "exchange_information": {
            "source_uri": plan.EXCHANGE_INFO_URL,
            "retrieved_at": "2027-05-02T01:00:05+00:00",
            "raw_payload_sha256": "e" * 64,
        },
        "development_reuse": {
            "dataset_id": plan.DEV_DATASET_ID,
            "artifact_digest": plan.DEV_ARTIFACT_DIGEST,
            "roster_digest": "f" * 64,
            "archive_count": 670,
        },
        "source_archives": {
            "count": 760,
            "roster_digest": "9" * 64,
            "fetched_count": 90,
        },
        "dataset": {"dataset_id": "a" * 64, "artifact_digest": "c" * 64},
    }
    receipt.update(overrides.pop("receipt", {}))
    receipt_bytes = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    (root / plan.RECEIPT_NAME).write_bytes(receipt_bytes)
    stamps = np.array(
        ["2021-01-01T01:00:00", "2027-05-01T00:00:00"], dtype="datetime64[s]"
    )
    dataset = SimpleNamespace(
        dataset_id="a" * 64, symbols=FS1_REGISTRY.symbols, timestamps=stamps, n_bars=2
    )
    arguments: dict[str, Any] = {
        "source_manifest": {
            f"{plan.VISION_ROOT}/futures/um/monthly/klines/BTCUSDT/1h/x.zip": "1" * 64,
            plan.SOURCE_KEY_RECEIPT: hashlib.sha256(receipt_bytes).hexdigest(),
            plan.SOURCE_KEY_EXCHANGE_INFO: "e" * 64,
            plan.SOURCE_KEY_DATASET_ARTIFACT: "c" * 64,
            plan.SOURCE_KEY_DEV_ROSTER: "f" * 64,
        },
        "build_policy_digest": FS1_REGISTRY.build_policy_digest(),
        "authorization_reference": AUTH,
        "fetched_utc": "2027-05-02T01:00:00Z",
        "sealed_utc": "2027-05-02T02:00:00Z",
    }
    arguments.update(overrides)
    manifest_bytes = seal.manifest_bytes_from_dataset(dataset, **arguments)  # type: ignore[arg-type]
    (root / plan.MANIFEST_NAME).write_bytes(manifest_bytes)
    return manifest_bytes, receipt_bytes


def test_the_record_contains_every_required_item(tmp_path: Path) -> None:
    root = tmp_path / "root"
    manifest_bytes, receipt_bytes = write_real_root(root)
    text = seal_record.render_seal_record(root, SEALED)

    assert f"`{SEALED}`" in text
    assert FROZEN_PARAMS_DIGEST in text and "FROZEN_PARAMS_DIGEST" in text
    assert FS1_REGISTRY.build_policy_digest() in text
    assert hashlib.sha256(manifest_bytes).hexdigest() in text
    assert hashlib.sha256(receipt_bytes).hexdigest() in text
    assert "2027-05-02T01:00:00Z" in text
    assert AUTH in text
    assert "Scoring allowed at or after | 2027-05-02T00:00Z" in text
    assert "a" * 64 in text
    assert "retrieved 2027-05-02T01:00:05+00:00" in text
    assert "--allow-network" in text
    assert "<ISSUE-810-G4-SCORING-AUTHORIZATION-COMMENT-URL>" in text
    assert f"--manifest-sha256 {hashlib.sha256(manifest_bytes).hexdigest()}" in text
    assert f"--sealed-code-sha {SEALED}" in text
    assert "contains no economic result" in text
    assert "not the G4 scoring authorization" in text


def test_low_a_and_low_b_notes_are_stated_exactly(tmp_path: Path) -> None:
    root = tmp_path / "root"
    write_real_root(root)
    text = seal_record.render_seal_record(root, SEALED)

    assert "LOW-A:" in text and "Revision 2" in text and "PR #1049" in text
    assert "LOW-B:" in text
    assert "zero-byte claimed output" in text
    assert "removed by hand" in text and "never silently" in text
    assert "`output_exists`" in text
    assert seal_record.LOW_A_NOTE in text and seal_record.LOW_B_PROCEDURE in text
    assert "no official checksum check" in text


def test_a_builder_commit_that_differs_from_the_sealed_commit_is_disclosed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    write_real_root(root)
    other = seal_record.render_seal_record(root, "e" * 40)
    assert "differs from the sealed code commit" in other
    same = seal_record.render_seal_record(root, SEALED)
    assert "differs from the sealed code commit" not in same


def test_the_generator_refuses_inconsistent_inputs(tmp_path: Path) -> None:
    def reasons(root: Path, sha: str = SEALED) -> tuple[str, ...]:
        with pytest.raises(AcquisitionRefusal) as caught:
            seal_record.render_seal_record(root, sha)
        return caught.value.reasons

    good = tmp_path / "good"
    write_real_root(good)
    assert reasons(good, "abc") == ("sealed_code_sha_malformed",)
    assert reasons(good, "D" * 40) == ("sealed_code_sha_malformed",)

    empty = tmp_path / "empty"
    empty.mkdir()
    assert reasons(empty) == ("manifest_missing", "receipt_missing")

    tampered = tmp_path / "tampered"
    write_real_root(tampered)
    receipt_path = tampered / plan.RECEIPT_NAME
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    assert "receipt_sha256_not_bound_by_manifest" in reasons(tampered)

    wrong_dataset = tmp_path / "wd"
    write_real_root(wrong_dataset, receipt={"dataset": {"dataset_id": "9" * 64}})
    assert "receipt_dataset_id_differs_from_manifest" in reasons(wrong_dataset)

    early = tmp_path / "early"
    write_real_root(early, fetched_utc="2027-05-01T12:00:00Z")
    assert "manifest_fetched_before_settlement" in reasons(early)

    other_policy = tmp_path / "policy"
    write_real_root(other_policy, build_policy_digest="0" * 64)
    assert "manifest_build_policy_mismatch" in reasons(other_policy)

    other_params = tmp_path / "params"
    write_real_root(other_params, receipt={"registry_params_digest": "0" * 64})
    assert "receipt_params_digest_differs_from_registry" in reasons(other_params)

    unbound = tmp_path / "unbound"
    write_real_root(
        unbound,
        source_manifest={
            plan.VISION_ROOT + "/x.zip": "1" * 64,
            plan.SOURCE_KEY_RECEIPT: "2" * 64,
        },
    )
    got = reasons(unbound)
    assert "exchange_information_not_bound_by_manifest" in got
    assert "dataset_artifact_not_bound_by_manifest" in got
    assert "development_roster_not_bound_by_manifest" in got


def test_the_cli_prints_the_record_and_never_posts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    write_real_root(root)
    code = cli.main(
        ["seal-record", "--output-root", str(root), "--sealed-code-sha", SEALED]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert out == seal_record.render_seal_record(root, SEALED)
    assert out.startswith("## FS1 seal record (Issue #810)")
    bad = cli.main(
        ["seal-record", "--output-root", str(root), "--sealed-code-sha", "zz"]
    )
    assert bad == 2
    assert "sealed_code_sha_malformed" in capsys.readouterr().err


def test_the_cli_import_closure_has_no_network_client_without_allow_network(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    write_real_root(root)
    program = (
        "import sys\n"
        "from trade_rl.evaluation.forward_acquisition.cli import main\n"
        "code = main(sys.argv[1:])\n"
        "banned = ('urllib.request', 'http.client', 'ssl', 'ftplib', 'requests', "
        "'urllib3', 'httpx', 'aiohttp')\n"
        "loaded = sorted(m for m in sys.modules if m in banned or "
        "m.startswith('trade_rl.integrations'))\n"
        "print('CLOSURE', code, loaded)\n"
    )

    def closure(*argv: str) -> str:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", program, *argv],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[2],
        )
        line = [x for x in completed.stdout.splitlines() if x.startswith("CLOSURE")]
        assert line, completed.stderr[-2000:]
        return line[-1]

    # plan only, every pre-flight refusal, and the printer never load a network client
    assert (
        closure(
            "acquire", "--authorization", AUTH, "--output-root", str(tmp_path / "new")
        )
        == "CLOSURE 0 []"
    )
    assert (
        closure(
            "acquire",
            "--authorization",
            "bad",
            "--output-root",
            str(tmp_path / "new"),
        )
        == "CLOSURE 2 []"
    )
    assert (
        closure(
            "acquire",
            "--authorization",
            AUTH,
            "--output-root",
            str(root),
            "--allow-network",
        )
        == "CLOSURE 2 []"
    )
    assert (
        closure(
            "acquire",
            "--authorization",
            AUTH,
            "--output-root",
            str(tmp_path / "new"),
            "--allow-network",
        )
        == "CLOSURE 2 []"
    )
    assert (
        closure("seal-record", "--output-root", str(root), "--sealed-code-sha", SEALED)
        == "CLOSURE 0 []"
    )

    # positive control: the maintained integration does bring a network client in
    control = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys, trade_rl.integrations.binance; "
            "print('urllib.request' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert control.stdout.strip() == "True"


def test_an_authorized_run_reaches_the_network_only_after_every_preflight_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from trade_rl.evaluation.forward_acquisition import acquire

    created: list[Path] = []

    def spy(cache_root: Path) -> Any:
        created.append(cache_root)
        raise AssertionError("a pre-flight refusal must not create a transport")

    monkeypatch.setattr(acquire, "_real_transport", spy)
    taken = tmp_path / "taken"
    taken.mkdir()
    with pytest.raises(AcquisitionRefusal):
        acquire.run_acquire(
            authorization=AUTH,
            output_root=taken,
            allow_network=True,
            command_line="x",
        )
    assert created == []
    # the production wiring uses the lazily created real transport
    assert acquire._environment().network_transport is acquire._real_transport


@pytest.fixture(scope="module")
def mini_world(tmp_path_factory: pytest.TempPathFactory) -> tuple[Scenario, Path]:
    base = tmp_path_factory.mktemp("sealed")
    scenario = make_scenario(base / "world")
    root = base / "root"
    scenario.run(root)
    return scenario, root


def test_the_generator_accepts_a_real_acquisition_output(
    mini_world: tuple[Scenario, Path],
) -> None:
    scenario, root = mini_world
    registry = scenario.registry
    inputs = seal_record.SealRecordInputs(
        manifest_bytes=(root / plan.MANIFEST_NAME).read_bytes(),
        receipt_bytes=(root / plan.RECEIPT_NAME).read_bytes(),
        sealed_code_sha=CODE_SHA,
    )
    text = seal_record._render(
        inputs, registry=registry, frozen_params_digest=registry.digest()
    )
    manifest = seal.parse_manifest(inputs.manifest_bytes)
    assert manifest.dataset_id in text
    assert registry.digest() in text
    assert "2026-04-02T00:00Z" in text  # miniature scoring end plus the margin
    assert datetime.fromisoformat(manifest.fetched_utc) >= datetime(
        2026, 4, 2, tzinfo=UTC
    )
    with pytest.raises(AcquisitionRefusal) as caught:
        seal_record._render(inputs, registry=registry, frozen_params_digest="0" * 64)
    assert "registry_digest_differs_from_frozen" in caught.value.reasons
