from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest

from tools import ppo_checkpoint_actions as transport


def _archive(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return path


def _receipt(**updates: object) -> dict[str, Any]:
    identity = {
        "repository": "owner/repo",
        "repository_id": 12,
        "code_sha": "a" * 40,
        "workflow": "PPO Feature Checkpoint",
        "workflow_ref": "owner/repo/.github/workflows/ppo-feature-checkpoint.yml@refs/heads/main",
        "workflow_sha": "b" * 40,
        "github_sha": "b" * 40,
    }
    receipt = transport._new_receipt(
        {"GITHUB_RUN_ID": "34", "GITHUB_RUN_ATTEMPT": "2"},
        "arm",
        identity,
        transport._runtime_provenance(),
    )
    receipt.update(
        {
            "status": "failed",
            "checkpoint_protocol_digest": "c" * 64,
            "approved_protocol_digest": "c" * 64,
            "review_reference": "https://github.com/owner/repo/pull/738#review-1",
            "finished_at": "2026-09-21T00:00:00+00:00",
        }
    )
    receipt.update(updates)
    return receipt


def test_parse_artifact_references_rejects_bool_aliases_and_extra_fields() -> None:
    valid = json.dumps([{"id": 1, "run_id": 2, "sha256": "a" * 64}])
    assert transport.parse_artifact_references(valid) == (
        transport.ArtifactReference(1, 2, "a" * 64),
    )

    for payload in (
        '[{"id":true,"run_id":2,"sha256":"' + "a" * 64 + '"}]',
        json.dumps([{"id": 1, "run_id": 2, "sha256": "A" * 64}]),
        json.dumps([{"id": 1, "run_id": 2, "sha256": "a" * 64, "name": "extra"}]),
        json.dumps(
            [
                {"id": 1, "run_id": 2, "sha256": "a" * 64},
                {"id": 1, "run_id": 2, "sha256": "a" * 64},
            ]
        ),
    ):
        with pytest.raises(ValueError):
            transport.parse_artifact_references(payload)


def test_archive_digest_mismatch_fails_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path / "input.zip", {"checkpoint/file": b"safe"})
    destination = tmp_path / "extracted"
    called = False

    def forbidden_extract(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(transport, "safe_extract_zip", forbidden_extract)
    with pytest.raises(ValueError, match="archive SHA-256"):
        transport.extract_verified_archive(archive, destination, "0" * 64)

    assert not called
    assert not destination.exists()


@pytest.mark.parametrize(
    "member",
    [
        "../escape.txt",
        "/absolute.txt",
        "C:/drive.txt",
        r"folder\backslash.txt",
        "checkpoint//empty-component.txt",
    ],
)
def test_safe_extractor_rejects_unsafe_zip_member_paths(
    tmp_path: Path, member: str
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    if "\\" in member:
        archive_path = _archive(archive_path, {member.replace("\\", "X"): b"bad"})
        archive_path.write_bytes(archive_path.read_bytes().replace(b"X", b"\\"))
    else:
        archive_path = _archive(archive_path, {member: b"bad"})
    destination = tmp_path / "unpacked"

    with pytest.raises(ValueError, match="ZIP member path"):
        transport.safe_extract_zip(archive_path, destination)

    assert not destination.exists()


def test_safe_extractor_rejects_symlinks_duplicates_and_file_directory_collisions(
    tmp_path: Path,
) -> None:
    symlink_zip = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink_zip, "w") as archive:
        member = zipfile.ZipInfo("checkpoint/link")
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, b"target")
    with pytest.raises(ValueError, match="symlink"):
        transport.safe_extract_zip(symlink_zip, tmp_path / "symlink-out")

    duplicate_zip = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate_zip, "w") as archive:
        archive.writestr("checkpoint/file", b"one")
        archive.writestr("checkpoint/file", b"two")
    with pytest.raises(ValueError, match="duplicate"):
        transport.safe_extract_zip(duplicate_zip, tmp_path / "duplicate-out")

    collision = _archive(
        tmp_path / "collision.zip",
        {"checkpoint/path": b"file", "checkpoint/path/child": b"child"},
    )
    with pytest.raises(ValueError, match="file/directory"):
        transport.safe_extract_zip(collision, tmp_path / "collision-out")


def test_checkpoint_tree_merge_accepts_equal_duplicates_and_rejects_collisions(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    target = tmp_path / "merged"
    for root in (left, right):
        (root / "fits" / "baseline" / "ppo0").mkdir(parents=True)
        (root / "fits" / "baseline" / "ppo0" / "fit.json").write_bytes(b"same")
    (right / "attempts").mkdir()
    (right / "attempts" / "failed-fit").write_bytes(b"preserved")

    transport.merge_checkpoint_trees(left, target)
    transport.merge_checkpoint_trees(right, target)

    assert (target / "fits" / "baseline" / "ppo0" / "fit.json").read_bytes() == b"same"
    assert (target / "attempts" / "failed-fit").read_bytes() == b"preserved"

    conflict = tmp_path / "conflict"
    (conflict / "fits" / "baseline" / "ppo0").mkdir(parents=True)
    (conflict / "fits" / "baseline" / "ppo0" / "fit.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint path collision"):
        transport.merge_checkpoint_trees(conflict, target)


def test_source_root_locator_requires_one_valid_dataset_study_pair_and_keeps_extras(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "source-zip"
    (extracted / "dataset").mkdir(parents=True)
    (extracted / "study").mkdir()
    (extracted / "dataset" / "manifest.json").write_bytes(b"manifest")
    (extracted / "dataset" / "arrays.npz").write_bytes(b"arrays")
    (extracted / "study" / "plan.json").write_bytes(b"plan")
    (extracted / "successor-index.json").write_bytes(b"authority index")

    assert transport.find_source_root(extracted) == extracted
    assert (transport.find_source_root(extracted) / "successor-index.json").is_file()

    duplicate = extracted / "nested"
    (duplicate / "dataset").mkdir(parents=True)
    (duplicate / "study").mkdir()
    (duplicate / "dataset" / "manifest.json").write_bytes(b"m")
    (duplicate / "dataset" / "arrays.npz").write_bytes(b"a")
    (duplicate / "study" / "plan.json").write_bytes(b"p")
    with pytest.raises(ValueError, match="exactly one"):
        transport.find_source_root(extracted)


def test_operator_approval_is_required_and_compares_outer_protocol_digest() -> None:
    with pytest.raises(ValueError, match="approved protocol digest"):
        transport.validate_operator_approval("arm", "", "review", "c" * 64)
    with pytest.raises(ValueError, match="review reference"):
        transport.validate_operator_approval("finalize", "c" * 64, "", "c" * 64)
    with pytest.raises(ValueError, match="does not match"):
        transport.validate_operator_approval("arm", "d" * 64, "review", "c" * 64)
    assert (
        transport.validate_operator_approval("arm", "c" * 64, "review ref", "c" * 64)
        == "review ref"
    )
    assert transport.validate_operator_approval("prepare", "", "", None) is None


def test_checkpoint_commands_are_one_prepare_or_one_complete_arm() -> None:
    protocol: dict[str, Any] = {
        "core_protocol": {
            "seeds": [0, 1, 2, 3, 4],
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "factors": {"baseline": {}, "btc_relative": {}},
        },
        "execution": {
            "scenario_names_by_factor": {
                "baseline": ["base"],
                "btc_relative": ["base", "cost_2x", "latency_1"],
            }
        },
    }
    source = Path("source")
    checkpoint = Path("checkpoint")
    prepare = transport.commands_for_stage(
        "prepare", source, checkpoint, protocol, factor=None, seed=None
    )
    assert len(prepare) == 1
    assert "prepare" in prepare[0]
    assert all(
        "fit" not in command and "replay-cell" not in command for command in prepare
    )

    arm = transport.commands_for_stage(
        "arm", source, checkpoint, protocol, factor="btc_relative", seed=3
    )
    assert [command[3] for command in arm] == [
        "fit",
        "replay-cell",
        "replay-cell",
        "replay-cell",
        "replay-cell",
        "replay-cell",
        "replay-cell",
        "assemble-arm",
    ]
    assert all("--seed" in command and "3" in command for command in arm)


def test_prior_receipt_binds_checkout_and_workflow_identity() -> None:
    reference = transport.ArtifactReference(90, 34, "d" * 64)
    metadata = {
        "id": 90,
        "expired": False,
        "name": "ppo-feature-checkpoint-34-2",
        "digest": f"sha256:{'d' * 64}",
        "workflow_run": {
            "id": 34,
            "repository_id": 12,
            "head_repository_id": 12,
            "head_sha": "b" * 40,
        },
    }
    identity = {
        "repository": "owner/repo",
        "repository_id": 12,
        "code_sha": "a" * 40,
        "workflow": "PPO Feature Checkpoint",
        "workflow_ref": "owner/repo/.github/workflows/ppo-feature-checkpoint.yml@refs/heads/main",
        "workflow_sha": "b" * 40,
        "github_sha": "b" * 40,
    }
    transport.validate_artifact_metadata(metadata, reference, repository_id=12)
    receipt = _receipt()
    transport.validate_prior_receipt(
        receipt, reference, metadata, identity, protocol_digest="c" * 64
    )

    changed_workflow = dict(receipt, workflow_sha="e" * 40, github_sha="e" * 40)
    with pytest.raises(ValueError, match="workflow identity"):
        transport.validate_prior_receipt(
            changed_workflow,
            reference,
            metadata,
            identity,
            protocol_digest="c" * 64,
        )


def test_artifact_api_digest_is_required() -> None:
    reference = transport.ArtifactReference(90, 34, "d" * 64)
    metadata = {
        "id": 90,
        "expired": False,
        "name": "ppo-feature-checkpoint-34-2",
        "digest": None,
        "workflow_run": {
            "id": 34,
            "repository_id": 12,
            "head_repository_id": 12,
            "head_sha": "b" * 40,
        },
    }
    with pytest.raises(ValueError, match="API digest"):
        transport.validate_artifact_metadata(metadata, reference, repository_id=12)


def test_producer_run_api_binds_workflow_attempt_repo_and_head() -> None:
    identity = {
        "repository": "owner/repo",
        "repository_id": 12,
        "workflow": "PPO Feature Checkpoint",
    }
    reference = transport.ArtifactReference(90, 34, "d" * 64)
    receipt = _receipt()
    run = {
        "id": 34,
        "run_attempt": 2,
        "path": transport.WORKFLOW_PATH,
        "name": "PPO Feature Checkpoint",
        "head_sha": "b" * 40,
        "repository": {"id": 12},
        "head_repository": {"id": 12},
    }
    transport.validate_producer_run(run, reference, identity, receipt)

    changed = dict(run, path=".github/workflows/ci.yml")
    with pytest.raises(ValueError, match="workflow identity"):
        transport.validate_producer_run(changed, reference, identity, receipt)


def test_producer_api_lookup_targets_the_receipt_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt()
    identity = {
        "repository": "owner/repo",
        "repository_id": 12,
        "workflow": "PPO Feature Checkpoint",
    }
    reference = transport.ArtifactReference(90, 34, "d" * 64)
    run = {
        "id": 34,
        "run_attempt": 2,
        "path": transport.WORKFLOW_PATH,
        "name": "PPO Feature Checkpoint",
        "head_sha": "b" * 40,
        "repository": {"id": 12},
        "head_repository": {"id": 12},
    }
    requested: list[str] = []

    def api_json(url: str, **_kwargs: object) -> dict[str, Any]:
        requested.append(url)
        return run

    monkeypatch.setattr(transport, "_api_json", api_json)
    transport._validate_run_api_record(
        "owner/repo",
        reference,
        receipt,
        identity,
        token="secret",
        deadline=999999999.0,
    )
    assert requested == [
        "https://api.github.com/repos/owner/repo/actions/runs/34/attempts/2"
    ]


def test_prepare_ignores_ui_default_factor_and_runs_no_economics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    commands: list[list[str]] = []

    def fake_run_stage(stage_commands: list[list[str]], **kwargs: Any) -> None:
        commands.extend(stage_commands)
        assert kwargs["on_success"] is not None
        kwargs["on_success"](1, stage_commands[0])

    monkeypatch.setattr(transport, "available_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(transport, "_check_checkout", lambda _sha: None)
    monkeypatch.setattr(transport, "_api_json", lambda *_args, **_kwargs: {"id": 12})
    monkeypatch.setattr(transport, "_download_source", lambda **_kwargs: source)
    monkeypatch.setattr(transport, "run_stage_commands", fake_run_stage)
    monkeypatch.setattr(
        transport,
        "_checkpoint_module",
        lambda: SimpleNamespace(
            validate_checkpoint_protocol=lambda *_args: {
                "core_protocol": {"seeds": [0, 1, 2, 3, 4]},
                "execution": {},
            }
        ),
    )
    output = tmp_path / "output"
    environment = {
        "CHECKPOINT_OUTPUT": str(output),
        "CHECKPOINT_STAGE": "prepare",
        "CHECKPOINT_CODE_SHA": "a" * 40,
        "CHECKPOINT_ARTIFACTS_JSON": "[]",
        "CHECKPOINT_FACTOR": "baseline",
        "CHECKPOINT_GITHUB_TOKEN": "secret-token",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_RUN_ID": "34",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_WORKFLOW": "PPO Feature Checkpoint",
        "GITHUB_WORKFLOW_REF": "owner/repo/.github/workflows/ppo-feature-checkpoint.yml@refs/heads/main",
        "GITHUB_WORKFLOW_SHA": "b" * 40,
        "GITHUB_SHA": "b" * 40,
    }

    assert transport.execute_from_environment(environment) == 0
    assert [command[3] for command in commands] == ["prepare"]
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["status"] == "succeeded"
    assert receipt["checkpoint_protocol_digest"]


def test_failed_arm_publishes_failed_receipt_with_protocol_and_completed_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    protocol: dict[str, Any] = {
        "core_protocol": {
            "seeds": [0, 1, 2, 3, 4],
            "symbols": ["BTCUSDT"],
            "factors": {"baseline": {"enabled": True}},
        },
        "execution": {"scenario_names_by_factor": {"baseline": ["base"]}},
    }
    digest = transport.content_digest(protocol)
    input_reference = transport.ArtifactReference(90, 34, "d" * 64)

    def fake_run_stage(commands: list[list[str]], **kwargs: Any) -> None:
        assert commands[0][3] == "fit"
        completed_fit = checkpoint / "fits" / "baseline" / "ppo0" / "fit.json"
        completed_fit.parent.mkdir(parents=True)
        completed_fit.write_bytes(b"validated completed fit evidence")
        kwargs["on_success"](1, commands[0])
        failing_child = (
            "import sys; "
            "sys.stderr.write('PREFIX' * 2000 + "
            "'secret-token DISTINCTIVE_FAILURE_MARKER'); "
            "sys.exit(23)"
        )
        transport.run_process_group(
            [sys.executable, "-c", failing_child],
            checkpoint_root=checkpoint,
            deadline=kwargs["deadline"],
            log_directory=kwargs["log_directory"],
            command_index=2,
            memory_reader=lambda: 8 * 1024**3,
            poll_seconds=0.03,
        )

    monkeypatch.setattr(transport, "available_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(transport, "_check_checkout", lambda _sha: None)
    monkeypatch.setattr(transport, "_api_json", lambda *_args, **_kwargs: {"id": 12})
    monkeypatch.setattr(transport, "_download_source", lambda **_kwargs: source)
    monkeypatch.setattr(
        transport,
        "_restore_checkpoint_inputs",
        lambda *_args, **_kwargs: (checkpoint, protocol, digest),
    )
    monkeypatch.setattr(transport, "run_stage_commands", fake_run_stage)
    monkeypatch.setattr(
        transport,
        "_checkpoint_module",
        lambda: SimpleNamespace(validate_checkpoint_protocol=lambda *_args: protocol),
    )
    output = tmp_path / "output"
    environment = {
        "CHECKPOINT_OUTPUT": str(output),
        "CHECKPOINT_STAGE": "arm",
        "CHECKPOINT_CODE_SHA": "a" * 40,
        "CHECKPOINT_ARTIFACTS_JSON": json.dumps([input_reference.to_mapping()]),
        "CHECKPOINT_APPROVED_PROTOCOL_DIGEST": digest,
        "CHECKPOINT_REVIEW_REFERENCE": "reviewed protocol 9df2",
        "CHECKPOINT_FACTOR": "baseline",
        "CHECKPOINT_SEED": "0",
        "CHECKPOINT_GITHUB_TOKEN": "secret-token",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_RUN_ID": "35",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_WORKFLOW": "PPO Feature Checkpoint",
        "GITHUB_WORKFLOW_REF": "owner/repo/.github/workflows/ppo-feature-checkpoint.yml@refs/heads/main",
        "GITHUB_WORKFLOW_SHA": "b" * 40,
        "GITHUB_SHA": "b" * 40,
    }

    with pytest.raises(
        transport.TransportError, match="DISTINCTIVE_FAILURE_MARKER"
    ) as error:
        transport.execute_from_environment(environment)
    assert len(str(error.value).encode("utf-8")) <= 500

    assert (checkpoint / "fits/baseline/ppo0/fit.json").read_bytes() == (
        b"validated completed fit evidence"
    )
    child_diagnostic = transport._read_child_diagnostic(
        output / "logs" / "command-0002.log"
    )
    assert len(child_diagnostic.encode("utf-8")) <= (
        transport.MAX_CHILD_DIAGNOSTIC_BYTES
    )
    assert "DISTINCTIVE_FAILURE_MARKER" in child_diagnostic
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["status"] == "failed"
    assert receipt["checkpoint_protocol_digest"] == digest
    assert receipt["approved_protocol_digest"] == digest
    assert receipt["review_reference"] == "reviewed protocol 9df2"
    assert receipt["completed_commands"] == [
        {"index": 1, "command": "fit", "exit_status": 0}
    ]
    assert "DISTINCTIVE_FAILURE_MARKER" in receipt["error_message"]
    assert "secret-token" not in receipt["error_message"]
    assert len(receipt["error_message"].encode("utf-8")) <= 500
    metadata = {
        "id": 91,
        "expired": False,
        "name": "ppo-feature-checkpoint-35-1",
        "digest": f"sha256:{'e' * 64}",
        "workflow_run": {
            "id": 35,
            "repository_id": 12,
            "head_repository_id": 12,
            "head_sha": "b" * 40,
        },
    }
    transport.validate_artifact_metadata(
        metadata,
        transport.ArtifactReference(91, 35, "e" * 64),
        repository_id=12,
    )
    transport.validate_prior_receipt(
        receipt,
        transport.ArtifactReference(91, 35, "e" * 64),
        metadata,
        {
            "repository": "owner/repo",
            "repository_id": 12,
            "code_sha": "a" * 40,
            "workflow": "PPO Feature Checkpoint",
            "workflow_ref": "owner/repo/.github/workflows/ppo-feature-checkpoint.yml@refs/heads/main",
            "workflow_sha": "b" * 40,
            "github_sha": "b" * 40,
        },
        protocol_digest=digest,
        approved_protocol_digest=digest,
        review_reference="reviewed protocol 9df2",
    )


def test_transport_roundtrips_prepare_failed_arm_resume_and_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_buffer = io.BytesIO()
    with zipfile.ZipFile(
        source_buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("dataset/manifest.json", b"synthetic manifest")
        archive.writestr("dataset/arrays.npz", b"synthetic arrays")
        archive.writestr("study/plan.json", b"synthetic plan")
        archive.writestr("successor-index.json", b"authority index")
    source_archive = source_buffer.getvalue()
    source_sha256 = hashlib.sha256(source_archive).hexdigest()
    source_identity = {
        "artifact_id": 700,
        "run_id": 70,
        "name": "synthetic-frozen-source",
        "sha256": source_sha256,
    }
    monkeypatch.setattr(transport, "SOURCE_ARTIFACT_ID", source_identity["artifact_id"])
    monkeypatch.setattr(transport, "SOURCE_ARTIFACT_RUN_ID", source_identity["run_id"])
    monkeypatch.setattr(transport, "SOURCE_ARTIFACT_NAME", source_identity["name"])
    monkeypatch.setattr(transport, "SOURCE_ARTIFACT_SHA256", source_sha256)

    source_workflow_run = {
        "id": source_identity["run_id"],
        "repository_id": 12,
        "head_repository_id": 12,
        "head_sha": "c" * 40,
    }
    artifact_store: dict[int, dict[str, Any]] = {
        source_identity["artifact_id"]: {
            "raw": source_archive,
            "metadata": {
                "id": source_identity["artifact_id"],
                "expired": False,
                "name": source_identity["name"],
                "digest": f"sha256:{source_sha256}",
                "workflow_run": source_workflow_run,
            },
        }
    }
    producer_runs: dict[tuple[int, int], dict[str, Any]] = {}
    api_paths: list[str] = []

    protocol: dict[str, Any] = {
        "core_protocol": {
            "seeds": [0, 1, 2, 3, 4],
            "symbols": ["BTCUSDT"],
            "factors": {"baseline": {"enabled": True}, "btc_relative": {}},
        },
        "execution": {
            "scenario_names_by_factor": {
                "baseline": ["base"],
                "btc_relative": ["base", "cost_2x", "latency_1"],
            }
        },
    }
    protocol_digest = transport.content_digest(protocol)
    current_stage = "prepare"
    first_arm_failure_injected = False
    fit_invocations = 0
    resumed_fit_reused = False

    def fake_api_json(url: str, *, token: str, deadline: float) -> dict[str, Any]:
        del token, deadline
        path = urlsplit(url).path
        api_paths.append(path)
        if path == "/repos/owner/repo":
            return {"id": 12}
        artifact_marker = "/actions/artifacts/"
        if artifact_marker in path:
            artifact_id = int(path.rsplit(artifact_marker, 1)[1])
            return artifact_store[artifact_id]["metadata"]
        run_marker = "/actions/runs/"
        if run_marker in path and "/attempts/" in path:
            run_and_attempt = path.rsplit(run_marker, 1)[1].split("/attempts/")
            run_key = (int(run_and_attempt[0]), int(run_and_attempt[1]))
            return producer_runs[run_key]
        raise AssertionError(f"unexpected fake GitHub API path: {path}")

    def fake_download_response(url: str, *, token: str, deadline: float) -> io.BytesIO:
        del token, deadline
        path_parts = urlsplit(url).path.split("/")
        artifact_id = int(path_parts[-2])
        return io.BytesIO(artifact_store[artifact_id]["raw"])

    def register_output_artifact(
        output: Path,
        *,
        artifact_id: int,
        conclusion: str,
    ) -> transport.ArtifactReference:
        receipt = json.loads((output / "receipt.json").read_bytes())
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(
            archive_buffer, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            checkpoint = output / "checkpoint"
            for path in sorted(checkpoint.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(checkpoint).as_posix()
                    archive.writestr(f"checkpoint/{relative}", path.read_bytes())
            archive.writestr("receipt.json", (output / "receipt.json").read_bytes())
        raw = archive_buffer.getvalue()
        archive_sha256 = hashlib.sha256(raw).hexdigest()
        run_id = receipt["run_id"]
        attempt = receipt["run_attempt"]
        artifact_store[artifact_id] = {
            "raw": raw,
            "metadata": {
                "id": artifact_id,
                "expired": False,
                "name": receipt["artifact_name"],
                "digest": f"sha256:{archive_sha256}",
                "workflow_run": {
                    "id": run_id,
                    "repository_id": 12,
                    "head_repository_id": 12,
                    "head_sha": receipt["github_sha"],
                },
            },
        }
        producer_runs[(run_id, attempt)] = {
            "id": run_id,
            "run_attempt": attempt,
            "path": transport.WORKFLOW_PATH,
            "name": receipt["workflow"],
            "head_sha": receipt["github_sha"],
            "repository": {"id": 12},
            "head_repository": {"id": 12},
            "status": "completed",
            "conclusion": conclusion,
        }
        return transport.ArtifactReference(artifact_id, run_id, archive_sha256)

    def fake_run_stage(commands: list[list[str]], **kwargs: Any) -> None:
        nonlocal first_arm_failure_injected, fit_invocations, resumed_fit_reused
        checkpoint = kwargs["checkpoint_root"]
        on_success = kwargs["on_success"]
        for index, command in enumerate(commands, start=1):
            name = command[3]
            if name == "prepare":
                checkpoint.mkdir()
                (checkpoint / "protocol.snapshot.json").write_bytes(
                    transport.canonical_json_bytes(protocol)
                )
            elif name == "fit":
                fit_path = checkpoint / "fits" / "baseline" / "ppo0" / "fit.json"
                if fit_path.is_file():
                    resumed_fit_reused = True
                else:
                    fit_invocations += 1
                    fit_path.parent.mkdir(parents=True)
                    fit_path.write_bytes(b"synthetic completed fit")
            elif name == "replay-cell":
                scenario = command[command.index("--scenario") + 1]
                (checkpoint / "cells/baseline/ppo0" / scenario / "symbol-0").mkdir(
                    parents=True, exist_ok=True
                )
                (
                    checkpoint
                    / "cells/baseline/ppo0"
                    / scenario
                    / "symbol-0"
                    / "cell.json"
                ).write_bytes(b"synthetic completed cell")
            elif name == "assemble-arm":
                arm = checkpoint / "arms" / "baseline" / "ppo0"
                arm.mkdir(parents=True, exist_ok=True)
                (arm / "result.json").write_bytes(b"synthetic assembled arm")
            elif name == "finalize":
                comparison = checkpoint / "comparison"
                comparison.mkdir()
                (comparison / "comparison.json").write_bytes(
                    b"synthetic final comparison"
                )
            else:
                raise AssertionError(f"unexpected child command: {name}")
            on_success(index, command)
            if (
                current_stage == "arm"
                and not first_arm_failure_injected
                and name == "fit"
            ):
                first_arm_failure_injected = True
                raise transport.TransportError("injected interruption after fit")

    monkeypatch.setattr(transport, "available_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(transport, "_check_checkout", lambda _sha: None)
    monkeypatch.setattr(transport, "_api_json", fake_api_json)
    monkeypatch.setattr(transport, "_download_response", fake_download_response)
    monkeypatch.setattr(
        transport,
        "_checkpoint_module",
        lambda: SimpleNamespace(
            validate_checkpoint_protocol=lambda source, root: (
                protocol
                if source.is_dir() and root.is_dir()
                else (_ for _ in ()).throw(AssertionError("source/checkpoint missing"))
            )
        ),
    )
    monkeypatch.setattr(transport, "run_stage_commands", fake_run_stage)

    def environment(
        stage: str,
        run_id: int,
        output: Path,
        references: tuple[transport.ArtifactReference, ...],
    ) -> dict[str, str]:
        return {
            "CHECKPOINT_OUTPUT": str(output),
            "CHECKPOINT_STAGE": stage,
            "CHECKPOINT_CODE_SHA": "a" * 40,
            "CHECKPOINT_ARTIFACTS_JSON": json.dumps(
                [reference.to_mapping() for reference in references]
            ),
            "CHECKPOINT_APPROVED_PROTOCOL_DIGEST": (
                "" if stage == "prepare" else protocol_digest
            ),
            "CHECKPOINT_REVIEW_REFERENCE": (
                "" if stage == "prepare" else "reviewed synthetic protocol"
            ),
            "CHECKPOINT_FACTOR": "baseline",
            "CHECKPOINT_SEED": "0" if stage == "arm" else "",
            "CHECKPOINT_GITHUB_TOKEN": "synthetic-secret",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_RUN_ID": str(run_id),
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_WORKFLOW": "PPO Feature Checkpoint",
            "GITHUB_WORKFLOW_REF": "owner/repo/.github/workflows/ppo-feature-checkpoint.yml@refs/heads/main",
            "GITHUB_WORKFLOW_SHA": "b" * 40,
            "GITHUB_SHA": "b" * 40,
        }

    current_stage = "prepare"
    prepare_output = tmp_path / "prepare"
    assert (
        transport.execute_from_environment(
            environment("prepare", 101, prepare_output, ())
        )
        == 0
    )
    prepare_reference = register_output_artifact(
        prepare_output, artifact_id=801, conclusion="success"
    )

    current_stage = "arm"
    failed_arm_output = tmp_path / "arm-failed"
    with pytest.raises(transport.TransportError, match="injected interruption"):
        transport.execute_from_environment(
            environment("arm", 102, failed_arm_output, (prepare_reference,))
        )
    failed_arm_reference = register_output_artifact(
        failed_arm_output, artifact_id=802, conclusion="failure"
    )
    failed_receipt = json.loads((failed_arm_output / "receipt.json").read_bytes())
    assert failed_receipt["checkpoint_protocol_digest"] == protocol_digest
    assert (failed_arm_output / "checkpoint/fits/baseline/ppo0/fit.json").is_file()

    resumed_arm_output = tmp_path / "arm-resumed"
    assert (
        transport.execute_from_environment(
            environment(
                "arm",
                103,
                resumed_arm_output,
                (prepare_reference, failed_arm_reference),
            )
        )
        == 0
    )
    assert fit_invocations == 1
    assert resumed_fit_reused
    resumed_reference = register_output_artifact(
        resumed_arm_output, artifact_id=803, conclusion="success"
    )

    current_stage = "finalize"
    final_output = tmp_path / "finalize"
    assert (
        transport.execute_from_environment(
            environment(
                "finalize",
                104,
                final_output,
                (prepare_reference, failed_arm_reference, resumed_reference),
            )
        )
        == 0
    )
    final_receipt = json.loads((final_output / "receipt.json").read_bytes())
    assert final_receipt["status"] == "succeeded"
    assert final_receipt["checkpoint_protocol_digest"] == protocol_digest
    assert (final_output / "checkpoint/comparison/comparison.json").is_file()
    assert any(path.endswith("/actions/runs/102/attempts/1") for path in api_paths)
    assert any(path.endswith("/actions/runs/103/attempts/1") for path in api_paths)


def test_main_redacts_token_from_displayed_child_diagnostic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CHECKPOINT_GITHUB_TOKEN", "secret-token")

    def fail() -> int:
        raise transport.TransportError(
            "checkpoint command failed; child output tail: secret-token"
        )

    monkeypatch.setattr(transport, "execute_from_environment", fail)
    with pytest.raises(SystemExit, match="1"):
        transport.main()
    stderr = capsys.readouterr().err
    assert "[redacted]" in stderr
    assert "secret-token" not in stderr


@pytest.mark.skipif(os.name != "posix", reason="process-group contract is Linux-only")
@pytest.mark.parametrize("parent_exit_code", [0, 7])
def test_process_group_cleanup_kills_writer_after_leader_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent_exit_code: int
) -> None:
    pid_path = tmp_path / "writer.pid"
    marker_path = tmp_path / "writes.txt"
    child_code = (
        "import os,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "stream=open(os.environ['WRITER_MARKER'],'a',encoding='ascii'); "
        "exec(\"while True:\\n stream.write('x\\\\n'); stream.flush(); time.sleep(0.03)\")"
    )
    parent_code = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[1]]); "
        "open(sys.argv[2],'w',encoding='ascii').write(str(child.pid)); "
        "time.sleep(0.2); sys.exit(int(sys.argv[3]))"
    )
    monkeypatch.setenv("WRITER_MARKER", str(marker_path))
    command = [
        sys.executable,
        "-c",
        parent_code,
        child_code,
        str(pid_path),
        str(parent_exit_code),
    ]

    def run() -> None:
        transport.run_process_group(
            command,
            checkpoint_root=tmp_path,
            deadline=time.monotonic() + 30,
            log_directory=tmp_path / "logs",
            command_index=1,
            memory_reader=lambda: 8 * 1024**3,
            poll_seconds=0.03,
        )

    if parent_exit_code:
        with pytest.raises(transport.TransportError, match="exited with status 7"):
            run()
    else:
        run()

    assert pid_path.is_file()
    before = marker_path.read_bytes()
    time.sleep(0.15)
    assert marker_path.read_bytes() == before


def test_failed_command_propagates_without_deleting_completed_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    cell = checkpoint / "completed-cell.json"
    cell.write_bytes(b"complete evidence")
    calls: list[list[str]] = []

    def fail_second(command: list[str], **_kwargs: object) -> None:
        calls.append(command)
        if len(calls) == 2:
            raise RuntimeError("child command failed")

    commands = [["first"], ["failed"], ["unreached"]]
    with pytest.raises(RuntimeError, match="child command failed"):
        transport.run_stage_commands(
            commands,
            checkpoint_root=checkpoint,
            deadline=999999999.0,
            command_runner=fail_second,
        )

    assert calls == commands[:2]
    assert cell.read_bytes() == b"complete evidence"


def _review_body(
    *,
    code_sha: str = "a" * 40,
    workflow_sha: str = "b" * 40,
    protocol_digest: str = "c" * 64,
    artifact_id: int = 90,
    run_id: int = 34,
    artifact_sha256: str = "d" * 64,
) -> str:
    payload = {
        "schema": "ppo_feature_review_evidence_v1",
        "outcome": "G0_G1_CLEAR_G2_EVIDENCE_BOUND",
        "code_sha": code_sha,
        "workflow_sha": workflow_sha,
        "protocol_digest": protocol_digest,
        "prepare_artifact": {
            "id": artifact_id,
            "run_id": run_id,
            "sha256": artifact_sha256,
        },
        "result_blind": True,
        "economic_execution_authorized": True,
        "unused_data_accessed": False,
        "final_data_accessed": False,
        "source_review_reference": "https://github.com/owner/repo/pull/744#issuecomment-5752780311",
    }
    return (
        "<!-- ppo-feature-review-evidence-v1 -->\n"
        + transport.canonical_json_bytes(payload).decode("utf-8")
        + "\n"
    )


def test_review_reference_must_be_exact_same_repo_pr_comment_url() -> None:
    assert transport.parse_review_reference(
        "https://github.com/owner/repo/pull/744#issuecomment-5752780311",
        repository="owner/repo",
    ) == (744, 5752780311)

    for value in (
        "reviewed",
        "https://github.com/other/repo/pull/744#issuecomment-5752780311",
        "https://github.com/owner/repo/issues/744#issuecomment-5752780311",
        "https://github.com/owner/repo/pull/744#pullrequestreview-5261754458",
    ):
        with pytest.raises(ValueError, match="review reference"):
            transport.parse_review_reference(value, repository="owner/repo")


def test_review_evidence_binds_comment_body_code_protocol_and_prepare_artifact() -> None:
    reference = transport.ArtifactReference(90, 34, "d" * 64)
    body = _review_body()
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    comment = {
        "html_url": "https://github.com/owner/repo/pull/744#issuecomment-5752780311",
        "issue_url": "https://api.github.com/repos/owner/repo/issues/744",
        "body": body,
    }
    pull = {
        "head": {
            "sha": "a" * 40,
            "repo": {"id": 12},
        }
    }

    evidence = transport.validate_review_evidence_comment(
        comment,
        pull,
        review_reference=comment["html_url"],
        review_record_sha256=digest,
        repository="owner/repo",
        repository_id=12,
        code_sha="a" * 40,
        workflow_sha="b" * 40,
        protocol_digest="c" * 64,
        prepare_reference=reference,
    )

    assert evidence["prepare_artifact"] == reference.to_mapping()
    assert evidence["result_blind"] is True
    assert evidence["economic_execution_authorized"] is True


@pytest.mark.parametrize(
    "mutation",
    ("body_digest", "code_sha", "protocol", "artifact", "authorization", "result_blind"),
)
def test_review_evidence_rejects_unbound_or_non_authorizing_record(
    mutation: str,
) -> None:
    reference = transport.ArtifactReference(90, 34, "d" * 64)
    body = _review_body()
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    comment = {
        "html_url": "https://github.com/owner/repo/pull/744#issuecomment-5752780311",
        "issue_url": "https://api.github.com/repos/owner/repo/issues/744",
        "body": body,
    }
    pull = {"head": {"sha": "a" * 40, "repo": {"id": 12}}}
    kwargs = {
        "review_reference": comment["html_url"],
        "review_record_sha256": digest,
        "repository": "owner/repo",
        "repository_id": 12,
        "code_sha": "a" * 40,
        "workflow_sha": "b" * 40,
        "protocol_digest": "c" * 64,
        "prepare_reference": reference,
    }
    if mutation == "body_digest":
        kwargs["review_record_sha256"] = "e" * 64
    elif mutation == "code_sha":
        kwargs["code_sha"] = "f" * 40
    elif mutation == "protocol":
        kwargs["protocol_digest"] = "e" * 64
    elif mutation == "artifact":
        kwargs["prepare_reference"] = transport.ArtifactReference(91, 34, "d" * 64)
    else:
        raw = json.loads(body.split("\n", 1)[1])
        if mutation == "authorization":
            raw["economic_execution_authorized"] = False
        else:
            raw["result_blind"] = False
        body = (
            "<!-- ppo-feature-review-evidence-v1 -->\n"
            + transport.canonical_json_bytes(raw).decode("utf-8")
            + "\n"
        )
        comment["body"] = body
        kwargs["review_record_sha256"] = hashlib.sha256(
            body.encode("utf-8")
        ).hexdigest()

    with pytest.raises(ValueError, match="review"):
        transport.validate_review_evidence_comment(comment, pull, **kwargs)


def test_operator_approval_rejects_free_text_review_reference() -> None:
    with pytest.raises(ValueError, match="review reference"):
        transport.validate_operator_approval(
            "arm",
            "c" * 64,
            "reviewed",
            "c" * 64,
            repository="owner/repo",
            review_record_sha256="e" * 64,
        )
