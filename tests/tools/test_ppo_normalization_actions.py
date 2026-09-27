from __future__ import annotations

import json

import pytest

from tools import ppo_normalization_actions as actions

HEAD = "a" * 40


def test_execution_request_binds_exact_result_blind_authorities() -> None:
    payload = actions.build_execution_request()

    assert "execution_code_sha" not in payload

    assert payload == {
        "schema": "ppo_normalization_execution_request_v1",
        "protocol_sha256": (
            "0013470ed5858eaa3b9391f97f4b18f772495d21c128832f74e1c50b090df304"
        ),
        "implementation_digest": (
            "1de7baa54625c106a75a97047f40e085ad7b1bddd00012aa342ba9f1233de97c"
        ),
        "implementation_seal_sha256": (
            "68fd92cfe823498625e7f44b0a2c5781748f0cfa3f454010d72ab674ac18272a"
        ),
        "fresh_reconstruction_sha256": (
            "fd3d9319550d9010102aac35cee9f46c05265bcc770f1ace41a67a3a65a69611"
        ),
        "assurance_review_sha256": (
            "bfdd2f74fd4c43372617091cb3be1b010c42347cf3cea680bee2b41555fc9d95"
        ),
        "source_artifact": {
            "run_id": 34803217815,
            "artifact_id": 10331899302,
            "sha256": (
                "89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce"
            ),
        },
        "economic_execution_authorized": True,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    actions.validate_execution_request(payload)


def test_execution_request_rejects_authority_or_claim_drift() -> None:
    payload = actions.build_execution_request()
    payload["implementation_seal_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="implementation seal"):
        actions.validate_execution_request(payload)

    payload = actions.build_execution_request()
    payload["economic_result_inspected"] = True
    with pytest.raises(ValueError, match="economic_result_inspected"):
        actions.validate_execution_request(payload)


def test_request_comment_is_canonical_and_exact_head_bound() -> None:
    request = actions.build_execution_request()
    request_sha = actions.content_digest(request)
    body = (
        actions.REVIEW_REQUEST_MARKER
        + "\n"
        + json.dumps(
            {"execution_code_sha": HEAD, "request_sha256": request_sha},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    parsed = actions.parse_execution_request_comment(body)
    assert parsed == {
        "execution_code_sha": HEAD,
        "request_sha256": request_sha,
    }

    with pytest.raises(ValueError, match="canonical"):
        actions.parse_execution_request_comment(body.replace(",", ", "))


def test_execution_pr_may_change_only_the_request_record() -> None:
    actions.validate_execution_pr_files(
        ["report/ppo-normalization-execution-request.json"]
    )
    with pytest.raises(ValueError, match="only"):
        actions.validate_execution_pr_files(
            [
                "report/ppo-normalization-execution-request.json",
                "trade_rl/evaluation/ppo_normalization_activation.json",
            ]
        )


def test_activation_payload_binds_runtime_and_all_pre_execution_evidence() -> None:
    request = actions.build_execution_request()
    provenance = {
        "schema_version": "candidate_run_provenance_v1",
        "implementation": {
            "schema_version": "candidate_run_implementation_v1",
            "files": [],
        },
        "implementation_digest": actions.IMPLEMENTATION_DIGEST,
        "runtime_environment": {
            "schema_version": "candidate_run_runtime_v1",
            "python": {"implementation": "CPython", "version": "3.12.12 test"},
            "os": {"family": "Linux", "release": "test-kernel"},
            "machine": "x86_64",
            "packages": {
                "trade-rl": "0.3.0",
                "numpy": "1.26.4",
                "gymnasium": "0.29.1",
                "lightgbm": "4.6.0",
                "stable-baselines3": "2.3.2",
                "torch": "2.4.1",
            },
        },
        "runtime_environment_digest": "c" * 64,
        "research_context_digest": None,
    }

    activation = actions.build_execution_activation(request, provenance)

    assert activation["schema"] == "ppo_normalization_execution_activation_v1"
    assert activation["protocol_sha256"] == actions.PROTOCOL_SHA256
    assert activation["implementation_digest"] == actions.IMPLEMENTATION_DIGEST
    assert (
        activation["implementation_seal_sha256"] == actions.IMPLEMENTATION_SEAL_SHA256
    )
    assert (
        activation["fresh_reconstruction_sha256"] == actions.FRESH_RECONSTRUCTION_SHA256
    )
    assert activation["assurance_review_sha256"] == actions.ASSURANCE_REVIEW_SHA256
    assert activation["provenance"] == provenance
    assert activation["economic_execution_authorized"] is True
    assert activation["economic_result_inspected"] is False
    assert activation["unused_data_accessed"] is False
    assert activation["final_test_accessed"] is False
    assert activation["production_eligible"] is False
    assert activation["live_trading_authorized"] is False

    authority = actions.activation_authority_payload(activation)
    assert authority == {
        "schema": "ppo_normalization_execution_activation_authority_v1",
        "activation_sha256": actions.content_digest(activation),
    }


def test_activation_rejects_wrong_implementation_before_tag_or_fit() -> None:
    request = actions.build_execution_request()
    provenance = {
        "implementation_digest": "d" * 64,
    }
    with pytest.raises(ValueError, match="implementation digest"):
        actions.build_execution_activation(request, provenance)


def test_verifier_authority_binds_github_artifact_metadata() -> None:
    payload = actions.build_verifier_artifact_authority(
        repository_id=1103009698,
        run_id=123,
        artifact_id=456,
        artifact_sha256="e" * 64,
        code_sha=HEAD,
        workflow_sha="b" * 40,
        activation_digest="c" * 64,
        implementation_digest=actions.IMPLEMENTATION_DIGEST,
        verification_set_sha256="d" * 64,
    )
    assert payload["schema"] == (
        "ppo_normalization_replication_verifier_artifact_authority_v1"
    )
    assert payload["artifact_api_digest"] == f"sha256:{'e' * 64}"
    assert payload["run_id"] == 123
    assert payload["artifact_id"] == 456


def test_static_activation_tag_is_repository_global_one_shot() -> None:
    assert actions.ACTIVATION_TAG == "activation/ppo-normalization-corrected-v1"


def _provenance() -> dict[str, object]:
    return {
        "schema_version": "candidate_run_provenance_v1",
        "implementation": {
            "schema_version": "candidate_run_implementation_v1",
            "files": [],
        },
        "implementation_digest": actions.IMPLEMENTATION_DIGEST,
        "runtime_environment": {
            "schema_version": "candidate_run_runtime_v1",
            "python": {"implementation": "CPython", "version": "3.12.12 test"},
            "os": {"family": "Linux", "release": "test-kernel"},
            "machine": "x86_64",
            "packages": {
                "trade-rl": "0.3.0",
                "numpy": "1.26.4",
                "gymnasium": "0.29.1",
                "lightgbm": "4.6.0",
                "stable-baselines3": "2.3.2",
                "torch": "2.4.1",
            },
        },
        "runtime_environment_digest": "c" * 64,
        "research_context_digest": None,
    }


def test_activation_authority_transition_requires_committed_null_state(
    tmp_path,
) -> None:
    target = tmp_path / "target"
    authority = target / "trade_rl/evaluation/ppo_normalization_activation.json"
    authority.parent.mkdir(parents=True)
    authority.write_bytes(
        actions.canonical_json_bytes(
            {
                "schema": "ppo_normalization_execution_activation_authority_v1",
                "activation_sha256": None,
            }
        )
    )
    request = actions.build_execution_request()
    activation = actions.build_execution_activation(request, _provenance())

    digest = actions._write_activation_authority(target, activation)

    assert digest == actions.content_digest(activation)
    assert json.loads(authority.read_bytes()) == {
        "schema": "ppo_normalization_execution_activation_authority_v1",
        "activation_sha256": digest,
    }

    with pytest.raises(ValueError, match="null state"):
        actions._write_activation_authority(target, activation)


def test_evidence_tree_digest_rejects_symlink(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "payload.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    link = root / "link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="symlink"):
        actions._tree_digest(root)


def test_request_snapshot_requires_green_exact_head_ci_and_fixed_request(
    monkeypatch,
) -> None:
    request = actions.build_execution_request()
    raw = actions.canonical_json_bytes(request)
    request_sha = actions.hashlib.sha256(raw).hexdigest()
    body = (
        actions.REVIEW_REQUEST_MARKER
        + "\n"
        + json.dumps(
            {"execution_code_sha": HEAD, "request_sha256": request_sha},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    event = {
        "repository": {"full_name": "shuntatsu/trade_rl", "id": 1103009698},
        "issue": {"number": 999, "pull_request": {}},
        "comment": {"body": body, "user": {"login": "operator"}},
    }

    def fake_api(url, *, token, method="GET", payload=None, allow_not_found=False):
        assert token == "token"
        if "/collaborators/operator/permission" in url:
            return {"permission": "write"}
        if url.endswith("/pulls/999"):
            return {
                "state": "open",
                "draft": True,
                "head": {"sha": HEAD, "repo": {"id": 1103009698}},
                "base": {"ref": "main"},
            }
        if url.endswith("/branches/main"):
            return {"commit": {"sha": "b" * 40}}
        if "/compare/" in url:
            return {
                "behind_by": 0,
                "merge_base_commit": {"sha": "b" * 40},
                "files": [
                    {"filename": "report/ppo-normalization-execution-request.json"}
                ],
            }
        if "/contents/report/ppo-normalization-execution-request.json" in url:
            import base64

            return {"encoding": "base64", "content": base64.b64encode(raw).decode()}
        if "/actions/runs?" in url:
            return {
                "workflow_runs": [
                    {
                        "id": 321,
                        "name": "CI",
                        "head_sha": HEAD,
                        "status": "completed",
                        "pull_requests": [{"number": 999}],
                    }
                ]
            }
        if "/actions/runs/321/jobs" in url:
            return {
                "jobs": [
                    {"name": name, "status": "completed", "conclusion": "success"}
                    for name in (
                        "Lean Core",
                        "PPO Runtime",
                        "Human Guide",
                        "Generic Independent Research Review",
                    )
                ]
            }
        if url.endswith("/pulls/770"):
            return {
                "merged": True,
                "head": {"sha": actions.IMPLEMENTATION_CODE_SHA},
                "merge_commit_sha": actions.IMPLEMENTATION_MERGE_SHA,
            }
        if "/git/ref/tags/seal/ppo-normalization-implementation-20260927-v2" in url:
            return {
                "object": {
                    "sha": actions.IMPLEMENTATION_SEAL_TAG_OBJECT_SHA,
                    "type": "tag",
                }
            }
        if f"/git/tags/{actions.IMPLEMENTATION_SEAL_TAG_OBJECT_SHA}" in url:
            return {
                "tag": actions.IMPLEMENTATION_SEAL_TAG,
                "object": {"sha": actions.IMPLEMENTATION_CODE_SHA, "type": "commit"},
                "message": (
                    f"implementation_seal_sha256={actions.IMPLEMENTATION_SEAL_SHA256}\n"
                ),
            }
        if "/git/ref/tags/review/ppo-normalization-replication-v2" in url:
            return {
                "object": {
                    "sha": actions.REVIEW_TAG_OBJECT_SHA,
                    "type": "tag",
                }
            }
        if f"/git/tags/{actions.REVIEW_TAG_OBJECT_SHA}" in url:
            return {
                "tag": actions.REVIEW_TAG,
                "object": {"sha": actions.IMPLEMENTATION_CODE_SHA, "type": "commit"},
                "message": "review",
            }
        if "/actions/artifacts/" in url:
            return {
                "id": actions.SOURCE_ARTIFACT_ID,
                "expired": False,
                "name": "source",
                "workflow_run": {
                    "id": actions.SOURCE_ARTIFACT_RUN_ID,
                    "repository_id": 1103009698,
                    "head_repository_id": 1103009698,
                },
                "digest": f"sha256:{actions.SOURCE_ARTIFACT_SHA256}",
            }
        if "/git/ref/tags/activation/ppo-normalization-corrected-v1" in url:
            assert allow_not_found is True
            return None
        raise AssertionError(url)

    monkeypatch.setattr(actions, "_api_json", fake_api)

    outputs = actions._request_outputs(event, token="token")

    assert outputs["execution_code_sha"] == HEAD
    assert outputs["request_sha256"] == request_sha
    assert len(outputs["activation_identity"]) == 64


def test_request_snapshot_rejects_missing_generic_review_gate(monkeypatch) -> None:
    def fake_api(url, *, token, method="GET", payload=None, allow_not_found=False):
        if "/actions/runs?" in url:
            return {
                "workflow_runs": [
                    {
                        "id": 1,
                        "name": "CI",
                        "head_sha": HEAD,
                        "status": "completed",
                        "pull_requests": [{"number": 9}],
                    }
                ]
            }
        if "/actions/runs/1/jobs" in url:
            return {
                "jobs": [
                    {"name": name, "status": "completed", "conclusion": "success"}
                    for name in ("Lean Core", "PPO Runtime", "Human Guide")
                ]
            }
        raise AssertionError(url)

    monkeypatch.setattr(actions, "_api_json", fake_api)
    with pytest.raises(ValueError, match="Generic Independent Research Review"):
        actions._require_green_exact_head_ci(
            "shuntatsu/trade_rl",
            token="token",
            pull_number=9,
            head_sha=HEAD,
        )


def test_green_exact_head_ci_accepts_latest_review_triggered_full_run(
    monkeypatch,
) -> None:
    def fake_api(url, *, token, method="GET", payload=None, allow_not_found=False):
        if "/actions/runs?" in url:
            assert "event=" not in url
            return {
                "workflow_runs": [
                    {
                        "id": 2,
                        "name": "CI",
                        "event": "pull_request_review",
                        "head_sha": HEAD,
                        "status": "completed",
                        "pull_requests": [{"number": 9}],
                    },
                    {
                        "id": 1,
                        "name": "CI",
                        "event": "pull_request",
                        "head_sha": HEAD,
                        "status": "completed",
                        "pull_requests": [{"number": 9}],
                    },
                ]
            }
        if "/actions/runs/2/jobs" in url:
            return {
                "jobs": [
                    {"name": name, "status": "completed", "conclusion": "success"}
                    for name in (
                        "Lean Core",
                        "PPO Runtime",
                        "Human Guide",
                        "Generic Independent Research Review",
                    )
                ]
            }
        raise AssertionError(url)

    monkeypatch.setattr(actions, "_api_json", fake_api)

    assert (
        actions._require_green_exact_head_ci(
            "shuntatsu/trade_rl",
            token="token",
            pull_number=9,
            head_sha=HEAD,
        )
        == 2
    )


def test_latest_exact_head_ci_failure_invalidates_older_green(monkeypatch) -> None:
    def fake_api(url, *, token, method="GET", payload=None, allow_not_found=False):
        if "/actions/runs?" in url:
            return {
                "workflow_runs": [
                    {
                        "id": 2,
                        "name": "CI",
                        "event": "pull_request_review",
                        "head_sha": HEAD,
                        "status": "completed",
                        "pull_requests": [{"number": 9}],
                    },
                    {
                        "id": 1,
                        "name": "CI",
                        "event": "pull_request_review",
                        "head_sha": HEAD,
                        "status": "completed",
                        "pull_requests": [{"number": 9}],
                    },
                ]
            }
        if "/actions/runs/2/jobs" in url:
            return {
                "jobs": [
                    {
                        "name": name,
                        "status": "completed",
                        "conclusion": (
                            "failure"
                            if name == "Generic Independent Research Review"
                            else "success"
                        ),
                    }
                    for name in (
                        "Lean Core",
                        "PPO Runtime",
                        "Human Guide",
                        "Generic Independent Research Review",
                    )
                ]
            }
        raise AssertionError(url)

    monkeypatch.setattr(actions, "_api_json", fake_api)

    with pytest.raises(ValueError, match="Generic Independent Research Review"):
        actions._require_green_exact_head_ci(
            "shuntatsu/trade_rl",
            token="token",
            pull_number=9,
            head_sha=HEAD,
        )


def test_execution_failure_receipt_contains_no_economic_payload() -> None:
    receipt = actions.build_execution_failure_receipt(
        repository="shuntatsu/trade_rl",
        repository_id=1103009698,
        run_id=123,
        run_attempt=1,
        execution_code_sha=HEAD,
        request_sha256="b" * 64,
        activation_sha256="c" * 64,
        activation_tag_created=True,
        phase="slot",
        slot="candidate_normalized_seed2",
        slot_consumed=True,
    )

    assert receipt["schema"] == "ppo_normalization_execution_failure_v1"
    assert receipt["activation_tag_created"] is True
    assert receipt["slot"] == "candidate_normalized_seed2"
    assert receipt["slot_consumed"] is True
    assert receipt["economic_result_inspected"] is False
    assert receipt["production_eligible"] is False
    assert receipt["live_trading_authorized"] is False
    assert not any(
        key in receipt
        for key in ("error", "error_message", "result", "return", "metrics", "pnl")
    )


def test_execute_failure_publishes_only_non_economic_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "output" / "execution"
    source = tmp_path / "source"
    source.mkdir()
    request = actions.build_execution_request()
    provenance = _provenance()
    spec = SimpleNamespace(slot="candidate_normalized_seed2")

    monkeypatch.setattr(
        actions,
        "_read_request_from_checkout",
        lambda *_args, **_kwargs: request,
    )
    monkeypatch.setattr(actions, "build_candidate_run_provenance", lambda: provenance)
    monkeypatch.setattr(
        actions, "_create_activation_tag", lambda *_args, **_kwargs: "d" * 40
    )
    monkeypatch.setattr(
        actions,
        "_write_activation_authority",
        lambda _target, activation: actions.content_digest(activation),
    )
    monkeypatch.setattr(actions, "_download_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        actions,
        "prepare_replication_execution",
        lambda root, _activation: root.mkdir(parents=True),
    )
    monkeypatch.setattr(actions, "replication_arm_specs", lambda: (spec,))
    monkeypatch.setattr(
        actions,
        "execute_replication_slot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    monkeypatch.setattr(
        actions,
        "replication_slot_state",
        lambda *_args, **_kwargs: {
            "consumed": True,
            "failed": True,
            "result_published": False,
        },
    )

    class FakeTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self.path = tmp_path / "temporary-source"
            self.path.mkdir(exist_ok=True)

        def __enter__(self):
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(actions.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)

    with pytest.raises(RuntimeError, match="fit failed"):
        actions.execute_main(
            {
                "GITHUB_REPOSITORY": "shuntatsu/trade_rl",
                "GITHUB_REPOSITORY_ID": "1103009698",
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "1",
                "GITHUB_TOKEN": "token",
                "EXECUTION_CODE_SHA": HEAD,
                "REQUEST_SHA256": "b" * 64,
                "TARGET_ROOT": str(target),
                "OUTPUT_ROOT": str(output),
            }
        )

    receipt_path = output.parent / "execution-failure.json"
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["phase"] == "slot"
    assert receipt["slot"] == spec.slot
    assert receipt["slot_consumed"] is True
    assert receipt["activation_tag_created"] is True
    assert receipt["economic_result_inspected"] is False
    assert not (output / "transport.json").exists()
    assert not any(
        key in receipt
        for key in ("error", "error_message", "result", "return", "metrics", "pnl")
    )


def test_fallback_failure_receipt_recovers_consumed_activation_from_tag(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "output" / "execution"
    activation = "c" * 64
    request_sha = "b" * 64
    tag_object_sha = "d" * 40
    record = {
        "schema": "ppo_normalization_repository_activation_v1",
        "execution_code_sha": HEAD,
        "request_sha256": request_sha,
        "activation_sha256": activation,
        "implementation_digest": actions.IMPLEMENTATION_DIGEST,
        "implementation_seal_sha256": actions.IMPLEMENTATION_SEAL_SHA256,
        "fresh_reconstruction_sha256": actions.FRESH_RECONSTRUCTION_SHA256,
        "assurance_review_sha256": actions.ASSURANCE_REVIEW_SHA256,
        "run_id": 123,
        "run_attempt": 1,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }

    def fake_api(url, *, token, method="GET", payload=None, allow_not_found=False):
        assert token == "token"
        if "/git/ref/tags/activation/ppo-normalization-corrected-v1" in url:
            return {"object": {"sha": tag_object_sha, "type": "tag"}}
        if f"/git/tags/{tag_object_sha}" in url:
            return {
                "tag": actions.ACTIVATION_TAG,
                "object": {"sha": HEAD, "type": "commit"},
                "message": actions.canonical_json_bytes(record).decode("utf-8"),
            }
        raise AssertionError(url)

    monkeypatch.setattr(actions, "_api_json", fake_api)

    assert (
        actions.ensure_failure_receipt_main(
            {
                "GITHUB_REPOSITORY": "shuntatsu/trade_rl",
                "GITHUB_REPOSITORY_ID": "1103009698",
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "1",
                "GITHUB_TOKEN": "token",
                "EXECUTION_CODE_SHA": HEAD,
                "REQUEST_SHA256": request_sha,
                "OUTPUT_ROOT": str(output),
            }
        )
        == 0
    )

    receipt = json.loads((output.parent / "execution-failure.json").read_bytes())
    assert receipt["activation_sha256"] == activation
    assert receipt["activation_tag_created"] is True
    assert receipt["phase"] == "workflow_failure"
    assert receipt["slot"] is None
    assert receipt["slot_consumed"] is None
    assert receipt["economic_result_inspected"] is False
    assert not any(
        key in receipt
        for key in ("error", "error_message", "result", "return", "metrics", "pnl")
    )


def test_fallback_failure_receipt_preserves_existing_detailed_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "output" / "execution"
    receipt_path = output.parent / "execution-failure.json"
    original = actions.build_execution_failure_receipt(
        repository="shuntatsu/trade_rl",
        repository_id=1103009698,
        run_id=123,
        run_attempt=1,
        execution_code_sha=HEAD,
        request_sha256="b" * 64,
        activation_sha256="c" * 64,
        activation_tag_created=True,
        phase="slot",
        slot="control_raw_seed0",
        slot_consumed=True,
    )
    actions._write_canonical_once(receipt_path, original)
    monkeypatch.setattr(
        actions,
        "_api_json",
        lambda *_args, **_kwargs: pytest.fail("existing receipt must not be replaced"),
    )

    assert (
        actions.ensure_failure_receipt_main(
            {
                "GITHUB_REPOSITORY": "shuntatsu/trade_rl",
                "GITHUB_REPOSITORY_ID": "1103009698",
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "1",
                "GITHUB_TOKEN": "token",
                "EXECUTION_CODE_SHA": HEAD,
                "REQUEST_SHA256": "b" * 64,
                "OUTPUT_ROOT": str(output),
            }
        )
        == 0
    )
    assert json.loads(receipt_path.read_bytes()) == original


def test_render_request_is_head_independent(tmp_path, capsys) -> None:
    output = tmp_path / "request.json"

    assert actions.render_request_main([str(output)]) == 0

    payload = json.loads(output.read_bytes())
    assert payload == actions.build_execution_request()
    assert "execution_code_sha" not in payload
    assert (
        capsys.readouterr().out.strip()
        == actions.hashlib.sha256(output.read_bytes()).hexdigest()
    )


def test_verification_transport_receipt_is_required_before_reveal(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "artifact"
    replication = root / "replication"
    replication.mkdir(parents=True)
    (replication / "placeholder").write_bytes(b"evidence")
    activation = "a" * 64
    verification_set = "b" * 64
    monkeypatch.setattr(actions, "_tree_digest", lambda _root: "c" * 64)
    monkeypatch.setattr(
        actions, "_verification_set_digest", lambda _root: verification_set
    )
    actions._write_canonical_once(
        root / "verification-transport.json",
        {
            "schema": "ppo_normalization_verification_transport_v1",
            "activation_sha256": activation,
            "verification_set_sha256": verification_set,
            "replication_tree_sha256": "c" * 64,
            "all_slots_verified": True,
            "no_refit": True,
        },
    )

    assert (
        actions._validate_verification_transport_tree(
            root,
            expected_activation_digest=activation,
            expected_verification_set_sha256=verification_set,
        )
        == replication
    )

    payload = json.loads((root / "verification-transport.json").read_bytes())
    payload["no_refit"] = False
    (root / "verification-transport.json").write_bytes(
        actions.canonical_json_bytes(payload)
    )
    with pytest.raises(ValueError, match="verification transport"):
        actions._validate_verification_transport_tree(
            root,
            expected_activation_digest=activation,
            expected_verification_set_sha256=verification_set,
        )


def test_workflow_artifact_tree_redownload_uses_raw_zip_digest(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []

    def fake_download(
        repository,
        reference,
        *,
        repository_id,
        token,
        destination,
        deadline,
    ):
        calls.append(("download", reference))
        destination.write_bytes(b"archive")
        return {"id": reference.artifact_id}

    def fake_extract(archive, extracted, expected_sha):
        calls.append(("extract", expected_sha))
        extracted.mkdir()
        (extracted / "evidence.json").write_text("{}", encoding="utf-8")

    from types import SimpleNamespace

    monkeypatch.setattr(
        actions,
        "_checkpoint_transport",
        lambda: SimpleNamespace(
            download_artifact_archive=fake_download,
            extract_verified_archive=fake_extract,
        ),
    )

    root = actions._download_workflow_artifact_tree(
        "shuntatsu/trade_rl",
        repository_id=1103009698,
        token="token",
        artifact_id=42,
        run_id=84,
        sha256="d" * 64,
        temp_root=tmp_path,
        label="verification",
    )

    assert root == tmp_path / "verification"
    reference = calls[0][1]
    assert reference.artifact_id == 42
    assert reference.run_id == 84
    assert reference.sha256 == "d" * 64
    assert calls[1] == ("extract", "d" * 64)


def test_request_transport_has_no_top_level_project_imports() -> None:
    import ast
    from pathlib import Path

    tree = ast.parse(Path(actions.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)

    assert not any(
        name == "trade_rl" or name.startswith("trade_rl.") for name in imported
    )
    assert "tools.ppo_checkpoint_actions" not in imported


def test_stdlib_canonical_json_matches_repository_canonical_json() -> None:
    from trade_rl.artifacts import (
        canonical_json_bytes as repository_canonical_json_bytes,
    )

    payload = {
        "z": None,
        "a": [True, False, 1, 1.25, "text"],
        "nested": {"beta": "β", "alpha": 2},
    }

    assert actions.canonical_json_bytes(payload) == repository_canonical_json_bytes(
        payload
    )
    assert (
        actions.content_digest(payload)
        == actions.hashlib.sha256(repository_canonical_json_bytes(payload)).hexdigest()
    )
