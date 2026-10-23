"""Source/review refusal before data access, with fake Git and JSON only."""

import importlib
import json

import pytest


@pytest.mark.parametrize("reviewer", ["gpt-6-luna", "UNKNOWN"])
def test_implementation_helper_or_unknown_model_cannot_be_independent(
    tmp_path, monkeypatch, reviewer
):
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_runner")
    protocol = {"scope": "ARTIFICIAL_ADMISSION_ONLY"}
    monkeypatch.setattr(module, "source_pins", lambda: {"runner.py": "a" * 64})
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(
            {
                "scope": protocol["scope"],
                "status": "APPROVED_FOR_TIER_D_DEVELOPMENT_EXECUTION",
                "reviewer_model": reviewer,
                "protocol_digest": module.json_sha256(protocol),
                "source_sha256": module.source_pins(),
            }
        )
    )
    with pytest.raises(ValueError, match="different-model"):
        module.validate_external_review(path, protocol)


@pytest.mark.parametrize("change", ["head", "dirty", "untracked", "bytes"])
def test_market_source_rejects_uncommitted_or_changed_package(
    tmp_path, monkeypatch, change
):
    module = importlib.import_module("tools.tier_d_nonrl.tier_d_protocol")
    package = tmp_path / "tools/tier_d_nonrl"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_bytes(b'"""manufactured source"""\n')
    monkeypatch.setattr(module, "PACKAGE", package)
    monkeypatch.setattr(module, "REPO", tmp_path)
    if change == "untracked":
        (package / "extra.py").write_bytes(b"pass\n")

    def fake_git(command, **kwargs):
        args = command[1:]
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40 if change == "head" else "a" * 40
        if args[:1] == ["status"]:
            return b" M tracked.py\n" if change == "dirty" else b""
        if args[:1] == ["ls-files"]:
            return b"tools/tier_d_nonrl/__init__.py\x00"
        if args[:1] == ["show"]:
            return b"different\n" if change == "bytes" else source.read_bytes()
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "check_output", fake_git)
    with pytest.raises(ValueError):
        module.validate_committed_source({"source_head": "a" * 40})
