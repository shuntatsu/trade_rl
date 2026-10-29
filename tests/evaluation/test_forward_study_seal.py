"""FS1 seal: every refusal path, with fakes only (no dataset, no network, no P&L)."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_rl.evaluation.forward_study import seal
from trade_rl.evaluation.forward_study.registry import (
    FROZEN_PARAMS_DIGEST,
    FS1_REGISTRY,
    MANIFEST_SCHEMA,
)
from trade_rl.evaluation.forward_study.seal import (
    CodeIdentity,
    Fs1Refusal,
    ScoringPermit,
    parse_manifest,
    require_permit,
)

AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-123456789"
FETCH_AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-111111111"
SHA = "a" * 40
# scoring end 2027-05-01T00:00Z plus the 24 h settlement margin
AFTER_WINDOW = datetime(2027, 5, 2, 0, 0, tzinfo=UTC)


def manifest_dict(**overrides: Any) -> dict[str, Any]:
    first = np.datetime64(FS1_REGISTRY.dataset_first_row_latest)
    last = np.datetime64(FS1_REGISTRY.scoring_end)
    value: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_id": "b" * 64,
        "symbols": list(FS1_REGISTRY.symbols),
        "first_timestamp": str(first),
        "last_timestamp": str(last),
        "n_bars": int((last - first) / np.timedelta64(1, "h")) + 1,
        "build_policy_digest": FS1_REGISTRY.build_policy_digest(),
        "source_manifest": {"BTCUSDT-1h-2026-11.zip": "d" * 64},
        "authorization_reference": FETCH_AUTH,
        "fetched_utc": "2027-05-02T01:00:00",
        "sealed_utc": "2027-05-02T02:00:00",
    }
    value.update(overrides)
    return value


def manifest_raw(**overrides: Any) -> bytes:
    return (
        json.dumps(manifest_dict(**overrides), sort_keys=True, indent=2) + "\n"
    ).encode()


def good(**overrides: Any) -> dict[str, Any]:
    raw = overrides.pop("manifest_bytes", manifest_raw())
    kwargs: dict[str, Any] = {
        "now": AFTER_WINDOW,
        "authorization": AUTH,
        "expected_params_digest": FROZEN_PARAMS_DIGEST,
        "sealed_code_sha": SHA,
        "code_identity": CodeIdentity(SHA, True),
        "manifest_bytes": raw,
        "expected_manifest_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
    }
    kwargs.update(overrides)
    return kwargs


def authorize(**kwargs: Any) -> ScoringPermit:
    """The private factory, so a test can inject a registry or a frozen digest."""

    registry = kwargs.pop("registry", FS1_REGISTRY)
    frozen = kwargs.pop("frozen_params_digest", FROZEN_PARAMS_DIGEST)
    return seal._authorize(registry=registry, frozen_params_digest=frozen, **kwargs)


def reasons_of(**overrides: Any) -> tuple[str, ...]:
    with pytest.raises(Fs1Refusal) as caught:
        authorize(**good(**overrides))
    return caught.value.reasons


def test_every_precondition_met_returns_a_permit() -> None:
    permit = authorize(**good())
    assert isinstance(permit, ScoringPermit)
    assert permit.params_digest == FROZEN_PARAMS_DIGEST
    assert permit.authorization == AUTH and permit.code_sha == SHA
    assert permit.manifest.dataset_id == "b" * 64
    assert permit.authorized_at_utc == "2027-05-02T00:00:00Z"
    assert require_permit(permit) is permit


def test_refuses_before_the_scoring_end_by_wall_clock() -> None:
    just_before = AFTER_WINDOW - timedelta(seconds=1)
    assert reasons_of(now=just_before) == ("scoring_window_not_closed",)
    # The scoring end itself (the open of the final row) is not enough: the final bar
    # must be complete and settled, i.e. end + the 24 h settlement margin.
    at_end = datetime(2027, 5, 1, 0, 0, tzinfo=UTC)
    assert reasons_of(now=at_end) == ("scoring_window_not_closed",)
    assert reasons_of(now=datetime(2026, 10, 30, tzinfo=UTC)) == (
        "scoring_window_not_closed",
    )


def test_the_clock_is_compared_in_utc() -> None:
    plus_nine = timezone(timedelta(hours=9))
    # 2027-05-02T08:59 in +09:00 is still 2027-05-01T23:59 UTC.
    early = datetime(2027, 5, 2, 8, 59, tzinfo=plus_nine)
    assert reasons_of(now=early) == ("scoring_window_not_closed",)
    authorize(**good(now=datetime(2027, 5, 2, 9, 0, tzinfo=plus_nine)))
    with pytest.raises(ValueError):
        authorize(**good(now=datetime(2027, 6, 1)))


@pytest.mark.parametrize("missing", [None, ""])
def test_refuses_without_an_authorization_reference(missing: str | None) -> None:
    assert reasons_of(authorization=missing) == ("authorization_missing",)


@pytest.mark.parametrize(
    "bad",
    [
        "yes",
        "https://github.com/shuntatsu/trade_rl/issues/810",
        "https://github.com/shuntatsu/trade_rl/issues/811#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/issues/8100#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/issues/81#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/pull/810#issuecomment-1",
        "https://github.com/other/trade_rl/issues/810#issuecomment-1",
        "http://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1 ",
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-abc",
    ],
)
def test_refuses_a_malformed_authorization_reference(bad: str) -> None:
    assert reasons_of(authorization=bad) == ("authorization_malformed",)


def test_refuses_when_the_params_digest_differs_from_the_seal_record() -> None:
    assert reasons_of(expected_params_digest="0" * 64) == ("params_digest_mismatch",)
    assert reasons_of(expected_params_digest=None) == ("params_digest_mismatch",)


def test_refuses_when_the_registry_is_not_the_frozen_one() -> None:
    altered = replace(FS1_REGISTRY, max_drawdown_limit_pct=30.0)
    reasons = reasons_of(registry=altered, expected_params_digest=altered.digest())
    assert reasons == ("registry_digest_differs_from_frozen",)
    # An operator pasting the frozen digest does not hide the alteration either.
    both = reasons_of(registry=altered)
    assert both == ("registry_digest_differs_from_frozen", "params_digest_mismatch")


def test_refuses_when_the_code_is_not_the_sealed_commit() -> None:
    assert reasons_of(code_identity=CodeIdentity("b" * 40, True)) == (
        "code_sha_mismatch",
    )
    assert reasons_of(code_identity=CodeIdentity(SHA, False)) == ("code_tree_dirty",)
    assert reasons_of(code_identity=CodeIdentity(None, False)) == (
        "code_identity_unavailable",
    )


@pytest.mark.parametrize("bad", [None, "", "abc", "A" * 40, "a" * 39])
def test_refuses_without_a_full_sealed_code_sha(bad: str | None) -> None:
    assert "sealed_code_sha_missing" in reasons_of(sealed_code_sha=bad)


def test_refuses_without_a_manifest() -> None:
    assert reasons_of(manifest_bytes=None, expected_manifest_sha256=None) == (
        "manifest_missing",
    )


def test_refuses_when_the_manifest_digest_is_not_the_recorded_one() -> None:
    assert reasons_of(expected_manifest_sha256="0" * 64) == (
        "manifest_sha256_mismatch",
    )
    assert reasons_of(expected_manifest_sha256=None) == ("manifest_sha256_mismatch",)
    tampered = manifest_raw(dataset_id="e" * 64)
    assert reasons_of(
        manifest_bytes=tampered,
        expected_manifest_sha256=hashlib.sha256(manifest_raw()).hexdigest(),
    ) == ("manifest_sha256_mismatch",)


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[]",
        json.dumps({**manifest_dict(), "extra": 1}).encode(),
        json.dumps(manifest_dict(schema_version="other")).encode(),
        json.dumps(manifest_dict(dataset_id="XYZ")).encode(),
        json.dumps(manifest_dict(source_manifest={})).encode(),
        json.dumps(manifest_dict(source_manifest={"a": "short"})).encode(),
        json.dumps(manifest_dict(authorization_reference="yes")).encode(),
        json.dumps(manifest_dict(n_bars=1.5)).encode(),
        json.dumps(manifest_dict(symbols=[])).encode(),
    ],
)
def test_refuses_an_invalid_manifest(raw: bytes) -> None:
    assert reasons_of(manifest_bytes=raw) == ("manifest_invalid",)
    with pytest.raises(ValueError):
        parse_manifest(raw)


def test_refuses_a_manifest_for_another_window_or_roster() -> None:
    wrong_last = manifest_raw(last_timestamp="2027-04-30T00:00:00")
    assert reasons_of(manifest_bytes=wrong_last) == ("manifest_window_mismatch",)
    wrong_symbols = manifest_raw(symbols=["BTCUSDT", "ETHUSDT"])
    assert reasons_of(manifest_bytes=wrong_symbols) == ("manifest_window_mismatch",)
    late_first = manifest_raw(first_timestamp="2021-03-01T00:00:00")
    assert reasons_of(manifest_bytes=late_first) == ("manifest_window_mismatch",)


def test_all_unmet_preconditions_are_reported_together() -> None:
    reasons = reasons_of(
        now=datetime(2026, 10, 30, tzinfo=UTC),
        authorization=None,
        expected_params_digest="0" * 64,
        code_identity=CodeIdentity("b" * 40, False),
        manifest_bytes=None,
        expected_manifest_sha256=None,
    )
    assert set(reasons) == {
        "scoring_window_not_closed",
        "authorization_missing",
        "params_digest_mismatch",
        "code_sha_mismatch",
        "code_tree_dirty",
        "manifest_missing",
    }


def test_a_permit_cannot_be_forged_by_construction() -> None:
    real = authorize(**good())
    forged = ScoringPermit(
        registry=real.registry,
        manifest=real.manifest,
        manifest_sha256=real.manifest_sha256,
        authorization=real.authorization,
        params_digest=real.params_digest,
        code_sha=real.code_sha,
        authorized_at_utc=real.authorized_at_utc,
    )
    with pytest.raises(TypeError):
        require_permit(forged)
    with pytest.raises(TypeError):
        require_permit(object())


def test_the_authorization_pattern_is_pinned_to_issue_810() -> None:
    assert seal.AUTHORIZATION_PATTERN.pattern.endswith(
        r"/issues/810#issuecomment-[1-9][0-9]*$"
    )
    other = FETCH_AUTH.replace("issues/810", "issues/811")
    with pytest.raises(ValueError):
        parse_manifest(manifest_raw(authorization_reference=other))


def test_the_manifest_must_carry_the_frozen_build_policy_digest() -> None:
    wrong = manifest_raw(build_policy_digest="c" * 64)
    assert reasons_of(manifest_bytes=wrong) == ("manifest_build_policy_mismatch",)
    assert FS1_REGISTRY.build_policy["gap_policy"] == "non_tradable_rows"
    assert FS1_REGISTRY.build_policy["end_time"] == "2027-05-01T00:00:00+00:00"
    assert tuple(FS1_REGISTRY.build_policy["symbols"]) == FS1_REGISTRY.symbols  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        # fetched before the final bar plus margin settled
        ({"fetched_utc": "2027-05-01T12:00:00"}, "manifest_fetched_before_settlement"),
        # sealed before it was fetched
        (
            {"fetched_utc": "2027-05-02T05:00:00", "sealed_utc": "2027-05-02T04:00:00"},
            "manifest_fetched_before_settlement",
        ),
    ],
)
def test_the_recorded_fetch_time_must_follow_settlement(
    overrides: dict[str, str], expected: str
) -> None:
    assert reasons_of(manifest_bytes=manifest_raw(**overrides)) == (expected,)
    with pytest.raises(ValueError):
        parse_manifest(manifest_raw(fetched_utc="not a time"))


def test_the_public_entrypoints_have_no_registry_or_digest_seams() -> None:
    parameters = set(inspect.signature(seal.authorize_scoring).parameters)
    assert "registry" not in parameters and "frozen_params_digest" not in parameters
    permit = seal.authorize_scoring(**good())
    assert permit.registry is FS1_REGISTRY
    # a non-frozen registry is refused by the public factory even with its own digest
    altered = replace(FS1_REGISTRY, max_drawdown_limit_pct=30.0)
    with pytest.raises(Fs1Refusal) as caught:
        seal.authorize_scoring(**good(expected_params_digest=altered.digest()))
    assert "params_digest_mismatch" in caught.value.reasons


GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def git(repo: Path, *args: str) -> str:
    import os

    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def make_repo(path: Path) -> tuple[Path, str, str]:
    path.mkdir()
    git(path, "init", "-q")
    (path / "tracked.txt").write_text("one\n", encoding="utf-8")
    git(path, "add", "tracked.txt")
    git(path, "commit", "-q", "-m", "root")
    root = git(path, "rev-parse", "HEAD")
    (path / "tracked.txt").write_text("two\n", encoding="utf-8")
    git(path, "commit", "-q", "-am", "second")
    return path, root, git(path, "rev-parse", "HEAD")


def test_code_identity_is_clean_only_without_modified_or_untracked_files(
    tmp_path: Path,
) -> None:
    repo, root, head = make_repo(tmp_path / "repo")
    assert seal.current_code_identity(repo, root_commit=root) == CodeIdentity(
        head, True
    )
    (repo / "scratch.txt").write_text("untracked\n", encoding="utf-8")
    assert seal.current_code_identity(repo, root_commit=root) == CodeIdentity(
        head, False
    )
    (repo / "scratch.txt").unlink()
    (repo / "tracked.txt").write_text("edited\n", encoding="utf-8")
    assert seal.current_code_identity(repo, root_commit=root) == CodeIdentity(
        head, False
    )


def test_code_identity_ignores_ignored_files_and_requires_the_pinned_root(
    tmp_path: Path,
) -> None:
    repo, root, head = make_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore")
    head = git(repo, "rev-parse", "HEAD")
    (repo / "build.log").write_text("ignored\n", encoding="utf-8")
    assert seal.current_code_identity(repo, root_commit=root) == CodeIdentity(
        head, True
    )
    # another repository (different root commit) is not the trade_rl repository
    assert seal.current_code_identity(repo, root_commit="f" * 40) == CodeIdentity(
        None, False
    )


def test_code_identity_is_unavailable_outside_a_repository(tmp_path: Path) -> None:
    assert seal.current_code_identity(tmp_path) == CodeIdentity(None, False)


def test_a_foreign_copy_of_the_module_is_refused_even_with_the_right_root_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolates the module-location check: the root commit matches, the path does not."""

    repo, root, head = make_repo(tmp_path / "repo")
    foreign = repo / "vendor" / "copy" / "pkg" / "seal.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("# a copy of the module inside a trade_rl-like repo\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "vendor copy")
    head = git(repo, "rev-parse", "HEAD")
    assert seal.current_code_identity(repo, root_commit=root) == CodeIdentity(
        head, True
    )

    monkeypatch.setattr(seal, "__file__", str(foreign))
    assert foreign.resolve().parents[3] == repo.resolve()
    # root commit is right and the tree is clean, but this file is not the repository's
    # tracked trade_rl/evaluation/forward_study/seal.py
    assert seal.current_code_identity(None, root_commit=root) == CodeIdentity(
        None, False
    )

    # control: the same module file at the tracked location is accepted
    tracked = repo / "trade_rl" / "evaluation" / "forward_study" / "seal.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("# tracked copy\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "tracked location")
    head = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(seal, "__file__", str(tracked))
    assert seal.current_code_identity(None, root_commit=root) == CodeIdentity(
        head, True
    )


def test_the_real_checkout_identity_matches_git_when_history_is_available() -> None:
    root = Path(__file__).resolve().parents[2]

    def real_git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout

    roots = real_git("rev-list", "--max-parents=0", "HEAD")
    if roots is None or seal.TRADE_RL_ROOT_COMMIT not in roots.split():
        pytest.skip(
            "not a full git checkout of the trade_rl repository (shallow or no git)"
        )
    identity = seal.current_code_identity()
    head = real_git("rev-parse", "HEAD")
    status = real_git("status", "--porcelain", "--untracked-files=all")
    assert head is not None and status is not None
    assert identity.git_head == head.strip()
    # reports a dirty tree as dirty (and a clean one as clean); never silently clean
    assert identity.tracked_tree_clean == (status.strip() == "")


def test_manifest_round_trip_is_canonical_and_strict() -> None:
    raw = manifest_raw()
    parsed = parse_manifest(raw)
    assert parsed.symbols == FS1_REGISTRY.symbols
    assert parsed.authorization_reference == FETCH_AUTH
    assert np.datetime64(parsed.last_timestamp) == np.datetime64(
        FS1_REGISTRY.scoring_end
    )
