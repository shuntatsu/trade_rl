from scripts.verify_served_identity import (
    identity_matches,
    main,
    verify_with_retries,
)


def _identity(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "ready",
        "active_version": "v123",
        "bundle_digest": "d" * 64,
        "release_git_sha": "a" * 40,
    }
    payload.update(overrides)
    return payload


def test_identity_matches_ready_or_degraded_exact_identity() -> None:
    for status in ("ready", "degraded"):
        assert identity_matches(
            _identity(status=status),
            expected_version="v123",
            expected_digest="d" * 64,
            expected_release_git_sha="a" * 40,
        )


def test_identity_rejects_previous_active_bundle() -> None:
    assert not identity_matches(
        _identity(active_version="previous", bundle_digest="e" * 64),
        expected_version="v123",
        expected_digest="d" * 64,
        expected_release_git_sha="a" * 40,
    )


def test_identity_rejects_unavailable_or_code_mismatch() -> None:
    assert not identity_matches(
        _identity(status="unavailable"),
        expected_version="v123",
        expected_digest="d" * 64,
        expected_release_git_sha="a" * 40,
    )
    assert not identity_matches(
        _identity(release_git_sha="b" * 40),
        expected_version="v123",
        expected_digest="d" * 64,
        expected_release_git_sha="a" * 40,
    )


def test_verifier_retries_until_exact_identity_is_observed() -> None:
    responses = iter(
        [
            _identity(active_version="previous"),
            _identity(status="degraded"),
        ]
    )
    sleeps: list[float] = []

    assert verify_with_retries(
        url="https://serving.example/ready",
        expected_version="v123",
        expected_digest="d" * 64,
        expected_release_git_sha="a" * 40,
        attempts=2,
        interval_seconds=0.25,
        fetch=lambda url: responses.__next__(),
        sleep=sleeps.append,
    )
    assert sleeps == [0.25]


def test_verifier_returns_false_after_transport_failures() -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    def fail(url: str) -> dict[str, object]:
        attempts.append(url)
        raise OSError("unreachable")

    assert not verify_with_retries(
        url="https://serving.example/ready",
        expected_version="v123",
        expected_digest="d" * 64,
        expected_release_git_sha="a" * 40,
        attempts=3,
        interval_seconds=0.5,
        fetch=fail,
        sleep=sleeps.append,
    )
    assert len(attempts) == 3
    assert sleeps == [0.5, 0.5]


def test_cli_returns_nonzero_when_live_identity_never_matches(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.verify_served_identity.verify_with_retries",
        lambda **kwargs: False,
    )

    result = main(
        [
            "--url",
            "https://serving.example/ready",
            "--version",
            "v123",
            "--digest",
            "d" * 64,
            "--release-git-sha",
            "a" * 40,
            "--attempts",
            "1",
            "--interval-seconds",
            "0",
        ]
    )

    assert result == 1
