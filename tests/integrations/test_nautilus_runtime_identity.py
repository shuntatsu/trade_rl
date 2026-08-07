from __future__ import annotations

from importlib import metadata

import pytest

from trade_rl.integrations.nautilus.runtime_identity import (
    EXPECTED_NAUTILUS_VERSION,
    NautilusRuntimeUnavailableError,
    NautilusRuntimeVersionError,
    probe_nautilus_runtime,
    require_nautilus_runtime,
)


def test_nautilus_version_is_exactly_pinned() -> None:
    assert EXPECTED_NAUTILUS_VERSION == "1.230.0"


def test_probe_is_import_safe_when_optional_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", missing)

    identity = probe_nautilus_runtime()

    assert identity.installed is False
    assert identity.compatible is False
    assert identity.package_version is None


def test_require_rejects_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", missing)

    with pytest.raises(NautilusRuntimeUnavailableError):
        require_nautilus_runtime()


def test_require_rejects_version_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "version", lambda _: "1.229.0")

    with pytest.raises(NautilusRuntimeVersionError, match="1.230.0"):
        require_nautilus_runtime()
