"""Runtime identity and exact-version guard for optional NautilusTrader usage."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib import metadata

NAUTILUS_PACKAGE_NAME = "nautilus_trader"
EXPECTED_NAUTILUS_VERSION = "1.230.0"


class NautilusRuntimeUnavailableError(RuntimeError):
    """Raised when a Nautilus-only path is requested without the optional wheel."""


class NautilusRuntimeVersionError(RuntimeError):
    """Raised when the installed NautilusTrader wheel does not match the contract."""


@dataclass(frozen=True, slots=True)
class NautilusRuntimeIdentity:
    """Serializable identity for the installed optional Nautilus runtime."""

    package_name: str
    package_version: str | None
    python_version: str
    platform: str
    installed: bool
    compatible: bool


def probe_nautilus_runtime() -> NautilusRuntimeIdentity:
    """Return runtime identity without importing ``nautilus_trader`` itself."""

    try:
        version = metadata.version(NAUTILUS_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        version = None

    return NautilusRuntimeIdentity(
        package_name=NAUTILUS_PACKAGE_NAME,
        package_version=version,
        python_version=platform.python_version(),
        platform=f"{sys.platform}-{platform.machine().lower()}",
        installed=version is not None,
        compatible=version == EXPECTED_NAUTILUS_VERSION,
    )


def require_nautilus_runtime() -> NautilusRuntimeIdentity:
    """Require the exact NautilusTrader release before touching upstream APIs."""

    identity = probe_nautilus_runtime()
    if not identity.installed:
        raise NautilusRuntimeUnavailableError(
            "NautilusTrader is optional; install the 'nautilus' extra before using "
            "the Nautilus execution runtime."
        )
    if not identity.compatible:
        raise NautilusRuntimeVersionError(
            "NautilusTrader runtime mismatch: expected "
            f"{EXPECTED_NAUTILUS_VERSION}, got {identity.package_version}."
        )
    return identity
