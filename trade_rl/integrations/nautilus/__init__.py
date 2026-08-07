"""Optional NautilusTrader integration boundary.

Importing this package must not require the optional ``nautilus_trader`` wheel.
Concrete runtime APIs validate the exact wheel before importing upstream modules.
"""

from trade_rl.integrations.nautilus.runtime_identity import (
    EXPECTED_NAUTILUS_VERSION,
    NautilusRuntimeIdentity,
    NautilusRuntimeUnavailableError,
    NautilusRuntimeVersionError,
    probe_nautilus_runtime,
    require_nautilus_runtime,
)

__all__ = (
    "EXPECTED_NAUTILUS_VERSION",
    "NautilusRuntimeIdentity",
    "NautilusRuntimeUnavailableError",
    "NautilusRuntimeVersionError",
    "probe_nautilus_runtime",
    "require_nautilus_runtime",
)
