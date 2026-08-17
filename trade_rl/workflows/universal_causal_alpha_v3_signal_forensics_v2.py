"""Read-only Causal Alpha V3 Signal Forensics V2 orchestration."""

from __future__ import annotations

from pathlib import Path

from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics import (
    load_causal_alpha_v3_signal_forensics,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2_contracts import (
    CausalAlphaV3SignalForensicsReportV2,
)


def load_causal_alpha_v3_signal_forensics_v2(
    root: Path,
) -> CausalAlphaV3SignalForensicsReportV2:
    """Load V1 canonical forensics and bind prospective V2 diagnostics."""

    resolved_root = Path(root)
    base = load_causal_alpha_v3_signal_forensics(resolved_root)
    diagnostics_root = resolved_root / "signal" / "diagnostics"
    if diagnostics_root.exists() or diagnostics_root.is_symlink():
        raise ValueError(
            "V3 signal forensics V2 diagnostic path requires complete sidecar binding"
        )
    return CausalAlphaV3SignalForensicsReportV2(
        base_forensics_digest=base.digest,
        base_forensics=base,
        sidecar_mode="historical_unavailable",
        sidecar_analysis=None,
        unavailable_analyses=base.unavailable_analyses,
    )
