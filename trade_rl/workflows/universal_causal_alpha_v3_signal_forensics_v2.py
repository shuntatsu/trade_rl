"""Read-only Causal Alpha V3 Signal Forensics V2 orchestration."""

from __future__ import annotations

from pathlib import Path

from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics import (
    CausalAlphaV3UnavailableAnalysis,
    load_causal_alpha_v3_signal_forensics,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2_analysis import (
    build_causal_alpha_v3_signal_forensics_v2_analysis,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2_contracts import (
    CausalAlphaV3SignalForensicsReportV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2_loader import (
    load_causal_alpha_v3_signal_forensics_v2_sidecars,
)


def _complete_sidecar_unavailable_analyses() -> tuple[
    CausalAlphaV3UnavailableAnalysis, ...
]:
    return (
        CausalAlphaV3UnavailableAnalysis(
            analysis="availability_error_causal_attribution",
            reason=(
                "Availability-stratified diagnostic summaries are descriptive "
                "associations and cannot establish that missing-feature availability "
                "caused forecast error."
            ),
        ),
        CausalAlphaV3UnavailableAnalysis(
            analysis="canonical_ridge_model_digest_reconstruction",
            reason=(
                "Diagnostic sidecars persist a research projection of ridge state, but "
                "not every canonical model-digest input required for independent digest "
                "reconstruction."
            ),
        ),
        CausalAlphaV3UnavailableAnalysis(
            analysis="market_regime_classification",
            reason=(
                "The persisted Signal evidence does not contain authored market-regime "
                "labels, and V2 does not invent them from external or reconstructed data."
            ),
        ),
        CausalAlphaV3UnavailableAnalysis(
            analysis="overlapping_row_independent_confidence",
            reason=(
                "Realized diagnostic rows may overlap in label windows and remain "
                "descriptive rather than independent confidence samples."
            ),
        ),
        CausalAlphaV3UnavailableAnalysis(
            analysis="profitability_or_production_go_inference",
            reason=(
                "Signal Forensics V2 is research-only diagnostic evidence and does not "
                "establish strategy profitability, promotion eligibility, or Production GO."
            ),
        ),
        CausalAlphaV3UnavailableAnalysis(
            analysis="row_feature_error_attribution",
            reason=(
                "The sidecar stores row availability counts/fractions and per-feature "
                "marginals, not the exact unavailable-feature set for each decision row."
            ),
        ),
    )


def load_causal_alpha_v3_signal_forensics_v2(
    root: Path,
) -> CausalAlphaV3SignalForensicsReportV2:
    """Load V1 canonical forensics and bind complete V2 diagnostics when present."""

    resolved_root = Path(root)
    base = load_causal_alpha_v3_signal_forensics(resolved_root)
    diagnostics_root = resolved_root / "signal" / "diagnostics"
    if diagnostics_root.exists() or diagnostics_root.is_symlink():
        bound = load_causal_alpha_v3_signal_forensics_v2_sidecars(resolved_root)
        analysis = build_causal_alpha_v3_signal_forensics_v2_analysis(bound)
        return CausalAlphaV3SignalForensicsReportV2(
            base_forensics_digest=base.digest,
            base_forensics=base,
            sidecar_mode="sidecar_complete",
            sidecar_analysis=analysis,
            unavailable_analyses=_complete_sidecar_unavailable_analyses(),
        )
    return CausalAlphaV3SignalForensicsReportV2(
        base_forensics_digest=base.digest,
        base_forensics=base,
        sidecar_mode="historical_unavailable",
        sidecar_analysis=None,
        unavailable_analyses=base.unavailable_analyses,
    )


__all__ = ["load_causal_alpha_v3_signal_forensics_v2"]
