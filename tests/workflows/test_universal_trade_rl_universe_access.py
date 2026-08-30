from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from trade_rl.domain.universal_trade_rl_universe import (
    UniversalTradeRLSymbolExclusion,
    UniversalTradeRLUniverseConfig,
)
from trade_rl.workflows.universal_trade_rl_universe_access import (
    UniversalTradeRLAccessPhase,
    UniversalTradeRLAdmissionAuthorization,
    UniversalTradeRLUniverseAccess,
)
from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
    build_universal_trade_rl_universe_manifest,
)

_FROZEN_GENERATION_DIGEST = "f" * 64
_SELECTION_EVIDENCE_DIGEST = "e" * 64


def _manifest() -> UniversalTradeRLUniverseManifest:
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("LINKUSDT",),
        admission_symbols=("AVAXUSDT",),
        exclusions=(
            UniversalTradeRLSymbolExclusion(
                symbol="LUNA2USDT",
                reason="insufficient_contiguous_history",
            ),
        ),
    )
    sources = tuple(
        UniversalTradeRLSymbolSource(
            symbol=symbol,
            dataset_digest=char * 64,
            first_timestamp_ns=1,
            last_timestamp_ns=100,
            row_count=100,
        )
        for symbol, char in (
            ("AVAXUSDT", "a"),
            ("BTCUSDT", "b"),
            ("ETHUSDT", "c"),
            ("LINKUSDT", "d"),
            ("LUNA2USDT", "e"),
        )
    )
    return build_universal_trade_rl_universe_manifest(config=config, sources=sources)


def _authorization(
    manifest: UniversalTradeRLUniverseManifest,
    *,
    universe_manifest_digest: str | None = None,
    frozen_generation_digest: str = _FROZEN_GENERATION_DIGEST,
    selection_evidence_digest: str = _SELECTION_EVIDENCE_DIGEST,
) -> UniversalTradeRLAdmissionAuthorization:
    return UniversalTradeRLAdmissionAuthorization(
        universe_manifest_digest=(
            manifest.digest
            if universe_manifest_digest is None
            else universe_manifest_digest
        ),
        frozen_generation_digest=frozen_generation_digest,
        selection_evidence_digest=selection_evidence_digest,
    )


def _admission_access() -> UniversalTradeRLUniverseAccess:
    manifest = _manifest()
    return UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest,
        phase=UniversalTradeRLAccessPhase.ADMISSION,
        authorization=_authorization(manifest),
        frozen_generation_digest=_FROZEN_GENERATION_DIGEST,
        selection_evidence_digest=_SELECTION_EVIDENCE_DIGEST,
    )


def test_train_exposes_only_train_fit_scope() -> None:
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=_manifest(),
        phase=UniversalTradeRLAccessPhase.TRAIN,
    )

    assert access.fit_symbols == ("BTCUSDT", "ETHUSDT")
    assert access.evaluation_symbols == ()
    assert access.admission_symbols == ("AVAXUSDT",)
    access.require_fit_scope(("BTCUSDT", "ETHUSDT"))
    access.require_normalization_scope(("BTCUSDT",))
    access.require_calibration_scope(("ETHUSDT",))
    assert not hasattr(access, "open")


def test_development_can_evaluate_but_not_fit() -> None:
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=_manifest(),
        phase=UniversalTradeRLAccessPhase.DEVELOPMENT,
    )

    assert access.fit_symbols == ("BTCUSDT", "ETHUSDT")
    assert access.evaluation_symbols == ("LINKUSDT",)
    access.require_evaluation_scope(("LINKUSDT",))
    with pytest.raises(PermissionError, match="Train-only"):
        access.require_fit_scope(access.evaluation_symbols)
    with pytest.raises(PermissionError, match="Train-only"):
        access.require_normalization_scope(("LINKUSDT",))
    with pytest.raises(PermissionError, match="Train-only"):
        access.require_calibration_scope(("LINKUSDT",))


def test_admission_requires_matching_authorization() -> None:
    manifest = _manifest()

    with pytest.raises(PermissionError, match="authorization"):
        UniversalTradeRLUniverseAccess.for_phase(
            manifest=manifest,
            phase=UniversalTradeRLAccessPhase.ADMISSION,
            frozen_generation_digest=_FROZEN_GENERATION_DIGEST,
            selection_evidence_digest=_SELECTION_EVIDENCE_DIGEST,
        )


def test_authorized_admission_exposes_only_admission_evaluation_scope() -> None:
    access = _admission_access()

    assert access.fit_symbols == ()
    assert access.evaluation_symbols == ("AVAXUSDT",)
    access.require_evaluation_scope(("AVAXUSDT",))
    with pytest.raises(PermissionError, match="Train-only"):
        access.require_fit_scope(access.admission_symbols)
    with pytest.raises(PermissionError, match="Train-only"):
        access.require_normalization_scope(access.admission_symbols)
    with pytest.raises(PermissionError, match="Train-only"):
        access.require_calibration_scope(access.admission_symbols)


def test_admission_rejects_wrong_universe_digest() -> None:
    manifest = _manifest()

    with pytest.raises(PermissionError, match="universe"):
        UniversalTradeRLUniverseAccess.for_phase(
            manifest=manifest,
            phase=UniversalTradeRLAccessPhase.ADMISSION,
            authorization=_authorization(
                manifest,
                universe_manifest_digest="a" * 64,
            ),
            frozen_generation_digest=_FROZEN_GENERATION_DIGEST,
            selection_evidence_digest=_SELECTION_EVIDENCE_DIGEST,
        )


def test_admission_rejects_changed_frozen_generation_digest() -> None:
    manifest = _manifest()

    with pytest.raises(PermissionError, match="generation"):
        UniversalTradeRLUniverseAccess.for_phase(
            manifest=manifest,
            phase=UniversalTradeRLAccessPhase.ADMISSION,
            authorization=_authorization(manifest),
            frozen_generation_digest="d" * 64,
            selection_evidence_digest=_SELECTION_EVIDENCE_DIGEST,
        )


def test_admission_rejects_changed_selection_evidence_digest() -> None:
    manifest = _manifest()

    with pytest.raises(PermissionError, match="Selection"):
        UniversalTradeRLUniverseAccess.for_phase(
            manifest=manifest,
            phase=UniversalTradeRLAccessPhase.ADMISSION,
            authorization=_authorization(manifest),
            frozen_generation_digest=_FROZEN_GENERATION_DIGEST,
            selection_evidence_digest="d" * 64,
        )


@pytest.mark.parametrize(
    "phase",
    (UniversalTradeRLAccessPhase.TRAIN, UniversalTradeRLAccessPhase.DEVELOPMENT),
)
def test_pre_admission_phases_forbid_authorization(
    phase: UniversalTradeRLAccessPhase,
) -> None:
    manifest = _manifest()

    with pytest.raises(PermissionError, match="forbid"):
        UniversalTradeRLUniverseAccess.for_phase(
            manifest=manifest,
            phase=phase,
            authorization=_authorization(manifest),
        )


def test_admission_requires_expected_generation_and_selection_identity() -> None:
    manifest = _manifest()
    authorization = _authorization(manifest)

    with pytest.raises(PermissionError, match="generation"):
        UniversalTradeRLUniverseAccess.for_phase(
            manifest=manifest,
            phase=UniversalTradeRLAccessPhase.ADMISSION,
            authorization=authorization,
            selection_evidence_digest=_SELECTION_EVIDENCE_DIGEST,
        )
    with pytest.raises(PermissionError, match="Selection"):
        UniversalTradeRLUniverseAccess.for_phase(
            manifest=manifest,
            phase=UniversalTradeRLAccessPhase.ADMISSION,
            authorization=authorization,
            frozen_generation_digest=_FROZEN_GENERATION_DIGEST,
        )


def test_scope_validators_reject_unsorted_or_duplicate_input() -> None:
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=_manifest(),
        phase=UniversalTradeRLAccessPhase.DEVELOPMENT,
    )

    with pytest.raises(ValueError, match="sorted"):
        access.require_fit_scope(("ETHUSDT", "BTCUSDT"))
    with pytest.raises(ValueError, match="unique"):
        access.require_fit_scope(("BTCUSDT", "BTCUSDT"))


def test_evaluation_scope_cannot_cross_phase_role() -> None:
    development = UniversalTradeRLUniverseAccess.for_phase(
        manifest=_manifest(),
        phase=UniversalTradeRLAccessPhase.DEVELOPMENT,
    )
    admission = _admission_access()

    with pytest.raises(PermissionError, match="evaluation"):
        development.require_evaluation_scope(("AVAXUSDT",))
    with pytest.raises(PermissionError, match="evaluation"):
        admission.require_evaluation_scope(("LINKUSDT",))


def test_authorization_is_immutable_and_digest_bound() -> None:
    manifest = _manifest()
    authorization = _authorization(manifest)

    assert authorization.schema_version == "universal_trade_rl_admission_authorization_v1"
    assert len(authorization.digest) == 64
    with pytest.raises(FrozenInstanceError):
        authorization.universe_manifest_digest = "a" * 64  # type: ignore[misc]
