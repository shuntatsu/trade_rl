"""Phase-aware U0 universe access firewall for Universal Trade RL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

UNIVERSAL_TRADE_RL_ADMISSION_AUTHORIZATION_SCHEMA: Final = (
    "universal_trade_rl_admission_authorization_v1"
)


class UniversalTradeRLAccessPhase(str, Enum):
    """U0 research phases with progressively narrower data access."""

    TRAIN = "train"
    DEVELOPMENT = "development"
    ADMISSION = "admission"


@dataclass(frozen=True, slots=True)
class UniversalTradeRLAdmissionAuthorization:
    """Immutable authorization to reveal one frozen Admission universe."""

    universe_manifest_digest: str
    frozen_generation_digest: str
    selection_evidence_digest: str
    schema_version: str = UNIVERSAL_TRADE_RL_ADMISSION_AUTHORIZATION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != UNIVERSAL_TRADE_RL_ADMISSION_AUTHORIZATION_SCHEMA:
            raise ValueError("unsupported Universal Trade RL Admission authorization schema")
        require_sha256(
            self.universe_manifest_digest,
            field="Admission authorization universe manifest digest",
        )
        require_sha256(
            self.frozen_generation_digest,
            field="Admission authorization frozen generation digest",
        )
        require_sha256(
            self.selection_evidence_digest,
            field="Admission authorization Selection evidence digest",
        )
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("Universal Trade RL Admission authorization digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "universe_manifest_digest": self.universe_manifest_digest,
            "frozen_generation_digest": self.frozen_generation_digest,
            "selection_evidence_digest": self.selection_evidence_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _canonical_scope(symbols: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(symbols, tuple):
        raise TypeError(f"{field} must be an immutable tuple")
    if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise ValueError(f"{field} symbols are invalid")
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"{field} symbols must be unique")
    if symbols != tuple(sorted(symbols)):
        raise ValueError(f"{field} symbols must be sorted")
    return symbols


def _role_symbols(
    manifest: UniversalTradeRLUniverseManifest,
    role: UniversalTradeRLSymbolRole,
) -> tuple[str, ...]:
    return tuple(entry.symbol for entry in manifest.entries if entry.role is role)


@dataclass(frozen=True, slots=True)
class UniversalTradeRLUniverseAccess:
    """Immutable phase-specific view of the frozen U0 universe."""

    universe_manifest_digest: str
    phase: UniversalTradeRLAccessPhase
    train_symbols: tuple[str, ...]
    development_symbols: tuple[str, ...]
    admission_symbols: tuple[str, ...]
    fit_symbols: tuple[str, ...]
    evaluation_symbols: tuple[str, ...]
    admission_authorization_digest: str | None = None

    def __post_init__(self) -> None:
        require_sha256(
            self.universe_manifest_digest,
            field="universe access manifest digest",
        )
        if not isinstance(self.phase, UniversalTradeRLAccessPhase):
            raise TypeError("universe access phase is invalid")
        train = _canonical_scope(self.train_symbols, field="Train role")
        development = _canonical_scope(
            self.development_symbols,
            field="Development role",
        )
        admission = _canonical_scope(self.admission_symbols, field="Admission role")
        if set(train) & set(development) or set(train) & set(admission):
            raise ValueError("universe access role scopes overlap")
        if set(development) & set(admission):
            raise ValueError("universe access role scopes overlap")
        _canonical_scope(self.fit_symbols, field="fit scope")
        _canonical_scope(self.evaluation_symbols, field="evaluation scope")
        if self.phase is UniversalTradeRLAccessPhase.TRAIN:
            if self.fit_symbols != train or self.evaluation_symbols:
                raise ValueError("Train universe access contract is invalid")
            if self.admission_authorization_digest is not None:
                raise ValueError("Train universe access must forbid Admission authorization")
        elif self.phase is UniversalTradeRLAccessPhase.DEVELOPMENT:
            if self.fit_symbols != train or self.evaluation_symbols != development:
                raise ValueError("Development universe access contract is invalid")
            if self.admission_authorization_digest is not None:
                raise ValueError(
                    "Development universe access must forbid Admission authorization"
                )
        else:
            if self.fit_symbols or self.evaluation_symbols != admission:
                raise ValueError("Admission universe access contract is invalid")
            if self.admission_authorization_digest is None:
                raise ValueError("Admission universe access requires authorization")
            require_sha256(
                self.admission_authorization_digest,
                field="Admission universe access authorization digest",
            )

    @classmethod
    def for_phase(
        cls,
        *,
        manifest: UniversalTradeRLUniverseManifest,
        phase: UniversalTradeRLAccessPhase,
        authorization: UniversalTradeRLAdmissionAuthorization | None = None,
        frozen_generation_digest: str | None = None,
        selection_evidence_digest: str | None = None,
    ) -> UniversalTradeRLUniverseAccess:
        if not isinstance(manifest, UniversalTradeRLUniverseManifest):
            raise TypeError("universe access requires a valid manifest")
        if not isinstance(phase, UniversalTradeRLAccessPhase):
            raise TypeError("universe access phase is invalid")

        train = _role_symbols(manifest, UniversalTradeRLSymbolRole.TRAIN)
        development = _role_symbols(
            manifest,
            UniversalTradeRLSymbolRole.DEVELOPMENT,
        )
        admission = _role_symbols(manifest, UniversalTradeRLSymbolRole.ADMISSION)

        if phase is not UniversalTradeRLAccessPhase.ADMISSION:
            if (
                authorization is not None
                or frozen_generation_digest is not None
                or selection_evidence_digest is not None
            ):
                raise PermissionError(
                    "Train/Development phases forbid Admission authorization context"
                )
            return cls(
                universe_manifest_digest=manifest.digest,
                phase=phase,
                train_symbols=train,
                development_symbols=development,
                admission_symbols=admission,
                fit_symbols=train,
                evaluation_symbols=(
                    ()
                    if phase is UniversalTradeRLAccessPhase.TRAIN
                    else development
                ),
            )

        if authorization is None:
            raise PermissionError("Admission authorization is required")
        if not isinstance(authorization, UniversalTradeRLAdmissionAuthorization):
            raise TypeError("Admission authorization contract is invalid")
        if frozen_generation_digest is None:
            raise PermissionError("Admission frozen generation identity is required")
        if selection_evidence_digest is None:
            raise PermissionError("Admission Selection evidence identity is required")
        require_sha256(
            frozen_generation_digest,
            field="expected frozen generation digest",
        )
        require_sha256(
            selection_evidence_digest,
            field="expected Selection evidence digest",
        )
        if authorization.universe_manifest_digest != manifest.digest:
            raise PermissionError("Admission authorization universe identity mismatch")
        if authorization.frozen_generation_digest != frozen_generation_digest:
            raise PermissionError("Admission authorization generation identity mismatch")
        if authorization.selection_evidence_digest != selection_evidence_digest:
            raise PermissionError("Admission authorization Selection identity mismatch")
        return cls(
            universe_manifest_digest=manifest.digest,
            phase=phase,
            train_symbols=train,
            development_symbols=development,
            admission_symbols=admission,
            fit_symbols=(),
            evaluation_symbols=admission,
            admission_authorization_digest=authorization.digest,
        )

    def require_fit_scope(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        scope = _canonical_scope(symbols, field="fit scope")
        if not set(scope).issubset(self.train_symbols):
            raise PermissionError("Universal Trade RL fit scope is Train-only")
        if self.phase is UniversalTradeRLAccessPhase.ADMISSION:
            raise PermissionError("Universal Trade RL fit scope is Train-only")
        return scope

    def require_normalization_scope(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        try:
            return self.require_fit_scope(symbols)
        except PermissionError as error:
            raise PermissionError(
                "Universal Trade RL normalization scope is Train-only"
            ) from error

    def require_calibration_scope(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        try:
            return self.require_fit_scope(symbols)
        except PermissionError as error:
            raise PermissionError(
                "Universal Trade RL calibration scope is Train-only"
            ) from error

    def require_evaluation_scope(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        scope = _canonical_scope(symbols, field="evaluation scope")
        if not set(scope).issubset(self.evaluation_symbols):
            raise PermissionError(
                "Universal Trade RL evaluation scope is not authorized for this phase"
            )
        return scope


__all__ = [
    "UNIVERSAL_TRADE_RL_ADMISSION_AUTHORIZATION_SCHEMA",
    "UniversalTradeRLAccessPhase",
    "UniversalTradeRLAdmissionAuthorization",
    "UniversalTradeRLUniverseAccess",
]
