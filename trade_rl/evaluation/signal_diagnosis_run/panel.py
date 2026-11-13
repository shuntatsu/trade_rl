"""Dense arrays of the development Dataset and their admission checks (SD2).

``SignalPanel`` holds only what the diagnosis reads: bar-close timestamps, closes,
the 167 protocol features with their availability masks and ``available_at``. Every
check here runs before a single statistic is computed, and a violation of the window
is a refusal (an exception), never a flag.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.signal_diagnosis.protocol import (
    ForbiddenEvaluationWindowError,
    SignalDiagnosisProtocol,
    SignalDiagnosisProtocolError,
)

NS_PER_HOUR = 3_600_000_000_000
NS_PER_DAY = 24 * NS_PER_HOUR


def parse_instant_ns(text: str) -> int:
    """Nanoseconds since the epoch of a canonical ``...Z`` instant of the protocol."""

    return int(np.datetime64(text[:-1], "ns").astype(np.int64))


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    dataset_id: str
    artifact_digest: str
    arrays_digest: str
    feature_config_digest: str

    def mismatches(self, protocol: SignalDiagnosisProtocol) -> tuple[str, ...]:
        data = protocol.data
        return tuple(
            name
            for name in (
                "dataset_id",
                "artifact_digest",
                "arrays_digest",
                "feature_config_digest",
            )
            if getattr(self, name) != getattr(data, name)
        )


@dataclass(frozen=True, slots=True)
class SignalPanel:
    """``(time, symbol, feature)`` arrays aligned to the protocol roster."""

    identity: DatasetIdentity
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    timestamps_ns: np.ndarray
    close: np.ndarray
    features: np.ndarray
    feature_available: np.ndarray
    available_at_ns: np.ndarray

    def __post_init__(self) -> None:
        rows, symbols = len(self.timestamps_ns), len(self.symbols)
        if self.close.shape != (rows, symbols):
            raise SignalDiagnosisProtocolError("close must be (time, symbol)")
        shape = (rows, symbols, len(self.feature_names))
        if self.features.shape != shape or self.feature_available.shape != shape:
            raise SignalDiagnosisProtocolError("features must be (time, symbol, name)")
        if self.available_at_ns.shape != (rows, symbols):
            raise SignalDiagnosisProtocolError("available_at must be (time, symbol)")

    @property
    def n_rows(self) -> int:
        return int(self.timestamps_ns.shape[0])


def panel_from_dataset(
    dataset: MarketDataset, *, artifact_digest: str, arrays_digest: str
) -> SignalPanel:
    """Dense panel of a verified ``MarketDataset`` (no value is interpreted here)."""

    return SignalPanel(
        identity=DatasetIdentity(
            dataset_id=dataset.dataset_id,
            artifact_digest=artifact_digest,
            arrays_digest=arrays_digest,
            feature_config_digest=dataset.feature_config_digest,
        ),
        symbols=tuple(dataset.symbols),
        feature_names=tuple(dataset.feature_names),
        timestamps_ns=dataset.timestamps.astype("datetime64[ns]").astype(np.int64),
        close=np.asarray(dataset.close, dtype=np.float64),
        features=np.asarray(dataset.features),
        feature_available=np.asarray(dataset.feature_available, dtype=np.bool_),
        available_at_ns=dataset.resolved_array("available_at")
        .astype("datetime64[ns]")
        .astype(np.int64),
    )


def protocol_feature_names(protocol: SignalDiagnosisProtocol) -> tuple[str, ...]:
    return tuple(name for family in protocol.families for name in family.feature_names)


def admit_panel(panel: SignalPanel, protocol: SignalDiagnosisProtocol) -> None:
    """Refuse a panel that is not the protocol roster inside the protocol window.

    The window is the half-open bar-period interval; bar-close timestamps lie in
    ``[evaluation_start, evaluation_end]``. A bar closing after the end belongs to the
    FS1 warm-up span or the sealed window and is refused unconditionally, as is a bar
    closing before the start.
    """

    data = protocol.data
    if panel.symbols != data.symbols:
        raise SignalDiagnosisProtocolError(
            f"panel symbols {panel.symbols} are not the protocol roster"
        )
    names = protocol_feature_names(protocol)
    if set(panel.feature_names) != set(names) or len(panel.feature_names) != len(names):
        raise SignalDiagnosisProtocolError(
            "panel features are not exactly the 167 protocol features"
        )
    stamps = panel.timestamps_ns
    if stamps.size < 3 or bool(np.any(np.diff(stamps) != NS_PER_HOUR)):
        raise SignalDiagnosisProtocolError("panel must be a regular hourly grid")
    if int(stamps[0]) < parse_instant_ns(data.evaluation_start):
        raise SignalDiagnosisProtocolError(
            "panel holds bars closing before the protocol window start"
        )
    if int(stamps[-1]) > parse_instant_ns(data.evaluation_end):
        raise ForbiddenEvaluationWindowError(
            "panel holds bars closing after the protocol window end "
            f"{data.evaluation_end}: FS1 warm-up, sealed or unused data is refused"
        )
    if not np.isfinite(panel.close).all() or bool((panel.close <= 0.0).any()):
        raise SignalDiagnosisProtocolError("close must be finite and positive")


def admit_artifact_manifest(
    manifest: Mapping[str, Any], protocol: SignalDiagnosisProtocol
) -> None:
    """Refuse a published Dataset artifact from its manifest alone, before any array.

    The manifest is small JSON; no array payload is needed or read. It must hash to its
    own ``artifact_digest`` (so every field below is authenticated), and the artifact,
    arrays and Dataset identities and the feature-config digest must equal the pins of
    the consumed development Dataset: bytes with those digests are that Dataset, whose
    coverage ends at the window end by construction. The roster must be the protocol
    roster, the clock a continuous hourly grid, and the number of bars must fit the
    protocol window (``[start, end]`` closes hourly), so a Dataset that extends into the
    FS1 warm-up span, the sealed window or unused data is refused from metadata. The
    array payload is verified and every bar is checked later (``admit_panel``).
    """

    data = protocol.data
    body = {key: value for key, value in manifest.items() if key != "artifact_digest"}
    claimed = manifest.get("artifact_digest")
    if type(claimed) is not str or content_digest(body) != claimed:
        raise SignalDiagnosisProtocolError(
            "artifact manifest does not hash to its artifact_digest"
        )
    scalars = manifest.get("dataset")
    arrays = manifest.get("arrays")
    if not isinstance(scalars, Mapping) or not isinstance(arrays, Mapping):
        raise SignalDiagnosisProtocolError("artifact manifest is malformed")
    pins = (
        ("artifact_digest", claimed, data.artifact_digest),
        ("arrays_digest", manifest.get("arrays_digest"), data.arrays_digest),
        ("dataset_id", manifest.get("dataset_id"), data.dataset_id),
        (
            "feature_config_digest",
            scalars.get("feature_config_digest"),
            data.feature_config_digest,
        ),
    )
    mismatched = [name for name, found, pinned in pins if found != pinned]
    if mismatched:
        raise SignalDiagnosisProtocolError(
            f"artifact is not the pinned development Dataset: {mismatched}"
        )
    if tuple(scalars.get("symbols") or ()) != data.symbols:
        raise SignalDiagnosisProtocolError("artifact symbols are not the roster")
    if (
        scalars.get("calendar_kind") != "continuous"
        or scalars.get("periods_per_year") != 24 * 365
    ):
        raise SignalDiagnosisProtocolError("artifact is not a continuous hourly grid")
    stamps = arrays.get("timestamps")
    shape = stamps.get("shape") if isinstance(stamps, Mapping) else None
    if (
        not isinstance(shape, list | tuple)
        or len(shape) != 1
        or type(shape[0]) is not int
    ):
        raise SignalDiagnosisProtocolError("artifact manifest has no timestamps shape")
    span = parse_instant_ns(data.evaluation_end) - parse_instant_ns(
        data.evaluation_start
    )
    most_bars = span // NS_PER_HOUR + 1
    if shape[0] > most_bars:
        raise ForbiddenEvaluationWindowError(
            f"artifact holds {shape[0]} bars, more than the {most_bars} of the "
            f"protocol window ending {data.evaluation_end}: FS1 warm-up, sealed or "
            "unused data is refused before any array is read"
        )
