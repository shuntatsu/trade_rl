"""Cached serving runtime with atomic bundle hot-swap and stateful inference."""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

import numpy as np

from mars_lite.env.observation import (
    ObservationSchema,
    ObservationState,
    ProgressMode,
    build_observation,
)
from mars_lite.serving.audit_store import AuditStore
from mars_lite.serving.bundle import ServingBundle
from mars_lite.serving.contracts import (
    InferenceRequest,
    InferenceResponse,
    InferenceState,
)
from mars_lite.serving.registry import ModelRegistry

_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class PolicyLike(Protocol):
    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[Any, Any]: ...


DecisionFn = Callable[
    [np.ndarray, InferenceState, np.ndarray | None, np.ndarray | None],
    tuple[np.ndarray, Mapping[str, Any]],
]
GuardrailFn = Callable[
    [np.ndarray, np.ndarray, InferenceState, float, np.ndarray],
    tuple[np.ndarray, Mapping[str, Any]],
]
RiskFn = Callable[[np.ndarray, InferenceState, Sequence[str]], Mapping[str, Any]]
ComponentFactory = Callable[[ServingBundle], "RuntimeComponents"]


def validate_release_git_sha(value: str) -> str:
    """Validate and normalize the immutable code identity used by strict serving."""

    if _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError("release_git_sha must be a 40-character hexadecimal SHA")
    return value.lower()


@dataclass(frozen=True)
class FeatureSnapshot:
    snapshot_id: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    feature_history: np.ndarray
    global_features: np.ndarray
    close_history: np.ndarray
    data_age_hours: float

    def validate(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id is required")
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("snapshot symbols must be unique and non-empty")
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("feature_names must be unique and non-empty")
        features = np.asarray(self.feature_history)
        if features.ndim != 3:
            raise ValueError(
                "feature_history must have shape (bars, symbols, features)"
            )
        if features.shape[0] == 0:
            raise ValueError("feature_history must contain at least one bar")
        if features.shape[1:] != (len(self.symbols), len(self.feature_names)):
            raise ValueError("feature_history dimensions do not match schema")
        if len(set(self.global_feature_names)) != len(self.global_feature_names):
            raise ValueError("global_feature_names must be unique")
        if not all(
            isinstance(name, str) and name for name in self.global_feature_names
        ):
            raise ValueError("global_feature_names must contain non-empty strings")
        globals_ = np.asarray(self.global_features)
        if globals_.ndim != 1 or not np.isfinite(globals_).all():
            raise ValueError("global_features must be a finite one-dimensional array")
        if len(globals_) != len(self.global_feature_names):
            raise ValueError("global feature dimensions do not match schema")
        close = np.asarray(self.close_history)
        if close.ndim != 2 or close.shape != (features.shape[0], len(self.symbols)):
            raise ValueError("close_history dimensions do not match feature history")
        if not np.isfinite(close).all() or np.any(close <= 0):
            raise ValueError("close_history must be finite and positive")
        if not math.isfinite(self.data_age_hours) or self.data_age_hours < 0:
            raise ValueError("data_age_hours must be finite and non-negative")


@dataclass(frozen=True)
class RuntimeComponents:
    model: PolicyLike
    decide: DecisionFn
    apply_guardrails: GuardrailFn
    evaluate_risk: RiskFn
    include_observation_risk_state: bool
    serving_progress: float = 0.0
    vol_lookback: int = 0
    htf_feature_name: str | None = None


@dataclass(frozen=True)
class ReadinessState:
    status: str
    active_version: str | None
    bundle_digest: str | None
    reason: str | None = None
    release_git_sha: str | None = None


@dataclass(frozen=True)
class _LoadedRuntime:
    bundle: ServingBundle
    components: RuntimeComponents
    loaded_at: float


class ServingRuntime:
    def __init__(
        self,
        *,
        registry: ModelRegistry,
        audit_store: AuditStore,
        component_factory: ComponentFactory | None = None,
        release_git_sha: str | None = None,
        strict_release_binding: bool = False,
    ) -> None:
        self.registry = registry
        self.audit_store = audit_store
        if component_factory is None:
            from mars_lite.serving.runtime_defaults import default_component_factory

            component_factory = default_component_factory
        self.component_factory = component_factory
        if strict_release_binding and release_git_sha is None:
            raise ValueError("release_git_sha is required in strict serving mode")
        self.release_git_sha = (
            validate_release_git_sha(release_git_sha)
            if release_git_sha is not None
            else None
        )
        self.strict_release_binding = bool(strict_release_binding)
        self._lock = threading.RLock()
        self._loaded: _LoadedRuntime | None = None
        self._readiness = ReadinessState(
            "unavailable",
            None,
            None,
            reason="not loaded",
            release_git_sha=self.release_git_sha,
        )

    def readiness(self) -> ReadinessState:
        with self._lock:
            return self._readiness

    @property
    def active_version(self) -> str | None:
        with self._lock:
            return self._loaded.bundle.version if self._loaded is not None else None

    def active_bundle(self) -> ServingBundle | None:
        """Return the immutable bundle currently loaded in memory."""
        with self._lock:
            return self._loaded.bundle if self._loaded is not None else None

    def refresh(self) -> bool:
        try:
            bundle = self.registry.get_active_bundle()
            if self.strict_release_binding:
                assert self.release_git_sha is not None
                if bundle.git_sha.lower() != self.release_git_sha:
                    raise ValueError(
                        "bundle git sha mismatch: "
                        f"bundle={bundle.git_sha.lower()} "
                        f"running={self.release_git_sha}"
                    )
            with self._lock:
                if (
                    self._loaded is not None
                    and self._loaded.bundle.bundle_digest == bundle.bundle_digest
                ):
                    self._readiness = ReadinessState(
                        "ready",
                        bundle.version,
                        bundle.bundle_digest,
                        release_git_sha=self.release_git_sha,
                    )
                    return True
            components = self.component_factory(bundle)
            observation_dim = bundle.metadata.get("observation_dim")
            if (
                isinstance(observation_dim, bool)
                or not isinstance(observation_dim, int)
                or observation_dim <= 0
            ):
                raise ValueError("metadata.observation_dim must be a positive integer")
            probe = np.zeros(observation_dim, dtype=np.float32)
            components.model.predict(probe, deterministic=True)
            candidate = _LoadedRuntime(bundle, components, time.time())
            with self._lock:
                self._loaded = candidate
                self._readiness = ReadinessState(
                    "ready",
                    bundle.version,
                    bundle.bundle_digest,
                    release_git_sha=self.release_git_sha,
                )
            self.audit_store.append_event(
                event_type="bundle_loaded",
                model_version=bundle.version,
                bundle_digest=bundle.bundle_digest,
                payload={
                    "status": "ready",
                    "release_git_sha": self.release_git_sha,
                },
            )
            return True
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            with self._lock:
                if self._loaded is None:
                    self._readiness = ReadinessState(
                        "unavailable",
                        None,
                        None,
                        reason=reason,
                        release_git_sha=self.release_git_sha,
                    )
                else:
                    self._readiness = ReadinessState(
                        "degraded",
                        self._loaded.bundle.version,
                        self._loaded.bundle.bundle_digest,
                        reason=reason,
                        release_git_sha=self.release_git_sha,
                    )
            self.audit_store.append_event(
                event_type="bundle_load_rejected",
                payload={
                    "reason": reason,
                    "release_git_sha": self.release_git_sha,
                },
            )
            return False

    def infer(
        self, request: InferenceRequest, snapshot: FeatureSnapshot
    ) -> InferenceResponse:
        with self._lock:
            loaded = self._loaded
        if loaded is None:
            return InferenceResponse(
                status="no_signal",
                request_id=request.request_id,
                market_snapshot_id=request.market_snapshot_id,
                model_version="",
                bundle_digest="",
                target_weights=None,
                reasons=("serving runtime is unavailable",),
            )
        bundle = loaded.bundle
        components = loaded.components
        try:
            request.validate(
                tuple(bundle.metadata["symbols"]),
                require_observation_risk_state=(
                    components.include_observation_risk_state
                ),
            )
            snapshot.validate()
            if snapshot.snapshot_id != request.market_snapshot_id:
                raise ValueError("market snapshot identity mismatch")
            if not self.audit_store.claim_request(
                request.request_id, request.payload_hash()
            ):
                return self._rejected(
                    request, bundle, "duplicate request_id replay", event="replay"
                )

            latest, recent_returns, htf_trend = self._prepare_features(
                bundle, components, snapshot
            )
            symbols = tuple(bundle.metadata["symbols"])
            current = request.state.weights_array(
                symbols,
                require_observation_risk_state=(
                    components.include_observation_risk_state
                ),
            )
            observation = build_observation(
                per_symbol_features=latest,
                global_features=np.asarray(snapshot.global_features, dtype=np.float64),
                state=ObservationState(
                    weights=current,
                    portfolio_value=request.state.portfolio_value,
                    peak_value=request.state.peak_value,
                    progress=components.serving_progress,
                    vol_scale=(
                        request.state.vol_scale
                        if request.state.vol_scale is not None
                        else 1.0
                    ),
                    dd_scale=(
                        request.state.dd_scale
                        if request.state.dd_scale is not None
                        else 1.0
                    ),
                    disagreement_scale=(
                        request.state.disagreement_scale
                        if request.state.disagreement_scale is not None
                        else 1.0
                    ),
                    est_port_vol=(
                        request.state.est_port_vol
                        if request.state.est_port_vol is not None
                        else 0.0
                    ),
                ),
                schema=ObservationSchema(
                    include_risk_state=components.include_observation_risk_state,
                    version=int(bundle.metadata["observation_schema_version"]),
                    progress_mode=cast(
                        ProgressMode, bundle.metadata["observation_progress_mode"]
                    ),
                ),
            )
            expected_dim = int(bundle.metadata["observation_dim"])
            if observation.shape != (expected_dim,):
                raise ValueError(
                    f"observation dimension {observation.shape[0]} != {expected_dim}"
                )
            raw_action, _ = components.model.predict(observation, deterministic=True)
            target, decision_info = components.decide(
                np.asarray(raw_action, dtype=np.float64).reshape(-1),
                request.state,
                recent_returns,
                htf_trend,
            )
            target = np.asarray(target, dtype=np.float64).reshape(-1)
            if target.shape != current.shape or not np.isfinite(target).all():
                raise ValueError("decision pipeline returned invalid target weights")
            final, guardrail = components.apply_guardrails(
                target,
                current,
                request.state,
                snapshot.data_age_hours,
                latest,
            )
            final = np.asarray(final, dtype=np.float64).reshape(-1)
            if final.shape != current.shape or not np.isfinite(final).all():
                raise ValueError("guardrails returned invalid target weights")
            risk = dict(components.evaluate_risk(final, request.state, symbols))
            if risk.get("approved") is not True:
                reason = str(risk.get("reason", "pre-trade risk rejected"))
                return self._rejected(
                    request,
                    bundle,
                    reason,
                    guardrail=guardrail,
                    risk=risk,
                    event="risk_rejected",
                )
            response = InferenceResponse(
                status="ok",
                request_id=request.request_id,
                market_snapshot_id=request.market_snapshot_id,
                model_version=bundle.version,
                bundle_digest=bundle.bundle_digest,
                target_weights={
                    symbol: float(weight) for symbol, weight in zip(symbols, final)
                },
                reasons=tuple(str(x) for x in guardrail.get("triggered", ())),
                guardrail=dict(guardrail),
                pre_trade_risk=risk,
            )
            self.audit_store.append_event(
                event_type="inference",
                request_id=request.request_id,
                model_version=bundle.version,
                bundle_digest=bundle.bundle_digest,
                payload={
                    "response": response.to_dict(),
                    "decision": dict(decision_info),
                },
            )
            return response
        except Exception as exc:
            return self._rejected(
                request,
                bundle,
                f"{type(exc).__name__}: {exc}",
                event="inference_rejected",
            )

    def _rejected(
        self,
        request: InferenceRequest,
        bundle: ServingBundle,
        reason: str,
        *,
        guardrail: Mapping[str, Any] | None = None,
        risk: Mapping[str, Any] | None = None,
        event: str,
    ) -> InferenceResponse:
        response = InferenceResponse(
            status="rejected",
            request_id=request.request_id,
            market_snapshot_id=request.market_snapshot_id,
            model_version=bundle.version,
            bundle_digest=bundle.bundle_digest,
            target_weights=None,
            reasons=(reason,),
            guardrail=dict(guardrail or {}),
            pre_trade_risk=dict(risk or {}),
        )
        self.audit_store.append_event(
            event_type=event,
            request_id=request.request_id,
            model_version=bundle.version,
            bundle_digest=bundle.bundle_digest,
            payload=response.to_dict(),
        )
        return response

    @staticmethod
    def _prepare_features(
        bundle: ServingBundle,
        components: RuntimeComponents,
        snapshot: FeatureSnapshot,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        symbols = tuple(bundle.metadata["symbols"])
        expected_names = tuple(bundle.preprocessing["feature_names"])
        if snapshot.symbols != symbols:
            raise ValueError("snapshot symbol order does not match bundle")
        if snapshot.feature_names != expected_names:
            raise ValueError("snapshot feature names do not match bundle")
        expected_globals = tuple(bundle.preprocessing["global_feature_names"])
        if snapshot.global_feature_names != expected_globals:
            raise ValueError("snapshot global feature names do not match bundle")
        history = np.asarray(snapshot.feature_history, dtype=np.float64).copy()
        feature_norm = bundle.preprocessing["feature_norm"]
        if feature_norm == "rank_gauss":
            from mars_lite.features.feature_pipeline import _gaussian_rank_transform

            history = _gaussian_rank_transform(
                history,
                int(bundle.preprocessing.get("rank_window", 250)),
                int(bundle.preprocessing.get("rank_min_periods", 40)),
            )
        mask = bundle.preprocessing.get("feature_mask")
        if mask is not None:
            mask_array = np.asarray(mask, dtype=bool)
            if mask_array.shape != (history.shape[2],):
                raise ValueError("feature mask length does not match snapshot")
            history[:, :, ~mask_array] = 0.0
        if int(bundle.preprocessing["post_mask_dim"]) != history.shape[2]:
            raise ValueError("post-mask feature dimension mismatch")
        latest = history[-1]

        recent_returns = None
        if components.vol_lookback > 0 and len(snapshot.close_history) > 1:
            count = min(len(snapshot.close_history), components.vol_lookback + 1)
            close = np.asarray(snapshot.close_history[-count:], dtype=np.float64)
            recent_returns = np.diff(close, axis=0) / close[:-1]
        htf_trend = None
        if components.htf_feature_name is not None:
            try:
                index = expected_names.index(components.htf_feature_name)
            except ValueError as exc:
                raise ValueError("HTF feature required by bundle is missing") from exc
            htf_trend = latest[:, index]
        return latest, recent_returns, htf_trend
