"""Authoritative U2 Development replay boundary after pre-development freeze."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from trade_rl.domain.common import require_sha256
from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.workflows.universal_trade_rl_u1_contract import UniversalTradeRLU1Contract
from trade_rl.workflows.universal_trade_rl_u2_contract import UniversalTradeRLU2Contract
from trade_rl.workflows.universal_trade_rl_u2_development_closure import (
    UniversalTradeRLU2AuthoritativeDevelopmentLock,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation import (
    UniversalTradeRLU2DevelopmentScopeClosure,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation_dataset import (
    U2EvaluationSourceArtifactLoader,
    U2EvaluationSourceArtifactLocator,
)
from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    UniversalTradeRLU2DevelopmentLock,
    UniversalTradeRLU2PreDevelopmentContract,
    universal_trade_rl_u2_evaluation_seed,
)
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    U2ReplayEnvironmentFactory,
    UniversalTradeRLU2DevelopmentReplaySession,
    UniversalTradeRLU2ReplayEvidence,
    UniversalTradeRLU2ReplayRequest,
    _build_universal_trade_rl_u2_development_replay_session_unlocked,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    UniversalTradeRLU2TimePartition,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

U2_DEVELOPMENT_REPLAY_AUTHORITY_SCHEMA: Final = (
    "universal_trade_rl_u2_development_replay_authority_v1"
)


def _canonical_scope_dataset_mapping(
    scope_closure: UniversalTradeRLU2DevelopmentScopeClosure,
) -> tuple[tuple[str, str], ...]:
    by_symbol: dict[str, str] = {}
    for scope in scope_closure.scopes:
        previous = by_symbol.setdefault(
            scope.concrete_symbol,
            scope.evaluation_dataset_digest,
        )
        if previous != scope.evaluation_dataset_digest:
            raise ValueError(
                "U2 Development scope closure has inconsistent evaluation dataset identity"
            )
    return tuple(sorted(by_symbol.items()))


def require_universal_trade_rl_u2_development_lock_for_replay(
    *,
    predevelopment_contract: UniversalTradeRLU2PreDevelopmentContract,
    base_lock: UniversalTradeRLU2DevelopmentLock,
    development_lock: UniversalTradeRLU2AuthoritativeDevelopmentLock,
    manifest: UniversalTradeRLUniverseManifest,
    u2_contract: UniversalTradeRLU2Contract,
    supplied_scope_closure: UniversalTradeRLU2DevelopmentScopeClosure,
    source_tree_digest: str,
    lockfile_digest: str,
    evaluation_runtime_identity_digest: str,
) -> UniversalTradeRLU2AuthoritativeDevelopmentLock:
    """Fail closed unless the exact pre-open lock authorizes this replay closure."""

    if not isinstance(
        predevelopment_contract, UniversalTradeRLU2PreDevelopmentContract
    ):
        raise TypeError("U2 Development replay requires a pre-development contract")
    if not isinstance(base_lock, UniversalTradeRLU2DevelopmentLock):
        raise TypeError("U2 Development replay requires a base Development lock")
    if not isinstance(
        development_lock,
        UniversalTradeRLU2AuthoritativeDevelopmentLock,
    ):
        raise TypeError("U2 Development replay requires an authoritative lock")
    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U2 Development replay requires a universe manifest")
    if not isinstance(u2_contract, UniversalTradeRLU2Contract):
        raise TypeError("U2 Development replay requires a U2 contract")
    if not isinstance(
        supplied_scope_closure,
        UniversalTradeRLU2DevelopmentScopeClosure,
    ):
        raise TypeError("U2 Development replay requires a Development scope closure")

    for field_name, value in (
        ("source tree digest", source_tree_digest),
        ("lockfile digest", lockfile_digest),
        ("evaluation runtime identity digest", evaluation_runtime_identity_digest),
    ):
        require_sha256(value, field=f"U2 Development replay {field_name}")

    if predevelopment_contract.universe_manifest_digest != manifest.digest:
        raise ValueError(
            "U2 Development replay pre-development universe identity mismatch"
        )
    if predevelopment_contract.u2_contract_digest != u2_contract.digest:
        raise ValueError("U2 Development replay pre-development U2 identity mismatch")
    if base_lock.predevelopment_contract_digest != predevelopment_contract.digest:
        raise ValueError(
            "U2 Development replay base-lock pre-development identity mismatch"
        )
    if base_lock.universe_manifest_digest != manifest.digest:
        raise ValueError("U2 Development replay base-lock universe identity mismatch")
    if base_lock.u2_contract_digest != u2_contract.digest:
        raise ValueError("U2 Development replay base-lock U2 identity mismatch")
    if base_lock.u1_contract_digest != u2_contract.u1_contract_digest:
        raise ValueError("U2 Development replay base-lock U1 identity mismatch")
    if base_lock.u1_normalizer_digest != u2_contract.u1_normalizer_digest:
        raise ValueError("U2 Development replay base-lock normalizer identity mismatch")

    if development_lock.base_lock_digest != base_lock.digest:
        raise ValueError(
            "U2 Development replay authoritative base-lock digest mismatch"
        )
    if (
        development_lock.predevelopment_contract_digest
        != predevelopment_contract.digest
    ):
        raise ValueError(
            "U2 Development replay authoritative pre-development identity mismatch"
        )
    if development_lock.universe_manifest_digest != manifest.digest:
        raise ValueError(
            "U2 Development replay authoritative universe identity mismatch"
        )
    if development_lock.u2_contract_digest != u2_contract.digest:
        raise ValueError("U2 Development replay authoritative U2 identity mismatch")
    if development_lock.u1_contract_digest != u2_contract.u1_contract_digest:
        raise ValueError("U2 Development replay authoritative U1 identity mismatch")
    if development_lock.u1_normalizer_digest != u2_contract.u1_normalizer_digest:
        raise ValueError("U2 Development replay authoritative normalizer mismatch")
    if (
        development_lock.replay_authority_schema
        != U2_DEVELOPMENT_REPLAY_AUTHORITY_SCHEMA
    ):
        raise ValueError("U2 Development replay authority schema drifted")

    if supplied_scope_closure.universe_manifest_digest != manifest.digest:
        raise ValueError("U2 Development replay scope universe identity mismatch")
    if supplied_scope_closure.u2_contract_digest != u2_contract.digest:
        raise ValueError("U2 Development replay scope U2 identity mismatch")
    if base_lock.development_scope_closure_digest != supplied_scope_closure.digest:
        raise ValueError("U2 Development replay scope closure digest mismatch")
    if base_lock.evaluation_dataset_digests != _canonical_scope_dataset_mapping(
        supplied_scope_closure
    ):
        raise ValueError("U2 Development replay evaluation dataset mapping mismatch")

    if base_lock.source_tree_digest != source_tree_digest:
        raise ValueError("U2 Development replay source-tree lock identity mismatch")
    if base_lock.lockfile_digest != lockfile_digest:
        raise ValueError("U2 Development replay lockfile identity mismatch")
    if (
        base_lock.evaluation_runtime_identity_digest
        != evaluation_runtime_identity_digest
    ):
        raise ValueError("U2 Development replay runtime identity mismatch")
    if base_lock.development_numeric_open_count != 0:
        raise ValueError(
            "U2 Development replay lock requires zero prior Development opens"
        )
    if base_lock.admission_numeric_open_count != 0:
        raise ValueError("U2 Development replay lock requires zero Admission opens")

    return development_lock


def build_authoritative_universal_trade_rl_u2_development_replay_session(
    *,
    predevelopment_contract: UniversalTradeRLU2PreDevelopmentContract,
    base_lock: UniversalTradeRLU2DevelopmentLock,
    development_lock: UniversalTradeRLU2AuthoritativeDevelopmentLock,
    manifest: UniversalTradeRLUniverseManifest,
    time_partition: UniversalTradeRLU2TimePartition,
    u2_contract: UniversalTradeRLU2Contract,
    u1_contract: UniversalTradeRLU1Contract,
    policy_contract: UniversalTradePolicyContract,
    normalizer: UniversalTradeSequenceNormalizer,
    supplied_scope_closure: UniversalTradeRLU2DevelopmentScopeClosure,
    artifact_locators: Mapping[str, U2EvaluationSourceArtifactLocator],
    source_loader: U2EvaluationSourceArtifactLoader,
    environment_factory: U2ReplayEnvironmentFactory,
    source_tree_digest: str,
    lockfile_digest: str,
    evaluation_runtime_identity_digest: str,
) -> UniversalTradeRLU2DevelopmentReplaySession:
    """Validate the frozen Development lock before any numeric source load."""

    require_universal_trade_rl_u2_development_lock_for_replay(
        predevelopment_contract=predevelopment_contract,
        base_lock=base_lock,
        development_lock=development_lock,
        manifest=manifest,
        u2_contract=u2_contract,
        supplied_scope_closure=supplied_scope_closure,
        source_tree_digest=source_tree_digest,
        lockfile_digest=lockfile_digest,
        evaluation_runtime_identity_digest=evaluation_runtime_identity_digest,
    )
    return _build_universal_trade_rl_u2_development_replay_session_unlocked(
        manifest=manifest,
        time_partition=time_partition,
        u2_contract=u2_contract,
        u1_contract=u1_contract,
        policy_contract=policy_contract,
        normalizer=normalizer,
        supplied_scope_closure=supplied_scope_closure,
        artifact_locators=artifact_locators,
        source_loader=source_loader,
        environment_factory=environment_factory,
    )


def replay_universal_trade_rl_u2_development_scope(
    *,
    session: UniversalTradeRLU2DevelopmentReplaySession,
    request: UniversalTradeRLU2ReplayRequest,
    model: Any | None = None,
) -> UniversalTradeRLU2ReplayEvidence:
    """Replay one scope only when its preregistered common RNG seed is used.

    ``UniversalTradeRLU2DevelopmentReplaySession`` remains the lower-level
    deterministic replay engine used by synthetic diagnostics. U2 Development
    Selection is authoritative only through this boundary, which rejects a
    seed mismatch before environment creation or numeric stepping.
    """

    if not isinstance(session, UniversalTradeRLU2DevelopmentReplaySession):
        raise TypeError("U2 Development replay authority requires a replay session")
    if not isinstance(request, UniversalTradeRLU2ReplayRequest):
        raise TypeError("U2 Development replay authority requires a replay request")

    scope = session.scope(request.scope_digest)
    expected_seed = universal_trade_rl_u2_evaluation_seed(
        u2_contract_digest=session.u2_contract.digest,
        scope_digest=scope.digest,
    )
    if request.evaluation_seed != expected_seed:
        raise ValueError(
            "U2 Development replay evaluation seed must equal the scope-common RNG seed"
        )
    return session.replay(request, model=model)


__all__ = [
    "U2_DEVELOPMENT_REPLAY_AUTHORITY_SCHEMA",
    "build_authoritative_universal_trade_rl_u2_development_replay_session",
    "replay_universal_trade_rl_u2_development_scope",
    "require_universal_trade_rl_u2_development_lock_for_replay",
]
