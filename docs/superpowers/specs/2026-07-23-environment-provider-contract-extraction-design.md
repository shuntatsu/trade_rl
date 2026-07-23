# Environment Provider Contract Extraction Design

Date: 2026-07-23

## Problem

After PR #114, `ResidualMarketEnv.__init__()` delegates deterministic observation-contract construction, but it still owns a second static construction cluster:

- trend-strategy and `MarketInputResolver` reconciliation;
- compatibility wrapping of causal alpha providers;
- alpha enablement and artifact identity resolution;
- static factor-basis validation and copying;
- factor-count inference and consistency checks;
- factor artifact identity resolution;
- provider minimum-index validation and aggregation.

These operations complete before reward, execution, observation, episode, or mutable Gymnasium state is created. They are deterministic for the supplied dataset and provider/config arguments, and their error order is part of the maintained constructor contract.

Keeping this cluster inline makes provider identity policy difficult to review and leaves helper methods on the mutable environment facade even though they do not use episode state.

## Considered approaches

### 1. Extract all remaining constructor wiring

Move providers, risk objects, action specification, reward construction, runtime services, and mutable state into one large environment factory.

Rejected. This would cross unrelated ownership boundaries and make behavior-preserving characterization unnecessarily broad.

### 2. Extract provider resolution plus action and risk configuration

Move provider resolution, portfolio-risk provider selection, action-spec validation, and episode schedule validation together.

Rejected for this PR. Although mostly construction-time behavior, the combined boundary would mix market-input identity with risk and episode policy.

### 3. Extract only the provider contract

Create a typed builder that owns trend/alpha/factor provider resolution, artifact identities, static factor basis, factor count, and provider-derived minimum start index.

Chosen. It is the smallest independently testable seam, removes the three provider helper methods from the facade, and does not alter risk, action, reward, execution, observation, or mutable state.

## Goal

Extract provider resolution into a typed immutable contract while preserving:

- the public `ResidualMarketEnv` constructor signature;
- every exposed environment field and type;
- validation order and exact error strings;
- provider artifact digests;
- static factor-basis copy semantics;
- factor-count inference;
- provider minimum-index contribution;
- environment digest inputs and runtime behavior.

## Chosen boundary

Create `trade_rl.rl.environment_provider_contract` containing:

```python
class AlphaProvider(Protocol):
    @property
    def artifact_digest(self) -> str: ...

    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray: ...


class FactorBasisProvider(Protocol):
    @property
    def artifact_digest(self) -> str: ...

    @property
    def n_factors(self) -> int: ...

    def basis_at(self, dataset: MarketDataset, index: int) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class EnvironmentProviderContract:
    market_input_resolver: MarketInputResolver | None
    trend_strategy: TrendStrategy
    alpha_provider: AlphaProvider | Callable[[MarketDataset, int], np.ndarray] | None
    alpha_enabled: bool
    alpha_contract: AlphaContract
    alpha_artifact_digest: str | None
    static_factor_basis: np.ndarray | None
    factor_basis_provider: FactorBasisProvider | Callable[[MarketDataset, int], np.ndarray] | None
    factor_artifact_digest: str | None
    factor_count: int
    minimum_start_index: int


class EnvironmentProviderContractBuilder:
    def __init__(
        self,
        dataset: MarketDataset,
        *,
        trend_strategy: TrendStrategy | None,
        market_input_resolver: MarketInputResolver | None,
        alpha_provider: AlphaProvider | Callable[[MarketDataset, int], np.ndarray] | None,
        alpha_enabled: bool,
        alpha_artifact_digest: str | None,
        alpha_contract: AlphaContract | None,
        factor_basis: np.ndarray | None,
        factor_basis_provider: FactorBasisProvider | Callable[[MarketDataset, int], np.ndarray] | None,
        factor_artifact_digest: str | None,
        factor_count: int | None,
    ) -> None: ...

    def build(self) -> EnvironmentProviderContract: ...
```

`AlphaProvider` and `FactorBasisProvider` move to this module. `environment.py` imports them so existing imports from `trade_rl.rl.environment` continue to resolve to the same protocol objects.

## Ownership and data flow

The builder receives only constructor arguments related to trend, alpha, and factor providers plus the dataset. It returns all resolved values and the provider-derived minimum start index.

`ResidualMarketEnv.__init__()` performs one builder call, assigns the returned fields, then continues with composer, risk, configuration, action, reward, execution, observation, runtime services, and mutable state exactly as before.

The builder does not import the environment facade, risk models, rewards, simulation, observation assembly, episode services, or mutable book state.

## Behavioral invariants

### Trend and market-input resolution

Preserve:

- explicit `trend_strategy` precedence;
- resolver trend fallback;
- default `TrendStrategy()` fallback;
- compatibility construction of `MarketInputResolver` only when the supplied alpha provider exposes both `predict` and `identity_digest`;
- exact mismatch error: `market_input_resolver trend differs from trend_strategy`.

### Alpha contract

Preserve:

- resolver-owned `alpha_enabled` when a resolver exists;
- constructor `alpha_enabled` otherwise;
- exact missing-provider error: `alpha_enabled requires an alpha_provider`;
- default `AlphaContract()` construction;
- explicit artifact digest precedence;
- provider `artifact_digest`, then provider `identity_digest`, fallback order;
- exact required and SHA-256 validation behavior.

### Factor contract

Preserve:

- `float64` conversion and copied static factor basis;
- exact shape and finite-value errors;
- factor-count inference from a provider `n_factors` integer;
- rejection of bool, non-integer, or negative factor counts;
- static-basis count reconciliation and exact mismatch error;
- deterministic `static_factor_basis_v1` content digest when no explicit/provider digest exists;
- exact required and SHA-256 validation behavior.

### Minimum start index

Preserve:

- trend minimum history as the initial value;
- alpha-provider then factor-provider validation order;
- missing `minimum_index` defaulting to zero;
- rejection of bool, non-integer, negative, or out-of-range indices;
- exact provider-specific error strings;
- maximum of all valid provider minimums.

### Public environment fields

The environment continues to expose identical values/types for:

- `market_input_resolver`;
- `trend_strategy`;
- `alpha_provider`;
- `alpha_enabled`;
- `alpha_contract`;
- `alpha_artifact_digest`;
- `_static_factor_basis`;
- `factor_basis_provider`;
- `factor_artifact_digest`;
- `_minimum_start_index` before later risk/reward/sequence contributions.

## Architecture constraints

- `ResidualMarketEnv.__init__()` must delegate to `EnvironmentProviderContractBuilder`.
- The constructor must not directly construct `MarketInputResolver`, call provider-digest/factor-basis/factor-count helpers, or manually iterate provider minimum indices.
- `_resolve_provider_digest`, `_validated_static_basis`, and `_resolve_factor_count` must no longer exist on `ResidualMarketEnv`.
- The constructor source span must be reduced from 321 lines to at most 270 lines.
- Provider policy must not be duplicated in `environment.py`.
- The new module must not depend on risk, reward, simulation, observation, or environment-runtime modules.

## Testing strategy

1. Add an architecture RED test requiring the new module, typed contract, constructor delegation, helper removal, forbidden low-level provider construction, and a 245-line constructor bound.
2. Add direct characterization tests for default, explicit resolver, compatibility alpha wrapping, artifact digest precedence, static basis copy/validation, factor count inference/reconciliation, and provider minimum indices.
3. Assert every existing error string and relevant validation order.
4. Assert representative `ResidualMarketEnv` fields and environment digests remain stable for equivalent inputs.
5. Run focused environment/provider tests, full static checks, complete pytest/coverage, Import Linter, Ubuntu/Windows compatibility, training-image checks, and PostgreSQL Catalog on the exact PR head.
6. Add a measured critical branch-coverage ratchet only if the exact-head report supports it.

## Non-goals

- No portfolio-risk provider extraction.
- No action-spec, reward, episode schedule, runtime-service, or mutable-state extraction.
- No change to provider interfaces or supported legacy compatibility behavior.
- No observation, action, reward, risk, execution, or training-policy change.
- No production-readiness, profitability, or exchange-realism claim.
- No claim that `AUD-RL-001` is fully resolved.

Production remains `NO-GO`.