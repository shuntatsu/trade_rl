# Environment Portfolio-Risk Contract Extraction Design

## 1. Context

`ResidualMarketEnv.__init__()` still owns a small but independent portfolio-risk input policy after the provider, observation, and runtime-service extractions. It currently:

- creates the default `PortfolioRiskModel`;
- selects `RollingPortfolioRiskInputsProvider` when advanced portfolio-risk limits require causal covariance, beta, or stress inputs;
- validates the provider identity as a SHA-256 digest;
- validates the provider minimum index against the dataset;
- merges that minimum index with the provider-contract minimum index.

This policy is deterministic construction-time work. It does not depend on books, order books, reset state, actions, observations, or executor state.

## 2. Approaches considered

### A. Extract only the portfolio-risk input contract — selected

Add a typed contract and builder dedicated to default model/provider selection, identity validation, and minimum-index aggregation.

Advantages:

- smallest behavior-preserving seam;
- keeps validation order explicit;
- avoids coupling pre-trade and emergency-risk concerns;
- directly removes portfolio-risk input policy from the environment facade;
- independently testable with no Gymnasium mutable state.

### B. Create one broad environment-risk construction builder

Move pre-trade defaults, emergency monitor construction, portfolio-risk selection, and execution-leverage compatibility checks together.

Rejected because those concerns have different inputs and error-order requirements. Combining them would create another mixed-responsibility constructor boundary.

### C. Leave the code inline

Rejected because the block already has a complete input/output contract and is one of the explicitly recorded `AUD-RL-001` construction-density items.

## 3. Maintained boundary

Create `trade_rl/rl/environment_portfolio_risk_contract.py` with:

```python
@dataclass(frozen=True, slots=True)
class EnvironmentPortfolioRiskContract:
    portfolio_risk: PortfolioRiskModel
    inputs_provider: PortfolioRiskInputsProvider | None
    minimum_start_index: int
```

and:

```python
class EnvironmentPortfolioRiskContractBuilder:
    def __init__(
        self,
        dataset: MarketDataset,
        *,
        portfolio_risk: PortfolioRiskModel | None,
        inputs_provider: PortfolioRiskInputsProvider | None,
    ) -> None: ...

    def build(self, *, minimum_start_index: int) -> EnvironmentPortfolioRiskContract: ...
```

The builder receives the dataset only for `n_bars`, receives the optional maintained model/provider, and receives the current minimum index from `EnvironmentProviderContractBuilder`.

## 4. Exact behavior and validation order

`build()` must preserve this order:

1. Use the supplied `PortfolioRiskModel`, or create `PortfolioRiskModel()`.
2. Use the supplied provider.
3. When the model requires advanced inputs and no provider was supplied, create `RollingPortfolioRiskInputsProvider()`.
4. When a provider exists, validate `identity_digest` using `require_sha256(..., field="portfolio_risk_inputs_provider.identity_digest")`.
5. Read `minimum_index` only after the digest succeeds.
6. Reject boolean, non-integer, negative, or `>= dataset.n_bars` minimum indices with `ValueError("portfolio risk inputs minimum_index is invalid")`.
7. Return `max(minimum_start_index, provider.minimum_index)` when a provider exists, otherwise preserve the supplied minimum index.

The builder does not add a new validation for the incoming minimum index. That value is already validated by the earlier provider contract, and adding another check could change exception order.

## 5. Facade integration

`ResidualMarketEnv.__init__()` will invoke the new builder after `EnvironmentProviderContractBuilder` and before config/action/reward construction.

It assigns:

- `self.portfolio_risk` from `contract.portfolio_risk`;
- `self.portfolio_risk_inputs_provider` from `contract.inputs_provider`;
- `self._minimum_start_index` from `contract.minimum_start_index`.

`self.composer` and `self.pre_trade_risk` remain in the facade. The environment no longer imports `RollingPortfolioRiskInputsProvider` or calls `require_sha256` for this concern.

## 6. Tests and permanent controls

Add direct characterization for:

- default model with no provider;
- supplied model/provider identity preservation;
- automatic rolling provider selection for advanced risk;
- no automatic provider for basic risk;
- provider digest failure occurring before `minimum_index` access;
- boolean, non-integer, negative, and out-of-range minimum indices;
- maximum aggregation with the incoming minimum index;
- environment facade integration and digest preservation.

Add an architecture test that:

- requires local ownership of the contract and builder;
- requires one builder invocation in `ResidualMarketEnv.__init__()`;
- prohibits inline `RollingPortfolioRiskInputsProvider()` and portfolio-risk provider digest/minimum-index validation in the facade constructor;
- reduces the constructor source-span limit from 240 to 220 lines, subject to the formatted implementation result.

The extracted module must reach 100% statement and branch coverage and be added to `[tool.trade_rl.critical_coverage.files]` at 100.0.

## 7. Non-goals

This change does not:

- alter `PortfolioRiskModel.constrain()`;
- alter rolling covariance, beta, or stress calculations;
- move pre-trade risk or emergency-risk construction;
- change action, reward, execution, observation, or reset behavior;
- change the public `ResidualMarketEnv` constructor;
- establish exchange realism, profitability, or production readiness.

## 8. Architecture status

`AUD-RL-001` remains `OPEN RISK, FURTHER REDUCED` after this extraction. Remaining constructor density will still include config/action/episode validation, reward/executor construction, and mutable Gymnasium state initialization.

Production remains `NO-GO`.