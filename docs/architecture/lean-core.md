# Lean core architecture

## Purpose

Trade RL is a research system for testing one symbol-agnostic long/short strategy across multiple markets under causal data, one execution/accounting ledger, hard risk constraints, and unused-data evaluation. The maintained path is intentionally small; adding infrastructure or model families is not a substitute for stronger evidence.

## Runtime and research flow

```text
point-in-time data
        |
        v
MarketDataset + immutable dataset artifact
        |
        +--------------------+
        |                    |
        v                    v
symbol-agnostic strategy   hard risk / feasibility
        |                    |
        +---------+----------+
                  v
           MarketExecutor
                  |
                  v
              BookState
                  |
                  v
       independent evaluation replay
                  |
                  v
      immutable metrics + raw returns
```

Data supplies observable state. Strategy supplies an economic intent. Risk/execution may constrain what is feasible. `MarketExecutor` and `BookState` are the single execution/accounting authority. Evaluation consumes the same frozen strategy and lower-layer contracts; it does not redefine economics.

## Strategy contract

A strategy uses only state observable at the decision time and returns one logical intent:

```text
SHORT
FLAT
LONG
```

When the intent remains LONG or remains SHORT, the maintained behavior is **quantity hold** rather than automatic weight rebalancing every decision. A new target quantity is chosen when intent changes; hard risk may subsequently de-risk an existing position when required.

### Responsibility split

Strategy owns economic decisions such as entry, hold, exit, reversal, and signal/forecast interpretation.

Risk and execution own feasibility and safety, including gross/absolute exposure limits, leverage, margin/insolvency, emergency drawdown handling, liquidity/participation capacity, minimum notional, tick/lot constraints, inactive or untradable state, and fail-closed invalid inputs. Do not duplicate the same entry rule in both layers.

## Universal fit and evaluation contract

The initial learned candidates are universal rather than one model per symbol:

- no symbol ID feature or symbol-specific embedding/coefficient in the initial model;
- one feature schema and one fit cutoff;
- supervised fit data comes only from `fit_symbol_names`;
- Ridge and LightGBM equalize total sample weight by fit symbol so row count alone does not let one symbol dominate;
- PPO training episodes round-robin the symbols in `fit_symbol_names` while each episode trades one active symbol;
- evaluation scope is separate from fit scope;
- the same frozen model/policy/strategy is replayed **independently** for every evaluation symbol.

Per-symbol evidence is authoritative. Aggregate P&L must not hide a losing symbol behind gains elsewhere.

## Causality and data contract

At minimum:

```text
feature_available_time <= decision_time
label_end_time < fit_cutoff
```

Normalization, scaling, imputation, feature selection, or threshold selection must not use future information. Development/final future data must not feed backward into model selection. Dense bar rows are not automatically independent statistical samples, and correlated symbols exposed to the same calendar shock are not fully independent observations.

Point-in-time funding, basis, mark/index, volume, and other economic fields are usable only when their information availability is verified.

## Execution and accounting contract

`MarketExecutor` + `BookState` are the P&L source of truth. The maintained invariants include:

- submission and fill are separate events;
- fees are charged once at realized fill;
- spread and impact are not double-counted;
- partial fills update positions using realized fill quantity;
- funding is applied once for the applicable timestamp, sign, and quantity;
- borrow, mark-to-market, and liquidation are distinct accounting channels;
- terminal mark-to-market is not silently treated as a forced close;
- execution fields carried by the dataset remain active even when an additional experiment overlay is zero.

OHLCV does not prove queue position or hidden liquidity, so the system must not claim those microstructure details without evidence.

## Independent replay and evidence

Evaluation replays the same frozen strategy independently for each selected symbol through the same risk, execution, and accounting path. Each symbol × strategy result retains metrics and raw interval returns needed for paired or block-based statistical analysis.

Immutable result publication prevents a later rerun from silently overwriting the evidence chain. Research conclusions are based on unused-data evidence and symbol-level robustness, not on whether the architecture can produce a successful run.
