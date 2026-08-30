# Universal Trade RL Zero-Shot + Optional Transfer Design

## 1. Purpose

Build a universal single-symbol trading RL system that learns reusable market structure from many symbols, can be deployed zero-shot on a previously unseen symbol selected by the user, and can optionally adapt to that symbol through a constrained transfer-learning layer.

The production objective is not portfolio allocation across many symbols at once. At deployment time, the user selects one symbol, assigns a fixed initial capital amount, and the agent trades only that symbol. The primary objective is to maximize after-cost terminal wealth under hard risk constraints.

Formally, for a selected symbol `s` and initial capital `W_0`, the policy objective is:

`maximize E[log(W_T / W_0)]`

subject to:

- hard risk limits are never violated;
- catastrophic drawdown / liquidation paths are fail-closed;
- execution cost and turnover are included in realized economics;
- the policy remains usable without symbol identity features;
- zero-shot deployment on unseen symbols is a first-class capability;
- transfer learning is optional and must outperform the zero-shot baseline before adoption.

## 2. Relationship to Causal Alpha V9/V10/V11

The existing Causal Alpha line is retained as research infrastructure and potential teacher/signal evidence, not as the final product architecture.

- V9 provides evidence that pooled symbol-independent forecasting can contain economic signal, but its tail robustness is insufficient.
- V10/r21 provides authoritative execution lifecycle, risk, artifact, and replay evidence. That execution/risk layer is frozen as the reference rather than repeatedly redesigned.
- V11 isolates policy hypotheses such as neutral expiry, after-cost entry, sign calibration, and sizing. These experiments remain useful for deciding which predictive and policy signals should be exposed to RL.
- The final Trade RL policy is allowed to learn dynamic entry, hold, exit, reduce, and sizing decisions, but only after the underlying state, execution, and evaluation contracts are sufficiently causal and auditable.

The RL project therefore consumes the proven infrastructure from Causal Alpha rather than extending the V10 hierarchy indefinitely.

## 3. Core design principle

The system must learn **market behavior**, not memorize **symbol identity**.

A symbol such as BTCUSDT, ETHUSDT, or a future unseen XYZUSDT should be represented primarily through normalized market state:

- log returns and multi-horizon returns;
- rolling volatility and range statistics;
- normalized trend / momentum state;
- volume and liquidity statistics normalized to local history;
- funding, basis, or derivative context when causally available;
- realized drawdown and risk state;
- current exposure, holding duration, and unrealized/realized PnL state;
- causal forecast outputs, uncertainty, and calibrated after-cost edge where validated.

Absolute symbol names, one-hot symbol IDs, symbol-specific coefficients, and hand-coded symbol exceptions are excluded from the universal base policy.

## 4. Deployment model

The production usage model is:

1. The user selects one tradable symbol.
2. The user assigns a fixed capital amount.
3. A frozen Universal Base RL policy is loaded.
4. The symbol is converted into the same normalized observation contract used during training.
5. The policy runs zero-shot with no symbol-specific retraining required.
6. Risk and execution controls remain authoritative outside the learned policy.
7. If optional transfer learning is enabled, an adapted policy is produced from historical data available before deployment and compared against the frozen zero-shot policy.
8. The adapted policy is accepted only if pre-registered evidence shows improvement without violating robustness or risk constraints.

The user must always be able to choose zero-shot operation even when transfer artifacts exist.

## 5. Universe partitioning

Training data is divided by symbol role as well as by time.

### 5.1 Train universe

The Train universe is intentionally broad. Additional symbols are valuable because they increase coverage of market states rather than merely increasing row count.

Desired diversity includes:

- high / low volatility;
- high / lower liquidity;
- persistent trend / mean-reversion behavior;
- bull / bear / sideways regimes;
- differing funding and basis states;
- mature and newer markets where data quality is sufficient;
- differing reaction speeds to market-wide shocks.

Train symbols may be used for:

- representation / feature statistics that are explicitly fit on training data;
- forecast models;
- RL training trajectories;
- training-time reward estimation;
- causal calibration performed strictly inside training folds.

### 5.2 Development universe

Development symbols are not used to fit the final confirmatory model instance. They are used to develop architecture and policy choices such as:

- observation composition;
- reward design;
- action-space design;
- RL algorithm and stable hyperparameter ranges;
- teacher / auxiliary-signal inclusion;
- transfer-learning mechanism design;
- stopping and deployment criteria.

Repeated inspection of these symbols makes them development evidence, not final generalization evidence.

### 5.3 Unseen Admission universe

A separate set of symbols is reserved for final zero-shot generalization testing.

These symbols are not used to fit:

- base policy parameters;
- forecast parameters;
- feature normalization parameters;
- calibration parameters;
- liquidity / volatility thresholds derived from cross-symbol population statistics;
- reward coefficients;
- architecture or hyperparameter decisions.

Their historical market data may be materialized and integrity-checked operationally, but no statistics derived from their economic outcomes may influence model selection before Admission.

Admission is opened only after the base architecture, model/config identities, and checkpoint selection rules are frozen.

## 6. Time partitioning inside each universe

Symbol holdout alone is insufficient. Every train/development process remains causal in time.

- Training windows may only use labels whose end precedes the decision cutoff.
- Development comparisons use forward chronological blocks.
- Admission symbols use an untouched chronological evaluation interval.
- Transfer-learning evaluation splits the unseen symbol history into adaptation-history and subsequent untouched transfer-evaluation periods.
- No adaptation label may cross the deployment/evaluation boundary.

## 7. Observation architecture

The Universal Base RL observation is divided into explicit domains.

### 7.1 Market state

Normalized price, trend, volatility, volume, liquidity, and derivative-context features.

### 7.2 Forecast state

Optional causal model outputs such as:

- fast expected return;
- slower context forecast;
- ensemble dispersion / uncertainty;
- calibrated probability or lower-confidence after-cost edge.

Forecast state is auxiliary information. The RL policy is not forced to reproduce a hard V10-style hierarchy.

### 7.3 Position state

- current realized weight / exposure;
- position sign;
- holding duration;
- unrealized and realized PnL state represented in scale-free terms;
- distance from local peak / drawdown;
- recent turnover and execution-cost state.

### 7.4 Safety state

The policy may observe risk budget, but hard enforcement remains outside the neural policy. The policy cannot override PreTrade / hard-risk controls.

## 8. Action architecture

The preferred long-term interface separates strategic intent from execution representation.

Strategic intents are conceptually:

- HOLD;
- ENTER_LONG;
- ENTER_SHORT;
- EXIT;
- REDUCE;
- ADD / increase exposure when allowed.

The first RL version may use a bounded continuous target exposure if it integrates better with the existing environment, but the design must preserve a semantic distinction between:

- policy decision;
- target compiler / sizing;
- execution adapter;
- final risk-projected target;
- realized exposure.

This prevents internal target churn from being mistaken for economic policy churn.

## 9. Reward and optimization target

The reward must align with terminal wealth rather than proxy prediction accuracy.

Preferred base reward:

`r_t = delta_log_after_cost_wealth`

Hard-risk events are prevented by the environment. Additional soft penalties may be introduced only when justified and separately versioned, for example:

- drawdown penalty;
- excessive turnover penalty beyond already-realized costs;
- exposure instability penalty.

No penalty should double-count execution fees already included in wealth.

Checkpoint/model selection must be based on forward after-cost economic evidence, not training reward alone.

## 10. Evaluation hierarchy

The evaluation design distinguishes model usefulness, zero-shot generalization, and deployment safety.

### 10.1 Development Selection

Used for comparing architecture and policy candidates. Important metrics include:

- symbol-balanced after-cost wealth;
- median symbol wealth;
- positive-scope fraction;
- lower-tail / CVaR metrics;
- maximum drawdown;
- turnover and cost retention;
- meaningful execution support;
- hard-risk and unexplained execution failures.

A single losing development symbol is diagnostic evidence, not automatically proof that the universal architecture is invalid. However, repeated concentration of losses in a market regime must be treated as a generalization failure rather than averaged away.

### 10.2 Zero-shot Admission

The frozen base policy is applied to unseen symbols without adaptation.

Admission must answer:

1. Does the policy produce meaningful execution on unseen symbols?
2. Is aggregate symbol-balanced after-cost wealth positive?
3. Is the majority of unseen symbol/scope evaluations positive?
4. Is downside bounded enough that the system is deployable with the intended capital/risk contract?
5. Are failures explainable by identified market regimes rather than implementation drift?

No retuning is allowed after viewing an Admission result for the same generation.

### 10.3 Deployment gate

Even a universal policy should not trade obviously invalid inputs. Deployment checks may reject a symbol for non-learning reasons such as:

- insufficient history;
- missing required market fields;
- unacceptable liquidity / execution capability;
- stale or malformed data;
- unsupported market mechanics.

This is different from rejecting a symbol merely because its identity was not seen during training.

## 11. Optional transfer learning

Transfer is secondary to zero-shot capability.

### 11.1 Baseline

Every transfer experiment must retain the frozen zero-shot result as the control.

### 11.2 Adaptation data

Only historical data available before the intended deployment boundary may be used.

### 11.3 Preferred adaptation order

Start with the smallest trainable surface:

1. calibration / normalization adapter if needed;
2. small policy head or adapter module;
3. selected upper policy layers;
4. full fine-tuning only as a later research arm.

The shared representation should remain frozen initially to reduce symbol-specific overfitting and catastrophic forgetting.

### 11.4 Acceptance rule

Transfer is accepted only if it improves subsequent untouched after-cost performance over zero-shot while preserving risk and robustness. If adaptation degrades performance, zero-shot remains the production policy.

## 12. Relationship between symbol count and generalization

Increasing training symbols is encouraged, but symbol count is not itself a success metric.

New symbols should improve coverage of market-state space. Before adding a symbol to Train, record data-quality and regime-coverage reasons. Avoid creating a training universe dominated by many highly correlated symbols that merely duplicate the same market behavior.

Where feasible, monitor diversity using symbol-independent descriptors such as volatility, liquidity, trend persistence, correlation to market beta, and funding/basis characteristics. These descriptors are for dataset coverage and audit; symbol identity still does not enter the base policy.

## 13. Transfer and zero-shot research protocol

Recommended staged program:

### Phase U1: Universal supervised / teacher evidence

Finish V11 policy research and determine which causal forecast/calibration signals carry robust after-cost information worth exposing to RL.

### Phase U2: Universal Base RL

Train RL on the Train universe only. Validate algorithm/reward/action design on Development symbols and chronological development windows.

### Phase U3: Zero-shot symbol Admission

Freeze U2 and open the Unseen Admission universe once. No transfer is used here; this measures the core product capability.

### Phase U4: Optional transfer

For the same conceptual class of unseen symbols, compare frozen zero-shot against pre-deployment adaptation using a separate untouched transfer-evaluation interval or separate held-out symbol cohort.

### Phase U5: Production deployment

User selects one supported market data stream and capital amount. Default to zero-shot base policy; use an adapted policy only when a valid transfer artifact is explicitly available and has passed its own gate.

## 14. Invariants

- No symbol identity feature in the Universal Base RL.
- No future leakage across time or symbol-holdout boundaries.
- Hard risk remains external and authoritative.
- Execution costs are included in realized wealth.
- V10/r21 lifecycle evidence remains the execution reference until a separately approved execution-generation change.
- Admission symbols do not influence architecture or parameter selection before the generation is frozen.
- Transfer cannot overwrite the universal base artifact.
- Zero-shot and adapted results are always independently reproducible and digest-bound.
- Production always knows whether it is running `zero_shot` or a specific adaptation artifact.

## 15. Failure modes and required evidence

### Symbol memorization

Detect by symbol-ID exclusion, unseen-symbol evaluation, and optionally representation probes. Reject hidden per-symbol coefficients or lookup behavior.

### Correlated-universe illusion

A large Train universe may still contain nearly identical market regimes. Track coverage diversity and evaluate genuinely different unseen symbols.

### Development leakage

Any symbol/scope repeatedly inspected during architecture decisions is Development thereafter and cannot be promoted as untouched Admission evidence.

### Transfer overfitting

Require untouched post-adaptation evaluation and direct zero-shot comparison.

### Cash-like false success

Require meaningful execution and exposure support. A policy that stays flat cannot pass merely through low drawdown.

### Cost-insensitive alpha

Reject policies whose gross edge does not survive realistic execution cost.

### Reward hacking

Reconcile reward, step economics, after-cost wealth, execution cost, and lifecycle traces. Final Selection/Admission uses economic observables, not reward alone.

## 16. Alternatives considered

### Zero-shot only

Simplest and strongest generalization claim, but may leave useful symbol-specific adaptation value unused. Kept as the mandatory baseline rather than the only production mode.

### Transfer-first per symbol

Could improve familiar symbols but weakens the core universal capability and risks turning the system into a collection of symbol-specific models. Rejected as the primary architecture.

### Universal base + optional transfer

Chosen approach. It preserves a genuinely reusable base policy while allowing adaptation when evidence proves it adds value.

## 17. Acceptance criteria for the architecture

The architecture is considered implemented, not economically validated, when all of the following are true:

1. Train, Development, and Unseen Admission symbol roles are immutable and artifact-bound.
2. Base-policy training cannot access Admission-derived fit/calibration statistics.
3. The base observation contains no symbol identity and is scale-normalized.
4. The RL reward reconciles to after-cost wealth.
5. Policy intent, execution target, risk-projected target, and realized exposure remain separately traceable.
6. A frozen base policy can be evaluated on an unseen symbol without retraining.
7. Optional transfer creates a separate adaptation artifact without mutating the base policy.
8. Zero-shot vs adapted outcomes can be compared on untouched future data.
9. Hard-risk violations and unexplained execution failures remain zero in admissible runs.
10. Flat/no-execution policies cannot pass economic gates.

Economic success is a later evidence claim, not implied by implementation completion.

## 18. Decision log

1. Production operates one user-selected symbol with fixed starting capital, not a simultaneous multi-symbol portfolio.
2. The Universal Base RL learns symbol-independent market structure.
3. Unseen-symbol zero-shot capability is the primary product behavior.
4. Transfer learning is optional and subordinate to the zero-shot baseline.
5. Training symbols are expanded to improve regime coverage, not to memorize identities.
6. Development and unseen Admission symbols are separated from Train.
7. Admission is a one-way confirmatory boundary after architecture/config freeze.
8. Causal Alpha V11 remains policy/teacher research; V10/r21 execution/risk evidence remains frozen infrastructure.
9. Final optimization target is after-cost wealth under hard safety constraints.
