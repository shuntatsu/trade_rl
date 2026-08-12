# Universal Train-Only Causal Alpha Teacher Design

## Objective

Replace the rejected hindsight-Oracle and fixed-trend behavior-cloning teachers
with a deterministic, train-only-fitted causal alpha teacher. The teacher must
produce economically admissible target-weight paths before PPO, Lagrangian PPO,
or Discounted Lagrangian PPO is allowed to start.

The change does not alter the pure-growth reward:

`reward_t = 100 * log(net_equity_after / net_equity_before)`

Fees, spread, impact, funding, and borrow costs remain represented exactly once
through net equity. Baseline underperformance and drawdown remain separate gate,
risk, and constraint concerns rather than new scalar reward penalties.

## Approved Direction

The approved approach is a pooled supervised alpha model fitted only on the nine
training symbols. It predicts forward returns from the existing causal
multi-timeframe feature payload and converts those predictions into a stateful,
turnover-aware target-weight path.

There is no mandatory or minimum holding period. A target may reverse at the next
decision when evidence is strong enough. Churn control instead uses entry/exit
hysteresis, a target-weight no-trade band, and a maximum target-weight change per
decision. These controls shape the teacher's action path; they do not add a
second transaction-cost penalty to reward.

## Data and Leakage Boundary

The immutable Universal runtime manifest remains authoritative:

- PostgreSQL-backed Binance data materialized into verified artifacts;
- causal 15m, 1h, 4h, and 1d features;
- nine train, three validation, and three test symbols;
- the fixed research interval and manifest digests; and
- the existing 720h finite-horizon environment and execution contract.

Only train-symbol datasets may enter model fitting, preprocessing-statistic
fitting, teacher configuration selection, BC, or critic warm-start. Validation
and test symbols remain untouched until their existing downstream evaluation
stages.

Teacher episodes use a chronological per-symbol split. For each of the nine
training symbols, complete earlier episodes are used for teacher selection and
BC, and that symbol's latest complete 720h episode is reserved as its causal BC
holdout. The resulting holdout set therefore contains exactly one untouched
complete episode per train symbol. An episode's predictor may use only labels
whose realization ends strictly before that episode starts. Predictions carry an
explicit knowledge cutoff, and any prediction whose cutoff is not earlier than
its decision index is rejected.

The signal fitter does not reuse the full-range Universal observation normalizer
for its regression inputs. Each expanding fit computes its own finite feature
location and scale from the allowed prefix. This prevents later distribution
statistics from entering an earlier prediction.

## Supervised Signal Model

### Inputs

The model consumes the existing target-local feature vector from each
single-symbol Universal dataset. These features already contain the maintained
multi-timeframe 15m/1h/4h/1d information and availability channels. A row is
eligible only when:

- the symbol is active and tradable;
- every selected feature is finite and causally available;
- the complete forward label horizon lies within the fit prefix; and
- the decision-to-execution delay can be represented without crossing the fit
  cutoff.

Unavailable values are not future-filled. Constant or unavailable feature
columns in a fit prefix are set to zero after scaling and recorded in the model
artifact.

### Labels

Two gross forward log-return labels are fitted independently:

- 24 hours; and
- 72 hours.

Each label begins at the first executable bar after the decision and ends at the
configured horizon. Labels are training targets only and are never placed in a
policy observation. Execution costs are not subtracted from each label; economic
selection runs the generated target path through the real execution simulator,
which accounts for costs once.

### Estimator

The first implementation uses deterministic pooled ridge regression implemented
with NumPy. It shares coefficients across train symbols and includes the existing
causal instrument descriptor so it can learn stable scale differences without a
ticker identity lookup. Ridge is selected because it is auditable,
dependency-free in the current environment, deterministic, and cheap enough for
repeated expanding fits.

The artifact records feature names, coefficient arrays, fit ranges, label
horizons, ridge strength, scaling statistics, constant masks, sample counts,
generator code/config digests, and knowledge cutoffs. Fitting fails closed on
non-finite arrays, insufficient samples, rank or shape mismatch, or identity
drift.

## Prediction-to-Target Conversion

The two predicted returns are combined by one train-selected member of this
small, declared family:

- 24h prediction only;
- 72h prediction only; or
- equal-weight 24h/72h prediction.

The combined score is mapped to a desired signed exposure with a bounded `tanh`
transform. A stateful controller then applies:

1. an entry threshold for moving out of cash;
2. a lower exit threshold for returning toward cash;
3. sign-change hysteresis, requiring the entry threshold in the new direction;
4. the existing target-weight no-trade band; and
5. a maximum absolute target-weight change per decision.

No clock-based holding lock is applied. If the score crosses the required
threshold, the controller may change or reverse the target immediately. The
result is still projected through the maintained pre-trade and portfolio-risk
controls by the normal environment path.

The target generator starts from each episode contract's declared initial
weight, not an assumed zero position. Its state transition and all submitted,
suppressed, rejected, and executed changes are included in the evidence.

## Train-Only Model and Controller Selection

All hyperparameters come from a bounded, predeclared grid. The grid covers:

- ridge strength;
- 24h/72h/equal horizon combination;
- score scale;
- entry and exit thresholds with `exit < entry`;
- target-weight no-trade band; and
- maximum target-weight change per decision.

Selection uses expanding, chronological fits and complete earlier 720h episodes
from train symbols. Every candidate is replayed with the production execution
model. Ranking is lexicographic:

1. maximize the lower-tail net return across symbol-episodes;
2. maximize mean net return;
3. minimize turnover per day; and
4. minimize total execution cost.

A candidate is not selectable if its gross return is negative on a majority of
train symbol-episodes, if it produces no meaningful trades, or if it violates
the existing risk contract. The complete grid, every candidate metric, and the
selected configuration digest are persisted before the causal holdouts run.

None of the nine per-symbol latest holdout episodes is consulted during
selection. After selection, each holdout prediction path is generated from a
predictor fitted only on data whose labels are fully realized before that
holdout's start, and the selected controller is evaluated exactly once on all
nine untouched holdout episodes.

## Universal Integration

A new explicit `behavior_cloning_teacher` value identifies this path. The name
must describe the fitted causal teacher and must not reuse `trend_baseline` or
`oracle` identity. The Universal teacher factory returns the existing
episode-aligned batch interface so BC, hierarchical labels, critic return
targets, and causal economic evaluation can remain structurally unchanged.

The teacher digest includes:

- dataset, partition, feature-schema, and instrument-context digests;
- chronological episode contracts;
- signal-model and controller configuration;
- fit and knowledge-cutoff ranges;
- prediction/target array digests; and
- the code identity used to generate the artifact.

The three algorithm families reuse the same admitted teacher batch. Algorithm
gamma may still affect critic targets, but it cannot change teacher actions or
the BC economic admission decision.

## Evidence and Admission

The existing reconstruction and causal economic gates remain fail closed.
Before RL starts, artifacts must include:

- expanding-fit sample counts and cutoff ranges;
- train and holdout prediction correlation and directional accuracy;
- prediction distributions by symbol and horizon;
- policy/teacher action RMSE, correlation, and direction agreement;
- gross and net return, baseline excess, drawdown, and reward totals;
- turnover per day, absolute target deltas, sign flips, submitted/executed
  changes, trade count, and execution cost; and
- aggregate confidence bounds plus per-symbol results across all nine causal
  holdout episodes.

Admission requires the current BC thresholds, including the causal net-return
lower bound. Gates are not relaxed to admit the new teacher. In addition, the
teacher itself must have non-negative aggregate gross return and may not depend
on one profitable symbol while most train-symbol holdouts lose gross value.

If the teacher fails, critic warm-start and PPO updates remain zero. The failure
generation, selected grid record, fitted artifact, logs, and Docker state are
preserved and summarized in `report/` before another hypothesis is tested.

## Testing Strategy

Unit tests prove:

- forward labels start at the first executable bar and end at the exact horizon;
- no fit sample or scaling statistic crosses a prediction knowledge cutoff;
- unavailable and constant features are handled deterministically;
- pooled ridge fitting and serialization are byte-stable;
- the controller implements entry/exit/sign hysteresis without a holding lock;
- no-trade and target-delta limits reduce submitted target changes;
- non-zero initial episode weights are respected; and
- all digests change when a relevant model, controller, or data identity changes.

Integration tests prove:

- validation and test symbols never reach signal fitting or teacher selection;
- the chronological split reserves exactly one latest complete holdout episode
  for each of the nine train symbols and excludes all nine from selection;
- the three algorithms share identical teacher actions;
- BC and critic warm-start consume the new batch through the maintained generic
  instrument surface; and
- a failed economic holdout performs zero PPO updates.

Execution validation proceeds in this order:

1. targeted unit and integration tests;
2. Ruff and mypy on all changed Python modules;
3. a train-only counterfactual/economic diagnostic;
4. CUDA BC causal holdout admission;
5. deterministic reproduction of any passing admission;
6. PPO/Lagrangian/discounted three-update economic smoke; and
7. only after all gates pass, canonical three algorithms by three seeds at
   524,288 timesteps followed by the existing audits and final report.

## Non-Goals

- Adding transaction-cost penalties to the scalar reward.
- Choosing parameters from the causal holdout, validation symbols, or test
  symbols.
- Enforcing a minimum 24h or any other clock-based holding period.
- Replacing real PostgreSQL-backed multi-timeframe data with a simplified path.
- Treating a profitable teacher or completed training run as sealed research
  success.
- Weakening economic, risk, reproducibility, or artifact-identity gates.
