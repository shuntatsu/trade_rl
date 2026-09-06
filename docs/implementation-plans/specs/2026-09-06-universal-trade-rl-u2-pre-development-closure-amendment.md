# Universal Trade RL U2 Pre-Development Closure Amendment

Status: **Normative U2 V1 amendment**  
Production: **NO-GO**  
Admission: **SEALED**  
Real U2 training: **NO-GO until the real pre-development contract is frozen**  
Development numeric evaluation: **NOT OPENED**

This amendment closes the remaining research-degree-of-freedom surface before any real U2 training or Development numeric outcome is consumed. It supplements the U2 Base PPO preregistration, the timeout/robustness amendment, the source/FIT/routing identity amendment, and the 2026-09-05 Development replay amendments. U0/U1 economics, PPO hyperparameters, economic thresholds, and Admission semantics are unchanged.

---

## 1. Objective

Before real training, freeze a content-addressed **pre-development contract** that fixes:

- minimum research role cardinality;
- the Development common-random-number rule;
- exact Selection metric formulas;
- exact cross-seed bootstrap panel reduction;
- the no-exact-resume rule;
- required training exposure evidence.

After all three exact-final checkpoints exist, but before any Development numeric source is opened, freeze a second content-addressed **Development lock** binding the exact checkpoints, evaluation scope/data identities, runtime/source identities, and the pre-development contract.

Changing any bound semantic after Development is opened requires a new U2 generation. It is not a same-generation repair or tuning opportunity.

---

## 2. Non-goals

This amendment does not:

- change U0 generic universe semantics;
- change U1 observation/action/reward/risk/execution/accounting;
- change the U2 60/10/10/20 partition;
- change PPO architecture, hyperparameters, seeds, or fixed timestep budget;
- change existing economic threshold values;
- open Development or Admission numeric data;
- authorize Production;
- claim exact mid-episode or bitwise PPO resume.

---

## 3. U2 research role cardinality

The generic U0 universe contract remains intentionally reusable and requires only non-empty roles. U2 V1 adds a stricter experiment-specific gate:

```text
Train       >= 9 symbols
Development >= 3 symbols
Admission   >= 3 symbols
```

These counts are checked from the frozen U0 manifest metadata. A smaller real role set is a technical/research NO-GO for this U2 generation; it must not be repaired by reading Development/Admission outcomes and reallocating symbols.

Synthetic unit tests may continue to use smaller manifests when testing lower-level U0/U1 contracts, but a real U2 pre-development contract cannot be built from them.

---

## 4. Development common random numbers

Training seed and evaluation/execution RNG are different experimental axes.

U2 V1 therefore supersedes the earlier rule `evaluation_seed = candidate training seed`.

For every canonical Development scope, derive one evaluation seed from immutable scope identity only:

```text
seed_material = {
  schema_version: "universal_trade_rl_u2_evaluation_crn_v1",
  u2_contract_digest,
  scope_digest,
  allowed_evaluation_seeds: (0, 1, 2)
}

index = int(content_digest(seed_material)[0:8], 16) % 3
evaluation_seed = (0, 1, 2)[index]
```

Consequences:

- training seed 0/1/2 candidates use the same evaluation seed for one scope;
- cash / constant-long / constant-short use that same scope seed;
- the seed is independent of candidate quality and numeric outcomes;
- changing a checkpoint cannot change evaluation RNG;
- changing the scope identity changes the seed deterministically.

This is a common-random-number design for paired comparisons. Execution-RNG stress testing is a separate future experiment and is not part of U2 V1 Selection.

---

## 5. Exact Selection metric contract

One leaf is identified by:

```text
(training_seed, cell, concrete_symbol, tile_identity)
```

All economic values come from the maintained U1 replay of the same action/risk/execution path. Gross and net returns are two views of that same replay; U2 must not rerun a separate cost-free environment to manufacture gross economics.

### 5.1 Leaf log growth

For replay simple returns `r_t`:

```text
leaf_net_log_growth   = sum(log1p(net_simple_return_t))
leaf_gross_log_growth = sum(log1p(gross_simple_return_t))
leaf_net_wealth       = exp(leaf_net_log_growth)
leaf_gross_wealth     = exp(leaf_gross_log_growth)
```

Every return must be finite and greater than `-1`.

### 5.2 Per-symbol and symbol-balanced wealth

For a selected cell/scope grouping:

```text
symbol_net_log_growth   = sum(leaf_net_log_growth for the symbol)
symbol_gross_log_growth = sum(leaf_gross_log_growth for the symbol)

symbol_net_wealth   = exp(symbol_net_log_growth)
symbol_gross_wealth = exp(symbol_gross_log_growth)

symbol_balanced_net_wealth   = exp(mean(symbol_net_log_growth))
symbol_balanced_gross_wealth = exp(mean(symbol_gross_log_growth))
```

Each concrete symbol has one equal vote regardless of row count.

### 5.3 Median / minimum symbol wealth

```text
median_symbol_net_wealth  = median(symbol_net_wealth)
minimum_symbol_net_wealth = min(symbol_net_wealth)
```

### 5.4 Positive scope fraction

```text
positive_net_scope_fraction = mean(leaf_net_log_growth > 0)
```

Zero is not positive.

### 5.5 CVaR10

```text
N = number of mandatory leaves in the grouping
K = max(1, ceil(0.10 * N))
scope_net_return_cvar10 = mean(sorted(leaf_net_log_growth)[0:K])
```

### 5.6 Turnover

For one leaf:

```text
turnover_per_day = turnover_total / (decision_count * 0.25 / 24)
```

The p95 gate is:

```text
np.quantile(turnover_per_day_values, 0.95, method="linear")
```

### 5.7 Meaningful execution

For U2 V1 one leaf has meaningful execution iff:

```text
executed_change_count > 0
OR turnover_total > 1e-6
```

The existing gate remains `meaningful_execution_symbol_fraction_required = 1.0`: every symbol in the mandatory grouping must have at least one meaningful-execution leaf.

### 5.8 Positive gross log-growth retention

Only when symbol-balanced gross log growth is positive:

```text
retention = symbol_balanced_net_log_growth / symbol_balanced_gross_log_growth
```

The existing threshold remains `>= 0.50`. A non-positive gross-growth grouping does not obtain a free retention pass; it already fails the gross-wealth `> 1` gate.

---

## 6. Cross-seed D bootstrap contract

The existing one-dimensional moving-block bootstrap must not be fed an arbitrary flattening of `(seed, symbol, time)` rows.

For D1 and D2 separately:

1. compute paired candidate-minus-cash **net log excess** at each decision timestamp;
2. at each timestamp, take the equal-weight mean across Development symbols;
3. at each timestamp, take the median across the three training seeds;
4. preserve chronological order;
5. run the maintained moving-block mean bootstrap on that one-dimensional temporal series.

For D1+D2 aggregate, D1 and D2 remain separate temporal segments. Bootstrap blocks may wrap inside one segment according to the maintained moving-block implementation but **must not cross the D1/D2 boundary**.

Frozen bootstrap parameters:

```text
confidence_level = 0.95
resamples        = 2000
bootstrap_seed   = 0
block_length     = ceil(sqrt(segment_length)), capped to segment length
quantile_method  = linear
pass condition   = lower 95% CI > 0
```

Candidate and cash series must have identical timestamp/symbol/scope closure before any bootstrap calculation.

---

## 7. Resume semantics

U2 V1 does not claim exact mid-episode or bitwise PPO resume.

Normative semantics:

```text
exact_mid_episode_resume_supported = false
restart_from_timestep_zero_required = true
intermediate_checkpoint_role        = recovery_debug_evidence_only
selection_checkpoint_rule           = exact final fixed-budget checkpoint only
```

If a real training member is interrupted, the Selection-authoritative member must restart from timestep zero with the same frozen generation and seed. A failed economic result is not an interruption and cannot be retried to obtain a better seed outcome.

Any older wording such as `deterministic_resumable=true` is not authority for exact optimizer/vector-environment trajectory continuation and is superseded for U2 V1 scientific claims by this amendment.

---

## 8. Training exposure evidence

Balanced routing guarantees exact equality only within complete environment-local routing cycles. Fixed-budget training may end inside an episode or routing cycle.

For every real training seed, evidence must record at least:

```text
worker_index
concrete_symbol
completed_episode_count
decision_step_count
partial_final_episode_step_count
routing_cycle_count
```

No post-hoc sample weighting or resampling is allowed to repair exposure imbalance. The evidence is diagnostic/interpretive, but absence of the evidence is a technical NO-GO for Development opening.

---

## 9. Pre-development contract artifact

Before real PPO training, materialize a canonical artifact binding at least:

```text
U0 universe manifest digest
U2 training contract digest
role cardinality contract
evaluation CRN contract
Selection metric contract
bootstrap panel contract
resume contract
training exposure evidence contract
Production = NO-GO
Admission = SEALED
```

Construction is metadata-only and performs zero market numeric source opens.

---

## 10. Development lock artifact

After exact-final checkpoints for seeds `(0,1,2)` exist, but before any Development numeric source open, materialize a canonical Development lock binding at least:

```text
pre-development contract digest
U0 universe manifest digest
U1 contract digest
U1 normalizer digest
U2 contract digest
exact seed->final-checkpoint digest mapping
Development scope-closure digest
exact common-view evaluation dataset digests
source-tree digest
lockfile digest
evaluation runtime identity digest
Development numeric open count = 0
Admission numeric open count = 0
Admission status = SEALED
Production status = NO-GO
```

The mapping order is canonical. Missing/extra/reordered seed entries, duplicate symbols, wrong scope identity, or non-zero pre-lock numeric-open counts fail closed.

After Development is opened, changing any field above requires a new U2 generation.

The only supported public U2 V1 numeric Development session entry is the lock-bound `build_authoritative_universal_trade_rl_u2_development_replay_session(...)`. The unlocked lower builder is an internal synthetic/integration-test helper, is not exported through `universal_trade_rl_u2_replay.__all__`, and is not an authorized real-Development API. The authoritative entry must validate the exact base lock, authoritative lock, scope closure, evaluation dataset mapping, source-tree digest, lockfile digest, evaluation-runtime identity, and zero prior Development/Admission opens before the first source-loader call.

---

## 11. Acceptance Criteria

1. Real U2 pre-development closure rejects Train < 9, Development < 3, or Admission < 3 before numeric access.
2. Evaluation seed is a deterministic function of U2 contract + scope only and is independent of training seed/checkpoint.
3. Candidate and all baselines in one scope are required to use the same derived evaluation seed.
4. Selection metric formulas are represented by a canonical machine contract whose digest changes if any formula/tolerance/quantile rule changes.
5. Bootstrap contract first collapses symbols equally, then training seeds by median, preserves time order, and prevents D1/D2 cross-boundary blocks.
6. Exact-resume claims are explicitly disabled for U2 V1.
7. Training exposure evidence schema is fixed before real training.
8. Development lock requires exact final checkpoints for seeds `(0,1,2)` and exact canonical evaluation dataset identities.
9. Development lock can be built only while Development and Admission numeric open counts are both zero.
10. Existing U0/U1 economics, U2 PPO recipe, economic threshold values, time partition, and Admission firewall remain unchanged.
11. Targeted, falsification, integration, static/type/architecture, full-suite/build, and exact-head CI gates pass on one final HEAD.
12. Actual numeric Development session construction requires the exact authoritative Development lock and rejects lock/scope/dataset/source-tree/lockfile/runtime drift before any source-loader call; no unlocked builder is part of the supported public U2 V1 API.
13. Replay evidence stores same-path gross and net simple-return series from one U1 execution path, and both series independently reconcile to their bound wealth ratios; gross economics are never obtained by rerunning a separate cost-free environment.

---

## 12. Quality Gate

Development numeric evaluation remains **NOT OPENED** until:

```text
real U0/U1 freeze
+ fresh strong-stack verification
+ real U2 pre-development contract freeze
+ exact seed 0/1/2 final checkpoint closure
+ training exposure evidence closure
+ Development lock publication
+ exact-head software Quality Gate
```

all succeed.

Admission remains **SEALED** and Production remains **NO-GO**.