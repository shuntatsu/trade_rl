# Universal Trade RL U2 Robustness / Timeout Amendment

Status: **Normative U2 V1 amendment**  
Production: **NO-GO**  
Admission: **SEALED**

This amendment is written before any U2 runtime/training implementation or Development result. It supersedes the affected parts of:

- `2026-09-03-universal-trade-rl-u2-base-ppo-selection-design.md`
- `2026-09-03-universal-trade-rl-u2-base-ppo-selection.md`

The reason for the amendment is independent falsification review. Three loopholes were found in the preregistration itself:

1. seed 1/2 could hide one losing symbol behind winning symbols while still passing a seed-balanced aggregate;
2. U1 external truncation could be accidentally treated as an economic terminal by the PPO return target;
3. execution-only training settings such as CUDA/vector mode were identity-bound in principle but not explicitly preregistered.

A fourth ordering ambiguity is also closed: U2 time metadata must exist before the real U1 normalizer can use `FIT end` as its cutoff, while U2 *training* still starts only after U1 is frozen.

---

## 1. U0 → time partition → U1 → U2 ordering

The real production-candidate generation must be constructed in this order:

```text
1. freeze real U0 universe/source identity
2. materialize U2 time-partition metadata from U0 metadata only
3. fit/materialize real U1 normalizer with cutoff == U2 FIT end
4. materialize real U1 contract
5. create Train-only RL_TRAINING provenance with cutoff == U2 FIT end
6. materialize U2 training contract / BASE_TRAINING identity
7. only then start U2 PPO training
```

Creating `u2_time_partition.json` in step 2 is **preregistration metadata**, not U2 training and not Development access. Its builder accepts U0 timestamps/count metadata only and must not read numeric market arrays, including Admission arrays.

Therefore the maintained statements:

```text
real U1 artifacts must be frozen before U2 training
```

and:

```text
U2 FIT end determines the real U1 normalization cutoff
```

are not contradictory.

---

## 2. Fixed execution/runtime settings for U2 V1

The complete resolved `ResidualTrainingConfig.digest_payload()` is part of the U2 training-contract identity. In addition to the algorithmic values in the base preregistration, U2 V1 fixes:

```text
policy                    = MultiInputPolicy
observation_encoder       = hierarchical_sequence_v2
policy_actor_head          = shared_target_v1

device                    = cuda
cuda_runtime_mode         = deterministic
vector_environment_mode   = in_process

sequence_compile           = false
sequence_compile_mode      = reduce-overhead
sequence_transfer_mode     = synchronous
hierarchical_gate_temperature = 1.0

checkpoint_interval_steps = 32768
max_checkpoints            = 8
max_policy_parameters      = 12000000
max_rollout_buffer_bytes   = 805306368

tensorboard_enabled        = true
tensorboard_log_interval   = 1
```

`vector_environment_mode=in_process` is chosen for U2 V1 before economic results because it avoids making the first U1 Base-PPO experiment depend on the historical compact subprocess observation transport. It also gives a single-process environment scheduling order for easier exact falsification. This is a mechanics decision, not a Development-tuned parameter.

If the real training host cannot satisfy the deterministic CUDA or resource contract, the result is **technical NO-GO**. It must not silently fall back to `performance`, CPU, fewer environments, subprocess mode, compilation, or different transfer mode under the same U2 generation.

### 2.1 Complete config, no hidden defaults

The U2 contract builder must construct one explicit `ResidualTrainingConfig`, serialize the complete resolved `digest_payload()`, and bind that payload/digest.

All algorithm-inactive fields must equal the maintained inactive defaults accepted by `ResidualTrainingConfig`. A repository default changing later must therefore either:

- leave the explicitly constructed U2 values unchanged, or
- change the U2 training-contract digest and require a new reviewed generation.

No SB3/library default that can affect learning may remain outside the frozen resolved contract.

---

## 3. External truncation is a timeout, not an economic terminal

U1 fixes:

```text
episode_hours = 720
terminated = false at normal horizon
truncated = true at normal horizon
liquidate_on_end = false
finite_horizon_observation = false
```

U2 PPO must preserve the continuing-task interpretation across this artificial sample boundary.

### 3.1 Required vector-adapter contract

At a normal 720h U1 horizon, the SB3-facing transition must expose the maintained equivalent of:

```text
done = true                     # vector API boundary only
TimeLimit.truncated = true
terminal_observation = exact final U1 observation
```

The final observation must retain the same U1 normalizer / policy-state semantics as any other observation.

### 3.2 Required PPO bootstrap contract

The value target must bootstrap from the final observation **exactly once** for timeout truncation.

Conceptually:

```text
training_target_reward = environment_reward + gamma * V(terminal_observation)
```

or an algebraically equivalent maintained GAE/timeout implementation.

This bootstrap correction is **not U1 reward shaping**. The environment/economic reward remains exactly:

```text
100 * log(W_after / W_before)
```

and economic telemetry/Selection must store the unmodified environment reward and realized wealth. The timeout value term exists only inside the training return/critic target.

### 3.3 Forbidden timeout behavior

Reject any implementation that:

- treats normal `truncated=True` as `terminated=True` for critic target purposes;
- sets timeout terminal value to zero;
- bootstraps twice;
- adds terminal liquidation or terminal PnL;
- includes the bootstrap value term in after-cost economic return;
- loses `terminal_observation` or `TimeLimit.truncated` through an info/VecEnv adapter.

### 3.4 Required falsification test

An integration test must use a controlled/mock value function with known `V_terminal` and prove, at the exact 720h truncation boundary:

1. raw U1 reward equals realized wealth reward;
2. vector info identifies timeout truncation;
3. final observation is the pre-reset terminal observation, not the next episode reset observation;
4. PPO rollout target contains exactly one `gamma * V_terminal` bootstrap;
5. no corresponding wealth/economic bonus exists.

The same test must be run through the actual U2 vectorization mode (`in_process`, `n_envs=8` at production contract; a smaller explicit test fixture may be used for unit speed if the production builder itself remains fixed).

---

## 4. Stronger seed robustness — no losing symbol hidden by winners

The original U2 preregistration required the primary seed 0 to pass all B/C/D core gates, but seed 1/2 robustness only constrained symbol-balanced seed wealth. That was insufficient.

### 4.1 Per-seed D core gate

For **each seed independently** (`0`, `1`, `2`), each of:

- D1
- D2
- D1+D2 aggregate

must pass the full D core gate:

```text
symbol coverage                       = complete
symbol_balanced_gross_wealth          > 1.0
symbol_balanced_net_wealth            > 1.0
median_symbol_net_wealth              >= 1.0
minimum_symbol_net_wealth             >= 1.0
positive_net_scope_fraction           >= 0.50
scope_net_return_cvar_10              >= -0.01
turnover_p95_per_day                  <= 1.0
meaningful_execution_symbol_fraction  = 1.0
hard_risk_violation_count             = 0
unexplained_execution_rejection_count = 0
```

When aggregate gross log growth is positive:

```text
net_log_growth / gross_log_growth >= 0.50
```

is also required for that seed/cell.

Thus a seed with one Development symbol below cash fails even if its other symbols make the balanced average positive.

### 4.2 Cross-seed gate remains additional

After all three seeds independently pass D core, D1, D2, and D1+D2 must also satisfy the cross-seed requirements:

```text
median_seed_symbol_balanced_net_wealth > 1.0
worst_seed_symbol_balanced_net_wealth  >= 1.0
all_seed_hard_risk_violations           = 0
all_seed_turnover_p95_per_day           <= 1.0
paired moving-block bootstrap lower 95% CI of excess vs cash > 0
```

The cross-seed gate does not replace the per-seed/per-symbol gate.

### 4.3 B/C scope across non-primary seeds

B/C1/C2 remain mandatory pass/fail gates for primary seed 0. Seeds 1/2 B/C metrics may be recorded as diagnostic robustness evidence but do not create additional Selection degrees of freedom and cannot replace seed 0.

### 4.4 Required falsification cases

Selection tests must prove rejection when:

- seed 1 has balanced D wealth >1 but one symbol wealth <1;
- seed 2 has excellent D1 but fails D2;
- every seed is individually >cash but cross-seed bootstrap lower CI <=0;
- seed 1 is economically best but seed 0 remains the only Admission candidate;
- seed 0 passes while any one of seed 1/2 fails D core: overall Development Selection fails.

---

## 5. Updated Development Selection AND rule

Development Selection passes only when **all** are true:

1. seed 0 passes B core;
2. seed 0 passes C1 core;
3. seed 0 passes C2 core;
4. seed 0 passes D1 and D2 core;
5. seed 1 passes D1 and D2 core;
6. seed 2 passes D1 and D2 core;
7. every seed passes the D1+D2 aggregate core;
8. D1 cross-seed robustness passes;
9. D2 cross-seed robustness passes;
10. D1+D2 cross-seed robustness passes;
11. timeout/bootstrap contract evidence is valid for the exact training runtime;
12. all identity/source/scope closure checks pass;
13. Admission numeric access count remains zero.

Otherwise:

```text
selected_checkpoint = null
admission_eligible   = false
production_eligible  = false
```

No partial-pass state may authorize Admission.

---

## 6. Updated implementation-plan obligations

The implementation plan is amended as follows.

### Task 3 — U2 contract

Add RED tests for every fixed runtime field in Section 2 and for the complete resolved `ResidualTrainingConfig.digest_payload()`.

### Task 5 — environment/vector contract

Add integration tests that U2 uses `in_process`, contexts remain `None`, and U1 horizon truncation survives the vector adapter as timeout metadata + exact terminal observation.

### Task 6 — PPO assembly

Add a timeout bootstrap oracle proving exactly one terminal-value bootstrap and no economic reward mutation.

### Task 8 — Selection

Replace the weaker seed-lottery test with per-seed D core tests plus the additional cross-seed gate in Section 4.

### Task 11 — synthetic E2E

Include one synthetic case where seed 1 has positive balanced return but one losing symbol. The full Development decision must reject it.

### Task 13 — real execution preflight

Verify the real host supports the exact `cuda + deterministic + in_process` contract before any economic run. Failure is technical NO-GO, not an implicit config amendment.

---

## 7. Quality-gate update

U2 software cannot be called training-ready unless independent review confirms:

- no losing Development symbol can be hidden inside seed aggregation;
- no losing seed can be hidden inside cross-seed aggregation;
- normal U1 horizon truncation receives exactly one critic bootstrap;
- bootstrap terms never enter economic Selection metrics;
- the complete runtime config is explicit and identity-bound;
- U2 time-partition preregistration occurs before real U1 normalization, while real PPO still waits for U1 freeze.

These requirements are part of U2 V1 and must not be relaxed after Development results.
