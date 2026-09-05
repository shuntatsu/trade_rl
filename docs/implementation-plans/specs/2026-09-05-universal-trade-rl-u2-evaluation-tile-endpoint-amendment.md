# Universal Trade RL U2 Evaluation Tile Endpoint Amendment

Status: **Normative U2 V1 amendment**  
Production: **NO-GO**  
Admission: **SEALED**  
Real U2 training: **NO-GO**  
Development numeric evaluation: **NOT OPENED**

This amendment is written before any U2 Development numeric evaluation, Admission numeric access, or economic result. It resolves only the boundary semantics between the preregistered U2 evaluation tiles and the maintained U1 rollout/execution runtime.

It does **not** change the 60/10/10/20 time partition, U2 time-partition artifact schema or payload, PPO configuration, checkpoint rules, economic thresholds, U1 reward/risk/execution semantics, or Admission authorization rules.

---

## 1. Why this amendment exists

The U2 time partition preregisters each 720-hour evaluation tile as a half-open bar range:

```text
[S, E)
E - S = 2,880 15-minute bars
```

Implementation-time falsification review found that this range cannot be passed directly to the maintained evaluator without defining what those 2,880 bars mean.

The maintained `evaluate_action_path()` contract is:

```text
expected_count = stop - start - 1
```

and it resets the environment with:

```text
start_idx    = start
episode_bars = expected_count
```

The maintained U1 stateful executor processes one decision from `previous_index` into `processing_index = previous_index + 1`. Therefore one decision consumes the next bar as its economic outcome.

Consequently:

- evaluator range `(S, E)` executes only `E - S - 1 = 2,879` decisions;
- evaluator range `(S, E + 1)` executes 2,880 decisions but allows the final outcome to escape the preregistered tile and, at a window boundary, enter the next time partition.

Both interpretations violate U2 V1.

---

## 2. Normative tile meaning

For U2 V1, a preregistered tile `[S, E)` denotes the **economic outcome bars**, not the initial-state row set.

The exact contract is:

```text
tile outcome range          = [S, E)
decision_count              = E - S = 2,880
initial state bar           = S - 1
evaluate_action_path range  = (S - 1, E)
outcome bars consumed       = S, S+1, ..., E-1
```

Equivalently:

```text
E - (S - 1) - 1 = E - S = 2,880 decisions
```

The evaluator must obtain this mapping from the maintained U2 tile contract. Callers must not independently reconstruct the `-1` offset.

---

## 3. Boundary isolation

For adjacent preregistered tiles:

```text
left  outcome tile = [S, E)
right outcome tile = [E, F)
```

their economic outcome sets are disjoint.

The right tile uses `E - 1` as its initial-state bar. That row is the left tile's final outcome and is allowed to be reused only as already-known state/history. It is **not** counted again as a right-tile economic outcome.

Thus:

```text
left outcomes  = S ... E-1
right outcomes = E ... F-1
```

with no overlapping outcome bar.

---

## 4. Cross-window history is allowed, cross-window outcomes are not

For the first tile of `DEV_FUTURE_1`, `DEV_FUTURE_2`, or `ADMISSION_FUTURE`, the initial-state bar `S - 1` may belong to the immediately preceding time window.

This is permitted because the original U2 preregistration already allows observation lookback to extend before a partition boundary when that history is available at decision time.

The invariant is stricter for economic outcomes:

- C/D outcome bars must not enter FIT;
- D1 outcomes must not enter D2;
- D2 outcomes must not enter Admission;
- Admission outcomes must remain inside `ADMISSION_FUTURE`;
- no evaluator may obtain an extra bar after `E - 1` to complete a 720-hour episode.

---

## 5. Candidate and baseline scope equality

Candidate, cash, constant-long, and constant-short replays for one U2 scope must use the **exact same**:

```text
tile identity
evaluation range = (S - 1, E)
decision_count   = E - S
initial state    = cash
U1 Risk / Execution / Accounting identity
source dataset identity
```

A baseline replay using `(S, E)`, `(S, E + 1)`, or any independently generated endpoint is not same-scope evidence and must be rejected.

---

## 6. Artifact identity is unchanged

This amendment adds no serialized field to `UniversalTradeRLU2EpisodeTile` and changes no existing time-partition payload.

The following remain unchanged:

- `UNIVERSAL_TRADE_RL_U2_TIME_PARTITION_SCHEMA`;
- time-window boundaries;
- tile `start_bar_index` / `stop_bar_index_exclusive`;
- tile timestamps;
- time-partition artifact digest for the same preregistered metadata.

`decision_count` and `evaluation_range` are derived runtime properties of the existing tile identity.

---

## 7. Required verification

Before Development numeric evaluation is allowed, one exact HEAD must prove at minimum:

1. every U2 evaluation tile has `decision_count == 2,880`;
2. every tile maps to evaluator range `(start_bar_index - 1, stop_bar_index_exclusive)`;
3. `stop - start - 1 == 2,880` for that evaluator range;
4. consumed outcome indices equal the tile's exact `[S, E)` range;
5. adjacent tiles have disjoint outcome ranges;
6. time-partition payload and digest round-trip unchanged;
7. actual U2 evaluator uses the tile-provided range rather than reconstructing endpoints;
8. candidate and all mandatory baselines use the identical range;
9. related U2 contract/integration tests, static checks, architecture checks, full suite, package/build, and exact-head CI pass.

Development remains unopened until Task 7 implements and verifies the actual evaluator path. Admission remains sealed and Production remains NO-GO.
