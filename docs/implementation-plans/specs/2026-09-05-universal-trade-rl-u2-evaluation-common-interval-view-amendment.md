# Universal Trade RL U2 Evaluation Common-Interval View Amendment

Status: **Normative U2 V1 amendment**  
Production: **NO-GO**  
Admission: **SEALED**  
Real U2 training: **NO-GO**  
Development numeric evaluation: **NOT OPENED**

This amendment is written before any U2 Development numeric evaluation, Admission numeric access, or economic result. It resolves the index-space contract between frozen per-symbol source datasets and the common-interval U2 evaluation tiles.

It does not change the U0 universe split, U2 60/10/10/20 time partition, PPO recipe, U1 economics, Selection thresholds, or Admission authorization.

---

## 1. Problem

U2 time-partition bar indices are defined relative to the common decision-clock interval:

```text
common_first = max(symbol.first_timestamp)
common_last  = min(symbol.last_timestamp)
```

Frozen source datasets may start at different timestamps. Therefore a tile index `S` in U2 common coordinates is not necessarily source-row `S` in every symbol's original dataset.

Using U2 tile indices directly against a raw source dataset can silently evaluate the wrong historical interval while all array bounds remain valid.

This is a technical NO-GO failure mode.

---

## 2. Canonical evaluation dataset per symbol

For every Train or Development symbol used by U2 Development evaluation, derive one canonical common-interval `MarketDatasetView` from its exact frozen source dataset.

Given source metadata:

```text
source_first_timestamp_ns
source_row_count
source_dataset_digest
```

and the U2 partition:

```text
common_first_timestamp_ns
common_last_timestamp_ns
common_bar_count
```

compute:

```text
offset_ns = common_first_timestamp_ns - source_first_timestamp_ns

offset_ns >= 0
offset_ns % U2_DECISION_STEP_NS == 0

view_start = offset_ns // U2_DECISION_STEP_NS
view_stop  = view_start + common_bar_count

0 <= view_start < view_stop <= source_row_count
```

The endpoint implied by the source grid must satisfy:

```text
source_first_timestamp_ns + (view_stop - 1) * U2_DECISION_STEP_NS
    == common_last_timestamp_ns
```

The evaluation dataset identity is exactly the maintained `MarketDatasetView` identity:

```text
content_digest({
    "dataset_id": source_dataset_digest,
    "schema_version": DATASET_VIEW_SCHEMA,
    "start": view_start,
    "stop": view_stop,
})
```

---

## 3. Scope identity

Every U2 Development evaluation scope binds both identities:

```text
source_dataset_digest      = exact frozen full-source dataset ID
evaluation_dataset_digest  = exact derived common-interval view ID
```

and also binds:

```text
evaluation_source_start_bar_index
evaluation_source_stop_bar_index_exclusive
```

The scope object must independently recompute the common-view identity from its full-source ID and source-view range. A caller-supplied digest/range mismatch is rejected even if the outer scope artifact digest is recomputed.

---

## 4. Tile coordinate system

After the exact common-interval view is materialized, all U2 tile coordinates are interpreted against that view.

For a preregistered tile `[S, E)`:

```text
common-view outcome bars   = [S, E)
initial-state row           = S - 1
evaluate_action_path range  = (S - 1, E)
decision_count              = E - S = 2,880
```

No per-symbol source offset is applied to the tile after the common view has been created.

This preserves one identical time coordinate system for every Train and Development symbol.

---

## 5. Numeric source loading boundary

Scope construction remains metadata-only.

The later Development source loader must:

1. require locator keys to equal exactly the Train + Development symbols required by the scope closure;
2. reject Admission or extra symbols before any loader call;
3. load each full source at most once;
4. require verified canonical source identity and exact frozen symbol/source metadata;
5. materialize exactly the scope-bound common `MarketDatasetView`;
6. require the materialized dataset ID to equal `evaluation_dataset_digest`;
7. return no raw full source as the evaluation dataset;
8. perform no normalization, fitting, calibration, model update, or threshold estimation.

Admission numeric source opens remain zero.

---

## 6. Locator independence

Filesystem or object-store locator values are operational addresses only.

Moving byte-identical canonical source artifacts must not change:

- scope closure digest;
- source dataset identity;
- common-view identity;
- tile coordinates;
- evaluation range;
- candidate/baseline same-scope identity.

---

## 7. Required verification

Before Development numeric evaluation is allowed, one exact HEAD must prove:

1. different source start timestamps produce different source-local common-view ranges where required;
2. every scope binds the exact maintained `MarketDatasetView` identity;
3. spoofing the common-view digest or source range fails closed;
4. repeated metadata-only scope construction is deterministic;
5. scope construction has no artifact-loader/numeric-data input surface;
6. actual Development source materialization matches the precomputed common-view identity;
7. Admission locators/sources are rejected before loader invocation;
8. actual candidate and all mandatory baselines use the same common-view dataset and tile range;
9. targeted/falsification/integration/static/architecture/full-suite/package/build/exact-head CI gates pass.

Development numeric evaluation remains unopened until the actual source-loader and evaluator integration are separately implemented and verified. Admission remains SEALED and Production remains NO-GO.
