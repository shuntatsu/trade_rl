# Universal Trade RL U2 Development Evaluation Access Amendment

Status: **Normative U2 V1 amendment**  
Production: **NO-GO**  
Admission: **SEALED**  
Real U2 training: **NO-GO**  
Development numeric evaluation: **NOT OPENED**

This amendment is written before any U2 Development numeric evaluation, Admission numeric access, or economic result. It resolves one role-access mismatch between the generic U0 phase firewall and the orthogonal symbol × time evaluation preregistered by U2.

It does not change U0 generic access semantics for other research generations.

---

## 1. Conflict discovered during Task 7 preparation

The generic U0 Development access contract exposes:

```text
fit_symbols        = Train symbols
evaluation_symbols = Development symbols
```

That contract is correct for a simple symbol-OOS Development evaluation.

U2 V1, however, preregisters an orthogonal symbol × time matrix:

```text
A  Train       × SEEN_TIME_PROBE       diagnostic only
B  Development × SEEN_TIME_PROBE       mandatory
C1 Train       × DEV_FUTURE_1          mandatory
C2 Train       × DEV_FUTURE_2          mandatory
D1 Development × DEV_FUTURE_1          mandatory
D2 Development × DEV_FUTURE_2          mandatory
E  Admission   × ADMISSION_FUTURE      sealed
```

Therefore C1/C2 require **evaluation-only access to Train symbols after FIT**, which cannot truthfully be represented by treating Train symbols as generic `evaluation_symbols`, nor by misusing the generic `fit_symbols` authorization for post-FIT outcomes.

---

## 2. Generic U0 firewall remains unchanged

U2 V1 must not broaden `UniversalTradeRLUniverseAccess.for_phase(DEVELOPMENT)` globally.

The following generic semantics remain unchanged:

```text
TRAIN phase:
  fit scope        = Train
  evaluation scope = none

DEVELOPMENT phase:
  fit scope        = Train
  evaluation scope = Development

ADMISSION phase:
  evaluation scope = Admission only after exact authorization
```

No existing U0/U1 consumer gains additional source access from this amendment.

---

## 3. U2-specific Development evaluation scope

U2 V1 introduces a narrower, generation-specific Development evaluation contract derived only from the frozen U0 manifest, the frozen U2 time partition, and the frozen U2 contract.

It authorizes **evaluation**, not fit, for exactly:

```text
Train symbols:
  SEEN_TIME_PROBE
  DEV_FUTURE_1
  DEV_FUTURE_2

Development symbols:
  SEEN_TIME_PROBE
  DEV_FUTURE_1
  DEV_FUTURE_2
```

The Selection-relevant cells remain exactly B/C1/C2/D1/D2. A remains diagnostic-only.

The contract must not authorize:

- any Admission symbol;
- any `ADMISSION_FUTURE` outcome;
- any new normalization, calibration, training, threshold fitting, architecture selection, checkpoint selection, or model update;
- any post-FIT bar to enter U1 normalizer or PPO training;
- any symbol outside the frozen Train/Development role sets.

---

## 4. Fit authority remains Train × FIT only

This amendment does not redefine fit access.

The only U2 fit/training authority remains:

```text
symbols = frozen Train role
outcomes <= FIT end
normalizer cutoff == FIT end
RL_TRAINING provenance cutoff == FIT end
```

A source artifact that is opened for U2 Development evaluation is evaluation input only. Its post-FIT bars must never be reused by training, normalization, calibration, checkpoint selection, or threshold fitting.

---

## 5. Admission remains sealed

U2 Development evaluation scope construction must be possible without constructing Admission access and without supplying Admission authorization.

Before exact U2 Admission authorization:

```text
Admission numeric source opens = 0
Admission evaluation scopes     = unavailable
Admission outcome bars          = unavailable
```

The U0 generic Admission firewall remains the authority for eventual Admission access.

---

## 6. Scope identity requirements

Each U2 Development evaluation scope must bind at least:

```text
U0 universe manifest digest
U2 time partition digest
U2 training contract digest
symbol role
concrete symbol
cell
source window
tile identity / tile coordinates
evaluation range
selection role (diagnostic or mandatory)
```

Scope closure must be derived from metadata only. Constructing A/B/C/D scopes must not open numeric market arrays.

The canonical scope order is deterministic and must not depend on filesystem locator order or numeric outcomes.

---

## 7. Required verification

Before Development numeric evaluation is allowed, one exact HEAD must prove:

1. A/B/C1/C2/D1/D2 assignments exactly match preregistration;
2. C1/C2 use Train symbols only and are evaluation-only;
3. B/D1/D2 use Development symbols only;
4. no Admission symbol or `ADMISSION_FUTURE` tile can enter Development scope closure;
5. each scope consumes the preregistered tile-provided `evaluation_range`;
6. scope construction performs zero numeric source loads;
7. later Development source loading accepts only the exact frozen Train/Development source identities required by those scopes;
8. fit/training code still rejects all post-FIT outcomes;
9. Admission numeric access count remains zero;
10. targeted, falsification, integration, static, architecture, full-suite, package/build, and exact-head CI gates pass.

Production remains NO-GO. Admission remains SEALED.
