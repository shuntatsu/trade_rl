# Signal diagnosis run (SD2)

Status: Active

Specification of the runner that applies the frozen `signal_diagnosis_protocol_v4`
(SHA-256 `1f48b065b03bc38299a751bef89456b57d5afc6a7d8b917668b1b461f852d0b3`, Issue
#1146) once to the consumed development Dataset and writes one verdict. The runner is
`trade_rl/evaluation/signal_diagnosis_run/`. It was verified on seeded synthetic panels
only: **no real-data statistic exists**, and every later result is a known-used, post
hoc, exploratory screening label, never an edge, G4, G5, unused-data or live claim. The
protocol is the contract; this document records what the runner does with it and where
the protocol left a choice. The G0-G2 contract and the author packet are in
`architecture/research-assurance.md` (section *SD2*).

## 1. What one run computes

| Layer | Computed | Role |
|---|---|---|
| D0 | availability check (`available_at` against the bar), a future-perturbation probe of every maintained feature calculation on seeded synthetic bars, a label-copy screen of the 167 Dataset features, a per-(symbol, feature, calendar year) z-score profile against the expanding past | validity flags and context |
| D1 | zero and drift baselines inside every cell, per-vintage beta slope on the market mean | context, never decisive |
| D2 | 167 features by horizons {4, 24, 72} by transforms {raw, standardise-clip3}: daily rank and linear IC against the raw and the demeaned target, one-sided EWC p-values in both directions, Holm and BH over the 167 features per (horizon, transform, target, direction), the sign payoff at cost x1 and x2 | reported, never decisive |
| D3 | 16 decision cells = 8 families x {`ridge_inner_cv`, `lightgbm_fixed`}, primary horizon 24, primary transform, 65 vintages (first fit 2021-04-01, 30-day refit, expanding window) = 1040 fits plus 5 folds x 5 penalties of inner CV per Ridge vintage; targets {raw, demeaned}; statistic = mean daily rank-IC increment over the drift baseline; p-value = `ewc_mean_test`; BH over the 16 cells per target inside the verdict | decisive |
| D4 | k {1, 2} x cost multiplier {1, 2} = 4 payoff grids per cell: drift-excess payoff, daily mean, 20-day moving-block interval (2000 resamples, seed 0, 95%); decisive numbers are k = 1, interval at x1 and point at x2 | decisive, only for a cell that clears both BH gates and the year rule |
| D5 | year (decisive: calendar years 2022-2025, covered months, sign), symbol, regime (sign of the causal trailing 30-day market return) and 1% per-tail extreme-day exclusion, for every cell | year decisive, the rest reported |
| Verdict | `classify_signal_diagnosis` on the 16 cells with `PValueProvenance` (frozen procedure id and eligible days) and the validity flags | INVALID / NO_SIGNAL / SIGNAL_NOT_TRADABLE / SIGNAL_CANDIDATE |

The protocol assigns D0, D1, D2 and part of D5 to card SD2 and D3, D4 to card SD3. Its
decisive cells are family-by-model cells that need the prequential fits and its budget
is `single_run`, so one invocation performs all layers. Models are trained on the raw
forward log return and scored against both targets. They are fitted at the primary
horizon and the primary transform only; the secondary horizons and the raw transform
appear in D2.

## 2. Refusals

Before opening any Dataset arrays file, the CLI reads only `manifest.json` and
admits its metadata: the manifest must hash to its own `artifact_digest`, match the
pinned artifact, arrays, Dataset and feature-config digests, declare the five-symbol
roster and a continuous hourly grid, and fit no more bars than the protocol window.
Malformed, tampered, foreign or out-of-window manifests are refused **without opening
`arrays.npz`**. This includes any manifest declaring FS1 warm-up, sealed or unused
coverage. A filesystem `Path.open` spy verifies zero arrays-file opens for these
refusals; URL, clean-tree and LightGBM checks precede even the manifest read.

Only after manifest admission does the CLI open `arrays.npz` once, hold the raw bytes
in memory, and compare their SHA-256 to the pinned arrays digest. A payload digest
mismatch therefore **does read** those bytes once, but never decodes them. The
Dataset decoder `load_dataset_from_verified_bytes` uses exactly the verified
in-memory manifest and array bytes, without another file read; a post-verification
directory swap cannot change what is decoded. The valid positive control checks
one arrays-file open and identical bytes at the decoder, and the injected file-swap
regressions preserve that boundary. This software-level check does not claim an
atomic directory-swap defense or a bound on large payload allocations. Before any statistic: no well-formed G4 authorization comment URL; a dirty Git working
tree (untracked files included); a plan other than the protocol's; a missing LightGBM;
a non-empty output directory. A protocol digest other than the pinned one or a Dataset
identity other than the pinned `dataset_id`, artifact, arrays and feature-config
digests produce an INVALID verdict and nothing else. After loading, a bar closing before the window
start or after 2026-08-01T00:00Z (the FS1 warm-up span and the sealed window), a roster
other than the five symbols, features other than the 167 protocol names or an irregular
hourly grid are refused. A cell that fails is saved with its error and the run ends
without a verdict (the single run is consumed).

A synthetic run (`synthetic_only`) carries no URL, refuses the development identity and
may widen the refit interval or skip the reported-only layers. Tests use it.

## 3. Interpretations of the protocol

1. A row's label is usable only if both bars were available at their own timestamp
   (`available_at <= timestamp`); otherwise it is dropped, never imputed. A cell where a
   feature is flagged available while `available_at` is later than the bar is a
   dataset contradiction and sets the `available_at_violation` flag.
2. The model training set follows the shared causal builder (label closes strictly
   before the cutoff); the drift baseline follows the protocol text (label matured at
   or before the cutoff). A row maturing exactly at the cutoff is in the second only.
3. Train-window standardisation uses the training rows of each (symbol, feature)
   column, population standard deviation, scale 1 when it is below 1e-12.
4. Inner CV: the time-ordered training rows are cut into six blocks at row-count
   quantiles; fold j validates on block j and trains on rows whose decision time is
   earlier than the block start minus 24 bars (the purge); the score is the symbol-
   balanced weighted validation MSE; ties go to the larger penalty. The Ridge is the
   shared fitter; the CV is computed from moments and pinned equal to it by test.
5. The year increment is the mean of the year's valid daily increments on the demeaned
   target; a covered month holds at least 15 valid daily increments of that series.
6. The symbol slice is the forecast's mean daily rank IC within one symbol (the drift is
   constant within a symbol-day, so an increment is undefined).
7. D2 standardisation statistics come from the vintage in force at the decision time.
8. D4 uses the raw forward log return as the realised label and 14 bp x multiplier as the
   round-trip cost from the protocol.

## 4. Output layout

`run_record.json` (digests, authorization URL and digest, code HEAD, tree id, dirty
flag, module digest, versions, plan, Dataset identity, vintage cutoffs, availability and
leakage reports, status), `verdict.json` (label, reasons, per-cell p-values and BH
adjusted values, validity flags, D4-evaluated cells, claim ceiling), `cells/<family>__<model>.json`
(16 files), `d0_profile.json`, `d1_baselines.json`, `d2_features.json` and
`forecasts.npz` (float32 forecast per cell). Files are written once into a new
directory. The text holds no clock value; the same panel, protocol and code give
byte-identical files.

## 5. G4 authorization

The protocol defines no authorization text. The owner's G4 authorization is a GitHub
issue or pull-request comment whose URL is passed as `--g4-authorization-url`
(`https://github.com/shuntatsu/trade_rl/(issues|pull)/<n>#issuecomment-<id>`); the
runner cannot authenticate its author. The URL is bound to the protocol digest as the
run request's authorization digest. See `architecture/research-assurance.md` for what
the comment should name.

## 6. Cost of a run

A full-size synthetic panel (46,752 hourly bars, five symbols, the 167 feature names,
seeded random values) ran every layer on the protocol schedule (65 vintages, 30-day
refit) with the real LightGBM in 345 seconds on one workstation: about 160 seconds for
D0 to D2 and about 20 to 45 seconds per family for the fits and the cell statistics.
Expect minutes, not hours, on the real Dataset; this is a synthetic measurement, not a
measurement on real data, and the real feature distributions may change the LightGBM
time.

## 7. Not authorized

This document and the runner authorize no real-data execution, no read of the FS1
warm-up span, sealed window or any unused data, no replay, no order and no claim.
