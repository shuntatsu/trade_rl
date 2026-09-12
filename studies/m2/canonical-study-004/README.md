# Canonical M2 Study 004

Status: Pre-registered combined correction lineage; no baseline result exists yet.

## Purpose

Study 004 is the first Canonical M2 lineage that combines both protocol corrections required before resuming real-data M2 research:

1. explicit Dataset-bound execution economics from PR #484 / Issue #481;
2. PPO Observation v2 from PR #491 / Issue #489.

The research question and all ordinary Study 001 research degrees of freedom remain unchanged: can one symbol-ID-free universal strategy family beat simple controls robustly across five long-lived Binance USD-M perpetual markets under one frozen development protocol?

This record is fixed before Study 004 StudyPlan creation, baseline execution, baseline-result inspection, candidate execution, or controlled Experiment execution.

## Frozen implementation integration

Study 004 code base is the clean two-parent integration commit:

- Study 004 integration SHA: `135cbdaad1a40cd064b8b08492d591d385a2c89e`
- tree: `afe75e0d45fd401e5e6819c7b78a1d099666cc53`
- parent 1 / execution economics: `558919861f1fa86956664405b8946ab923b2d863`
- parent 2 / Observation v2: `b3c20f01966c8f4e944255ee7df5cd4258684a0f`
- main base for both parents: `4780882f0caf2312fbb33108573ad53b4690e776`

Before this research branch was created, an isolated integration gate verified the exact two-parent tree. The gate passed focused cross-contract tests, explicit falsification mutations, Ruff, format, production Mypy, architecture-tooling Mypy, the full test suite, package build, source-closure validation, sdist-to-wheel rebuild, and clean-installed smoke checks. The temporary integration workflow was not present in the tested tree and was deleted afterwards.

Study 004 does not imply that PR #484 or PR #491 has been merged to `main`. They remain separate development PRs; this research lineage freezes their exact verified integration without changing `main`.

## Superseded attempts and chronology

Study 001 remains immutable historical evidence. It is zero-trading-cost diagnostic evidence under the maintained Dataset-authoritative replay contract and is not eligible for canonical cost-aware advancement.

Study 002 is not used as a canonical lineage. Its preregistration chronology did not precede the separate real-cost bootstrap run, and it did not include Observation v2.

Study 003 is not used as a canonical lineage. Its preregistration included the execution-economics correction but not Observation v2. Run `34679691431` was cancelled while bootstrap was still in progress and before baseline execution; it produced no baseline artifact.

A separate pre-Observation-v2 real-cost baseline attempt, run `34679928915`, was also cancelled before artifact publication. It produced zero Actions artifacts and is not research evidence.

No outcome from either cancelled baseline path is used to choose Study 004 research settings.

## Reused immutable Dataset input

Study 004 will not redownload or rebuild the historical market Dataset before baseline. It will reuse only the already-published Dataset bytes from the successful real-cost bootstrap artifact below:

- source bootstrap run: `34679031604`
- source artifact name: `canonical-m2-real-cost-bootstrap-v2-34679031604`
- source artifact ID: `10293251579`
- source artifact outer SHA-256: `1bc5bfffc314a0aee951dbba2b5fb8a05e0ea7754bd5ee4faa4dc0f035145ce2`
- Dataset ID: `13334380e5e52fae6b69688a4270de736958c384a7991234ea5635887b06adc0`
- Dataset artifact digest: `c1fb38f2805f452c423a648b7daf14ea1e910aae63c8bb5e24429323b14f02c2`
- bootstrap config digest: `8d39d389d8a6b3e57439b45862063d14ccf1b3fdfd06135681804814202db425`
- raw source roster digest: `0293e7575533bb3bd8876f5b130495491b2496144bff2ac9a5d0ba78553fe6d0`
- Binance Vision resolution digest: `ba0a247f5b32d82493e6ad67d106ccb60b1dedafb7948870179afbc904d15fcf`

The source artifact was generated from production code equivalent to final PR #484 for Dataset construction; the remaining #484 delta was test-only. PR #491 does not modify Dataset-building paths. Reusing the exact Dataset bytes therefore avoids introducing a second raw-download/source-metadata draw into the comparison.

Only the source artifact's `bootstrap/dataset` directory is eligible for reuse. Its pre-Observation-v2 StudyPlan, whose Study digest was `ebd2b8fcefdb71c02f7c2b5dd9d77f795edd99ec48ed0cfc07dfb7187a3db35a`, is explicitly not reused as Study 004 authority.

Study 004 must construct a new StudyPlan from the reused Dataset under the frozen integrated code so that Observation v2 fields enter resolved-run identity. Before baseline execution, an immutable Study 004 rebind artifact must prove that:

- Dataset ID remains exactly `13334380e5e52fae6b69688a4270de736958c384a7991234ea5635887b06adc0`;
- Dataset artifact digest remains exactly `c1fb38f2805f452c423a648b7daf14ea1e910aae63c8bb5e24429323b14f02c2`;
- source roster and Vision-resolution digests remain unchanged;
- the new baseline resolved config identifies Observation v2, including `ppo_observation_schema = ppo_observation_v2` and an empty `ppo_global_feature_names` roster;
- the new Study digest differs from Study 001 and from the pre-Observation-v2 real-cost Study digest;
- no baseline evidence exists inside the rebind artifact.

## Frozen research conditions

The Study 001 research question, symbols, clocks, baseline hyperparameters, seed roster, Experiment budget, and allowed controlled factors remain unchanged.

Ordered symbols:

1. `BTCUSDT`
2. `ETHUSDT`
3. `BNBUSDT`
4. `XRPUSDT`
5. `ADAUSDT`

Time contract:

- source start: `2021-01-01T00:00:00Z`
- fit cutoff: `2023-01-01T00:00:00Z`
- development start: `2023-01-01T00:00:00Z`
- development stop: `2025-01-01T00:00:00Z`
- base timeframe: `1h`
- feature timeframes: `4h`, `1d`

Baseline degrees of freedom:

- rule signal: `1h__log_return_24bar`
- feature set: exactly the 12 names in `bootstrap.json`
- fit symbols: all five Study symbols
- PPO seeds: `0,1,2,3,4`
- PPO budget: `100000` timesteps per seed run
- gross budget: `0.5`
- initial capital: `100000`
- paired/bootstrap analysis count: `2000`
- bootstrap seed: `1729`
- Experiment budget: `12`
- rule entry/exit: `0.01 / 0.0025`
- forecast entry/exit: `0.0025 / 0.0005`

Allowed controlled factors remain:

- `FEATURE_SET`
- `RULE_SIGNAL`
- `RULE_THRESHOLDS`
- `FORECAST_THRESHOLDS`
- `PPO_TRAINING_BUDGET`

`FIT_SYMBOL_SCOPE` and `GROSS_BUDGET` remain excluded from within-Study variation.

## Execution-economics correction

Study 004 freezes `canonical_m2_research_assumption_v1`, schema `execution_economics_profile_v1`:

- generic fee: `0.0005`
- maker fee add-on: `0.0`
- taker fee add-on: `0.0`
- spread: `0.0002`
- maximum participation: `0.05`
- borrow available: `true`
- borrow rate: `0.0`

Impact and stochastic slippage remain zero in the maintained zero-overlay replay. These are reproducible research assumptions, not a claim about any person's historical account-specific Binance fee tier.

## Observation v2 correction

Study 004 freezes the PPO Observation v2 contract before baseline:

- local feature values;
- local feature availability;
- normalized local feature staleness;
- current intent;
- current weight;
- no global regime channels;
- `ppo_global_feature_names = []`.

Global full-universe regime channels are intentionally excluded because they can reveal fit-scope-external symbols during PPO training. Observation schema and global-feature roster are Study-fixed resolved-run identity fields.

## Baseline quality gate

Baseline execution is forbidden until a separate immutable preregistration artifact and a separate immutable Study 004 rebind artifact have been uploaded and re-downloaded successfully.

The baseline must then run all five PPO seeds against the exact rebound StudyPlan. Before any controlled Experiment begins, an independent post-artifact audit must verify at least:

- exact Study/Dataset/source/runtime/implementation provenance;
- all five expected seed runs and complete strategy/symbol rosters;
- finite raw return arrays and summary consistency;
- nonzero realized execution costs on trading strategies, while cash remains zero-cost;
- Dataset-authoritative execution economics with zero runtime overlay;
- Observation v2 identity in the baseline resolved config;
- no source/dataset drift from the frozen Dataset input;
- no unexpected seed sensitivity in deterministic strategies;
- immutable baseline artifact identity and re-download verification.

Any violation invalidates the baseline as canonical evidence.

## Experiment boundary

No Experiment 0001/0002 or other candidate is preregistered by this document. No candidate probe may begin until the Study 004 baseline passes its independent quality gate. The next Experiment choice must be preregistered separately after baseline inspection according to the Study protocol.

No profitability, winner, Production, or deployment conclusion is implied by this preregistration.
