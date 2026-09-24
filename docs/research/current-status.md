# Current research status

更新基準: 2026-09-22 (JST)

## 結論

Trade RLの現在地は、**lean core、5候補+3 controlsの共通比較基盤、provenance-bound candidate Run Core、Controlled Experiment Loop v1、Canonical M2 bootstrap toolingを実装し、`market_build_v3` / `portable_feature_numerics_v1`、real-cost-assumption Dataset、fit-scope-safe PPO Observation v2を固定したportable Canonical real-data baselineを、結果前のplan-only preregistrationからfresh post-Artifact verificationまで完了した**段階である。

一方、**Portable Controlled Experiment 0001は独立再検証まで完了し、formal decisionはKEEP_BASELINE**である。事前登録した唯一のcandidateはmean-reversionで5 / 5 symbolsをbaseline比改善し、median turnoverも低下したが、candidate total returnが正だったのは1 / 5 symbolsだけだった。結果前に固定したformal ruleではこの条件がKEEP_BASELINEに該当する。したがって現在も次を主張しない。

- profitabilityは未証明。
- winnerは未選定。
- Production/live order routingは未認可。
- PPOやforecastがruleを上回るという結論はない。

次の研究上の本質的作業は、新しいmodel familyやbootstrap toolingを増やすことではない。Experiment 0001のKEEP_BASELINEをcurrent development authorityとして維持し、次に検証するControlled Factorを結果を見る前にpreregisterしたうえで、同じfactor-isolation・raw-return・cost/cash・fresh post-Artifact verification契約でdevelopment Experimentを積み上げることである。

## 研究目的

### Separate paired funding-carry development capability

A separate BTC/ETH spot-long/perpetual-short carry comparison is complete.
It has no forecasting or learned selector.
The source roster is fixed to official hourly/funding archives covering reused
2023-2024 development time; 150 raw archives were acquired and hashed before any
carry economic replay. Software review covers equal-quantity holding, actual
exits, signed funding, fee conservation and futures collateral excluding spot.
The strict-clock source attempt stopped before economic replay: monthly and
independent daily spot archives both lack the 2023-03-24 13:00 UTC open for BTC
and ETH. Binance's official incident report documents a spot halt from 11:27 to
14:00 UTC. Preserve that failed attempt; a separately identified source revision
explicitly masks intersecting halt bins and uses a preceding-close stale mark
only for a missing whole halted bin. Unknown gaps still fail. Software validation
and source acquisition alone are not profit evidence. The subsequently frozen
halt-aware protocol
`aa67889f4facdbad927437cbc65d6ef1f9b6f3e8e6ab2deb3c446570a2183aef`
completed all twelve predeclared replays. The decision is
`PROSPECTIVE_PAPER_DESIGN_REQUIRED`, with all twelve development screens passing.
Shared-account net return over the full two years was +5.3031%, with +2.0269% in
2023, +3.2111% in 2024 and ledger maximum drawdown 0.6145%. Double fees/spread
returned +4.9830%, one-hour initial-entry delay +5.3028%, and 10x lower
participation +5.3031%. Independent BTC and ETH base returns were +5.0491% and
+5.5622%; every base year and every stress full-period return was positive.
All replays completed 17544 intervals, actually closed all positions and had no
stop, unmatched hedge, collateral breach or canonical termination.

The fixed research contract used 10000 USDT shared capital (5000 for each
independent pair), gross 0.5, monthly equal-base quantity targets, common lot
0.001 and minimum notional 10. Fees were spot 10bp/perpetual 5bp plus 5bp adverse
execution per leg; participation was 1% of preceding hourly quote volume.
The irreversible carry drawdown stop was 10%, below the user's 20% research
tolerance. Base qualification required both years and full-period net profit;
all nine stresses required full-period profit. Every arm additionally required
drawdown below 10%, complete execution, actual flatness and no stop/invalidity.
The tests do not prove profitability at larger capital or under venue-specific
account rules. Hourly marks and coarse halt masks do not establish intrabar
drawdown. The apparent equity spike at the March spot halt reflects asynchronous
stale spot versus current perpetual valuation; it is not a realized windfall.

An independent auditor, without importing the simulator, reconstructed all twelve
cash/quantity/equity paths from recorded fills and the frozen market arrays,
including costs, signed funding, capacity, actual flatness and collateral checks.
Shared base final cash was 10530.3127 USDT, funding income 564.1770 and execution
costs 31.1492. The final selection SHA-256 is
`8196ceebe637d39d906a8f1559f37a7f7783bef8eee8de24d3f00c8f3b1f3776`.
This is positive reused-development evidence, not unused-future validation or
permission for live orders. Perpetual-close mark proxies, assumed fees/rules,
instant wallet transfers and unobserved order-book execution remain limitations.
The next active design is `docs/specs/funding-carry-paper.md`. Completed PPO and
carry studies retain their original frozen source and evidence.

Forward source capture has passed permanent CI and real public-feed acquisition.
The separate recorded-depth paper matcher models partial bid/ask fills, declared
fees and adverse pricing on the canonical account. These are software components;
current public-rule capture and verified snapshot readers are available, while
an append-only protocol-bound journal has been implemented and tested with
concurrent writes and interrupted transactions. The deterministic paper engine
now composes saved decisions, verified later quotes, partial fills and settled
funding on the existing account, and verifies every committed result on restart.
Funding uses exact quantities held before the settlement timestamp, including
when payment evidence arrives after an exit. Revisions, late/missing funding and
observation gaps remain visible permanent quality failures. A standalone
source/runtime-bound collection component now seals its identity before start,
refreshes public rules, supervises pending decisions and durably halts on failures
or interrupted cycles. The independent real-public-data supervisor smoke executed
four entry and four exit fills, reopened identically and reconciled to exact zero
quantities. Final cash was 9987.8435571 from 10000 virtual USDT, fees 7.4208051,
and funding income zero. This brief software check demonstrates execution costs,
not profitability. Its earlier helper-error attempt is preserved as failed.

The cadence CLI and fixed prospective screen now support seal/run/status/evaluate.
The screen lasts ninety UTC days with three thirty-day blocks and unchanged
10000 virtual USDT/default cost assumptions. It requires positive full-period
and each-block returns, net profit above recorded fees, drawdown below 10%,
funding receipts in every block, no unpaid announced settlement, complete
source/control evidence and actual terminal flatness. Fee headroom is assessed
on the realized trajectory, not a separate doubled-fee strategy replay. Minute
observations cannot guarantee intraminute risk. Status output is operational;
only full offline replay at the fixed deadline may decide the paper screen.
The first formal prospective screen is sealed under protocol
`843467870d85d0b085e65cf904e9d458287c14fc1e31615b9aaf77029f0669b7`.
Its fixed UTC start is 2026-09-17 20:05, close is 2026-12-16 20:05, and terminal
observation deadline is 20:08 that day. The dedicated detached worktree
`funding-carry-forward-20260918` and its private locked environment remain frozen;
the collector was launched before start. Its source implementation digest is
`5e093425ee775639b2ac840831e8a71f30c34966b6150367e40bdc0b7f6ed349`
and runtime environment digest is
`0c56fc67d16f583d49b5dea7a3878e3fcba0f58e6abd0e9dd66e51393c994cc9`.
The declared period is not complete. No future profit or deployment qualification
has been established, and neither a partial return nor software CI can pass it.

A subsequent real one-minute CLI software probe exposed partial ETH spot depth:
the 20-level capture filled 0.3821 ETH against a 0.51 ETH perpetual short. The
engine rejected the unmatched hedge, actually flattened all four legs on the
next observation, and exited with code 1. Final cash was 9988.8165415 USDT,
fees 6.7969229 and funding zero. Preserve that rejected software attempt.
New capture uses an explicitly versioned 100-level profile; the old v1 reader
remains available for evidence verification. This expands observed price levels,
while retaining 10% participation, costs, risk and qualification conditions.
It does not establish that later quotes would have filled the earlier orders.
The separate 100-level software probe then completed its whole minute cadence
and terminal grace, with four completed cycles, six journal events and eight
fills. It reopened identically, ended actually flat and had no quality failure.
Final cash was 9987.8380018 USDT, fees 7.4237759 and funding zero. This remains
excluded from the formal economic screen. The dedicated study environment also
passed all 155 paper and forward-source tests before sealing.

### Directional development study under a 20% drawdown budget

The complete 13-arm study returned `NO_QUALIFIED_CANDIDATE`; all five PPO seeds
lost money. Their net returns were -17.1134%, -15.4319%, -13.3033%, -13.7667%,
and -15.0823%, with median -15.0823%. Channel breakout returned +1.4456% but
failed the 2024-positive and terminal-flat gates; the constant-long control
returned +14.8059% and also retained terminal holdings. No stress replay was
reached because no candidate passed the base screen. Independent raw-return
arithmetic and file/model hashes passed; this is not an independent order-ledger
reconstruction. The original immutable protocol is
`aa9bf63e43668a2edbc0c8a54a5c6e689a277de4fb9e2af7862892914e783000`.

The subsequent five-seed risk-only comparison completed under separately frozen
protocol `eafa0327189021986116aefce33936a34e56409de7b3b53e8ba337c969e149c4`,
bound to all baseline result hashes and the completed selection. Its frozen
source passed both permanent CI jobs; exact source revision, workflow identity,
and raw snapshot hashes are retained with the study's run-context evidence.
This software verification is not economic evidence.
Its decision is `KEEP_BASELINE`: paired net-return improvements occurred for only
three of five seeds, below the registered four-seed requirement. Candidate net
returns were +12.2341%, +4.3244%, -12.5830%, -19.5275%, and -17.5039%; the median
was -12.5830%. The median paired improvement was +0.7203 percentage points;
this differs from subtracting the two family medians. No seed passed all base
gates, so stress replays were not reached. Seed 0 had positive full and both-year
returns but retained terminal holdings. All five completed 262144 training steps
and all 17544 evaluation intervals. The independent audit verified saved model
parameters, raw-return arithmetic, hashes, and the aggregate decision, but did
not independently reconstruct an order ledger. Positive individual seeds do not
qualify this PPO family. The default training risk remains unchanged.

The risk-only relative gate requires at least four paired net-return wins,
a positive five-seed median paired delta, complete baseline and candidate
replays, candidate ledger drawdowns at most 20%, and no increased termination
count per seed. Cost and turnover are diagnostics. The absolute profitability,
terminal-flat, and stress gates remain separate; relative improvement alone
never authorizes deployment. Baseline model/result/selection hashes, dataset,
source snapshot, and identical runtime are frozen before fitting candidates.

Source and synthetic execution diagnostics also identified an admission/ledger
quantity inconsistency: accumulated floating-point lot additions/subtractions
can leave approximately one lot that admission accepts but fill allocation
rejects. Genuine below-minimum-notional residuals are a separate issue under the
current execution contract. Neither terminal holdings nor thresholds are altered
in the risk comparison. A seed that did end flat still lost 13.7667%, so terminal
flatness alone does not explain the observed lack of profitability. A future
execution fix requires its own reviewed contract and fresh evidence; historical
results remain unchanged.

The user subsequently explicitly prioritized PPO and data/training improvements.
A train-only feature audit and source audit found that the default training risk
does not match the directional evaluator: training allows gross/per-symbol 1.0
and drawdown start/stop 1.0, while evaluation uses gross 0.5, per-symbol 0.1,
drawdown start 0.1 and stop 0.2. The completed comparison changed only these
training risk settings. Training still has one active symbol per account while
evaluation shares cash across five, and policy inputs lack account drawdown.
The optional risk configuration does not resolve these remaining mismatches.
This is not a reward
cost omission: the current PPO reward already uses log net return after costs.
Fill counts combine policy and risk actions and cannot alone diagnose churning.
Feature-scale normalization and the sealed interleaved experiment remain separate.

An opt-in fit-only PPO feature standardizer and bound model-bundle capability
are implemented separately from the completed, immutable risk comparison.
The durable preprocessing/model contract is in `architecture/lean-core.md`.
Synthetic scope, masking, default compatibility, balanced statistics and
model/transform reload checks and full software CI pass. The five-seed normalized
real-data comparison completed under frozen protocol
`464278c3390abd20e6741c7176dcc7382a3679f5351de8aa245f734fc971ff0c`.
The decision is `RELATIVE_IMPROVEMENT_ONLY`: four of five paired returns improved,
and the median paired delta was +3.8399 percentage points. Net returns were
-8.2816%, -19.0971%, -9.4633%, -9.1378%, and -12.0839%; median -9.4633% versus
the original -15.0823%. All five remained unprofitable and retained terminal
holdings; no seed passed the base screen and no stress replay was reached.
Median turnover increased from 48.5957 to 364.0627, a diagnostic to investigate,
not proof of policy-driven churn. All five completed 262144 training steps and
17544 evaluation intervals. An independent arithmetic/hash/model audit also
reconstructed fit-only balanced normalizer moments from raw source arrays and
confirmed the aggregate gates. It did not independently reconstruct order ledgers.
The final comparison SHA-256 is
`fab5e1e13c35263533f3f36f62ce7a7017fe92e5413b58a56f61f84ba910df33`.
The isolated comparison retained original default risk, source, budget and gates;
it does not establish operational profit or reopen the sealed interleaved study.

The user subsequently broadened the search to other RL algorithms, ensembles
and additional data. These are permitted future candidates, subject to the same
cost, drawdown and out-of-sample evidence requirements. The completed
standardization study changed only its registered preprocessing factor.
Exact fill-quantity accounting now preserves accepted lot counts and genuine
remainders across book/order updates and resume; capacity allocation searches
integer lots against the actual monetary bound. This separate implementation
does not alter any active frozen study or its historical results.

### PPO BTC-relative feature ablation: G4 completed, KEEP_BASELINE

PPO BTC-relative feature ablationはG3/G4まで独立監査済み。
absolute base profitabilityは0 / 5 symbols。doubled-cost / one-bar-latency stressは0 / 5 symbolsでfailし、
development decisionは`KEEP_BASELINE`。G5とProduction/live eligibilityは未判定である。

A dedicated private runner, `trade_rl.evaluation.ppo_feature_study`, and its
result-blind protocol are implemented. The first preregistered attempt was
started on 2026-09-20, then stopped during the fifth baseline arm's development
replay when the host approached its memory-commit limit. Four baseline arms had
completed; the fifth model was saved, but its ledger replay and the paired study
were incomplete. The attempt was not finalized, and its economic outputs were
not inspected. Its partial write-once artifacts remain preserved under
`output/ppo-btc-relative-feature-ablation-20260920` and must not be combined with
a later protocol.

They test one narrow question: whether adding three existing BTC-relative
return features helps the current PPO candidate under its fixed development
screens. The exact frozen successor Dataset/Study artifact is preserved in the
companion `directional-profit-bot/output/source-successor` worktree. Its
Dataset, artifact, StudyPlan, evaluation-Dataset, and protocol identities were
re-resolved and matched the fixed values. The replacement run uses the same
fixed source and economic comparison, with a memory-bounded ledger observer.
Do not substitute the separate 2024–2026 realdata generation. The
baseline feature roster is compared with the same roster plus
`1h__relative_return_to_btc_1bar`, `4h__relative_return_to_btc_1bar`, and
`1d__relative_return_to_btc_1bar`. This is a PPO feature ablation, not a
comparison of PPO against other RL families.

The protocol pairs two arms over the same five seeds (0–4), sequential PPO
layout, 262,144 requested training steps per seed, full fit-symbol roster,
pre-2023 fit cutoff, `max_gross=0.5`, `max_abs_weight=0.1`, no turnover cap,
drawdown start 0.1 / stop 0.2, and the same execution/accounting implementation.
That is ten fits in total. Evaluation uses five independent 10,000 USDT accounts
(one per BTC/ETH/BNB/XRP/ADA symbol), each with gross budget 0.1, over the same
17,544 development intervals from 2023-01-01 00:00 to 2025-01-01 00:00. Return
intervals are grouped by their start timestamp: 2023 has 8,760 intervals and
2024 has 8,784; the final 2025 timestamp closes the last 2024 interval.

Candidate base and stress cells all carry a global hard veto: any seed-symbol
cell with ledger drawdown above 20%, termination, a non-flat terminal position,
or an active order remainder blocks admission. A seed votes for the absolute
return screen only when full and every-year returns are positive in base and
both stresses. Four of five same-seed full-return deltas must be strictly
positive for the paired vote, with positive five-seed median full and each-year
deltas. These economic votes never relax the hard veto; all candidate medians
use all five seeds. The candidate alone is replayed under doubled execution
cost and one-bar latency, across every seed-symbol cell. A pass can request a
prospective paper study only.

The independent result-blind G0–G2 review passed on 2026-09-20 for the original
protocol
digest `09ec5e9e051c7867686dcac9290f6d6a32120c8db459069439386c286f8bbf44`,
implementation digest `58f1e0e801b094df5fc5b8dfe683f8f55edcc5955dc5251e77244497f56dbb62`,
and source snapshot `e04210707a33d812bd3e41b7907528658d17f94869d1950772d48389d3d4bce2`.
The reviewer independently rehashed the implementation and source snapshot;
the protocol digest was supplied as the fixed binding. That review authorized
only the original exact source; it does not carry over to the memory fix. The
replacement protocol was prepared on 2026-09-21 at
`output/ppo-btc-relative-feature-ablation-20260921-r1`, with protocol digest
`a25aa21fcfb2b4c17c83f7fc465a49b1e08171704742563511a24e9933b07fb3`,
implementation digest `f3db8d4f070f3d3bd21c73cd35462c5f87405c79774140ff3e7e4c00162313e4`,
and source snapshot `8a994f7e3d7f6961edff9363f8c65b52e534a391970bde43d3f6f4281b27dc16`.
Its independent result-blind review completed on 2026-09-21: G0 PASS, G1 PASS,
and G2 PASS for this exact binding. The baseline seed-0 fit started on
2026-09-21 and was safely interrupted before fit completion after available
physical memory fell to 1.18 GB on a 15.75 GB host (92% load). Its arm directory
contains only `started.json`; no model, result, ledger, or economic output was
published or inspected. The partial start marker is preserved. A retry needs a
new write-once output root, and must not combine with the 2026-09-20 attempt.
The reviewers confirmed G0 and G1 for the fixed paired mechanism. G2 PASS is
bound to the exact implementation
digest and source snapshot above; its ledger validator does not independently
recompute P&L from persisted order/fill events, and its ledger schema does not
carry an expected symbol index. The current generator passes the symbol index
through single-symbol replay, keeps other symbols flat, and validates row
identity, so reviewers found no current-generation mismatch. These are limits
of the evidence verifier, not a claim that event-level P&L has been independently
reconstructed. No economic outputs from either attempt were inspected when the
reviews were performed. These reviews do not establish completion of repository
quality gates or any economic result. The Dataset and 2023–2024
development interval have already been used by prior research, so this
experiment is not confirmatory, is not an unused-data validation, and cannot
establish general profitability. No production or live-trading claim follows
from its outcome.

After the r1 interruption, source review found `run_arm` retained both the raw
Dataset and the immutable price-channel-augmented Dataset for the full fit and
replay. The code now releases the unused raw reference immediately after
augmentation in both `expected_protocol` and `run_arm`; a weak-reference test
was RED before this change and passes at fit entry. The 61 feature-study tests,
Ruff, and package mypy passed. New exact protocol
`output/ppo-btc-relative-feature-ablation-20260921-r2` binds protocol digest
`e5e11eda3c3206087752e184381931eedc8d94efd6fc21679457b6cfb71633b0`,
implementation digest `7d685cd2f83e59c149171f7c367a95332faff69f56a66542a7124a1faa5a908d`,
and source snapshot `cb55da0e9d53a2e4af53aed0ab8f5fc25e98238ff73292681b9ec71c108f7b37`.
Its independent result-blind review completed on 2026-09-21 with G0 PASS, G1
PASS, and G2 PASS for this exact binding. The G2 reviewer confirmed the
lifetime-only change does not alter Dataset values or training/evaluation
semantics; the memory reduction itself has not yet been measured in a full fit.
The reviewer did not reload the companion Dataset/Study artifacts; local
`reserve_study()` re-resolved their identities while preparing r2. G2 retains
the earlier ledger-verifier limitations: no event-level independent P&L
reconstruction and no expected symbol index in the ledger schema. Baseline
seed 0 fit completed and entered replay on 2026-09-21. Replay was safely
interrupted when host available physical memory reached 1.479 GB of 15.75 GB
(90% load), below the 1.5 GB stop line. Preserve r2 as incomplete: it contains
the fitted `model.zip`, `started.json`, and base ledgers for symbols 0, 1, and
2 of 5; symbols 3 and 4 and the arm result were not published. No economic
output was read. Do not finalize this root or combine it with the original or
r1 partial attempts. Any retry needs a new output root and at least 4.0 GB of
available physical memory at preflight; interrupt again if availability falls
below 1.5 GB.

The checkpoint runner is implemented at
`trade_rl/evaluation/ppo_feature_checkpoint.py`. It uses a separate protocol
identity and atomic completion boundaries for fits, individual
seed-symbol-scenario replays, arm assembly, and comparison. A retry verifies
completed evidence before reuse; missing work may be rerun, while tampered
completed evidence fails closed. An interrupted fit restarts rather than
resuming partial optimizer or rollout state. The original, r1, and r2 roots
remain preserved as incomplete evidence and are not inputs to this runner.

The exact Linux protocol was prepared on 2026-09-21 under immutable tag
`seal/ppo-btc-relative-checkpoint-20260921-v1`. Hosted preparation run
[35537964829](https://github.com/shuntatsu/trade_rl/actions/runs/35537964829)
succeeded and published artifact `10612544268` (raw ZIP SHA-256
`7f776884512bab19f32b50b13e2f47bb7bd3928ef8c2b562c469cb1d6dbbc7b3`). The
outer protocol digest is
`e04d0146fea39bdbb95e1b78ed5b94b2fead296f67774f1007f0496236e9b5d2`; the
source snapshot digest is
`7fb2b501b79dda20bc32ddf69b0e5b56d881f1d2bffce68649a7020ec085eb97`, with
all 170 snapshot files matching Git. Fresh independent result-blind review
passed G0, G1, and G2 with disposition
`G0_G1_CLEAR_G2_EVIDENCE_BOUND`; its record digest is
`2579a7214cf74a394821d80568be7864a963a9b5fa1a0e9c7197e1585591fe9b` ([review
record](https://github.com/shuntatsu/trade_rl/pull/744#issuecomment-5752780311)).

All ten fixed arms have been dispatched under that same frozen protocol: baseline
seeds 0–4 are runs 35538782635, 35538813730, 35538815588, 35538817265, and
35538818919; candidate seeds 0–4 are runs 35538820441, 35538822037,
35538823806, 35538825367, and 35538826669. The unchanged 12-feature baseline
and 15-feature candidate use five seeds and 262,144 requested steps per fit;
their preregistered roster contains 100 replay cells. Each replay starts with
an independent 10,000 USDT per-symbol account, and every candidate cell remains
subject to the 20% drawdown hard veto. The shared-portfolio, stress, and
development-only limits above remain in force.

The exact hosted run completed on 2026-09-21 as run `35542548393`, attempt 1,
and published artifact `10614569766`. Its ZIP is 589,319,909 bytes with
SHA-256 `3997d64142c9143085ea954ef340905c4764ce58d264e21ade791b8d41f7120f`.
The frozen source head is `b876f1c5b5a7`; checkpoint protocol digest is
`e04d0146fea39bdbb95e1b78ed5b94b2fead296f67774f1007f0496236e9b5d2`, and the
canonical core-protocol digest is
`e37701b92ddfb93f3bc6d528e1affebcd692db929415fa1b27ff65ea94dbd475`. The
independent G3 audit passed: its record SHA-256 is
`3a2b8d353aa4e176e0dacb1bf9963cfb2995e905d51611b37641f96d5fbf8e5a`. It
validated the exact archive roster, source provenance, ten completed 262,144-
step fits, 100 replay ledgers, and their checkpoint-to-arm identity before any
economic payload was opened.

An independent G4 audit then reaggregated all 100 cells and 17,544 intervals
per cell from the exact artifact. It checked ledger equity/return chains,
preregistered start-year slices, recorded drawdown traces, terminal quantities,
active orders, and arm-cell equality without importing the frozen evaluator or
simulator. Its source-comparison recomputation is `MATCH`; both report
`KEEP_BASELINE`. The committed [G4 audit record](../../report/ppo-btc-relative-feature-ablation-g4-20260922.json)
has SHA-256 `cd01a193ca2a6fa34355873fdf21b62f5f425977b2d05a388259f69cf753f481`;
the [exact audit script](../../report/independent_ppo_feature_g4_audit.py) has
SHA-256 `1dc742dd479617e1aacca5251fd3c9e5a46951e8e284c562e877b52f40d86e9e`.

| G4 gate | Result |
|---|---|
| Candidate hard guards: drawdown at most 20%, no termination, terminal flat, no active remainder | PASS; zero violations; maximum across candidate base and stress cells 19.91485% |
| Absolute base profitability | FAIL; 0 / 5 symbols qualify |
| Paired relative screen | FAIL overall; ETHUSDT alone qualifies |
| Doubled-cost and one-bar-latency stresses | FAIL; 0 / 5 symbols qualify |
| At least four common symbols | FAIL; 0 qualify |
| G5 unused-future and production/live eligibility | NOT ESTABLISHED |

Median full-period returns (baseline → candidate) were BTC −14.11% → −14.48%,
ETH −13.18% → −6.66%, BNB −12.06% → −17.82%, XRP −5.09% → −17.38%, and ADA
−15.29% → −15.34%. Every candidate median and every baseline median was
negative. ETH showed relative improvement, but not positive absolute returns
or passing stress results. `KEEP_BASELINE` therefore means no feature-augmented
candidate qualified; it does **not** establish that the baseline is profitable.
The study reuses 2023–2024 development data and does not qualify a prospective
paper stage, a winner, or live trading.

The audit reaggregated P&L from portfolio-value snapshots persisted in these
ledgers. It did not fetch raw market bars or independently replay source fills,
so it does not establish a second market-data-to-execution oracle. The audit
used the persisted full-ledger maximum-drawdown trace and cross-checked it
against independently recomputed interval-end equity drawdown; it cannot
rebuild every intrainterval mark without the frozen Dataset. Annual returns
were grouped by registered interval-start year; the artifact's interval-end
year diagnostic is not the admission oracle.

The checkpoint implementation passed the full Linux repository, PPO runtime,
distribution, clean-install, and Guide browser checks at the implementation
head; its independent code review found no blocking integration issue. Hosted
execution uses the manual `ppo-feature-checkpoint.yml` workflow on standard
Ubuntu runners because the local host did not retain the required free memory.
It retrieves the frozen source from artifact `10331899302`, run `34803217815`,
with outer SHA-256
`89e899427f23fa46929c8be1e71fd49abe0d1d465c7a7f796a0874426b885bce`.
The publisher's verification stopped at a SHA-prefix comparison. Its existing
verification-only recovery, run `34803432434`, subsequently passed independent
source reconstruction and the verification audit for this same bundle.
Recovery artifact `10332575500` has outer SHA-256
`2067c38da3c1f45927748e2cc7481f11268710b2b386a5bd3b9a289394937589`;
its downloaded bytes, bundle binding, Dataset/Study identities, and explicit
no-P&L assertions have been checked. It does not replace the new study review.
The preparation artifact and independent review are bound to the same frozen
source, workflow revision, Linux runtime, and outer protocol digest. Each arm
dispatch reuses that binding, saves completed checkpoint evidence, and
preserves failed attempts. Run dispatch alone does not establish G3 or G4.
This transport does not import the original, r1, or r2 partial roots.
After the strategy-owned feature schema was integrated, the checkpoint replay
wrapper was updated to retain the loaded policy's feature names. A focused
counterexample failed before the forwarding fix; the real SB3 roundtrip also
checks that loaded and replayed policies retain the fitted schema. This is an
identity-preservation repair and does not change the registered feature roster.

### PPO 4h indicator smoke: preregistered, not yet executed

A new development-only smoke is preregistered for the single-symbol PPO path.
The user's shorthand is fixed before results as **MACD + ATR + DI± + Ichimoku
on the native 4h feature clock**. The selected observation roster is exactly
ten local features: three 4h MACD values, 4h ATR%, 4h +DI/-DI, and four 4h
Ichimoku distances/cloud descriptors. No symbol ID, cross-sectional feature,
normalization change, shared-cash training, reward change, threshold search, or
PPO hyperparameter change is part of this smoke.

The decision clock remains the maintained 1h clock. Only the selected input
features come from the causal 4h-native feature stream; changing the trading
clock to one action every four hours would be a separate factor and is not
mixed into this test. One common policy is fit across the full five-symbol
pre-2023 roster, while every 2023-2024 replay uses an independent 10,000 USDT
single-symbol account. The fixed seed is 0 and the requested training budget is
100,000 PPO steps. Base execution is accompanied by doubled-cost and one-bar
latency stresses under the maintained hard-risk contract.

This is intentionally a one-seed screening experiment on already-used
development data. It can only promote the hypothesis to a new preregistered
five-seed study. Promotion requires all hard guards in all 15 replay cells,
positive base return on at least four of five symbols, positive cross-symbol
base median, positive 2023 and 2024 base medians, and positive medians under
both fixed stresses. Otherwise the lineage stops after the smoke. Even a pass
does not establish general profitability, unused-data evidence, production
eligibility, or live-trading readiness. Economic execution remains blocked
until exact-head Full CI and a fresh result-blind G0-G2 review are bound to the
fixed contract. For this smoke, the formal review must be posted by a GitHub
principal distinct from the execution PR author. A same-principal canonical PASS
payload plus a valid annotated tag is insufficient because the author could create
both; the structured external-AI/fresh-read-only/independence fields remain required
semantic attestations in addition to principal separation. The concrete
`reviewer_model` value is recorded as non-empty provenance and is not hard-coded
as a validator allow-list. The preregistration/code PR has been integrated
without economic authorization, so the remaining authorization lifecycle is
carried by a dedicated **open draft execution PR** on
`research/ppo-4h-indicator-smoke-execution`. Draft state is a required machine invariant through review authorization and the evidence-only trigger transition, preventing the merge-before-review failure mode even while repository branch protection is unavailable. Review-status and trigger-time authorization also compare the reviewed code against live current `main`; if `main` advances, the old exact-head CI/review cannot be reused. The trigger gate
requires that PR to target `main`, belong to this repository, and have its head
equal the exact reviewed code SHA. The exact review `commit_id`, exact
review-body hash, and canonical `ppo_4h_indicator_source_review_v3` payload
must have matching code/contract identity, `review_tag`,
`review_tag_object_sha`, `reviewer_independence=ESTABLISHED`,
`reviewer_kind=external_ai`, non-empty `reviewer_model`,
`reviewer_context=fresh_read_only`, result-blind G0/G1/G2 outcomes, no blocking
findings, and development-only authorization before source download or PPO
fitting may start. The review tag is versioned as
`review/ppo-4h-indicator-smoke-vN` and authorization binds tag name + annotated
tag object SHA + reviewed commit SHA. GitHub API validation follows tag ref ->
annotated tag object -> commit and then refetches the tag ref, so lightweight
tags, deleted or retargeted tags, tags for another commit, and mid-check ref
mutation are rejected. Trigger evidence uses the versioned
`ppo_4h_indicator_smoke_review_v2` schema and repeats the same tag
name/object SHA; the pre-tag v1 trigger schema is not accepted for this
lifecycle. The review-body SHA-256 freezes the full canonical body, including
reviewer provenance. The gate also compares the authenticated review principal
against the execution PR author and rejects equality; the v3 fields and live tag
objects then provide the remaining machine-checked review-process attestation.
GitHub now exposes the waiting state through
an `Independent Research Review` check that runs only after Lean Core, real-SB3
PPO Runtime, and Human Guide verification succeed. Review submit/edit/dismiss
events repeat those checks and then refetch the current formal-review inventory;
only reviews passing the execution gate's canonical validator can satisfy the check.
The posting GitHub principal must currently have repository `write` or `admin`
permission; a public GitHub account without write authority cannot carry the
authorization record. The inventory is chronological and only each posting
GitHub principal's latest review is eligible, so a later blocking or otherwise
non-authorizing review from that same principal supersedes its earlier
authorization while another principal's later review does not rewrite it. If a
review is BLOCKED, its review tag is deleted and the review is not reused after
branch changes. The repaired exact HEAD must complete Full CI again before a
new versioned tag such as `...-v2` is created and reviewed; only the APPROVED
review tag is retained for the trigger. Because a Git ref has no trusted
creation-time authority, creating the review tag only after exact-head Full CI
is a procedural invariant rather than a timestamp check enforced by the
gate.
Before review authorization the
open execution PR head equals the reviewed code SHA; the only subsequent source
transition is the review-evidence-only trigger commit, which becomes the PR head
while the formal review stays bound to its parent code SHA. The trigger gate
refetches the bound review, current reviewer permission, and full review
inventory immediately before accepting the evidence transition; a superseded,
permission-revoked, or same-ID review whose URL/body/commit/state/login changed
between the individual fetch and inventory refresh is rejected. `PENDING` therefore means software verification completed but the exact-head
independent result-blind review is still absent or invalid; `READY` means only
that the authenticated review-evidence transition may proceed, not that PPO
economics may run directly.

The repository now has an A2C intent adapter, sequential CPU fitter, and
algorithm-specific inference bundle over the same `Discrete(3)` environment.
Synthetic CPU fit/save/load tests check the explicit config, rollout rounding,
nominal fit-budget metadata, feature binding, and A2C policy family. This is
software validation only: no research A2C candidate fit, replay,
PPO-versus-A2C comparison, or economic evidence exists. The next research step
remains an algorithm comparison before ensembling, using fixed data, features,
fit scope, account, and execution contract. Test DQN only as a later, separate
factor because it adds replay-buffer and exploration settings.
Do not combine policies unless independent candidates first pass the same
out-of-sample gates and their errors show useful complementarity.

Data improvement should also be isolated from the learner comparison. The
Binance aggTrades parser is not connected to the canonical Dataset feature
builder, and those records do not provide exchange publication or client
receipt times. Any signed-volume-flow feature therefore needs a documented
availability lag and verified raw coverage before it enters a study. Broader
regime and symbol coverage is preferable to adding many unverified indicators.

The separately preregistered corrected-accounting replication in Issue #645 then
ran the unchanged 13-arm screen from frozen source `c80652a12678` plus only
the accounting-correction source `95a4831bfbd5`. Official run `35287338444` on
workflow HEAD `6dcbde815de7` completed exactly once with all 13 immutable arm
slots and an independent no-refit publication audit. Its canonical selection is
`NO_QUALIFIED_CANDIDATE`, `winner=null`, and `production_eligible=false`; no
candidate qualified. Run-summary Artifact `10528400988` has API digest
`sha256:8d07b6b6119bf71416742c33ef3d2c0eb4fe14c8c6ce329208daa281f8950bc9`,
and independent audit Artifact `10527862342` has API digest
`sha256:2d6c88ee811a795543e1d772d3b5f4c7162661da38a706a6ee7ce3324913587b`.
This corrected-accounting development evidence does not alter the earlier pre-fix
Issue #640 result and does not validate the later #657/#658/#659
reduce-only/profile semantics now present on `main`. It accessed no unused/final
data and authorizes neither production nor live trading. Existing profitable
individual seeds do not authorize selecting a model or changing its qualification
gate.

### Directional exit diagnostics

A separate unchanged constant-long diagnostic has now completed all 17544
development intervals on the corrected accepted-lot ledger. Protocol
`666c864868f09dce66e3c1ae87dea934756ed00380177135aaeda08404ccee22`
binds source/runtime and an observer-only raw order/account trace. Net return
was +14.02045%, with ledger maximum drawdown 11.78340%, but exact final holdings
were BTC 0, ETH 0.004, BNB 0, XRP 0.1 and ADA 1. All three nonzero final exits
were rejected under the unchanged minimum-notional contract. This diagnostic
does not qualify the control or revise the original study.

An independent Fraction/Decimal audit reconciled all intervals, 193 fills and
2192 funding boundaries against the frozen dataset arrays, including exact
inventories, cash, costs, capacity and ledger drawdown. It verified source/helper
pins and the genuine final residuals. It did not reconstruct original exchange
archives, every admission decision, or tick-level funding eligibility. The result
SHA-256 is
`a9aaa8c028edaf808752160ef34f509867b6ee6d6b83666b16c2e50e345670d5`.

Its 2024 return was only +0.02473%: material positions had already been reduced
to residual amounts during 2023. Source inspection and a fixed-price projection
example show that feeding each constrained target into the next proposal applies
the historical drawdown scale repeatedly. At a fixed 15% drawdown, a 50% gross
proposal becomes 25%, 12.5%, 6.25%, and so on. This risk-contract investigation
is separate from the explicit reduce-only execution design in
`docs/specs/reduce-only-exits.md`. No risk change or minimum-notional exception
has yet been applied to a new economic comparison.

The common simulator provides explicit MARKET reduce-only orders, exact fill-time
inventory bounds and persistence/event evidence. An opt-in source-bound USD-M
one-way MARKET profile enables same-side reduction reconciliation and its
per-order venue-notional exception, retaining runtime minima and source quantity
bounds. False/true profiles support a matched comparison; omission retains default
behavior. Current exchange filters on historical bars are a declared assumption,
not point-in-time source recovery. The shared-cash directional evaluator accepts
the profile explicitly, records actual execution-policy identity and exact terminal
inventory, and now has observer-only interval ledger evidence for independent
reconciliation without changing replay economics. These are software contracts,
not economic improvement evidence.

Issue #661 preregistered the matched Stage C economic comparison but could not
establish its required current-rule source authority. Official source run
`35337609697` on Ubuntu and recovery run `35343362615` on macOS both stopped
at the first Binance USD-M `exchangeInfo` request with HTTP 451 and published no
source artifact. Two separately frozen transport recoveries also produced no valid
`exchangeInfo` bytes: a single direct GET failed before validation, and a single
live raw-body fetch returned Binance's HTTP-451 restricted-location error JSON.
No successful response was substituted, no alternate host/mirror was shopped, and
no model fit or ordinary/reduce-only economic replay occurred. The terminal Stage C
status is therefore `SOURCE_AUTHORITY_UNAVAILABLE`: the reduce-only economic
hypothesis is **untested, not rejected**. No unused/final data was accessed and
nothing from this stopped lineage authorizes production or live trading.

On 2026-09-17 the user selected 20% as a research drawdown tolerance. A separate
directional study uses the frozen successor Dataset from run 34803217815,
artifact 10331899302, fits before 2023, and screens 2023-2024 only. Its immutable
protocol is assembled by `directional_study.expected_protocol` and is retained
with the raw study evidence; the maintained contract is described here.
This study uses 10000 USDT simulated capital and one shared account; it does
not replace the earlier independent-symbol Study or reopen sealed experiments.
It compares the five maintained families, a fixed 20/10-day channel breakout,
and three controls. Sequential PPO uses five seeds and 262144 steps per seed;
the opt-in interleaved capability is outside this experiment's treatment.

Qualification requires positive full and both-year returns, ledger drawdown
at most 20%, complete replay, no termination, and actual terminal flatness.
Double-cost and extra-bar-delay stresses plus per-symbol diagnostics precede
prospective paper eligibility. PPO needs four of five seeds passing both base
and stress gates; medians always include all five. There is no profitable
candidate claim before the write-once selection artifact completes, and even
a qualified development candidate requires prospective paper evidence.

The channel entry uses the prior 480 hourly bars and exit the prior 240 bars,
excluding the decision bar from both extrema. Qualification ranks full net
return, then turnover, then fixed complexity order (trend, mean reversion,
channel breakout, ridge24, lightgbm24, PPO); controls cannot win. Risk reduction
starts at 10% historical maximum drawdown and requests flat at 20%, but price
gaps can exceed the bound and therefore fail the gate. A terminal next-open
mark/close at 2025-01-01 00:00 is the endpoint of the last 2024 interval; no later
2025 or 2026 evaluation is performed by these studies.

The write-once CLI entrypoints are
`python -m trade_rl.evaluation.directional_study` and
`python -m trade_rl.evaluation.ppo_risk_study`. Both expose `prepare`, `run`,
and `finalize` with required `--source` and `--output`; `run` additionally takes
one frozen `--arm`. The risk-only CLI also requires `--baseline` pointing to the
completed original study with its exact source snapshot and model/result hashes.
It accepts only that registered baseline, not an arbitrary profitable rerun.
An output directory or arm cannot be overwritten. Production source/runtime
must remain byte-identical between prepare and final publication; raw source
snapshots permit later independent inspection after main advances.

一つの銘柄ID非依存strategy/model/policyを学習・凍結し、各銘柄へ独立に適用する。その結果がpoint-in-time data、同一execution/accounting、hard risk、明示的なexecution-cost assumptionsの下でcontrolsを超え、unused dataでも再現するかを検証する。

Aggregate P&Lだけで成功を判定せず、各symbolの結果とraw interval returnsを保持する。

## 初期比較: 5 candidates + 3 controls

| 名前 | 系統 | 現行役割 |
|---|---|---|
| `trend` | rule | 単一signalのtrend + hysteresis |
| `mean_reversion` | rule | 単一signalのmean reversion + hysteresis |
| `ridge24` | forecast | universal Ridge 24h forecast + 共通controller |
| `lightgbm24` | forecast | universal shallow LightGBM 24h forecast + 共通controller |
| `ppo` | RL | universal teacher-free PPO |
| `cash` | control | 常時FLAT |
| `constant_long` | control | 常時LONG |
| `constant_short` | control | 常時SHORT |

初回比較へTransformer、SAC、TD3、TQC、複数horizon ensemble、大規模hyperparameter gridを追加しない。

判断原則:

- ruleが同等以上ならruleを優先する。
- forecastが明確に優位でPPOが上乗せしないならforecastを優先する。
- PPOはunused dataでも単純候補へ安定して上乗せする場合だけ残す。
- 差が不明ならより単純な候補を残す。
- 全候補が弱ければ **no winner** を正しい結論とする。
- controlsはbenchmarkであり、Study winnerとは呼ばない。controlが最良ならno winnerである。

## Universal fit contract

Universal model/policyにsymbol ID、symbol-specific embedding、symbol-specific coefficientを初期状態では入れない。

共通条件:

- 同じfeature schemaを使う。
- `fit_symbol_names` で事前登録した銘柄だけをfitへ使う。
- fit scope外の銘柄をtraining row/episodeへ混ぜない。
- content-verified Datasetでfit scopeを全symbolより狭める場合は、selected featureの情報依存もfit scope内へ閉じる。cross-sectional rank / dispersionのようなuniverse-dependent featureはsubset fitでrejectし、reference-relative / correlation / betaはreference symbolがfit scope内にある場合だけ許す。FeatureKind provenanceを復元できないverified subset fitはfail closedとし、identity provenanceのないlegacy/synthetic経路だけをunseen-symbol isolationの証拠には使わない。
- 同じfit cutoffを使う。
- 同じfrozen strategy/model/policyを評価対象の各銘柄へ適用する。
- 評価symbolごとにfresh strategy/controller wrapperを生成し、学習済みmodel/policy weightだけを共有する。前symbolのwrapper内部状態を次symbolへ持ち越さない。
- 評価は各銘柄を独立portfolioとしてReplayする。

### Ridge / LightGBM

Eligible row数の多い銘柄がtrainingを支配しないよう、各fit symbolの総sample weightを等しくする。Ridgeはweighted statistics/normal equationを使い、LightGBMは同じweightを`sample_weight`へ渡す。

### PPO

初回real-data M2のPPO Observation v2は、selected local feature values、availability / finite mask、normalized local staleness、current intent、current weightだけをこの順序で使う。symbol IDは含めない。Training episodeはfit symbolをround-robinし、各episodeは一つのactive symbolだけを扱う。

現行datasetのglobal regimeは全dataset symbolから集計されるため、fit-symbol subset外の情報がtrainingへ混入しないよう初回M2のpolicy inputから除外した。Observation v2は空のglobal rosterをsemantic identityへ明示bindする。global contextは、fit-scope-safeなreference universeを事前固定できる場合にだけ別Controlled Factorとして検証する。

PPO fitの既定layoutは既存互換の`sequential`である。複数fit symbolを宣言するsequential fitでは、SB3の2048-step rollout丸め後の実効budgetが全fit symbolへ最低1 full agent episodeずつ届くことをfit前に要求し、後半symbolが0 transitionになる設定をfail closedにする。これは最低coverage保証であり、完全なsample均等化や性能改善を意味しない。実装上はopt-inの`interleaved`も選べ、fit symbolごとのfixed-symbol `PPOTradingEnv`を`DummyVecEnv`へ束ね、明示した`rollout_steps_per_env`ごとに全envからrolloutを集める。 `PPOTradingEnv`は実際にepisodeで売買するactive symbol scopeと、selected featureが参照してよいinformation symbol scopeを分離する。direct constructionではinformation scopeを省略するとactive scopeと同一として検証し、fitter経由ではsequential/interleavedとも全fit symbol rosterをinformation scopeとして明示する。したがってinterleavedの各slotが1銘柄activeでも、fit内reference-relative featureを誤ってrejectせず、fit外symbol依存は#707の共有validatorで拒否する。これは学習sample schedulingだけを変えるunevaluated capabilityであり、Observation v2、reward、hard risk、network、entropy係数を変更しない。Directional PPOではfitとdevelopment replayが同じbase execution economicsを共有し、Dataset由来のborrowを両方で課す。さらにcurrent directional fitはfinite-horizon末尾を無料resetにせず、latencyを考慮して最後のagent decision後にcanonical FLAT settlement区間を予約する。settlementはagent actionではなくenvironment terminal transitionとしてrisk/executionを通し、内部settlement barへ追加discountを掛けず、その実現log wealth changeをterminal rewardへ加算する。capacity等で残余が残ればflatと偽装しない。これはper-symbol training endpointの補正であり、shared-cash evaluationとのcross-symbol accounting差は別途残る。旧interleaved prereg/evaluatorはこのborrow修正前のimplementation authorityへbindされているため、current economicsでの実行authorityとしてはobsoleteであり、結果を見ずにfresh protocolを作り直す必要がある。なおSB3はwhole rollout単位で学習するため、同じcaller `total_timesteps`でもlayoutごとのrealized `model.num_timesteps`はわずかに異なり得る。実比較では両方をevidenceへ保存する。また、vector envのreset seed差がexecution randomnessへ混入しないよう、interleavedは`slippage_std > 0`を拒否する。

PPOのconstructor/policy constructionについて、current implementationが実際に依存する主要defaultはsourceへ明示bindする。対象はlearning rate、rollout長、batch、epoch、discount/GAE、clip、advantage normalization、entropy/value係数、gradient clip、gSDE/target-KL、およびMlpPolicyのTanh・orthogonal init・FlattenExtractor・shared extractor・Adam epsである。これは値を変更する探索ではなく、pinned runtimeで既に有効だった値をsource contractへ昇格する変更である。一方、SB3/PyTorch内部algorithm implementationまでrepositoryへ複製したわけではないため、library versionとruntime provenanceは引き続きtraining implementation identityの一部であり、dependency変更時のsemantic equivalenceを自動仮定しない。

## Causality and evaluation rules

- `feature_available_time <= decision_time` を守る。
- supervised labelは `label_end_time < fit_cutoff` で完結する。
- future由来のscaler/normalization/imputation/feature selectionを禁止する。
- fit symbol subsetを使うtrainingでは、そのsubsetを情報scopeとして扱う。content-verified Datasetのselected cross-asset featureがholdout symbol universeへ依存する場合はfit前にrejectし、reference-dependent featureはreference symbolがfit scope内にある場合だけ許す。verified transformで元build configを追跡できないstrict subsetもfail closedにする。
- development/final期間をfitやthreshold調整へ戻さない。
- 同calendar shockを受ける複数銘柄を完全独立標本とみなさない。
- 同じfrozen strategyを各symbolへ独立Replayし、`UniversalStrategyComparison.by_symbol`を主要結果として扱う。

各symbol × strategyでは少なくともtotal return、Sharpe、Sortino、maximum drawdown、turnover、total execution cost、funding P&L、borrow cost、trade/rebalance/termination diagnostics、raw interval returnsを保持する。

## Execution economics contract

Execution economicsはfeature configurationではなくDataset environment semanticsである。

- build-level authorityは`trade_rl.data.build.ExecutionEconomicsProfile`が持つ。
- `MarketBuildConfig`はfeature/build authorityのまま維持する。
- profile省略時はlegacy build behavior/content identityを維持する。
- 明示profileは既存economic-semantics経路を通り、immutable Dataset economic arrays/content identityへbindする。
- Canonical M2 bootstrap v1のreader/payload/digest互換は維持する。
- Canonical M2 bootstrap v2は明示的なexecution economicsを必須とし、zero economicsへのsilent fallbackを禁止する。
- runtimeは`zero_overlay_dataset_fields_authoritative`を維持し、Dataset economicsへ第二のcost overlayを重ねない。

初回real-data M2で採用したeconomicsは再現可能なresearch assumptionであり、historical/account-specific venue truthではない。static spread、zero borrow cost、Dataset-authoritative market-impact/slippage不在は残存realism limitationである。

## M1 / M2 / M3

### M1 — Lean core: complete

実装済み:

- causal/point-in-time `MarketDataset`
- deterministic filesystem dataset artifact
- canonical `MarketExecutor + BookState` accounting
- hard-risk projection
- quantity-preserving independent symbol replay
- DB/UI/teacher pipelineなしで成立するcore CI

### M2 — Canonical real-data baseline and Controlled Experiment 0001 verified

実装・検証済み:

- 5 candidates + 3 controls
- universal Ridge / LightGBM / teacher-free PPO fit
- symbol-balanced supervised fit
- fit-symbol scopeの明示
- symbol-ID-free PPO
- fit-scope-safe PPO Observation v2（local values + availability/finite mask + normalized staleness + portfolio state、global policy rosterは空）
- `lean_candidate_result_v2`によるObservation contractのRun evidence bindingとhistorical v1 reader互換
- `resolved_run_config_v2`によるStudy identity binding、historical v1 read互換、v1 Studyへのv2 mutation拒否
- 全symbol独立comparison
- shared candidate config resolution
- in-memory candidate execution seam
- implementation/runtime/research-context provenance生成
- immutable 3-file candidate-run artifact
- semantic candidate-artifact identityとraw file integrity evidence
- append-only Study/Experiment state machineとprocess-safe mutation lock
- Study-owned multi-seed EvidenceSetとdeterministic seed invariance
- one-factor resolved delta verificationとunaffected-strategy raw-return invariance
- paired/bootstrap/seed analysis、ACCEPT-only lineage、FAILED/INVALID terminal、WINNER/NO_WINNER freeze
- strict `CanonicalM2BootstrapConfig` とsingle `ppo_seeds` authority
- exact Binance exchange-info / Vision plan / raw archive rosterのfreeze
- source同期後のcache-only network cut
- canonical dataset artifact + immutable StudyPlanのwhole-root atomic publication
- bootstrap manifestによるsource / dataset / Study / provenance identity binding
- `inspect_canonical_m2_bootstrap` によるnetwork-free再検証
- build-level execution economicsのDataset identity bindingとbootstrap v2 closure
- `market_build_v3` / `portable_feature_numerics_v1` によるCPU-portableなidentity-bound feature numerics
- persisted `market_build_v2` artifactのhistorical reader互換
- heterogeneous AMD / Intel hosted runnerでのfull priced Dataset byte identity一致
- portable Dataset / StudyPlanの結果前plan-only preregistrationと独立再構築
- real-cost-assumption portable Datasetを使ったbaseline-only Studyの実行
- baseline Artifactを別runnerで再取得し、Dataset / StudyPlan / EvidenceSet / raw Candidate Runsをpublication indexに依存せず独立再構築・再計算するpost-Artifact verification

現在のcanonical portable baselineで確認したこと:

- build semanticsは `market_build_v3` / `portable_feature_numerics_v1` で、hash-only roundingやtolerance弱体化は使わない。
- same sealed sourceからのfull priced Dataset identityはheterogeneous AMD / Intel hosted runner間でbyte-identicalに再現した。
- 結果を見る前のplan-only preregistrationはrun `34702660287`、Artifact ID `10300479733`、outer digest `sha256:60127cc2f24c7e8b3dcb5c6157ca49a5dd2f5b60c44f1020e4d540405e34e1f4` としてbaseline結果より先に封印した。
- portable Dataset IDは `d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f`、Dataset artifact digestは `77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8`、Study digestは `3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79` である。
- portable baselineはrun `34700123151`、Artifact ID `10301701698`、outer digest `sha256:f814fe4e205f8714c4344238911aae16e89ce0279908265feb1fdc85069b2a0a`、EvidenceSet fingerprint `526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29` である。
- preregistrationとbaselineは同じDataset artifactと**完全に同一のStudyPlan**へbindされ、preregistrationはplan-only、baselineはbaseline-onlyのままでExperiment countは0、未freezeである。
- preregisteredな5 PPO seedsと5 symbols × 8 strategiesの完全なbaseline evidenceが存在する。
- tradeが発生した175 observationsすべてで`total_cost > 0`、cash 25 observationsはzero-trade / zero-cost / zero-return、aggregate realized trading costは正である。
- fresh post-Artifact verifier run `34704606059` はsealed source、portable preregistration、baseline Artifactを再取得し、Dataset / StudyPlanを独立再構築したうえでraw Candidate Runsからreturn・cost・seed invarianceを再検証した。verifier Artifact IDは `10300932825`、outer digestは `sha256:77c26e8cdeec023a750582ec5aebe3870ca84728e580c9e27d1b2e420368a9` である。
- Portable Controlled Experiment 0001 (#511) は結果前preregistrationを封印・fresh verificationした唯一のcandidate EvidenceSetを再実行せずに完遂し、fresh runnerで公開result Artifactを再取得して独立再検証した。formal decisionは`KEEP_BASELINE`。mean-reversionのfactor effectは5 / 5 symbolsで正、median excess total returnは`+0.16996869069426646`、candidate positive-total-return symbolsは1 / 5、candidate median turnoverは`467.48617120292243`（baseline `858.3114067468092`）だった。unaffected raw-return equality 150 checks、deterministic metric seed invariance 1120 checks、tradeあり175 / 175 positive-cost、cash 25 zero-trade / zero-cost / zero-returnもGreenである。
- publication indexは独立再構築後のcross-checkにだけ使い、結果のoracleにはしていない。
- `research/m2-canonical-study-004` はpre-portable `market_build_v2` numericsで生成されたimmutable historical evidenceとしてhead/treeを維持するが、current canonical inputとしてはportable successorにsupersedeされた。旧Studyを書き換えたり削除したりしない。
- pre-portable Study 004 Experiment 0001 (#498) はfail-closedし、result Artifactも結果解釈も存在しないhistorical lineとして保持する。portable lineageのExperiment 0001は結果前preregistrationからfresh result re-verificationまで完了し、`KEEP_BASELINE`をdevelopment decisionとして固定した。
- PPO cross-seed candidate-metric aggregation defect (#476) はcurrent mainで修正済みだが、Issue #511は修正前にfreeze/startしたimplementation provenanceを維持する。PPO aggregate candidate metricsはformal decisionのoracleに使わず、raw-return equality controlとしてのみ扱う。
- baseline成立はresearch environment / identity / evidence pathの検証であり、profitability、winner、Production readinessを意味しない。

未完了:

1. Experiment 0001の`KEEP_BASELINE`を維持し、次のControlled Factorを結果を見る前にpreregisterする。
2. 次のExperimentでもone-factor delta、unaffected-strategy raw-return invariance、deterministic metric seed invariance、cost/cash semantics、fresh post-Artifact verificationを必須にする。
3. development Experimentを事前登録して継続し、最終的にwinnerをfreezeするかno-winnerと判断する。
4. winner候補が成立した場合だけsealed unused-future / final-testへ進み、Production/live tradingとは引き続き分離する。

**Portable Canonical real-data baselineは結果前preregistrationからpost-Artifact独立検証まで完了し、Portable Controlled Experiment 0001もfresh result re-verificationまで完了してformal decisionはKEEP_BASELINEである。** このExperimentは相対改善とturnover低下を示したが、candidate profitabilityやwinnerを成立させなかった。Baseline成立やKEEP_BASELINE decisionはProduction readinessの証拠ではない。

### M3 — Finalize and delete: not started

M2で候補をfreezeした後だけ進む。

1. 未使用future / zero-shot評価を一度だけ開く。
2. pre-registered stressを実行する。
3. 支持されなかったstrategy familyと専用test/extra/dead adapterを削除する。
4. README/config/CI/testsを採用構成へさらに縮約する。
5. Production認可は研究結果とは別に扱う。

## Canonical M2 bootstrap

Canonical M2 bootstrapはresearch runそのものではなく、real development Studyの入力を固定するpreparation stepである。

入力JSONは少なくとも次を事前登録する。

- Binance USD-M market
- ordered symbol roster
- base timeframe / feature timeframes
- exact data start / exclusive stop
- baseline signal/features/fit symbols/fit cutoff/development window
- rule / forecast thresholds
- PPO training budget
- ordered `ppo_seeds`
- gross budget / initial capital
- allowed controlled factors / experiment budget
- bootstrap count / seed
- bootstrap v2では明示的なexecution economics profile
- final-eligibleな新規Studyを作るbootstrap v3では、さらにunused `final_evaluation_start` / `final_evaluation_stop_exclusive`

baseline JSONに`ppo_seed`は持たず、`ppo_seeds[0]`だけがbaseline seed authorityである。

実行入口:

```bash
uv run --extra forecast-gbm --extra train-sb3 \
  python -m trade_rl.evaluation.experiments.bootstrap.cli \
  --config <canonical-bootstrap-config.json> \
  --output <new-bootstrap-dir>
```

このCLIは `bootstrap_canonical_m2_study` を呼ぶ薄いfilesystem adapterである。`--output`は存在していてはならない。

成功したbootstrapは概ね次を持つ。

```text
<new-bootstrap-dir>/
  bootstrap.json
  bootstrap-manifest.json
  source/
    exchange-info/{exchange-info.raw.json,manifest.json}
    vision-plan.json
    vision-cache/**
  dataset/{manifest.json,arrays.npz}
  study/{plan.json,.mutation.lock}
```

source同期後のdataset buildはcache-onlyで、missing cacheをnetwork fallbackで補わない。whole rootはstaging内で完成・検証してから最後にrenameする。成功直後のStudyはPlanだけで、baselineはまだ実行されていない。

Inspection:

```python
from trade_rl.evaluation.experiments import inspect_canonical_m2_bootstrap

result = inspect_canonical_m2_bootstrap("<new-bootstrap-dir>")
```

Inspectionは保存されたconfig、source roster、dataset artifact、StudyPlan、bootstrap manifestをnetwork-freeで相互検証する。bootstrap v2では、configured economics、生成/reloadしたDataset economic arrays、identity-bound profileの一致もfail-closedで確認する。

## Development Run Core

必要な入力は次の3つである。

1. canonical filesystem dataset artifact
2. JSON run config
3. 存在していない新しいoutput directory

現行runnerが受理するconfig keyは次である。

```json
{
  "signal_name": "<rule signal feature name>",
  "feature_names": ["<feature A>", "<feature B>"],
  "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
  "fit_cutoff": "2026-01-01T00:00:00",
  "evaluation_start": "2026-01-01T00:00:00",
  "evaluation_stop_exclusive": "2026-02-01T00:00:00",
  "rule_entry_threshold": 0.10,
  "rule_exit_threshold": 0.02,
  "forecast_entry_threshold": 0.01,
  "forecast_exit_threshold": 0.002,
  "ppo_total_timesteps": 100000,
  "ppo_seed": 0,
  "gross_budget": 0.5,
  "initial_capital": 100000.0
}
```

`evaluation_start` と `evaluation_stop_exclusive` はdataset timestampへexact matchする必要がある。Evaluation startはfit cutoffより前にできない。

Standalone実行:

```bash
uv run --extra forecast-gbm --extra train-sb3 \
  python -m trade_rl.evaluation.runs.candidate \
  --dataset <dataset-artifact-dir> \
  --config <run-config.json> \
  --output <new-result-dir>
```

Controlled StudyでRunを事前登録contextへbindする場合は、higher-level workflowから同じRun CoreへSHA-256 `research_context_digest`を渡す。CLIにも `--research-context-digest <sha256>` がある。Standalone Runでは省略できる。

既存output directoryへの上書きは拒否する。実行前後でimplementation/runtime provenance digestが変化した場合もpublishしない。

出力:

```text
<new-result-dir>/
  summary.json
  returns.npz
  provenance.json
```

- `summary.json`: dataset artifact identity、resolved run config/scope、各symbol × strategy metrics/diagnostics。
- `returns.npz`: paired comparison/block-bootstrap等に使うraw interval-return series。`allow_pickle=False`で検証可能なnumeric 1D arraysだけを正当なevidenceとする。
- `provenance.json`: exact Python source-byte manifest、runtime/dependency roster、optional research-context digest。

3ファイルを一つのimmutable Run evidenceとして扱う。Semantic artifact identityはNPZ ZIP compressionの違いでは変えず、検証済みarray content、summary、provenanceへbindする。raw file SHA-256/sizeはtamper検出用evidenceとして別に保持できる。

## Development後の判断順序

結果を見たら、architecture/model sizeを増やす前に次を確認する。

1. controlsより本当に上か。
2. 全symbolでreturn符号、drawdown、costが許容可能か。
3. 特定一銘柄だけが利益を作っていないか。
4. turnover/costでgross edgeが消えていないか。
5. ruleとforecastの差は十分か。
6. PPOが単純候補へ本当に上乗せしているか。
7. raw returnsが特定time block/regimeだけに依存していないか。
8. 差が弱ければ単純側を残す。
9. 全て弱ければno-winnerとする。

No-winnerの後に最初に疑う順序は、model sizeではなく **information source → feature causality/quality → horizon → regime hypothesis → period → model complexity** とする。

PPO controlled comparisonのcross-symbol candidate metricsは、各symbol内でfrozen seedを先に集約してからsymbol間summaryへ進む明示契約へ更新した。total return / turnover / total costはseed中央値、maximum drawdownはseed内worstを使い、factor effectは従来どおりseedごとのpaired excessの中央値をsymbol代表値とする。新規factor-effectは`controlled_evidence_comparison_v2`、persisted v1はhistorical first-seed semanticsでinspection互換を維持する。

## Final unused-data / stress protocol

Developmentで繰り返し見た期間をfinal testと呼ばない。Candidate、feature、threshold、seed policy、cost/riskをfreezeした後、未使用futureを一度だけ開く。

採用候補には事前固定したstressを適用する。

- fee adverse
- spread adverse
- impact adverse
- +1 decision latency
- capacity reduction
- initial capital sensitivity
- funding / borrow coverage check

Stress結果を見てから合格thresholdを変更しない。

Controlled Experiment Loop自体からsealed unused-futureを開かない。Development StudyをWINNER/NO_WINNERへfreezeした後、別subsystemでのみfinal authorizationを扱う。

research-governance側では `RESEARCH-001` の機械可読化として、`StudyResearchContext` / `ConsumedEvidence` と `controlled_study_plan_v3` を追加した。新規Studyは、仮説・observation/model/hyperparameter/evaluation design/result interpretationへ使った既知development evidenceのdigest、canonical time scope、利用目的とparent context digestをStudy identityへbindできる。新規final-eligible research lineは `canonical_m2_bootstrap_config_v4` でfinal windowとresearch contextをresult前に固定し、final startがdevelopment Datasetまたは申告済みconsumed-evidence scopeの終了以前にある場合はfail closedにする。historical `canonical_m2_bootstrap_config_v3` / `controlled_study_plan_v2` を含む既存artifactは当時の意味を維持し、contextを後付けして再分類しない。この機構は申告済みconsumptionを固定するもので、研究者やAIが閲覧した全情報の完全な申告を自動証明するものではない。

現行codeには、final境界として `trade_rl.evaluation.final_test` の**authorization capabilityだけ**がある。frozen `WINNER` のStudyPlan/StudyFreeze/winner evidence/winner strategyと未使用windowをcanonical one-shot artifactへbindするが、final Datasetを読まず、P&L/stressを実行しない。したがってM3 final economic evaluation自体は未実行であり、authorization capabilityやresearch-context infrastructureのGreenをfinal evidenceとして数えない。

## Superseded evidenceの扱い

旧zero-cost canonical Studyは削除・再解釈せず、diagnostic evidenceとして保持する。新しいreal-cost Dataset / Study / EvidenceSetは別identityであり、旧Studyをin-place mutationしていない。

PPO Observation v2確定前に生成されたeconomics-only baselineもdiagnostic evidenceとしてのみ扱う。Current canonical M2のStudy semantic identityを満たさないため、Controlled Experimentのbaselineとして採用しない。

## 現在の次アクション

現時点の次アクションは、Run Coreやbootstrap toolingをさらに拡張することではない。

> 現在のcanonical baselineをimmutable inputとして、一つのControlled Factorを結果を見る前に事前登録し、最初のControlled Experimentを実行・独立検証する。

旧teacher-selection runのrejectは旧mandatory teacher経路を再採用する根拠でも、現候補のprofitabilityを示す証拠でもない。現在の候補は現在のlean contract上で改めて評価する。
