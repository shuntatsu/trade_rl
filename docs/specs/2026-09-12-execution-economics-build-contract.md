# Execution Economics Build Contract

Status: Active

## Objective

Canonical M2 を実コスト前提で再bootstrapできるように、market dataset build に明示的・決定論的な execution economics contract を追加する。

Execution economics は feature configuration ではなく dataset environment semantics として扱い、zero execution overlay の Candidate Run が dataset の fee / spread / participation / borrow semantics をそのまま使用する。

旧 Canonical M2 Study とその immutable artifacts は変更しない。旧 Study は zero-trading-cost diagnostic evidence として保持する。

## Current defect

現在の `MarketDatasetBuilder` は `build_market_economic_semantics()` を呼ぶ際、fee / maker fee / taker fee / spread / max participation / borrow を渡していない。そのため defaults により次が dataset へ入る。

- fee = 0
- maker fee = 0
- taker fee = 0
- spread = 0
- max participation = 1
- borrow available = true
- borrow rate = 0

Canonical M2 Study は `zero_overlay_dataset_fields_authoritative` を採用するため、この dataset は事実上 zero-trading-cost Study になる。

## Responsibility boundary

### Build-level authority

新しい `ExecutionEconomicsProfile` を `trade_rl/data/build/economics.py` に置く。

この型が次を所有する。

- profile name
- generic fee rate
- maker fee rate
- taker fee rate
- spread rate
- max participation rate
- borrow availability
- borrow rate
- schema version
- strict payload parse / canonical payload
- `build_market_economic_semantics()` に渡す値

`MarketBuildConfig` には execution economics を追加しない。`MarketBuildConfig.canonical_payload()` は feature/build semantics の正本であり、execution-cost-only change で `feature_config_digest` を変えてはならない。

### One data flow

```text
JSON / Canonical bootstrap v2
        ↓
ExecutionEconomicsProfile
        ↓
MarketDatasetBuilder.build(..., execution_economics=profile)
        ↓
build_market_economic_semantics(...)
        ↓
immutable MarketDataset economic arrays
        ↓
dataset content identity / artifact
        ↓
zero-overlay executor
```

ad-hocな第二代入経路は作らない。

### Legacy behavior

`execution_economics` を指定しない通常buildは現行defaultsをそのまま使用する。既存dataset content identityを変えない。

## Fee combination contract

現在の executor は generic `fee_rate`、order type に応じた maker/taker fee、spread component、impact/slippage overlay を加算する。

したがって profile では、generic fee と maker/taker fee を同時に正値にすることを禁止する。

1. generic fee model: `fee_rate >= 0`, maker/taker = 0
2. venue-specific fee model: `fee_rate = 0`, maker/taker >= 0

すべて0も有効とする。negative maker rebate は現行 execution-cost contract が非負rateのみを許すため今回扱わない。

## Spread contract

`spread_rate` は既存executorが使用する per-notional spread cost input とする。executorの現行規則を変更しない。

- market / stop-market: multiplier 1.0
- limit: multiplier 0.5

## Borrow contract

`borrow_available` と `borrow_rate` を dataset environment semantics として明示可能にする。`borrow_available=false` の場合 `borrow_rate` は0でなければならない。

Canonical M2 は Binance USDⓈ-M linear perpetualを対象とするため、research assumption profile では `borrow_available=true`, `borrow_rate=0` とする。perpetual financing は既存 funding semantics で扱い、spot-style borrow costを新たに推定しない。

## Canonical M2 bootstrap evolution

既存 `canonical_m2_bootstrap_config_v1` は immutable historical bootstrap を再読できるよう変更しない。

新しい real-cost canonical lineage では `canonical_m2_bootstrap_config_v2` を使い、top-level `execution_economics` を必須にする。

同じ `CanonicalM2BootstrapConfig` reader は v1/v2 を明示dispatchする。

- v1: `execution_economics` を持たない。旧payload/digest semantics維持。
- v2: `execution_economics` が必須。profile canonical payloadをbootstrap identityへbindする。

v2が profile 不在で zero economics へfallbackすることは禁止する。

## Canonical M2 research assumption profile

新しい canonical real-cost baseline には、既存 `ExecutionCostConfig` のmaintained defaultで使われてきた研究用仮定を dataset authorityへ移す。

```text
name                   = canonical_m2_research_assumption_v1
fee_rate               = 0.0005
maker_fee_rate         = 0.0
taker_fee_rate         = 0.0
spread_rate            = 0.0002
max_participation_rate = 0.05
borrow_available       = true
borrow_rate            = 0.0
```

これは account-specific / historical Binance fee schedule の事実主張ではない。再現可能な research assumption である。

現在dataset schemaに impact/slippage arrayはないため、新canonicalでも impact/slippage は zero overlay のままになる。この限界を明示し、historical execution realismとは呼ばない。

## JSON market build contract

`load_market_build_request()` のrootに optional `execution_economics` objectを追加する。objectは shared profile parserを使ってstrict parseし、未知fieldはfail closedする。

`MarketDatasetBuildRequest` は profile を `MarketBuildConfig` と分離して保持する。

## Binance build contract

`build_binance_market_dataset()` は optional `execution_economics` を受け取り、`MarketDatasetBuilder.build()` へそのまま渡す。metadata/source acquisition、feature preset、InstrumentContractはこのprofileを所有しない。

## Identity and provenance

Dataset IDは既に economic arraysをcontent identityへbindしているため、economics-only changeで dataset IDは変わる。

`feature_config_digest` と normalization content/digestは economics-only changeで不変でなければならない。

profileのcanonical payloadは、明示profileがある場合だけ dataset identity metadataへ `execution_economics` として追加する。profile省略時はmetadataも追加せず、legacy dataset IDを維持する。

## Artifact contract

既存 `MarketDataset` artifact schemaがeconomic arraysを保持していることを利用する。schema追加やsecond artifactは作らない。

## Acceptance Criteria

1. Profile省略時のmarket build behaviorとdataset identityが変更前と一致する。
2. Strict JSON build requestからprofileを指定でき、unknown fieldは拒否される。
3. Profileは唯一のBuilder経路から`build_market_economic_semantics()`へ届く。
4. Dataset economic arraysがprofile値を保持する。
5. Economics-only changeでdataset IDが変わる。
6. Economics-only changeでfeature_config_digestとnormalization digest/contentは変わらない。
7. Artifact round-tripがeconomic arraysとdataset IDをexact保持する。
8. Zero overlay replayでtradeが発生した場合、non-zero fee/spread profileによりnon-zero total_costが発生し、既存executor式と一致する。
9. generic feeとmaker/takerのdouble countをvalidationで禁止する。
10. Profile外のstrict build-config unknown-key拒否を維持する。
11. Bootstrap v1のpayload/digest/read compatibilityを維持する。
12. Bootstrap v2ではprofile必須で、profileはbootstrap identityへbindされる。
13. 新Canonical M2 bootstrap/baselineは旧zero-cost lineageとは異なるdataset/artifact/Study/EvidenceSet identityを持つ。
14. 旧Study/artifactsを変更せず、zero-trading-cost diagnosticと明示する。

## Invariants

- Frozen research SHA / old Study / old artifacts are immutable.
- `MarketBuildConfig` remains feature/build authority.
- Dataset ID remains content-addressed to resolved economic arrays.
- Funding remains separate from fee/spread/borrow.
- All strategies/seeds in one EvidenceSet share the same dataset economics.
- Study controlled-factor isolation is unchanged.
- Historical bootstrap v1 remains readable.
- No new runtime execution default silently overrides dataset authority.

## Failure Modes

- JSON profile is parsed but dropped before Builder.
- Builder profile affects replay but not dataset identity.
- Economics contaminates feature/normalization digest.
- Omitting profile changes legacy dataset ID.
- Generic fee and maker/taker are both charged.
- Spread is multiplied with a new convention inconsistent with executor behavior.
- max participation remains 1.0 despite profile.
- positive spot-like borrow cost is introduced to USDⓈ-M without evidence.
- Artifact reload drops economic arrays.
- v1 bootstrap reader breaks.
- v2 canonical bootstrap accepts missing economics or falls back to zero.
- Old Experiment 0001 is accidentally treated as evidence in the repaired lineage.

## Test Oracle

Observe directly parsed profile, strict payload round-trip, Builder arrays, dataset ID, feature/normalization digests, published/reloaded artifact, executor interval/total cost, bootstrap v1 compatibility, bootstrap v2 identity, and new baseline cost diagnostics.

## Required Test Layers

- Unit/contract: profile validation and strict parser.
- Integration: Builder arrays, identity separation, Binance builder plumbing.
- Artifact: publish/load exact round-trip.
- Replay/executor: deterministic one-trade cost regression using zero overlay.
- Bootstrap: v1 compatibility, v2 required profile, end-to-end plumbing.
- Static: Ruff / Format / Mypy / architecture checks.
- Distribution: build, source closure, clean install, CLI/package identity.
- Falsification: mutate/drop/double-count/profile fallback cases.
- Research artifact verification: new canonical baseline with observed non-zero cost.

## Quality Gate

Do not mark complete unless all Acceptance Criteria are evidenced, final exact-head CI is green, current main is included, temporary spec/plan/workflows are cleaned under current-only docs policy, old frozen evidence is unchanged, and remaining realism limits are recorded.
