# Canonical M2 Study Bootstrap v1

Status: Active

Date: 2026-09-09

## 結論

Canonical M2 Study Bootstrap v1 は、Binance USD-M Futures の実データを **pre-registered config → exact source freeze → canonical MarketDataset artifact → immutable Controlled StudyPlan** まで一つの再現可能な手順で準備する。

このbootstrapは研究結果を生成しない。`run_baseline`、Experiment実行、winner/no-winner判断、sealed unused-futureへのアクセスは行わない。成功時の終端は **「datasetとStudyPlanがfreezeされ、baseline実行前」** である。

既存の責務を再実装しない。

- raw market acquisition / Vision cache: `trade_rl.integrations.binance`
- canonical dataset build/publication: `trade_rl.data`
- StudyPlan resolution/immutability: `trade_rl.evaluation.experiments`
- candidate execution: 既存 `evaluation/runs`。bootstrapでは実行しない

新しい責務は、これらをpre-registration契約の下で順序付け、source/network driftを閉じ、最終bundleを一度だけpublishすることだけである。

## 1. Objective

実M2 development Studyを開始する前に、次を同一のimmutable bootstrap bundleへbindする。

1. strict pre-registration config
2. exact Binance exchange metadata bytes
3. 使用した全Binance Vision archive bytesとcontent evidence
4. canonical `MarketDataset` artifact identity
5. resolved baseline Study config
6. Study seed policy / experiment budget / allowed factor policy
7. implementation/runtime provenance

これにより「どのsource bytes・dataset・config・code/runtime条件をfreezeしてStudyを開始したか」を後から再構築できるようにする。

## 2. Non-goals

v1では次を行わない。

- `run_baseline`
- Experiment definition / execution / verification / comparison / decision
- feature自動探索
- threshold grid/random search
- symbol自動選別
- seed選別
- winner自動選定
- stress orchestration
- sealed unused-future / M3 final test
- Production/live routing
- DB / dashboard / web UI
- remote dataset registry
- resumable download protocol
- Binance Spot study
- Binance COIN-M study

Spotはshort/fundingの経済契約がUSD-M Studyと異なり、COIN-Mは現行linear `BookState`で安全に表現できないため、v1は `market == "usds-m"` のみを受理する。

## 3. Architecture

新規packageはControlled Experiment Loopのdevelopment preparation boundaryとして置く。

```text
trade_rl/evaluation/experiments/
  bootstrap/
    __init__.py
    config.py
    binance.py
    cli.py
```

provider固有のnetwork/cache semanticsは `evaluation` へ複製しない。必要なlower-layer強化は `trade_rl.integrations.binance` に置く。

```text
bootstrap/config.py
  strict JSON parse + canonical config identity
            ↓
bootstrap/binance.py
  acquire/freeze exact Binance source evidence
            ↓
integrations/binance
  verified Vision cache + frozen exchangeInfo + cache-only transport
            ↓
trade_rl.data
  build + atomic dataset artifact publication
            ↓
evaluation.experiments.create_study
  resolved immutable StudyPlan
            ↓
bootstrap root final validation + atomic publication
```

依存方向は既存契約を維持する。

```text
integrations -X-> evaluation
runs         -X-> experiments
bootstrap     -> integrations/data/experiments workflow
bootstrap    -X-> sealed final-test authorization
```

## 4. Bootstrap config contract

入力はUTF-8 JSON object一つである。unknown keyは拒否する。研究条件に暗黙defaultを入れない。

以下は**schema形状の例であり、symbol universe・期間・featureを研究上推奨するものではない**。

```json
{
  "schema_version": "canonical_m2_bootstrap_config_v1",
  "research_question": "Can one universal policy beat simple controls on frozen development data?",
  "market": "usds-m",
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "base_timeframe": "1h",
  "feature_timeframes": ["4h", "1d"],
  "data_start": "2024-01-01T00:00:00+00:00",
  "data_stop_exclusive": "2025-07-01T00:00:00+00:00",
  "baseline": {
    "signal_name": "1h__log_return_24bar",
    "feature_names": ["1h__log_return_24bar", "1h__realized_volatility_24bar"],
    "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
    "fit_cutoff": "2025-01-01T00:00:00",
    "evaluation_start": "2025-01-01T00:00:00",
    "evaluation_stop_exclusive": "2025-07-01T00:00:00",
    "rule_entry_threshold": 0.10,
    "rule_exit_threshold": 0.02,
    "forecast_entry_threshold": 0.01,
    "forecast_exit_threshold": 0.002,
    "ppo_total_timesteps": 100000,
    "gross_budget": 0.5,
    "initial_capital": 100000.0
  },
  "ppo_seeds": [0, 1, 2, 3, 4],
  "allowed_factors": ["FEATURE_SET", "PPO_TRAINING_BUDGET"],
  "max_experiments": 12,
  "n_bootstrap": 2000,
  "bootstrap_seed": 0
}
```

`baseline.ppo_seed`は入力させない。Study seed policyの二重正本を作らず、bootstrapが `ppo_seeds[0]` をbaseline `CandidateRunConfig.ppo_seed`へ設定する。

### 4.1 Required validation

- `schema_version` exact match
- `market == "usds-m"`
- `symbols` はnon-empty / unique / order-preserving
- `feature_timeframes` はuniqueでbase timeframeを含めない
- time fieldsはtimezone-aware UTCへ正規化可能
- `data_start < fit_cutoff <= evaluation_start < evaluation_stop_exclusive <= data_stop_exclusive`
- `data_start` / `data_stop_exclusive` はbase + feature timeframesすべてのnative clock境界へalignする
- `data_stop_exclusive` はUTCの月初00:00:00である
- `fit_symbol_names` はnon-empty / uniqueで`symbols`のsubset
- `feature_names` はnon-empty / unique
- `ppo_seeds` は2個以上のunique non-negative int
- `allowed_factors` はnon-empty / uniqueで現行`ControlledFactor`だけ
- `max_experiments > 0`
- `n_bootstrap > 0`
- `bootstrap_seed >= 0`
- baselineのthreshold/budget/capital/timestepsは既存 `CandidateRunConfig` と同じvalidatorを通す

config parserはfeature index、symbol index、timestamp indexを独自解決しない。dataset作成後、既存 `resolve_candidate_run_spec()` / `create_study()` を唯一のresolution authorityとして使用する。

### 4.2 Config identity

normalized payloadをcanonical JSONとして `content_digest()` し、`bootstrap_config_digest`とする。JSONのkey順・空白差はidentityを変えない。

`bootstrap.json`にはnormalized payloadを保存する。

## 5. Source freeze contract

### 5.1 Exchange metadata

bootstrap開始時に `FrozenBinanceExchangeInfoTransport` とlive `BinancePublicTransport`を使ってUSD-M `exchangeInfo`を一度だけ取得し、次を保存する。

```text
source/exchange-info/
  exchange-info.raw.json
  manifest.json
```

既存contractに従い、少なくとも次を検証する。

- raw bytes SHA-256
- source URI
- market
- retrieved_at
- schema version

`FrozenBinanceExchangeInfoTransport`へ `load_exchange_information(...)` compatibility methodを追加する。これは既存のfrozen snapshot loaderを再利用してdeep-mutable JSON copyと `"frozen:exchange-info"` source markerを返し、新しいmetadata parserは作らない。

Dataset identityへ渡す`metadata_evidence`はfrozen snapshotをcanonical payload化したものとし、少なくともmarket、source URI、raw SHA-256、retrieved_at、evidence schemaをbindする。

### 5.2 Vision archive plan

必要intervalは次のordered unique setである。

```text
(base_timeframe, *feature_timeframes)
```

`plan_binance_vision_cache()`を唯一のURL planning authorityとして、全symbol・全interval・必要funding archiveのURL集合を事前計画する。

planを `source/vision-plan.json` にcanonical JSONで保存する。

### 5.3 Vision raw cache

network-enabled `BinancePublicTransport(cache_root=<staging>/source/vision-cache)` でplan全件を同期する。

各archiveは既存 `binance_vision_raw_cache_v1` evidenceを持ち、少なくとも次を検証する。

- exact URL
- SHA-256
- byte size
- acquired_at
- downloader
- ETag if supplied
- Last-Modified if supplied

bootstrap成功後もraw archive bytesとsidecarを最終bundle内へ保持する。datasetだけ残してsource bytesを捨てない。

### 5.4 Network cut

source同期後、dataset buildはnetwork-enabled transportを使用してはならない。

lower integration layerへcache-only capabilityを追加し、cache miss時はHTTPへfallbackせず`BinanceTransportError`でfail closedする。

`BinancePublicTransport` にbackward-compatibleな `allow_network: bool = True` を追加する。`allow_network=False`では、verified Vision cache hitだけを許し、それ以外の `_request_bytes()` を拒否する。default `True` は現行behaviorを維持する。

bootstrap側のprivate composite transportは次だけをdelegateする。

- klines/funding: `BinancePublicTransport(cache_root=..., allow_network=False)`
- metadata: `FrozenBinanceExchangeInfoTransport(delegate=None)`

USD-M fundingの現行Vision実装は未完了末尾月でRESTへfallbackし得る。その経路も`allow_network=False`で拒否される。さらにv1 config自体が`data_stop_exclusive`をUTC月境界に限定し、通常経路ではREST funding fallbackを必要としない条件をpre-registerする。

Dataset build完了時、source markerがVision cache + frozen metadata以外を示した場合はbootstrap失敗とする。

## 6. Dataset build and publication

source freeze後に既存 `build_binance_market_dataset()` を使う。

- market: `usds-m`
- symbols: config order
- interval: `base_timeframe`
- feature_timeframes: config
- start/end: config
- transport mode: Vision
- transport: cache-only composite
- metadata evidence: frozen exchangeInfo evidence

独自のfeature implementation、metadata parser、MarketDataset builderをbootstrapへ作らない。

結果は既存 `publish_market_dataset_artifact()` で次へpublishする。

```text
dataset/
  manifest.json
  arrays.npz
```

publication後に必ず再load/inspectし、次を確認する。

- loaded `dataset_id` == built `dataset_id`
- artifact digestがinspect結果と一致
- symbol rosterがconfigとexact match
- dataset timestamp rangeがpre-registered rangeを満たす
- baseline configが既存resolverで解決可能
- fit scopeにeligible rowが存在する

## 7. Study creation

Dataset publication後、既存 `create_study()` を呼ぶ。

Bootstrap configから次を渡す。

- `research_question`
- canonical `dataset_root`
- baseline `CandidateRunConfig` (`ppo_seed = ppo_seeds[0]`)
- `ppo_seeds`
- `allowed_factors`
- `max_experiments`
- `n_bootstrap`
- `bootstrap_seed`

出力は既存Study contractのまま:

```text
study/
  plan.json
  .mutation.lock
```

bootstrapは `run_baseline()` を呼ばない。`baseline/` が存在した状態でbootstrap successを返してはならない。

## 8. Bootstrap provenance

bootstrap開始前に既存 `build_candidate_run_provenance()` でimplementation/runtime digestを取得する。

source acquisition、dataset build、Study作成、最終validation後に再取得し、開始前後で次が一致しなければfinal rootをpublishしない。

- implementation digest
- runtime environment digest

最終StudyPlanがbindしたimplementation/runtime digestもbootstrap開始前後のdigestとexact matchしなければならない。

これによりbootstrap途中のsource code/runtime driftをhidden conditionとして残さない。

## 9. Output artifact and identity

CLI:

```bash
uv run --extra forecast-gbm --extra train-sb3 \
  python -m trade_rl.evaluation.experiments.bootstrap.cli \
  --config <bootstrap-config.json> \
  --output <nonexistent-bootstrap-root>
```

成功時layout:

```text
<bootstrap-root>/
  bootstrap.json
  bootstrap-manifest.json
  source/
    exchange-info/
      exchange-info.raw.json
      manifest.json
    vision-plan.json
    vision-cache/
      **/*.bin
      **/*.json
  dataset/
    manifest.json
    arrays.npz
  study/
    plan.json
    .mutation.lock
```

`bootstrap-manifest.json`は少なくとも次へbindする。

- schema version
- bootstrap config digest
- Vision plan digest
- ordered raw source roster `{url, sha256, size_bytes}`
- raw source roster digest
- frozen exchangeInfo raw SHA-256 / source URI / retrieved_at
- dataset id
- dataset artifact schema/digest
- StudyPlan digest
- implementation digest
- runtime environment digest

`bootstrap_digest`はmanifestのidentity fieldsをcanonical digestして計算する。filesystem absolute path、temporary staging pathはidentityへ入れない。

`CanonicalM2BootstrapResult`は少なくとも次を返すimmutable valueとする。

- final root
- bootstrap digest
- bootstrap config digest
- dataset id
- dataset artifact digest
- StudyPlan digest

## 10. Atomicity and overwrite policy

`--output`は存在していてはならない。output parentはregular directoryとして扱い、final outputをsymlink越しにpublishしない。

bootstrap全体はoutput parent配下のtemporary staging rootへ構築する。

```text
parse config
→ capture start provenance
→ acquire/freeze source
→ verify complete source
→ network cut
→ build/publish dataset
→ create StudyPlan
→ validate final graph
→ capture end provenance
→ write bootstrap manifest
→ staging root rename to output
```

どのstepでも失敗した場合、final `--output` pathは存在してはならない。staging rootはcleanupする。

既存outputへ追記・repair・resume・overwriteしない。v1では失敗後のdownload resumeより、valid-looking partial Studyを絶対に残さないことを優先する。

## 11. Failure modes

最低限、次をfail closedにする。

### Config

- unknown/missing key
- invalid schema version
- non-USD-M market
- duplicate/empty symbol or feature
- invalid/naive timestamp
- timestamp order violation
- native-clock misalignment
- data stop not UTC month boundary
- invalid seed policy
- unsupported controlled factor

### Source

- missing Vision URL
- empty archive
- invalid ZIP/CSV
- missing or malformed sidecar
- URL/size/SHA mismatch
- duplicate/irregular kline timestamps
- missing required funding archive
- REST/network attempt after network cut
- malformed/tampered exchangeInfo cache
- exchangeInfo symbol not TRADING / missing symbol

### Dataset

- noncanonical identity
- partial publication
- reload identity mismatch
- dataset/config symbol mismatch
- unresolved feature/signal
- invalid fit scope
- evaluation timestamps not exact dataset timestamps

### Study

- StudyPlan dataset digest mismatch
- baseline resolved config mismatch
- seed policy mismatch
- implementation/runtime digest mismatch
- unexpected baseline evidence already present

### Filesystem

- output exists
- symlink target where regular file/directory is required
- partial staging failure
- final rename failure

## 12. Public API policy

v1のintentional public surfaceは次に固定する。

```python
CanonicalM2BootstrapConfig
CanonicalM2BootstrapResult
load_canonical_m2_bootstrap_config(...)
bootstrap_canonical_m2_study(...)
```

provider-specific helper、cache-only composite、manifest codecはbootstrap package privateとする。

`trade_rl.evaluation` root facadeへbootstrap APIを再exportしない。利用者は明示的に `trade_rl.evaluation.experiments.bootstrap` からimportする。

lower integrationでは次のbackward-compatible public extensionを行う。

```python
BinancePublicTransport(..., allow_network: bool = True)
FrozenBinanceExchangeInfoTransport.load_exchange_information(...)
```

## 13. Test Oracle

単にCLI exit code 0を成功条件にしない。

観測する正本:

- normalized bootstrap config payload/digest
- raw exchangeInfo bytes SHA
- Vision plan URL roster
- each raw archive bytes SHA/size/URL
- network call count after source freeze == 0
- dataset id/artifact digest/reload equality
- StudyPlan digest and dataset binding
- StudyPlan implementation/runtime binding
- absence of `baseline/`
- bootstrap manifest digest/reference graph
- final output exists only after all validation
- failure時final output absent

## 14. Required test layers

### Unit / contract

- strict config parse
- canonical config digest
- UTC/month-boundary validation
- raw source roster canonical digest
- manifest contract
- cache-only network rejection

### Integration

fake Binance transportを使い、次を一つのcontract testで通す。

```text
source freeze
→ cache-only rebuild
→ dataset publication/reload
→ Study creation
→ final bundle reconstruction
```

### Regression / falsification

- post-freeze REST funding fallback attempt
- self-consistent sidecar tamper
- exchangeInfo byte tamper
- source code/runtime drift
- partial archive acquisition
- partial dataset publication
- Study created in staging but final bootstrap publication fails
- pre-existing output
- symlink/path substitution
- baseline accidentally executed

### Architecture

- `integrations -> evaluation`依存なし
- `evaluation/runs -> evaluation/experiments`依存なし
- bootstrapからsealed final-test importなし
- permanent bootstrap filesのrequired layout

### Final repository gate

同一final HEADで最低限:

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run pytest -q tests
uv build
```

package identityとGitHub Actionsも同一HEADで確認する。

## 15. Acceptance Criteria

実装成功は次をすべて満たす場合だけとする。

1. strict pre-registration configがcanonical digestを持つ。
2. v1はUSD-M Futures以外をfail closedする。
3. data stopはUTC月境界へpre-registerされる。
4. 必要な全Vision URLを既存plannerで事前計画する。
5. raw Vision bytesとsidecar evidenceを最終bundleに保持する。
6. exchangeInfo exact bytesとmanifestを最終bundleに保持する。
7. source freeze後のnetwork accessを構造的に拒否する。
8. dataset buildはcache-only sourceだけで成立する。
9. metadata evidenceがdataset identityへbindされる。
10. dataset artifactをatomic publicationし、再load identityを検証する。
11. baseline config resolutionは既存Run Core authorityを再利用する。
12. StudyPlanがdataset/config/seed/budget/factor/provenanceへbindされる。
13. bootstrapはbaselineを実行しない。
14. bootstrap開始前後のimplementation/runtime digest driftを拒否する。
15. output rootはappend/overwriteせず一度だけpublishされる。
16. failure時にvalid-looking final outputを残さない。
17. manifestのreference graphを再構築・検証できる。
18. bootstrapからsealed unused-futureへ到達できない。
19. 現行Candidate Run / CEL public contractを壊さない。
20. mainの既存testsにsemantic regressionを起こさない。
21. targeted + full tests、static checks、distribution build、package identity、exact-head CIが成功する。
22. falsification reviewでCritical/High未解決がない。

## 16. Quality Gate and remaining limitations

このbootstrapがGreenでも次は保証しない。

- Binance source自体の経済的正しさ
- 選んだsymbol universeの研究妥当性
- feature edge
- profitability
- candidate winner
- unused-future robustness
- Production suitability

実装後の次工程で初めて、実データavailability/listing/warmup/compute costを**performanceを見る前に**調査し、実際のcanonical Study configをpre-registerする。その後 `run_baseline()` からM2 development Studyを開始する。
