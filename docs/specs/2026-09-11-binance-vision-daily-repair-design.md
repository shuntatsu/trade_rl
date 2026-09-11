# Binance Vision Daily Repair Design

Status: Active

## 結論

Canonical M2 bootstrap の source freeze は、pre-registration から決まる既存 `vision-plan.json` を primary source plan として維持する。その primary plan の official Binance Vision monthly kline archive に timestamp 欠損がある場合だけ、欠損した UTC day の official daily kline archive を deterministic repair source として追加する。

repair は hidden fallback ではない。`source/vision-resolution.json` に、primary plan digest、symbol/timeframe ごとの missing open timestamps、repair daily URLs を保存する。raw source roster は primary + repair の実際に使用した全 archive bytes を SHA-256 / size とともに bind する。bootstrap manifest writer は v2 に進め、`vision_resolution_digest` を bootstrap identity に含める。既存 v1 artifact reader は維持する。

研究条件は変更しない。symbol、period、timeframe、feature、threshold、seed、budget は bootstrap config のままである。

## 背景と観測事実

Canonical M2 study 001 の pre-registered range は 2021-01-01T00:00:00Z から 2025-01-01T00:00:00Z、symbols は BTCUSDT / ETHUSDT / BNBUSDT / XRPUSDT / ADAUSDT、base timeframe は 1h、feature timeframes は 4h / 1d である。

実データ bootstrap では primary Binance Vision archive の存在確認 960/960 は成功したが、dataset build が `Binance kline range must be complete and exactly regular` で停止した。timestamp-only diagnostic では BTC / ETH / BNB / ADA は 1h / 4h / 1d の全 clock が完全だった。一方 XRPUSDT は 2022-02-26..2022-02-28 と 2022-04-01..2022-04-02 に対応する bar が monthly archive から欠落していた。

同日の official Binance Vision daily archive を別 probe で確認すると、対象5日すべて HTTP 200 で、1h archive は各24 rows を保持していた。したがって今回の failure は market suspension でも preregistration error でもなく、official monthly archive の content completeness defect である。

診断は timestamp のみを使用し、price / return / P&L / strategy outcome は観測していない。

## Objective

Official Binance Vision の primary monthly archive が内部的に不完全でも、同一 official source family の daily archiveを明示的・再現可能・fail-closed に使用して pre-registered clock を復元し、canonical bootstrap を変更前と同じ research config で作成可能にする。

## Non-goals

- bootstrap config の symbol / period / timeframe / feature / threshold / PPO budget / seed / experiment budgetを変更しない。
- missing OHLC を補間・forward-fill・synthetic生成しない。
- REST kline を repair source にしない。
- funding source policy を変更しない。
- `BinancePublicTransport` の通常の AUTO/VISION 利用を hidden daily fallback に変更しない。
- monthly archive が完全な series に daily archive を追加取得しない。
- final unused-future data を開かない。
- XRP または 2022 年固有の production exception を追加しない。

## Authority と責務境界

### Primary plan

既存 `source/vision-plan.json` と `_vision_plan_payload(config)` は変更しない。これは pre-registration から純粋に決まる primary acquisition contract であり、既存 `canonical_m2_vision_plan_v1` の意味を維持する。

### Resolution plan

新規 `source/vision-resolution.json` が runtime resolution の authority になる。schema は `canonical_m2_vision_resolution_v1` とする。

最低限、次を保持する。

```json
{
  "schema_version": "canonical_m2_vision_resolution_v1",
  "primary_plan_digest": "<sha256>",
  "repairs": [
    {
      "symbol": "<symbol>",
      "timeframe": "<timeframe>",
      "missing_open_ms": [123],
      "daily_urls": ["https://data.binance.vision/..."]
    }
  ]
}
```

`repairs` は bootstrap config の symbol order、その中で `(base_timeframe, *feature_timeframes)` order に従う。repair が不要な series は entry を持たない。`missing_open_ms` は strictly increasing / unique / requested range 内 / native interval aligned でなければならない。`daily_urls` は missing timestamps が属する UTC day を exact に覆う official Binance Vision daily kline URL の ordered unique 集合だけを許可する。

### Raw source roster

`raw_source_roster` は primary plan URLs と resolution plan の repair URLs の ordered unionを対象とする。各memberの `url / sha256 / size_bytes` 契約は既存のまま維持する。source role は resolution plan から導出できるため roster item schema へ重複保持しない。

### Dataset reader

Canonical bootstrap 専用の frozen dataset transport が persisted resolution plan に従って primary rows と repair rows をmergeする。通常の `BinancePublicTransport.load_klines(..., mode=VISION)` の外部挙動は変更しない。

repair merge は timestamp key で行う。primary row が存在する timestamp は primary が authority である。repair archiveに同じ timestamp が存在する場合、row content が一致しなければ fail-closed とする。一致する overlap は重複publicationせず、primary rowを保持する。primary に存在しない expected timestampだけ repair rowから補う。

merge後は start inclusive / stop exclusive の全 expected open timestamp が exactly one row 存在し、strictly increasing かつ native interval exact でなければ source freezeを失敗させる。

## Acquisition data flow

1. bootstrap config から既存 primary Vision plan を生成する。
2. exchange-info を既存契約でfreezeする。
3. primary plan URLを既存 cache contractで同期・byte integrity検証する。
4. primary kline bytesだけを読み、各 symbol/timeframe の timestamp coverage を検査する。
5. missing expected open timestampsをdeterministically列挙する。
6. missing timestampが属するUTC dayからofficial daily repair URLsを導出する。
7. repair URLsだけ追加同期し、各 raw byteを既存 sidecar SHA-256 evidenceで検証する。
8. primary + repair rowsをmergeし、overlap conflictとfull clock coverageを検証する。
9. `vision-resolution.json` をpublishする。
10. primary + repair ordered unionからraw source rosterを作る。
11. network-disabled frozen transportだけをdataset buildへ渡す。
12. source / dataset / Study / manifestをstaging rootで相互検証してからwhole-root renameする。

source sync後のdataset buildとinspectionは従来通りnetwork-freeでなければならない。

## Inspection data flow

`inspect_canonical_m2_bootstrap` は network accessなしで以下を再構築する。

1. bootstrap configから primary plan を再計算し `vision-plan.json` と一致確認する。
2. primary raw bytesから missing timestamps と expected repair URLs を再計算する。
3. `vision-resolution.json` がその結果とexact一致することを確認する。
4. primary + repair raw source roster の全memberをsidecar / SHA-256 / sizeで検証する。
5. repair merge後の全native clockがexactly regularであることを確認する。
6. dataset artifact、StudyPlan、bootstrap manifestとの既存identity checksを行う。

保存されたrepair planを盲信せず、primary bytesから再導出することが重要である。

## Manifest schema evolution

新規writerは `canonical_m2_bootstrap_manifest_v2` を出力する。v2はv1 fieldsを全て維持し、次を追加する。

```text
vision_resolution_digest
```

`bootstrap_digest` は従来どおりmanifest body全体のcontent digestなので、resolution digestを通じてrepair topologyもidentityに入る。

readerは以下を維持する。

- v1: `vision-resolution.json` を要求せず、既存 primary-only artifact を従来契約でinspectする。
- v2: `vision-resolution.json` を必須とし、manifestの`vision_resolution_digest`とrecomputed resolution digestの一致を要求する。

v1 artifactをv2へ自動migration・rewriteしない。

## Invariants

- bootstrap config digestはrepair導入前後で不変。
- primary `vision-plan.json` payload/digestは同じconfigなら不変。
- repair decisionはtimestamp coverageだけから決まり、価格・return・P&Lを参照しない。
- daily repair URLは official `https://data.binance.vision/data/` 配下のみ。
- primary rowはrepair rowで上書きしない。
- overlap content conflictはfail-closed。
- repair後も1本でもmissing/duplicate/backwards/irregular rowがあればdatasetをpublishしない。
- source sync完了後のdataset build / inspectionでnetwork accessしない。
- repair raw bytesを含む全使用sourceがraw source rosterに含まれる。
- unrelated symbols/timeframesにrepair sourceを追加しない。
- funding behaviorは不変。
- ordinary `BinancePublicTransport` public behaviorは不変。

## Failure Modes

- primary archiveにmissing barsがあるがdaily archiveも存在しない。
- daily archiveは存在するがrequired timestampが仍欠損している。
- daily repairとprimary overlapのrow contentが異なる。
- repair URLがrequested symbol/timeframe/dayと一致しない。
- repair URLがofficial Vision prefix外。
- resolution fileのmissing timestamps / URL order / digestがtamperされる。
- repair raw bytesまたはsidecarがtamperされる。
- raw source rosterからrepair URLが欠落または余計なURLが入る。
- v1 artifactを誤ってv2として解釈する、または逆。
- network-disabled inspectionがnetwork fallbackしてしまう。
- repair実装がXRP/2022など既知fixtureへhard-codeされる。

すべてfail-closedとする。

## Test Oracle

正しさは単なるbootstrap成功ではなく、次の観測で判定する。

- synthetic monthly primary archiveにarbitrary day gapを作ると、そのdayだけdaily repair URLがresolved planへ入る。
- gapのないseriesではrepair listが空で追加network requestがない。
- repair後のopen timestampsがrequested clockのexpected sequenceと完全一致する。
- primary/daily overlapが同一なら1 rowへcollapseし、異なるなら失敗する。
- daily archive自体がincompleteなら失敗する。
- resolution file / repair raw bytes / sidecar / roster tamperをinspectionが検出する。
- `allow_network=False` のinspection/rebuildでnetwork requestが発生しない。
- v1 synthetic artifactのinspection契約が維持される。
- v2 manifest digestがresolution digest変更を検出する。
- existing bootstrap / cache / transport unit and integration testsがGreenを維持する。
- full repository CI、distribution closure、clean install、package identityが成功する。

## Required Test Layers

- Unit: timestamp gap detection、UTC day → repair URL resolution、merge/conflict semantics。
- Integration: cache sync + frozen source inspection + bootstrap manifest v1/v2。
- Regression: current complete Vision series behavior、public transport behavior。
- Falsification: hard-coded known dates/symbolsでは通らないarbitrary synthetic gaps、tampered resolution/repair bytes、incomplete daily repair。
- Static: Ruff / Format / Mypy / architecture tooling。
- Full: complete pytest suite、build、distribution source closure、clean installed smoke、package identity。
- Real-data confirmation: canonical M2 configを変更せず再実行し、bootstrap inspection成功後にのみbaselineへ進む。

## Quality Gate

次を全て満たすまでproduction修正を完了扱いしない。

1. REDで現行primary-only freezeがincomplete monthly archiveをrepairできないことを再現する。
2. GREENでgeneric synthetic gapをdaily repairし、hard-coded XRP/dateなしで通る。
3. conflict / still-missing / tamper / offline failure modesを検証する。
4. v1 reader compatibilityを確認する。
5. exact final HEADでfull hardened CIを通す。
6. merge後main exact HEADでも同じrequired checksを確認する。
7. canonical M2 research branchへ新mainをhistory-preserving mergeする。
8. bootstrap config raw SHA-256とconfig digest不変を確認する。
9. real-data bootstrapを再実行し、resolved repair evidenceとnetwork-free inspectionを保存する。
10. pristine bootstrap成功後だけdevelopment Study copyでbaselineを実行する。

## Expected file scope

主な変更候補は以下に限定する。

- `trade_rl/evaluation/experiments/bootstrap/binance.py`
- `trade_rl/evaluation/experiments/bootstrap/workflow.py`
- 必要なら `trade_rl/integrations/binance/vision.py` または小さな既存integration helper
- bootstrap / Binance integration tests
- `docs/architecture/controlled-experiment-loop.md`
- `docs/research/current-status.md` はreal-data再実行結果が出た後だけ更新する

通常のmarket dataset / simulation / strategy / risk behaviorは変更しない。
