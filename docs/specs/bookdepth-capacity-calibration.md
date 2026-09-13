# bookDepth capacity calibration preregistration

Status: Active

Issue: #527  
Stacked implementation PR: #528  
Dependency: #523 / PR #524

## 目的

Binance Vision USD-M daily `bookDepth` を、既存の固定 `max_participation_rate = 0.05` を検証・校正するための **capacity evidence** として使う。ただし、評価期間のreturn/P&Lを見て都合のよいexecution assumptionを選ぶことは禁止する。

この段階では `bookDepth` の ±1%..±5% cumulative depth から top-of-book spread やmarket impactを推定しない。providerが提供していない意味を補って見かけ上のバックテスト利益を改善しない。

## 封印するcanonical lineage

- Dataset ID: `d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f`
- Study digest: `3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79`
- market: Binance USD-M (`usds-m`)
- reference volume timeframe: `1h`
- symbols: `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `XRPUSDT`, `ADAUSDT`
- calibration start: `2021-01-01T00:00:00Z`
- calibration stop exclusive: `2023-01-01T00:00:00Z`
- evaluation start: `2023-01-01T00:00:00Z`

`2023-01-01T00:00:00Z` 以降のliquidity summary、return、strategy metric、candidate result、Experiment 0003 evidence、evaluation P&Lをcalibration ruleの決定へ使わない。

## deterministic archive roster

データを見た後の日付選択を防ぐため、calibration archive rosterを次で固定する。

- 2021年1月から2022年12月まで
- 毎月 UTC の1日と15日
- 48 planned days / symbol
- 240 planned symbol-days
- planned archiveがmissing / invalidでも近接日や後日のarchiveで置換しない
- selected archive内のvalid snapshotはすべて使用する

raw archive bytes、URL、SHA-256、sizeを取得証拠のauthorityとし、PR #524のstrict parserでunsupported / malformed evidenceをfail closedにする。

## calibration statistic

各valid snapshotについて、maintained bandsのうち `-1` と `+1` だけをcapacity calibrationへ使う。

1. `bid_depth_1pct = depth[-1]`
2. `ask_depth_1pct = depth[+1]`
3. `worst_side_depth = min(bid_depth_1pct, ask_depth_1pct)`
4. 同一symbolの **直前に完全終了した1h bar** のvolumeだけを参照する
5. `depth_to_volume_ratio = worst_side_depth / previous_completed_1h_volume`
6. denominatorがfiniteかつpositiveでない観測はvalid calibration observationにしない
7. missing snapshotやinvalid rowを補間・修復しない

symbolごとに、frozen roster内の全valid aligned snapshotsから

`Q05 = 5th percentile(depth_to_volume_ratio)`

を計算し、

`symbol_cap = min(0.05, 0.10 * Q05)`

とする。

`Q05`、`0.10` book-utilization fraction、`0.05` ceiling、`{-1,+1}` band、sample days `{1,15}` は、calibration numeric summaryを見る前に固定する。結果を見てこれらを探索・変更しない。

## coverage / invalidation

calibrated Datasetを作る前に以下を満たす必要がある。

- 各symbolが2021年と2022年の両方で少なくとも1つのvalid planned archiveを持つ
- accepted archiveはすべてstrict parserを通る
- 各symbolでvalidation/alignment後に30 distinct valid planned days以上ある
- `Q05` がfiniteかつstrictly positive

1つでも満たさないsymbolがある場合、このcalibrationは `INVALID` とする。replacement day、別symbolの値、2023年以降の値、都合のよいfallback capを導入しない。

## Dataset / Study identity

現行 `ExecutionEconomicsProfile` の `max_participation_rate` はscalarであり、5 symbolの別々のcalibrated capを忠実に表現できない。calibration結果を見た後に都合のよいscalarへ圧縮してはならない。

次段階では、結果を見る前に次のどちらかを別preregistrationで固定する。

1. per-symbol capacity authorityをDataset identityへ明示的に組み込む
2. per-symbol capをscalarへ落とす場合は、保守的aggregation ruleをQ05を見る前に固定する

どちらの場合も新しいDataset identityと新しいStudy identityを作る。既存canonical Dataset / Study / frozen Experiment evidenceをin-placeで変更しない。

## P&L seal

このprotocolのcalibration phaseではevaluation P&Lを実行・閲覧しない。calibrated economicsでevaluationを許可するのは、次をすべて満たした後だけとする。

1. protocolがcontent digest付きで封印済み
2. raw evidenceからcalibration artifactを独立再構築済み
3. calibrated Dataset / Study identityを別途preregister済み
4. Experiment 0003 (#522) がterminalであり、本作業がそのcandidate selection / decision ruleへ影響しない

その後のevaluationでも、new economics下でmodel / thresholdを探索しない。frozen strategy/run configurationを再利用し、controlを含む全strategy/symbolを開示する。

## failure conditions

以下のどれかが起きた場合はcalibration evidenceを無効とする。

- 2023年以降の観測がcalibration statisticへ入る
- missing/invalid planned dayを未登録日で置換する
- current/future incomplete-hour volumeを使う
- `bookDepth` をtop-of-book spreadとして扱う
- invalid source rowsをrepair / interpolateする
- numeric summaryやP&Lを見た後にformula / roster / band / percentile / multiplierを変更する
- symbol間でliquidity値を代用する
- raw archive rosterからcalibration artifactを再構築できない
- existing canonical Dataset / Studyをin-place変更する

## 現在の状態

この文書と `BookDepthCapacityCalibrationProtocol` は **preregistrationだけ** を定義する。現時点ではQ05、`symbol_cap`、calibrated P&Lを生成していない。先にprotocol artifactを封印・独立再取得検証し、その後にstructural availabilityを確認する。
