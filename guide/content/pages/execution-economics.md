## このページで答える問い

fee、spread、participation、borrowなどの実行条件をどこで決め、どこでP&Lへ反映し、二重課金をどう防ぐのか。

## 会計の正本は一つ

```text
ExecutionEconomicsProfile
          ↓
MarketDatasetの実行条件fields
          ↓
PreTradeRiskで実行可能なtargetへ制約
          ↓
MarketExecutor + BookState
          ↓
fill / fee / spread / funding / borrow / return
          ↓
Run diagnostics / raw evidence
```

Canonical M2では、Datasetへ解決済みの実行条件を持たせ、実行時の追加overlayはゼロに固定します。

## Datasetが実行条件を所有する

構築時の`ExecutionEconomicsProfile`でfee、spread、participation上限、borrow等の研究仮定を解決し、Dataset fieldsとcontent identityへ固定します。

その後の候補strategyは同じDataset economicsを使います。

## zero overlayが意味すること

実行時overlayがzeroでも、取引コストがzeroという意味ではありません。

```text
Dataset fee/spread/...  = 有効
追加runtime overlay      = 0
```

Dataset側で既に持っているコストをruntime側でももう一度控除しないための契約です。

## MarketExecutorがfillと会計を処理する

共通executorは少なくとも次を区別・記録します。

- order submissionとrealized fill
- fee
- spread / slippageとしてモデル化された実行条件
- participation / liquidity capacity
- funding
- borrow
- mark-to-market
- margin / liquidation状態
- `BookState`更新

partial fillなら、要求数量ではなくrealized fill quantityだけをpositionへ反映します。

## 研究仮定と実市場truthを混同しない

現行のfee・spread等は、比較可能な研究のために固定した**再現可能な仮定**です。過去の特定accountにおけるBinance実績feeや、観測不能なqueue position、hidden liquidityまで再現したとは主張しません。

| 項目 | 現在の扱い |
| --- | --- |
| fee | Datasetが正本 |
| spread | Datasetが正本 |
| participation上限 | Datasetが正本 |
| borrow | Datasetが正本 |
| runtime追加overlay | Canonical M2ではzero |
| Dataset-authoritative market impact model | 未導入 |

## 二重課金を防ぐ不変条件

- feeはrealized fillへ一度だけ計上する。
- spread / impactを複数channelから重複控除しない。
- fundingは対象時刻・符号・quantityへ一度だけ適用する。
- forced closeとterminal mark-to-marketを混同しない。
- strategy独自のP&L計算を正本にしない。

## 検証で見る値

Run evidenceにはturnover、total cost、funding P&L等のdiagnosticsを残します。利益だけを見て、実際にはほとんど取引していない、あるいはcostが計上されていない候補を誤って採用しないためです。

## まだ保証していないこと

現行execution economicsを通ったbacktest利益が、そのままproduction利益になるとは限りません。account-specific fee、実際のslippage/impact、latency、capacity、stressは別途検証が必要です。
