## このページで答える問い

1本の市場バーを処理するとき、**現在時点の観測 → 戦略判断 → hard risk → 約定・会計 → 次のBookState** はどの順番で進むのか。

重要なのは、戦略の出力をそのまま注文にしないことと、戦略ごとに別会計を持たせないことです。

## 処理順

```text
MarketDataset
    │
    │ 1. 現在時点の観測を組み立てる
    ▼
Replay
    │
    │ 2. strategy.decide(observation)
    ▼
Strategy
    │
    │ 3. SHORT / FLAT / LONG を返す
    ▼
Replay
    │
    │ desired_quantity → proposal_weight
    │
    │ 4. hard riskを適用
    ▼
PreTradeRisk.constrain
    │
    │ target_weight
    │
    │ 5. 制約済み目標を約定・会計へ渡す
    ▼
MarketExecutor.execute_interval
    │
    │ fill / cost / interval net return
    ▼
BookState
    │
    │ 6. 帳簿と処理位置を次のバーへ引き継ぐ
    └───────────────────────────────→ Replay

最後に 7. returns / decisions / diagnostics を結果へまとめる
```

この順序はクリックしなくても全体を読めるように固定しています。

## 1. 観測を作る

現在バーのfeature、availability、staleness、global feature情報に、**現在の売買意図と現在ウェイト**を加えて`StrategyObservation`を作ります。

未来のバーを見てから観測を作ることはありません。判断に入る情報は現在時点までに利用可能なものに限定します。

## 2. 戦略がintentを返す

戦略へ現在の`StrategyObservation`を渡し、`SHORT` / `FLAT` / `LONG`の論理的なポジション意図を受け取ります。

ここではまだfillも手数料も発生していません。**戦略の責務は経済判断であって、約定や会計ではありません。**

## 3. 希望数量とproposalへ変換する

intentが変わった場合、gross budgetから目標ウェイトを計算し、現在価格と現在の資産額を使って希望保有数量へ変換します。

```text
PositionIntent
    ↓
desired_quantity
    ↓
proposal_weight
```

`proposal_weight`はまだ実行可能性を保証しない、risk適用前の提案値です。

## 4. hard riskを適用する

`PreTradeRisk`はproposalへ、turnover、単一ウェイト上限、gross上限、drawdown縮小、emergency flatten、reduce-onlyなどの制約を適用します。

```text
proposal_weight
      ↓
PreTradeRisk.constrain
      ↓
target_weight
```

約定器へ渡るのは、戦略が最初に希望した値ではなく**制約後の`target_weight`**です。

## 5. 約定と会計を行う

`MarketExecutor`が次の1区間を共通ルールで処理します。

- 注文とrealized fillを区別する
- fee / spread / funding / borrow等を共通経路で計上する
- partial fillならrealized quantityだけpositionへ反映する
- 区間のnet returnを作る
- 新しい`BookState`を返す

戦略ごとに別の損益計算を実装しません。

## 6. BookStateを引き継ぐ

約定後の`BookState`を次のバーの共通状態として採用します。同時に今回のintentと次の処理index、累積cost・turnover等を引き継ぎます。

このため、次の観測も**実際に約定・会計された状態**から始まります。

## 7. 結果を確定する

全区間を処理した後、raw interval returns、decisions、execution diagnosticsを`SingleSymbolReplayResult`へまとめます。

比較に使うのはこの共通経路を通った結果です。

## 不変条件

- 判断前に未来バーを読まない。
- strategy intentを直接fill扱いしない。
- hard riskを通す前のproposalを約定しない。
- fee / spread等をstrategy側で別途控除しない。
- 次のバーは約定後の共通`BookState`から始める。
- 比較対象strategyは同じexecution/accountingを使う。

## 失敗すると何が壊れるか

| 破る契約 | 起こる問題 |
| --- | --- |
| 未来情報を観測へ入れる | leakageで見かけの利益が増える |
| intentを直接約定する | riskやfill制約を無視した非現実的P&Lになる |
| strategyごとに会計を変える | 戦略比較が同一条件でなくなる |
| costを複数箇所で引く | 二重課金になる |
| 約定前stateを次バーへ持つ | portfolio状態とP&Lが食い違う |
