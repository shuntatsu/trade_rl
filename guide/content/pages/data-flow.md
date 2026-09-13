## このページで答える問い

市場データをどの順番で固定・加工し、未来情報を混ぜずにstrategyと検証証拠へ渡すのか。

## データの流れ

```text
取得対象を事前に決める
        ↓
生データとmetadataを固定する
        ↓
causalな特徴量を構築する
        ↓
MarketDataset artifactを固定する
        ↓
同じDatasetを候補strategyへ渡す
        ↓
hard risk + 共通execution/accountingでreplayする
        ↓
raw returns / summary / provenanceをevidenceとして固定する
```

結果を見てから都合の良い取得元、期間、featureへ差し替えないことが前提です。

## 1. 取得対象を結果より前に決める

市場、銘柄、時間足、期間、取引所metadataなど、どの市場証拠を使うかを先に決めます。

## 2. 生データを固定する

source URL、SHA-256、size、取引所metadata等を保存し、再取得時にも同じ入力だったか照合できる状態にします。

## 3. 未来情報を混ぜずに特徴量を作る

基本条件は次です。

```text
feature_available_time <= decision_time
```

加えて、fit cutoffより後の情報をscaler、normalization、imputation、feature selectionへ戻しません。supervised labelもfit boundaryの内側で完結させます。

## 4. 同じ入力から同じDataset identityを作る

標準特徴量は`portable_feature_numerics_v1`の固定したscalar/reduction semanticsで計算します。確認済みのAMD / Intel hosted runner間で微小なCPU差がDataset identityへ混ざらないようにしています。

これはroundingやtoleranceで差を隠す方式ではありません。内容そのものが一致することを目標にします。

## 5. Datasetを変更不能なartifactへする

MarketDatasetの配列、availability、staleness、execution economics、identityをartifactへ固定し、publish後に再読込してcontent identityを検証します。

同じpublication先が既に存在する場合は、静かに上書きせずfail closedにします。

## 6. 候補strategyは同じDatasetを見る

rule、forecast、PPOなどの候補を、別々の都合の良いDatasetで評価しません。共通のcausal data contractから論理的な売買意図を生成します。

## 7. 約定後の結果をevidenceへする

strategy出力はhard riskと`MarketExecutor + BookState`を通し、summaryだけでなくraw interval returnsとprovenanceも残します。

結果の意味を後から検証できるようにするためです。

## 不変条件

- `feature_available_time <= decision_time`。
- fit-scope外の情報をtraining transformへ戻さない。
- Dataset identityを一致させるための後付けroundingをしない。
- strategy間でDataset、risk、execution/accounting条件を変えない。
- aggregate結果だけでなく銘柄ごとのraw evidenceを保持する。

## 典型的なfailure mode

| failure | 何が壊れるか |
| --- | --- |
| future dataがfeatureへ混入 | leakageで見かけのedgeが生まれる |
| result後にsource rosterを変更 | 実験の事前条件が失われる |
| strategyごとにDatasetを変える | 比較可能性が失われる |
| Dataset identityをtoleranceで丸める | 本当の再現性差を見えなくする |
| summaryだけ保存 | 後から独立再計算できない |
