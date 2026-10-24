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

新規final-eligible M2研究は、データ同期前に条件を固定します。一般はconfig v4、独立口座PPOはv5、共有資金PPOはv6を使い、protocol・baseline/risk・final window・`StudyResearchContext`を別identityへbindします。bootstrapはDatasetとStudyPlanまでを作り、学習・replayはG0-G2 review後に別実行します。

現行のStudy CLIは外部reviewの完了を認証しないため、この順序は運用上のrelease prerequisiteとして扱います。

## 2. 生データを固定する

source URL、SHA-256、size、取引所metadata等を保存し、再取得時にも同じ入力だったか照合できる状態にします。

## 3. 未来情報を混ぜずに特徴量を作る

基本条件は次です。

```text
feature_available_time <= decision_time
```

加えて、fit cutoffより後の情報をscaler、normalization、imputation、feature selectionへ戻しません。supervised labelは終了時刻がfit cutoffより前で、Datasetが宣言するstart/end価格sourceの利用可能時刻もcutoffまでのものだけを使います。forecast training builderはcutoffを`datetime64[ns]`へ値を失わず変換できる場合だけ受け入れ、`NaT`、範囲外、nanosecond未満の端数はscopeやlabel rowを調べる前に停止します。Dataset境界ではtimestampsと明示available_atをnsへ値を失わず変換できる入力だけ受け入れます。範囲外へのwrap、nanosecond未満の端数の切り捨て、NaTを拒否し、有効なcoarser-unit timestampの入力表現は維持し、available_atはnsへ正規化します。

`LeanCandidateConfig`を直接構築するときのtyped `fit_cutoff`も、ns正規化前に元unitとのexact round-tripを検証します。範囲外、`NaT`、nanosecond未満の端数を拒否します。入力前に既にwrapされた`datetime64[ns]`から元時刻を復元することはできません。

PPOを含む候補群を学習する前に、学習対象の銘柄について、学習に使う価格の利用可能時刻が締切までとDatasetに記録されているかを確認します。報酬計算や最終決済で使う末尾の価格も対象です。遅れて届いた価格があれば、学習期間を短くして続けずに停止します。ここで確認するのはDatasetに記録された利用可能時刻であり、実際のvenue arrivalを独立に証明するものではありません。

fit cutoffとは別に、single-symbol/shared-cash replayとPPO環境は、次に処理する各bar行の `available_at` がその行のtimestampより後なら、帳簿・risk・strategy・executionへ渡す前に停止します。直接`MarketExecutor.execute_interval`のtarget admissionでも、有効なtargetが非zero、または帳簿にnonzeroのexact holdingがある銘柄のtarget rowが遅着なら、valuation・reconciliation前に停止します。同時刻のavailabilityは通し、`information_available=False`だけでは拒否しません。flat accountのzero targetによるpending order cancelは許し、未選択flat銘柄や未処理tailは調べません。これはDatasetの宣言時刻に対する範囲限定runtime guardです。明示`mark_price`・外部帳簿markの独立時刻、明示`OrderIntent`実行、直接close liquidation、過去のvenue arrivalまでは保証しません。cutoff preflightが通っても、実行行のavailability contractを代替するものではありません。

また、fit対象銘柄をDataset全体より狭める場合は、training rowだけでなく**selected featureが参照する銘柄範囲**もfit scope内へ閉じます。cross-sectional rankやdispersionのように全銘柄universeへ依存するfeatureはholdout銘柄を間接的に見てしまうためsubset fitでは使いません。BTC等のreference-relative featureは、そのreference銘柄がfit scopeに含まれる場合だけ許可します。

## 4. 同じ入力から同じDataset identityを作る

標準特徴量は`portable_feature_numerics_v1`の固定したscalar/reduction semanticsで計算します。確認済みのAMD / Intel hosted runner間で微小なCPU差がDataset identityへ混ざらないようにしています。

これはroundingやtoleranceで差を隠す方式ではありません。内容そのものが一致することを目標にします。

## 5. Datasetを変更不能なartifactへする

MarketDatasetの配列、availability、staleness、execution economics、identityをartifactへ固定し、publish後に再読込してcontent identityを検証します。

annualization clockの`periods_per_year`はpositive integerで固定し、boolean・fractional・non-finiteな値はDataset境界で拒否します。

同じpublication先が既に存在する場合は、静かに上書きせずfail closedにします。

## 6. 候補strategyは同じDatasetを見る

rule、forecast、PPOなどの候補を、別々の都合の良いDatasetで評価しません。共通のcausal data contractから論理的な売買意図を生成します。

## 7. 約定後の結果をevidenceへする

strategy出力はhard riskと`MarketExecutor + BookState`を通し、summaryだけでなくraw interval returnsとprovenanceも残します。

結果の意味を後から検証できるようにするためです。

## 不変条件

- `feature_available_time <= decision_time`。
- forecast training cutoffはlosslessな`datetime64[ns]`値である。
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
