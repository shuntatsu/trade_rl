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

bootstrapのデータ期間・最終評価期間のUTC日時は、ナノ秒時刻へ正確に変換できる範囲だけを受け付けます。日時文字列や時差の端数が読み込み時に失われる場合も拒否し、別の日時への折り返しや端数の切り捨てでデータ範囲を変更しません。正しいUTC等価な時差と既存の時間足境界の条件は維持します。

bootstrap設定、Vision計画・解決情報、manifestのJSONを読み込む際は、検証済みの通常ファイルを開き、読み取り前後のファイル識別情報とbyte数を照合します。読み取り中に検出した差し替え、消失、変更はJSONを解釈する前に拒否します。 JSONの同じobject内で同名キーが重複している場合も、後から現れた値で上書きせず拒否します。ファイルシステム全体の原子性や入力総量の上限を保証するものではありません。

現行のStudy CLIは外部reviewの完了を認証しないため、この順序は運用上のrelease prerequisiteとして扱います。

## 2. 生データを固定する

source URL、SHA-256、size、取引所metadata等を保存し、再取得時にも同じ入力だったか照合できる状態にします。

CSVのevent timeと `available_at` は、ISO日時本体のfractionをnanosecondまで保って読み込みます。timezone offsetに小数秒を含む場合はmicrosecond精度で反映し、6桁より細かい桁にnon-zero digitがあれば拒否します（余分な桁がすべてzeroなら値は失われません）。integer epochは既存の秒/millisecond判定を整数演算で適用し、範囲外やnanosecond未満の精度を失う値は拒否します。1ns遅れたavailabilityも同時刻へ丸めず、後段の特徴量生成まで遅れとして保ちます。

RawMarketSeriesは通常のNumPy配列・bufferからの入力を正規化する前に、残っているマスク情報と所有元を検査します。所有元を確認できない特殊配列やsubclassは拒否します。直接構築するMarketDatasetでも、価格・特徴量・利用可能時刻・取引費用などの必須・任意配列を、変換やalias比較より先に同じ方法で検査します。欠損は従来の明示的なavailability配列で表し、正常な入力のDataset identityは変えません。既に欠損情報を捨てた入力から欠損を復元する仕組みではなく、既存の非配列coercionや市場データの真正性を保証するものでもありません。

## 3. 未来情報を混ぜずに特徴量を作る

基本条件は次です。

```text
feature_available_time <= decision_time
```

加えて、fit cutoffより後の情報をscaler、normalization、imputation、feature selectionへ戻しません。supervised labelは終了時刻がfit cutoffより前で、start/end両端の行がDatasetの`information_available`（行が存在し、定時に公開済み）であり、両端の公開時刻もfit cutoffより前のものだけを使います。欠損barは直前closeを繰り越した値なので、labelの始点にも終点にも使いません。遅れて公開された行は、cutoff前に届いていても学習labelに使いません。forecast training builderのcutoffと`CausalForecastTrainingSet`へ直接渡す`label_end_times`は、元のdatetime64 unitと正のmultiplier metadataから正確な整数演算でns値を求め、signed-nanosecond範囲を確認してから正規化します。multiplierが0以下のunit metadataは値の意味が定まらないため拒否します。固定長unitでは正確な整数比を使い、calendar `Y` / `M`は暦日の境界をns値へ変換します。`NaT`、範囲外へのwrap、nanosecond未満の端数はscopeやlabel maturity比較より前に拒否します。cast後に元unitへ戻るかだけでは、overflowを検出できません。availabilityはDatasetが宣言する値であり、独立したvenue到着証拠ではありません。Dataset境界ではtimestampsと明示available_atもns時計で検証し、範囲外へのwrap、nanosecond未満の端数の切り捨て、NaTを拒否します。有効なcoarser-unit timestampsの入力表現は維持し、available_atはnsへ正規化します。Candidate Runの`fit_cutoff`、`evaluation_start`、`evaluation_stop_exclusive`も、元ISO値のcalendar/range/precisionを確認してからnsへ正規化し、範囲外のwrapとnon-zero sub-nanosecond桁を拒否します。timezone offsetのUTC等価化は維持します。入力前に既にwrapされた`datetime64[ns]`値の元時刻は復元できません。

`LeanCandidateConfig`を直接構築するときのtyped `fit_cutoff`も、Dataset、`RawMarketSeries`、forecast cutoff/labelsと共有するinternal data helperで整数演算のlosslessなns正規化を行います。範囲外、`NaT`、nanosecond未満の端数を拒否します。入力前に既にwrapされた`datetime64[ns]`から元時刻を復元することはできません。

Ridge / LightGBM strategy adapterは、保存済みobservation timestampが`NaT`、またはmodelの`fit_cutoff`がobservation timestampより後なら、feature-unavailable fallbackやpredictionの前に拒否します。同時刻は許します。これは宣言fit cutoffに対するas-of floorで、modelの生成・配布時刻や過去のvintage availabilityは証明しません。`StrategyObservation`構築前に既にwrapされたtimestampも復元できません。

Task18向けの別経路は単純リターンを直接学習します。既存の対数リターンとは型と予測APIを分け、両価格端点の情報が利用可能でcutoffより厳密に前であることを要求します。4設定×3 vintageの12モデル、52,216予測、52,216費用行を凍結し、外部pinを検証してから再計算なしで参照します。人工native完了receiptはledger・結果・検証・receiptの4ファイルを新規directoryに一括公開し、復元時は実行せずsource oracleで再照合します。実約定の費用と事前費用推定は区別し、歴史的市場実行や認可の証拠にしません。人工停止handoffは外部pin付きの全祖先世代を順に照合し、元run IDのまま累積した完成armを除くpending subsetだけを新runへ渡します。人工後継runは実native区間の観測を定期保存し、例外でFAILEDになった場合も観測済みの進捗だけを保持します。人工48-armの検証driverは同じ凍結planを使い、完了した4ファイルの外部pinとbatchの系譜を最後に照合します。

PPOを含む候補群を学習する前に、学習対象の銘柄について、学習に使う価格の利用可能時刻が締切までとDatasetに記録されているかを確認します。報酬計算や最終決済で使う末尾の価格も対象です。遅れて届いた価格があれば、学習期間を短くして続けずに停止します。ここで確認するのはDatasetに記録された利用可能時刻であり、実際のvenue arrivalを独立に証明するものではありません。

fit cutoffとは別に、single-symbol/shared-cash replayとPPO環境は、次に処理する各bar行の `available_at` がその行のtimestampより後なら、帳簿・risk・strategy・executionへ渡す前に停止します。直接`MarketExecutor.execute_interval`のtarget admissionでも、有効なtargetが非zero、または帳簿にnonzeroのexact holdingがある銘柄のtarget rowが遅着なら、valuation・reconciliation前に停止します。同時刻のavailabilityは通し、`information_available=False`だけでは拒否しません。flat accountのzero targetによるpending order cancelは許し、未選択flat銘柄や未処理tailは調べません。これはDatasetの宣言時刻に対する範囲限定runtime guardです。明示`mark_price`・外部帳簿markの独立時刻、明示`OrderIntent`実行、直接close liquidation、過去のvenue arrivalまでは保証しません。cutoff preflightが通っても、実行行のavailability contractを代替するものではありません。

globalのmarket return平均・標準偏差は、そのバーで利用できる銘柄の1-bar returnだけから計算し、古い集計値を繰り越しません。利用可能な行の`global_feature_staleness_hours`は0時間です。returnが1件もなく欠損する行でも数値placeholderは0.0ですが、`global_feature_available=False`と`global_feature_missing_reason=1`によって「経過時間不明の欠損」を表します。0.0だけで有効と判断してはいけません。古いDataset artifactに保存済みの欠損時1.0は、元のidentityのまま読み込み、書き換えません。

追加のnative calendar APIでは、指定されたUTC週・暦月・固定間隔と完全なsource memberを確認し、derived barの利用可能時刻を全memberのpublicationの最大値とします。時刻とOHLCVは欠損マスク付き配列やその所有元が追跡できるviewを拒否します。すでに呼出し前に失われた欠損情報は復元できません。この検査は取引所データの真正性・finalityや既存Datasetの長時間足対応を証明しません。

別の opt-in Spot 受信データ用入口は、収集済みのUTC combined streamの確定フラグ・銘柄・時間足・時刻を照合し、受信時刻を利用可能時刻として同じ集約処理へ渡します。未確定バーや欠損・重複、UTC以外のstreamを代用しません。ネットワーク収集や接続・受信時計の真正性、先物の資金調達料、既存Datasetへの接続は別の課題です。

明示したnative calendarの価格・出来高特徴量を判断時刻に対応付ける追加APIでは、週・暦月・固定間隔の連続した完全なバーを使用し、特徴量の窓はバー本数、鮮度は実経過時間で扱います。公開遅延と銘柄の有効期間を考慮します。資金調達・cross-assetのnative特徴量は未対応です。別のopt-in overlayでは、元のDatasetと同じ銘柄契約を持つ完全な上位足をsource digestで固定し、3日・UTC週・暦月など複数の上位足の特徴量を既存の基準足Datasetへ一括追加できます。公開時刻・鮮度・基準足のavailabilityを検査し、既存の価格・費用・資金調達・年率・約定時間足は変更しません。新しいcontent identityで保存・再読込し、既存の共通replayへ渡せます。粗い足そのものの約定、取引所データの真正性、実データによる学習や取引の認可を意味しません。

native sourceの時点整列では、判断時点までに公開されたeventのうち最も新しいcloseを選びます。長期履歴は固定サイズの基準足chunkに分けて処理し、遅れて公開された古いeventで最新値を巻き戻さず、利用可能時刻・経過時間・鮮度の数値契約を保ちます。

一目均衡表は従来の固定期間を維持しつつ、明示設定で転換線・基準線・先行スパンBの期間を足の本数で指定できます。可視雲の変位を使う場合は、判断行より過去の計算値を使う方式を明示し、最長の必要窓から判断行までの欠損・取引可能性・公開時刻を検査します。設定を変えると新しいDataset識別子になります。雲の上・中・下は、雲の上端を超えた距離、下端を下回った距離、雲の内側（端を含む）で厳密に0という値で区別でき、雲の色と捻じれは先行スパンAとBの差の符号で表します。遅行スパンは、変位と同じ本数だけ過去のcloseとの比較で、未来の行を読みません。変位の数え方はプラットフォームで異なり、TradingView相当は変位25です（既定値は変更しません）。

MACDは明示設定でfast・slow・signalの期間を、ストキャスティクス%Dは%Kのlookbackと平滑化期間を足の本数で指定できます。明示%Dの通常の有限正値float32/float64価格配列では、各窓の最大・最小を共有する単調dequeで求め、長い%K窓の繰り返し走査を省きます。%Dの平均は従来の固定順序の計算を維持します。MACD・ATR・ADX・方向性指標・EMAの作業配列も連続区間内に限定します。この作業配列の限定は欠損時のwarm-up、公開時刻、従来の設定・数値・保存identityを変えません。別の修正として、ADXの利用可否は連続した使用可能区間の2×lookback−1本目から有効になり、後に行が続くかどうかに依存しなくなりました。既に有効だった値は変わらず、ちょうどその長さの区間を含む新しいbuildではDataset識別子が変わり得ます。方向性指標はlookback本目からで変わりません。長い%D平滑化窓の計算量や全プロセスのメモリ上限を保証する改善ではありません。

`adx_causal_v1` を明示すると、同じADXを、欠損に加えて取引不可の足でもwarm-upをやり直す設定で使えます。期間pの場合、連続した利用可能な足が2p−1本に達した時点で最初の値を確定し、次の足を待たず、依存する全行の公開時刻を確認します。旧 `adx` も同じ2p−1本で有効になります（研究リセットに伴い、追加1本を要求する互換挙動は残しません）。両者は別kindのidentityであり、保存済みデータは置き換えません。

1時間足を基準に15分足などのcross-asset特徴量だけを選ぶ場合、不要なlog-return公開列は要求しません。内部では既存のnative raw sourceから1-bar returnを計算し、native clockの連続性、銘柄の有効期間、sourceの`available_at`をすべて確認したうえで基準足へ対応付けます。遅延して未利用のデータを過去の値として使ったり、基準足の価格でnative値を合成したりしません。基準足とそれ以上のnative timeframeでは、従来の明示1-bar source条件を維持します。

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

Scenario v2には、その既存native four-arm結果を各arm後に保存し、before/after provenanceとsource records、外部identity、expected result digestをstrict reloadで確認してからcompleteにするone-shot local attemptもあります。これはsoftware-integrity recordであり、別の帳簿や損益計算、研究判定を追加しません。

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
