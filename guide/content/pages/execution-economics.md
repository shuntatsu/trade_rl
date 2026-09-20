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

注文の消化率と実売買金額も分けます。`fill_ratio` と `unfilled_turnover` はsubmission時のreference priceで要求量と約定量を同じ基準へ揃え、価格変動だけで「全量約定」に見えないようにします。一方、`filled_turnover` はactual fill priceで実際に売買したnotionalを使います。

LIMITのcost分類も注文名だけでは決めません。そのbarで初めてeligibleになったLIMITがprocessing openですでにmarketableなら、そのfillはliquidity-takingとしてtaker feeとfull spreadを使います。openではmarketableでなくbar内touchまでrestしたLIMITはmaker扱いです。また、前バーから残っていたLIMITは次バーopenでcrossしても既にrestingしていた注文なのでmaker扱いを維持します。この区別はfee/spreadの分類だけで、fill価格・fill数量・capacityを変更しません。

区間収益は役割を分けます。`interval_net_return` は実際のfill・cost・funding・borrow・dividend・cash interestをすべて反映した最終equityの収益です。`interval_gross_return` は**同じ実約定経路**から明示cash flowを取り除いて価格損益を分離した診断値で、実際のfill priceを使います。OPENから期末までのasset returnへ事後position weightを掛ける近似や、costを0にして戦略を再実行した反実仮想ではありません。

session calendarでclose-to-close間隔がnominal barより長い場合、closed-session gapのcash interest / borrowはnext-open fill前の保有・現金へ適用し、processing bar分だけをfill後の状態へ適用します。したがって、週末をまたいで新しく建てたpositionへ週末分のborrowを遡及させません。continuous cadenceでは従来どおり1 bar分をfill後の状態へ適用します。

発注単位の整数個数を正確に記録し、保有数量と注文残量の両方へ同じ約定を反映します。小数の表示値を繰り返し加減して、最後の1単位が決済できなくなることを防ぎます。現金は実際の約定数量から計算し、最小発注額に届かない本当の端数は保持します。

流動性の上限で部分約定になる場合も、注文数量と約定金額の上限に収まる最大の発注単位数を求めます。逆算時の小数誤差を理由に、有効な1単位を落としたり、上限を超えたりしません。

明示的な決済専用の成行注文は、約定時に残っている反対方向の保有数量までに制限します。他の注文が先に約定していても、決済注文が新しいポジションを作ることはありません。決済できなくなった注文残量は理由付きで失効し、費用は約定した数量だけにかかります。

取引所の保存済みルールと対象データを検証した専用設定を選ぶと、同じ方向の保有量を減らす注文を自動で決済専用にできます。この場合だけ取引所由来の最小発注額を免除し、別途設定した実行時の下限や、数量の上下限・刻み幅は守ります。新規建てと売買方向の反転は通常の最小発注額に従います。設定を省略した既存研究の動作は変わりません。現在のルールを過去の相場へ適用することは研究上の仮定であり、利益の改善は別の比較で確認します。

## participation capacityの時間境界

現行executorはnext-openであり、decision row `t` の注文は最初にrow `t+1` のopenで約定可能になります。participation capacityには2つの明示的なmodeがあります。

- `processing_bar_volume_capacity=True` は既存互換modeです。約定を処理するbar全体の最終volumeをcapacity poolに使います。同じbarのopen時点では最終volumeはまだ確定していないため、point-in-timeで観測済みのliquidityとはみなしません。既存canonical runの意味を変えないためdefaultとして残します。
- `processing_bar_volume_capacity=False` はcausal stress modeです。直前に完全終了したbarのvolumeだけをcapacity authorityにし、base-volumeならその前barcloseでmarket notionalへ換算します。現在barの最終volumeをfill capacityへ使いません。

volumeの単位もcapacity計算に残します。QUOTE_NOTIONALは従来どおりquote notionalだけで上限を持ちます。BASE_ASSET / CONTRACTSはそのquote-notional上限に加え、raw base量 / raw contract数から得るnative quantity上限も同じparticipation率で適用します。そのため、LIMIT/STOPのactual fill priceがcapacity referenceより低くても、価格差を使って観測されたbase量やcontract数より多く約定させません。trigger segmentの利用可能volume fractionもquote/notionalとnative quantityの両poolへ同じ割合で掛かります。

`False` は「次barの流動性を正しく予測できる」という主張ではありません。同一barの未来volumeへ依存しない条件でedgeが残るかを見るための、より保守的なstressです。modeはexecution-policy identityへ含まれるため、既存runを後から別modeとして読み替えません。

## 研究仮定と実市場truthを混同しない

現行のfee・spread等は、比較可能な研究のために固定した**再現可能な仮定**です。過去の特定accountにおけるBinance実績feeや、観測不能なqueue position、hidden liquidityまで再現したとは主張しません。

| 項目 | 現在の扱い |
| --- | --- |
| fee | Datasetが正本 |
| spread | Datasetが正本 |
| participation上限 | Datasetが正本 |
| capacity volume source | execution-policyでlegacy / causal stressを区別 |
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
