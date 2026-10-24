## 結論

現在のTrade RLは、**再現可能な実データbaselineとControlled Experimentを検証できる研究基盤までは成立しているが、継続的な利益性やwinner strategyはまだ証明していない**段階です。

Portable Controlled Experiment 0001は独立再検証まで完了し、formal decisionは **KEEP_BASELINE** です。

## Task 18 Datasetの準備状況と次のゲート

Task 18のSourceVerifier 30cは、既存の公開`market_build_v3` / `identity_v7` Datasetをpublic loaderで読み、Dataset identity・scalar・aliasを確認しました。これはloader/source適合の確認で、48-cell比較やP&Lの証拠ではありません。Tier D preparation attemptでは4つのcost producerを開始し、4 configuration × 3 vintageの12 fitを完了しました。その後、Windows RSS samplerがvenv redirector（約4.4 MB）だけを測り、実Python child（約248.8 MB）を含まなかったためtechnical stopとなりました。native比較は0 / 48 armsで全arm未着手、このattemptではnative replayとP&Lを実行していません。attemptとprovenanceは保持され、科学的parameterは変えていません。同じ既消費Dataset・同じ設定はsampler修正後のTier D technical retryに再利用できます。

比較protocolのdeclared fee/spread/impactとfunding仮定は結果を見る前に固定済みですが、歴史的な市場実績へのcalibrationは未確立です。Funding markはbar-close proxyとraw rateの仮定に基づき、実際のsettlement markではありません。

Issue #810の次の比較は既存Datasetを使うNONRL 48-cellで、RL実データfit/replayより先に進めます。RLはfundingを含む金融clockとrollout/decision protocolを固定し、fee-only synthetic marketでno-trade learnabilityを通過するまで実データへ進めません。既存の6-fit PPO fee-only learnabilityは`FAILED`のままで、このtechnical attemptでも変更されません。native CASH controlやpolicy dropoutは学習済みno-trade能力や利益性を示しません。

### 条件付きfour-arm Ridge開発screen

attempt 01では全12行の計算が完了しましたが、screen全体の判定は `FAILED_SCREEN` です。3つのシナリオすべてでcash以外の3 armの費用反映後リターンは正でした。一方、latency 2 barsでは正のcost proxy armが2.6777%で、position-aware `cost=0` の2.8971%を下回り、必要な比較条件を満たしませんでした。全行は終端flat・残注文0で、実現ledger最大DDは20%未満でしたが、勝者や将来利益の証拠ではありません。全12行とDD・費用・fundingの詳細は正本の研究statusに記録しています。rootの別readbackとN2の71/71 source-M34 metadata/public-loader checksは完了しました。詳細reportとevidence packageは[M34のGitHub release](https://github.com/shuntatsu/trade_rl/releases/tag/research%2Fissue810-fourarm-M34-20261024-attempt01)に公開済みです（release ID `422708600`、tagはM34）。

別のopt-in software診断では、frozen Ridge予測とzero forecastを同一の成熟済みlog-return pairで比較できます。target endpointはscoring-as-of stopより前、target closeはその時刻までに利用可能であることを求めます。horizon labelが時間足で重なる場合、pair countは独立sample数や有意性を表しません。この診断はM34のscreenを再計算せず、予測vintage、portfolio return、winner、利益性の証拠にはなりません。

費用はDatasetの `fee_rate=0.0005`（5 bp）と `spread_rate=0.0002`（2 bp）を使い、追加overlayは0です。10 bp/weight-turnoverは意思決定用proxyで、ledger feeではありません。ledgerの費用とfundingは最終wealthに含まれ、再度差し引きません。fundingはbar-close proxyで、実際のexchange settlement markとの一致は未確認です。元の48-cell gateは未完了で、registryには1 interrupted・47 unattempted・0 completeが残っています。入力は過去利用のclosureが不明なTier Uです。PPO/RL、unused-period、将来利益、live運用の証拠ではありません。再実行やretuneはしません。

## PPO中期保有期間の次期設計

従来の独立口座版 `ppo_holding_duration_v1` は、PPOを主役に、Observation v3を共通で使うH=0と3 / 7 / 14 / 21日相当の最低保有期間を比較します。データ、評価期間、費用、初期資金、リスク条件、5 seedを揃え、seed内の銘柄平均を先に取ってからseed中央値でprimary scoreを決めます。H=0と候補の全独立口座で終端決済後の建玉と未約定注文をなくし、実現DDを20%以下に保つ条件も採点へ含めます。

共通資金版 `ppo_shared_cash_holding_duration_v2` も実装しました。5銘柄をseedごとに一つの100,000 USDT口座で評価し、個別銘柄リターンの平均ではなくportfolio全体のreturn/DDで選びます。bootstrap v6 / StudyPlan v6 / comparison v4でv1と分離し、終端flat・残注文なし・最大DD 20%以下・H=0比のseed中央値が正を条件、seed中央値returnをscoreとします。PPO学習はsingle-symbolのままです。mocked bootstrap / synthetic lifecycle testsは通過しましたが、G0-G2独立reviewは未完了で、v2 fit・経済replayは未実行です。利益性や勝者は未確認です。

正確なBinance Dataset、開発・最終期間、immutable StudyPlanは2026-10-01にresult-blindで固定し、ネットワークを使わない独立検査で同じidentityを再確認しました。5銘柄を2023-01から2026-08まで開発評価し、2026-11から2027-11を未使用期間として予約しています。これは研究を始めるための準備であり、H=0も保有期間候補もまだ学習・経済評価していません。まずG0-G2のfresh independent reviewとcontract checksを閉じてからH=0の学習へ進みます。Study workflowは外部レビューを認証するgateを持たず、書き手が用意したレビューJSONも承認証拠として受け付けません。G0-G2が閉じるまでは実行者が学習・候補実行を止める必要があります。G2は新しい実装へのfresh independent result-blind reviewと関連contract checksが終わるまで未確立です。Guide説明の人間確認はfingerprint更新前に必要な別の文書ゲートで、G2 oracleとは別です。PPO利益や保有期間の勝者は未確認で、利益性の主張には別の未使用期間評価が必要です。

## 検証済みの証拠

- `market_build_v3` と `portable_feature_numerics_v1` を固定。
- 価格付きDataset全体のidentityを、確認済みのAMD / Intel hosted runner間でbyte-identicalに再現。
- 結果を見る前のportable preregistrationをbaseline実行より先に封印。
- 5銘柄 × 8戦略 × 5 seedのportable baseline evidenceを生成。
- 取引あり175 observationsすべてでtotal costが正値であることを確認。
- 別runnerでDataset / StudyPlanを再構築し、生のCandidate Runsから独立再計算。
- Portable Controlled Experiment 0001をfresh result re-verificationまで完了。
- Experiment 0001で影響外のraw return一致150 checksと、決定的metricのseed不変性1120 checksを検証。

## Experiment 0001で分かったこと

mean-reversion candidateはbaseline比で5 / 5銘柄を改善し、median turnoverも低下しました。

しかしcandidate total returnが正だったのは **1 / 5銘柄** でした。

| 観測 | 結果 |
| --- | --- |
| baseline比で改善したmean-reversion銘柄 | 5 / 5 |
| candidate total returnが正の銘柄 | 1 / 5 |
| formal decision | `KEEP_BASELINE` |

事前登録ruleではpositive candidate symbolが3 / 5以下ならKEEP_BASELINEです。そのため、改善幅が見えてもruleを書き換えてcandidateを採用していません。

## 現在維持するもの

現在のdevelopment基準はbaselineのままです。

PPO feature standardizationには、historical comparisonとは別に **current corrected economics用のreplication software boundary** があります。raw / normalizedをseed 0..4でfresh fitする10 slotsを固定し、両armはfit-only normalization以外を共通化します。execution rootはprepareだけがstagingからatomicに公開し、slot claim/failureはprepared rootのidentityから導出します。bundleはmanifestとparent pathを安全に確認してから読込み、fit/reloadとも262,144 timestepsを必須にします。source/runtimeも長いfit/replay後、resultを保存する前に再確認します。

economic activationは非Pythonのcanonical `ppo_normalization_activation.json` で別管理し、source authorityの `activation_sha256=null` を維持します。activationはreview済みimplementationに加えてimplementation seal・fresh reconstruction・result-blind assurance reviewのdigestをbindします。one-shot transport capabilityはcurrent `main` を含むopen Draft request PR、exact-head Core / real-PPO / Guide / independent-review Green、固定source Artifact、未使用のstatic activation tagを再検証してからactivationを作る契約です。request PR HEADは承認provenanceでありeconomic sourceではなく、request recordが#770のreview/seal済みsource SHAを別にbindします。

one-shot authorizationはGitHub Actions run `36356182462`で既に消費され、tag `activation/ppo-normalization-corrected-v1`が作成されました。10 slotsのfitは完了し、execution artifact `10946283192`（SHA-256 `0c0d5335c389f13fe7b1ff22bd53d97025cc4b39253f280843baeba06024a81f`）が公開されましたが、fresh no-refit verifierは `fresh bundle replay differs from published result` で失敗し、finalizerは実行されていません。sourceをたどると、slot公開時に方向性評価結果のtop-level `schema` がslot用schemaへ置き換わる一方、fresh replayには元schemaが残り、保存payloadからそのfieldを除外した状態で比較していたため、全slotで不一致になる実装でした。このPRではreplay側の未保存schemaだけを比較対象から除き、その他のfieldは厳密一致させる回帰テストを追加しました。元artifactのno-refit再検証はまだ行っておらず、経済結果も確認していないため、artifactは引き続き未検証です。one-shot requestを再triggerしたり、この10 slotsを再fitしてはなりません。local `verified.json`だけではindependent verificationとは扱わず、未検証artifactからcomparisonやeconomic dispositionを確定しません。unused-future evaluation、production eligibility、live authorizationも成立していません。

trainingは既存のone-active-symbol episode / `risk_config=None`、evaluationはshared-cash accountと10%/20% drawdown hard riskという共通のtrain/eval差を残します。この差は両arm共通なのでnormalization-only比較のfactorは変えませんが、shared-cash問題そのものを学習済みだという主張はできません。したがって、この境界の実装完了はprofitability、unused-data validation、production/live適格性を意味しません。

次のControlled Experimentでも、変更要因を結果より前に一つ固定し、factor isolation、unaffected raw-return equality、metric invariance、cost semanticsを再検証します。

PPOには、既定sequentialを維持したままlayoutだけを比較できるopt-in要因も追加しました。合成CPUデータで同じ2,048 transitionsを学習した速度確認ではinterleaved/512の中央値が13.9%短くなりましたが、policy hashは異なり、実データの速度・経済性は未検証です。この結果だけではPPOの利益性や採用可否を判断せず、正式比較では実際のtransition数を揃えます。

## まだ主張しないこと

- 継続的なprofitability。
- winner strategyの選定。
- PPOやforecastがrule strategyより優れること。
- unused future dataで同じedgeが続くこと。
- Production/live order routingの認可。

## 残っている制約

| 制約 | 現在の意味 |
| --- | --- |
| portable numerics | repositoryが支援する標準Dataset構築と確認済みrunner環境が対象。任意platformまでの普遍保証ではない |
| fee / spread | 再現可能な研究仮定。account-specificな実績値そのものではない |
| market impact / slippage | Dataset-authoritativeなモデルは未導入 |
| Experiment 0001 implementation | freeze済み旧implementationで完了。後発修正を遡及適用していない |

PPO aggregate metricはExperiment 0001のformal decision oracleには使っていません。

## 次の研究ゲート

1. developmentで一因子Experimentを積む。
2. winner / no-winner判断を事前ruleに従って固定する。
3. 一般のfinal-eligibleな新規Studyはbootstrap v4 / StudyPlan v3、PPO保有期間protocolはbootstrap v5 / StudyPlan v5でunused windowを事前登録する。final startはdevelopment Datasetと申告済みconsumed-evidence scopeの両方より後に置き、historical StudyPlanへwindowを後付けしない。
4. そのStudyがWINNERになった場合だけ、実装済みのfinal authorization gateでStudy freeze・winner evidence・事前登録windowをone-shot artifactへbindする。
5. authorizationとは別の将来consumerが、そのartifactを検証して初めてsealed unused-futureを開く。
6. final evaluation後もexecution stress、capacity、account-specific economicsを別途確認する。

**authorization gateが実装済みであることは、final Datasetを開いたこと・final P&Lを得たこと・production適格性を意味しません。** 現時点ではfinal economic evaluationそのものは未実行です。

**「baselineを再現できる」から「実運用で継続的に儲かる」までには、まだ複数の反証ゲートが残っています。**
