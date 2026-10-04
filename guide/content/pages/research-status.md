## 結論

現在のTrade RLは、**再現可能な実データbaselineとControlled Experimentを検証できる研究基盤までは成立しているが、継続的な利益性やwinner strategyはまだ証明していない**段階です。

Portable Controlled Experiment 0001は独立再検証まで完了し、formal decisionは **KEEP_BASELINE** です。

## PPO中期保有期間の次期設計

従来の独立口座版 `ppo_holding_duration_v1` は、PPOを主役に、Observation v3を共通で使うH=0と3 / 7 / 14 / 21日相当の最低保有期間を比較します。データ、評価期間、費用、初期資金、リスク条件、5 seedを揃え、seed内の銘柄平均を先に取ってからseed中央値でprimary scoreを決めます。H=0と候補の全独立口座で終端決済後の建玉と未約定注文をなくし、実現DDを20%以下に保つ条件も採点へ含めます。

共通資金版 `ppo_shared_cash_holding_duration_v2` も実装しました。5銘柄をseedごとに一つの100,000 USDT口座で評価し、個別銘柄リターンの平均ではなくportfolio全体のreturn/DDで選びます。bootstrap v6 / StudyPlan v6 / comparison v4でv1と分離し、終端flat・残注文なし・各seedの最大DD 20%以下に加え、candidateのseed中央値returnが正で、H=0比のpaired return中央値も正の場合だけeligibleとし、seed中央値returnをscoreとします。H=0を上回っても、候補が損失のままなら利益目標は満たしません。この絶対return条件は開発用の選別基準で、将来の利益を示すものではありません。 OHLCの高値・安値stressはバー内の価格順序を再現できないため、実現価格経路ではなく保守的な価格幅による下落stressです。これは研究専用の評価診断で、実運用の「1銘柄ずつ独立account」は変えません。PPO学習はsingle-symbolのままです。各時間帯の約定・保有・リスク判断を完全な記録として保存し、学習と評価の売買判断を同じ市場データで照合するテスト、複数銘柄を扱う時の資金・費用計算を手計算で照合するテストを備えます。EvidenceSet生成時とStudy再読込時のsource-binding oracleは、順序付き会計遷移に記録されたopen / mark価格、fundingの発生・rate・timestamp・multiplier、split、delisting、dividend、cash / borrow rate、経過時間を元Datasetの該当行へ照合します。borrow chargeのrate multiplierは台帳値を信用せず、登録済みexecution overlayから解決した値と照合します。canonical Dataset identityも全identity arrayから再検証し、source行の変更に加え、volume・参加上限・fee・closeなどidentity arrayの変更とborrow-rate multiplierの改ざんを、保存ledgerを固定したmutation testで拒否します。Study再読込時も元Dataset artifactとStudyPlan固定のexecution overlayを照合し、Run・EvidenceSet・analysis digestを再計算した一貫改変を拒否するテストを追加しました。これらの契約テストは通過しましたが、独立レビューとrepository quality gatesは未完了で、G0-G2は未確立です。2026-10-04 00:40 JSTにH=0 baselineの実行を開始しましたが、現headのfresh G0-G2 review未完了のため停止しました。旧source SHA 370a97e6からの実行で、結果と出力は確認・検証していません。利益性やwinnerは未確認です。

正確なBinance Dataset、開発・最終期間、immutable StudyPlanを含むcanonical v6 bootstrapは2026-10-03にresult-blindで作成し、ネットワークを使わない検査に通りました。workspace-only packetは `report/ppo-shared-cash-canonical-v6-20261003/` にあり、Studyには固定済みplanだけが保存されています。5銘柄を2023-01から2026-08まで開発評価し、2026-11から2027-11を未使用期間として予約しています。baseline runは旧sourceで開始後、現headのG0-G2 review未完了のため停止しました。出力と部分P&Lは未確認で、candidate experimentは実行していません。G0-G2とcontract checksを閉じ、baseline/evidence状態の確認後に次へ進みます。Study workflowは外部レビューを認証するgateを持たず、書き手が用意したレビューJSONも承認証拠として受け付けません。G0-G2が閉じるまでは実行者が学習・候補実行を止める必要があります。G2は新しい実装へのfresh independent result-blind reviewと関連contract checksが終わるまで未確立です。Guide説明の人間確認はfingerprint更新前に必要な別の文書ゲートで、G2 oracleとは別です。PPO利益や保有期間の勝者は未確認で、利益性の主張には別の未使用期間評価が必要です。

v11の共有口座replayは、終値リターン列とは別に、各バーの高値・安値から保守的な最大下落stressを計算します。有利な価格でpeakを更新してから、不利な価格で下落を測り、各約定の直後にも記録します。OHLCにはバー内の価格順がないため、これは実現経路の再現ではなく、保守的な価格幅のstressです。総リターンは引き続き区間ごとのリターン列で照合します。約定証跡には割当ロットから計算した数量と台帳に実際に適用した数量差分を別々に保存し、no-lotで現在数量を閉じる時の微小な端数差も検証します。

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
