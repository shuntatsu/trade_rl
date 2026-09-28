## 結論

現在のTrade RLは、**再現可能な実データbaselineとControlled Experimentを検証できる研究基盤までは成立しているが、継続的な利益性やwinner strategyはまだ証明していない**段階です。

Portable Controlled Experiment 0001は独立再検証まで完了し、formal decisionは **KEEP_BASELINE** です。

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

economic activationは非Pythonのcanonical `ppo_normalization_activation.json` で別管理し、現在も `activation_sha256=null` のためfail-closedです。activationはreview済みimplementationに加えてimplementation seal・fresh reconstruction・result-blind assurance reviewのdigestをbindします。さらにone-shot transport capabilityを別に実装し、current `main` を含むopen Draft request PR、exact-head Core / real-PPO / Guide / independent-review Green、固定source Artifact、未使用のstatic activation tagを再検証してから初めてactivationを作る契約にしています。request PR HEADは承認provenanceでありeconomic sourceではありません。request recordが#770のreview/seal済みsource SHAを別にbindし、fit/replayとno-refit verifierはそのsealed sourceをcheckoutするため、後からcurrent `main` に入ったPython変更を実験factorへ混入させません。10 slotsは同じactivation runtimeで実行し、全slot完了まではexecution artifactを公開しません。途中失敗で公開できるのは経済値を含まないfailure receiptだけです。

local `verified.json` だけではindependent verificationとは扱わず、complete execution artifactを別のno-refit verifierがid/run/raw digest付きで再取得し、10 slotsのverification identityをfresh verifier artifact authorityへbindした後だけfinalizerがcomparisonを公開します。ただし、このtransportが実装済みであること自体はeconomic authorizationではありません。現在はrequest activation tagも作成されておらず、corrected-economicsのfit/replay/P&Lは未実行です。

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
3. final-eligibleな新規Studyでは、development resultより前にbootstrap v3 / StudyPlan v2でunused windowを固定する。final startはdevelopment Datasetの最終timestampより後に置き、historical StudyPlan v1へwindowを後付けしない。
4. そのStudyがWINNERになった場合だけ、実装済みのfinal authorization gateでStudy freeze・winner evidence・事前登録windowをone-shot artifactへbindする。
5. authorizationとは別の将来consumerが、そのartifactを検証して初めてsealed unused-futureを開く。
6. final evaluation後もexecution stress、capacity、account-specific economicsを別途確認する。

**authorization gateが実装済みであることは、final Datasetを開いたこと・final P&Lを得たこと・production適格性を意味しません。** 現時点ではfinal economic evaluationそのものは未実行です。

**「baselineを再現できる」から「実運用で継続的に儲かる」までには、まだ複数の反証ゲートが残っています。**
