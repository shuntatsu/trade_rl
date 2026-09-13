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

次のControlled Experimentでも、変更要因を結果より前に一つ固定し、factor isolation、unaffected raw-return equality、metric invariance、cost semanticsを再検証します。

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
3. その後だけsealed final-testへ進む。
4. final-test後もexecution stress、capacity、account-specific economicsを別途確認する。

**「baselineを再現できる」から「実運用で継続的に儲かる」までには、まだ複数の反証ゲートが残っています。**
