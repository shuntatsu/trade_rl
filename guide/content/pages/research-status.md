## 結論

Trade RLは再現可能なbaselineとcontrolled experimentを検証できる段階ですが、継続的な利益性や優位なstrategyはまだ証明していません。Portable Controlled Experiment 0001の正式判断は `KEEP_BASELINE` です。候補は5銘柄すべてでbaselineを上回りましたが、候補自身のリターンが正だったのは1銘柄でした。

## PPO中期保有期間の比較

次はPPOを使い、H=0と3 / 7 / 14 / 21日相当の最低保有期間を同じデータ・費用・リスク条件・5 seedで比べる設計です。v1は銘柄ごとの独立口座、v2は5銘柄を一つの100,000 USDT口座で評価する別protocolです。v2ではportfolio全体のリターンとdrawdownで判定し、Candidate Run v8に各intervalの完全なexecution ledgerを保存します。

v2のcapital境界、ledger保存、同一市場でのPPO学習環境とreplayの照合、multi-symbol手計算oracleを追加しました。PPOの学習自体はsingle-symbolのままなので、shared-cash portfolioを学習したことにはなりません。20% drawdownは研究対象の適格性条件であり、価格gap後の最大損失を保証する上限ではありません。

5銘柄、2023-01〜2026-08の開発期間、2026-11〜2027-11の未使用期間、immutable StudyPlanはresult-blindで固定済みです。fresh independent reviewと関連contract checksが終わるまでG0-G2は未確立で、v2のPPO fit・経済replayは未実行です。次はレビューを閉じ、H=0から新規学習して事前登録済み候補と比較します。

## まだ主張しないこと

- PPOやforecastがrule strategyより優れること。
- 中期保有が利益を生むこと、またはwinnerが選ばれたこと。
- 未使用期間でも同じedgeが続くこと。
- Production/live order routingの適格性。

費用とslippageは再現可能な研究仮定で、口座固有の実績値ではありません。優位性の主張には、凍結したwinnerを未使用期間で一度だけ検証する別ゲートが必要です。
