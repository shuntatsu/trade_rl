# Causal Alpha V5 r7 実データ検証・GPT引き継ぎレポート

## GPTへ渡す依頼

以下は、Binance実データ・9銘柄・銘柄別独立long/shortを前提にした Causal Alpha V5 の固定Signal検証結果です。

目的は「同時点の全銘柄比較」ではなく、各銘柄で独立に売買し、手数料・スリッページ等控除後の資産を最大化することです。無出来高・無変動域での無駄売買、損失を伴う頻繁なポジション変更を避け、4時間から数日続く波を保持できる設計を求めています。

V5はOOMを修正して全Signal区間を完走しましたが、slow laneは648サンプル中active 0件で棄却されました。一方、既存V4 fast 4h laneは固定Signalゲートを通過しています。下記の証拠を前提に、holdoutを覗かず、閾値を事後的に緩めず、fast-firstで可変保持期間を持つ次世代設計を検討してください。

## 結論

- r7はCalibration 8/8、Signal 8/8を完走した。
- 実行終了は `signal_rejected`（exit 2）であり、クラッシュではない。`OOMKilled=false`。
- V5 slow laneは72 scope、9銘柄、8独立episodeの完全な期待scopeを評価した。
- slow laneのactiveは `0 / 648`、active coverageは `0.0%`。
- 無条件slow方向精度は平均 `48.6111%`（50%超過分 `-1.3889%`）。
- 無条件Rank ICは平均 `+0.0275463` だが95% CI下限は `-0.0530093`。
- 無条件top-bottom spreadは平均 `+0.00442344` だが95% CI下限は `-0.00271794`。
- 648件の非active理由は `confidence_abstain=374`、`direction_disagreement=274`。edge/cost hurdleへ到達したactive候補は0件。
- V4 fast 4h laneは同じrunでSignal合格 (`v4_fast_lane_passed=true`)。
- よって「confidence閾値だけを下げる」は不適切。方向精度そのものが50%未満で、区間安定性も不足している。
- Selection、Admission、BC、RLは仕様どおり未実行。資産曲線・net wealth・turnover・execution costに関する成功主張はできない。

## 再現情報

| 項目 | 値 |
|---|---|
| Repository worktree | `C:\dev\trade_rl\.worktrees\causal-alpha-v5-research` |
| Branch | `codex/causal-alpha-v5-research` |
| HEAD | `e06d8e1efd90e5495d7dda1fe48dd0f98d5b97b3` |
| Source-tree digest | `42a46663d47f8145dc543bbc53a8e065830a8274d03886113072fd828b4490a2` |
| Lockfile digest | `95dddd1ed146c4738004a0f3c97458737184cb5c03c730167af46f345e9c213b` |
| Runtime manifest digest | `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0` |
| V4 context manifest digest | `bc91783061182e41415d45a714049737ae16564a47d0e1ca14d004cc4c5c7357` |
| Docker image | `trade-rl-causal-alpha-v5:e06d8e1efd90-6726b3737df9` |
| Docker image digest | `sha256:ae6695061d1810bfc37321d6d006b76c39eba4b184d5663b98592e7618e878fc` |
| Container | `trade-rl-causal-alpha-v5-prod-20260825-r7` (`0460458b3010...`) |
| User | `trainer` (non-root) |
| Started (UTC) | `2026-08-24T20:12:49.957054484Z` |
| Finished (UTC) | `2026-08-24T20:37:59.939235611Z` |
| Exit / OOM | `2 / false` |
| Artifact root | `/workspace/var/runs/causal-alpha-v5-prod-20260825-r7` in volume `trade-rl-training-data` |

主要artifact:

- `calibration/evidence.json`: Calibration 8 fitのdigestを固定。
- `signal/evidence.json`: Signal棄却と6理由を固定。
- `signal/diagnostics.json`: bootstrap実数値と72 scopeの全内訳。
- `result.json`: terminal `signal_rejected`。

主要digest:

| Artifact | Digest |
|---|---|
| Run manifest | `b65b8b74d85f58a32c3eaeb79841fbeaa59a073580800037b894586554407083` |
| Calibration evidence | `1680fa1e01a576e748e6a29a7d254aaf7b23a7fd1c80820676ca2ee4635edbc6` |
| Signal evidence | `3df8d8427444eec96da35811f066d7a067e0fe1b8d719e66bd06241a603237f4` |
| Signal diagnostics | `6545d3d26fa2a095fc7c978d602be58361bc7492b35ae62651c1caec4fd21dc9` |
| Terminal evidence | `18e352a4a4819cd56cd4f3f87ecaa8fd1636b03fac857302bd38b1310541fc8a` |
| Result envelope | `7fa33d742c61943984792413714cc06433b1fcbca4976d489b577e03e981117e` |

## 固定Signal統計

| 指標 | 平均 | 95% CI下限 | 95% CI上限 | p値 | 判定 |
|---|---:|---:|---:|---:|---|
| Unconditional Rank IC | 0.0275463 | -0.0530093 | 0.1194444 | 0.2593741 | fail |
| Unconditional top-bottom spread | 0.00442344 | -0.00271794 | 0.00866482 | 0.00009999 | fail (下限基準) |
| Unconditional direction accuracy excess | -0.0138889 | -0.0879630 | 0.0570988 | 1.0 | fail |
| Selective direction accuracy excess | -0.5 | -0.5 | -0.5 | 1.0 | fail (active 0) |

固定棄却理由:

1. `unconditional_rank_ic_lower_ci`
2. `unconditional_top_bottom_spread_lower_ci`
3. `unconditional_direction_accuracy_excess_mean`
4. `selective_direction_accuracy_excess_lower_ci`
5. `active_coverage`
6. `scope_active_support`

Support:

| 項目 | 値 |
|---|---:|
| Raw scopes | 72 |
| Independent episodes | 8 |
| Symbols | 9 |
| Raw non-overlap samples | 648 |
| Raw direction samples | 648 |
| Active samples | 0 |
| Active direction samples | 0 |
| Overall active coverage | 0.0% |
| Confidence abstain | 374 |
| Direction disagreement | 274 |

## 銘柄別slow Signal

各銘柄は8 episode、raw 72サンプル。すべてactive 0で、全8 scopeがminimum active supportに不合格。

| Symbol | Rank IC mean | Direction accuracy | Spread mean | Active coverage |
|---|---:|---:|---:|---:|
| APTUSDT | 0.035417 | 47.2222% | -0.007950 | 0% |
| ARBUSDT | 0.120833 | 47.2222% | 0.001736 | 0% |
| BCHUSDT | 0.068750 | 48.6111% | 0.010241 | 0% |
| BNBUSDT | -0.137500 | 51.3889% | -0.000797 | 0% |
| BTCUSDT | -0.087500 | 54.1667% | -0.000249 | 0% |
| LINKUSDT | 0.077083 | 43.0556% | -0.004962 | 0% |
| LTCUSDT | -0.025000 | 52.7778% | -0.004668 | 0% |
| SOLUSDT | 0.075000 | 45.8333% | 0.029172 | 0% |
| XRPUSDT | 0.120833 | 47.2222% | 0.017288 | 0% |

銘柄ごとに符号・順位・spreadが一致しておらず、「どの銘柄でも勝てる」slow universal signalの証拠にはなっていない。BTC/LTC/BNBは方向精度が50%を超えるがrank/spreadが弱いか負。AR/BCH/SOL/XRPはrank/spreadに一部正の兆候があるが方向精度は50%未満。

## Episode別slow Signal

| Episode | Decision range | Rank mean | Direction excess | Spread mean |
|---:|---|---:|---:|---:|
| 0 | 8527–11408 | 0.031481 | 0.006173 | 0.022778 |
| 1 | 11408–14289 | -0.050000 | -0.154321 | -0.007779 |
| 2 | 14289–17170 | 0.279630 | 0.129630 | 0.024840 |
| 3 | 17170–20051 | 0.116667 | 0.043210 | -0.007087 |
| 4 | 20051–22932 | -0.064815 | -0.104938 | -0.003575 |
| 5 | 22932–25813 | 0.131481 | -0.179012 | 0.020473 |
| 6 | 25813–28694 | -0.125926 | 0.055556 | -0.003029 |
| 7 | 28694–31575 | -0.098148 | 0.092593 | -0.011233 |

時系列で符号が頻繁に反転しており、単一の24h/72h slow zero-point校正がレジームを跨いで安定していない。

## 実装上の修正履歴

主な確定コミット:

- `cde12b71`: artifact-bound実データstage実装。
- `be8a837f`: 72hを含むcalibration splitを実データ上で成立させた。
- `4de44a9f`: calibrationメモリを制限。
- `9fc5d1ef`: cutoff fitをgate単位にstream化しholdout隔離を強化。
- `9e186f34`: V5/V4 Signal fitを二段分解し同時保持を解消。
- `e53caf4d`: 不要なruntime/context/prepared_v3参照をStage前に解放しOOMを解消。
- `e06d8e1e`: Signal scalar/scope診断をimmutable artifactに追加。

OOM経緯:

- r2/r3: 複数cutoff fitの蓄積でOOM。
- r4: Signal最終cutoffでV5 fitとV4 fitを同時保持してOOM。
- r5: 二段分解後も不要な準備入力が残り、最終V4 fitでOOM。
- r6/r7: 参照解放後、Calibration 8/8 + Signal 8/8を完走。定常メモリは概ね約3.0–3.1 GiB。r7はOOMなし。

## 解釈と次設計への制約

### 確認できたこと

- 銘柄ごとの環境、forecast、target、replayは独立している。9銘柄同時の横断ランキングで売買する設計ではない。
- universal/shared modelは銘柄間で学習を共有するが、symbol-ID固有lookupや銘柄別interceptは使わない。
- reward契約は純粋な控除後net equityの対数成長を維持している。
- ただしV5はSignalで止まったため、経済replayとreward/資産推移はまだ評価されていない。
- 1分足を追加すべき証拠はまだない。gross edgeがpositiveでnetだけcost負け、またはbar-path感応度が大きい段階に到達していない。

### やるべきでないこと

- `minimum_selective_confidence=1.0`だけを事後的に下げる。
- 失敗した銘柄・episodeを除外する。
- Signalを飛ばしてSelection、Admission、BC、RLを実行する。
- holdoutを見て設計・閾値を調整する。
- active 0を「損失なし」や「安全」と解釈して成功扱いする。

### 次の有力仮説

V4 fast 4h laneが通り、V5 slow laneが全面停止したため、次世代はslow-anchor-firstではなくfast-firstが妥当。

候補構造:

1. 4h fast signalをエントリー/方向の一次信号にする。
2. 24h/72hは独立した方向決定ではなく、保持・縮小・exit hysteresis・レジーム判定に限定する。
3. 各銘柄のposition state、PnL、cost、turnover、holding durationを独立に評価する。
4. shared modelで銘柄横断一般化を狙いつつ、銘柄IDによる暗記は許可しない。
5. rewardは `scale * log(net_equity_after / net_equity_before)` のまま変更しない。
6. slowの不一致時は即flipせず、既存positionの保持または段階的risk reductionを優先する。
7. 固定Signal通過後にのみSelectionでnet wealth、turnover、cost、holding durationを評価する。
8. 1分足は、Selectionでgross positive/net non-positiveかつcost消費が大きい場合のexecution-only仮説として後段で検討する。

## 検証状態

- V5対象テスト: `94 passed`。
- 変更範囲Ruff: pass。
- 変更範囲Mypy (`--follow-imports=skip`): pass。
- Docker image内torch compile probe: pass。
- Docker実データrun: completed through fixed Signal terminal, exit 2, OOM false。
- Selection/Admission/BC/RL: not run by fixed fail-closed ordering。

ホストのfull pytest collectionはoptional dev環境にtorchがないため不可。Docker training imageにはtorchがあり、実runで使用済み。リポジトリ全体の既存Windows/type issueをV5完了と混同しないこと。

