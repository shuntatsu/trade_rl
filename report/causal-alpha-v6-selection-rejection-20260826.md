# Causal Alpha V6 実データSelection棄却・GPT引き継ぎレポート

## GPTへ渡す依頼

以下は、Binance実データ、9銘柄、銘柄別に独立したlong/short、純粋な控除後net-log equity報酬を前提にした Causal Alpha V6 fast-first 戦略の固定検証結果です。

目的は同時点の銘柄横断ランキングではなく、shared modelで一般化しながら各銘柄の独立売買で控除後資産を増やすことです。4h fast signalでentryし、24h/72hは保持・縮小・exitだけに使い、無出来高・無変動域の無駄売買、損失を伴う頻繁なposition変更を避ける設計です。

V6はSignalを通過し、9銘柄 x 8独立episode x 2候補のSelection経済replayを完走しました。しかしfast-onlyとfast+slow-retentionは経済結果が完全同値で、どちらも手数料前から資産を減らし固定Selection gateで棄却されました。下記の証拠を前提に、holdoutを開かず、閾値を事後的に緩めず、entry方向・forecast calibration・regime適合を根本から見直してください。

## 結論

- r3はSignal 8/8とSelection 8/8を完走した。
- 終了は正規の `selection_rejected`（exit 3）であり、クラッシュではない。`OOMKilled=false`。
- Signalは両候補とも72 scope、9銘柄、8独立episode、207,360 decisionで合格した。
- Selectionは候補ごと72 scope、合計144 replayを維持されたexecution/accounting simulatorで評価した。
- fast-onlyとfast+slow-retentionは全Selection経済指標が完全同値だった。slow retentionは実現損益を改善していない。
- symbol-balanced gross wealthは `0.9439408691`、net wealthは `0.9385296752`。手数料前から約5.61%、控除後約6.15%減った。
- net/gross retentionは `0.9942674441`。costによる追加悪化は約0.54 percentage pointであり、主因はcostではなくgross edgeの欠如。
- net-positive scopeは `15 / 72 = 20.8333%`。median symbol net wealthは `0.9635612312`、minimumはSOLの `0.8337952115`。
- 9銘柄中、gross wealthが1を超えたのはLTCだけ（`1.0035418326`）で、netでは `0.9983602284`。残り8銘柄はgross段階から負けた。
- turnover p95は `0.191596555 / day`、sign flipは0。target changeは約0.56回/day、executed changeは約0.47回/day、closed tradeは約37 scope-daysに1回で、高速churnが主因ではない。
- hard risk violationとunexplained execution rejectionはいずれも0。
- Admission、BC、RLはfail-closed順序どおり未実行。holdoutは開いていない。
- aggregate grossが負なので、1分足を追加してexecutionだけ改善してもuniversal戦略は救えない。LTC単体にはcost感応の兆候があるが、失敗銘柄を除外する根拠には使わない。

## 再現情報

| 項目 | 値 |
|---|---|
| Repository worktree | `C:\dev\trade_rl\.worktrees\causal-alpha-v5-research` |
| Branch | `codex/causal-alpha-v5-research` |
| Run HEAD | `d1abd3a0201da1331e26c8950166cd6ddd9dccc7` |
| Report creation commit | `5d608b809016e974b723e88f7a4a23b9a2cbef55` |
| Source-tree digest | `f65b5aa54002dac4259c2042ccc27908d6905a68fc5ff306ba296ec2d1ccf0a0` |
| Lockfile digest | `95dddd1ed146c4738004a0f3c97458737184cb5c03c730167af46f345e9c213b` |
| Runtime manifest digest | `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0` |
| Runtime manifest raw SHA-256 | `e1f47c0d05cdebc27d65027bf7ce5170c01796e0fd6c3a2f36e2a1428d8887f3` |
| V4 context manifest digest | `bc91783061182e41415d45a714049737ae16564a47d0e1ca14d004cc4c5c7357` |
| Research config digest | `2d876d33b016d90b1dff9f4227f78189e8b64c31fb26bd497c2d2afe6e287222` |
| Target config digest | `067855764dae501ad2f7c93861a83ce19dba89dda0aa73dc4b215facd8951173` |
| Generator code digest | `c64e8f0ceb451a4ed05d398ce50d2a1be68691b67074d1350d85c21f975ca33e` |
| Run manifest digest | `6c1dc84c847a47ad83921a7b26bfacbda4af8aa355ac7e18aea26f060786c860` |
| Docker image | `trade-rl-causal-alpha-v6:d1abd3a0201d-6726b3737df9` |
| Docker image digest | `sha256:36f4641dca4b7d934726de81324c8653698153a18b286968ea63a5e20fcfd9ab` |
| Container | `trade-rl-causal-alpha-v6-prod-20260826-r3` (`4e3c92a88dbd...`) |
| User | `trainer` (non-root) |
| Started (UTC) | `2026-08-25T15:33:29.801395221Z` |
| Finished (UTC) | `2026-08-25T16:41:04.154461246Z` |
| Exit / OOM | `3 / false` |
| Artifact root | `/workspace/var/runs/causal-alpha-v6-prod-20260826-r3` in volume `trade-rl-training-data` |

## Artifact証跡

| Artifact | Canonical artifact/evidence digest | Raw file SHA-256 |
|---|---|---|
| `authored-config.json` | `a73bb90de2f9a053eef9da3470275fdf51edb4c344dc4e70bb3fb043705d24ed` | `9774560942e82a15a754c6f3a2f3ab8e72cf1ac04a017d7cf28b9c638779fc84` |
| `signal/evidence.json` | `1576305ee986475bb3752916df3b150e4305a66783808d0b51b43426ff28c32d` | `b26b7e5f02ddb2b620c9a5116353755e06bfb1b8e66707b546e0843c7cbafdb1` |
| `signal/diagnostics.json` | envelope-bound Signal diagnostics | `6dddb7d57e497130197d9f44576ac632ef9af08372381a7da0751f7e9033d5fd` |
| `selection/evidence.json` | `2ebdb606d635e6b50ca1c8eeeb3d5a882d0f99229dd44deada6d9febb1486d86` | `4ec1b43a8fe78c0fe9f0c3c9be4ebc0225474fe75b69cad380d0ca0d1ec8c98f` |
| `result.json` terminal evidence | `9b89402014b2609d26907e90739e8228cc1bd9815320936422e3d015a2d624af` | `ef01892af518901d3c032a293518e84d62fd31be0575245314f3fc210205648d` |

`admission/evidence.json`、research package、BC artifact、RL artifactは存在しないことを確認した。

## 固定Signal結果

| 指標 | fast-only | fast+slow-retention |
|---|---:|---:|
| Raw scopes | 72 | 72 |
| Independent episodes | 8 | 8 |
| Symbols | 9 | 9 |
| Decisions | 207,360 | 207,360 |
| Actionable decisions | 207,360 | 207,360 |
| Non-flat targets | 113,912 | 113,912 |
| Target changes | 872 | 871 |
| Sign flips | 0 | 0 |
| Slow direction accuracy | 52.8405% | 52.8405% |
| Signal gate | pass | pass |

V4 fast laneも同じrunで合格した。Signal段階ではretention候補がtarget changeを1件だけ減らしたが、Selectionの実行損益・turnover・cost・trade数は完全同値になった。

## 固定Selection結果

| 指標 | fast-only | fast+slow-retention | Gate |
|---|---:|---:|---|
| Symbol-balanced gross wealth | 0.9439408691 | 0.9439408691 | fail (`> 1`) |
| Symbol-balanced net wealth | 0.9385296752 | 0.9385296752 | fail (`> 1`) |
| Median symbol net wealth | 0.9635612312 | 0.9635612312 | fail (`>= 1`) |
| Minimum symbol net wealth | 0.8337952115 | 0.8337952115 | fail (`>= 1`) |
| Positive net scope fraction | 20.8333% | 20.8333% | fail (`>= 50%`) |
| Scope net-return CVaR 10% | -0.0660374079 | -0.0660374079 | diagnostic |
| Worst symbol-episode net return | -0.1168818465 | -0.1168818465 | diagnostic |
| Net/gross retention | 0.9942674441 | 0.9942674441 | pass-like diagnostic |
| Turnover p50 / day | 0.0055808232 | 0.0055808232 | pass |
| Turnover p95 / day | 0.1915965550 | 0.1915965550 | pass (`<= 1`) |
| Meaningful execution scopes | 43 / 72 | 43 / 72 | pass |
| Total target changes | 1,217 | 1,217 | diagnostic |
| Total executed changes | 1,023 | 1,023 | diagnostic |
| Total closed trades | 58 | 58 | diagnostic |
| Total sign flips | 0 | 0 | diagnostic |
| Total execution cost | 5,671.611111 | 5,671.611111 | diagnostic |
| Hard risk violations | 0 | 0 | pass |
| Unexplained execution rejections | 0 | 0 | pass |

両候補の固定棄却理由:

1. `symbol_balanced_gross_wealth`
2. `symbol_balanced_net_wealth`
3. `minimum_symbol_net_wealth`
4. `median_symbol_net_wealth`
5. `positive_net_scope_fraction`

Selection全体は `no_eligible_candidate`。selected candidateはnull。

## 銘柄別資産

両候補で同値。各銘柄は8独立episodeを連結したsymbol-balanced wealth。

| Symbol | Gross wealth | Net wealth | Net損益 | Meaningful scopes |
|---|---:|---:|---:|---:|
| APTUSDT | 0.9829506077 | 0.9824287329 | -1.7571% | 5 / 8 |
| ARBUSDT | 0.9939174986 | 0.9917234501 | -0.8277% | 5 / 8 |
| BCHUSDT | 0.9851193355 | 0.9835342614 | -1.6466% | 4 / 8 |
| BNBUSDT | 0.9615086336 | 0.9552331873 | -4.4767% | 5 / 8 |
| BTCUSDT | 0.9113323139 | 0.9079427127 | -9.2057% | 4 / 8 |
| LINKUSDT | 0.9718786051 | 0.9635612312 | -3.6439% | 5 / 8 |
| LTCUSDT | 1.0035418326 | 0.9983602284 | -0.1640% | 5 / 8 |
| SOLUSDT | 0.8440215051 | 0.8337952115 | -16.6205% | 5 / 8 |
| XRPUSDT | 0.8570403317 | 0.8475339698 | -15.2466% | 5 / 8 |

「どの銘柄でも勝てる」という目的に対して、netで勝った銘柄は0/9。BTC、SOL、XRPのgross lossが特に大きい。LTCだけはgross positive / net slightly negativeでexecution cost感応の局所例だが、universal aggregateはgross negativeであり、1分足を全体へ導入する根拠にはならない。

## Reward・cost・holding/churnの解釈

- run configは `scale=100`、`absolute_growth_weight=1`、その他の報酬項0で、純粋なnet-log equity成長。
- 全144 replay metric構築時に `reward_total == 100 * net_log_return` を `rel_tol=1e-9, abs_tol=1e-12` でfail-closed検証した。Selection evidenceが生成されたため、全scopeがこの不変条件を通過した。
- V6 artifact内のreturnは対数収益率、wealthは `exp(log_return)`。r1で単純複利returnをlog returnとして扱うadapter bugを検出し修正した。
- gross wealth 0.94394に対してnet wealth 0.93853。costは損失を増やしたが、gross loss自体を反転させる規模ではない。
- 72 scopeは各30日なので合計2,160 scope-days。target change 1,217は約0.563/day、executed change 1,023は約0.474/day。
- sign flip 0、closed trade 58は約0.0269/day、すなわち約37.2 scope-daysに1 closed trade。高速なlong/short反転や短期churnは発生していない。
- `liquidate_on_end=false`のためepisode末にopen positionが残り得る。Selection summaryは個々のholding-duration配列を保存しないが、低turnover、flip 0、closed trade頻度が長期保持を裏づける。
- `total_submitted_change_count=121,786`はtarget変更数ではない。保持中にcurrent weightが価格変動でtargetから微小にずれるため毎decisionでproposal判定されるが、実target changeは1,217、実execution changeは1,023、turnover p95は0.192/dayに抑えられている。

## 実装上の修正履歴

V6主実装コミット:

- `fe2abef5`: fast-first二段分解設計と実装計画。
- `e5863dd3`〜`59d742be`: target、Signal、replay、Selection、Admission、pipeline、artifact-bound stage、runnerを実装。
- `52a448e6`: replay evaluatorの単純複利returnをlog-return単位へ正規化。報酬関数は変更していない。
- `d1abd3a0`: V4 fit後RMSE予測を4096-row block化し、数学・fit対象を変えずOOM peakを解消。

実run経緯:

| Run | Terminal | 原因 / 証拠 | 対応 |
|---|---|---|---|
| r1 | exit 5, OOM false | Signal 8/8後、reward単位adapter mismatch | RED回帰後 `52a448e6` |
| r2 | exit 137, OOM true | Selection 2/8後、cutoff 37337 fit単体でも再現 | RMSE全量予測を特定 |
| fit diagnostic | old image exit 137 | preparation RSS約2.95 GiBから37337 fitでOOM | RMSE無効化時は約3.0 GiBで成功 |
| fixed fit diagnostic | exit 0 | 新imageでRMSE有効の37337 fit成功、RSS約2.91 GiB | root-cause fix確認 |
| r3 | exit 3, OOM false | Signal 8/8 + Selection 8/8完走 | 正規Selection棄却 |

## 根本原因と次設計への制約

### 確認できたこと

- 各銘柄の環境、position、PnL、cost、reward、target pathは独立。横断rankingで売買していない。
- shared V4 modelは9銘柄で学習を共有するが、symbol ID lookupや銘柄固有interceptを使わない。
- 4h fast-firstはSignal livenessを満たすが、Selectionのrealized gross PnLへ変換できていない。Signal passはprofitability passではない。
- 24h/72h retentionはSignal target changeを1件減らしただけで、実現経済指標はfast-onlyと完全同値。現在のslow stateは行動に実効的影響を与えていない。
- cost、turnover、hard risk、execution rejectionは主因ではない。方向・entry timing・forecast calibrationが主因。
- 波を長く保持すること自体は実現しているが、誤った方向を長く保持すると損失が拡大する。holding durationを伸ばすだけでは改善しない。

### やるべきでないこと

- Selection結果を見てtarget threshold、confirmation count、gate thresholdを緩める。
- SOL/XRP/BTC等の失敗銘柄や負けepisodeを除外する。
- gross-negativeを1分足executionで救えると仮定する。
- Selectionを飛ばしてholdout Admission、BC、RLを実行する。
- slow retentionの経済同値を「安定」と解釈する。

### 次の有力な研究仮説

1. Selectionデータ内だけで、long/short別、entry reason別、forecast confidence別、volatility/liquidity/regime別のgross PnL attributionを追加し、方向が逆転する条件を特定する。
2. Signal livenessとは別に、4h forecastの符号calibrationとentry時点のrealized gross edgeを直接検証する。rank/accuracyが通ってもtradable entry edgeが負なら棄却する。
3. fast proposalをそのままpositionへ写すのではなく、shared regime representationで「取引可能な波」と「逆行しやすい局面」を分離する。ただしsymbol ID暗記は許可しない。
4. slow contextを残すなら、hold/reduce/exitのどのdecisionを変えたかとそのcounterfactual gross PnLを証跡化する。経済同値ならslow branchを削除する。
5. 1分足はaggregate gross-positive戦略が得られた後、LTCのようなgross-positive/net-negative scopeでexecution-only検証する。今は導入しない。
6. rewardは引き続き純粋な控除後net-log equityとし、補助penaltyで見かけの学習報酬を改善しない。

## 検証状態

- V4–V6 targeted suite: `297 passed in 43.84s`。
- V6全テスト: `83 passed`（reward fix時点）。
- Ruff: pass。
- Mypy: pass。
- Import Linter: 13 contracts kept、0 broken。
- Docker build: provenance、frozen lock、source digest、torch compile、non-root probe pass。
- 問題cutoff 37337の固定fit診断: old image OOM、new image pass。
- Docker実データr3: Signal 8/8、Selection 8/8、exit 3、OOM false。
- Admission/BC/RL: fixed fail-closed orderingにより未実行。

## 非主張

- learned policyが利益を出したとは主張しない。BC/RLは実行していない。
- fast-onlyまたはslow-retentionのどちらかが採用可能とは主張しない。selected candidateはnull。
- 1分足が不要と恒久決定したとは主張しない。現時点のaggregate gross-negativeを救う手段ではないという判断。
- positive interim Signalや低turnoverを資産増加成功とは扱わない。最終Selection wealthは明確に1未満。
