# GPT向け：Causal Alpha V10 現在の問題点と判断依頼

作成日: 2026-08-28 (JST)
最終更新: 2026-08-29 (JST, V10 r7 完走結果・実現weight traceを追記)

## 1. 目的と固定条件

目的は、Binance USDⓈ-M の複数銘柄で、各銘柄の long/short を独立に売買し、手数料・約定制約を含む after-cost の総資産を最大化すること。特定銘柄だけに合わせず、どの銘柄でも再現性のあるトレードを学習させる。

固定条件:

- 純粋な報酬は 100 * net_log_return。
- 銘柄IDを特徴量に入れず、全銘柄を同一の普遍モデルで扱う。
- Signal → Selection → Admission → BC/RL の順序を守る。
- Signal/Selection のゲート閾値、holdout 分離、コスト計算を緩和しない。
- 実データは PostgreSQL/DB-backed Binance ワークフローを使用する。
- 1分足はまだ導入していない。現在の15分足で戦略の有効性と約定可能性を先に検証している。

## 2. 実装と実行 identity

- 作業ブランチ: main
- worktree: C:\dev\trade_rl
- 最新コードコミット: dd190deb255e56d8917e9ac312dc1d446302b4e
- 今回の修正: V10 の3候補は候補ごとに fast/slow fit identity が異なるため、scope pairing の共通キーを各 replay の V6 calibration fit に正規化した。さらに ActionPathStepEconomics に realized weight を保存し、policy trace metadata の property/callable 両形式を読み取るようにした。候補固有の target/replay digest は保持し、回帰テストを追加。
- ソースツリーダイジェスト: 00c166e28b29410088de950caa46976a656aee7ab6a3285dad3fef9fd5a0fe84
- lockfile digest: 95dddd1ed146c4738004a0f3c97458737184cb5c03c730167af46f345e9c213b
- runtime manifest digest: 6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0
- 実行イメージ: trade-rl-causal-alpha-v10:dd190deb255e-6726b3737df9
- 実行イメージ manifest: sha256:1c18ddb03d2683b1177353d19d72a8e7e14dcc5d2b46eb99627fdf4adc1cb3ba
- run generation: causal-alpha-v10-prod-20260829-r7
- output root: /workspace/var/runs/causal-alpha-v10-prod-20260829-r7
- runtime DB volume: trade-rl-training-data:/workspace/var

### V10 設計

- fast horizon 4時間。fast fit は4週間、4時間非重複ラベル。
- slow horizon 72時間。slow fit は12週間、72時間非重複ラベル。
- Signal は slow qualified scope 72/72。
- Selection は v8_robust_control、v9_nonlinear_control、hierarchical_wave の3候補。
- hierarchical wave は slow の最後の qualified direction をラッチし、fast と一致して2回連続確認で entry。fast逆方向2回、slow逆方向2回、slow neutral 6回で exit。直接反転はしない。
- target magnitude は0.1。実行シミュレータは15分足、no-trade band は0.05。

## 3. V9 の既知の失敗（全216 Selection 完了済み）

V9 nonlinear control の正式な Selection は rejected。

    balanced gross wealth : 1.0129087626
    balanced net wealth   : 1.0064626469
    minimum symbol net    : 0.9586933631
    median symbol net     : 0.9981460217
    positive scope frac   : 0.4305555556
    CVaR10                : -0.0221627281
    turnover p95          : 0.0861466
    retention             : 0.9936360
    rejection reasons     : minimum_symbol_net_wealth,
                            median_symbol_net_wealth,
                            positive_net_scope_fraction

V9 attribution:

- long net log return -0.103633、short +0.174549。
- confidence q1 -0.050294、q4 +0.108271。
- liquidity q1/q2 が負、q3/q4 が正。
- volatility q1/q4 が負、q2/q3 が正。

単に取引頻度や時間足を増やしても、long 側の弱さ・低流動性・極端な volatility での損失は解決しない可能性が高い。

## 4. V10 r3 の中間実データ（正式判定前）

Signal は 72/72 で通過済み。Selection は途中状態で、スナップショット時点の leaf は28件。正式な Selection gate、Admission、BC/RL、holdout は未実施。

hierarchical_wave の episode 08:

| symbol | net wealth | executed | target changes | submitted | downstream suppression | flat fraction |
|---|---:|---:|---:|---:|---:|---:|
| APTUSDT | 1.000000000 | 0 | 3 | 208 | 208 | 0.928 |
| ARBUSDT | 1.000000000 | 0 | 15 | 512 | 512 | 0.822 |
| BCHUSDT | 1.000000000 | 0 | 0 | 0 | 0 | 1.000 |
| BNBUSDT | 1.000000000 | 0 | 0 | 0 | 0 | 1.000 |
| BTCUSDT | 0.999328161 | 2 | 2 | 130 | 128 | 0.956 |
| LINKUSDT | 1.014348395 | 9 | 1 | 495 | 486 | 0.828 |
| LTCUSDT | 1.000000000 | 0 | 0 | 0 | 0 | 1.000 |
| SOLUSDT | 1.000000000 | 0 | 0 | 0 | 0 | 1.000 |
| XRPUSDT | 1.027188324 | 7 | 3 | 1343 | 1336 | 0.533 |

同じ episode の V9:

    APT 1.024332153 / exec 16 / net +0.024040842
    ARB 1.004164049 / exec 11 / net +0.004155404
    BCH 0.998828765 / exec 10 / net -0.001171921
    BNB 0.970252807 / exec 26 / net -0.030198616
    BTC 1.014815032 / exec 11 / net +0.014706361
    LINK 1.041343700 / exec 17 / net +0.040511899
    LTC 0.974651629 / exec 19 / net -0.025675176
    SOL 1.034077830 / exec 19 / net +0.033510044
    XRP 1.009961100 / exec 15 / net +0.009911815

中間データであり、候補選択の根拠にはしていない。

## 5. 現在の問題点・仮説

### A. target path と実行可能性の不整合

hierarchical target は liquidity/risk cap をそのまま target magnitude にしている。cap が simulator の no-trade band 0.05 以下になると、target は変化しても controller が注文を抑制する。その結果、submitted_change_count と downstream_no_trade_suppression_count が大量に積み上がる。

これは無駄トレードだけでなく、has_meaningful_execution=false になり、候補が全銘柄 gate を満たせない原因になり得る。一方、cap を無視して固定0.1を要求すると risk projection とコストを過大にする可能性がある。

判断依頼:

1. entry は cap > no-trade band の時だけ許可し、それ以下は flat を維持するべきか。
2. 保有中の cap 低下が band 未満の場合、target を小さくして抑制するか、前回 target を維持して risk layer に任せるか。
3. no-trade band を新しいハイパーパラメータにせず、既存 simulator contract と一致する固定 execution eligibility として扱うべきか。
4. suppression を診断として記録しつつ、economic gate の canonical metric の意味を変えずに済むか。

### B. 選択性が強すぎる可能性

slow/fast 一致、2回連続確認、liquidity median 以上、volatility q25-q75 を同時要求している。Signal は通っているが、episode によっては entry が0になる。4〜8時間の波では、72時間 ownership が正しくても確認が遅すぎる可能性がある。

confirmation を1回へ緩める等は holdout tuning になり得るため、変更するなら事前登録した別候補として Selection を最初からやり直す必要がある。

### C. long/short と regime の非対称性

V9 は short が利益源で long が損失源だった。V10 も方向別、銘柄別、liquidity/volatility 条件別に after-cost を分解する必要がある。全体 net wealth が正だけでは採用せず、minimum symbol、median、positive scope、CVaR、turnover、retention を同時に確認する。

### D. 1分足は現時点の主問題ではない

15分足で target changes と downstream suppression の不整合が既に見えている。1分足は約定遅延・intrabar順序・volume participation の証拠が得られた場合だけ検討する。1分足にしても cap/実行帯や long 側予測の問題は解決しない。

### E. 再開時の writer lock

途中でコンテナを停止すると .causal-alpha-v10.lock が残り、再開が V4 output root already has an active or unrecovered writer lock で fail-closed になった。実行中コンテナがないことを確認後、対象 run root の lock だけを回収する必要がある。immutable leaf は回収後に再利用できる。

## 6. GPTに判断してほしいこと

1. risk cap を守りつつ no-trade suppression を減らす、事前登録可能な target compiler の設計。
2. slow 72h / fast 4h / 2回 confirmation が4h〜7日の波に妥当か。変更する場合の比較候補。
3. 銘柄IDなしで、long/short 独立売買の calibration・position sizing・loss containment を改善する方法。
4. 15分足から1分足へ進むべき実証条件。単なる取引数増加を改善と見なさない評価方法。
5. V10 Selection reject 後に優先すべき根本候補（feature、label horizon、entry/exit ownership、execution gate）。
6. Selection pass 後にのみ Admission/BC/RL を開く段階ゲートを維持した改善手順。

## 7. 再現用の重要パス

- V10 Signal (r5): /workspace/var/runs/causal-alpha-v10-prod-20260829-r5/signal/evidence.json
- V10 Selection evidence (r5): /workspace/var/runs/causal-alpha-v10-prod-20260829-r5/selection/evidence.json
- V10 Selection leaf (r5): /workspace/var/runs/causal-alpha-v10-prod-20260829-r5/selection/replays/
- V9 formal run: /workspace/var/runs/causal-alpha-v9-prod-20260828-r3
- V10 spec: docs/implementation-plans/specs/2026-08-28-causal-alpha-v10-hierarchical-wave-design.md
- V10 plan: docs/superpowers/plans/2026-08-28-causal-alpha-v10-hierarchical-wave.md

確認コマンド:

    docker logs --tail 20 trade-rl-causal-alpha-v10-prod-20260829-r5
    docker run --rm --mount type=volume,source=trade-rl-training-data,target=/workspace/var trade-rl-causal-alpha-v10:d1355cdf59ae-6726b3737df9 python -c "from pathlib import Path; print(len(list(Path('/workspace/var/runs/causal-alpha-v10-prod-20260829-r5/selection/replays').rglob('*.json'))))"

## 8. 現時点の結論

V10 r3 の中間値からは結論しない。修正後 r5 の正式 Selection は Signal を通過したが、3候補すべて数値ゲート不通過で rejected となった。Admission、BC/RL、holdout は未実施であり、「学習成功」「1分足が必要」とは結論しない。

## 9. V10 r5 修正後の正式結果

### 実行 identity と stage

- Signal evidence: `/workspace/var/runs/causal-alpha-v10-prod-20260829-r5/signal/evidence.json`
- Selection evidence: `/workspace/var/runs/causal-alpha-v10-prod-20260829-r5/selection/evidence.json`
- Terminal result: `/workspace/var/runs/causal-alpha-v10-prod-20260829-r5/result.json`
- run manifest digest: `41105fbe3df38c196768bbdb85a2af366239609a5f92de3e710af8bd20957ed9`
- V4 context manifest digest: `bc91783061182e41415d45a714049737ae16564a47d0e1ca14d004cc4c5c7357`
- config digest: `e838186df1cd268f650a539c2b6412f0331ee59784c73747ff615e1871461a46`
- Signal artifact digest: `02e4bc43bb474cff607e133cf1a7c076bc695f5ce46ba69da5d68ad82546cc4b`
- Selection evidence digest: `b67d4f96e08af41624d2ca89ac63d58480f0b270e499ac6d940d5a4d2e24fecb`
- Terminal result artifact digest: `92ce1022a3f9ad6e476e9a86b0e6aece569cdd1d2472cb6a3f0506aee1efafba`

Signal は `72/72` qualified slow scope で passed。Selection は `216` replay leaf、`paired_scope_count=72` まで生成され、候補 fit identity 差による従来の `scope_pairing` 誤判定は解消した。最終 rejection reason は `no_eligible_candidate` のみであり、ゲート閾値を緩和していない。

### 候補別 after-cost economics

| candidate | balanced gross | balanced net | minimum symbol net | median symbol net | positive scope | CVaR10 | meaningful scopes | executed | target changes | submitted | cost | closed trades | rejection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v8 robust control | 0.996468 | 0.995010 | 0.986459 | 0.995424 | 0.152778 | -0.004985 | 36 | 133 | 77 | 54,831 | 1,308.37 | 0 | symbol gross/net, min, median, positive |
| v9 nonlinear control | 1.012909 | 1.006463 | 0.958693 | 0.998146 | 0.430556 | -0.022163 | 62 | 840 | 2,656 | 176,235 | 5,299.14 | 198 | min, median, positive |
| hierarchical wave | 0.996954 | 0.994428 | 0.984927 | 0.996621 | 0.194444 | -0.005604 | 51 | 283 | 101,565 | 585 | 2,245.31 | 53 | symbol gross/net, min, median, positive |

各候補とも `hard_risk_violation_count=0`、`unexplained_execution_rejection_count=0`。V9 の利益は銘柄普遍ではなく、long net log return `-0.103633` に対して short `+0.174549` と強く非対称で、BNBUSDT の net wealth は `0.958693`。hierarchical wave も APTUSDT `0.985259`、ARBUSDT `0.984927` が弱く、全銘柄の資産最大化条件を満たさない。

hierarchical wave の attribution では、long `-0.028970`、short `-0.016585` と両方向が負。slow state `mixed` は net `-0.038908`、`risk_projection` transition は net `-0.044106`、volatility q4 は `-0.025047`。一方、V9 は confidence q4 `+0.108271` に対し q1 `-0.050294` で、confidence と方向の calibration/損失制御が根本課題である。

### 次に見るべき根本課題

1. `risk_projection` と mixed slow-state での target ownership を事前登録した別候補として修正し、同じ Signal→Selection gate を再実行する。
2. V9 の long/short 非対称、低流動性、high-volatility losses を calibration と position sizing の問題として切り分ける。銘柄IDの追加や symbol exclusion はしない。
3. 1分足はまだ導入しない。15分足で execution/risk projection の損失が残っているため、まず target compiler と loss containment を検証する。
4. Selection pass が得られるまで Admission/BC/RL を開始しない。

## 10. V10 r7（最新コードでの正式再開結果）

### 実行結果

r6 は、`last_step_trace_metadata` が property であるケースを evaluator が callable と誤認したため、診断用の途中実行として破棄した。property/callable の両方を読む修正を入れ、同じコード・同じ DB-backed runtime で r7 を最初から再実行した。以下を正式な最新結果とする。

- Signal evidence: `/workspace/var/runs/causal-alpha-v10-prod-20260829-r7/signal/evidence.json`
- Selection evidence: `/workspace/var/runs/causal-alpha-v10-prod-20260829-r7/selection/evidence.json`
- Terminal result: `/workspace/var/runs/causal-alpha-v10-prod-20260829-r7/result.json`
- replay leaves: `/workspace/var/runs/causal-alpha-v10-prod-20260829-r7/selection/replays/`（216件）
- Signal artifact digest: `04f1038ee5c566f3b1f45dbab10deafcb6f10aa248381a842ce186eb4dbe9a2a`
- Signal evidence digest: `2f8334385ddd335265956b0490417424602c8318b0af268eb2e38e6d776f2474`
- Selection artifact digest: `4e6532cf2d5afbeb4f31fb7aeaaa78bb7235dd506b8b5477026208c18f695871`
- Selection evidence/final digest: `ab22b091c4b7bb9acadcc9d0d96b1051f5ee9c7ab3b450e7e8b01ef8512e128c`
- Terminal result artifact digest: `7e9c048507eb6e103748e99dfb87551f3c2c4aa581602060dfa1f3f8accfc109`
- run manifest digest: `f3837b2c3e818deb76bcf9289cc0173c2155c6da13fb05021c54e7f557c55983`
- V4 context manifest digest: `bc91783061182e41415d45a714049737ae16564a47d0e1ca14d004cc4c5c7357`

Signal は `72/72` qualified slow scope。Selection は `paired_scope_count=72`、216 replay leaf を完走したが、`passed=false`、`selected_candidate=null`、`promotion_eligible=false`、terminal status は `selection_rejected`。したがって Admission/BC/RL は開始していない。

| candidate | eligible | meaningful scopes | minimum net wealth | median net wealth | positive scope | retention | rejection reasons |
|---|---:|---:|---:|---:|---:|---:|---|
| v8_robust_control | no | 36 | 0.986459 | 0.995424 | 0.152778 | 0.998537 | symbol gross/net, minimum, median, positive |
| v9_nonlinear_control | no | 62 | 0.958693 | 0.998146 | 0.430556 | 0.993636 | minimum, median, positive |
| hierarchical_wave | no | 51 | 0.984927 | 0.996621 | 0.194444 | 0.997467 | symbol gross/net, minimum, median, positive |

### 必須 action trace と attribution の検証

action は変更していない。全216 leafで `action_path_step_trace_v1` を保存し、各 step に次を保持する。

`decision index`, `current weight before action`, `requested target`, `PreTrade後のprojected target`, `execution後のrealized weight`, `active risk cap`, `active liquidity cap`, `fast head mean/std/qualified direction`, `abs(mean)-std-edge_margin`, `after-cost entry objective`, `slow head mean/std/direction`, `position origin (inherited/native entry/flat)`, `hierarchy reason`, `gross return`, `net return`, `cost`, `turnover`, `submitted`, `suppressed`, `executed`。

`ActionPathStepEconomics` は `causal_alpha_v7_step_economics_v2` として `realized_weights` を保存し、V7/V8 attribution の exposure 分類は requested target ではなく、その interval の simulator-authoritative な realized exposure を使う。再検証結果は次の通り。

- trace present/schema/decision alignment/realized weight shape: `216/216`
- step economics digest 再計算一致: `216/216`
- trace gross/net log と attribution 一致: `216/216`
- trace gross/net log と v6 gross/net wealth 一致: `216/216`
- trace の総 decision 数: `207,360`（72 leaf × 2,880 step、candidateごと）

### hierarchical_wave の実現エクスポージャ診断

以下は72 leafを合算した trace 分類で、return は各 step の `log1p` を合算してから `expm1` した値。従って Selection の symbol-balanced wealth そのものではなく、原因切り分け用の累積値である。

| 分類（realized exposure/reason） | steps | gross return | net return | cost | 判断 |
|---|---:|---:|---:|---:|---|
| realized long | 67,018 | -2.006% | -2.994% | 998.40 | long側は保有区間を含めて負 |
| realized short | 36,713 | -0.720% | -1.617% | 885.81 | short側も after-cost で負 |
| realized flat | 103,629 | +0.003% | -0.358% | 361.10 | flat区間にも execution cost が残る |
| neutral signal hold（realized non-flat かつ fast qualified=0） | 47,372 | -1.758% | -2.436% | 682.23 | signal=0後の hold が明確に負 |
| any projected target != requested target | 23,474 | -1.910% | -2.983% | 1,068.55 | projection区間は負だが、保護効果の反実仮想ではない |
| risk_cap_projection | 86 | -2.665% | -3.687% | 1,049.34 | 局所flatならnet logを0にできる区間 |
| risk_cap_flatten | 76 | -0.317% | -0.652% | 337.82 | 局所flat候補を登録すべき区間 |
| inherited origin | 52,669 | -3.314% | -4.608% | 1,310.32 | 最も強い ownership/episode boundary 問題の証拠 |
| native_entry origin | 132 | +0.631% | +0.162% | 467.20 | entry直後の小標本は僅かに正、結論には不足 |

hierarchical の `entry` reason は66回だが、realized exposure が long になったのは11 step、short は4 step、flat のままが51 step。entry試行の大半が実約定ポジションになっておらず、entry校正だけでなく execution eligibility/no-trade band との不整合がある。

slow qualified direction は long 29,967 step（gross -0.396%、net -0.605%）、short 37,104 step（gross +0.161%、net -0.185%）。72h slow の方向自体も after-cost で一貫して正ではない。

trace reason の大半は `cadence_hold=193,989`。submitted/suppressed/executed は `585/461/283`。無駄な注文を減らす必要はあるが、取引数を増やすだけでは Selection gate を満たさない。

### 根本原因の判定

| 観測 | r7で確認できたこと | 暫定判断 |
|---|---|---|
| longの新規entry直後から負 | realized long entry は11 step、netはほぼゼロ。long全保有はnet -2.994% | entry直後校正の単独結論は保留。long保有/exitを分離する候補が必要 |
| entry直後は正、signal=0後のholdが負 | neutral signal hold net -2.436% | neutral expiry/exit不足が有力 |
| risk projection前に損失、projection後に改善 | projection区間は net -2.983%、risk_cap_projection は -3.687% | 現runだけでは保護機構の改善とは言えない |
| projection後も即flatより悪い | flat-on-breach反実仮想を同時には計算していない | 事前登録した flat-on-breach candidate が必要 |
| inheritedだけ大幅に負 | inherited net -4.608% | episode boundary ownership が第一候補 |
| slow qualified方向の成績 | long net -0.605%、short net -0.185% | 72h slowを単独のownership sourceにしない比較候補が必要 |

次の比較は、同じ Signal→Selection gate を維持したまま、事前登録した別候補として行う。局所flat（そのstepのrealized exposureとcostを0に置く）では、risk_cap_projection の net-log 改善余地は `+0.03756`、risk_cap_flatten は `+0.00654`、inherited は `+0.04718`、neutral signal hold は `+0.02467` だった。ただしこれは将来の状態遷移を再計算しない局所反実仮想であり、採用判定には flat-on-breach の完全再 replay が必要である。優先順位は (1) inherited を flat/reset として扱う boundary ownership、(2) neutral signal の expiry/flat化、(3) risk breach の flat-on-breach counterfactual、(4) slow 72h を regime filter に限定した短い ownership。1分足導入は、この比較で15分足の execution/risk 問題を解消した後に、約定遅延・intrabar順序・volume participation の実証がある場合だけ検討する。

### r7再現確認

```text
docker run --rm --mount type=volume,source=trade-rl-training-data,target=/workspace/var \
  trade-rl-causal-alpha-v10:dd190deb255e-6726b3737df9 \
  python -c "from pathlib import Path; print(len(list(Path('/workspace/var/runs/causal-alpha-v10-prod-20260829-r7/selection/replays').rglob('*.json'))))"
```

期待値は `216`。実行コードは commit `dd190deb255e56d8917e9ac312dc1d446302b4e`、image manifest `sha256:1c18ddb03d2683b1177353d19d72a8e7e14dcc5d2b46eb99627fdf4adc1cb3ba` に固定されている。

