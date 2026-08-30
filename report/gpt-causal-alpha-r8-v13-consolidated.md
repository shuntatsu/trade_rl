# GPT向け統合報告：Causal Alpha r8〜V13（V11実データ研究）

作成日: 2026-08-31 (JST)  
目的: Binance USDⓈ-M の各銘柄で long/short を銘柄ごとに独立して売買し、手数料・約定・リスク制約込みの after-cost 資産を最大化する。全銘柄を同時に同じ方向へ持つことや銘柄間の同時性は目的ではない。

## 結論

- r8〜r21 の正式 Selection と、V11 の独立4 study arm（L1/E1/C1/S1）を同じDB-backed実データ契約で調査した。
- V11 の目的は r21 の execution lifecycle を凍結したまま、V9 の損失原因を一つずつ切り分けることだった。V8 cash sanity、V9 exact control、各 treatment を同一runへ混ぜず、armごとに別output rootを使用した。
- L1、E1、C1 は Selection reject。S1は固定契約の実行可能性を検証してpreflight stopとなった。
- Selection 合格がない限り Admission、BC、RL、holdout の開封は行わない。したがって現時点で「学習済み戦略が勝てる」「production authorization済み」とは主張しない。
- V12 は V11 terminal evidence 後にのみ進む条件付き分岐。追加horizonの因果label materializationとartifact bindingが必要であり、4h/72h固定のV11成果物を流用してV12性能を主張しない。V13は15分戦略のSelection合格が先条件で、raw 1mデータの存在だけでは開始しない。

## 1. 固定した目的・評価契約

- DB-backed Binance source、凍結V4 context、15分decision cadence、報酬 `100 * net_log_return`。
- universal model（銘柄ID・銘柄別係数なし）。各銘柄の long/short はポジション状態を銘柄単位で管理し、銘柄間の同時方向は評価目的にしない。
- fast horizon 4h、slow horizon 72h、target magnitude 0.1、fee/impact/liquidity/risk/confirmation/no-tradeの契約は変更しない。
- Selection は9銘柄 × 8 economic scope、候補ごと72 leaf。各V11 armは `cash sanity + exact V9 control + treatment` の3候補、合計216 replay leaf。
- ゲートは既存V8 numerical gateをそのまま適用: symbol-balanced gross/net、minimum/median symbol wealth、positive scope fraction、CVaR、meaningful execution、hard-risk、reconciliation。
- `balanced net > 1` だけでは合格にしない。全銘柄で資産が増えることを必要条件とする。

## 2. r8〜r21の履歴

| 世代 | 内容 | 結果 |
|---|---|---|
| r8 | flatten-then-reset | reject: balanced net 0.994555、minimum 0.984710、positive scope 0.194444 |
| r10 | neutral-fast-expiry | reject: balanced net 0.994428、minimum 0.984927、positive scope 0.194444 |
| r11 | flatten-on-risk-breach | reject: balanced net 0.994475、minimum 0.984710、positive scope 0.194444 |
| r12 | fast-only ownership | reject: balanced net 0.992970、minimum 0.984927、positive scope 0.138889 |
| r13〜r19 | Signal/fallback、provenance、identity、bindingの修正過程 | 性能判定なし。失敗ログ・成果物は保存 |
| r20 | split Signal→Selectionを初めて完走 | Selection reject。hierarchical net 0.999443 |
| r21 | lifecycle/reduce-only/risk evidence hardening | Selection reject。r20と経済結果は同一、V10 lifecycle referenceとして凍結 |

r20/r21の詳細は旧handoffを重複させず、`report/gpt-v10-r8-r20-handoff-20260830.md` と `report/gpt-v10-r21-lifecycle-handoff-20260830.md` を参照する。r21 V9は balanced gross 1.018550、balanced net 1.013175 だが、minimum symbol net 0.960973、positive scope 0.375 のため普遍戦略として不合格だった。

## 3. r21を基準にした根因

r21で execution lifecycle の証跡は十分になった。hierarchical wave は target change 372、submitted 133、executed 186、closed trade 93、cost 1,316.24、balanced net 0.999443で、r20から経済結果は不変だった。target changeはnumeric requested targetの変化であり、submission/executionと同じ注文ファネルではない。内部target churnを利益改善指標にしない。

V11 D1は、r21 leafへ存在しないfast metadataを後付けで0埋めしない。r21と同一fit/forecast/runtimeからV9 targetを再生成し、V6 target digestがleafと一致した場合だけsignal配列をdecision indexでr21のauthoritative step economicsへ結合した。これにより、実現exposureに基づくentry/exitと、signalのneutralを分離した。

## 4. 実装したV11契約

追加した主なファイル:

- `trade_rl/learning/causal_alpha_v11.py`: immutable config、候補/arm、digest、S1 sizing feasibility。
- `trade_rl/learning/causal_alpha_v11_policy.py`: exact V9 control、L1/E1/C1/S1の独立compiler、trace policy。
- `trade_rl/learning/causal_alpha_v11_calibration.py`: pooled symbol-free long/short ridge calibration。label end < knowledge cutoffを検証。
- `trade_rl/learning/causal_alpha_v11_diagnostics.py`: D1のentry→neutral→exit、entry oracle、MAE/MFE、exposure効率。
- `trade_rl/workflows/universal_causal_alpha_v11_gates.py`: unchanged V8 gate adapter、arm identity、S1 stop。
- `trade_rl/workflows/universal_causal_alpha_v11_stage_entry.py` と `scripts/run_universal_causal_alpha_v11_research.py`: DB-backed、restart-safe、digest-bound runner。

主要コミット:

```text
0571b944 docs: design causal alpha v11 policy research
c9e17a98 feat: define causal alpha v11 policy contracts
ce48013b feat: add causal alpha v11 policy experiments
a5370b41 feat: add pooled v11 sign calibration
b86b58ed feat: add v11 trade diagnostics
f3bc4af7 feat: add v11 independent selection gates
329131d7 feat: add db-backed causal alpha v11 research
```

`ActionPathStepTrace` はdecision index、current/requested/projected/realized weight、risk/liquidity cap、fast mean/std/qualified direction、edge margin、after-cost objective、slow mean/std/direction、position origin、hierarchy reason、gross/net/cost/turnover、submitted/suppressed/executedをstep単位で保存する。`ActionPathStepEconomics.realized_weights` とstep traceはreconcile時に一致を要求し、D1のexposure-hour/entry/exit attributionはrequested targetではなくrealized weightを使う。V10/r21 schemaは変更しない。

## 5. D1（全V11 arm共通のbehavior-neutral診断）

72 scope、325 trade、325 entry、neutral observed trade 236、right-censored 32。固定4h entry edgeは `direction * realized_4h_label - 2 * one_way_cost`。

| 指標 | 値 |
|---|---:|
| mean entry edge | -0.0007943378 |
| mean entry→first neutral net log | -0.000092253 |
| mean first neutral→exit net log | +0.000454720 |

entry時点で既に平均edgeが負で、neutral後だけが主損失という仮説は支持されない。従ってL1単独で解決する根拠は弱く、E1/C1のentry quality/方向校正が必要という順序になった。D1のper-symbol/per-scope/per-trade JSONは各arm `diagnostics/scopes/` と `diagnostics/evidence.json` に保存した。

根因分類は次のとおりである。

- **entry負 / sign-calibration候補**: mean entry edgeが`-0.0007943378`。entry直後からの質を先に直すべきで、C1はpooled long/short校正を試したがgateを改善しなかった。
- **neutral expiry**: neutral→exitの平均net logは`+0.000454720`。neutral後保有が一貫した損失源という証拠はなく、L1は`neutral_expiry_2`で悪化した。
- **risk projection / post-projection flat**: r21のhard-risk violationは0、unexplained exit/rejectionも0。risk-cap/reduce-onlyはlifecycle evidenceとして保存されたが、target change増加はsubmission/executionに接続しない内部churnである。
- **inherited losses**: 全r21/V11 control leafで初期current/realized weightはflat。初期保有を原因とする損失ではない。
- **slow ownership**: r8〜r12でslow neutral、boundary reset、risk flatten、fast-only ownershipを個別に試してもuniversal gateを通らず、hierarchical waveはnegative/reference controlとして凍結した。

## 6. V11 arm別実データ結果

全armで `cash sanity` は意味あるexecutionなし、`V9 control` は同じr21挙動、treatmentのみ差分。以下はafter-cost Selectionの主要値。`eligible=false` は数値ゲート不合格を意味する。

| study arm / treatment | balanced gross | balanced net | minimum net | median net | positive scope | meaningful scopes | closed trades | cost | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `neutral_expiry_2` (L1) | 0.997342 | 0.987997 | 0.933031 | 0.991857 | 0.305556 | 53 | 584 | 8,380.97 | reject |
| `after_cost_entry` (E1) | 1.015764 | 1.010636 | 0.959900 | 1.010841 | 0.416667 | 50 | 282 | 4,177.38 | reject |
| `sign_calibrated_entry` (C1) | 0.988745 | 0.986894 | 0.963623 | 0.990113 | 0.111111 | 23 | 99 | 1,452.80 | reject |
| `calibrated_edge_sizing` (S1) | — | — | — | — | — | — | — | — | preflight stopped |

L1/E1/C1のcash/control値、artifact digestは各output rootで固定されている。E1はV9よりbalanced net、median、costを改善したが、minimum 0.959900とpositive scope 0.416667でgateを通らない。L1はentry-to-neutralの負を変えず全体を悪化させ、C1はpooled long/short校正を加えてもpositive scope 0.111111まで低下した。C1 terminal result digestは`1de25f0ed7e089458030e46314d23dda2f638c65950a8a4a7263282ef78e49b0`、Selection evidence digestは`61bd7f827c7e73c70d82e645977f06de00608931876b2e7b652f13d453cbf369`。

execution funnel（treatment）も保存した。target changeはpolicy/compilerのnumeric変化、submittedはenvironment submission、executedはfilled turnoverであり、順序どおりに単調減少する指標ではない。

| arm | CVaR10 | target changes | submitted | executed | closed trades | turnover p50/p95 | hard-risk / unexplained rejection |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | -0.017431 | 4,062 | 89,205 | 1,306 | 584 | 0.066095 / 0.117952 | 0 / 0 |
| E1 | -0.022233 | 2,259 | 159,234 | 702 | 282 | 0.023435 / 0.074782 | 0 / 0 |
| C1 | -0.015532 | 1,117 | 76,164 | 262 | 99 | 0 / 0.054825 | 0 / 0 |

L1/E1/C1はいずれもr21 target/step/lifecycle reconciliationを壊していない。C1は方向校正を加えたが、entry coverageとpositive scopeが低下し、改善候補としては不採用である。

## 7. trace/lifecycle監査

- L1/E1/C1各runは異なるoutput root、`study_arm`/config digestを持つ。r21 input、outer artifact、V6 target digest、V10 wrapper digestを毎scopeで照合。
- 既存r21 leafとのexact V9 controlはgross/net/cost/turnover、requested/projected/realized weight、lifecycle transitionを比較する。
- `ActionPathStepEconomics` と `ActionPathStepTrace` のrealized weight、step economics、digestを再照合する。欠落、schema mismatch、hard-risk、unexplained rejection、reconciliation差分はfail-closed。
- positive interim netをlearned-policy upliftと呼ばない。V8 cash fallback、V9 control、treatment、Admission、BC/RLは別の結論として報告する。

## 8. V12 horizon researchの扱い

V11全armがterminalになった後、V12条件は「V11 policy candidateが不合格ならH1/H2へ進む」として成立する。ただしV12は別familyであり、V11の4h/72h targetを流用した性能主張はしない。

事前登録する比較は、H1 `fast=4h/8h/16h/24h, slow=72h固定`、H2 `H1 winner固定, slow=72h/120h/168h`。horizon winnerはinner walk-forwardだけで決め、未観測outerを一度だけ開く。8h/16h/120h/168h labelは凍結15分価格からcausalにmaterializeし、label end、knowledge cutoff、source/target digestを保存する必要がある。

現行の凍結V4契約・V10 fitは4h/24h/72h horizonを前提としており、追加horizonのartifact-bound materializerとV12 runnerはまだ成果物として存在しない。24h labelはschema上存在するが、V9/V10 wave fitter・config・runnerがselectable horizonとして受け付けない。従ってこの世代ではV12 performance runを捏造せず、`blocked_preflight`（requested fast `4/8/16/24h`、slow `72/120/168h`、available `4/24/72h`、missing `8/16/120/168h`、`performance_run_attempted=false`、`selection_evidence_digest=null`、stop reason `v12_horizon_label_materialization_and_binding_required`）として扱う。これはSelection rejectではなく、因果labelを作らずに既存labelから補間しないための安全停止である。

## 9. V13 one-minute execution fidelity

PostgreSQL `public.rl_klines` にはBinance 1m raw rows（16 symbols、2023-07-06〜2026-07-05）が存在する。しかし maintained runtime/point-in-time artifact と execution contract は15m/1h/4h/1dを前提にしており、1mのpaired lifecycle bindingはまだない。V13は、まず15分戦略がSelectionを通過し、target streamを凍結した後にのみ開始する。現状はV11 candidateが未合格なのでV13は blocked。

将来のV13は、同一15m target pathを15m simulatorと1m simulatorへpaired入力し、fill timing/price、intrabar order、volume participation、partial fill、slippage、cost、rejection、hard-riskだけを比較する。signal、direction、entry/exit、target magnitude、confirmation、horizon選択を1m結果から再調整しない。

## 10. Docker/runtimeと成果物

- branch: `main`
- code commit/image revision: `329131d7a9b26b2e96f8e03879e7aa49fafceb3c`
- image: `trade-rl-causal-alpha-v11:329131d7-6726b3737df9`
- image manifest: `sha256:e3b18b2225e79e0b4f68651e8c63b65ae66a86f88118c75e961049cb731cbbed`
- source tree digest: `14036cb0a9e0b7ba3ce74cf9bf8cf8912e1021271c65bcf2c912d7b4ffd062e0`
- lock digest: `95dddd1ed146c4738004a0f3c97458737184cb5c03c730167af46f345e9c213b`
- runtime manifest: `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`
- V4 context manifest: `bc91783061182e41415d45a714049737ae16564a47d0e1ca14d004cc4c5c7357`
- volume/network: `trade-rl-training-data`, `trade_rl_default`

V11実データrunは`329131d7` imageで完了した。その後、既存V10 risk projectionのno-trade band境界に対する回帰修正を`dcf7a877`へ取り込み、49件の対象テストで検証した。この修正はV11 compiler/selection pathの挙動を変更しないため、既存V11 artifactのrun identityを差し替えていない。

V11 output roots:

```text
/workspace/var/runs/causal-alpha-v11-neutral-expiry-2-prod-20260830-r1
/workspace/var/runs/causal-alpha-v11-after-cost-entry-prod-20260831-r1
/workspace/var/runs/causal-alpha-v11-sign-calibrated-entry-prod-20260831-r1
/workspace/var/runs/causal-alpha-v11-calibrated-edge-sizing-prod-20260831-r1
```

L1/E1/C1はそれぞれ216 replay leafを生成済み。C1 containerは2026-08-30T20:12:17Z〜21:18:52Z、exit 3、OOM false。S1は72/72 diagnosticsを生成後、216 replay metricを作らずpreflight stopした（2026-08-30T21:19:42Z〜21:32:02Z、exit 4、OOM false）。S1 sizing evidenceは`generated_nonzero_count=75840`、`executable_nonzero_count=0`、`maximum_absolute_target=0.07354263`、`entry_threshold=0.1`、rejection `entry_threshold`、feasibility digest `858b5314737aa58184906caa6b408d49409eaa8a34282ef985f0045e1690cafb`。S1 terminal result digestは`03ba8d765eea766fc14c1ab39eff38b51da5a2860f48127e725961b1367b1f3a`、Selection evidence digestは`5bc45fbf00ab57c63254020ce4fd6d0fbae89e61612c7f436f1d7225de7b80c8`。各runの`selection/evidence.json`、`diagnostics/evidence.json`、`result.json`、terminal artifact digest、container exit/OOMを保存した。

| arm | selection evidence | diagnostics evidence | terminal result | status |
|---|---|---|---|---|
| L1 | `d3bb188dd413faaf5af548b929acf2f1affc74637742ec48605deccbef619e6c` | `1d088da7450b5d8d9de6fbeb3f217342755d0dae83e9b4c674c2c5cd2bffd456` | `0a706819cde88b25c83f201e76a9b984876673835d1cec40bfcc947fe69574f3` | selection_rejected |
| E1 | `08de171e1f261671b56e335b610917cdd6acbbbccb75b5982d7bfca684884a33` | `c05b9abee788a308b664e49d4d59940952f6abc0f1278c9671f76784018b0529` | `f6467ceb0f5e9aa6be76da1a1b7095f6b037f17af757219148b20da6dd2e7018` | selection_rejected |
| C1 | `61bd7f827c7e73c70d82e645977f06de00608931876b2e7b652f13d453cbf369` | `43faa84de310f19d14da1fa26add0f387a11bb5cad84155f2ebdf08a15862b26` | `1de25f0ed7e089458030e46314d23dda2f638c65950a8a4a7263282ef78e49b0` | selection_rejected |
| S1 | `5bc45fbf00ab57c63254020ce4fd6d0fbae89e61612c7f436f1d7225de7b80c8` | `e46ae25fff17cfe9ae6e399f23f87adadd3d5d0e0cb41e30878fd42474116aed` | `03ba8d765eea766fc14c1ab39eff38b51da5a2860f48127e725961b1367b1f3a` | preflight_stopped |

## 11. 検証と最終判断

V11 focused tests、r21 lifecycle/V10関連 testsは最終修正後 `49 passed`。Ruff check/format、`git diff --check`を通し、変更外の既存mypy warning（`trade_rl/telemetry/_indexed_storage.py:172` unreachable）以外の新規エラーはない。全体suiteの実行では4,494 passed / 44 skippedだったが、既存のpyproject v7 entry-point期待値と、別系統のsequence behavior-cloning gate support=0の2テストが残った。今回のV11コードおよびrisk projection修正の対象テストは合格しており、この2件をV11 Selection成功と混同しない。

最終的な判断は次の順序で分離する。

1. Signal: V9には一部predictive informationがあるが、universal robustnessは未成立。
2. Selection: V8/V9/V11 treatmentのいずれも現時点で銘柄横断gateを通過していない。
3. Admission: 未実施。holdout未開封。
4. BC/RL: 未実施。learned-policy upliftの証拠なし。
5. Production: authorizationなし。

従って次に根本から見るべき順は、D1でentry負が確認されたため、`after-cost entry quality → pooled long/short calibration → 実行可能性を壊さないsizing → V12 horizon materialization`。1分足はexecution fidelityのpaired比較に限定し、15分戦略を置き換える根拠にはまだ使わない。
