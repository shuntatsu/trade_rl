# GPT向け追加報告：Causal Alpha V10 r21 execution lifecycle 再評価

作成日: 2026-08-30 (JST)

関連する全体報告: `report/gpt-v10-r8-r20-handoff-20260830.md`

## 1. 結論

r20 後に execution lifecycle と reduce-only 契約を明示する修正を取り込み、DB-backed Binance 実データで r21 を最初から再実行した。Signal 72/72、Selection 216/216 は正常完走したが、全候補が数値ゲートを満たさず `selection_rejected` となった。Admission、BC、RL は開始していない。

r21 の after-cost economics は r20 と同一であり、収益性は改善していない。唯一の主要な数値差は hierarchical wave の `total_target_change_count` が 292 から 372 に増えたことだが、submitted 133、executed 186、closed trades 93、cost 1,316.24、balanced net 0.999443 はすべて不変だった。

従って今回の修正は、曖昧だった reduce-only intent、risk projection、execution lifecycle を正しく証跡化したが、資産最大化戦略の改善にはなっていない。次は lifecycle の追加修正ではなく、signal、entry quality、方向別 calibration、position sizing、loss containment を含む戦略側の根本再設計が必要である。

## 2. r21 の変更内容

ユーザー更新で次を導入した。

- reduce-only intent を signal delay、PreTrade、risk、execution まで明示的に伝播。
- 最終 risk projection から authoritative hard-risk evidence を保存。
- `ActionPathLifecycleTrace` を全 replay leaf に永続化。
- V10 replay leaf schema を v2 から `causal_alpha_v10_replay_leaf_v3` に更新。
- hierarchical policy の risk-cap reduction を、無条件 flatten ではなく reduce-only intent として扱う。
- V8/V9 control、4時間 fast、72時間 slow、報酬、手数料、ゲート、銘柄、15分足は変更しない。

旧 r20 の v2 leaf は再利用せず、r21 の216葉をすべて新規生成した。

## 3. 実行 identity

- branch: `main`
- user update head: `de9db08614dc2f131d3018bd292e07fde8332bad`
- validation fix commit / image revision: `3254e0d8eb93d6d87ea35a01dd6d44b23b188a51`
- image: `trade-rl-causal-alpha-v10:3254e0d8-6726b3737df9`
- image manifest: `sha256:c2c8e4792ecfcb588ce56356b51634f23ac1ef790d6eccfe14adb13b067d6afc`
- source tree digest: `b60d4dac6d6763a2f1b11065ad731252499cdb154ca7798a11259f357d09aa25`
- lockfile digest: `95dddd1ed146c4738004a0f3c97458737184cb5c03c730167af46f345e9c213b`
- runtime manifest digest: `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`
- V4 context manifest digest: `bc91783061182e41415d45a714049737ae16564a47d0e1ca14d004cc4c5c7357`
- output root: `/workspace/var/runs/causal-alpha-v10-prod-20260830-r21`
- container: `trade-rl-causal-alpha-v10-prod-20260830-r21`
- boundary mode: `flat_start_activation`
- started: `2026-08-30T05:14:14.447415865Z`
- finished: `2026-08-30T06:24:39.772420696Z`
- container result: exit code 3、OOM false

Artifact binding:

- Signal run manifest: `8f9154b1fac10e11d3641d583f74ff2d0a764c6a88f59770dcfc8aef721157b4`
- Selection run manifest: `a1ad424c5853de9b6395a022393ee0d5ce7fda145c532b3036350230af6b1e29`
- dual-run binding digest: `48b00fa89df9845eba4ac185a53d2ebad1aca8be4fc6e6622b75ea111e6890be`
- Signal evidence digest: `40eb827e0bf9f793c87e733660b189396af4399ab4109d7d49773f19dcf0f197`
- Selection evidence digest: `5643532480c36c1c1913d63c0e60c9e8150b073ed265e45059076c2f0ee31393`
- terminal result artifact digest: `c20417f38d069d220e0a9c944186c11bcf3d212f5da16d6c883f55421da84cc8`

## 4. 候補別結果

| candidate | balanced gross | balanced net | minimum net | median net | positive scope | CVaR10 | meaningful scopes | target changes | submitted | executed | closed trades | cost | rejection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v8 robust control | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0 | 0 | 0.00 | `symbol_balanced_*`, `positive_net_scope_fraction`, `no_meaningful_execution` |
| v9 nonlinear control | 1.018550 | 1.013175 | 0.960973 | 1.004164 | 0.375000 | -0.022370 | 50 | 2,568 | 166,345 | 736 | 292 | 4,372.66 | `minimum_symbol_net_wealth`, `positive_net_scope_fraction` |
| hierarchical wave | 1.000906 | 0.999443 | 0.997673 | 0.999449 | 0.194444 | -0.000951 | 33 | 372 | 133 | 186 | 93 | 1,316.24 | `symbol_balanced_net_wealth`, `minimum_symbol_net_wealth`, `median_symbol_net_wealth`, `positive_net_scope_fraction` |

全候補で `hard_risk_violation_count=0`、`unexplained_execution_rejection_count=0`。選択結果は `selected_candidate=null`、`promotion_eligible=false`、rejection は `no_eligible_candidate`。

V9 は全体 balanced net が1を超えるが、最低銘柄 wealth 0.960973、positive scope 0.375 のため「どの銘柄でも勝てる」普遍戦略ではない。hierarchical wave は downside を V9 より狭く抑える一方、after-cost の全体資産も中央値も1未満である。

## 5. r20 と r21 の差

| 項目 | r20 | r21 | 解釈 |
|---|---:|---:|---|
| replay leaf schema | v2 | v3 | lifecycle trace を追加 |
| hierarchical target changes | 292 | 372 | risk-cap/reduce-only の意図変化をより多く表現 |
| hierarchical submitted | 133 | 133 | execution へ届く注文意図は増えていない |
| hierarchical executed | 186 | 186 | 実現取引は増えていない |
| hierarchical closed trades | 93 | 93 | entry/exit 数は不変 |
| hierarchical balanced net | 0.999443 | 0.999443 | after-cost 資産は不変 |
| hierarchical total cost | 1,316.24 | 1,316.24 | コストも不変 |
| V9 economics | r20値 | 完全一致 | control 挙動は維持 |
| V8 economics | r20値 | 完全一致 | control 挙動は維持 |

target change の増加が submission、execution、exposure、wealth に接続していない。これは「無駄なポジション変動で損失が増えた」というより、execution 前に抑制または同値化される内部 target churn が増えた状態である。収益性の改善信号として扱ってはいけない。

## 6. 全216 leaf の監査結果

- leaf: 216（各候補72）
- replay parse: 216/216
- outer artifact digest: 216/216
- step economics digest: 216/216
- trace gross/net log-return → attribution: 216/216、最大絶対誤差 0
- attribution → V6 metric: 216/216、最大絶対誤差 `9.08e-15`
- execution cost reconciliation: 216/216
- decision count: 622,080
- initial `current_weights` flat: 216/216
- initial `realized_weights` flat: 216/216
- hard-risk evidence available: 216/216
- hard-risk violation leaf: 0
- hierarchical unexplained exit leaf: 0
- Signal、Selection、binding、result の outer digest: すべて一致
- Selection payload digest: 一致
- terminal result → Selection evidence binding: 一致

hierarchical lifecycle の207,360 decisions:

```text
transition:
  flat  207,174
  entry      93
  exit       93

flatten initiator:
  not_applicable             207,267
  execution_intent_flatten        93
  unexplained                       0

risk evidence:
  reduce_only + reduce_only_satisfied  40
  reversal_hysteresis                 93
  hard-risk violation                  0
```

今回追加した execution lifecycle は監査可能で、exit 93件はすべて execution intent に帰属した。しかし経済結果は不変である。

## 7. コード検証

- lifecycle/reduce-only/V8/V10 関連テスト: 47 passed
- Ruff check / format check / `git diff --check`: pass
- 今回追加コードで検出した mypy 2件は修正済み。
- mypy 全体には変更外の既存 `trade_rl/telemetry/_indexed_storage.py:172` unreachable 警告が1件残る。

## 8. GPT に判断してほしいこと

1. target changes だけ増え、submitted/executed/economics が不変な場合、どの compiler 境界を計測・簡素化すべきか。
2. hierarchical wave の exposure が93 entryしかない一方、それを単に増やすと損失とコストを増やす恐れがある。entry quality と exposure efficiency をどう事前登録すべきか。
3. V9 の全体利益を保ちながら、minimum wealth 0.960973 と positive scope 0.375 を改善する universal な方向別 calibration / sizing / loss containment をどう設計すべきか。
4. fast 4時間、slow 72時間を固定したまま改善する候補と、8時間〜7日の horizon grid を試す候補をどう分離すべきか。
5. 1分足導入はまだ必要条件が観測されていない。15分足で signal/selection を改善する実験と、execution fidelity 用1分足の比較をどう切り分けるべきか。
6. Selection 不合格を維持したまま、次の最小実験を signal redesign、position sizing、loss containment のどれから始めるべきか。

## 9. 再現パス

- r21 Signal: `/workspace/var/runs/causal-alpha-v10-prod-20260830-r21/signal/evidence.json`
- r21 Selection: `/workspace/var/runs/causal-alpha-v10-prod-20260830-r21/selection/evidence.json`
- r21 result: `/workspace/var/runs/causal-alpha-v10-prod-20260830-r21/result.json`
- r21 replay leaves: `/workspace/var/runs/causal-alpha-v10-prod-20260830-r21/selection/replays/`
- runtime volume: `trade-rl-training-data:/workspace/var`

確認例:

```powershell
docker inspect trade-rl-causal-alpha-v10-prod-20260830-r21 --format 'status={{.State.Status}} exit={{.State.ExitCode}} OOM={{.State.OOMKilled}}'
docker run --rm --mount type=volume,source=trade-rl-training-data,target=/workspace/var trade-rl-causal-alpha-v10:3254e0d8-6726b3737df9 python -c "from pathlib import Path; print(len(list(Path('/workspace/var/runs/causal-alpha-v10-prod-20260830-r21/selection/replays').rglob('*.json'))))"
```
