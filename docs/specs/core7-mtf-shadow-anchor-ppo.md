# Core7 Multi-Timeframe Shadow-Anchor PPO & Non-RL Regime-Gate Benchmark Specification

Status: Active

## 1. 目的と位置づけ（Phase 0: Software Validity Only）

本仕様書は、[Issue #810](https://github.com/shuntatsu/trade_rl/issues/810) 配下の 実効 4 時間軸（4h/1d/3d/1w）＋ 4h OHLC 由来の 15m/1h 近似入力・11 銘柄・22 レッグ（`:spot` / `:perp` 交互配列）Shadow-Anchor PPO 統合アーキテクチャおよび固定因果 Non-RL レジームゲート対照ベンチマーク（`ANC_REGIME_GATE_NONRL`）の**Phase 0（ソフトウェア整合性・合成オラクル・境界契約のみ）**を定義する。

- 現在の証拠水準: `software_validity`（`edge_claim = false`, `production_eligible = false`, `live_trading_authorized = false`）。
- 次の許可行動クラス: `synthetic_verification_only`（合成入力と不変境界テストのみ。実市場データの読込・学習・リプレイは禁止）。
- 実市場データを用いる **Phase 1（Tier D 既存データ・正準エンジン整合性比較）** は、Phase 0 実装コミットへピン留めされた独立 G0–G2 レビューおよび別途発行されるオーナー G4 認可 URL（`https://github.com/shuntatsu/trade_rl/(issues|pull)/<id>#issuecomment-<id>`）なしには起動できない。
- いかなる優位性主張・実運用可否判定も Phase 1 からは導けず、別の事前登録済み前向き実験プロトコル（Phase 2）と G5 認可が必要となる。

詳細なパッケージ責務・依存方向は [architecture/package-boundaries.md](../architecture/package-boundaries.md#core7-mtf-shadow-anchor-ppo-boundary)、Research Assurance Gate の正本は [architecture/research-assurance.md](../architecture/research-assurance.md) を参照する。

## 2. 凍結プロトコル（`core7_mtf_shadow_anchor_ppo_v4`）

### 2.1 対象ユニバース（11 ペア / 22 交互レッグ）

`BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `SOLUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `AVAXUSDT`, `LINKUSDT`, `LTCUSDT`, `DOTUSDT` の各ペアについて `(sym:spot, sym:perp)` をこの順序で交互配置する（実データ研究時は完全 11 ペア必須、合成 G2 テストでは 2–11 ペアを許容）。

### 2.2 時間境界・データ整合性

- Pre-FS1 ハード境界: 全バーの `timestamp` および `available_at` は `2026-08-01T00:00:00`（UTC）**未満**でなければならない（`>= 2026-08-01T00:00:00` は fail-closed 拒否）。
- 時間軸入力: `MarketDataset` は 4h 足であり、実効時間軸は 4h/1d/3d/1w の 4 つである。15m・1h 入力は外部 `micro_15m`/`micro_1h` が供給されない限り同一 4h OHLC バーからの因果近似（proxy）であり、実 15m/1h データではない。外部 micro 配列を渡す場合は有限値・形状 `(T,P,3)`/`(T,P,2)` と、4h 行 timestamp 以前の `micro_available_at (T,P)` が必須で、欠落・遅延は拒否する。
- `SPOT_HALTS` ギャップ（例: Binance Spot 停止区間）は前方補間やゼロ埋めリターン結合を行わず非取引マスク（`spot_usable = False`, `carry_mask = False`）として遮断する。
- ウォークフォワード境界: `fit_stop_index <= start_index` を設定および重み成果物の両面で強制する。

### 2.3 比較アーム（5 アーム）

1. `CASH`: 常時ゼロウェイト対照。
2. `ANC_BIDIR_75_25`: 因果 6 時間軸 All-Weather 双方向 75/25 アンカー + 因果 `vol_cushion_tail`（Non-RL 固定基準）。
3. `ANC_REGIME_GATE_NONRL`: 6 次元因果機会状態（`xdisp_ema_30d`, `fr_lvl_14d`, `fr_pos_14d`, `anc_sr_20d`, `carry_sr_20d`, `anc_dd_30d`）から固定シグモイド／クリップ規則で算出する Non-RL レジームゲート `(g_alpha, g_carry)` を `ANC_BIDIR_75_25` へ適用する対照ベンチマーク。
4. `CORE7_SPARSE_PPO`: 41 次元因果状態 + 平滑化アクション 7 次元 + 口座/クッションスカラー 4 次元 = 52 次元観測を受ける純粋 NumPy `LinearClipMlpActorCritic` がソフト閾値（0.05）付き 7 因子（`H0, H2, H4, H5, M0, M1, M3`）残差を生成する主候補。
5. `ORTH6_CUSH_PPO`: 共線な `M0` を除外した 6 因子（`H0, H2, H4, H5, M1, M3`）残差（`a[:6]`）＋テールクッション微調整乗数（`a[6]` -> `[0.92, 1.06]`）を用いる直交副候補。

### 2.4 実行コスト・ストレスシナリオ・ハードリスク整合

共有ネイティブ実行エンジン `run_shared_portfolio_replay(..., settle_terminal_position=True, capture_ledger_evidence=True)` のみを使用し、第 2 のバックテスタや独自会計は持たない。

| Scenario | `fee_rate` | `spread_rate` | `impact_rate` | `signal_lag_bars` | `order_latency_bars` | `processing_bar_volume_capacity` |
|---|---:|---:|---:|---:|---:|---|
| `BASE` | `0.0004` | `0.0002` | `0.0002` | `0` | `0` | `False` |
| `LOW_COST` | `0.0002` | `0.0001` | `0.0001` | `0` | `0` | `False` |
| `HIGH_COST` | `0.0008` | `0.0004` | `0.0004` | `0` | `0` | `False` |
| `DELAY_1BAR` | `0.0004` | `0.0002` | `0.0002` | `1` | `0` | `False` |

- `DELAY_1BAR` 設計注記: 本リポジトリの共有約定エンジンでは `order_latency_bars` の `0` と `1` がいずれも次バー始値約定となり、`1` は単に `agent_stop_index` を 1 バー短縮するだけである。そのため `DELAY_1BAR` は `order_latency_bars=0` を維持しつつポリシー層 `ShadowAnchorMtfConfig(signal_lag_bars=1)` によって真正な 1 バー信号遅延を与える。
- 20% ハードドローダウン: `max_drawdown >= drawdown_stop(0.20)` のセルは `decision.py` で停止扱い（`STOP_FAIL_NO_RESCUE`）となり、`PASS_PARITY_ONLY` は発行されない。`PASS_PARITY_ONLY` は `scenarios == SCENARIOS` の完全集合でのみ定義され、部分集合は `INVALID`。
- 終端ダスト: セル要約は lean-core の terminal-dust 規則に従う（`dev_realdata_study.runner.terminal_dust` を再利用し、各セルの `ExecutionCostConfig` を渡す）。エンジン自身が終端クローズを `below_minimum_notional` で拒否したことだけが原因の残余はダストとして記録され、`terminal_flat` は false のまま `terminal_dust_only` が true となり、残余がすべてダストなら `terminal_position_not_flat` を付けず、ダストを mark から想定退出コストを引いた値で評価し、その退出コストを `total_return` から差し引く。他の理由の残余・評価不能なダストは従来どおり INVALID。これは今後の実行にのみ適用し、消費済みの v4 Phase 1 INVALID（Issue #810 comment 7023026438）や lowturn v1 INVALID（comment 7028195515）を再分類せず、いかなる経済実験も認可しない。
- 認可（fail-closed、`trade_rl` はネットワーク不使用）: `run_study` は凍結プロトコル digest と一致する `expected_protocol_digest` を必須とし、Pre-FS1 禁輸（`timestamp`・`available_at`・行整列した `features`/`global_features`/`funding_rate`/`funding_price_rate`、および外部 micro 配列の `micro_available_at`）を常に強制する（`enforce_pre_fs1=False` は拒否。引数は呼出互換のため残す。`run_causality_probe` に禁輸解除引数は無い）。G4 URL（`authorization_url`）は `trade_rl` 内では**形式検査のみ**（`authorization_check = url_format_only`）であり、コメントの実在・投稿者・本文は検証できない。証拠適格（`evidence_eligible`）には、tools 層（#1282 のアダプタ）が GitHub から取得して構築する `AuthorizationReceipt`（`authorization_url`・`comment_id`・`author_login`・`author_id`・`body_sha256`・`fetched_at`（tz 付き ISO-8601）・`quoted_protocol_digest`）を `authorization_receipt` として渡す必要がある。ランナーは receipt の形状、URL との一致、URL 内コメント ID との一致、引用 digest とプロトコル digest の一致のみを検査し（`authorization_check = receipt_shape_and_binding`、`authorization_authenticated = true`）、真正性自体は証明できない（呼出側証明）。receipt 無しの URL 実行は `scenarios == SCENARIOS` と完全 11 銘柄を要するが結果は `UNAUTHENTICATED_NOT_EVIDENCE`、URL 無し実行は `allow_synthetic_unauthenticated=True` と `protocol.mark_synthetic_dataset` の合成マーカー付き（identity payload なし）データセットに限られ結果は `SYNTHETIC_ONLY_NOT_EVIDENCE`。いずれも `INVALID` は `INVALID` のままで、算出ラベルは `underlying_outcome` に残り、パリティ／停止の証拠にならない。`decide()` の `evidence_eligible` は必須キーワード（既定なし）。合成マーカーは全 identity 配列（`mark_price`・`available_at`・`features`・`funding_price_rate` 等）に束縛されるが、誰でも付与できるため認可ではなく誤用防止の便宜に過ぎない。
- `processing_bar_volume_capacity=False`: 全シナリオで 1% 参加率上限を**先行完了バー `t - 1` の出来高**だけで判定し、約定バー自身の出来高ルックアヘッドを遮断する。
- ハードリスク整合: `PreTradeRiskConfig(max_gross=2.5, max_abs_weight=0.55, max_turnover=2.5, drawdown_start=0.10, drawdown_stop=0.20, emergency_turnover_override=True)` と `ExecutionCostConfig(max_leverage=2.5, maintenance_margin_rate=0.05, max_participation_rate=0.01, minimum_notional=10.0)` の整合を保つ。
- HOLD／Deadband とドローダウン保護: リバランス間隔外または L1 目標変化が `deadband=0.04` 未満のバーでは `hold_current_quantity_symbols` を返し不要なドリフト約定を防ぐ一方、現在または最大ドローダウンが `drawdown_start` に達した場合は HOLD を解除する。非リバランスバーでは直近リバランスの**リスク適用前**目標を再提案し、`PreTradeRisk` の `risk_scale` を一度だけ適用する（縮小済み現在ウェイトを再投入すると縮小が毎バー複利化しリバランスで跳ね戻るため）。リバランスバーの deadband もドローダウン中は解除する。ドローダウン中に `risk_scale < 1` の間は厳密数量 HOLD を使えない（リスクが保持ウェイトを再度縮小するため）。非リバランスバーで、`risk_scale` が前回判断から**わずかでも低下した場合は常に**意図目標へ再調整する（`risk_scale` 低下を deadband で吸収すると、露出がリスク意図より最大 0.04 L1 高いまま残るため）。`risk_scale` が不変または上昇し、かつリスク縮小後目標が現在ウェイトから deadband（L1 < 0.04）内であれば、リスクが現在ウェイトへ写像する逆像 `current / risk_scale` を提案して微小な再約定を抑え、乖離が deadband を超えた場合は意図目標へ再調整する。残余ターンオーバーは終値→次始値ギャップ由来の微小ドリフトで、**全モード（`ANC_BIDIR_75_25`・`ANC_REGIME_GATE_NONRL` の Non-RL 対照を含む）**でドローダウン中の非リバランスバーの一部は依然として再約定する（合成フィクスチャ seed 1 の実測: 非リバランス・ブリーチバー 45 本中 26 本が約定し 19 本はゼロ約定、非リバランス合計ターンオーバー約 0.56（うち 0.15 は `risk_scale` 低下バーの正当な縮小）、ブリーチ期間コストの約 5%、`risk_scale` 不変バーの 1 バー最大約 0.07。リプレイ試験の境界は 1 バー 0.10 以下・平均 0.02 以下の実測値であり実データでの保証ではない）。この再約定は凍結コストモデルで課金されるため結果をより保守的にするのみで、会計不具合ではなく、RL と Non-RL 対照に等しく掛かる回転効率の問題である。

## 3. 事前データ露出（Pre-FS1 Rounds 1–12）の開示

- Pre-FS1 開発窓（`2021-01-01` ～ `2026-07-31`）の 11 銘柄データは Round 1–12 の探索で既読（known-used Tier D データ）である。Round 1–11 では `DSR_null = 0.788 / 0.721 < 0.95` と多重検定の帰無水準を下回り、Round 12 のレジーム適応検証では `ANC_REGIME_GATE_NONRL` が低迷期のドローダウンを同等に抑えたことで `H13-A NOT SUPPORTED` / `H13-B NOT SUPPORTED` の帰無結果が得られている。
- そのため `ANC_REGIME_GATE_NONRL` を本仕様および `decision.py` 診断の正式対照ベンチマークとして組み込み、RL 各アームについて `ANC_BIDIR_75_25` と `ANC_REGIME_GATE_NONRL` 双方に対するリターン差・Sharpe 差・ドローダウン差・優位性判定フラグを強制出力する。
- 事前探索結果は研究上の参考背景に過ぎず、本プロトコルの `decision.py` 判定ラベルは構造・因果性・台帳整合・停止非救済のみを測る `PASS_PARITY_ONLY` / `STOP_FAIL_NO_RESCUE` / `INVALID` に限定される。

## 4. Phase 0 合成 G2 オラクル（FM1–FM3 + 独立 Decimal 台帳）

1. **FM1（ゼロ残差・HOLD・Non-RL 対照整合）**: `ShadowAnchorMlpWeights.zero_residual()` を与えた `core7_sparse_ppo` / `orth6_cush_ppo` が `anc_bidir_75_25` と目標重み・終端資産ともに `< 1e-12` で一致すること。非リバランスおよび deadband 内で不要約定がゼロであり、ドローダウン発動時に HOLD が解除され、リプレイ上で総ポジションが `risk_scale × 意図した総ポジション` に追従して複利縮小・跳ね戻りしないこと。
2. **FM2（因果性・`SPOT_HALTS`・Pre-FS1 境界）**: `k_cut` 以降の価格・出来高・Funding 摂動に対する `k_cut` 以前の全パック配列・ゲート・提案重み・HOLD タプルの差分が `== 0.0` であること。プローブは非自明な口座パス上で全決定バー（リバランス有無を問わない）とシナリオ遅延（`DELAY_1BAR`）ごとに実行し、`index + 1` 行を読む意図的リークポリシーを検出する。`SPOT_HALTS` が前方補間・NaN なくマスクされ、`>= 2026-08-01T00:00:00` および `fit_stop_index > start_index` が fail-closed に拒否されること。
3. **FM3（先行出来高ロック・リスク／レバレッジ整合・独立 Decimal 台帳）**: 1% 上限でクリップされたリバランス約定の実行バー出来高をショックしても `processing_bar_volume_capacity=False` では約定が不変であり（クリップが拘束することを断言）、`True` の陽性対照では変化すること、`synthesize_alternating_leg_weights` の直接テスト（スポット `>= 0`・`|w| <= 0.55`・総 `<= 2.5`）とリプレイ上の `minimum_notional=10` / 1% 参加率、`max_leverage=1.0` と `max_gross=2.5` の不整合が拒否されること、および本番会計を import しない独立 50 桁 `Decimal` 2 ペア 4 レッグ台帳が全バーの現金・純資産・コスト（手数料＋スプレッド＋`impact_rate * sqrt(participation)`）・Funding PnL・ターンオーバー・終端フラット性を `abs <= 2e-9` / `rel <= 2e-12` で再現すること。ランナーの `ledger_reconciled` は生約定から現金・数量・純資産・ターンオーバーを再構成して照合し、各項目の改ざんを検出すること。

## 4.1 既知の未検証事項

- `execution.py` の `order_minimum_notional` の reduce-only 分岐（`market_order_profile` と `reduce_only_exits` が必要）は Phase 0 の 4 シナリオでは到達不能のため未テスト。`symbol_fills.py` の `allocate_symbol_capacity` 呼出しの外側 `minimum_notional` はリクエスト個別値に常に上書きされる等価変異である。

## 5. Packet Pin

- Open Author Packet: `report/issue810-core7-mtf-ppo-research-assurance-v6.json`（`-v1.json`〜`-v5.json` は不変の履歴であり再ピンせず、旧判定も再分類しない）
- Pinned Implementation Head: `ed31fdd3c419db4cc49f74d035403bfcc89c0378`
- Re-pin reason: #1289 統合の v5 terminal-dust packet を不変の履歴として保持し、#1287 の `trade_rl/data/market.py` クロスアセット最新改訂値の由来情報保護を適用したソースを v6 に固定する。既存のプロトコルdigest・terminal-dust規則・費用・STOP・判定・実験認可は変更しない。v6 は **UNREVIEWED** であり、独立G0–G2と別途オーナーG4の認可なしに実市場 Phase 1、FS1、unused/final の読込、注文を許可しない。
- Pinned Protocol ID: `core7_mtf_shadow_anchor_ppo_v4`
- Pinned Protocol Digest: `af34928c682de43afce8e203a42841ec7bdb89eb9407fe0b9399b7821e2ca013`
- Pinned Packet Digest: `e45271d30f6c764b6a93035764c3cf7dcf936925f9208d9efbb00ea89925e3a9`
