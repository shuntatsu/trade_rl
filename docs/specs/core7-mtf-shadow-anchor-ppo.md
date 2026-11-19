# Core7 Multi-Timeframe Shadow-Anchor PPO & Non-RL Regime-Gate Benchmark Specification

Status: Active

## 1. 目的と位置づけ（Phase 0: Software Validity Only）

本仕様書は、[Issue #810](https://github.com/shuntatsu/trade_rl/issues/810) 配下の 6 時間軸・11 銘柄・22 レッグ（`:spot` / `:perp` 交互配列）Shadow-Anchor PPO 統合アーキテクチャおよび固定因果 Non-RL レジームゲート対照ベンチマーク（`ANC_REGIME_GATE_NONRL`）の**Phase 0（ソフトウェア整合性・合成オラクル・境界契約のみ）**を定義する。

- 現在の証拠水準: `software_validity`（`edge_claim = false`, `production_eligible = false`, `live_trading_authorized = false`）。
- 次の許可行動クラス: `synthetic_verification_only`（合成入力と不変境界テストのみ。実市場データの読込・学習・リプレイは禁止）。
- 実市場データを用いる **Phase 1（Tier D 既存データ・正準エンジン整合性比較）** は、Phase 0 実装コミットへピン留めされた独立 G0–G2 レビューおよび別途発行されるオーナー G4 認可 URL（`https://github.com/shuntatsu/trade_rl/(issues|pull)/<id>#issuecomment-<id>`）なしには起動できない。
- いかなる優位性主張・実運用可否判定も Phase 1 からは導けず、別の事前登録済み前向き実験プロトコル（Phase 2）と G5 認可が必要となる。

詳細なパッケージ責務・依存方向は [architecture/package-boundaries.md](../architecture/package-boundaries.md#core7-mtf-shadow-anchor-ppo-boundary)、Research Assurance Gate の正本は [architecture/research-assurance.md](../architecture/research-assurance.md) を参照する。

## 2. 凍結プロトコル（`core7_mtf_shadow_anchor_ppo_v1`）

### 2.1 対象ユニバース（11 ペア / 22 交互レッグ）

`BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `SOLUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `AVAXUSDT`, `LINKUSDT`, `LTCUSDT`, `DOTUSDT` の各ペアについて `(sym:spot, sym:perp)` をこの順序で交互配置する（実データ研究時は完全 11 ペア必須、合成 G2 テストでは 2–11 ペアを許容）。

### 2.2 時間境界・データ整合性

- Pre-FS1 ハード境界: 全バーの timestamp は `2026-08-01T00:00:00`（UTC）**未満**でなければならない（`>= 2026-08-01T00:00:00` は fail-closed 拒否）。
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
- `processing_bar_volume_capacity=False`: 全シナリオで 1% 参加率上限を**先行完了バー `t - 1` の出来高**だけで判定し、約定バー自身の出来高ルックアヘッドを遮断する。
- ハードリスク整合: `PreTradeRiskConfig(max_gross=2.5, max_abs_weight=0.55, max_turnover=2.5, drawdown_start=0.10, drawdown_stop=0.20, emergency_turnover_override=True)` と `ExecutionCostConfig(max_leverage=2.5, maintenance_margin_rate=0.05, max_participation_rate=0.01, minimum_notional=10.0)` の整合を保つ。
- HOLD／Deadband とドローダウン保護: リバランス間隔外または L1 目標変化が `deadband=0.04` 未満のバーでは `hold_current_quantity_symbols` を返し不要なドリフト約定を防ぐ一方、現在または最大ドローダウンが `drawdown_start` に達した場合は HOLD を解除して `PreTradeRisk` による縮小・緊急フラット化を妨げない。

## 3. 事前データ露出（Pre-FS1 Rounds 1–12）の開示

- Pre-FS1 開発窓（`2021-01-01` ～ `2026-07-31`）の 11 銘柄データは Round 1–12 の探索で既読（known-used Tier D データ）である。Round 1–11 では `DSR_null = 0.788 / 0.721 < 0.95` と多重検定の帰無水準を下回り、Round 12 のレジーム適応検証では `ANC_REGIME_GATE_NONRL` が低迷期のドローダウンを同等に抑えたことで `H13-A NOT SUPPORTED` / `H13-B NOT SUPPORTED` の帰無結果が得られている。
- そのため `ANC_REGIME_GATE_NONRL` を本仕様および `decision.py` 診断の正式対照ベンチマークとして組み込み、RL 各アームについて `ANC_BIDIR_75_25` と `ANC_REGIME_GATE_NONRL` 双方に対するリターン差・Sharpe 差・ドローダウン差・優位性判定フラグを強制出力する。
- 事前探索結果は研究上の参考背景に過ぎず、本プロトコルの `decision.py` 判定ラベルは構造・因果性・台帳整合・停止非救済のみを測る `PASS_PARITY_ONLY` / `STOP_FAIL_NO_RESCUE` / `INVALID` に限定される。

## 4. Phase 0 合成 G2 オラクル（FM1–FM3 + 独立 Decimal 台帳）

1. **FM1（ゼロ残差・HOLD・Non-RL 対照整合）**: `ShadowAnchorMlpWeights.zero_residual()` を与えた `core7_sparse_ppo` / `orth6_cush_ppo` が `anc_bidir_75_25` と目標重み・終端資産ともに `< 1e-12` で一致すること。非リバランスおよび deadband 内で不要約定がゼロであり、ドローダウン発動時に HOLD が解除されること。
2. **FM2（因果性・`SPOT_HALTS`・Pre-FS1 境界）**: `k_cut` 以降の価格・出来高・Funding 摂動に対する `k_cut` 以前の全 6 時間軸特徴・ゲート・提案重み差分が `== 0.0` であること。`SPOT_HALTS` が前方補間・NaN なくマスクされ、`>= 2026-08-01T00:00:00` および `fit_stop_index > start_index` が fail-closed に拒否されること。
3. **FM3（先行出来高ロック・リスク／レバレッジ整合・独立 Decimal 台帳）**: `processing_bar_volume_capacity=False` により実行バー自身の出来高ショックへ不変であること、`max_leverage=1.0` と `max_gross=2.5` の不整合が拒否されること、および本番会計を import しない独立 50 桁 `Decimal` 2 ペア 4 レッグ台帳が全バーの現金・純資産・コスト（手数料＋スプレッド＋`impact_rate * sqrt(participation)`）・Funding PnL・ターンオーバー・終端フラット性を `abs <= 2e-9` / `rel <= 2e-12` で再現すること。

## 5. Packet Pin

- Open Author Packet: `report/issue810-core7-mtf-ppo-research-assurance-v1.json`
- Pinned Implementation Head: `PENDING_STEP5_IMPL_SHA`
- Pinned Protocol ID: `core7_mtf_shadow_anchor_ppo_v1`
- Pinned Protocol Digest: `c61d96248f58e4515254cd17a4bc8562714273bce626d627da9eeb9a9c6968f4`
- Pinned Packet Digest: `PENDING_STEP5_PACKET_DIGEST`
