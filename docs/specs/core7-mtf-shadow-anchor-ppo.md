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

- Open Author Packet: `report/issue810-core7-mtf-ppo-research-assurance-v22.json`（v1〜v21、main正式v14とM1並行v14アーカイブはすべて不変のUNREVIEWED履歴。旧source pinを新コードの経済認可へ再利用しない）。
- Pinned Implementation Head: `ae83944b7d8773549428c5a71d056bc54763fe12`
- Pin lineage: v9のTD8失敗レシートreader保護（[PR #1322](https://github.com/shuntatsu/trade_rl/pull/1322)、digest `1a57f5b478dec5ad90fa844700730db092e14095b281421c2558c30372a0a57d`、実装HEAD `97cd1fd4e79a1398b25402b4283d773143b1fc6a`）は変更せず歴史として保持する。v10は最新mainのTD8変更とM1監査のPreTradeRisk数値・緊急マスク境界、失敗時stateful乱数復元を含む公開統合ソース `aa9282c74ffdf866831d0f0b423401a7e7e3ca29` を実装・プロトコルの両HEADへ固定する新しいsource-onlyパケット。現行ソースによる旧v9の実行・証拠再分類は認めない。G0–G2の独立レビュー前は **UNREVIEWED / economic_execution_authorized=false**、Core7 v4の経済ルール・費用・リスク閾値・過去研究結果・プロトコルdigestは変更しない。別のG4認可、Pre-FS1/FS1/unused/final禁輸およびno-live-order条件を維持する。
- v11 source-only lineage: v10のM1・TD8保護とそのimmutable packet digest `8927defcfb0417be50ce6af1a0b5335ccb14fd0d1f7e2367836eee179cb4fede` は変更しない。本v11はIssue #1134と#1131のPOSIX receipt root/leafおよびStudy JSON読取中のFD/inode/time identityを固定した公開source commit `a5224b86253e90a12bf23cbb7e8be732f82d4475` を新規ピンする。historical v2/v3 receipt、Study証拠・過去decision・Core7 v4経済protocol、費用／リスク閾値、数値resultには変更を加えない。UNREVIEWED / economic_execution_authorized=false / production_eligible=falseを維持し、G0–G2の別独立レビューとowner G4がない限りTier D実測、FS1/unused/final閲覧、利益の主張やlive注文を認可しない。
- v12 source-only lineage: v11の既公開read-identity保護と immutable packet digest `390db3ecc403f61e4f9bc5a30bc1dc0c6f54ab7e3af30dba072fc272e71c2a6b` を変更せず保存する。Geminiの２指摘（TD8の一時的なA→B→A差替えがsingle-entryチェックをすり抜ける問題と、M2のread中unlink時に生のOSErrorが出る問題）を、固定した可読directory FDでの`os.scandir(fd)`と型付き`ArtifactIntegrityError`で拒否した公開ソース `553f8828cfe00627edc721e1e5195e37dcdcddcc` を新たにピンする。v2/v3 receipt・既存Study・旧経済結果・Core7 v4プロトコルは不変。別G0–G2、owner G4承認前は **UNREVIEWED / economic_execution_authorized=false / production_eligible=false** を維持し、実データ・FS1/unused/final・live注文は引き続き認可しない。
- v13 source-only lineage: v12のTD8/M2レビュー是正とimmutable packet digest `76d0644bc61f4cbf22526d000ea8cac9fa9e8f18159679e41a0f3a2c36e9b2e7` は不変。M1 `BookState.settle_positions` における整数・NaN・文字列・多次元・masked-arrayの暗黙のbool変換による誤清算を、清算前の厳格な一次元bool型チェックへ修正したソースと、v12を含むTD8/M2ソースを2-parent統合した公開コミット `f9be9737fe27facdb1dfd3147e436196c3d6d278` を新規ピンする。合成REDと正常bool精算の回帰テストを保持。旧研究結果・Core7 v4の経済的条件・G0–G5認可は変更しない。新v13も **UNREVIEWED / economic_execution_authorized=false / production_eligible=false / live_trading_authorized=false** であり、実データ・FS1/unused/finalや注文を認可しない。
- Canonical merged-main v14 source-only lineage (#1307): `report/issue810-core7-mtf-ppo-research-assurance-v14.json` は既にmainに固定済み、digest `d10520a1e8acd26d871441d2d89e4a75a8235856cde8cd21d7c85b314c533158`、source `66d03d6d9a420474b137ce3a084bcca330db9954`。このv14はopt-in XS `engine_settlement` を含むsource-only packetであり、Core7の経済判定・採用は変えない。#1112並行ブランチで異なるv14が先に公開されていたため、mainの正本v14を上書きせず、M1側の旧v14を別名で原バイト保存した。両方は同じv13を親とする並行のUNREVIEWED source-only履歴で、mechanism/evidenceは同一。以降のM1 v15～v17を遡及的に再計算しない。
- M1 parallel v14 source-only lineage (immutable archived packet `report/issue1112-m1-core7-mtf-ppo-research-assurance-v14.json`): v13のmaskedな決済セレクタ拒否とimmutable packet digest `1d27476dd42d92029aafe9b2529da95dc62845a087f6fcd8d241cc4f31facf09` を変更せず保持する。M1 `BookState` の数量・評価価格・契約倍率・決済価格・回復率とsplit/配当/再評価/mark-to-marketの数値vectorが、`np.asarray` のmask消失により欠損値を利用する経路を拒否した公開ソース `f20fbb116ff73c20867a07a05d3f6d7877fcbf48` を新規ピンする。9件の合成回帰テストで不正入力時の口座不変性を確認する。v1〜v13の履歴、Core7 v4経済条件・過去結果・G0〜G5の認可は不変。新v14も **UNREVIEWED / economic_execution_authorized=false / production_eligible=false / live_trading_authorized=false** であり、実データ・FS1/unused/final・注文は認可しない。
- v15 source-only lineage: v14のimmutable digest `be04c734d837ad1037ce2ab692cadc6cc9979d2672f7d103dfd8b16d3a03b1cb` とv1〜v14のUNREVIEWED履歴を変更しない。入れ子のリスト・タプル・object配列の内部に置かれたNumPy masked arrayが、会計数値vectorの`np.asarray`で欠損情報を失い数値へ変換される経路を公開ソース `397646acfc357669f9b5d08709a7b8054d19820d` で拒否する。RED→GREENで４つの不正入力と正常な入れ子数値リストを検証し、失敗した決済のcash・正確な保有量・mark・カウンタ不変を確認する。Core7 v4の経済メカニズム・過去研究結果・G0〜G5認可は変えない。新v15も **UNREVIEWED / economic_execution_authorized=false / production_eligible=false / live_trading_authorized=false** であり、実データ・FS1/unused/final・注文は認可しない。
- v16 source-only lineage: v15のimmutable digest `460167479a86d52bec498abaf7fd6e7156b9b6d98e2bc85b40521038875844f9` とv1〜v15のUNREVIEWED履歴を変更せず保持する。Gemini 4 Hydrogen HighのLOW指摘で確認された`deque`・`UserList`や不透明な`__array__`アダプタが数値maskを隠して会計へ通す経路を公開ソース `13471eeda89e28262e2c1522b2aa6d0a54db938c` で拒否する。通常の数値ndarray・リスト・タプル・deque・UserListを維持し、欠損配列や不透明なアダプタはfail-closed。合成RED→GREENと旧packet不変性をテストし、経済ルール・過去結果・G0〜G5権限は変えない。新v16も **UNREVIEWED / economic_execution_authorized=false / production_eligible=false / live_trading_authorized=false**、実データ・FS1/unused/final・注文不許可。
- v17 source-only lineage: v16のimmutable digest `62a7005b01a9b5ea851fe4517c73c4fcf365c98cd602570e4a28913b9a27bd84` とv1〜v16のUNREVIEWED履歴は不変。Gemini 4 Hydrogen Highの現行HEAD検査で発見されたMEDIUM２件（ABC未登録の添字型duck sequenceと`__array_interface__`／`__array_struct__`のmask消失）およびLOWの動的入れ子`id`再利用に、公開ソース `9e1083a80f96569bf230f1ccd820df3576f5fdfb` で対応。添字値を検査時に一度だけsnapshotへ固定し、NumPyによる元オブジェクトの二重読みを防止。循環と不透明なバッファadapterをfail-closed、正常な数値ndarray・list/tuple/deque/UserList・安定した添字型の数値互換性を維持する。故障を示す合成RED→GREENと旧packetのdigest不変を検証。Core7 v4の経済プロトコル・過去結果・G0〜G5認可は不変。v17も **UNREVIEWED / economic_execution_authorized=false / production_eligible=false / live_trading_authorized=false**。実データ・FS1/unused/final・利益主張・注文は認可しない。
- v18 source-only main統合: v17のimmutable digest `d7a11ca94530fd2f2dca35ba57a928e0078b52f0a89ef14bf1725b509a01c11b` とM1欠損mask防御は変更しない。main側[PR #1307](https://github.com/shuntatsu/trade_rl/pull/1307)の既存正式v14（digest `d10520a1e8acd26d871441d2d89e4a75a8235856cde8cd21d7c85b314c533158`）を1 byteも書き換えず保持し、#1112の同時公開M1 v14（digest `be04c734d837ad1037ce2ab692cadc6cc9979d2672f7d103dfd8b16d3a03b1cb`）は `report/issue1112-m1-core7-mtf-ppo-research-assurance-v14.json` に原Git blobそのまま保管。双方のv14は同一v13から分岐し、既存M1 v15〜v17のバイト列・source pinは変更しない。PR #1307のデフォルトoffの`engine_settlement`を含むmainと、M1の会計数値vector fail-closed修正を、二親の公開統合ソース `82eb90a22a43b4452f2af7c3cc27c31768ae6667` に束縛したv18を新規追加する。Core7 v4経済protocol、既存試験結果、認可や入力Datasetは変えない。v18も **UNREVIEWED / economic_execution_authorized=false / production_eligible=false / live_trading_authorized=false**。G0〜G5・owner G4・FS1/unused/sealed/final・実経済研究・利益主張・売買注文を認可しない。
- v19 source-only opaque descriptor admission: v18の不変digest `ebd32575b29118d857ae7c0a207939dc4beb6c57f59eccd1f2febfe922c60c7b` とそのCore7ソース `82eb90a22a43b4452f2af7c3cc27c31768ae6667`、全ての旧v1〜v17 packetおよび２つの独立v14 packetを変更しない。会計数値ベクトルの `__array_interface__` / `__array_struct__` が初回の属性照会だけ `AttributeError` を返してmasked-backedな数値を隠し、後続のNumPy変換で誤受理されるREDを再現した。公開ソース `3cd4ae29abd73a7e19b9f166d8af308874e81a1a` は `inspect.getattr_static` でdescriptorを起動せず検出し、明示的にopaque値を拒否する。通常の型・入れ子の数値は保持し、失敗した決済の口座状態は不変。新v19も **UNREVIEWED / economic_execution_authorized=false / production_eligible=false / live_trading_authorized=false**。G0〜G5、FS1/unused/sealed/final、利益主張・取引注文は引き続き許可しない。
- v20 source-only masked memoryview owner: v19の不変digest `fb8bcc67cb696fc487a4fda704615c6150d23bedb51324c45b30f7ccdef792dc` と公開ソース `3cd4ae29abd73a7e19b9f166d8af308874e81a1a`、v1〜v18、mainの正式v14とM1別名保存v14を変更しない。`memoryview(MaskedArray)` は元のmaskを `.obj` に保持するのにNumPyへ渡す要素は通常の数値となり、会計vectorの欠損情報が消失する経路を合成REDで９件再現した。公開ソース `a6d1a936f8ea63f9c411d07313eea572a24912b9` では索引読みの前にmemoryviewのownerを検査して拒否し、正常な数値memoryviewの取り込みと失敗した決済の口座不変性を維持する。v20は **UNREVIEWED / economic_execution_authorized=false / production_eligible=false / live_trading_authorized=false**。元データ、G0〜G5、FS1/unused/final、利益主張、紙/実注文は認可しない。
- v21 source-only masked ndarray owner chain: v20の固定digest `d11973554924d1e4074cd33f1cb86729a60a2503867893931d74873b248b112a`、旧v1〜v19と正式#1307/M1並行v14は全てbyte不変。通常型 `np.ndarray` でも `.base` に ​MaskedArray の元のmaskが残る場合と、その配列をmemoryviewで包んだ場合に欠損情報が消失したまま会計に入る経路を、合成RED（9件＋複合5件）で確認した。公開ソース `1f1f65265835edb9bad565f7ca27915907c2fa79` は元の `ndarray.base` / `memoryview.obj` の型付き参照を検査し、mask保持元を持つ入力だけを拒否する。正常な数値配列・memoryviewは継続して認可され、清算失敗は原子的。v21も **UNREVIEWED / economic_execution_authorized=false / production_eligible=false / live_trading_authorized=false** であり、実データ・FS1/unused/final・利益主張・売買注文・G0〜G5経済認可を与えない。
- v22 source-only native numeric conversion boundary: v21 canonical digest `e8089f5e66d90da21fba8f3c390cbdff04cdbc493483f755fd80213c63426023` と旧v1〜v21・双方のv14を不変保持。独自の`__float__`／`__index__`がmasked ownerを隠す7件のRED、およびndarray派生型のdtype/tolist上書きによる4件のREDを再現。公開統合ソース `ae83944b7d8773549428c5a71d056bc54763fe12` はnative dtypeと要素、終端scalar型を検査し、不透明な変換を拒否する。通常の数値配列・安定した数値コンテナ・Decimal/Fraction・native NumPy scalarの受理と失敗時口座不変性を維持。callerが先に消失させたmaskは復元対象外。mainのsetup-uv固定SHAとLean Core skip表示契約は両方保持。v22も **UNREVIEWED / economic_execution_authorized=false / production_eligible=false / live_trading_authorized=false**。G0〜G5/G4・FS1/unused/sealed/final・利益主張・paper/live注文は許可しない。
- Pinned Protocol ID: `core7_mtf_shadow_anchor_ppo_v4`

- Pinned Protocol Digest: `af34928c682de43afce8e203a42841ec7bdb89eb9407fe0b9399b7821e2ca013`
- Pinned Packet Digest: `25fb38d34f587a2b106822900d2c6031a69b53aa77ec0b9009063b5c284e2f8f`
