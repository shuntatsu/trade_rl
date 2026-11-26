# 研究基盤と個別研究のリポジトリ分離計画

Status: Active — external study migration in progress; execution cutover and core cleanup remain separately gated.

## 目的と非目標

既存の `shuntatsu/trade_rl` を再利用可能な研究基盤として維持し、別のprivate repository
[shuntatsu/trade_rl_research](https://github.com/shuntatsu/trade_rl_research) に個別研究の実装、仮説、比較条件、実行、判定、証拠参照を段階的に集約する。
Python配布名 `trade-rl` とimport名 `trade_rl` は変更しない。
研究用の別配布名は `trade-rl-research`、import名は `trade_rl_research` とする。
新規の個別研究は研究repoを通常の作業場所とする。移行中の研究実装・台帳・検証状況は [研究リポジトリ](https://github.com/shuntatsu/trade_rl_research) で確認し、基盤側へ研究別進捗の複製を増やさない。公開範囲の変更、ライセンスの追加判断、既存sourceの削除は別途確認する。
本計画は基盤の現行source配置、研究結果、gate、実行権限を変更しない。

目標は研究候補の追加によって共通会計・データ・学習基盤の責務やリリースを変更しなくてよい状態。
三つ以上のリポジトリ、汎用plugin framework、既存engineの再実装、全フォルダの命名変更は非目標。
実行用の権限・データ・外部artifact storeはGitリポジトリの分離と同時に公開しない。

## 責務境界

| 所有対象 | 基盤 trade_rl | 研究 trade_rl_research |
|---|---|---|
| データ取得・PIT | MarketDataset、source clock、availability、Binance/market_store/cross_assetの共通adapter | study-specific universe、期間、feature roster、data selection |
| 特徴量 | 再利用可能で因果性・単位が検証済みの演算と変換 | 指標の組合せ、閾値、研究固有context |
| Strategy / learning | observation/action/PortfolioPolicy契約、reference control、汎用fit/runtime、model/checkpoint codec | 研究仮説に対応するpolicy、学習設定、model構成、候補roster |
| Risk / simulation | hard risk、注文・約定・funding・borrow・会計、native replay | 経済条件の事前登録とengineへの入力 |
| Evaluation | metrics、comparison、robustness、generic Run/Study/EvidenceSet、one-shot/seal・final authorization契約 | runner、protocol、比較群、判定規則、個別分析 |
| Assurance / ops | 共通schema、validator、source-bound review契約、基盤の配布・CI | 研究固有author/review packet、run admission、stage roster、研究リポジトリのCI |
| 研究の証拠 | 既存の固定履歴、基盤の再現性に必要な旧artifact reader | 新規のimmutable evidence参照、trial履歴と科学的判断 |

依存は `trade_rl_research -> trade_rl` のみ。基盤から研究のpackage/CLI/testsを
importしない。研究は共通engineを呼び出し、独自の注文・リスク・会計経路を作らない。
研究からの基盤改善依頼は、既存Issueへの追記または短いIssue（再現条件、期待契約、影響範囲）でよい。小規模なdocs/test/意味保存変更には新しい研究packetを要求せず、変更規模に応じた検証を行う。実験認可・正確な証拠保全・統合時の必須review/CIは従来どおりとする。
基盤にreference strategyが必要な場合は最小の比較用実装だけを残す。戦略を一般化した
という理由だけで学習済みcheckpointやstudy固有の閾値を基盤へ昇格させない。

## 目標フォルダ構成

基盤は現行の責務別7 packageを維持する。全面的な `src/` 移行は不要。
以下は段階移行後の目標であり、現行の公開path変更を宣言するものではない。

```text
trade_rl/
├── trade_rl/
│   ├── artifacts/
│   ├── data/
│   │   ├── artifacts/
│   │   ├── build/
│   │   └── features/
│   ├── integrations/
│   ├── risk/
│   ├── simulation/
│   ├── strategies/          # contract, reference controls, reusable training
│   └── evaluation/          # common replay, metrics, run/study/authorization
├── tests/                   # core unit/contract/integration/architecture
├── tools/                   # core maintenance, release and assurance adapters
├── docs/
│   ├── README.md
│   ├── AGENTS.md
│   ├── architecture/        # common contracts only
│   ├── specs/               # active core changes only
│   └── plans/               # active core migrations only
├── guide/                   # non-authoritative platform guide
├── deploy/                  # existing operational integration, initially retained
├── report/                  # existing immutable evidence, not new research output
├── LICENSE
└── LICENSES/
```

```text
trade_rl_research/
├── pyproject.toml
├── uv.lock
├── README.md
├── AGENTS.md
├── src/trade_rl_research/
│   ├── studies/
│   │   ├── cloud_regime_switch/
│   │   │   ├── __init__.py
│   │   │   ├── features.py
│   │   │   ├── policies.py
│   │   │   ├── protocol.py
│   │   │   ├── runner.py
│   │   │   └── analysis.py
│   │   ├── trend/
│   │   ├── carry/
│   │   ├── cross_sectional/
│   │   └── rl/
│   └── common/             # only proven reusable research helpers
├── protocols/
│   └── cloud_regime_switch/
│       └── dev_v1/
│           ├── protocol.json
│           ├── data_manifest.json
│           └── author_packet.json
├── tests/
│   ├── studies/            # mirrors Python study packages
│   ├── integration/        # installed core compatibility
│   └── architecture/       # no reverse import, no private core coupling
├── docs/
│   ├── README.md
│   ├── research/
│   │   ├── current-status.md
│   │   └── cloud-regime-switch.md
│   └── methodology/        # research-specific usage, links to core authority
├── evidence/              # tracked immutable small receipts/manifests only
│   └── <study>/<run-id>/
├── LICENSE
├── LICENSES/
└── .github/workflows/
```

`src/` は新リポジトリだけで採用し、installed packageへの依存をテストする。
study固有codeは研究単位でco-locateし、features/strategies/evaluationの全体横断フォルダへ
再び分散させない。研究群の細分化は実際の責務・独立test・依存に従う。
`common/` は2つ以上の実利用があり、共通化すべき契約が確認できた場合だけ増やす。
研究IDはcode、protocol、docs、test、evidence間で一貫させる。空のstudy folderや
将来用の抽象層は作らない。studyの固定machine-readable条件は `protocols/<study>/<protocol-version>/` に置き、Python側
で読み込む。docsの説明を第二の実行条件にしない。runnerへのentry pointは明示的に登録し、
任意moduleの自動発見や任意コードの実行を新設しない。
market data、checkpoint、巨大ledger、ログ、run directory、credentialはGit対象外の
既存artifact storageに保存する。小さいimmutable receiptのみ `evidence/` に記録し、
元データ・artifact・reviewの場所とdigestを参照する。seal/consumption registryは単一の
権威を維持し、リポジトリごとに独立した消費状態を作らない。

## 現行sourceの棚卸しと移行単位

現状調査は 2026-11-24 の main
`3bec2c65b551088d8af9465c97109c8cb5b9a8cc` に固定した。
Git追跡fileは `trade_rl/` 386、うち `evaluation/` 215、`strategies/` 56、
`tests/` 624、`tools/` 86、`docs/` 30、`report/` 52。
これは移動数ではない。既存 `tools.agent_repo.source_index.ImportCollector` で
以下12群を静的走査したところ101 Python moduleを候補として識別した。
再exportを含むimport調査であり、動的import、package初期化、serialization、実行時
resource、CI、外部consumerの完全なclosureを証明しない。各移行時に再走査する。

| 現行の研究群・source | 移行先候補 | 先に確認する境界 |
|---|---|---|
| `evaluation/carry_multisymbol/`, `evaluation/carry.py`, `strategies/carry.py` | carry | `paper/account.py` がcarryを参照。共通paired executionを残す。 |
| `evaluation/trend_stop_study/` | [固定履歴](https://github.com/shuntatsu/trade_rl_research/blob/main/migration/trend_stop_archive.json) | 終了済みstudyを削除。共通rule・native replayを維持。 |
| `evaluation/trend_oos/` | trend | 現行consumer・source-bound実行経路を確認して移管。 |
| `evaluation/agy_anchor_oos/`, `strategies/agy_anchor/` | trend/agy_anchor | 原著provenance、他研究、固定OOS判定。 |
| `evaluation/xs_lowvol_funding/` | cross_sectional | universe/delisting/stage、未使用範囲・owner認可。 |
| `evaluation/core7_mtf_study/`, `strategies/rl/shadow_anchor_mtf.py` | rl/core7 | 他研究9 module、source closure、v14 packet、checkpoint。 |
| `evaluation/lowturn_screen/` | lowturn | 他研究8 module、凍結candidate roster。 |
| `evaluation/decision_value_*.py`, `strategies/decision_value*.py` | decision_value | evaluation facade、paired action/label、Run Coreの5 consumer。汎用fitはcore側に残す可能性。 |
| `evaluation/directional*.py` | directional | 共通のsignal/decision契約と研究固有rosterを分ける。 |
| `evaluation/dev_realdata_study/` | historical_development | forward_study/fs1とregistryが依存。旧registryはcoreに固定維持。 |
| `evaluation/residual_rl/` | [固定履歴](https://github.com/shuntatsu/trade_rl_research/blob/main/migration/residual_rl_archive.json) | 終了済み。独自ledger・実行コード・結果のcurrent-tree複製を削除。元Git履歴は不変。 |
| `evaluation/ppo_*study.py`, `ppo_*checkpoint.py`, `ppo_*normalization*.py`, `ppo_4h_indicator_smoke.py` | rl/ppo | 固定activation JSON、PPO関連CI、optional extras、checkpoint互換性。 |
| `evaluation/rl_family_comparison/` | [研究側](https://github.com/shuntatsu/trade_rl_research/blob/main/migration/rl_family_comparison_sources.json) | 移管済みの個別比較契約・oracleの旧実装を削除。共通fitterは維持。 |
| `evaluation/experiments/bootstrap/capacity_robustness_prereg.py` | [固定履歴](https://github.com/shuntatsu/trade_rl_research/blob/main/migration/capacity_robustness_archive.json) | #541 終了済み個別研究。固定protocolと専用テストをGit履歴に保持し、共通bootstrap/materializationは維持。新規実行・結果再分類なし。 |

`tools/trend_breadth.py`、`trend_exit_arms_forward.py`、`mtf_combo_forward.py`、
`btc_trend_ensemble.py`、`agy_clean_all_forward.py`、`carry_multisymbol_screen.py` 等の
study専用runnerも該当研究群と一緒に移す候補。
`tools/agent_repo/`、branch hygiene、GitHub coordination、markdown verificationなどの
基盤運用は基盤側に残す。現状 `tools` は配布wheelの対象外なので、研究側から
`tools.*` のimportやcore checkoutの相対path参照を前提としない。共通で必要な
assurance/authorization capabilityは明示した配布APIまたは信頼済みversioned CLIへ
先に分離する。既存の `tools/core7_g4_comment.py` はrepository、owner、comment URL
に固定された研究固有adapterであり、単純なrepo名置換で新研究を認可しない。

共通packageの公開facade、既存Run/Study schema、portfolio policy codec、provider data、
artifact identity、`forward_study` / `forward_acquisition` の現行消費記録、
paper運用、training/inferenceの共有部品は初期移行から除外する。
研究固有部分と共通authorityが混在するものは個別の責務分離PRを先行させる。
現行 `tests/architecture/` は現在のpath/APIを検証しているため、物理移動時に
実装とテスト・docs・Guide・package-data・CI・source pinを同時に更新する。
テストを削除/緩和して移行を成立させない。

## 最初の移行パイロット

未統合の雲＋ダイバージェンス系を最初に対象とする。既存mainの研究群を先に
削る必要はない。対象PRは #1340 / #1341 / #1344 / #1345 と
[固定比較のDraft PR #1355](https://github.com/shuntatsu/trade_rl/pull/1355)。
元PR、branch、review、protocol、CI、保管済み研究証拠はそのまま保持する。

- generic divergence detector、Ichimoku/indicator bank、completed-bar alignmentは
  coreへの昇格候補。canonical source ownerとの重複を解消し、公開APIと因果性・単位・
  異常入力の契約を確定してから採用する。研究固有6時間/24時間閾値、context、quorum
  構成などは研究側で指定する。
- C2/C3、R1/R2、N0–N3、T0–T2、S0–S2のpolicyと研究固有入力契約の初期移植は
  [PR #1356の検証記録](https://github.com/shuntatsu/trade_rl/pull/1356) と別repoのsourceで確認する。
  既存 `PortfolioPolicy` / `run_shared_portfolio_replay` を唯一の執行経路とする。
  研究repoの初期合成テストは全feature adapter・runner・研究CI・認可・経済的同等性の
  完了証拠ではない。正式なcore配布とpinの受入は [Issue #1359](https://github.com/shuntatsu/trade_rl/issues/1359) で追跡する。
- #1355のrunner/summary・198セルprotocolは研究側へ置く候補。現行sourceが参照する
  `RawMarketSeries` のprivate helpers、MTFの `_aggregate` 等、core側の `tools.*`、
  Fedora固定path、hard-coded repository/owner approvalを洗い出す。
  公開extension seam、明示的なrun配置、review admissionへ移行してから外部packageで
  動かす。現在の承認条件を外さない。
- 未統合研究のソフトウェアは別repoで試験移植しても、#1355の元の198試行の実行条件、
  source HEAD、author packet、データidentityを上書きしない。移植結果は新しいsource
  identityとして扱い、既存承認を引き継がない。作業・レビューのroutingと経済runnerの委譲を区別し、実行権威の切替はcutover完了後に行う。
  既存の継続タスクと新しいrunnerが同じ試行を二重実行しないよう、承認主体と
  exclusive attempt/consumption authorityを切替前に確定する。

## 時間軸を限定しない共通基盤

基盤側の実装要件は [Issue #1357](https://github.com/shuntatsu/trade_rl/issues/1357)
で管理する。既存の #1340 の指標実装と調整し、重複した計算ownerを作らない。
研究固有の1h/4h/1dや固定年を共通APIの前提にしない。

- Native timeframe、decision cadence、execution clock、elapsed duration、
  calendar periodを別々の契約として扱う。指標のlookback/変位はnativeバー数、
  保有期限・cooldown・latencyはバー数か実時間かを明示する。
  26本の日足と26日間、26本の週足と26日間を混同しない。
- 現行の対応は15m/30m/1h/2h/4h/6h/8h/12h/1d。3d、1w等を段階的に拡張し、
  calendar monthや休場を持つ市場には別途calendar契約を設計する。
  28日足や固定30日足を月足と呼ばない。週足closeの曜日・UTC anchor、
  leap year、欠損、遅延、部分windowとwarmupを検証する。
- 一目均衡表は既存のunshifted featureとvisible shifted cloudを区別する。
  標準9/26/52/26のvisible cloudは最初の78 native barsが必要。
  period変更時は最長の入力依存範囲を検査し、未来の描画位置をdecision時点の
  入力にしない。任意の設定は新しいversioned identityにbindする。
- Source acquisition、完全なresampling、as-of alignment、feature mask、
  annualization、共通replay・費用・funding・borrow・terminal stateまで
  end-to-endで確認する。現行 `MarketDataset` は正規の連続週足でも年8760hを
  整数期間数に変換できず拒否する。この制約を丸めやcalendar種別の偽装で回避しない。
  native intervalとelapsed yearを表す新契約が必要ならschema/reader/identityを
  versioned migrationとして変更する。
- Opt-in `trade_rl.evaluation.native_annualization.evaluate_native_bar_annualization`
  は完全なnative gridと1区間1 simple returnを受け取り、ACT/365Fの実経過年と
  観測頻度をFractionで返す。riskは算術driftに対する区間時間比例の残差二乗和を
  実経過年で割る明示的な記述モデルとし、zero-target downsideも同じ実経過年で
  正規化する。固定等間隔では既存ratioの丸め前の定義に一致する（ゼロ分散に
  対する浮動小数点の偽の残差は除く）。月足を固定30日扱いせず、名目月次Sharpe、
  CAGR、独立増分、providerの真正性または取引可能性の主張ではない。
  既存ReturnSeries / PerformanceMetrics / Datasetのschema・identity・既定値は不変。
  Dataset/replayからの接続、source provenance、execution/carryは別の受入条件である。
- Opt-in `RollingPortfolioRiskInputsProvider(native_clock=...)` は、観測した
  close-to-close log returnのnative intervalと公開時刻を検証し、消費する価格窓は
  native binary64-compatibleな配列に限定する（より広い浮動小数点型、絶対値が2**53を
  超える64bit整数は変換前に拒否し、未来rowの値は検査しない）。ACT/365Fで
  年率共分散を算出する。等間隔では既存sample covarianceの整数丸め前の年率換算と
  一致する。不等間隔では総経過時間T、各区間t_iについて、総log returnの
  t_i/Tを差し引いた残差の外積和を、実効観測時間 T - sum(t_i²)/T で割る
  （一定の共分散と独立増分を仮定した線形drift推定の補正）。これは統計モデルの
  選択であり、独立性や定常性を実証するものではない。非年率共分散とstress quantile
  は観測バー単位のまま、固定時間のshockとはみなさない。clock、anchor、設定を
  別のprovider identityにbindし、旧providerのidentityと計算を変更しない。
  新しいDatasetのadmissionや実行時計、経済実験の認可を意味しない。
- 長い履歴（複数年・細かい足）を固定した件数上限や特定年に依存せず扱えること。
  必要なwarmupの実時間はnative intervalから求める。先読み防止と同じ結果を
  保ったまま計算量・peak memoryを測定し、期間延長で再計算量が過度に増えない
  ことをsynthetic benchmarkで確認する。

現行mainの合成120 barsによる確認では、9種類の既存時間足でcanonical
Ichimoku cloud-positionの値とavailabilityが一致し、69 barsがvalidだった。
#1340の固定sourceのvisible-cloud primitiveは日足・週足・固定28日足で
43 barsがreadyとなり、バー単位の値が一致した。ただし後者は指標演算のみで、
週足/月足のデータ取得・Dataset・執行・評価を実装済みとはいえない。
これらの具体的な拒否経路と受入条件は上記Issueに残した。

## 研究から基盤への改善依頼

個別研究は、自分の研究scopeの仮説、policy、runner、analysis、docs/testsを所有する。
基盤のbug、機能不足、数値精度、性能問題を見つけたら、研究側のagentは
[基盤Issue](https://github.com/shuntatsu/trade_rl/issues/new)に報告する。
既存のIssue/PR/Taskを検索して重複を避け、既存案件なら最小再現と影響範囲を追記する。
研究のために基盤を直接編集したり、独自の会計・時刻変換・risk実装で穴を埋めない。
未対応のcapabilityが必須なら研究をBLOCKEDとし、独立して進められる分析・合成testは
継続する。基盤のバグと研究固有の戦略成績不良は別の問題として扱う。

[Core request template](../../.github/ISSUE_TEMPLATE/core-request.md) に沿い、
研究repo/study/commit、利用中core commit/wheel digest、再現条件、期待値と実測値、
最小の合成再現、影響を受ける契約、希望する機能と受入テストを記載する。
秘密情報、未使用/封印データ、非公開の生取引記録をIssueに貼らない。
結果後の条件変更や経済実行の許可をIssueの存在から推定しない。

基盤側は既存Task/leaseとPR経路でtriage、修正、独立review、Full CI、merge、
versioned wheel公開とreadbackを行う。Issueには対応するPR/固定core SHA、
変更契約、検証結果を返す。研究側はそのIssueを参照して依存pinを明示更新し、
最小再現と研究固有の回帰テストを実行する。確認後にのみ研究側のBLOCKEDを解消する。
Issueがclosedでも旧環境が自動的に修正されたとはみなさない。
同じ基盤変更に複数の研究から依頼があっても実装ownerは一つにする。
研究repoのAGENTS.mdはこのrouting規則を明示している。既存のcore変更は引き続き基盤PR経路を用いる。

## その他の最適化方針

| 領域 | 変更方針と検証 |
|---|---|
| Data / cache | 研究別の価格取得・保存の複製を避け、既存market storeとverified artifactsを共用。data identity・availability・消費境界を固定。 |
| Feature compute | 同じ入力・期間・指標設定の計算を試行間で再利用する。causal windowの前計算やincremental更新はprefix/naive oracle parityと性能測定を通して採用。 |
| 実験実行 | runnerを共通engineに接続し、固定rosterの独立cellだけを並列化。資源上限・single-writer・exclusive publication・再開時のattempt identityを維持。完了済み結果を再実行で置換しない。 |
| 設定 / provenance | protocolを唯一のmachine-readable条件とし、設定の重複・hard-coded pathを排除。core/research/source/lock/receiptを一体として固定し、未承認や不一致はfail closed。 |
| テスト / CI | coreは完全なcontract/Full CIを維持。研究では変更studyの近傍testを高速反復し、最終固定wheelとの全関連testとpackagingを検証。economic runは通常CIから分離。 |
| docs / Agent | READMEを入口、恒久契約と研究固有の仮説・判定を別ownerにする。研究ごとのstatusとIssue/PRへの短いリンクで案内し、長い作業履歴の複製を避ける。 |
| 依存管理 | core releaseを自動追従しない。研究側の固定wheel更新PRでschema/API/旧artifact/研究結果の再現性を検査。private helperへの依存を新しい公開契約で解消。 |
| 運用 | 認証済み権限・approved source・進行中attemptを一つの権威で管理し、repo分離による二重run、二重消費、誤ったreview再利用を防ぐ。 |

最適化のためだけの抽象化、既存の固定研究を無条件に再実行すること、
研究codeの都合によるcoreの安全条件緩和は対象外とする。
性能については現状値を測定するまで改善率を主張しない。

## 配布・固定・CI

基盤の現行Python 3.12と `trade-rl` wheelを維持する。研究側は
レビュー済みcore commitから構築した配布wheelを使用する。versionだけで
sourceの一致を推定せず、core repository/full commit/tree、wheel SHA-256、
lock digest、research repository/full commit/tree、protocol digest、
Dataset/manifest/array digest、run recipe、execution/risk/cost、review/owner receipt、
attempt identity、ledger/output digestを一体としてbindする。
開発時のeditable installはローカル限定とし、正式evidenceでは未固定の
main、dirty checkout、暗黙のpath、競合する2つの `trade_rl` packageを拒否する。

core CI: 現行の完全なcore contract、architecture、distribution closure、
optional capability、DB・学習runtime、必要なintegrationを維持。
research CI: coreの固定wheelでimport/contract、独立synthetic oracle、
各study unit/integration、protocol/receipt schema、依存方向、docs/linksを実行。
通常CIは実市場の経済実験、sealed/unused input取得、本番注文を開始しない。
coreの更新は別PRで研究側の固定versionを更新し、全影響studyの互換性を再検査する。
core mainを研究PRの都合で直接変更せず、reviewer/CI結果の適用scopeをrepoごとに区別する。

## 文書・証拠の正本

現在の [docs index](../README.md)、[package boundaries](../architecture/package-boundaries.md)、
[research assurance](../architecture/research-assurance.md)、
[controlled experiment loop](../architecture/controlled-experiment-loop.md)、
[final authorization](../architecture/final-evaluation-authorization.md) は移行完了までは現行正本。
基盤docsには恒久的な技術契約・共通方法・現行の基盤作業のみを残す。
研究docsの入口ではstudyごとの状態、仮説、固定protocol、判定、限界、次gateへ案内する。
研究の進捗はIssue/PR、実験の生記録はimmutable artifact、現行判断は研究repoの
current-statusとし、同じ内容を両repoへ複製しない。分離途中は旧statusを正本のまま維持し、
移行した研究だけを明示的に参照先変更する。既存Guideの参照とfingerprintを検査する。

既存の `report/` のpacket、provenance、registry、negative/invalid result、source-bound
evidenceは改変しない。旧repoのexact Git commit/pathを参照可能なまま保持し、
必要なときはoriginal bytesとhashを照合してimmutable copyする。
Core7のpinned source closure、旧review、unused consumption、PPO activation、
closed study stop ruleは別repoに移しただけでは更新できない。新sourceでの新runは
新たなreviewと認可を必要とし、過去結果を新source由来に見せない。
`docs/history`、`docs/archive` は作成しない。ライセンス・third-party notices・由来の
確認を両repoで維持し、公開前にcredential/非公開artifactが含まれないことを確認する。

## 段階移行と完了条件

1. **設計・棚卸し**: 本計画と現行docs入口の整理。source/API/CIの削除なし。
2. **Pilot（部分完了）**: private repoに外部研究packageの最小構成とpolicyを初期import済み。
   core公開extension seamを確定し、旧sourceから雲研究を移植した合成fixtureで、
   同一固定入力と固定risk/costに対するsignal、decision、order/fill、
   terminal equity/cost/funding/borrow、ledgerのsemantic equalityを検証する。
   source/path/digestが変わる場合は新しいidentityを記録する。旧経済結果は読替えない。
3. **連携とcutover**: core wheelの検証、研究CI、研究側auth/registry/保存先を
   実環境でreadback。旧runner停止と新runnerへの一度限りの委譲を確認し、
   開いているPR、review、automation、CLI、docs、Guide、package resourcesを追跡する。
4. **既存研究の移行**: 進行中の研究、封印済み研究、閉じた研究を区別し、
   依存の少ない研究群から一群ずつ移植する。閉じた研究は再評価せず、
   旧exact-sourceへの参照だけでもよい。混在するcore責務は先に分離する。
5. **基盤の整理**: 外部research packageのimport・replay・artifact readbackと
   coreの公開consumer/CIが全て通ってから、移行済みsourceのみ削除する。
   書換えなしの履歴参照、old checkpoint compatibility、license、docs/Guide更新を確認。
   path維持だけのforwarding shimを恒久化しない。公開契約を外す場合はversioned
   migrationとconsumer更新を同一目的PRで行う。

各段階は専用branchとcurrent mainで検証し、独立レビューと必要なFull CIが揃うまで
mergeしない。途中で失敗すればcutoverを止め、旧repo/旧runner/旧artifactに戻る。
旧ref、既存source、試行記録を先に削除しない。二重実行や二重消費の検出、
未承認runの拒否、coreが研究をimportしないこと、両repoの責務・配布・docs/CI・
再現性・source provenanceが独立して検証できることを分離完了条件とする。
本計画の完了は実市場での経済優位性、G4/G5、production/live認可を意味しない。
