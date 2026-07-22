# Trade RL Studio

ローカル優先の`trade_rl`研究コンソールです。Vite、React、Strict TypeScript、FastAPIを使い、既存の正本ArtifactとWorkflowを操作します。

> Studioは探索を理解し、証拠を読み取るための画面です。Checkpoint選択、Sealed-test、Release承認、Bundle activation、取引所注文は実行しません。Production statusは常に`NO-GO`です。

## 実装済み

- 固定Top bar、Sidebar、Workspace、Status bar
- 1536×1024／1440×900でBrowser page全体の縦Scrollなし
- System、Dataset、Job、Run、Baseline、Fold安定性、`NO-GO`を集約したDashboard
- Data Labで正本Dataset artifactを検証して一覧・詳細表示
- 実験画面から検証済みConfigとDatasetを選び、Exploratory trainingを開始
- Run Centerで永続Job状態、PID、終了Code、Logを表示し、所有Processを安全停止
- Live Trainingで学習中の探索行動をSeed単位の市場Replayとして表示
- 「ほぼライブ」／「バッファ再生」と「ローソク足ごと」／「イベント圧縮」の切替
- Price chart、Position変更Marker、再生Cursor、現在Weight、探索区間PnL、Reward、Drawdown、Risk eventの同期表示
- 複数SeedのStreamを独立選択し、CursorとBrowser bufferを混在させず再生
- 決定論的`checkpoint-selection.json`から、選択Seed、明示Fold、評価Return、Range、Digest、Finalist状態を表示
- Compareで検証済みRunの指標、設定差、Fold、累積Wealthを比較
- Evidence ExplorerでRun manifest、Identity、Authorization、Artifact file closureを監査
- Serving MonitorでActive bundleとPaper inference snapshotをRead-only表示
- FastAPIによるDataset、Run、Config、Job、Training telemetry、Checkpoint評価証拠の型付きAPI
- API未起動時は明示的な`DEMO DATA`へFallback
- 直接取引所注文、API key入力、Live資金操作は未実装

## 起動

Repository rootでPython APIを起動します。

```bash
uv sync --extra studio --extra train-sb3
uv run trade-rl studio start --project-root .
```

別TerminalでReactを起動します。

```bash
npm ci --prefix studio
npm run dev --prefix studio
```

`http://127.0.0.1:5173`を開きます。Viteは`/api`を`127.0.0.1:8765`へ転送します。

## Live Training

1. `実験`WorkspaceからExploratory training jobを開始します。
2. Sidebarの`Live Training`を開きます。
3. 実行中または保存済みJobを選択します。
4. 複数Seedがある場合は`Seed`Selectorで表示対象を選びます。Seed変更時は受信CursorとBrowser bufferを初期化し、別SeedのRecordを混ぜません。
5. 初期状態の`バッファ再生`では、受信を継続しながら人間が追える速度でReplayします。
6. `ほぼライブ`へ切り替えると、最新受信位置へ追従します。
7. `ローソク足ごと`と`イベント圧縮`を切り替え、通常Sampleまたは重要なPosition、Risk、Episode eventを観察します。
8. 同じSeedの決定論的Checkpoint評価証拠がある場合、`Checkpoint evidence`SelectorでFoldを明示選択します。
9. 選択Foldの評価Return、`checkpoint_range`、Evaluation digest、Finalist状態を探索Replayとは別枠で確認します。

一時停止中もBackendからの受信を継続します。`最新へ`で最新位置へ戻ります。Browser内Recordは選択Seedごとに最大2,048件です。

Stable-Baselines3 Callbackは通常区間を既定32 decisionごとに間引き、Position変化、Risk、Emergency deleverage、Episode終了を優先して保存します。保存EventではVector environmentからPrimary symbol、時刻、判断区間OHLCを取得します。Auto-reset後のTerminal eventでも旧Episodeの明示Market indexからOHLCを復元します。学習再開時は既存JSONLの最終Sequenceを引き継ぎます。Telemetry書込例外は可視化だけを停止し、学習自体を停止しません。

各SeedのStreamは学習中に次の場所へAppend-only JSONLとして作成されます。

```text
<run-root>/.staging/<run-id>/seed-<seed>/telemetry/training-telemetry.jsonl
```

Runが公開または失敗隔離された後は、同じRun directoryとともに`runs/`または`failed/`へ移動します。Studio APIは既知Jobと宣言済みArtifact rootを経由してのみ読み取り、Project外Path、Symlink、未知Job、Stream identityとRecord seedの不一致を拒否します。

Checkpoint比較は、維持対象Walk-forward workflowが生成した次の証拠を読み取ります。Studio自身はCheckpoint評価、Candidate ranking、Seed finalist選択、Fold間Ranking、再学習を実行しません。

```text
<run-root>/{.staging,runs,failed}/<run-id>/**/checkpoint-selection.json
```

Readerは`checkpoint_selection_v2_seed_aware`、Fold identity、評価Range、有限Score、Policy/Evaluation digest、Candidate/Finalist identity、重複、Finalist score一致を検証します。UIは最高ScoreのFoldを自動選択せず、明示選択させます。不正な証拠は推測表示せず、`artifact_invalid`としてFail closedします。

主なAPI:

```text
GET /api/studio/jobs/{job_id}/telemetry/status?seed=7
GET /api/studio/jobs/{job_id}/telemetry/events?seed=7&after_sequence=0&limit=512
GET /api/studio/jobs/{job_id}/checkpoint-evaluations
```

Live Trainingは学習中の探索を理解する画面です。BUY／SELLはWeight変化の可視化であり、取引所注文ではありません。探索Telemetryはモデル選択、Sealed evaluation、収益性証拠、Release承認、Production authorizationではありません。探索区間PnLと決定論的Checkpoint評価は異なるRangeと過程の証拠であり、どちらも本番性能を保証しません。Production statusは`NO-GO`です。

## Telemetryの信頼境界と既知制約

Telemetryは診断用で、Model selection、Run identity、Sealed evaluation、Execution promotion、Serving approval、Release、Order executionから明示的に除外します。表示が停止または遅延しても、学習Artifactの正当性をTelemetryから推測しません。

現在のBackendはBooleanを厳密に解析し、Sparse indexを使ってJSONLをPage readし、同一Seedの重複StreamをAmbiguous artifactとして拒否します。`trade_rl.telemetry`はImport Linterで標準ライブラリ専用の正式Layerとして強制されています。

一方、1つのSeed Streamには複数Vector environmentのRecordが入り、Auto-reset後も明示Episode IDはありません。現在のUIは`environment_id`やEpisode境界で系列を選択・分割せず、Buffer全体を1本のPrice/Equity/PnL系列として表示します。そのため複数EnvironmentまたはResetをまたぐ表示は、単一の連続Portfolioを証明しません。この診断系列分離は[Post-remediation architecture audit](../docs/verification/2026-07-22-post-merge-architecture-audit.md)の独立修正項目です。

## Artifact探索範囲

既定:

- Dataset: `artifacts/datasets`, `var/quickstart/dataset`
- Run store: `artifacts/research`, `var/quickstart/artifacts`
- Training config: `configs`, `examples`
- Job state: `var/studio/jobs`
- Serving registry: `var/serving`
- Paper inference snapshot: `var/studio/paper-inference.json`

環境変数`TRADE_RL_STUDIO_DATASET_ROOTS`、`TRADE_RL_STUDIO_RUN_ROOTS`、`TRADE_RL_STUDIO_CONFIG_ROOTS`、`TRADE_RL_STUDIO_JOB_ROOT`、`TRADE_RL_STUDIO_SERVING_ROOT`、`TRADE_RL_STUDIO_PAPER_SNAPSHOT`でProject配下の相対Pathへ変更できます。Project外Pathは拒否されます。

## 検証

```bash
uv run pytest -q tests/telemetry tests/integrations/test_training_telemetry.py tests/studio
npm test --prefix studio -- --run
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
```

UIから開始できるのは既存の`trade-rl train run`を使うExploratory trainingだけです。Serving MonitorはRead-onlyで、Bundle activation、取引所注文、API key入力、Live資金操作を行いません。
