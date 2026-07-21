# Trade RL Studio

ローカル優先の `trade_rl` 研究コンソールです。Vite、React、strict TypeScript と FastAPI を使い、既存の正本artifactとworkflowを操作します。

## 実装済み

- 固定トップバー、サイドバー、ワークスペース、ステータスバー
- 1536×1024／1440×900でブラウザページ全体の縦スクロールなし
- システム、dataset、job、run、baseline、fold安定性、`NO-GO`を集約したダッシュボード
- Data Labで正本dataset artifactを検証して一覧・詳細表示
- 実験画面から検証済みconfigとdatasetを選び、exploratory trainingを開始
- Run Centerで永続job状態、PID、終了コード、ログを表示し、所有プロセスを安全停止
- Live Trainingで学習中の探索行動を市場リプレイとして表示
- Live Trainingの「ほぼライブ」／「バッファ再生」と「ローソク足ごと」／「イベント圧縮」の切替
- 価格チャート、ポジション変更マーカー、再生カーソル、現在weight、損益、reward、drawdown、最新イベントの同期表示
- Compareで検証済みrunの指標、設定差、fold、累積wealthを比較
- Evidence Explorerでrun manifest、identity、authorization、artifact file closureを監査
- Serving Monitorでactive bundleとpaper推論スナップショットを読み取り専用表示
- FastAPIによるdataset・run・config・job・training telemetryの型付きAPI
- API未起動時は明示的な`DEMO DATA`へフォールバック
- 直接取引所注文、APIキー入力、ライブ資金操作は未実装

## 起動

リポジトリ直下でPython APIを起動します。

```bash
uv sync --extra studio --extra train-sb3
uv run trade-rl studio start --project-root .
```

別ターミナルでReactを起動します。

```bash
npm ci --prefix studio
npm run dev --prefix studio
```

`http://127.0.0.1:5173`を開きます。Viteは`/api`を`127.0.0.1:8765`へ転送します。

## Live Training

1. `実験`ワークスペースからexploratory training jobを開始します。
2. サイドバーの`Live Training`を開きます。
3. 実行中または保存済みジョブを選択します。
4. 初期状態の`バッファ再生`では、受信を継続しながら人間が追える速度でリプレイします。
5. `ほぼライブ`へ切り替えると、最新受信位置へ追従します。
6. `ローソク足ごと`と`イベント圧縮`を切り替え、通常サンプルまたは重要なposition・risk・episodeイベントを観察します。

再生中に一時停止しても、ブラウザはバックエンドからの受信を継続します。`最新へ`を押すと最新位置へ戻ります。受信済みレコードはブラウザ内で最大2,048件に制限されます。

Stable-Baselines3の学習コールバックは、通常区間を既定32 decisionごとに間引き、position変化、risk、emergency deleverage、episode終了を優先して保存します。テレメトリ書き込みで例外が発生した場合、可視化だけを停止し、学習自体は停止しません。

各seedのストリームは、学習中は次の場所へappend-only JSON Linesとして作成されます。

```text
<run-root>/.staging/<run-id>/seed-<seed>/telemetry/training-telemetry.jsonl
```

runが公開または失敗隔離された後は、同じrunディレクトリとともに`runs/`または`failed/`配下へ移動します。Studio APIは既知のjobと宣言済みartifact rootを経由してのみ読み取り、プロジェクト外へのパス、symlink、未知のjobを拒否します。

主なAPI:

```text
GET /api/studio/jobs/{job_id}/telemetry/status
GET /api/studio/jobs/{job_id}/telemetry/events?after_sequence=0&limit=512
```

Live Trainingは学習中の探索行動を理解するための画面です。表示されたBUY／SELLはweight変化を視覚化したもので、取引所注文ではありません。決定論的なモデル評価はCheckpoint Evaluation側の責務であり、探索ロールアウトの損益を本番性能として扱いません。

## artifact探索範囲

既定では次を探索します。

- dataset: `artifacts/datasets`, `var/quickstart/dataset`
- run store: `artifacts/research`, `var/quickstart/artifacts`
- training config: `configs`, `examples`
- job state: `var/studio/jobs`
- serving registry: `var/serving`
- paper inference snapshot: `var/studio/paper-inference.json`

環境変数`TRADE_RL_STUDIO_DATASET_ROOTS`、`TRADE_RL_STUDIO_RUN_ROOTS`、`TRADE_RL_STUDIO_CONFIG_ROOTS`、`TRADE_RL_STUDIO_JOB_ROOT`、`TRADE_RL_STUDIO_SERVING_ROOT`、`TRADE_RL_STUDIO_PAPER_SNAPSHOT`で、プロジェクト配下の相対パスに変更できます。プロジェクト外へのパスは拒否されます。

## 検証

```bash
uv run pytest -q tests/telemetry tests/integrations/test_training_telemetry.py tests/studio
npm test --prefix studio -- --run
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
```

Studioは常に研究状態を`NO-GO`として表示します。UIから開始できるのは既存の`trade-rl train run`を使うexploratory trainingだけです。Serving Monitorは読み取り専用で、bundle activation、取引所注文、APIキー入力、ライブ資金操作を行いません。
