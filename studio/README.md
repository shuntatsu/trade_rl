# Trade RL Studio

ローカル優先の `trade_rl` 研究コンソールです。Vite、React、strict TypeScript と FastAPI を使い、既存の正本artifactとworkflowを操作します。

## 実装済み

- 固定トップバー、サイドバー、ワークスペース、ステータスバー
- 1536×1024／1440×900でブラウザページ全体の縦スクロールなし
- システム、dataset、job、run、baseline、fold安定性、`NO-GO`を集約したダッシュボード
- Data Labで正本dataset artifactを検証して一覧・詳細表示
- 実験画面から検証済みconfigとdatasetを選び、exploratory trainingを開始
- Run Centerで永続job状態、PID、終了コード、ログを表示し、所有プロセスを安全停止
- FastAPIによるdataset・run・config・jobの型付きAPI
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

`http://127.0.0.1:4173`を開きます。Viteは`/api`を`127.0.0.1:8765`へ転送します。

## artifact探索範囲

既定では次を探索します。

- dataset: `artifacts/datasets`, `var/quickstart/dataset`
- run store: `artifacts/research`, `var/quickstart/artifacts`
- training config: `configs`, `examples`
- job state: `var/studio/jobs`

環境変数`TRADE_RL_STUDIO_DATASET_ROOTS`、`TRADE_RL_STUDIO_RUN_ROOTS`、`TRADE_RL_STUDIO_CONFIG_ROOTS`、`TRADE_RL_STUDIO_JOB_ROOT`で、プロジェクト配下の相対パスに変更できます。プロジェクト外へのパスは拒否されます。

## 検証

```bash
pytest -q tests/studio
npm test --prefix studio -- --run
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
```

Studioは常に研究状態を`NO-GO`として表示します。UIから開始できるのは既存の`trade-rl train run`を使うexploratory trainingだけです。
