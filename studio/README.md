# Trade RL Studio

Trade RL Studioは、学習Job、Artifact、探索Telemetry、TensorBoard診断、Evaluation evidence、Read-only Serving状態を確認するローカル研究画面です。Vite、React、Strict TypeScript、FastAPIを使用します。

> Studioは取引画面ではありません。Checkpoint選択、Sealed-testの開封、Release承認、Bundle activation、取引所注文を実行しません。Production statusは`NO-GO`です。

## 起動

Repository rootでAPIを起動します。

```bash
uv sync --extra studio --extra train-sb3
uv run trade-rl studio start --project-root .
```

別ターミナル:

```bash
npm ci --prefix studio
npm run dev --prefix studio
```

`http://127.0.0.1:5173`を開きます。Viteは`/api`を`127.0.0.1:8765`へ転送します。

## 主な画面

- Dashboard: System、Dataset、Job、Run、Gate状態
- Data Lab: 検証済みDataset artifact
- Experiments: ConfigとDatasetを選んだExploratory training
- Run Center: Job、PID、Exit code、Log、Artifact
- Live Training: 市場Replayと学習診断
- Compare: Run、Fold、Wealth、設定差
- Evidence Explorer: Manifest、Identity、Authorization、File closure
- Serving Monitor: Active bundleとPaper inferenceのRead-only状態

UI全体は固定Workspaceを基本とし、主要画面でBrowser page全体の縦Scrollを避けます。

## Live Training

1. ExperimentsからExploratory jobを開始
2. Live TrainingでJobとSeedを選択
3. 複数Vector environmentがある場合はEnvironmentを選択
4. Producer-issued`episode_id`がある場合はCurrent episodeを選択
5. `バッファ再生`または`ほぼライブ`で表示
6. `ローソク足ごと`または`イベント圧縮`を選択

BUY／SELL MarkerはTarget exposureの変化であり、取引所注文ではありません。

`training_telemetry_v1`はSeed scopedのAppend-only JSONLです。各Recordは`environment_id`とnullableな`episode_id`を持ちます。UIは選択EnvironmentのCurrent episodeだけからPrice、PnL、Baseline、Drawdown、Event、Cursorを作ります。

Historical records with`null` episode identityは、Terminal eventとCounter rollbackを使って分割します。異なるSeed、Environment、Episodeを1本の連続Portfolioとして混ぜません。

Telemetry書込失敗はVisualizationを停止できますが、学習自体を停止しません。

## 学習診断

TensorBoard event fileからAllowlist済みの有限ScalarをLoopback API経由で読みます。BrowserはEvent fileやTensorBoard serverへ直接接続しません。

表示例:

- Learning rate、Loss、Approx KL、Clip fraction
- Explained variance、Reward、Portfolio value、Drawdown
- Cost、Action magnitude
- Clock別Attention share、Missing ratio
- Timeframe/Asset Attention entropy
- Gate平均と飽和率
- Sequence BlockのGradient norm

学習曲線とAttentionは最適化診断です。モデル選択、Checkpoint選択、Sealed評価、汎化性能、収益性のEvidenceではありません。

## Checkpoint evidence

Studioは維持対象Workflowが生成した`checkpoint-selection.json`をRead-only表示します。Foldを自動的に最高Scoreへ切り替えず、利用者が明示選択します。

Schema、Fold identity、Range、Score、Policy/Evaluation digest、Finalist identityが不正な場合は、推測表示せず`artifact_invalid`として拒否します。

## Trust boundary

Studio APIは、既知Jobと宣言済みArtifact rootの下だけを読みます。

拒否するもの:

- Project root escape
- Symlink
- Unknown job
- Seed identity不一致
- 同じSeedを名乗る複数Stream
- 不正なManifestやDigest

TelemetryとTensorBoardは、Fitting、Selection、Sealed evaluation、Run identity、Release、Serving activation、Order executionから除外します。

## Artifact探索範囲

既定:

```text
Dataset: artifacts/datasets, var/quickstart/dataset
Run store: artifacts/research, var/quickstart/artifacts
Training config: configs, examples
Job state: var/studio/jobs
Serving registry: var/serving
Paper snapshot: var/studio/paper-inference.json
```

環境変数でProject配下の相対Pathへ変更できます。Project外Pathは拒否します。

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

Studioの役割は探索とEvidenceの理解です。直接取引所注文、API key入力、Live資金操作は行いません。
