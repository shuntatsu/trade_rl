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
- Live Training: 市場・方策・学習信号・成績を同期表示する研究Replay
- Compare: Run、Fold、Wealth、設定差
- Evidence Explorer: Manifest、Identity、Authorization、File closure
- Serving Monitor: Active bundleとPaper inferenceのRead-only状態

UI全体は固定Workspaceを基本とし、主要画面でBrowser page全体の縦Scrollを避けます。

## Live Training

Live Trainingの市場Replayは、`lightweight-charts` 5.2.0を使ったDesktop向け研究Workspaceです。静的SVGではなく、Mouse wheelやDragによるZoom・Pan、Crosshair、複数Pane、Chart clickによるReplay位置選択を直接操作できます。TradingView attribution logoはLibraryのLicense条件に従って有効のまま保持します。

### 画面構成

上段の操作は用途ごとに分離します。

- `Run`: 常時表示する学習Job選択
- `対象を変更`: SeedとEnvironmentを一度に適用するPopover
- Replay transport: 先頭、前Event、再生・停止、次Event、最新、1x・4x・8x
- `最新へ追従`: 最新Telemetryへ自動追従するかを明示的に切替
- `表示項目`: 売買Event、Risk、Baseline、Executed weight、Reward・Cost、Drawdown
- `表示リセット`: 24H、既定Layer、最新追従へ戻す

Decorativeな`接続 / LIVE / Seed / Env`指標Cardは表示しません。SeedとEnvironmentは研究対象の選択情報としてPopoverとInspectorで確認します。

Chart直上には、Chartへ直接作用するControlだけを置きます。

- Symbol: `BTCUSDT`、`ETHUSDT`、`BNBUSDT`などTelemetryに存在するSymbol
- Timeframe: `15m`、`1h`、`4h`、`1d`
- 選択時点のOHLC
- 表示期間: `1H`、`24H`、`7D`、`全期間`

Chart直下にはRaw Replay recordを移動するScrubberを置きます。上部の表示期間はChart viewport、ScrubberはReplay cursorを変更するため、役割を混同しません。

### 同期Pane

一つのChart instanceに4つのPaneを持たせ、Time scaleとCrosshairを共有します。

1. Market: CandlestickとBUY、SELL、RISK、END Marker
2. Policy: Target weightとExecuted target
3. Learning: RewardとInterval cost
4. Performance: RL equity、Baseline equity、Drawdown

RewardとCost、EquityとDrawdownは単位が異なるため、必要なSeriesは左右の独立Price scaleを使用します。右側InspectorにはCrosshair previewを優先表示し、PointerがChart外へ出た場合はCommit済みReplay recordへ戻します。Chart clickはその時点をCommitしてReplayを停止します。

### ReplayとViewportの契約

Telemetry pollingはReplay停止中も継続します。

- HoverはPreviewだけを更新し、Replayを止めません。
- Chart click、Scrubber、前後Event移動はCursorをCommitしてReplayを停止します。
- Manual Zoom・Panは`最新へ追従`を解除します。
- 新しいTelemetry到着はSeriesだけを更新し、Manual viewportを勝手に24Hへ戻しません。
- Run、Seed、Environment、Symbol、Timeframe、表示期間、明示Resetの変更時だけViewportを再計算します。
- Programmaticな`setData`、Range preset、最新位置Scrollが発生させるRange通知は、User操作として扱いません。

### Telemetry変換

`training_telemetry_v1`はSeed scopedのAppend-only JSONLです。各Recordは`environment_id`とnullableな`episode_id`を持ちます。UIは選択EnvironmentのCurrent episodeだけからPrice、Weight、Reward、Cost、PnL、Baseline、Drawdown、Event、Cursorを作ります。

Historical records with`null` episode identityは、Terminal eventとCounter rollbackを使って分割します。異なるSeed、Environment、Episodeを1本の連続Portfolioとして混ぜません。

Timeframe aggregationは次のContractです。

- Open: Bucket内の最初の有限Open。なければ最初のClose
- High: 有限High、Open、Closeの最大
- Low: 有限Low、Open、Closeの最小
- Close: Bucket内の最後の有限Close。なければ最後のOpen
- Weight、Reward、Cost、Equity、Baseline、Drawdown: Bucket内の最後の有限値
- Inspector record: Market timeとSequence順で最後のRecord

Nanosecond timestampはMillisecondへ安全に正規化してからParseします。不正Timestampや非有限値へ架空の時刻・値を割り当てません。

BUY／SELL MarkerはTarget exposureの変化であり、取引所注文ではありません。表示中のPrimary assetは配列Index 0です。Primary weight deltaが正ならBUY、負ならSELL、0なら他Assetで`position` Eventが発生していても方向Markerを表示しません。RISKとENDは独立したEvent Markerです。

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

Frontend testは、Telemetry集約、Primary asset Marker、Event navigation、Atomic source選択、4 Pane構成、Crosshair preview、Chart click、Programmatic range通知、Manual viewport保持、Page統合を検証します。Playwright layout checkは、1440×900でReplayと学習診断がBrowser viewportからOverflowしないことを検証します。

Studioの役割は探索とEvidenceの理解です。直接取引所注文、API key入力、Live資金操作は行いません。
