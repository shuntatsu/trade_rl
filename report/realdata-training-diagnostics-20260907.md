# Binance 実データ学習診断レポート（2026-09-07）

> 状態: **確定**。全 `1,728/1,728` replay 後、cost-aware causal alpha selection が fail-closed で reject された。RL は実行していないため、学習成功とは判定しない。

## 1. 目的

学習後の最終利益だけでなく、次の7項目を実現された exposure と cost を含む telemetry で確認する。

1. action がほぼ0へ collapse していないか
2. exposure が実際に変化しているか
3. turnover が異常に大きくないか
4. gross では勝つが cost で net が負けていないか
5. Train では学ぶが Future で崩れていないか
6. unseen symbol だけ崩れていないか
7. seed 間の結果の分散が大きすぎないか

## 2. 実行・再現情報

- checkout: `C:\dev\trade_rl`
- branch: `main`
- commit at launch: `1edf778ed948ca9b87f1493db45dbf6fb1de77a7`
- runtime root: `C:\dev\trade_rl\var\realdata-current-20260907`
- runtime manifest: `C:\dev\trade_rl\var\realdata-current-20260907\runtime-manifest.json`
- manifest digest: `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`
- normalizer stats digest: `f08efbac3759d791213167092b2f6de4ebbf5e781a2bcab8d462384d688`
- database: PostgreSQL `trade_rl_db` on `localhost:5433`, health `healthy`
- data cache: `binance-usds-m-native-indicators-15x-20241113-20260705-v1`
- shared rows: `57504`
- train symbols: `APTUSDT, ARBUSDT, BCHUSDT, BNBUSDT, BTCUSDT, LINKUSDT, LTCUSDT, SOLUSDT, XRPUSDT`
- validation symbols: `AVAXUSDT, DOGEUSDT, ETHUSDT`
- test symbols: `ADAUSDT, OPUSDT, SUIUSDT`

実行コマンド:

```powershell
$root=(Resolve-Path 'var/realdata-current-20260907').Path
$out=Join-Path $root 'full-research'
uv run python scripts/run_universal_full_research.py `
  --selected-architecture u_medium_direct `
  --ppo-config examples/binance-multitimeframe/universal-u6-ppo.json `
  --lagrangian-config examples/binance-multitimeframe/universal-u6-lagrangian.json `
  --discounted-config examples/binance-multitimeframe/universal-u6-discounted.json `
  --runtime-manifest (Join-Path $root 'runtime-manifest.json') `
  --frozen-metadata-root (Join-Path $root 'frozen-metadata-usds-m') `
  --baseline supervised_allocator `
  --fold 0 --fold 1 `
  --output-root $out `
  --verbose 1
```

## 3. 現在の進捗

2026-09-07 19:54 JST に、causal teacher selection v2 は全 `1728/1728 replay` を完了した。最終状態は `causal_teacher_selection_rejected`、理由は `no admissible cost-aware causal alpha candidate`。Admission、BC、critic warm-start、PPO/Lagrangian/discounted RL は実行されていない。

証跡:

- progress: `C:\dev\trade_rl\var\realdata-current-20260907\full-research\_shared-causal-teacher\causal-teacher-progress.json`
- checkpoint: `C:\dev\trade_rl\var\realdata-current-20260907\full-research\_shared-causal-teacher\causal-teacher-selection-checkpoint-v2.jsonl`
- monitor snapshot: `C:\dev\trade_rl\var\realdata-current-20260907\full-research\monitor-snapshot.json`

## 4. 最終診断表（完了後に更新）

| # | 診断 | 必須 evidence | 状態 |
|---|---|---|---|
| 1 | action collapse | RL action の平均・標準偏差・near-zero率・飽和率 | **未実施**（RL未到達） |
| 2 | realized exposure | requested/projected/realized weight、non-zero exposure、delta、override | **未実施**（RL未到達） |
| 3 | turnover | filled turnover の mean/p95/max/day、fill count、override | **未実施**（RL未到達） |
| 4 | gross vs cost vs net | gross return、execution cost、net return | **teacherで確認、RL未実施** |
| 5 | Train vs Future | fold/phase 別の同一指標と gate | **未実施**（RL未到達） |
| 6 | seen vs unseen symbol | train/validation/test symbol 別のRL分解 | **未実施**（RL未到達） |
| 7 | seed dispersion | seed 0/1/2 の平均、分散、worst seed、gate | **未実施**（RL未到達） |

## 5. Teacher selection の最終結果

selection gate は全候補で `mean_net_return >= 0`、`lower_tail_net_return >= -0.05`、`positive gross episode fraction >= 0.5` を要求する。全12候補がこの3条件で不採用となった。turnover は全候補で `1.0/day` 以下、hard risk violation も観測されなかった。

| candidate | mean gross | mean net | lower-tail net | turnover/day | gross陽性率 | rejection |
|---|---:|---:|---:|---:|---:|---|
| `cost-aware-baseline` | -1.296% | -2.132% | -17.568% | 0.411 | 36.8% | mean net / tail / gross |
| `horizon-24h` | -0.904% | -1.915% | -16.317% | 0.491 | 35.4% | mean net / tail / gross |
| `horizon-72h` | -1.510% | -2.428% | -17.651% | 0.450 | 35.4% | mean net / tail / gross |
| `cost-multiplier-high` | -1.250% | -2.079% | -17.576% | 0.407 | 37.5% | mean net / tail / gross |
| `edge-margin-high` | -1.253% | -2.062% | -17.544% | 0.398 | 37.5% | mean net / tail / gross |
| `confirmation-one` | -1.390% | -2.297% | -17.381% | 0.446 | 35.4% | mean net / tail / gross |
| `confirmation-three` | -1.166% | -1.964% | -17.561% | 0.391 | 37.5% | mean net / tail / gross |
| `strong-reversal-low` | -1.304% | -2.142% | -17.531% | 0.411 | 34.7% | mean net / tail / gross |
| `scale-low` | -1.480% | -2.095% | -17.197% | 0.306 | 34.7% | mean net / tail / gross |
| `exposure-low` | -0.658% | -1.212% | -11.644% | 0.267 | 36.8% | mean net / tail / gross |
| `no-trade-high` | -1.243% | -1.982% | -17.528% | 0.364 | 36.8% | mean net / tail / gross |
| `delta-low` | -1.252% | -2.026% | -17.404% | 0.381 | 37.5% | mean net / tail / gross |

### Symbol別 teacher evidence

以下は12候補×16 episodeのtrain-symbol selection replay集計であり、RL policy telemetryではない。

| symbol | mean gross | mean net | turnover/day | total cost | gross陽性率 |
|---|---:|---:|---:|---:|---:|
| `APTUSDT` | -0.267% | -0.402% | 0.069 | 28,049.13 | 39.6% |
| `ARBUSDT` | -0.153% | -0.368% | 0.105 | 42,916.88 | 33.3% |
| `BCHUSDT` | -0.441% | -0.763% | 0.156 | 63,912.01 | 40.6% |
| `BNBUSDT` | **+0.390%** | **-0.808%** | 0.555 | 224,405.45 | 42.7% |
| `BTCUSDT` | -2.297% | -3.149% | 0.420 | 163,887.22 | 25.5% |
| `LINKUSDT` | -0.299% | -1.275% | 0.473 | 189,629.93 | 52.1% |
| `LTCUSDT` | -0.723% | -1.441% | 0.352 | 140,018.91 | 34.4% |
| `SOLUSDT` | -2.845% | -4.294% | 0.735 | 285,777.87 | 33.3% |
| `XRPUSDT` | -4.396% | -5.751% | 0.677 | 263,183.99 | 25.5% |

`BNBUSDT` はgrossがプラスでもnetがマイナスであり、costが利益を消している。全体としてはgross自体も候補平均でマイナスであるため、主因はcostだけではなく、予測・方向性の経済的妥当性不足も含む。

### Checkpoint integrity

- raw checkpoint lines: `1,729`
- unique replay identities: `1,728`
- duplicate: `candidate=0c5e40cf...`, `symbol=XRPUSDT`, `episode_index=15` が同一artifactで2行
- analysis: 同一identityを1件にdeduplicateして集計
- cause: rejection catch pathが最後の `episode_metric` を含む `latest_progress` を再保存していた
- raw artifactは証跡として保持し、修正後のコードではrejection statusだけを保存してmetricを再追記しない

## 6. 判定方針

- teacher selection の途中値は最終結果として扱わない。
- selection が reject された場合、RL 7項目は「未実施」と明記し、学習成功とは呼ばない。
- selection が pass した場合だけ Admission、PPO/Lagrangian/discounted の各候補と seed 別 telemetry を比較する。
- requested action だけでなく、`executed_target`、`weights_after`、`filled_turnover` を実行 attribution の根拠にする。
- 修正を入れた場合は、原因、変更、focused test、再実行root、前後比較を追記する。

## 7. 今回の修正と検証

selection gate、reward、action semantics、realized exposureの契約は変更していない。fail-closed時のprogress payloadから `episode_metric` だけを除外し、最後のreplayを二重にcheckpointへ追記しないようにした。

- changed production file: `C:\dev\trade_rl\trade_rl\workflows\universal_causal_alpha_teacher.py`
- regression test: `test_rejection_progress_does_not_replay_last_episode_metric`
- focused tests: `29 passed`
- Ruff: `All checks passed!`
- mypy: `Success: no issues found in 1 source file`
- current run outcome: selection rejected; fix was applied after preserving the raw run evidence, so this run was not silently rerun or reclassified

## 8. 環境整理

学習・DB稼働中のコンテナと現行DB volumeは保持した。停止済みコンテナ19個、旧トレーニングイメージ14個、Docker build cache約96GB、現行経路から参照されない `postgres:18.3` と `alpine:3.20` を削除した。DB volumeは削除していない。

## 9. 既存の関連レポート

- `C:\dev\trade_rl\report\gpt-causal-alpha-r8-v13-consolidated.md` は過去ランの selection rejected／RL未実行を扱う。今回の実行結果ではない。
