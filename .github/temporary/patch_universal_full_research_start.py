from pathlib import Path

path = Path("START.md")
text = path.read_text(encoding="utf-8")
marker = "\n## 13. Universal U3-U6 full-research trainingを実行する\n"
if marker not in text:
    text += r'''

## 13. Universal U3-U6 full-research trainingを実行する

Universal系のmaintained contractは、複数銘柄で1つのPolicyを学習し、Policy-facing symbol/actionを`INSTRUMENT`へ固定したまま、推論時は1銘柄だけを取引する構成です。U3は206 target-local market features + 9 continuous instrument descriptorsとsymbol-balanced train-only normalization、U4はsymbol-balanced Oracle BC + critic warm start、U5は4 architectureのablation + zero-shot Stage A、U6はU5選抜architectureだけをPPO / Lagrangian PPO / Discounted Lagrangian PPOへ接続します。

U6の実学習入口は次です。`--runtime-factory`は`module:function`形式のcallableで、keyword argumentsとして`algorithm`, architecture投影済み`run_config`, `context`を受け、`UniversalTrainingRuntime`を返す必要があります。`context`にはinstrument artifact root、PostgreSQL URL、dataset artifact root、train fold、normalizer/feature schema digestが含まれます。runtime factoryはU3 helper (`materialize_universal_train_datasets`, `fit_universal_shared_normalizer`, `bind_universal_normalizers`, `publish_universal_train_dataset_artifacts`)を使い、validation/test symbolを学習前処理へ混入させてはいけません。

```bash
uv sync --extra dev --extra train-sb3 --extra postgres

uv run python scripts/run_universal_full_research.py \
  --selected-architecture u_medium_direct \
  --ppo-config examples/binance-multitimeframe/training-target-weight-growth-ppo.json \
  --lagrangian-config examples/binance-multitimeframe/training-target-weight-constrained-growth.json \
  --discounted-config examples/binance-multitimeframe/training-target-weight-constrained-growth-discounted.json \
  --runtime-factory your_project.universal_runtime:build_runtime \
  --instrument-artifact-root artifacts/universal/instruments \
  --postgres-url "$TRADE_RL_POSTGRES_URL" \
  --dataset-artifact-root artifacts/universal/datasets \
  --fold-train-start 0 \
  --fold-train-stop 100000 \
  --normalizer-digest "$UNIVERSAL_NORMALIZER_DIGEST" \
  --feature-schema-digest "$UNIVERSAL_FEATURE_SCHEMA_DIGEST" \
  --baseline supervised_allocator \
  --fold 0 \
  --fold 1 \
  --output-root artifacts/universal/full-research
```

3つのauthored `TrainingRunConfig`は、training algorithm/cost-family/gammaの比較上必要な差分を除き、Environment / Risk / Reward / Trend / Action / Executionを同一にしてください。CLIは非training surfaceの差分を先に拒否し、U6本体もPPO対Lagrangianの非algorithm条件と、Lagrangian対Discountedのgamma以外の差分をfail-closedで拒否します。

学習完了時は`universal-full-research-training.json`が生成されます。この時点の`research_success=false`は意図した状態です。training完走はSoftware successであり、Research successではありません。U5のvalidation selectionと一度だけのsealed unseen-symbol test、paired baseline evidence、bootstrap lower bound、worst-symbol/worst-seed/pass-fraction gate、hard-safety violation=0がそろうまでは`NO-GO`を維持してください。

U5/U6の評価・状態遷移には既存の`StageAZeroShotEvaluationOrchestrator`、`UniversalResearchManifest`、`scripts/run_full_research_experiment.py`を使います。sealed evidenceを作る前に`zero_shot_gate_passed=true`を手動で偽装しないでください。
'''
path.write_text(text, encoding="utf-8")
