# Trade RL

[English](README.md) | **日本語**

Trade RLは、複数の金融商品に対してリスク制約付きの目標ウェイトを生成する、ポートフォリオ強化学習の研究・デプロイ用コードベースです。

## 状態

**Production: NO-GO（本番投入不可）。**

このリポジトリは、オフラインのControl Planeと、認証付き・読み取り専用のServing Planeに分離されています。コードとCIが正常であるだけでは、ライブ取引は許可されません。[`docs/ja/PRODUCTION_READINESS.md`](docs/ja/PRODUCTION_READINESS.md)の未完了項目に運用証拠が添付されるまで、本番投入はブロックされます。

このリポジトリ内のリターン、Sharpe比、ベンチマーク結果、合成データ実験は、将来の収益性を保証するものではありません。

## アーキテクチャ

```text
Control Plane
  データ -> 品質ゲート -> development／sealed holdout分割
         -> 学習 -> Walk-Forward／holdout評価
         -> release eligibility判定 + 必須risk policy
         -> 不変ServingBundle候補 -> 証拠 -> Registry
         -> 承認後の原子的activation -> live identity検証

Serving Plane
  認証済み口座状態 + キャッシュ済み市場スナップショット
         -> 共通Observation Builder -> Policy -> DecisionPipeline
         -> Guardrails -> Pre-Trade Risk判定 -> 読み取り専用レスポンス
```

アーキテクチャの唯一の正典は[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)です。日本語版は[`docs/ja/ARCHITECTURE.md`](docs/ja/ARCHITECTURE.md)にあります。

## セットアップ

Python 3.12と`uv`を推奨します。

```bash
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy mars_lite
uv run pytest --cov=mars_lite --cov-fail-under=70 tests/
```

## Control Plane

release候補を生成する前に、[`config/release-risk.example.json`](config/release-risk.example.json)をコピーして編集してください。`symbol_liquidity_caps`は、実際に解決されたBundleのsymbolを過不足なく含める必要があります。

完全な検証・学習パイプラインを実行します。

```bash
uv run python scripts/run_pipeline.py \
  --source postgres \
  --git-sha "$(git rev-parse HEAD)" \
  --model-version model-YYYYMMDD-N \
  --risk-config config/release-risk.example.json
```

release可能な実行には、空でないsealed holdout、すべての必須gate合格、完全なrisk policyが必要です。`--force`、`--skip-p0`、`--skip-wf`、`--skip-gate`のいずれかを使うと研究専用実行になり、reportは生成できても候補Bundleの構築・Registry登録はできません。意図的な研究実行には`--no-register`を使用します。

Baseline-Anchored Residual policyは現在、明示的な研究専用経路として提供されています。詳細は[`docs/ja/BASELINE_RESIDUAL_RL.md`](docs/ja/BASELINE_RESIDUAL_RL.md)を参照してください。sealed multi-fold release workflowが実装・証拠化されるまで、Registry登録はfail-closedで拒否されます。

適格な実行は、不変の候補Bundleを構築してRegistryへ登録します。ただし、候補を**activationしません**。activationは、証拠検証とEnvironment承認を通過した後、デプロイworkflowだけが実行します。

Registry操作:

```bash
uv run python scripts/manage_registry.py --registry-dir output/model_registry list
uv run python scripts/manage_registry.py --registry-dir output/model_registry show-active
```

## Serving Plane

必要な環境変数:

```text
TRADE_RL_SERVING_TOKEN      必須Bearer token
TRADE_RL_RELEASE_GIT_SHA    稼働中releaseの正確な40桁Git SHA
TRADE_RL_REGISTRY_DIR       Registryディレクトリ
TRADE_RL_AUDIT_DB           SQLite監査・リプレイ防止DB
TRADE_RL_DATA_DIR           市場データディレクトリ
TRADE_RL_ALLOWED_ORIGINS    カンマ区切りの許可origin
TRADE_RL_HOST               既定値 127.0.0.1
TRADE_RL_PORT               既定値 8001
```

Servingを開始します。

```bash
export TRADE_RL_RELEASE_GIT_SHA="$(git rev-parse HEAD)"
uv run python scripts/run_server.py
```

公開route:

- `GET /health`
- `GET /ready`
- `POST /api/signal/latest`（`Authorization: Bearer ...`が必要）

`/ready`は、active model version、Bundle digest、稼働中release Git SHAを返します。Productionのstrict servingは、稼働中コードと異なるGit SHAで構築されたBundleを拒否し、直前の健全なin-memory Bundleを維持します。

Serving Planeには、学習、モデル削除、昇格、rollback、Registry変更用のrouteはありません。

## ドキュメント

- [`docs/ja/ARCHITECTURE.md`](docs/ja/ARCHITECTURE.md) — 現行システムアーキテクチャ
- [`docs/ja/BASELINE_RESIDUAL_RL.md`](docs/ja/BASELINE_RESIDUAL_RL.md) — Baseline-Anchored Residual研究経路とrelease境界
- [`docs/ja/MODEL_LIFECYCLE.md`](docs/ja/MODEL_LIFECYCLE.md) — 候補、証拠、Registry、activation、rollback
- [`docs/ja/OPERATIONS.md`](docs/ja/OPERATIONS.md) — デプロイとインシデント対応
- [`docs/ja/SECURITY.md`](docs/ja/SECURITY.md) — 信頼境界と脅威
- [`docs/ja/TESTING.md`](docs/ja/TESTING.md) — テストと受入ゲート
- [`docs/ja/PRODUCTION_READINESS.md`](docs/ja/PRODUCTION_READINESS.md) — GO／NO-GOチェックリスト
- [`docs/ja/DECISIONS.md`](docs/ja/DECISIONS.md) — アーキテクチャ上の意思決定
- [`docs/ja/RESEARCH_HISTORY.md`](docs/ja/RESEARCH_HISTORY.md) — 非正典の研究履歴

日本語ドキュメント一覧は[`docs/ja/README.md`](docs/ja/README.md)を参照してください。

承認済みの仕様書と実装計画は`docs/superpowers/`に履歴資料として保持されています。

## ローカル検証

P0はrelease候補で解決された`horizon`と`decision_every`をそのまま使用します。明示的な`--p0-days`は合成データ期間だけを変更し、候補のタイミング設定を置き換えません。

Serving snapshotはcontent-addressedであり、順序付きschema、選択timestamp、feature値、global値、close履歴をhash化します。CSV providerはcompleted barだけを使い、`15m`、`1h`、`4h`、`1d`についてbar closeから鮮度を計算します。

取引所へ接続しないLocal GameDay:

```bash
uv run python scripts/run_local_gameday.py
```

旧dashboard serverは開発専用で、意図的な直接opt-inを除き`TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1`が必要です。filesystem Registryはsingle-nodeのローカル実装です。この検証の成功はmulti-node、testnet、本番GOを証明しません。
