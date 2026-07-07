# モデルロールバック運用手順書 (Model Rollback Runbook)

## 1. 概要
本手順書は、`mars_lite`における予測モデルの異常検知、パフォーマンス低下、ドリフト、あるいは取引システムの動作エラーなどの緊急時に、モデルレジストリ（`model_registry` CLI）を用いて安全かつ迅速に旧バージョンのモデルにロールバックまたは特定バージョンへアクティベートするための手順を定義します。

## 2. モデルレジストリ CLI の基本
モデルレジストリはファイルベース（デフォルト: `output/model_registry/`）で管理されており、以下のPythonモジュールCLIを介して操作します。

```bash
# 基本的な使用方法
python -m mars_lite.server.model_registry <command> [args]
```

### 主要なサブコマンド
* `list`: 登録されている全モデルと現在のアクティブモデルのメタデータを表示します。
* `rollback`: 直近でアクティブだった前のバージョンへロールバックします。
* `activate <version>`: 指定されたバージョンをアクティブに切り替えます。
* `register <model_path>`: 新しいモデルファイルをレジストリに登録します。

---

## 3. 運用手順

### ステップ1: 事前検証 (状態の確認)
ロールバック操作を行う前に、現在のレジストリ内のモデル一覧およびアクティブになっているモデルバージョンを確認します。

1. 次のコマンドを実行してレジストリの状態を取得します：
   ```bash
   python -m mars_lite.server.model_registry list
   ```
2. 出力されるJSON内の `active_version` の値を確認し、現在実行中と思われるモデルのバージョンを特定します。また、`history` から直近でアクティブだったバージョンの遷移を確認します。

### ステップ2: 緊急ロールバックの実行
何らかの不具合で直近のモデルデプロイを差し戻し、1つ前の正常だったモデルに戻す場合は、`rollback` コマンドを使用します。

1. 次のコマンドを実行します：
   ```bash
   python -m mars_lite.server.model_registry rollback
   ```
2. 実行が成功すると、ロールバック先のモデル情報がJSON形式で出力されます。
3. `history` の最後から2番目のバージョンが新たな `active_version` に設定されます。
   * ※ 注意: 履歴に2つ以上のモデルが登録されていない（戻し先がない）場合、`Error: no previous active model to roll back to` というエラーになります。

### ステップ3: 特定バージョンへの直接アクティベート
以前に動作確認が取れている特定のバージョン（例: `v2-stable` など）へピンポイントで切り替えたい場合、または自動ロールバック履歴が使用できない場合は、直接バージョンを指定してアクティベートします。

1. `list` コマンド等で切り替えたい有効なバージョン名（例: `model-1719888888000`）を確認します。
2. 次のコマンドを実行してアクティベートします：
   ```bash
   python -m mars_lite.server.model_registry activate <対象のバージョン名>
   ```
   例：
   ```bash
   python -m mars_lite.server.model_registry activate model-1719888888000
   ```
3. 指定したバージョンが存在しない場合、`Error: unknown model version: ...` と表示されます。

### ステップ4: 事後確認と検証
モデルの切り替えを実行した後は、以下の確認を必ず行ってください。

1. `list` コマンドで現在のアクティブバージョンが意図通り変更されたか確認します：
   ```bash
   python -m mars_lite.server.model_registry list
   ```
2. 取引システムやサービング側のプロセスが新しい（切り替え後の）モデルファイルを正しく読み込んでいるか、ログを確認します。
3. テスト環境（Shadow環境など）において予測値の出力が正常に行われ、ドリフトやエラーが発生していないことを確認します。

---

## 4. トラブルシューティング

### 4.1 ロック競合によるタイムアウトエラー (`TimeoutError`)
**事象**: `Failed to acquire registry lock...` というエラーが表示されて操作が失敗する。
* **原因**: 他のプロセスがモデル登録やアクティベーション処理を行っているか、前回の処理が異常終了してロックファイルが残留している可能性があります。
* **対処法**: 
  1. `output/model_registry/registry.json.lock` ファイルが存在するか確認します。
  2. 他にアクティブな登録/切替プロセスが動いていないことを確認した上で、残留したロックファイルを手動で削除します。
     ```bash
     rm output/model_registry/registry.json.lock
     ```
  3. ロールバックコマンドを再実行します。

### 4.2 物理ファイルの消失エラー (`FileNotFoundError`)
**事象**: `Model physical file does not exist: ...` というエラーが表示される。
* **原因**: レジストリデータベース（`registry.json`）には情報が残っているものの、対応するモデルの実ファイル（`.pt`や`.pkl`など）が `output/model_registry/models/` 配下から削除されている。
* **対処法**: 
  1. モデルファイルが誤って別場所に移動されていないか確認します。
  2. バックアップから物理ファイルを `output/model_registry/models/` 内に復元するか、別の利用可能なバージョンを直接アクティベート（`activate`）してください。

### 4.3 履歴不足エラー (`LookupError`)
**事象**: `no previous active model to roll back to` と表示されてロールバックできない。
* **原因**: レジストリの履歴（`history`）が1件以下であり、ロールバック可能な過去バージョンが存在しない。
* **対処法**: 
  1. 新たに正常なモデルを準備し、`register` コマンドで登録します：
     ```bash
     python -m mars_lite.server.model_registry register path/to/valid_model.pt --version <新規バージョン名>
     ```
  2. 登録した新規バージョンをアクティベートします。
