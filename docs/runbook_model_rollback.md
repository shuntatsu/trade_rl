# モデルロールバック運用手順書

## 1. 原則

モデルレジストリの`active_version`変更と、実際の推論プロセスが対象モデルをロードした状態は別である。ロールバック完了は、レジストリ、モデルファイル、サービングプロセス、API表示、注文状態が一致した時点とする。

## 2. 事前確認

```bash
python -m mars_lite.server.model_registry list
```

対象versionについて、モデルファイルの存在、SHA-256、Git SHA、設定hash、過去のShadow/Canary証拠を確認する。異常注文がある場合は先に新規リスクを停止し、必要に応じて緊急flattenを実行する。

## 3. 任意versionへのロールバック

```bash
python -m mars_lite.server.model_registry rollback \
  --target-version <verified-version>
```

直前versionへ戻すだけの場合は`--target-version`を省略できるが、インシデント対応では検証済みversionを明示することを推奨する。

## 4. サービング反映

1. 旧プロセスへの新規判断入力を停止する。
2. レジストリの対象モデルファイルhashを再検証する。
3. サービングプロセスを安全に再読込または再起動する。
4. API、ログ、メトリクスが対象versionとartifact hashを返すことを確認する。
5. 旧versionの判断と新versionの判断を同一注文ライフサイクルへ混在させない。
6. 取引所の注文・ポジションと内部台帳を再照合する。

## 5. 完了基準

- registryの`active_version`が対象version。
- 対象モデルファイルのSHA-256が登録値と一致。
- 推論プロセスとAPIが対象versionを報告。
- 新旧モデルの同時稼働または注文混在がない。
- 未約定注文とポジションが取引所と一致。
- ロールバック時刻、実行者、理由、前後version、hash、確認結果を意思決定ログへ記録。

## 6. 失敗時

対象ファイル不在、hash不一致、再読込失敗、照合不一致の場合は復旧完了とみなさない。新規リスクを停止したまま別の検証済みversionを選択するか、フラット状態を維持してインシデント対応へ移行する。
