# モデルライフサイクル

[English](../MODEL_LIFECYCLE.md) | **日本語**

## 1. 学習と評価

Control Planeは1つのfeature setを構築し、可能な場合はdevelopment dataとsealed holdout dataを分離します。そのうえでP0、PBT、Walk-Forward／cost sensitivity、最終学習、baseline比較、必要に応じた統計的有意性検査を実行します。

Gateに失敗した実行から、昇格可能なmodelを生成してはいけません。`--force`は研究専用であり、deploymentを許可しません。

## 2. 候補を構築する

合格したrunは、model、順序付きfeature schema、preprocessing、observation contract、post-processing、risk設定、評価identity、Git SHA、digestを含む完全な`ServingBundle`を1つ作成します。

Production互換modelは、observation progress modeとして`zero`を使用します。Episode相対progressはオンラインで再構築できないため、Bundle検証時に拒否されます。

## 3. 登録する

```bash
uv run python scripts/manage_registry.py \
  --registry-dir <registry> register <candidate-directory>
```

登録処理:

1. Source Bundleを検証する。
2. 同階層のtemporary directoryへcopyする。
3. Copy後のBundleを再検証する。
4. `versions/<version>`へ原子的にrenameする。
5. `active.json`は変更しない。

同じversion・同じdigestの再登録は、management CLI上でidempotentです。同じversionを異なる内容に再利用する操作は拒否されます。

## 4. 証拠を生成する

ShadowおよびCanary評価は、正確なmodel version、Git SHA、bundle identity、source run、必要に応じたparent evidenceを参照しなければなりません。Production証拠には、Canary結果、incident状態、approval ticket、Environment承認を追加します。

## 5. Activation

Activationは、Gate成功後のdeployment Control Planeでのみ実行します。

```bash
uv run python scripts/manage_registry.py \
  --registry-dir <registry> activate <version> \
  --evidence-identity <trusted-identity>
```

Activationは登録済みBundleを検証し、`active.json`を原子的に置換します。Modelファイルをcopyまたはrenameしません。

## 6. Serving

`ServingRuntime`はactive identityを読み、現在load中のversionを維持したまま新Bundleを検証・loadし、readiness check成功後にだけin-memory referenceを切り替えます。

ResponseにはServing中のmodel versionとbundle digestを含め、callerがidentityを検証できるようにします。

## 7. Rollback

Rollbackは、登録済みのknown-good versionを選択し、`active.json`を原子的に変更します。Servingは同じ検証済みhot-swap pathを通じて、そのversionへ戻ります。

## Registry不変条件

- Registry実装は1つだけ
- active pointerは1つだけ
- version directoryは不変
- すべての境界でdigestを検証
- 登録とactivationは別操作
- activation失敗時は以前のpointerを維持
- Served identityはactive identityと一致
- Serving Planeはmodel lifecycle変更操作を公開しない