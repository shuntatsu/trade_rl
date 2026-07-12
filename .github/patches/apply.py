from pathlib import Path


SECTIONS = {
    "README.md": """

## Local validation

P0 uses the release candidate's resolved `horizon` and `decision_every`. The explicit `--p0-days` option changes only the synthetic sample duration; it does not replace candidate timing.

Serving snapshots are content-addressed: the snapshot identity hashes the ordered schema, selected timestamps, feature values, global values, and close history. The CSV provider uses only a completed bar and measures age from that bar's close time for `15m`, `1h`, `4h`, and `1d` data.

Run the exchange-free local drill with:

```bash
uv run python scripts/run_local_gameday.py
```

The legacy dashboard server is development-only and requires `TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1` unless an intentional development caller opts in directly. The filesystem Registry is a single-node local implementation; passing this drill does not establish multi-node or testnet readiness.
""",
    "README.ja.md": """

## ローカル検証

P0はrelease候補で解決された`horizon`と`decision_every`をそのまま使用します。明示的な`--p0-days`は合成データ期間だけを変更し、候補のタイミング設定を置き換えません。

Serving snapshotはcontent-addressedであり、順序付きschema、選択timestamp、feature値、global値、close履歴をhash化します。CSV providerはcompleted barだけを使い、`15m`、`1h`、`4h`、`1d`についてbar closeから鮮度を計算します。

取引所へ接続しないLocal GameDay:

```bash
uv run python scripts/run_local_gameday.py
```

旧dashboard serverは開発専用で、意図的な直接opt-inを除き`TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1`が必要です。filesystem Registryはsingle-nodeのローカル実装です。この検証の成功はmulti-node、testnet、本番GOを証明しません。
""",
    "docs/ARCHITECTURE.md": """

## Local validation boundaries

The Control Plane's P0 gate preserves the candidate `horizon` and `decision_every`; `--p0-days` controls only synthetic runtime. The Serving Plane creates a content-addressed snapshot from the exact inference inputs after selecting the latest completed bar and computes staleness from bar close.

`mars_lite.server.signal_server` remains authoritative. The legacy dashboard is gated by `TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1`. The filesystem Registry is deliberately single-node and is not a distributed coordination mechanism.
""",
    "docs/ja/ARCHITECTURE.md": """

## ローカル検証の境界

Control PlaneのP0は候補の`horizon`と`decision_every`を保持し、`--p0-days`は合成データの実行期間だけを制御します。Serving Planeは最新のcompleted barを選んだ後、実際の推論入力からcontent-addressed snapshotを生成し、bar closeからstalenessを計算します。

`mars_lite.server.signal_server`が唯一の正規Servingです。旧dashboardは`TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1`で明示的に有効化する必要があります。filesystem Registryは意図的にsingle-nodeであり、分散協調機構ではありません。
""",
    "docs/OPERATIONS.md": """

## Local validation drill

Before any testnet exercise, run:

```bash
uv run python scripts/run_local_gameday.py
```

The command validates candidate-aligned P0 configuration (`--p0-days` affects only sample duration), content-addressed snapshot mutation, completed bar freshness, stale-data flattening, replay rejection, mismatched-bundle preservation, and rollback. It performs no exchange action or network request.

To launch the legacy training dashboard intentionally, set `TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1`; do not use it as the Serving Plane. The local filesystem Registry is single-node only.
""",
    "docs/ja/OPERATIONS.md": """

## ローカル検証ドリル

testnet演習より前に次を実行します。

```bash
uv run python scripts/run_local_gameday.py
```

このコマンドは、候補設定に一致するP0（`--p0-days`はsample期間のみ）、content-addressed snapshotのmutation、completed barの鮮度、stale dataのflatten、replay拒否、不一致Bundle時の健全runtime維持、rollbackを検証します。取引所操作や外部network requestは行いません。

旧training dashboardを意図的に起動する場合だけ`TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1`を設定し、Serving Planeとしては使用しません。ローカルfilesystem Registryはsingle-node専用です。
""",
    "docs/PRODUCTION_READINESS.md": """

## Local validation evidence

- [ ] Exact release head records candidate timing in P0 and uses `--p0-days` only for synthetic duration.
- [ ] Content-addressed snapshot tests prove selected-value mutation changes identity.
- [ ] Completed bar freshness tests pass for `1h`, `4h`, and `1d`.
- [ ] `uv run python scripts/run_local_gameday.py` passes all seven local scenarios.
- [ ] Legacy dashboard startup requires `TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1` or explicit development opt-in.
- [ ] Reviewers accept that the filesystem Registry remains single-node and local evidence is not multi-node or testnet evidence.
""",
    "docs/ja/PRODUCTION_READINESS.md": """

## ローカル検証証拠

- [ ] 正確なrelease headでP0が候補タイミングを記録し、`--p0-days`を合成データ期間だけに使用している。
- [ ] Content-addressed snapshot testで、選択済み値のmutationによりidentityが変化する。
- [ ] `1h`、`4h`、`1d`のcompleted bar鮮度testが合格している。
- [ ] `uv run python scripts/run_local_gameday.py`が7つのローカルscenarioすべてに合格している。
- [ ] 旧dashboard起動に`TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1`または明示的なdevelopment opt-inが必要である。
- [ ] filesystem Registryがsingle-nodeのままであり、ローカル証拠はmulti-nodeやtestnet証拠ではないとreviewerが確認している。
""",
}


for name, section in SECTIONS.items():
    path = Path(name)
    raw = path.read_bytes()
    uses_crlf = b"\r\n" in raw
    text = raw.decode("utf-8").replace("\r\n", "\n")
    marker = section.strip().splitlines()[0]
    if marker in text:
        raise RuntimeError(f"section already exists in {name}: {marker}")
    text = text.rstrip() + section + "\n"
    if uses_crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))
