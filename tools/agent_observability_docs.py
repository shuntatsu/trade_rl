from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _append_once(relative: str, marker: str, content: str) -> None:
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + content.rstrip() + "\n", encoding="utf-8")


def apply() -> None:
    for relative in (
        "examples/binance-multitimeframe/training-full.json",
        "examples/binance-multitimeframe/training-growth-optimal.json",
    ):
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        training = payload["training"]
        training["learning_rate"] = 0.00012
        training["learning_rate_schedule"] = "linear"
        training["learning_rate_final_ratio"] = 0.1
        training["tensorboard_enabled"] = True
        training["tensorboard_log_interval"] = 1
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    _append_once(
        "studio/README.md",
        "## 学習診断（TensorBoard scalar）",
        '''## 学習診断（TensorBoard scalar）

Studio の `Live Training` には `市場リプレイ` と `学習診断` の2つの表示があります。Run と Seed を選択し、`学習診断` を開くと、学習率、損失、Approx KL、Clip fraction、Explained variance、報酬、ポートフォリオ価値、ドローダウン、コスト、行動量を global step 軸で確認できます。

Studio は各 ensemble member のローカル TensorBoard event file を loopback API 経由で読みます。ブラウザは event file や TensorBoard server に直接接続せず、明示的な allowlist に含まれる有限 scalar だけを受け取ります。通常の利用では TensorBoard server の起動は不要です。

学習曲線は最適化状態の診断であり、汎化性能や収益性の証拠ではありません。モデル選択は Checkpoint 検証と Walk-forward 評価で行います。
''',
    )
    _append_once(
        "README.ja.md",
        "### 学習診断グラフ",
        '''### 学習診断グラフ

維持対象の full training 設定では TensorBoard scalar を出力できます。Studio で `Live Training → Run選択 → Seed選択 → 学習診断` と進むと、TensorBoard server を別途起動せずに学習率、PPO最適化指標、価値関数指標、取引・リスク指標を確認できます。

主な指標は `train/learning_rate`、`train/approx_kl`、`train/clip_fraction`、`train/entropy_loss`、`train/value_loss`、`train/explained_variance`、`trade_rl/drawdown_mean`、`trade_rl/interval_cost_mean` です。グラフが滑らかなことを理由にモデルやscheduleを選択してはいけません。Checkpoint検証とWalk-forward評価を固定した後に判断します。

生のTensorBoard画面をエンジニアリング診断に使う場合のみ、任意で次を実行できます。

```bash
uv run tensorboard --logdir var --host 127.0.0.1 --port 6006
```
''',
    )
    _append_once(
        "docs/operations/docker-gpu-full-training.md",
        "## Studio 学習診断の確認",
        '''## Studio 学習診断の確認

`training-full.json` と `training-growth-optimal.json` は linear learning-rate decay と TensorBoard scalar 出力を有効にしています。linear decay は候補設定であり、最適値とみなさないでください。

GPU training の開始後、Studio の `Live Training` で Run と Seed を選択し、`学習診断` を開きます。event file が作成されるまでは `未出力` と表示されます。resume 後は同じ seed/run identity の event file を統合し、global step 軸を継続します。市場リプレイの JSONL telemetry は独立して継続します。
''',
    )

    test_path = ROOT / "tests/examples/test_binance_multitimeframe_full_assets.py"
    text = test_path.read_text(encoding="utf-8")
    marker = "test_maintained_full_configs_enable_training_diagnostics"
    if marker not in text:
        text += '''\n\ndef test_maintained_full_configs_enable_training_diagnostics() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name in ("training-full.json", "training-growth-optimal.json"):
        payload = json.loads(
            (root / "examples" / "binance-multitimeframe" / name).read_text(
                encoding="utf-8"
            )
        )
        training = payload["training"]
        assert training["learning_rate_schedule"] == "linear"
        assert training["learning_rate_final_ratio"] == 0.1
        assert training["tensorboard_enabled"] is True
        assert training["tensorboard_log_interval"] == 1
'''
        test_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    apply()
