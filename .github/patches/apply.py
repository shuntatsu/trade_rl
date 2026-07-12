from pathlib import Path


def transform(path: str, replacements: list[tuple[str, str]]) -> None:
    target = Path(path)
    raw = target.read_bytes()
    uses_crlf = b"\r\n" in raw
    text = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"expected source fragment not found in {path}: {old[:60]!r}")
        text = text.replace(old, new, 1)
    if uses_crlf:
        text = text.replace("\n", "\r\n")
    target.write_bytes(text.encode("utf-8"))


transform(
    "mars_lite/server/metrics_server.py",
    [
        (
            '"""\nメトリクスサーバーモジュール\n\nFastAPIベースのWebSocket + REST APIサーバー。\n学習メトリクスのリアルタイム配信とモデル管理を提供。\n"""\n',
            '"""Legacy development-only training dashboard server.\n\nThis module is not the authoritative Serving Plane. Production and signal-serving\nprocesses must use :mod:`mars_lite.server.signal_server` through\n``scripts/run_server.py``.\n"""\n',
        ),
        ("import json\nimport time\n", "import json\nimport os\nimport time\n"),
        (
            "from mars_lite.learning.training_callback import MetricsHistory, get_metrics_history\n\n\nclass NumpyJSONEncoder",
            'from mars_lite.learning.training_callback import MetricsHistory, get_metrics_history\n\n\ndef _require_development_opt_in(development_only: bool) -> None:\n    if development_only or os.getenv("TRADE_RL_ENABLE_LEGACY_METRICS_SERVER") == "1":\n        return\n    raise RuntimeError(\n        "legacy metrics server is development-only; set "\n        "TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1 or pass development_only=True"\n    )\n\n\nclass NumpyJSONEncoder',
        ),
        (
            'def create_app(\n    metrics_history: Optional[MetricsHistory] = None,\n    output_dir: str = "./output",\n) -> "FastAPI":\n',
            'def create_app(\n    metrics_history: Optional[MetricsHistory] = None,\n    output_dir: str = "./output",\n    *,\n    development_only: bool = False,\n) -> "FastAPI":\n',
        ),
        (
            '    if not HAS_FASTAPI:\n        raise ImportError(\n            "FastAPI is required. Install with: pip install fastapi uvicorn[standard]"\n        )\n\n    app = FastAPI(\n',
            '    _require_development_opt_in(development_only)\n    if not HAS_FASTAPI:\n        raise ImportError(\n            "FastAPI is required. Install with: pip install fastapi uvicorn[standard]"\n        )\n\n    app = FastAPI(\n',
        ),
        (
            'def run_server(\n    host: str = "0.0.0.0",\n    port: int = 8001,\n    metrics_history: Optional[MetricsHistory] = None,\n    output_dir: str = "./output",\n) -> None:\n',
            'def run_server(\n    host: str = "0.0.0.0",\n    port: int = 8001,\n    metrics_history: Optional[MetricsHistory] = None,\n    output_dir: str = "./output",\n    *,\n    development_only: bool = False,\n) -> None:\n',
        ),
        (
            "    app = create_app(metrics_history, output_dir)\n    uvicorn.run(",
            "    app = create_app(\n        metrics_history, output_dir, development_only=development_only\n    )\n    uvicorn.run(",
        ),
        (
            'async def run_server_async(\n    host: str = "0.0.0.0",\n    port: int = 8001,\n    metrics_history: Optional[MetricsHistory] = None,\n    output_dir: str = "./output",\n) -> None:\n',
            'async def run_server_async(\n    host: str = "0.0.0.0",\n    port: int = 8001,\n    metrics_history: Optional[MetricsHistory] = None,\n    output_dir: str = "./output",\n    *,\n    development_only: bool = False,\n) -> None:\n',
        ),
        (
            "    app = create_app(metrics_history, output_dir)\n    config = uvicorn.Config(",
            "    app = create_app(\n        metrics_history, output_dir, development_only=development_only\n    )\n    config = uvicorn.Config(",
        ),
        (
            '    parser.add_argument("--output-dir", default="./output", help="Output directory")\n\n    args = parser.parse_args()\n',
            '    parser.add_argument("--output-dir", default="./output", help="Output directory")\n    parser.add_argument(\n        "--development-only",\n        action="store_true",\n        help="explicitly opt in to the legacy training dashboard",\n    )\n\n    args = parser.parse_args()\n',
        ),
        (
            "        output_dir=args.output_dir,\n    )\n",
            "        output_dir=args.output_dir,\n        development_only=args.development_only,\n    )\n",
        ),
    ],
)

transform(
    "scripts/train.py",
    [
        (
            "                output_dir=str(output_dir),\n            )\n",
            "                output_dir=str(output_dir),\n                development_only=True,\n            )\n",
        )
    ],
)
