from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "trade_rl" / "data"


def move(source: str, target: str) -> None:
    src = ROOT / source
    dst = ROOT / target
    if not src.is_file():
        raise RuntimeError(f"migration source missing: {source}")
    if dst.exists():
        raise RuntimeError(f"migration destination already exists: {target}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


def write(path: str, content: str) -> None:
    target = ROOT / path
    if target.exists():
        raise RuntimeError(f"new migration file already exists: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def rewrite_module(old: str, new: str) -> int:
    pattern = re.compile(rf"\b{re.escape(old)}\b")
    count = 0
    for base in (ROOT / "trade_rl", ROOT / "tests"):
        for path in sorted(base.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            replaced, changes = pattern.subn(new, text)
            if changes:
                path.write_text(replaced, encoding="utf-8")
                count += changes
    if count == 0:
        raise RuntimeError(f"expected at least one import reference for {old}")
    return count


VIEW = '''"""Immutable range-scoped views over a canonical market dataset."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset

DATASET_VIEW_SCHEMA: Final = "market_dataset_view_v1"


@dataclass(frozen=True, slots=True)
class MarketDatasetView:
    """A half-open, absolute-index view that cannot escape its assigned range."""

    dataset: MarketDataset
    start: int
    stop: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or isinstance(self.stop, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.stop, int)
            or not 0 <= self.start < self.stop <= self.dataset.n_bars
        ):
            raise ValueError("dataset view range is outside the dataset")

    @property
    def identity(self) -> str:
        return content_digest(
            {
                "dataset_id": self.dataset.dataset_id,
                "schema_version": DATASET_VIEW_SCHEMA,
                "start": self.start,
                "stop": self.stop,
            }
        )

    def subview(self, start: int, stop: int) -> MarketDatasetView:
        if not self.start <= start < stop <= self.stop:
            raise ValueError("requested subview is outside the parent range")
        return MarketDatasetView(self.dataset, start, stop)

    def materialize(self) -> MarketDataset:
        if self.stop - self.start < 3:
            raise ValueError("materialized dataset view requires at least three bars")
        kwargs: dict[str, Any] = {}
        for item in fields(MarketDataset):
            if not item.init or item.name.startswith("_"):
                continue
            value = getattr(self.dataset, item.name)
            if item.name == "dataset_id":
                kwargs[item.name] = self.identity
            elif item.name == "identity_payload_json":
                kwargs[item.name] = None
            elif isinstance(value, np.ndarray) and value.shape[:1] == (
                self.dataset.n_bars,
            ):
                kwargs[item.name] = value[self.start : self.stop]
            else:
                kwargs[item.name] = value
        return MarketDataset(**kwargs)


__all__ = ["DATASET_VIEW_SCHEMA", "MarketDatasetView"]
'''

ARTIFACTS_INIT = '''"""Canonical market-dataset artifact codec and immutable publication."""

from trade_rl.data.artifacts.codec import (
    DATASET_ARRAYS_NAME,
    DATASET_ARTIFACT_SCHEMA,
    DATASET_MANIFEST_NAME,
    DatasetArtifactFiles,
    load_dataset_files,
    verify_exact_artifact_files,
    write_market_dataset_files,
)
from trade_rl.data.artifacts.publication import (
    PublishedDatasetArtifact,
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
    publish_market_dataset_artifact,
)

__all__ = [
    "DATASET_ARRAYS_NAME",
    "DATASET_ARTIFACT_SCHEMA",
    "DATASET_MANIFEST_NAME",
    "DatasetArtifactFiles",
    "PublishedDatasetArtifact",
    "inspect_published_market_dataset_artifact",
    "load_dataset_files",
    "load_market_dataset_artifact",
    "publish_market_dataset_artifact",
    "verify_exact_artifact_files",
    "write_market_dataset_files",
]
'''

BUILD_INIT = '''"""Market-dataset build configuration and deterministic construction."""

from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.build.config import MarketDatasetBuildRequest, load_market_build_request

__all__ = ["MarketDatasetBuildRequest", "MarketDatasetBuilder", "load_market_build_request"]
'''

FEATURES_INIT = '''"""Causal market feature implementations grouped by responsibility."""

from trade_rl.data.features.core import calculate_feature_events

__all__ = ["calculate_feature_events"]
'''

DATA_INIT = '''"""Market data contracts, artifacts and validation."""

from trade_rl.data.artifacts import (
    DatasetArtifactFiles,
    PublishedDatasetArtifact,
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
    publish_market_dataset_artifact,
    write_market_dataset_files,
)
from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.data.market import MarketCalendarKind, MarketDataset

__all__ = [
    "DatasetArtifactFiles",
    "InstrumentExecutionRule",
    "MarketCalendarKind",
    "MarketDataset",
    "PublishedDatasetArtifact",
    "inspect_published_market_dataset_artifact",
    "load_market_dataset_artifact",
    "publish_market_dataset_artifact",
    "write_market_dataset_files",
]
'''


def remove_deprecated_writer() -> None:
    path = ROOT / "trade_rl/data/artifacts/publication.py"
    text = path.read_text(encoding="utf-8")
    if "import warnings\n" not in text:
        raise RuntimeError("publication source no longer has expected warnings import")
    text = text.replace("import warnings\n", "", 1)
    block = '''\n\ndef write_market_dataset_artifact(root: str | Path, dataset: MarketDataset) -> str:\n    \"\"\"Deprecated compatibility wrapper returning only the artifact digest.\"\"\"\n\n    warnings.warn(\n        \"write_market_dataset_artifact is deprecated; use \"\n        \"publish_market_dataset_artifact\",\n        DeprecationWarning,\n        stacklevel=2,\n    )\n    return publish_market_dataset_artifact(root, dataset).artifact_digest\n'''
    if text.count(block) != 1:
        raise RuntimeError("deprecated writer block did not match exactly once")
    text = text.replace(block, "", 1)
    text = text.replace('    "write_market_dataset_artifact",\n', "", 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    moves = (
        ("trade_rl/data/artifact_codec.py", "trade_rl/data/artifacts/codec.py"),
        ("trade_rl/data/artifact.py", "trade_rl/data/artifacts/publication.py"),
        ("trade_rl/data/config.py", "trade_rl/data/build/config.py"),
        ("trade_rl/data/builder.py", "trade_rl/data/build/builder.py"),
        ("trade_rl/data/features.py", "trade_rl/data/features/core.py"),
        ("trade_rl/data/cross_asset_features.py", "trade_rl/data/features/cross_asset.py"),
        ("trade_rl/data/economic_semantics.py", "trade_rl/data/features/economic.py"),
        ("trade_rl/data/multitimeframe.py", "trade_rl/data/features/multitimeframe.py"),
    )
    for source, target in moves:
        move(source, target)

    write("trade_rl/data/view.py", VIEW)
    write("trade_rl/data/artifacts/__init__.py", ARTIFACTS_INIT)
    write("trade_rl/data/build/__init__.py", BUILD_INIT)
    write("trade_rl/data/features/__init__.py", FEATURES_INIT)

    old_view = ROOT / "trade_rl/data/artifacts.py"
    if not old_view.is_file() or "class MarketDatasetView" not in old_view.read_text(encoding="utf-8"):
        raise RuntimeError("legacy data/artifacts.py did not contain expected MarketDatasetView")
    old_view.unlink()

    replacements = (
        ("trade_rl.data.artifact_codec", "trade_rl.data.artifacts.codec"),
        ("trade_rl.data.artifact", "trade_rl.data.artifacts.publication"),
        ("trade_rl.data.builder", "trade_rl.data.build.builder"),
        ("trade_rl.data.config", "trade_rl.data.build.config"),
        ("trade_rl.data.cross_asset_features", "trade_rl.data.features.cross_asset"),
        ("trade_rl.data.economic_semantics", "trade_rl.data.features.economic"),
        ("trade_rl.data.multitimeframe", "trade_rl.data.features.multitimeframe"),
    )
    for old, new in replacements:
        rewrite_module(old, new)

    remove_deprecated_writer()

    data_init = ROOT / "trade_rl/data/__init__.py"
    data_init.write_text(DATA_INIT, encoding="utf-8")

    # The test-first changes intentionally use the shorter build package export.
    market_artifact_test = ROOT / "tests/data/test_market_artifact.py"
    text = market_artifact_test.read_text(encoding="utf-8")
    text = text.replace(
        "from trade_rl.data.build.builder import MarketDatasetBuilder",
        "from trade_rl.data.build import MarketDatasetBuilder",
    )
    market_artifact_test.write_text(text, encoding="utf-8")

    # No executable import of a retired flat module may remain.
    retired = (
        "trade_rl.data.artifact_codec",
        "trade_rl.data.artifact",
        "trade_rl.data.builder",
        "trade_rl.data.config",
        "trade_rl.data.cross_asset_features",
        "trade_rl.data.economic_semantics",
        "trade_rl.data.multitimeframe",
    )
    leftovers: list[str] = []
    for base in (ROOT / "trade_rl", ROOT / "tests"):
        for path in sorted(base.rglob("*.py")):
            content = path.read_text(encoding="utf-8")
            if any(re.search(rf"\b{re.escape(name)}\b", content) for name in retired):
                leftovers.append(str(path.relative_to(ROOT)))
    if leftovers:
        raise RuntimeError(f"retired data module imports remain: {leftovers}")

    for old_name in (
        "artifact.py",
        "artifact_codec.py",
        "artifacts.py",
        "builder.py",
        "config.py",
        "features.py",
        "cross_asset_features.py",
        "economic_semantics.py",
        "multitimeframe.py",
    ):
        if (DATA / old_name).exists():
            raise RuntimeError(f"retired flat data path remains: {old_name}")

    (ROOT / "scripts/phase2a_migrate_data.py").unlink()
    (ROOT / ".github/workflows/phase2a-data-migration.yml").unlink()


if __name__ == "__main__":
    main()
