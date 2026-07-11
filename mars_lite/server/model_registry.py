"""File-backed model registry with activation and rollback support."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelEntry:
    version: str
    model_path: str
    metrics: dict[str, float]
    created_at: float


class ModelRegistry:
    """Persist model versions and active-model history under one directory."""

    def __init__(self, registry_dir: str | Path = "output/model_registry") -> None:
        self.registry_dir = Path(registry_dir)
        self.models_dir = self.registry_dir / "models"
        self.index_path = self.registry_dir / "registry.json"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.Lock()
        if not self.index_path.exists():
            self._save({"models": [], "active_version": None, "history": []})

    @contextlib.contextmanager
    def _lock(self):
        lock_path = self.registry_dir / "registry.json.lock"
        with self._thread_lock:
            start_time = time.time()
            acquired = False
            while time.time() - start_time < 5.0:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    acquired = True
                    break
                except FileExistsError:
                    time.sleep(0.05)
            if not acquired:
                raise TimeoutError(
                    f"Failed to acquire registry lock at {lock_path} within 5 seconds."
                )
            try:
                yield
            finally:
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass

    def register(
        self,
        model_path: str | Path,
        metrics: dict[str, float] | None = None,
        version: str | None = None,
    ) -> ModelEntry:
        source = Path(model_path)
        if not source.exists():
            raise FileNotFoundError(source)

        if metrics is not None:
            if not isinstance(metrics, dict):
                raise ValueError("metrics must be a dictionary")
            for k, v in metrics.items():
                if not isinstance(k, str):
                    raise ValueError("metrics keys must be strings")
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ValueError(
                        f"metrics values must be numeric, got {type(v)} for key '{k}'"
                    )

        with self._lock():
            data = self._load()

            if version is not None:
                if not re.match(r"^[a-zA-Z0-9_\-]+$", version):
                    raise ValueError(
                        f"Invalid version format: '{version}'. Only alphanumeric, hyphen, and underscore are allowed."
                    )
                if any(item["version"] == version for item in data["models"]):
                    raise ValueError(
                        f"Version '{version}' already exists in the registry."
                    )
            else:
                timestamp = int(time.time() * 1000)
                while True:
                    candidate = f"model-{timestamp}"
                    if not any(item["version"] == candidate for item in data["models"]):
                        version = candidate
                        break
                    timestamp += 1

            # SeedEnsemble.save()はディレクトリ(seed_*.zipを内包)を書くため、
            # source.suffixは空文字（拡張子なしの単一ファイルと区別できない）。
            # ディレクトリソースはそのままディレクトリとして登録する。
            target = self.models_dir / f"{version}{source.suffix}"

            # Path traversal validation
            resolved_target = target.resolve()
            resolved_models_dir = self.models_dir.resolve()
            try:
                resolved_target.relative_to(resolved_models_dir)
            except ValueError:
                raise ValueError(
                    f"Path traversal detected: version '{version}' points outside model directory."
                )

            try:
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
                entry = ModelEntry(
                    version=version,
                    model_path=str(target),
                    metrics=metrics or {},
                    created_at=time.time(),
                )

                data["models"].append(asdict(entry))
                self._set_active(data, version)
                self._save(data)
                return entry
            except Exception:
                if target.exists():
                    try:
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    except Exception:
                        pass
                raise

    def list_models(self) -> list[ModelEntry]:
        return [ModelEntry(**item) for item in self._load()["models"]]

    def get_active(self) -> ModelEntry:
        active_version = self._load()["active_version"]
        if active_version is None:
            raise LookupError("no active model")
        return self._entry_for(active_version)

    def activate(self, version: str) -> ModelEntry:
        with self._lock():
            entry = self._entry_for(version)
            data = self._load()
            self._set_active(data, version)
            self._save(data)
            return entry

    def rollback(self, target_version: str | None = None) -> ModelEntry:
        if target_version is not None:
            return self.activate(target_version)
        with self._lock():
            data = self._load()
            history = data["history"]
            if len(history) < 2:
                raise LookupError("no previous active model to roll back to")
            target_version_auto = history[-2]
            entry = self._entry_for(target_version_auto)

            history.pop()
            data["active_version"] = target_version_auto
            self._save(data)
            return entry

    def _entry_for(self, version: str) -> ModelEntry:
        for item in self._load()["models"]:
            if item["version"] == version:
                model_path = Path(item["model_path"])
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Model physical file does not exist: {model_path}"
                    )
                return ModelEntry(**item)
        raise KeyError(f"unknown model version: {version}")

    def _load(self) -> dict[str, Any]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self, data: dict[str, Any]) -> None:
        temp_path = self.index_path.with_suffix(".tmp")
        try:
            temp_path.write_text(
                json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
            )
            for attempt in range(5):
                try:
                    temp_path.replace(self.index_path)
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(0.1)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    @staticmethod
    def _set_active(data: dict[str, Any], version: str) -> None:
        data["active_version"] = version
        data["history"].append(version)
        if len(data["history"]) > 100:
            data["history"] = data["history"][-100:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage registered model versions.")
    parser.add_argument("--registry-dir", default="output/model_registry")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("model_path")
    register_parser.add_argument("--version")
    register_parser.add_argument("--metrics")

    subparsers.add_parser("list")
    activate_parser = subparsers.add_parser("activate")
    activate_parser.add_argument("version")
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument(
        "--target-version", default=None, help="Specific version to rollback to"
    )

    try:
        args = parser.parse_args(argv)
        registry = ModelRegistry(args.registry_dir)
        if args.command == "register":
            metrics = None
            if args.metrics:
                try:
                    metrics = json.loads(args.metrics)
                    if not isinstance(metrics, dict):
                        raise ValueError("metrics must be a JSON object")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON for metrics: {e}")
            print(
                json.dumps(
                    asdict(
                        registry.register(
                            args.model_path, metrics=metrics, version=args.version
                        )
                    )
                )
            )
        elif args.command == "list":
            print(json.dumps([asdict(entry) for entry in registry.list_models()]))
        elif args.command == "activate":
            print(json.dumps(asdict(registry.activate(args.version))))
        elif args.command == "rollback":
            print(
                json.dumps(
                    asdict(
                        registry.rollback(
                            getattr(args, "target_version", None)
                        )
                    )
                )
            )
        return 0
    except (ValueError, FileNotFoundError, KeyError, LookupError) as exc:
        import sys

        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
