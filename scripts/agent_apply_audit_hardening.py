from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    replace_once(
        "trade_rl/simulation/execution.py",
        '''        if not 0.0 <= self.tail_slippage_probability <= 1.0:\n            raise ValueError("tail_slippage_probability must be within [0, 1]")\n        if not 0.0 < self.max_leverage:\n''',
        '''        if not 0.0 <= self.tail_slippage_probability <= 1.0:\n            raise ValueError("tail_slippage_probability must be within [0, 1]")\n        if (\n            self.tail_slippage_probability > 0.0\n            and self.tail_slippage_multiplier < 1.0\n        ):\n            raise ValueError(\n                "tail_slippage_multiplier must be at least 1.0 when tail events are enabled"\n            )\n        if not 0.0 < self.max_leverage:\n''',
    )
    replace_once(
        "trade_rl/simulation/execution.py",
        '''    def execution_policy_payload(self) -> dict[str, object]:\n        return {\n            "allow_short": self.allow_short,\n            "limit_offset_rate": self.limit_offset_rate,\n            "max_leverage": self.max_leverage,\n            "max_participation_rate": self.max_participation_rate,\n            "order_latency_bars": self.order_latency_bars,\n            "order_type": self.order_type,\n            "partial_fill_carry": self.partial_fill_carry,\n            "path_mode": self.path_mode,\n            "processing_bar_volume_capacity": self.processing_bar_volume_capacity,\n            "schema_version": "execution_policy_v1",\n            "trigger_volume_fractions": list(self.trigger_volume_fractions),\n        }\n''',
        '''    def execution_policy_payload(self) -> dict[str, object]:\n        return {\n            "allow_short": self.allow_short,\n            "borrow_rate_multiplier": self.borrow_rate_multiplier,\n            "collateral_haircut": self.collateral_haircut,\n            "fee_rate": self.fee_rate,\n            "impact_rate": self.impact_rate,\n            "limit_offset_rate": self.limit_offset_rate,\n            "lot_size": self.lot_size,\n            "maintenance_margin_rate": self.maintenance_margin_rate,\n            "maker_fee_rate": self.maker_fee_rate,\n            "margin_mode": self.margin_mode,\n            "max_leverage": self.max_leverage,\n            "max_participation_rate": self.max_participation_rate,\n            "minimum_notional": self.minimum_notional,\n            "multiplier": self.multiplier,\n            "order_latency_bars": self.order_latency_bars,\n            "order_type": self.order_type,\n            "partial_fill_carry": self.partial_fill_carry,\n            "path_mode": self.path_mode,\n            "processing_bar_volume_capacity": self.processing_bar_volume_capacity,\n            "random_seed": self.random_seed,\n            "schema_version": "execution_policy_v2",\n            "slippage_std": self.slippage_std,\n            "spread_rate": self.spread_rate,\n            "tail_slippage_multiplier": self.tail_slippage_multiplier,\n            "tail_slippage_probability": self.tail_slippage_probability,\n            "taker_fee_rate": self.taker_fee_rate,\n            "tick_size": self.tick_size,\n            "trigger_volume_fractions": list(self.trigger_volume_fractions),\n        }\n''',
    )
    replace_once(
        "trade_rl/simulation/execution.py",
        '''    @property\n    def rate_per_turnover(self) -> float:\n        return self.multiplier * (self.fee_rate + self.spread_rate)\n\n''',
        "",
    )
    replace_once(
        "trade_rl/simulation/execution.py",
        '''        self.dataset = dataset\n        self.cost = cost or ExecutionCostConfig()\n        self.rule_stress = rule_stress or ExecutionRuleStress()\n''',
        '''        self.dataset = dataset\n        self.cost = cost or ExecutionCostConfig()\n        if self.cost.margin_mode == "isolated" and self.dataset.n_symbols != 1:\n            raise ValueError(\n                "isolated margin requires a per-symbol collateral ledger; "\n                "only single-asset execution is supported"\n            )\n        self.rule_stress = rule_stress or ExecutionRuleStress()\n''',
    )
    replace_once(
        "trade_rl/simulation/execution.py",
        '''    def _capacity_notional(\n        self,\n        prices: np.ndarray,\n        capacity_volume: np.ndarray,\n    ) -> np.ndarray:\n        result = np.empty_like(prices, dtype=np.float64)\n        for index, unit in enumerate(self.dataset.volume_units):\n            resolved = VolumeUnit(unit)\n            if resolved is VolumeUnit.QUOTE_NOTIONAL:\n                result[index] = capacity_volume[index]\n            else:\n                result[index] = prices[index] * capacity_volume[index]\n        return result\n\n''',
        "",
    )
    replace_once(
        "trade_rl/simulation/execution.py",
        "from trade_rl.data.contracts import VolumeUnit\n",
        "",
    )

    replace_once(
        "trade_rl/simulation/execution_promotion.py",
        'EXECUTION_EVIDENCE_SCHEMA = "execution_promotion_evidence_v1"',
        'EXECUTION_EVIDENCE_SCHEMA = "execution_promotion_evidence_v2"',
    )
    replace_once(
        "trade_rl/simulation/execution_promotion.py",
        '''    if not evidence.complete_order_evidence:\n        raise ExecutionPromotionError(\n            "execution promotion requires complete order evidence"\n        )\n''',
        '''    if evidence.order_event_count <= 0:\n        raise ExecutionPromotionError(\n            "execution promotion requires at least one order event"\n        )\n    if not evidence.complete_order_evidence:\n        raise ExecutionPromotionError(\n            "execution promotion requires complete order evidence"\n        )\n''',
    )

    replace_once(
        "trade_rl/artifacts/store.py",
        "import shutil\n",
        "import shutil\nimport uuid\n",
    )
    replace_once(
        "trade_rl/artifacts/store.py",
        '''def _atomic_write(path: Path, payload: bytes) -> None:\n    temporary = path.with_name(f".{path.name}.tmp")\n    with temporary.open("wb") as handle:\n        handle.write(payload)\n        handle.flush()\n        os.fsync(handle.fileno())\n    os.replace(temporary, path)\n    _fsync_directory(path.parent)\n''',
        '''def _atomic_write(path: Path, payload: bytes) -> None:\n    temporary = path.with_name(\n        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"\n    )\n    try:\n        with temporary.open("xb") as handle:\n            handle.write(payload)\n            handle.flush()\n            os.fsync(handle.fileno())\n        os.replace(temporary, path)\n        _fsync_directory(path.parent)\n    finally:\n        temporary.unlink(missing_ok=True)\n''',
    )
    replace_once(
        "trade_rl/artifacts/store.py",
        '''        _atomic_write(self.root / "latest.json", canonical_json_bytes(pointer))\n        return published\n''',
        '''        try:\n            _atomic_write(self.root / "latest.json", canonical_json_bytes(pointer))\n        except BaseException:\n            if published.is_dir() and not stage.exists():\n                _replace_directory(published, stage)\n                _fsync_directory(self.staging_root)\n                _fsync_directory(self.runs_root)\n            raise\n        return published\n''',
    )

    replace_once(
        "trade_rl/rl/checkpointing.py",
        '''import hashlib\nimport json\nimport shutil\nimport uuid\nfrom dataclasses import asdict, dataclass\nfrom pathlib import Path\nfrom typing import Any, Protocol\n''',
        '''import hashlib\nimport json\nimport os\nimport shutil\nimport stat\nimport tempfile\nimport uuid\nfrom contextlib import contextmanager\nfrom dataclasses import asdict, dataclass\nfrom pathlib import Path\nfrom typing import Any, BinaryIO, Iterator, Protocol\n''',
    )
    replace_once(
        "trade_rl/rl/checkpointing.py",
        '''def _file_digest(path: Path) -> str:\n    digest = hashlib.sha256()\n    with path.open("rb") as handle:\n        for chunk in iter(lambda: handle.read(1024 * 1024), b""):\n            digest.update(chunk)\n    return digest.hexdigest()\n''',
        '''@contextmanager\ndef _open_regular_binary(path: Path, *, field: str) -> Iterator[BinaryIO]:\n    path = Path(path)\n    if path.is_symlink():\n        raise ValueError(f"{field} must not be a symlink")\n    descriptor = os.open(\n        path,\n        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),\n    )\n    try:\n        if not stat.S_ISREG(os.fstat(descriptor).st_mode):\n            raise ValueError(f"{field} must be a regular file")\n        with os.fdopen(descriptor, "rb", closefd=True) as handle:\n            descriptor = -1\n            yield handle\n    finally:\n        if descriptor >= 0:\n            os.close(descriptor)\n\n\ndef _file_digest(path: Path, *, field: str = "checkpoint file") -> str:\n    digest = hashlib.sha256()\n    with _open_regular_binary(path, field=field) as handle:\n        for chunk in iter(lambda: handle.read(1024 * 1024), b""):\n            digest.update(chunk)\n    return digest.hexdigest()\n''',
    )
    replace_once(
        "trade_rl/rl/checkpointing.py",
        '''    if not path.is_file():\n        raise FileNotFoundError(f"checkpoint manifest is missing: {path}")\n    raw = json.loads(path.read_text(encoding="utf-8"))\n''',
        '''    if not path.exists():\n        raise FileNotFoundError(f"checkpoint manifest is missing: {path}")\n    with _open_regular_binary(path, field="checkpoint manifest") as handle:\n        raw = json.loads(handle.read().decode("utf-8"))\n''',
    )
    replace_once(
        "trade_rl/rl/checkpointing.py",
        '''    if not manifest.policy_path.is_file():\n        raise FileNotFoundError(f"checkpoint policy is missing: {manifest.policy_path}")\n    if _file_digest(manifest.policy_path) != manifest.policy_digest:\n        raise ValueError("checkpoint policy digest mismatch")\n    return manifest\n''',
        '''    if not manifest.policy_path.exists():\n        raise FileNotFoundError(f"checkpoint policy is missing: {manifest.policy_path}")\n    if (\n        _file_digest(manifest.policy_path, field="checkpoint policy")\n        != manifest.policy_digest\n    ):\n        raise ValueError("checkpoint policy digest mismatch")\n    return manifest\n\n\n@contextmanager\ndef verified_checkpoint_policy_copy(\n    manifest: CheckpointManifest,\n) -> Iterator[Path]:\n    """Yield a private immutable copy verified immediately before deserialization."""\n\n    with tempfile.TemporaryDirectory(prefix="trade-rl-checkpoint-") as temporary:\n        target = Path(temporary) / CHECKPOINT_POLICY_NAME\n        with (\n            _open_regular_binary(manifest.policy_path, field="checkpoint policy") as source,\n            target.open("xb") as destination,\n        ):\n            shutil.copyfileobj(source, destination)\n            destination.flush()\n            os.fsync(destination.fileno())\n        if _file_digest(target) != manifest.policy_digest:\n            raise ValueError("checkpoint policy changed during verified copy")\n        yield target\n''',
    )
    replace_once(
        "trade_rl/rl/checkpointing.py",
        '''    "save_policy_without_runtime_state",\n    "validate_checkpoint_algorithm_identity",\n]''',
        '''    "save_policy_without_runtime_state",\n    "validate_checkpoint_algorithm_identity",\n    "verified_checkpoint_policy_copy",\n]''',
    )

    replace_once(
        "trade_rl/integrations/sb3_checkpoint_assembly.py",
        '''    load_checkpoint_manifest,\n    validate_checkpoint_algorithm_identity,\n)\n''',
        '''    load_checkpoint_manifest,\n    validate_checkpoint_algorithm_identity,\n    verified_checkpoint_policy_copy,\n)\n''',
    )
    replace_once(
        "trade_rl/integrations/sb3_checkpoint_assembly.py",
        '''    loader: Any = _checkpoint_loader(algorithm_config)\n    model: Any = loader.load(\n        str(manifest.policy_path),\n        env=environment,\n        device=config.device,\n    )\n''',
        '''    loader: Any = _checkpoint_loader(algorithm_config)\n    with verified_checkpoint_policy_copy(manifest) as verified_policy_path:\n        model: Any = loader.load(\n            str(verified_policy_path),\n            env=environment,\n            device=config.device,\n        )\n''',
    )

    replace_once(
        "trade_rl/rl/replay.py",
        '''import hashlib\nimport json\nimport os\nimport shutil\nfrom dataclasses import asdict, dataclass\nfrom pathlib import Path\n''',
        '''import hashlib\nimport json\nimport os\nimport shutil\nimport stat\nimport tempfile\nfrom contextlib import contextmanager\nfrom dataclasses import asdict, dataclass\nfrom pathlib import Path\nfrom typing import BinaryIO, Iterator\n''',
    )
    replace_once(
        "trade_rl/rl/replay.py",
        '''def _digest(path: Path) -> str:\n    value = hashlib.sha256()\n    with path.open("rb") as handle:\n        for chunk in iter(lambda: handle.read(1024 * 1024), b""):\n            value.update(chunk)\n    return value.hexdigest()\n''',
        '''@contextmanager\ndef _open_regular_binary(path: Path, *, field: str) -> Iterator[BinaryIO]:\n    path = Path(path)\n    if path.is_symlink():\n        raise ValueError(f"{field} must not be a symlink")\n    descriptor = os.open(\n        path,\n        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),\n    )\n    try:\n        if not stat.S_ISREG(os.fstat(descriptor).st_mode):\n            raise ValueError(f"{field} must be a regular file")\n        with os.fdopen(descriptor, "rb", closefd=True) as handle:\n            descriptor = -1\n            yield handle\n    finally:\n        if descriptor >= 0:\n            os.close(descriptor)\n\n\ndef _digest(path: Path, *, field: str = "replay buffer") -> str:\n    value = hashlib.sha256()\n    with _open_regular_binary(path, field=field) as handle:\n        for chunk in iter(lambda: handle.read(1024 * 1024), b""):\n            value.update(chunk)\n    return value.hexdigest()\n''',
    )
    replace_once(
        "trade_rl/rl/replay.py",
        '''    raw = json.loads((path / REPLAY_MANIFEST).read_text(encoding="utf-8"))\n    manifest = ReplayBufferManifest(**raw)\n''',
        '''    with _open_regular_binary(\n        path / REPLAY_MANIFEST,\n        field="replay manifest",\n    ) as handle:\n        raw = json.loads(handle.read().decode("utf-8"))\n    manifest = ReplayBufferManifest(**raw)\n''',
    )
    replace_once(
        "trade_rl/rl/replay.py",
        '''    if _digest(replay) != manifest.replay_digest:\n        raise ValueError("replay buffer digest mismatch")\n    return manifest, replay\n''',
        '''    if _digest(replay) != manifest.replay_digest:\n        raise ValueError("replay buffer digest mismatch")\n    return manifest, replay\n\n\n@contextmanager\ndef verified_replay_buffer_copy(\n    manifest: ReplayBufferManifest,\n    replay: Path,\n) -> Iterator[Path]:\n    """Yield a private verified replay copy for unsafe pickle deserialization."""\n\n    with tempfile.TemporaryDirectory(prefix="trade-rl-replay-") as temporary:\n        target = Path(temporary) / REPLAY_FILE\n        with (\n            _open_regular_binary(replay, field="replay buffer") as source,\n            target.open("xb") as destination,\n        ):\n            shutil.copyfileobj(source, destination)\n            destination.flush()\n            os.fsync(destination.fileno())\n        if target.stat().st_size != manifest.size_bytes:\n            raise ValueError("replay buffer changed during verified copy")\n        if _digest(target) != manifest.replay_digest:\n            raise ValueError("replay buffer changed during verified copy")\n        yield target\n''',
    )
    replace_once(
        "trade_rl/rl/replay.py",
        '''    "load_replay_buffer_artifact",\n    "write_replay_buffer_artifact",\n]''',
        '''    "load_replay_buffer_artifact",\n    "verified_replay_buffer_copy",\n    "write_replay_buffer_artifact",\n]''',
    )

    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''from trade_rl.rl.replay import (\n    load_replay_buffer_artifact,\n    write_replay_buffer_artifact,\n)\n''',
        '''from trade_rl.rl.replay import (\n    load_replay_buffer_artifact,\n    verified_replay_buffer_copy,\n    write_replay_buffer_artifact,\n)\n''',
    )
    replace_once(
        "trade_rl/integrations/sb3_training.py",
        '''                model.load_replay_buffer(str(resume_path))\n''',
        '''                with verified_replay_buffer_copy(\n                    replay_manifest,\n                    resume_path,\n                ) as verified_resume_path:\n                    model.load_replay_buffer(str(verified_resume_path))\n''',
    )

    replace_once(
        "trade_rl/workflows/training_run.py",
        '''    resolved_run_id = run_id or resolved_created_at.strftime("run-%Y%m%dT%H%M%SZ")\n''',
        '''    resolved_run_id = run_id or resolved_created_at.strftime(\n        "run-%Y%m%dT%H%M%S.%fZ"\n    )\n''',
    )

    replace_once(
        "tests/evaluation/test_execution_promotion.py",
        '''def test_execution_evidence_uses_the_canonical_cost_policy_digest() -> None:\n    from trade_rl.simulation.execution import ExecutionCostConfig\n    from trade_rl.simulation.execution_promotion import execution_evidence_from_cost\n    from trade_rl.simulation.orders import execution_policy_digest\n\n    cost = ExecutionCostConfig(path_mode="conservative")\n    evidence = execution_evidence_from_cost(dataset_id="d" * 64, cost=cost)\n    expected = execution_policy_digest(\n        {\n            "allow_short": cost.allow_short,\n            "limit_offset_rate": cost.limit_offset_rate,\n            "max_leverage": cost.max_leverage,\n            "max_participation_rate": cost.max_participation_rate,\n            "order_latency_bars": cost.order_latency_bars,\n            "order_type": cost.order_type,\n            "partial_fill_carry": cost.partial_fill_carry,\n            "path_mode": cost.path_mode,\n            "processing_bar_volume_capacity": cost.processing_bar_volume_capacity,\n            "schema_version": "execution_policy_v1",\n            "trigger_volume_fractions": list(cost.trigger_volume_fractions),\n        }\n    )\n    assert evidence.execution_policy_digest == expected\n''',
        '''def test_execution_evidence_uses_the_complete_cost_policy_digest() -> None:\n    from trade_rl.simulation.execution import ExecutionCostConfig\n    from trade_rl.simulation.execution_promotion import execution_evidence_from_cost\n\n    cost = ExecutionCostConfig(path_mode="conservative")\n    evidence = execution_evidence_from_cost(dataset_id="d" * 64, cost=cost)\n\n    assert evidence.execution_policy_digest == cost.execution_policy_digest\n    assert evidence.execution_policy_digest != replace(\n        cost,\n        fee_rate=cost.fee_rate + 0.001,\n    ).execution_policy_digest\n''',
    )
    replace_once(
        "tests/simulation/test_critical_branch_coverage.py",
        "from trade_rl.data.contracts import VolumeUnit\n",
        "",
    )
    replace_once(
        "tests/simulation/test_critical_branch_coverage.py",
        '''    assert base._capacity_notional(np.array([100.0]), np.array([2.0])).tolist() == [\n        200.0\n    ]\n\n    quote = MarketExecutor(\n        market(volume_units=(VolumeUnit.QUOTE_NOTIONAL,)),\n        ExecutionCostConfig.zero(),\n    )\n    assert quote._capacity_notional(np.array([100.0]), np.array([2.0])).tolist() == [\n        2.0\n    ]\n\n''',
        "",
    )

    replace_once(
        "docs/superpowers/specs/2026-07-30-audit-hardening-design.md",
        '''`margin_mode="isolated"` is rejected until a per-symbol collateral ledger exists. The current proportional allocation of account-wide collateral is removed from the supported contract because it is not isolated margin.\n''',
        '''Multi-asset `margin_mode="isolated"` is rejected until a per-symbol collateral ledger exists. Single-asset isolated execution remains accepted because its collateral semantics are equivalent to cross margin. The current proportional allocation of account-wide collateral is not used as a multi-asset isolated-margin contract.\n''',
    )
    replace_once(
        "docs/superpowers/specs/2026-07-30-audit-hardening-design.md",
        '''Temporary files use process-unique names and exclusive creation. Run identifiers include microseconds and a random suffix. Publication uses a store-scoped lock and refuses concurrent mutation rather than sharing a fixed temporary filename.\n''',
        '''Temporary files use process-unique names and exclusive creation. Generated run identifiers include microseconds. Pointer-write failure rolls the published directory back to staging, and concurrent writers never share a temporary filename.\n''',
    )

    for relative in (
        "scripts/agent_apply_audit_hardening.py",
        ".github/workflows/agent-audit-hardening.yml",
        ".agent/audit-hardening-trigger",
    ):
        path = ROOT / relative
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
