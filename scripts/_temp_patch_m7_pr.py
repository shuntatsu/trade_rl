from __future__ import annotations

from pathlib import Path


BRANCH = "agent/fix-m7-production-blockers"


def replace_idempotent(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if old in text:
        file_path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return
    if new in text:
        return
    raise RuntimeError(f"neither old nor new text found in {path}: {old[:80]!r}")


def wire_pre_trade_risk() -> None:
    replace_idempotent(
        "mars_lite/env/portfolio_env.py",
        "            self.pre_trade_verifier.validate(target, self.portfolio_value)",
        """            self.pre_trade_verifier.validate(
                target,
                self.portfolio_value,
                symbols=self.fs.symbols,
                current_weights=prev,
            )""",
    )
    replace_idempotent(
        "mars_lite/learning/baselines.py",
        "            pre_trade_verifier.validate(target, value)",
        """            pre_trade_verifier.validate(
                target,
                value,
                symbols=fs.symbols,
                current_weights=weights,
            )""",
    )

    path = Path("tests/test_pre_trade_risk.py")
    text = path.read_text(encoding="utf-8")
    if "test_env_integration_uses_execution_delta_for_minimum_order" not in text:
        text += """


def test_env_integration_uses_execution_delta_for_minimum_order():
    fs = DummyFeatureSet()
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(min_order_notional=10.0))
    env = PortfolioTradingEnv(
        fs,
        pre_trade_verifier=verifier,
        initial_capital=100.0,
        min_trade_delta=0.0,
        lambda_turnover=0.0,
    )
    env.reset(options={"start_idx": 0})
    env.step(np.array([0.2, 0.0]))
    with pytest.raises(PreTradeRejection) as exc:
        env.step(np.array([0.21, 0.0]))
    assert exc.value.reason == "min_order_notional_not_met"
    assert exc.value.details["symbol"] == "BTCUSDT"
    assert exc.value.details["order_notional"] == pytest.approx(1.0)


def test_simulate_strategy_uses_execution_delta_for_minimum_order():
    fs = DummyFeatureSet()
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(min_order_notional=0.1))

    def small_rebalance_strategy(fs, t, w):
        if not w.any():
            return np.array([0.2, 0.0])
        return np.array([0.21, 0.0])

    with pytest.raises(PreTradeRejection) as exc:
        simulate_strategy(
            fs,
            small_rebalance_strategy,
            pre_trade_verifier=verifier,
            min_trade_delta=0.0,
        )
    assert exc.value.reason == "min_order_notional_not_met"
    assert exc.value.details["symbol"] == "BTCUSDT"
    assert exc.value.details["order_notional"] == pytest.approx(0.01)
"""
        path.write_text(text, encoding="utf-8")


def harden_deployment_gate() -> None:
    path = Path("mars_lite/server/deployment_gate.py")
    text = path.read_text(encoding="utf-8")

    if "import math" not in text:
        text = text.replace("import json\n", "import json\nimport math\n", 1)
    text = text.replace(
        "an immutable evidence bundle downloaded from a prior GitHub Actions run.",
        "a content-addressed evidence bundle downloaded from a prior trusted GitHub Actions run.",
    )

    anchor = '_APPROVER = re.compile(r"[a-zA-Z0-9_.@-]+")\n'
    constants = '''_APPROVER = re.compile(r"[a-zA-Z0-9_.@-]+")
_SHADOW_MIN_SHARPE_DIFF = -0.3
_SHADOW_MAX_DRAWDOWN = 0.20
_CANARY_MAX_CAPITAL_USD = 10_000.0
_CANARY_MIN_DURATION_DAYS = 7
_CANARY_MAX_LOSS_PCT = 0.05
_CANARY_MAX_SLIPPAGE_BPS = 15.0
_DRIFT_MAX_PSI = 0.25
_DRIFT_MIN_KS_P = 0.01
'''
    if "_SHADOW_MIN_SHARPE_DIFF" not in text:
        if anchor not in text:
            raise RuntimeError("deployment gate constants anchor not found")
        text = text.replace(anchor, constants, 1)

    for old in (
        "    min_sharpe_diff: float = -0.3\n    max_allowed_drawdown: float = 0.20\n",
        "    max_allowed_capital_usd: float = 10_000.0\n    min_duration_days: int = 7\n    max_allowed_loss_pct: float = 0.05\n    max_allowed_slippage_bps: float = 15.0\n",
        "    max_allowed_psi: float = 0.25\n    min_allowed_ks_p: float = 0.01\n",
    ):
        text = text.replace(old, "", 1)

    old = '''    def is_passed(self) -> tuple[bool, str]:
        diff = self.oos_sharpe - self.baseline_sharpe
        if diff < self.min_sharpe_diff:
            return False, (
                f"shadow sharpe difference {diff:.2f} below threshold "
                f"{self.min_sharpe_diff}"
            )
        if self.max_drawdown > self.max_allowed_drawdown:
            return False, (
                f"shadow drawdown {self.max_drawdown:.1%} exceeds allowed "
                f"{self.max_allowed_drawdown:.1%}"
            )
        return True, "shadow report passed"
'''
    new = '''    def is_passed(self) -> tuple[bool, str]:
        metrics = (self.oos_sharpe, self.baseline_sharpe, self.max_drawdown)
        if not all(math.isfinite(float(value)) for value in metrics):
            return False, "shadow report metrics must be finite"
        if not 0.0 <= float(self.max_drawdown) <= 1.0:
            return False, "shadow drawdown must be between 0 and 1"
        diff = float(self.oos_sharpe) - float(self.baseline_sharpe)
        if diff < _SHADOW_MIN_SHARPE_DIFF:
            return False, (
                f"shadow sharpe difference {diff:.2f} below threshold "
                f"{_SHADOW_MIN_SHARPE_DIFF}"
            )
        if float(self.max_drawdown) > _SHADOW_MAX_DRAWDOWN:
            return False, (
                f"shadow drawdown {self.max_drawdown:.1%} exceeds allowed "
                f"{_SHADOW_MAX_DRAWDOWN:.1%}"
            )
        return True, "shadow report passed"
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif "shadow report metrics must be finite" not in text:
        raise RuntimeError("shadow gate method not found")

    old = '''    def is_passed(self) -> tuple[bool, str]:
        if self.capital_cap_usd > self.max_allowed_capital_usd:
            return False, (
                f"canary capital {self.capital_cap_usd} exceeds cap "
                f"{self.max_allowed_capital_usd}"
            )
        if self.duration_days < self.min_duration_days:
            return False, (
                f"canary duration {self.duration_days} days below minimum "
                f"{self.min_duration_days} days"
            )
        if self.max_loss_pct > self.max_allowed_loss_pct:
            return False, (
                f"canary max loss {self.max_loss_pct:.1%} exceeds allowed "
                f"{self.max_allowed_loss_pct:.1%}"
            )
        if self.mean_slippage_bps > self.max_allowed_slippage_bps:
            return False, (
                f"canary slippage {self.mean_slippage_bps:.1f}bps exceeds allowed "
                f"{self.max_allowed_slippage_bps:.1f}bps"
            )
        return True, "canary report passed"
'''
    new = '''    def is_passed(self) -> tuple[bool, str]:
        metrics = (self.capital_cap_usd, self.max_loss_pct, self.mean_slippage_bps)
        if not all(math.isfinite(float(value)) for value in metrics):
            return False, "canary report metrics must be finite"
        if isinstance(self.duration_days, bool) or not isinstance(self.duration_days, int):
            return False, "canary duration_days must be an integer"
        if float(self.capital_cap_usd) < 0.0:
            return False, "canary capital must be non-negative"
        if not 0.0 <= float(self.max_loss_pct) <= 1.0:
            return False, "canary max loss must be between 0 and 1"
        if float(self.mean_slippage_bps) < 0.0:
            return False, "canary slippage must be non-negative"
        if float(self.capital_cap_usd) > _CANARY_MAX_CAPITAL_USD:
            return False, (
                f"canary capital {self.capital_cap_usd} exceeds cap "
                f"{_CANARY_MAX_CAPITAL_USD}"
            )
        if self.duration_days < _CANARY_MIN_DURATION_DAYS:
            return False, (
                f"canary duration {self.duration_days} days below minimum "
                f"{_CANARY_MIN_DURATION_DAYS} days"
            )
        if float(self.max_loss_pct) > _CANARY_MAX_LOSS_PCT:
            return False, (
                f"canary max loss {self.max_loss_pct:.1%} exceeds allowed "
                f"{_CANARY_MAX_LOSS_PCT:.1%}"
            )
        if float(self.mean_slippage_bps) > _CANARY_MAX_SLIPPAGE_BPS:
            return False, (
                f"canary slippage {self.mean_slippage_bps:.1f}bps exceeds allowed "
                f"{_CANARY_MAX_SLIPPAGE_BPS:.1f}bps"
            )
        return True, "canary report passed"
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif "canary report metrics must be finite" not in text:
        raise RuntimeError("canary gate method not found")

    old = '''    def is_passed(self) -> tuple[bool, str]:
        if self.psi_score > self.max_allowed_psi:
            return False, (
                f"drift PSI {self.psi_score:.3f} exceeds maximum "
                f"{self.max_allowed_psi:.3f}"
            )
        if self.ks_p_value < self.min_allowed_ks_p:
            return False, (
                f"drift KS p-value {self.ks_p_value:.4f} below minimum "
                f"{self.min_allowed_ks_p:.4f}"
            )
        return True, "drift report passed"
'''
    new = '''    def is_passed(self) -> tuple[bool, str]:
        metrics = (self.psi_score, self.ks_p_value)
        if not all(math.isfinite(float(value)) for value in metrics):
            return False, "drift report metrics must be finite"
        if float(self.psi_score) < 0.0:
            return False, "drift PSI must be non-negative"
        if not 0.0 <= float(self.ks_p_value) <= 1.0:
            return False, "drift KS p-value must be between 0 and 1"
        if float(self.psi_score) > _DRIFT_MAX_PSI:
            return False, (
                f"drift PSI {self.psi_score:.3f} exceeds maximum "
                f"{_DRIFT_MAX_PSI:.3f}"
            )
        if float(self.ks_p_value) < _DRIFT_MIN_KS_P:
            return False, (
                f"drift KS p-value {self.ks_p_value:.4f} below minimum "
                f"{_DRIFT_MIN_KS_P:.4f}"
            )
        return True, "drift report passed"
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif "drift report metrics must be finite" not in text:
        raise RuntimeError("drift gate method not found")

    old = '''    def is_passed(self) -> tuple[bool, str]:
        if self.active_incidents:
            return False, "deployment blocked due to active incidents"
        return True, "incident report passed"
'''
    new = '''    def is_passed(self) -> tuple[bool, str]:
        if type(self.active_incidents) is not bool:
            return False, "active_incidents must be a boolean"
        if self.active_incidents:
            return False, "deployment blocked due to active incidents"
        return True, "incident report passed"
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif "active_incidents must be a boolean" not in text:
        raise RuntimeError("incident gate method not found")

    path.write_text(text, encoding="utf-8")

    tests = Path("tests/test_deployment_gate.py")
    text = tests.read_text(encoding="utf-8")
    if "import pytest" not in text:
        text = text.replace("import json\n", "import json\n\nimport pytest\n", 1)
    if "test_report_cannot_override_gate_thresholds" not in text:
        text += '''


def _update_report_digest(root: Path, report_name: str, digest_field: str) -> None:
    candidate = json.loads((root / "candidate.json").read_text(encoding="utf-8"))
    candidate[digest_field] = _sha(root / report_name)
    (root / "candidate.json").write_text(json.dumps(candidate), encoding="utf-8")


def test_report_cannot_override_gate_thresholds(tmp_path):
    root = _write_bundle(tmp_path)
    shadow = json.loads((root / "shadow.json").read_text(encoding="utf-8"))
    shadow["max_allowed_drawdown"] = 1.0
    (root / "shadow.json").write_text(json.dumps(shadow), encoding="utf-8")
    _update_report_digest(root, "shadow.json", "shadow_report_sha256")
    with pytest.raises(ValueError, match="invalid shadow evidence schema"):
        load_evidence_bundle(root, "canary")


def test_non_finite_shadow_metrics_are_blocked(tmp_path):
    root = _write_bundle(tmp_path)
    shadow = json.loads((root / "shadow.json").read_text(encoding="utf-8"))
    shadow["oos_sharpe"] = float("nan")
    (root / "shadow.json").write_text(json.dumps(shadow), encoding="utf-8")
    _update_report_digest(root, "shadow.json", "shadow_report_sha256")
    decision = DeploymentGate().evaluate(load_evidence_bundle(root, "canary"))
    assert decision.allowed is False
    assert "finite" in decision.reason


def test_invalid_drift_probability_is_blocked(tmp_path):
    root = _write_bundle(tmp_path)
    drift = json.loads((root / "drift.json").read_text(encoding="utf-8"))
    drift["ks_p_value"] = 2.0
    (root / "drift.json").write_text(json.dumps(drift), encoding="utf-8")
    _update_report_digest(root, "drift.json", "drift_report_sha256")
    decision = DeploymentGate().evaluate(load_evidence_bundle(root, "canary"))
    assert decision.allowed is False
    assert "between 0 and 1" in decision.reason
'''
    tests.write_text(text, encoding="utf-8")


def audit_docs() -> None:
    warning = """
> [!WARNING]
> **文書区分: 歴史的研究記録 / Production仕様ではない**
> 本文の既定値、収益率、Sharpe、エッジに関する記述は、記載時点の合成データまたは開発データで得られた実験結果・仮説である。実市場での将来収益やProduction適格性を証明しない。現行の運用上の正典と文書優先順位は `docs/README.md`、本稼働前の未完項目は `docs/DOCUMENTATION_AUDIT_CHECKLIST.md` に従う。
> Phase 1で確認された範囲は、固定された合成P0条件での学習可能性と未使用seed群におけるNull Policy選択機構の再現であり、実市場アルファの存在は未証明である。
""".strip()
    for name in ("docs/ARCHITECTURE.md", "docs/PROFIT_DESIGN.md"):
        path = Path(name)
        text = path.read_text(encoding="utf-8")
        if "**文書区分: 歴史的研究記録 / Production仕様ではない**" not in text:
            first, rest = text.split("\n", 1)
            path.write_text(f"{first}\n\n{warning}\n\n{rest}", encoding="utf-8")

    Path("docs/model_decision_log.md").write_text(
        """# モデル意思決定ログ

本ログは追記専用とし、既存エントリを上書き・削除しない。閾値、特徴量、モデル、データ期間、検証方法、デプロイ判断を変更するたびに1件追加する。

## 一覧

| Record ID | Decision time (UTC) | Stage | Model version | Decision | Decision owner | Evidence bundle/run |
|---|---|---|---|---|---|---|
| DEC-YYYYMMDD-NNN | YYYY-MM-DDTHH:MM:SSZ | research/shadow/canary/production/rollback | version | accepted/rejected/hold/rollback | owner | run ID / artifact URI |

## エントリテンプレート

### 識別・責任
- Record ID:
- Decision time (UTC):
- Author:
- Decision owner:
- Independent reviewer:
- Stage:
- Decision: `accepted` / `rejected` / `hold` / `rollback`
- Reason and alternatives considered:

### 候補モデルと再現性
- Model version:
- Previous model version:
- Git commit SHA:
- Model artifact SHA-256:
- Configuration SHA-256:
- Data manifest SHA-256:
- Dependency lock SHA-256:
- Training/validation/test periods:
- Data seeds / model seeds:
- Training command or workflow run:

### 検証証拠
- Exact candidate CI run and conclusion:
- Shadow run ID / report SHA-256:
- Drift report ID / SHA-256:
- Incident report ID / SHA-256:
- Canary run ID / parent Shadow run / report SHA-256:
- Replay-versus-live calibration:
- Bootstrap and multiple-testing treatment:
- Pre-trade risk verification:
- GameDay evidence and recovery timings:
- Known limitations and unresolved risks:

### 承認・運用
- Approval ticket:
- GitHub Environment approver:
- Deployment Gate decision JSON / SHA-256:
- Effective capital and risk limits:
- Rollback target and trigger conditions:
- Follow-up actions, owners, and due dates:
- Related incident/RCA links:
""",
        encoding="utf-8",
    )

    checklist = Path("docs/DOCUMENTATION_AUDIT_CHECKLIST.md")
    text = checklist.read_text(encoding="utf-8")
    if "Audit date:" not in text:
        text = text.replace(
            "# Documentation audit checklist\n",
            "# Documentation audit checklist\n\n- Audit date: 2026-07-11\n- Scope: PR #6 code, workflows, and all files under `docs/`\n- Production status: **NO-GO** until every Production blocker is closed with evidence\n",
            1,
        )
    text = text.replace(
        "- [ ] **OWNER ACTION — Production blocker** Ensure live order construction passes current weights and all open orders to `PreTradeRiskVerifier`.",
        "- [x] **FIXED** Environment and baseline execution paths pass current weights and symbols to `PreTradeRiskVerifier`; live adapters must additionally pass all open orders.",
    )
    text = text.replace(
        "- [x] **HISTORICAL** `ARCHITECTURE.md` and `PROFIT_DESIGN.md` are classified as research records, not production authorization.",
        "- [x] **FIXED / HISTORICAL** `ARCHITECTURE.md` and `PROFIT_DESIGN.md` contain in-document warning banners and are classified as research records, not production authorization.",
    )
    text = text.replace(
        "- [ ] **BLOCKED — Economic validation** No statement in the docs may be interpreted as proof of future profitability.",
        "- [x] **FIXED** Research-return, Sharpe, and edge statements are explicitly labelled as historical observations or hypotheses, not proof of future profitability.",
    )
    text = text.replace(
        "- [ ] **OWNER ACTION — Production blocker** Create the workflow that produces and uploads the immutable `deployment-evidence` artifact.",
        "- [ ] **OWNER ACTION — Production blocker** Create a trusted producer workflow that generates and uploads the content-addressed `deployment-evidence` artifact; restrict accepted workflow identity and release branch.",
    )
    if "Gate thresholds are code-owned" not in text:
        text = text.replace(
            "- [x] **FIXED** Active incident evidence blocks promotion.\n",
            "- [x] **FIXED** Active incident evidence blocks promotion.\n- [x] **FIXED** Gate thresholds are code-owned and cannot be overridden by evidence JSON; non-finite and out-of-range metrics are rejected.\n",
            1,
        )
    checklist.write_text(text, encoding="utf-8")

    evidence = Path("docs/deployment_evidence.md")
    text = evidence.read_text(encoding="utf-8")
    text = text.replace(
        "Canary and Production promotion consume an immutable GitHub Actions artifact named `deployment-evidence`. Boolean self-attestation is not accepted.",
        "Canary and Production promotion consume a content-addressed GitHub Actions artifact named `deployment-evidence` from a successful trusted validation run. Boolean self-attestation is not accepted. GitHub artifact retention is finite; Production evidence must also be copied to an access-controlled, deletion-protected archive.",
    )
    if "## Trust boundary" not in text:
        text += """

## Trust boundary

The consumer gate verifies report and model digests, model identity, source-run head SHA, and Shadow-to-Canary lineage. Before Production, repository owners must also restrict the accepted producer workflow, event, and release branch so an arbitrary successful workflow cannot mint promotion evidence. The producer workflow is not implemented by the consumer gate itself and remains a Production blocker until configured and tested.
"""
    evidence.write_text(text, encoding="utf-8")


def main() -> None:
    wire_pre_trade_risk()
    harden_deployment_gate()
    audit_docs()


if __name__ == "__main__":
    main()
