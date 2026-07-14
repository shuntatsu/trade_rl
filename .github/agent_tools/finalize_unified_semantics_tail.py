from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Reject invalid prediction ranges before deriving row-wise default cutoffs.
replace_once(
    "trade_rl/artifacts/signals.py",
    '''    resolved_prediction_start = fit_stop if prediction_start is None else prediction_start
    resolved_prediction_stop = array.shape[0] if prediction_stop is None else prediction_stop
    if generator_digest is not None:
''',
    '''    resolved_prediction_start = fit_stop if prediction_start is None else prediction_start
    resolved_prediction_stop = array.shape[0] if prediction_stop is None else prediction_stop
    if resolved_prediction_start < fit_stop:
        raise ValueError("prediction range must start at or after fit_stop")
    if resolved_prediction_stop <= resolved_prediction_start:
        raise ValueError("signal prediction range must be non-empty and half-open")
    if generator_digest is not None:
''',
)

# Preserve sealed outer-test authorization records alongside execution diagnostics.
replace_once(
    "trade_rl/workflows/fold_runner.py",
    '''from trade_rl.evaluation.walk_forward.folds import IndexRange, WalkForwardFold
from trade_rl.evaluation.walk_forward.stitching import FoldOOSResult
''',
    '''from trade_rl.evaluation.walk_forward.folds import IndexRange, WalkForwardFold
from trade_rl.evaluation.walk_forward.sealed_test import (
    SealedTestAccessRecord,
    SealedTestLedger,
)
from trade_rl.evaluation.walk_forward.stitching import FoldOOSResult
''',
)
replace_once(
    "trade_rl/workflows/fold_runner.py",
    '''        require_aware_datetime(self.selected_at, field="selected_at")


@dataclass(frozen=True, slots=True)
class CandidateTrainingRequest:
''',
    '''        require_aware_datetime(self.selected_at, field="selected_at")

    @property
    def experiment_plan_digest(self) -> str:
        return content_digest(
            {
                "candidates": tuple(candidate.name for candidate in self.candidates),
                "dataset_id": self.dataset_id,
                "minimum_selection_uplift": self.minimum_selection_uplift,
                "schema_version": "fold_execution_plan_v2",
                "signal_digest": self.signal_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class CandidateTrainingRequest:
''',
)
replace_once(
    "trade_rl/workflows/fold_runner.py",
    '''    selected_oos: FoldOOSResult
    baseline_oos: FoldOOSResult

    def __post_init__(self) -> None:
''',
    '''    selected_oos: FoldOOSResult
    baseline_oos: FoldOOSResult
    sealed_test_access: SealedTestAccessRecord | None = None

    def __post_init__(self) -> None:
''',
)
replace_once(
    "trade_rl/workflows/fold_runner.py",
    '''        self.config = config
        self.trainer = trainer
        self.evaluator = evaluator

    @staticmethod
''',
    '''        self.config = config
        self.trainer = trainer
        self.evaluator = evaluator
        self._sealed_test_ledger = SealedTestLedger()

    @staticmethod
''',
)
replace_once(
    "trade_rl/workflows/fold_runner.py",
    '''        baseline_test = self._evaluate(
            fold=fold,
''',
    '''        sealed_test_access = self._sealed_test_ledger.authorize_once(
            experiment_plan_digest=self.config.experiment_plan_digest,
            dataset_id=self.config.dataset_id,
            fold_index=fold.fold_index,
            test_range=fold.test,
            selected_configuration=selected_configuration,
            selected_policy_digest=selected_policy_digest,
        )
        baseline_test = self._evaluate(
            fold=fold,
''',
)
replace_once(
    "trade_rl/workflows/fold_runner.py",
    '''            selected_oos=selected_oos,
            baseline_oos=baseline_oos,
        )
''',
    '''            selected_oos=selected_oos,
            baseline_oos=baseline_oos,
            sealed_test_access=sealed_test_access,
        )
''',
)

# Candidate bundles are immutable before an external release attestation is attached.
replace_once(
    "tests/e2e/test_research_to_serving_v2.py",
    '''        selection_digest="2" * 64,
        artifact_paths=("members/member-000/policy.zip", "policy-loader.json"),
''',
    '''        selection_digest="2" * 64,
        release_digest=None,
        artifact_paths=("members/member-000/policy.zip", "policy-loader.json"),
''',
)

# Serving accepts either the legacy in-bundle release pointer or the newer adjacent
# non-circular release attestation, while validating bundle identity in both cases.
replace_once(
    "trade_rl/serving/bundle.py",
    '''from trade_rl.domain.releases import ReleaseManifest
from trade_rl.domain.selection import PolicyMode
''',
    '''from trade_rl.domain.releases import ReleaseManifest
from trade_rl.domain.selection import PolicyMode
from trade_rl.release.attestation import (
    ReleaseAttestation,
    default_attestation_path,
    load_release_attestation as load_external_release_attestation,
)
''',
)
replace_once(
    "trade_rl/serving/bundle.py",
    '''from trade_rl.serving.release import (
    RELEASE_ATTESTATION_NAME,
    load_release_attestation,
)
''',
    '''from trade_rl.serving.release import (
    RELEASE_ATTESTATION_NAME,
    load_release_attestation as load_legacy_release_attestation,
)
''',
)
replace_once(
    "trade_rl/serving/bundle.py",
    '''    release: ReleaseManifest | None = None
    normalizer: object | None = None
''',
    '''    release: ReleaseManifest | ReleaseAttestation | None = None
    normalizer: object | None = None
''',
)
replace_once(
    "trade_rl/serving/bundle.py",
    '''    release: ReleaseManifest | None = None
    normalizer = None
''',
    '''    release: ReleaseManifest | ReleaseAttestation | None = None
    normalizer = None
''',
)
replace_once(
    "trade_rl/serving/bundle.py",
    '''        release = load_release_attestation(root)
        if release.digest != manifest.release_digest:
''',
    '''        release = load_legacy_release_attestation(root)
        if release.digest != manifest.release_digest:
''',
)
replace_once(
    "trade_rl/serving/bundle.py",
    '''        declared.add(RELEASE_ATTESTATION_NAME)
    for file in manifest.files:
''',
    '''        declared.add(RELEASE_ATTESTATION_NAME)
    else:
        external_path = default_attestation_path(root)
        if external_path.is_file():
            release = load_external_release_attestation(external_path)
            if release.bundle_digest != manifest.bundle_digest:
                raise ValueError("external release attestation bundle mismatch")
            if release.dataset_id != manifest.dataset_id:
                raise ValueError("external release attestation dataset mismatch")
            if release.selected_policy_digest != manifest.policy_digest:
                raise ValueError("external release attestation policy mismatch")
    for file in manifest.files:
''',
)
