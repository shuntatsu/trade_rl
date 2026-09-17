#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: issue640_official_run_arm.sh <arm>" >&2
  exit 2
fi

arm="$1"
case "$arm" in
  cash|constant_long|constant_short|trend|mean_reversion|ridge24|lightgbm24|channel_breakout|ppo0|ppo1|ppo2|ppo3|ppo4) ;;
  *)
    echo "unknown frozen arm: $arm" >&2
    exit 2
    ;;
esac

: "${TARGET_DIR:?}"
: "${SOURCE_DIR:?}"
: "${STUDY_DIR:?}"
: "${PUBLISH_ROOT:?}"
: "${STATUS_ROOT:?}"
: "${RUNNER_TEMP:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_NUMBER:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${TARGET_SHA:?}"

test "$GITHUB_RUN_NUMBER" = "1"
test "$GITHUB_RUN_ATTEMPT" = "1"

package="$PUBLISH_ROOT/$arm"
if [[ -e "$package" || -e "$STUDY_DIR/$arm" || -e "$STATUS_ROOT/$arm" ]]; then
  echo "local immutable arm slot already exists: $arm" >&2
  exit 3
fi
mkdir -p "$package" "$STATUS_ROOT"

cd "$TARGET_DIR"
ARM="$arm" uv run python - <<'PY'
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from trade_rl.artifacts import canonical_json_bytes

arm = os.environ["ARM"]
protocol_path = Path(os.environ["STUDY_DIR"]) / "protocol.json"
claim_path = Path(os.environ["RUNNER_TEMP"]) / "activation" / "activation.json"
if not protocol_path.is_file() or not claim_path.is_file():
    raise SystemExit("Issue 640 activation/protocol evidence is missing before arm")
protocol_raw = protocol_path.read_bytes()
claim_raw = claim_path.read_bytes()
protocol = json.loads(protocol_raw)
claim = json.loads(claim_raw)
if canonical_json_bytes(protocol) != protocol_raw or canonical_json_bytes(claim) != claim_raw:
    raise SystemExit("Issue 640 activation/protocol evidence is noncanonical")
activation = protocol.get("official_activation")
if not isinstance(activation, dict):
    raise SystemExit("Issue 640 official activation is missing from protocol")
if (
    activation.get("workflow_run_id") != int(os.environ["GITHUB_RUN_ID"])
    or activation.get("workflow_run_number") != 1
    or activation.get("workflow_run_attempt") != 1
):
    raise SystemExit("Issue 640 official activation identity drifted before arm")
path = Path(".github/scripts/issue640_directional_activation.py")
spec = importlib.util.spec_from_file_location("issue640_activation", path)
if spec is None or spec.loader is None:
    raise SystemExit("Issue 640 activation helper cannot be loaded")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.validate_activation_claim(
    claim,
    protocol,
    workflow_run_id=int(os.environ["GITHUB_RUN_ID"]),
    implementation_head=os.environ["TARGET_SHA"],
)
print(f"ISSUE640_ARM_{arm}_ACTIVATION_REBOUND=true")
print("ECONOMIC_RESULT_INTERPRETED=false")
PY

set +e
uv run python -m trade_rl.evaluation.directional_study run \
  --source "$SOURCE_DIR" \
  --output "$STUDY_DIR" \
  --arm "$arm"
status=$?
set -e

if [[ -d "$STUDY_DIR/$arm" ]]; then
  cp -a "$STUDY_DIR/$arm/." "$package/"
fi

ARM="$arm" ARM_EXIT_CODE="$status" PACKAGE="$package" uv run python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from trade_rl.artifacts import canonical_json_bytes

arm = os.environ["ARM"]
package = Path(os.environ["PACKAGE"])
status = int(os.environ["ARM_EXIT_CODE"])
payload = {
    "schema_version": "issue640_directional_arm_attempt_v1",
    "issue_number": 640,
    "arm": arm,
    "workflow_run_id": int(os.environ["GITHUB_RUN_ID"]),
    "workflow_run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
    "implementation_head": os.environ["TARGET_SHA"],
    "exit_code": status,
    "started": (package / "started.json").is_file(),
    "result_written": (package / "result.json").is_file(),
    "failure_written": (package / "failed.json").is_file(),
    "economic_result_interpreted": False,
    "unused_data_accessed": False,
    "production_eligible": False,
    "live_trading_authorized": False,
}
(package / "attempt.json").write_bytes(canonical_json_bytes(payload))
PY

printf '%s\n' "$status" > "$STATUS_ROOT/$arm"
echo "ISSUE640_ARM_${arm}_ATTEMPT_RECORDED=true"
echo "ECONOMIC_RESULT_INTERPRETED=false"
