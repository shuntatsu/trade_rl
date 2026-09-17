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
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${TARGET_SHA:?}"

package="$PUBLISH_ROOT/$arm"
mkdir -p "$package" "$STATUS_ROOT"

cd "$TARGET_DIR"
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
