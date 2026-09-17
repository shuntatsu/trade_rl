#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: issue640_run_arm.sh <arm>" >&2
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

: "${GITHUB_REPOSITORY:?}"
: "${GITHUB_RUN_ID:?}"
: "${GITHUB_RUN_ATTEMPT:?}"
: "${GITHUB_OUTPUT:?}"
: "${SOURCE_SHA:?}"
: "${RUNNER_TEMP:?}"
: "${GH_TOKEN:?}"

artifact="issue640-directional-arm-${arm}-v1"
existing="$(gh api "repos/$GITHUB_REPOSITORY/actions/artifacts?name=$artifact&per_page=100" --jq .total_count)"
if [[ "$existing" != "0" ]]; then
  echo "immutable arm slot already exists: $artifact" >&2
  exit 3
fi

workspace="${GITHUB_WORKSPACE:?}"
target="$workspace/target"
source_root="$RUNNER_TEMP/source"
study_root="$RUNNER_TEMP/study"
package="$RUNNER_TEMP/packages/$arm"
rm -rf "$package"
mkdir -p "$package"

cd "$target"
set +e
uv run python -m trade_rl.evaluation.directional_study run \
  --source "$source_root" \
  --output "$study_root" \
  --arm "$arm"
status=$?
set -e

if [[ -d "$study_root/$arm" ]]; then
  cp -a "$study_root/$arm/." "$package/"
fi

ARM="$arm" ARM_EXIT_CODE="$status" uv run python - <<'PY'
import os
from pathlib import Path

from trade_rl.artifacts.canonical import canonical_json_bytes

arm = os.environ["ARM"]
package = Path(os.environ["RUNNER_TEMP"]) / "packages" / arm
payload = {
    "schema_version": "issue640_directional_arm_attempt_v1",
    "issue_number": 640,
    "arm": arm,
    "workflow_run_id": int(os.environ["GITHUB_RUN_ID"]),
    "workflow_run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
    "source_head": os.environ["SOURCE_SHA"],
    "exit_code": int(os.environ["ARM_EXIT_CODE"]),
    "started": (package / "started.json").is_file(),
    "result_published": (package / "result.json").is_file(),
    "failure_published": (package / "failed.json").is_file(),
    "unused_data_accessed": False,
    "production_eligible": False,
    "live_trading_authorized": False,
}
(package / "attempt.json").write_bytes(canonical_json_bytes(payload))
PY

echo "exit_code=$status" >> "$GITHUB_OUTPUT"
exit 0
