#!/usr/bin/env bash
# Full 16-arm run, followed by figures + dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
RUN_DIR="$("$PY" -m eval.runner --repeats 3 --workers 8 "$@" | tail -n 1)"
echo "run: $RUN_DIR"
"$PY" -m eval.visualize --run "$RUN_DIR"
