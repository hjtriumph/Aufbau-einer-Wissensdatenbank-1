#!/usr/bin/env bash
# Phase 1: four main arms + equal-context control (the core story).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
"$PY" -m eval.runner --arms A "A'" B C "D'" --repeats 3 --workers 8 "$@"
