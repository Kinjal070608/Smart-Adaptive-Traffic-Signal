#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker command not found"
  exit 1
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
else
  echo "[ERROR] No Dockerfile found in the repository root"
  exit 1
fi

echo "[INFO] Building Docker image..."
docker build "$DOCKER_CONTEXT" -t smart-adaptive-traffic-signal:latest

echo "[INFO] Running openenv validate..."
if ! command -v openenv >/dev/null 2>&1; then
  echo "[ERROR] openenv command not found. Install it with: pip install openenv-core"
  exit 1
fi

cd "$REPO_DIR"
openenv validate

echo "[INFO] Validation complete. Docker build and openenv validate passed."
