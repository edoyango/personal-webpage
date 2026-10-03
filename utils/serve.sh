#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8080}"
export PUBLISH_DIR="${PUBLISH_DIR:-public}"
PUBLISH_ROOT="${ROOT_DIR}/${PUBLISH_DIR}"

echo "Building site..."
"${ROOT_DIR}/utils/build.sh"

echo "Serving ${PUBLISH_ROOT} at http://localhost:${PORT}/"
cd "${PUBLISH_ROOT}"
exec python3 -m http.server "${PORT}"
