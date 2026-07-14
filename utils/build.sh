#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLISH_DIR="${PUBLISH_DIR:-public}"
PUBLISH_ROOT="${ROOT_DIR}/${PUBLISH_DIR}"
HUGO_CACHEDIR="${HUGO_CACHEDIR:-${ROOT_DIR}/.hugo_cache}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-/}"

base_url() {
  local path="${1#/}"
  local base="${PUBLIC_BASE_URL%/}"

  if [[ -z "${base}" || "${base}" == "." ]]; then
    base="/"
  fi

  if [[ "${base}" == "/" ]]; then
    printf '/%s\n' "${path:+${path}/}"
  else
    printf '%s/%s\n' "${base}" "${path:+${path}/}"
  fi
}

build_site() {
  local source_dir="$1"
  local output_path="$2"
  local url_path="$3"
  local cache_path="$4"
  local destination="${PUBLISH_ROOT}/${output_path}"

  echo "Building ${source_dir} -> ${destination}"
  hugo \
    --source "${ROOT_DIR}/${source_dir}" \
    --destination "${destination}" \
    --baseURL "$(base_url "${url_path}")" \
    --cacheDir "${HUGO_CACHEDIR}/${cache_path}" \
    --cleanDestinationDir \
    --gc \
    --minify
}

rm -rf "${PUBLISH_ROOT}"
mkdir -p "${PUBLISH_ROOT}"

build_site "src/home" "." "" "home"
build_site "src/about" "about" "about" "about"
build_site "src/cv" "about/cv" "about/cv" "cv"
build_site "src/trips" "trips" "trips" "trips"
build_site "src/docs" "docs" "docs" "docs"
build_site "src/worknotes" "worknotes" "worknotes" "worknotes"

if [[ -f "${ROOT_DIR}/CNAME" ]]; then
  cp "${ROOT_DIR}/CNAME" "${PUBLISH_ROOT}/CNAME"
fi

"${ROOT_DIR}/utils/fix.sh" "${PUBLISH_ROOT}/trips/content/2019jpn-twn/index.html"
