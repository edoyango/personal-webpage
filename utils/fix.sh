#!/usr/bin/env bash
set -euo pipefail

# Script to replace STRAVA_ACTIVITY:<id> placeholders with a Strava activity
# embed. Content markdown marks embed spots with a `STRAVA_ACTIVITY:<id>`
# code span (Goldmark strips raw HTML comments, so a comment placeholder
# doesn't survive to the rendered output).

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <generated-html-file>" >&2
    exit 2
fi

file="$1"

perl -0777 -pi -e 's{<p><code>STRAVA_ACTIVITY:(\d+)</code></p>}{<div class="stravacontainer"><div class="strava-embed-placeholder" data-embed-type="activity" data-embed-id="$1"></div><script src="https://strava-embeds.com/embed.js"></script></div>}g' "${file}"
