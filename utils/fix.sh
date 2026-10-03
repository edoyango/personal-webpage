#!/usr/bin/env bash
set -euo pipefail

# Script to replace STRAVA_ACTIVITY:<id>[:<token>] placeholders with a Strava
# activity embed. Content markdown marks embed spots with a
# `STRAVA_ACTIVITY:<id>` or `STRAVA_ACTIVITY:<id>:<token>` code span (Goldmark
# strips raw HTML comments, so a comment placeholder doesn't survive to the
# rendered output).
#
# Strava now issues a per-activity `data-token` alongside newer embed codes
# (copy it from the "Embed" option on the activity page); without it,
# strava-embeds.com returns a "This content is unavailable" error. Older
# activities may still embed fine with no token, so it's optional here.

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <generated-html-file>" >&2
    exit 2
fi

file="$1"

perl -0777 -pi -e '
s{<p><code>STRAVA_ACTIVITY:(\d+)(?::([\w-]+))?</code></p>}{
    my ($id, $token) = ($1, $2);
    my $token_attr = defined $token ? qq{ data-token="$token"} : "";
    qq{<div class="stravacontainer"><div class="strava-embed-placeholder" data-embed-type="activity" data-embed-id="$id" data-style="standard" data-from-embed="false"$token_attr></div><script src="https://strava-embeds.com/embed.js"></script></div>};
}gex
' "${file}"
