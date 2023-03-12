#!/bin/bash

# Script to replace <!-- raw HTML omitted --> with strava activity

activityids="2164094022 2166398671 2168197783 2175930772 2178519713 2180914194 2183032370 2184904148 2187434992 2189995780 2191955784"

for i in $activityids
do
    sed -i '0,/<!-- raw HTML omitted -->/{s@@<div class="stravacontainer"><div class="strava-embed-placeholder" data-embed-type="activity" data-embed-id="'$i'"></div><script src="https://strava-embeds.com/embed.js"></script></div>@}' $1
done