#!/usr/bin/env bash
# Fetch open external tracking datasets into data/raw/external/ (gitignored).
#
#   scripts/fetch_external_data.sh                             # all datasets (~5 GB)
#   scripts/fetch_external_data.sh skillcorner                 # one dataset (all 20 matches)
#   scripts/fetch_external_data.sh skillcorner:1886347,1899585 # only these SkillCorner matches
#   scripts/fetch_external_data.sh metrica lastrow             # small ones, quick start
#   scripts/fetch_external_data.sh skillcorner2020             # SkillCorner 2019/20, 9 matches (~540 MB)
#   scripts/fetch_external_data.sh soccertrack                 # SoccerTrack v2 compact files (A17)
#   scripts/fetch_external_data.sh wyscout statsbomb           # event-only sources (B27), by name only
#
# Idempotent: a dataset with a .fetched marker is skipped. Delete the folder to re-fetch.
# SkillCorner is the exception: .fetched marks its base (metadata + events for all 20 matches), and
# tracking files (~90 MB each) are added per match. A match is usable once its tracking file exists.
# Match ids and metadata: data/raw/external/skillcorner/data/matches.json
# SoccerTrack v2 is not in the default list: it needs HF_TOKEN (gated dataset) and MTAH_HF_CACHE. It
# pulls ~20 MB of compact files from the cache, or builds them from the ~54 GB original game state
# when the cache is empty (~5 min on 4 cores, then pushes them). Idempotent, so no .fetched marker.
# Catalogue, licences and possession-label notes: docs/external-datasets.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/raw/external"
ALL=(metrica lastrow skillcorner idsse skillcorner2020)
EVENTS=(wyscout statsbomb)  # event-only sources (B27), fetched by name only

# Shallow-clone a GitHub repo, then drop .git.
clone() {  # clone <name> <repo-url> <branch>
  local dest="$OUT/$1"
  rm -rf "$dest"
  git clone --quiet --depth 1 --branch "$3" "$2" "$dest"
  rm -rf "$dest/.git"
}

# SkillCorner: base clone without the LFS tracking files, then tracking per match straight from
# GitHub's LFS media URL (no git-lfs needed). Same layout as scripts/cloud_env_setup.sh.
fetch_skillcorner() {  # fetch_skillcorner [id ...]   (no ids = all 20)
  local dest="$OUT/skillcorner" id f
  if [ ! -f "$dest/.fetched" ]; then
    rm -rf "$dest"
    GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --depth 1 https://github.com/SkillCorner/opendata.git "$dest"
    rm -rf "$dest/.git"
    find "$dest/data/matches" -name '*.jsonl' -size -2k -delete   # LFS pointers, not data
    date -u +%FT%TZ >"$dest/.fetched"
  fi
  [ $# -gt 0 ] || set -- $(ls "$dest/data/matches")
  for id in "$@"; do
    [ -d "$dest/data/matches/$id" ] || { echo "unknown skillcorner match: $id (see data/matches.json)" >&2; exit 1; }
    f="$dest/data/matches/$id/${id}_tracking_extrapolated.jsonl"
    [ -s "$f" ] && continue
    echo "  $id tracking"
    curl -fsSL --retry 3 -o "$f.part" \
      "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/$id/${id}_tracking_extrapolated.jsonl"
    mv "$f.part" "$f"
  done
}

# IDSSE / Sportec (DFL): 7 Bundesliga matches on figshare, ~2.6 GB of XML.
fetch_idsse() {
  local dest="$OUT/idsse"
  mkdir -p "$dest"
  curl -fsSL https://api.figshare.com/v2/articles/28196177 \
    | python3 -I -c 'import sys,json; [print(f["name"], f["size"], f["download_url"]) for f in json.load(sys.stdin)["files"]]' \
    | while read -r name size url; do
        if [ -f "$dest/$name" ] && [ "$(wc -c <"$dest/$name" | tr -d ' ')" = "$size" ]; then continue; fi
        echo "  $name ($((size / 1000000)) MB)"
        # .part + resume: figshare often times out mid-file; a re-run continues where it stopped
        [ -f "$dest/$name" ] && mv "$dest/$name" "$dest/$name.part"
        curl -fsSL --retry 5 -C - -o "$dest/$name.part" "$url"
        mv "$dest/$name.part" "$dest/$name"
      done
}

# SkillCorner 2019/20 (backlog A15): SkillCorner's first open release, 9 top-league matches, MIT.
# Deleted from their repo on 6 Oct 2025 (commit 741bdb798b) but still in its history: raw files
# pinned to the last commit that has them. Plain git, not LFS; ~56 MB of JSON per match.
SC2020_COMMIT=1b99d401ddc82e366c5c20d6860878668fd5858f
SC2020_IDS=(4039 3749 3518 3442 2841 2440 2417 2269 2068)
fetch_skillcorner2020() {
  local dest="$OUT/skillcorner2020" base id f
  base="https://raw.githubusercontent.com/SkillCorner/opendata/$SC2020_COMMIT"
  mkdir -p "$dest/data/matches"
  for f in LICENSE README.md data/matches.json; do
    [ -s "$dest/$f" ] || curl -fsSL --retry 3 -o "$dest/$f" "$base/$f"
  done
  for id in "${SC2020_IDS[@]}"; do
    mkdir -p "$dest/data/matches/$id"
    for f in match_data.json structured_data.json; do
      [ -s "$dest/data/matches/$id/$f" ] && continue
      echo "  $id $f"
      curl -fsSL --retry 3 -o "$dest/data/matches/$id/$f.part" "$base/data/matches/$id/$f"
      mv "$dest/data/matches/$id/$f.part" "$dest/data/matches/$id/$f"
    done
  done
}

# Wyscout public events (Pappalardo et al. 2019, CC BY 4.0, figshare): 1,941 matches of 7
# competitions 2017/18, events only (B27). ~80 MB zipped, ~1 GB unzipped.
fetch_wyscout() {
  local dest="$OUT/wyscout"
  mkdir -p "$dest"
  curl -fsSL --retry 5 -o "$dest/events.zip" https://ndownloader.figshare.com/files/14464685
  curl -fsSL --retry 5 -o "$dest/matches.zip" https://ndownloader.figshare.com/files/14464622
  curl -fsSL --retry 5 -o "$dest/teams.json" https://ndownloader.figshare.com/files/15073697
  (cd "$dest" && unzip -qo events.zip -d events && unzip -qo matches.zip -d matches && rm events.zip matches.zip)
}

# StatsBomb open data (StatsBomb public data licence: attribution), events only (B27). A fixed
# subset of men's competitions, ~580 matches, ~2 GB: World Cup 2022, Euro 2020 and 2024,
# Bundesliga 2023/24, Premier League 2015/16. No La Liga (keeps every Barcelona v Athletic out).
SB_RAW=https://raw.githubusercontent.com/statsbomb/open-data/master/data
SB_SEASONS="43/106 55/43 55/282 9/281 2/27"
fetch_statsbomb() {
  local dest="$OUT/statsbomb" cs
  mkdir -p "$dest/matches" "$dest/events"
  for cs in $SB_SEASONS; do
    mkdir -p "$dest/matches/${cs%/*}"
    curl -fsSL --retry 5 -o "$dest/matches/$cs.json" "$SB_RAW/matches/$cs.json"
  done
  cat "$dest"/matches/*/*.json \
    | python3 -I -c 'import sys,json,re; [print(m) for m in re.findall(r"\"match_id\"\s*:\s*(\d+)", sys.stdin.read())]' \
    | sort -u | while read -r id; do [ -s "$dest/events/$id.json" ] || echo "$id"; done \
    | xargs -P 8 -I{} curl -fsSL --retry 5 -o "$dest/events/{}.json" "$SB_RAW/events/{}.json"
}

fetch() {
  case "$1" in
    metrica)     clone metrica https://github.com/metrica-sports/sample-data.git master ;;
    lastrow)     clone lastrow https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row.git master ;;
    idsse)       fetch_idsse ;;
    skillcorner2020) fetch_skillcorner2020 ;;
    wyscout)     fetch_wyscout ;;
    statsbomb)   fetch_statsbomb ;;
    *) echo "unknown dataset: $1 (choose from: ${ALL[*]} ${EVENTS[*]} soccertrack)" >&2; exit 1 ;;
  esac
}

mkdir -p "$OUT"
for arg in "${@:-${ALL[@]}}"; do
  ds=${arg%%:*}
  if [ "$ds" = skillcorner ]; then
    ids=""; [ "$arg" = "${arg#*:}" ] || ids=${arg#*:}
    echo "== skillcorner: fetching ${ids:-all matches}"
    fetch_skillcorner ${ids//,/ }
    echo "== skillcorner: done ($(ls "$OUT"/skillcorner/data/matches/*/*.jsonl 2>/dev/null | wc -l | tr -d ' ') matches with tracking)"
    continue
  fi
  if [ "$ds" = soccertrack ]; then
    echo "== soccertrack: fetching (cache $([ -n "${MTAH_HF_CACHE:-}" ] && echo set || echo NOT set))"
    (cd "$ROOT" && "$ROOT/.venv/bin/python" -m src.ingest.soccertrack fetch)
    # V3b's whistle reads per-frame audio features (18 MB cache); without them the cue silently does
    # nothing on SoccerTrack (W5 close, F337). Best effort: a missing cache only warns at run time.
    (cd "$ROOT" && "$ROOT/.venv/bin/python" -m src.video.soccertrack_audio fetch) \
      || echo "== soccertrack audio: not fetched (the whistle cue will warn and give no effect)"
    continue
  fi
  if [ -f "$OUT/$ds/.fetched" ]; then echo "== $ds: already fetched"; continue; fi
  echo "== $ds: fetching"
  fetch "$ds"
  date -u +%FT%TZ >"$OUT/$ds/.fetched"
  echo "== $ds: done ($(du -sh "$OUT/$ds" | cut -f1))"
done
