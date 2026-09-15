#!/usr/bin/env bash
# A full run of one target, from a clean slate.
#
#   ./full-run.sh SN2026abgb
#   ./full-run.sh SN2026abgb --bands g r --max-images 20
#   ./full-run.sh SN2026abgb --work-root /tmp/scratch
#
# Options that every subcommand understands (--config, --data-root,
# --work-root, --bands, --log-level) are forwarded to ALL of the steps below.
# Anything else is passed to the final 'tttphot run' only.
#
# Steps 2-4 are cached, so re-running is cheap. Step 1 removes only this
# target's outputs; the downloads and the detector patterns survive.
set -euo pipefail

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "usage: $0 <TARGET> [flags]" >&2
    exit 1
fi
TARGET="$1"; shift

# Split the arguments: shared options go to every step, the rest only to 'run'.
COMMON=()
RUNONLY=()
while [ $# -gt 0 ]; do
    case "$1" in
        --config|--data-root|--work-root|--log-level)
            COMMON+=("$1" "$2"); shift 2 ;;
        --bands)
            COMMON+=("$1"); shift
            while [ $# -gt 0 ] && [[ "$1" != -* ]]; do COMMON+=("$1"); shift; done ;;
        *)
            RUNONLY+=("$1"); shift ;;
    esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
run() { printf '   $ %s\n' "$*"; "$@"; }

# A step that exits 2 hit a bug rather than a missing input, so stop: carrying
# on would bury the traceback under a benign-sounding message and then produce
# a light curve built on whatever the broken step left behind.
soft() {
    local what="$1"; shift
    printf '   $ %s\n' "$*"
    "$@" && return 0
    local code=$?
    if [ "$code" -eq 2 ]; then
        printf '\n   %s failed with an internal error (above). Stopping.\n' "$1" >&2
        exit 2
    fi
    printf '   (%s)\n' "$what"
    return 0
}

say "1/5  removing previous outputs for $TARGET"
run tttphot clean "$TARGET" "${COMMON[@]+"${COMMON[@]}"}"

say "2/5  resolving the position"
soft "not resolved; keeping whatever the target file has" \
    tttphot resolve "$TARGET" --write "${COMMON[@]+"${COMMON[@]}"}"

say "3/5  reference catalogues"
soft "no catalogue reached; the run will report no_zeropoint" \
    tttphot fetch-catalog "$TARGET" "${COMMON[@]+"${COMMON[@]}"}"

say "4/5  reference images for difference imaging"
soft "no template reached; the run falls back to direct photometry" \
    tttphot fetch-template "$TARGET" "${COMMON[@]+"${COMMON[@]}"}"

say "5/5  photometry"
run tttphot run "$TARGET" "${COMMON[@]+"${COMMON[@]}"}" "${RUNONLY[@]+"${RUNONLY[@]}"}"
