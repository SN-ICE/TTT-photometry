#!/usr/bin/env bash
# A full run of one target, from a clean slate.
#
#   ./full-run.sh SN2026abgb
#   ./full-run.sh SN2026abgb --bands g r --max-images 20
#   ./full-run.sh SN2026abgb --work-root /tmp/scratch
#   ./full-run.sh SN2026abgb --clean          # start the photometry from scratch
#
# Options that every subcommand understands (--config, --data-root,
# --work-root, --bands, --log-level) are forwarded to ALL of the steps below.
# Anything else is passed to the final 'tttphot run' only.
#
# Re-running is cheap: the catalogues and templates are cached, and the run
# itself measures only the frames that are new since last time. Pass --clean to
# throw the previous photometry away and measure everything again -- which is
# what to do after changing the configuration, since the existing rows were
# measured with the old one.
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

# --clean is meant for 'tttphot run', which does the removing itself, but the
# step below has to know about it too -- and it must not reach 'run' twice.
WANT_CLEAN=0
KEPT=()
for arg in "${RUNONLY[@]+"${RUNONLY[@]}"}"; do
    if [ "$arg" = "--clean" ]; then WANT_CLEAN=1; else KEPT+=("$arg"); fi
done
RUNONLY=("${KEPT[@]+"${KEPT[@]}"}")

if [ "$WANT_CLEAN" -eq 1 ]; then
    say "1/5  removing previous outputs for $TARGET"
    run tttphot clean "$TARGET" "${COMMON[@]+"${COMMON[@]}"}"
else
    say "1/5  keeping previous outputs (pass --clean to start again)"
fi

say "2/5  resolving the position"
soft "not resolved; keeping whatever the target file has" \
    tttphot resolve "$TARGET" --write "${COMMON[@]+"${COMMON[@]}"}"

say "3/5  reference catalogues"
soft "no catalogue reached; the run will report no_zeropoint" \
    tttphot fetch-catalog "$TARGET" "${COMMON[@]+"${COMMON[@]}"}"

say "4/5  reference images for difference imaging"
soft "no template reached; the run falls back to direct photometry" \
    tttphot fetch-template "$TARGET" "${COMMON[@]+"${COMMON[@]}"}"

say "5/5  photometry (only the frames that are new since the last run)"
run tttphot run "$TARGET" "${COMMON[@]+"${COMMON[@]}"}" "${RUNONLY[@]+"${RUNONLY[@]}"}"
