#!/usr/bin/env bash
# Monthly Reddit bot analysis pipeline.
# Run manually or via cron.
#
# Cron example (02:00 on 1st of each month):
#   0 2 1 * * cd /path/to/reddit-bot-analysis-repo && bash run_monthly.sh >> logs/run.log 2>&1
#
# Usage:
#   bash run_monthly.sh                              # V1: current month
#   bash run_monthly.sh --skip-collect               # V1: re-score without new pull
#   bash run_monthly.sh --v2                         # V2: previous calendar month
#   bash run_monthly.sh --v2 --year                  # V2: full year backfill (1000 posts/sub)
#   bash run_monthly.sh --v2 --month 2026-01         # V2: specific month
#   bash run_monthly.sh --v2 --subs india ipl        # V2: test subset

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

mkdir -p data data/v2 output output/v2 docs/data docs/data_v2 logs

LOG_FILE="logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "=== Reddit Bot Analysis — $(date) ===" | tee -a "$LOG_FILE"

# Parse flags
VERSION=1
SKIP_COLLECT=false
MONTH_ARG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --v2)           VERSION=2 ;;
    --skip-collect) SKIP_COLLECT=true ;;
    --year)         EXTRA_ARGS+=(--year) ;;
    --month)        MONTH_ARG="$2"; EXTRA_ARGS+=(--month "$2"); shift ;;
    --subs)
      shift
      EXTRA_ARGS+=(--subs)
      while [[ $# -gt 0 && "$1" != --* ]]; do
        EXTRA_ARGS+=("$1"); shift
      done
      continue ;;
    *) ;;
  esac
  shift
done

echo "Version: V${VERSION}  skip-collect: ${SKIP_COLLECT}" | tee -a "$LOG_FILE"

# ── V1 pipeline ───────────────────────────────────────────────────────────────
if [[ $VERSION -eq 1 ]]; then
  if $SKIP_COLLECT; then
    echo "[STEP 1] Skipping collection (--skip-collect)" | tee -a "$LOG_FILE"
  else
    echo "[STEP 1] Collecting data (V1)…" | tee -a "$LOG_FILE"
    python3 -u scripts/collect_data.py 2>&1 | tee -a "$LOG_FILE"
  fi

  echo "[STEP 2] Scoring (V1)…" | tee -a "$LOG_FILE"
  python3 -u scripts/analyze_data.py 2>&1 | tee -a "$LOG_FILE"

  echo "[STEP 3] Generating site data (V1)…" | tee -a "$LOG_FILE"
  python3 -u scripts/generate_site.py 2>&1 | tee -a "$LOG_FILE"

  echo "" | tee -a "$LOG_FILE"
  echo "=== Done (V1). Push: git add docs/ && git commit -m 'Monthly report' && git push ===" | tee -a "$LOG_FILE"

# ── V2 pipeline ───────────────────────────────────────────────────────────────
else
  if $SKIP_COLLECT; then
    echo "[STEP 1] Skipping collection (--skip-collect)" | tee -a "$LOG_FILE"
  else
    echo "[STEP 1] Collecting data (V2)…" | tee -a "$LOG_FILE"
    python3 -u scripts/collect_data_v2.py "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  fi

  echo "[STEP 2] Scoring (V2)…" | tee -a "$LOG_FILE"
  python3 -u scripts/analyze_data_v2.py ${MONTH_ARG:+--month "$MONTH_ARG"} 2>&1 | tee -a "$LOG_FILE"

  echo "[STEP 3] Generating site data (V2)…" | tee -a "$LOG_FILE"
  python3 -u scripts/generate_site.py --v2 ${MONTH_ARG:+--month "$MONTH_ARG"} 2>&1 | tee -a "$LOG_FILE"

  echo "" | tee -a "$LOG_FILE"
  echo "=== Done (V2). Push: git add docs/ data/v2/ output/v2/ && git commit -m 'V2 report' && git push ===" | tee -a "$LOG_FILE"
fi
