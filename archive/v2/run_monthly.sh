#!/usr/bin/env bash
# Monthly Reddit bot analysis pipeline (V2). V1 is archived — see archive/v1/.
# Run manually or via cron.
#
# Cron example (02:00 on 1st of each month):
#   0 2 1 * * cd /path/to/reddit-bot-analysis-repo && bash run_monthly.sh >> logs/run.log 2>&1
#
# Usage:
#   bash run_monthly.sh                        # previous calendar month
#   bash run_monthly.sh --skip-collect         # re-score without a new Reddit pull
#   bash run_monthly.sh --year                 # full rolling-year backfill (1000 posts/sub)
#   bash run_monthly.sh --month 2026-07        # specific month
#   bash run_monthly.sh --subs india ipl       # test subset

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

mkdir -p data/v2 output/v2 docs/data_v2 reports logs

LOG_FILE="logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "=== Reddit Bot Analysis (V2) — $(date) ===" | tee -a "$LOG_FILE"

# Parse flags
SKIP_COLLECT=false
MONTH_ARG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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

echo "skip-collect: ${SKIP_COLLECT}" | tee -a "$LOG_FILE"

if $SKIP_COLLECT; then
  echo "[STEP 1] Skipping collection (--skip-collect)" | tee -a "$LOG_FILE"
else
  echo "[STEP 1] Collecting data…" | tee -a "$LOG_FILE"
  python3 -u scripts/collect_data_v2.py "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

echo "[STEP 2] Refreshing account-risk model against the current corpus…" | tee -a "$LOG_FILE"
# Required before scoring, every run — not optional. final_score's rollup does
# an inner join against output/v2/account_risk_scores.csv; if this file is
# stale relative to whatever new accounts this month's collection brought in,
# those accounts silently vanish from the rollup instead of being scored.
# Reuses the already-fit, already-validated coefficients (no live API calls,
# no refit) — just rescales the current population against them, seconds to run.
python3 -u scripts/score_accounts.py 2>&1 | tee -a "$LOG_FILE"

echo "[STEP 3] Scoring…" | tee -a "$LOG_FILE"
python3 -u scripts/analyze_data_v2.py ${MONTH_ARG:+--month "$MONTH_ARG"} 2>&1 | tee -a "$LOG_FILE"

echo "[STEP 4] Generating site data…" | tee -a "$LOG_FILE"
# Deliberately NOT passing --month here: generate_site.py --month filters
# history.json to *only* that month, overwriting the full multi-month
# history instead of adding to it — confirmed by testing, not assumed. Every
# run rebuilds the complete site from whatever's in output/v2/ (cheap — no
# API calls, just re-reading already-computed analysis JSON), so the trend
# chart/leaderboard sparklines/drill-down never lose history.
python3 -u scripts/generate_site.py --v2 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Done. Push: git add docs/data_v2/ data/v2/ output/v2/ reports/findings.json && git commit -m 'Monthly V2 report' && git push ===" | tee -a "$LOG_FILE"
