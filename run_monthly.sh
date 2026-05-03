#!/usr/bin/env bash
# Monthly Reddit bot analysis pipeline.
# Run manually or via cron (e.g. first day of each month).
#
# Cron example (runs at 02:00 on the 1st of every month):
#   0 2 1 * * cd /path/to/reddit-bot-analysis-repo && bash run_monthly.sh >> logs/run.log 2>&1
#
# Usage: bash run_monthly.sh [--skip-collect]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

mkdir -p data output docs/data logs

LOG_FILE="logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "=== Reddit Bot Analysis — $(date) ===" | tee -a "$LOG_FILE"

SKIP_COLLECT=false
for arg in "$@"; do
  [[ "$arg" == "--skip-collect" ]] && SKIP_COLLECT=true
done

if $SKIP_COLLECT; then
  echo "[STEP 1] Skipping data collection (--skip-collect)" | tee -a "$LOG_FILE"
else
  echo "[STEP 1] Collecting data..." | tee -a "$LOG_FILE"
  python3 scripts/collect_data.py 2>&1 | tee -a "$LOG_FILE"
fi

echo "[STEP 2] Running analysis..." | tee -a "$LOG_FILE"
python3 scripts/analyze_data.py 2>&1 | tee -a "$LOG_FILE"

echo "[STEP 3] Generating site data..." | tee -a "$LOG_FILE"
python3 scripts/generate_site.py 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Done. Site data updated in docs/data/ ===" | tee -a "$LOG_FILE"
echo "    Push to GitHub to publish: git add docs/ && git commit -m 'Monthly report' && git push" | tee -a "$LOG_FILE"
