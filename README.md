# Reddit Bot Watch — Indian Subreddits

A bot-activity monitoring system for 25 Indian subreddits. Scores each community 0–100 across five signal dimensions and publishes results as a static GitHub Pages dashboard.

**Live dashboard:** https://karmaneggs.github.io/RedditWatch/

> **V1 archived.** The original 4-component pipeline (`data/`, `output/*.json`, `docs/data/`, and the V1 scripts) has been moved to `archive/v1/` — superseded by V2's comment-ring detection and calibrated weights. See `archive/v1/README` context below. Everything in this doc describes the current V2 pipeline.

---

## How it works

Each month, run:

```bash
bash run_monthly.sh --v2                    # previous calendar month
bash run_monthly.sh --v2 --month 2026-07    # specific month
bash run_monthly.sh --v2 --year             # full rolling-year backfill (long-running)
```

Three steps run automatically:

| Step | Script | Output |
|------|--------|--------|
| 1 | `collect_data_v2.py` | `data/v2/posts_YYYY-MM.csv` + `data/v2/commenters_YYYY-MM.csv` |
| 2 | `analyze_data_v2.py` | `output/v2/analysis_YYYY-MM_<timestamp>.json` + `analysis_latest.json` |
| 3 | `generate_site.py --v2` | `docs/data_v2/YYYY-MM.json` + `history.json` + `findings.json` |

Then push to publish:

```bash
git add docs/data_v2/ data/v2/ output/v2/ reports/findings.json logs/
git commit -m "Add YYYY-MM V2 report"
git push
```

To re-score existing data without a fresh Reddit pull:

```bash
bash run_monthly.sh --v2 --skip-collect
```

There is currently **no automated schedule** — every run above is manual/on-demand. See "Next steps" in project notes for setting up a recurring monthly job.

---

## Scoring system

Final score = weighted average of five components (0–100 each). Weights are **calibrated from the data itself**, not hand-picked — see [Weight calibration](#weight-calibration) below.

| Component | Current weight | What it measures |
|-----------|-----------------|-------------------|
| Engagement | 41.9% | UCR (upvote:comment ratio), score↔comment correlation, upvote-ratio uniformity |
| Account | 29.9% | Karma-per-day anomalies in posters + top-10 commenters, new-account %, unverified email |
| Ring | 12.2% | Early-comment timing burst, sub-5-min first-comment rate, commenter recurrence across posts |
| Distribution | 8% | Coefficient of variation of scores/comment counts, comment depth |
| Temporal | 8% | Posting-hour concentration, entropy vs. uniform 24h distribution, interval regularity |

Severity bands: **0–20 Low · 20–40 Moderate · 40–70 High · 70+ Critical**

### Weight calibration

Weights come from `scripts/analysis.py`, which reads the full historical corpus (`data/v2/*.csv`), computes each raw signal's coefficient of variation across sub-months, averages within each component, and writes the result to `reports/findings.json`. `analyze_data_v2.py` loads those weights at runtime (falls back to defaults if the file is missing).

**Known limitation:** this calibration is unsupervised — there is no labeled bot/human ground truth anywhere in the pipeline, so "weight" currently reflects how much a signal *varies* across subreddits, not a validated measure of how well it *identifies bots*. Treat the weights as a reasonable prior, not a proven ranking. See project notes for planned validation work (labeled sample, PCA over raw signals, post/account-level anomaly detection).

`overlap_rate` was measured and removed from the ring component after it came back ~91.5% everywhere — that turned out to be an organic Reddit first-mover effect, not a bot signal (kept in the data as `overlap_rate` for reference, just not scored).

---

## Setup

### Requirements

- Python 3.8+
- Reddit OAuth app credentials ([create one here](https://www.reddit.com/prefs/apps) — choose "script" type)

### Install

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your Reddit credentials
```

### `.env` format

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=BotWatch/1.0 by YourUsername
GITHUB_PAT=your_pat_if_needed
```

### Subreddits tracked

Edit `subreddits.txt` — one subreddit name per line, lines starting with `#` are ignored. Currently tracking 25 Indian subreddits including r/india, r/indiaspeaks, r/IndiaCricket, r/BollyBlindsNGossip, r/IndianStockMarket, r/JEENEETards, and others.

---

## Output

### Per-month data (`data/v2/`)

- `posts_YYYY-MM.csv` — up to 40 top posts per subreddit per month, with author account signals (karma, age, verified email)
- `commenters_YYYY-MM.csv` — comment rows for the top-10-scoring posts per (sub, month), tagged `in_top10` / `in_first5` for ring detection

### Monthly analysis JSON (`docs/data_v2/YYYY-MM.json`)

Per-subreddit breakdown across all five components plus the unified `final_score`, written by `generate_site.py` from `output/v2/analysis_latest.json`.

### History (`docs/data_v2/history.json`)

Tracks all months with per-subreddit severity labels and aggregate stats. Used by the trend chart.

### Findings (`docs/data_v2/findings.json`)

Copy of `reports/findings.json` — the calibrated weights, key findings, and trend/acceleration alerts, published for the dashboard's findings panel.

---

## Dashboard (GitHub Pages)

The `docs/` folder is served as a static site (V2 only — see archive note above). The dashboard includes:

- **Highlights panel** — avg score, tier counts, month-on-month gainers/losers
- **Component breakdown** — per-subreddit stacked bar view across all five components
- **Trend chart** — avg score + % Moderate+ and % High+ over time
- **Score distributions** — histograms for final score (severity bands) and each component
- **Findings panel** — calibrated-weight rationale and key findings from the full-corpus analysis
- **Subreddit detail cards** — per-sub metrics with anomaly highlighting

View locally:

```bash
python3 -m http.server 8080 --directory docs/
# open http://localhost:8080
```

---

## Full-corpus statistical report

```bash
python3 scripts/analysis.py                        # all months in data/v2/
python3 scripts/analysis.py --months 2026-04 2026-05
```

Produces `reports/analysis_<timestamp>.pdf` (+ `analysis_latest.pdf`) covering data quality, signal distributions/correlations, text intelligence (near-dupes, cross-sub keyword spread), cross-sub network analysis (account overlap, churn, Gini concentration), an event calendar overlay, and the weight calibration that feeds `reports/findings.json`.

---

## Notes

- Year-mode collection (`--year`) fetches up to 1,000 posts/subreddit and every commenter profile it encounters — expect several hours for a full 25-sub backfill; it checkpoints and resumes on interruption
- With `--skip-collect`, analysis + site generation takes ~3 seconds
- All credentials are gitignored via `.env`

---

**Last updated:** July 2026 · **Subreddits tracked:** 25 · **Pipeline:** V2
