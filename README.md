# Reddit Bot Watch — Indian Subreddits

A monthly bot-activity monitoring system for 25 Indian subreddits. Scores each community 0–100 across four signal dimensions and publishes results as a static GitHub Pages dashboard.

**Live dashboard:** https://karmaneggs.github.io/RedditWatch/

---

## How it works

Each month, run:

```bash
bash run_monthly.sh
```

Three steps run automatically:

| Step | Script | Output |
|------|--------|--------|
| 1 | `collect_data.py` | `data/reddit_data_YYYYMMDD.csv` |
| 2 | `analyze_data.py` | `output/analysis_YYYYMMDD.json` |
| 3 | `generate_site.py` | `docs/data/YYYY-MM.json` + `history.json` |

Then push to publish:

```bash
git add docs/ data/ output/ logs/
git commit -m "Add YYYY-MM monthly report"
git push
```

To re-score existing data without a fresh Reddit pull:

```bash
bash run_monthly.sh --skip-collect
```

---

## Scoring system

Final score = weighted average of four components (0–100 each):

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| User accounts | 35% | Karma-per-day anomalies in posters + top 5 commenters per post |
| Engagement | 30% | Upvote-to-comment ratio (UCR); upvote consensus |
| Temporal | 20% | Posting hour concentration; entropy vs uniform 24 h distribution |
| Distribution | 15% | Coefficient of variation of vote counts (low CV = suspicious uniformity) |

### Severity bands

Calibrated to observed avg score ~30 (±10 band logic):

| Score | Label | Interpretation |
|-------|-------|----------------|
| 0–20 | **Low** | Organic or near-organic. Isolated elevated sub-scores explained by content type. |
| 20–40 | **Moderate** | Elevated patterns within the expected ±10 range of the ecosystem avg. Watch across months. |
| 40–70 | **High** | Multiple components elevated simultaneously. Sustained High is a strong signal. |
| 70+ | **Critical** | Highly anomalous across all dimensions. Immediate investigation warranted. |

### Component thresholds

**User accounts (35%)**
- kpd > 500 = suspicious (top ~10% of observed accounts)
- kpd > 2,000 = very suspicious (top ~5%)
- Account age < 90 days AND kpd > 100 = flagged
- Poster and commenter signals blended 60/40
- Calibrated on May 2026 data (717 accounts, median kpd = 22.5)

**Engagement (30%)**
- UCR (avg score ÷ avg comments) scales to 65 pts, ceiling at UCR = 30
- Upvote ratio > 85% scales to 35 pts

**Temporal (20%)**
- % posts in top-3 busiest hours → up to 60 pts
- Entropy deficit vs perfectly uniform → up to 40 pts (max entropy = 4.58 bits)

**Distribution (15%)**
- Vote CV < 0.8 → up to 70 pts (scores); comments CV < 0.8 → up to 30 pts

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

### Monthly JSON (`docs/data/YYYY-MM.json`)

```json
{
  "month": "2026-06",
  "analysis_date": "2026-06-02T17:13:03",
  "subreddits": {
    "IndiaCricket": {
      "final_score": 48.9,
      "user_score": 32.7,
      "engagement_score": 66.3,
      "temporal_score": 48.6,
      "distribution_score": 52.6,
      "details": { ... }
    }
  }
}
```

### History (`docs/data/history.json`)

Tracks all months with per-subreddit severity labels and aggregate stats (avg score, % Moderate+, % High+). Used by the trend chart.

---

## Dashboard (GitHub Pages)

The `docs/` folder is served as a static site. The dashboard includes:

- **Highlights panel** — avg score, tier counts, month-on-month gainers/losers, distribution health checks
- **Component breakdown** — per-subreddit stacked bar view across all four components
- **Trend chart** — avg score + % Moderate+ and % High+ over time
- **Score distributions** — histograms for final score (severity bands) and each component
- **Scoring rubric + methodology** — threshold documentation
- **Subreddit detail cards** — per-sub metrics with anomaly highlighting

View locally:

```bash
python3 -m http.server 8080 --directory docs/
# open http://localhost:8080
```

---

## Notes

- Data collection takes 10–15 min due to Reddit API rate limiting (OAuth: 60 req/min, throttled to 54)
- 30 top posts per subreddit per month are collected
- With `--skip-collect`, analysis + site generation takes ~3 seconds
- All credentials are gitignored via `.env`

---

**Last updated:** June 2026 · **Subreddits tracked:** 25
