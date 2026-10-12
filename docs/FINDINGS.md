# Bot Activity in Indian Subreddits — Methodology & Findings

**Project:** [RedditWatch](https://karmaneggs.github.io/RedditWatch/) · **Tracked:** 25 Indian subreddits · **Period:** May–June 2026

---

## What This Is

A monthly automated scan of the 25 largest Indian subreddits, scoring each community on signals consistent with coordinated inauthentic behaviour — bot accounts, vote manipulation rings, and astroturfing. Every score is reproducible: raw data and analysis scripts are public in this repository.

This is not a definitive verdict on any subreddit. It is a monitoring system — repeated High scores over multiple months are a signal worth investigating; a single month is context.

---

## Scoring System (V1 — May & June 2026)

Four components, each scored 0–100:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| User + Commenter | 40% | Karma-per-day anomalies in post authors and top 5 commenters. Calibrated to observed distribution: median kpd = 22.5, p90 = 600, p95 = 2,400 |
| Engagement | 25% | Upvote-to-comment ratio (UCR); upvote ratio variance (uniform ratios across posts = suspicious) |
| Temporal Patterns | 20% | Posting hour concentration; entropy vs. uniform distribution; inter-post interval regularity |
| Score Distribution | 15% | Coefficient of variation of vote counts (very low CV = suspiciously uniform) |

**Severity bands** (calibrated to observed average ~29):

| Score | Label | Interpretation |
|-------|-------|----------------|
| 0–20 | Low | Organic or near-organic patterns |
| 20–40 | Moderate | Elevated but within expected ecosystem range |
| 40–70 | High | Multiple signals elevated simultaneously |
| 70+ | Critical | Not yet observed in tracked data |

---

## Key Findings — June 2026

| Rank | Subreddit | Score | Notes |
|------|-----------|-------|-------|
| 1 | r/IndiaCricket | **47.3** HIGH | 20% of posts have both suspicious poster AND suspicious commenters (fully coordinated). Active IPL season context. |
| 2 | r/indiasocial | **43.9** HIGH | Extreme upvote-to-comment ratio (32.3×). High engagement with minimal discussion. |
| 3 | r/unitedstatesofindia | **43.9** HIGH | 16.7% fully coordinated posts. Low upvote ratio variance (uniform voting). |
| 4 | r/indiaspeaks | **43.3** HIGH | Score-comment correlation: 0.23 (votes without discussion). Upvote ratio std: 0.016 — eerily uniform across all posts. |
| 5 | r/ipl | **40.2** HIGH | High posting hour concentration during match windows (temporal score 73). |

**Ecosystem average:** 28.0 / 100 across 25 subreddits.  
**76% of subreddits** score Moderate or above (≥20). **20% score High** (≥40).

### Notable structural findings

**Cross-subreddit coordination:** 18 accounts posted in 2+ tracked subreddits in June. 2 accounts (`Broad-Research5220`, `GiveMeSomeSunshine3`) posted in 3+ subreddits — the strongest coordination signal available at post level.

**Engagement without discussion:** r/indiaspeaks and r/unitedstatesofindia show the clearest disconnect between upvote counts and comment activity. Posts receive high uniform upvotes but generate proportionally little discussion — consistent with vote ring behaviour rather than organic interest.

**Political sub pattern:** The three highest-scoring non-sports subs (indiasocial, unitedstatesofindia, indiaspeaks) all show different political alignments but similar manipulation patterns. The signal appears across the political spectrum, not concentrated in one direction.

---

## What We Are Not Measuring (Honestly)

- **Individual post authors are not named or accused.** The analysis is at subreddit aggregate level.
- **High UCR can be organic.** Sports and entertainment subs naturally attract more upvotes than comments during events.
- **Low score-comment correlation has alternative explanations.** Study resources (r/UPSC) or question-answer subs may have this pattern organically.
- **30 posts/month is a small sample.** Distribution metrics (CV, correlation) are noisy at this sample size.
- **We cannot distinguish sophisticated bot rings from coordinated human networks.** The signals detect coordination, not automation specifically.

---

## Data Collection

Each month: top 30 posts from the last 30 days per subreddit, collected via Reddit OAuth API (60 req/min, throttled to 54). Post author and top 5 commenter account data fetched for each post.

**V1 collected fields per post:** score, upvote_ratio, num_comments, created_utc, author total_karma, account_age_days, commenter avg_kpd, commenter suspicious count.

---

## V2 — In Development

A more robust collection schema is being developed (`collect_data_v2.py`) that addresses the main V1 limitations:

**New data per run:**
- Individual commenter rows (not aggregated) — enables recurrence and cohort analysis
- Top 10 commenters by score AND first 5 by timestamp — detects comment rings directly
- `link_karma` / `comment_karma` separated — karma farmer signal
- `comment_created_utc` — early burst detection (ring comments arrive within minutes)
- `author_verified_email` — bot accounts rarely verify

**New signals V2 will score:**
- Comment ring rate (% overlap between early and promoted commenters per post)
- Account creation cohort detection (batch-created accounts posting together)
- % posts/comments from new accounts (<90 days old)
- Commenter recurrence (same accounts across 3+ posts in same sub)
- Early comment timing burst (std of first-comment timestamps)

V2 backfill covers January–June 2026. Scores will be published alongside V1 for comparison.

---

## Reproducibility

```bash
git clone https://github.com/karmaNeggs/RedditWatch
pip install -r requirements.txt
cp .env.example .env          # add Reddit OAuth credentials
bash run_monthly.sh           # collect + score + publish
```

All raw data, analysis code, and historical scores are in this repository. The scoring formula is fully documented in `scripts/analyze_data.py`.

---

*Last updated: June 2026. Data and methodology subject to revision as calibration improves.*
