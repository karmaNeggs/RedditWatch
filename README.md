# RedditWatch 1.0

A bot/spam-influence tracker for 45 India-focused subreddits. Scores the accounts behind each subreddit's best-performing content each month against an account-removal model trained and cross-validated on **684 real, live-checked Reddit accounts** (banned / deleted / still active — not a proxy label), and publishes results as a static GitHub Pages dashboard.

**Live site:** https://karmaneggs.github.io/RedditWatch/ · **Dashboard:** [`docs/bot-spam-compass.html`](docs/bot-spam-compass.html) · **Methodology:** [`docs/methodology.html`](docs/methodology.html) · **Whitepaper:** [`docs/v3-research/whitepaper.html`](docs/v3-research/whitepaper.html)

10 deduplicated features (down from ~67 candidates), tuned XGBoost, repeated 5×10-fold CV AUC 0.793 ± 0.054 (re-validated 2026-09-01 on the 25-month corpus; imputed within-fold). Subreddit-level prevalence reports posters, commenters, and combined as three separate metrics (checked directly: they carry meaningfully different risk, not assumed), measured over each subreddit's top-100 posts by karma and the top/latest 10 commenters on each. Full 25-month history (2024-08→2026-08), published as **append-only vintages** under `docs/data_v3/v{series}/` — current series **v1.2.0**, with v1.1.0 and v1.0.0 preserved verbatim. Monthly refresh is `python3 scripts/v3_stage8_monthly_refresh.py`: it loads a **pinned** model and appends the new month without rewriting published ones. Retraining requires an explicit `--retrain`; a methodology change bumps the series version and publishes a new vintage beside the old rather than restating history. Figures are comparable within a series, never across. Full methodology, every number, and honest limitations: [`docs/v3-research/whitepaper.md`](docs/v3-research/whitepaper.md). **Start at `V3_PLAN.md` → the "🚦 START HERE" block** for exactly where things stand and what's next.

> **V1 and V2 are archived, not deleted.** They're retained below for provenance and because their underlying pipelines (`scripts/collect_data_v2.py` etc.) still exist and technically still run — but neither is the live methodology, and the site no longer links to V2's report page. V1's 4-component pipeline (`data/`, `output/*.json`, `docs/data/`) lives in `archive/v1/`. V2 (comment-ring detection, calibrated logistic-regression weights, plateaued at ROC-AUC 0.663 — an account-level ceiling per the literature) is documented in the rest of this file as a historical record of how the project got here. Don't extend either without checking `V3_PLAN.md` first.

---

## How it works

Each month, run:

```bash
bash run_monthly.sh --v2                    # previous calendar month
bash run_monthly.sh --v2 --month 2026-07    # specific month
bash run_monthly.sh --v2 --year             # full rolling-year backfill (long-running)
```

Four steps run automatically:

| Step | Script | Output |
|------|--------|--------|
| 1 | `collect_data_v2.py` | `data/v2/posts_YYYY-MM.csv` + `data/v2/commenters_YYYY-MM.csv` |
| 2 | `score_accounts.py` | `output/v2/account_risk_scores.csv` — refreshes the account-risk model against whatever's in the corpus *now*, including any new accounts this month's collection just brought in |
| 3 | `analyze_data_v2.py` | `output/v2/analysis_YYYY-MM_<timestamp>.json` + `analysis_latest.json` |
| 4 | `generate_site.py --v2` | `docs/data_v2/YYYY-MM.json` + `history.json` + `findings.json` |

Step 2 is required every run, not optional — `analyze_data_v2.py`'s scoring does
an inner join against `account_risk_scores.csv`; if it's stale relative to the
current corpus, new accounts silently vanish from that month's score instead
of being counted. It reuses the already-fit coefficients (no live API calls,
seconds to run) — it isn't a refit, just a rescale of the current population.

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

`final_score` = the % of a (subreddit, month)'s posting/commenting activity
coming from accounts in **that month's own top risk-decile**, per a
validated model — not a blend of hand-weighted heuristic components (that
approach was tried, tested against real evidence, and replaced; see below).

**The model**: ~10 account-level features (account age, log-transformed
karma/day, log-transformed link-karma ratio, comment velocity, ring-timing
signals, etc.) → one `LogisticRegression` → a 0–100 risk score per account.
Fit against the one real label this project has — whether a random sample of
accounts is currently suspended/gone from Reddit, a weak but real proxy for
"bad account." Current cross-val AUC ≈ 0.66 (0.5 = coin flip) — a real,
modest signal, not a confident classifier; treat every score as directional,
not a verdict. Exact current coefficients, AUC, and a genuine forward-looking
backtest (does an early reading predict later attrition, not just describe
concurrent decline?) are on the live **Methodology** page — that page reads
`reports/findings.json` directly, so it never goes stale the way a hardcoded
number in this README would.

**"High risk"** is computed fresh each month — the top decile of *that
month's own active accounts*, not a fixed all-time population cutoff. An
earlier version used a fixed global threshold and it produced a near-
monotonic 13-month score climb driven almost entirely by the active
population trending younger over time, not by any real change in relative
risk (confirmed: the climb hit all 25 subreddits in lockstep regardless of
topic, and the underlying population's median account age genuinely fell
over the same window). Month-relative thresholding fixed that.

**Severity bands** (Low/Moderate/High/Critical) are percentiles of the
actual observed score distribution across every (subreddit, month) scored so
far — recalibrated by `score_accounts.py` on every run, not fixed cutoffs.
Current values are on the **Report** page's "how to read this" box.

Six legacy heuristic components (account/ring/engagement/temporal/
distribution/network — what `final_score` used to be, hand-weighted) are
still computed and published, but purely as descriptive context, not inputs
— they were never individually validated against real evidence. See the
Report page's "Descriptive signals" section for the honest distinction.

`overlap_rate` and `avg_comment_depth` were measured and dropped after
testing showed them either organic (not a bot signal) or genuinely inert
(zero effect on the fitted model) — see `scripts/anomaly_detection.py`'s
`FEATURE_COLS` comment for the full history of what's been tried and why
each change was made or rejected.

### Refitting the model

`scripts/scale_weak_labels.py` does the actual fit (live Reddit API calls to
check account status — checkpointed/resumable, since a large sample takes
hours; re-run the same command to continue). Run this far less often than
monthly — it's a real time investment, not a routine step:

```bash
python3 scripts/scale_weak_labels.py --n 8000              # first call starts, re-run to resume/continue
python3 scripts/scale_weak_labels.py --refit-only           # refit on already-collected labels + current FEATURE_COLS, no API calls — use this when testing feature changes
python3 scripts/backtest_predictive.py                      # refresh the early/late predictive-validity check
python3 scripts/score_accounts.py                           # apply the refit coefficients to the current population
```

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

- `posts_YYYY-MM.csv` — up to 40 top posts per subreddit per month (plus a
  ~20-post random supplement from `/new.json` for months collected after the
  survivorship-bias fix, tagged `sample_type` — calibration-only, never part
  of the reported score), with author account signals (karma, age, verified
  email)
- `commenters_YYYY-MM.csv` — comment rows for the top-10-scoring posts per (sub, month), tagged `in_top10` / `in_first5` for ring detection

### Monthly analysis JSON (`docs/data_v2/YYYY-MM.json`)

Per-subreddit `final_score` (the validated account-risk rollup) plus the six
legacy components as diagnostic detail, plus a `narrative` block (headline,
toppers/movers, curated findings) — written by `generate_site.py` from
`output/v2/analysis_latest.json`.

### History (`docs/data_v2/history.json`)

Tracks all months with per-subreddit severity labels and aggregate stats. Used by the trend chart.

### Findings (`docs/data_v2/findings.json`)

Copy of `reports/findings.json` — model coefficients, AUC, severity bands, and the predictive backtest, published for the Methodology page.

---

## Dashboard (GitHub Pages)

The `docs/` folder is served as a static site — `index.html` is the root
page GitHub Pages serves at https://karmaneggs.github.io/RedditWatch/. Both
pages read live data from `docs/data_v2/` at load time; neither hardcodes a
number that can go stale.

- **`index.html`** ("Report") — headline banner (biggest single move this
  month), full ranked leaderboard (all 25 subs, severity-colored, per-row
  sparkline), a subreddit-drill-down trend view, toppers/movers, curated
  findings, validated composite trends, the legacy components as clearly-
  labeled descriptive context, and a glossary
- **`methodology.html`** — the model's actual coefficients (chart), current
  AUC, the early/late predictive backtest, severity-band derivation, and an
  explicit "what's validated vs. what's descriptive" explanation

The earlier dashboard (`index.html`/`insights.html` built for the old
weighted-component scoring system) has been removed — still in git history
if needed, but no longer published.

View locally:

```bash
python3 -m http.server 8080 --directory docs/
# open http://localhost:8080
```

---

## Full-corpus statistical report

Run this on a slower cadence than the monthly collection — quarterly is
reasonable, since it re-derives its own diagnostics from the *entire*
history and would otherwise chase single-month noise.

```bash
python3 scripts/analysis.py                        # all months in data/v2/
python3 scripts/analysis.py --months 2026-04 2026-05
```

Produces `reports/analysis_<timestamp>.pdf` (+ `analysis_latest.pdf`)
covering data quality, signal distributions/correlations, text intelligence
(near-dupes, cross-sub keyword spread), cross-sub network analysis (account
overlap, churn, Gini concentration), and an event calendar overlay. **Note:**
this script still writes `calibrated_weights`/`pca_weights` keys into
`reports/findings.json` from an earlier scoring approach — `analyze_data_v2.py`
no longer reads them for anything; the model's real coefficients live in
`account_model` (written by `score_accounts.py`/`scale_weak_labels.py`,
[see Scoring system](#scoring-system)). Harmless to run, just a vestigial
output worth knowing isn't load-bearing anymore.

---

## Notes

- Year-mode collection (`--year`) fetches up to 1,000 posts/subreddit and every commenter profile it encounters — expect several hours for a full 25-sub backfill; it checkpoints and resumes on interruption
- With `--skip-collect`, `run_monthly.sh` (collect skipped + rescore + site) takes well under a minute
- `scale_weak_labels.py` (the model refit) is checkpointed the same way — a
  large `--n` can take hours of live API calls; re-running the identical
  command resumes from where it left off rather than restarting
- All credentials are gitignored via `.env`

---

**Last updated:** July 2026 · **Subreddits tracked:** 25 · **Pipeline:** V2
