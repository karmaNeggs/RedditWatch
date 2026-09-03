# RedditWatch — monthly runbook

The project is complete. Ongoing work is one incremental run per month, then publish.
This file is the operational procedure. For *why* anything is the way it is, see
[`V3_PLAN.md`](V3_PLAN.md) (design record) and
[`docs/v3-research/whitepaper.md`](docs/v3-research/whitepaper.md) (methodology).

**Current series: v1.2.0 · pinned model: 1.1.0 · severity baseline: 1.2.0**

---

## When to run

Run for month `M` **a few days after `M` ends**, not on the 1st.

Collection takes each subreddit's top-100 posts *by karma*, and posts from the final days of
the month haven't accumulated their karma yet. Running on 2026-09-01 for August yielded only
39 posts dated Aug 31 versus ~170 on a typical day — a systematic under-sample of every
month's tail. Waiting ~3–5 days fixes it at no cost.

---

## The run

Five commands. Substitute the target month for `2026-08`.

```bash
# 0. Health check — which subs does Arctic Shift actually have data for this month?
#    Decides Tier-1 (Arctic Shift) vs Tier-2 (live Reddit) per sub. Writes output/v3/tracker_freshness.json
python3 scripts/v3_tracker_freshness.py

# 1. Collect the month (~20 min for 45 subs; resumable — completed cells are skipped)
python3 -u scripts/v3_collect.py --only-months 2026-08 --workers 4

# 2. Rebuild the corpus.  ALL FOUR, IN THIS ORDER.
python3 scripts/v3_stage0_build.py
python3 scripts/v3_account_features.py
python3 scripts/v3_botmarker_composite.py     # NOT optional — see warning below
python3 scripts/v3_feature_sanitise.py

# 3. Score + publish (loads the pinned model, appends the new month, rebuilds the dashboard)
python3 scripts/v3_stage8_monthly_refresh.py

# 4. Verify, then publish
git add -A && git commit -m "2026-08 monthly run" && git push
```

### ⚠️ `v3_botmarker_composite.py` is required, not optional

Its docstring calls it unsupervised prep work, but line 91 does
`CREATE OR REPLACE TABLE account_features AS SELECT af.*, bm.removal_rate_pctl, …` — it *adds*
the percentile columns. **Two of the deployed model's 10 features (`removal_rate_pctl`,
`thin_history_score`) exist only after it runs.** Skip it and `account_features` has 50 columns
instead of 58, and Stage 8 trains an **8-feature** model with no error and no warning. This
happened on 2026-09-01 by following an older rebuild chain that omitted it.

---

## What a correct run looks like

```
Loaded pinned model_version=1.1.0 (10 features, trained on n=684).
Scored ~38,000 accounts (>= 10 contributions) under model_version=1.1.0.
Suppressed N (sub, month, role) cell(s) below the MIN_SCORED_PER_CELL=15 floor.
Loaded frozen severity baseline 1.2.0 (2024-08..2026-08, n=1105).
Series v1.2.0: 1 month(s) appended (2026-09), 25 kept frozen.
```

Check all five:

| Line | Expected | If wrong |
|---|---|---|
| `Loaded pinned model` | **not** "Trained model_version=…" | You passed `--retrain`. Don't — see below. |
| feature count | **10** | `v3_botmarker_composite.py` didn't run. Redo step 2. |
| `Loaded frozen severity baseline` | **not** "Froze severity baseline …" | `output/v3/severity_baseline.json` is missing. Restore it; do not let it re-derive. |
| months appended | **exactly 1** (the new month) | More than 1 means a vintage directory is missing. Stop. |
| months kept frozen | all previously published | 0 frozen means you're rewriting history. Stop. |

Then open `docs/bot-spam-compass.html` and confirm the new month appears and the trend's earlier
months are unchanged.

---

## Rules that keep published numbers stable

The pipeline used to silently rewrite its own history: an ordinary refresh once changed **94.5%
of 1,076 published subreddit-months** and flipped **24.6% of severity labels**. Four causes were
fixed; these rules are what keep them fixed.

1. **Never pass `--retrain` on a monthly run.** It refits the model, which restates every month.
   Retraining is a deliberate release: bump `MODEL_VERSION`, bump `SERIES_VERSION`, publish a new
   vintage beside the old.
2. **Never delete `output/v3/severity_baseline.json`.** Bands are frozen to a declared window. A
   P50/P80/P95-of-observed band set is self-normalizing — 50/20/5% of months land in each band
   *by construction* — so re-deriving them would hide any real ecosystem-wide change.
3. **Never edit a published month file.** `docs/data_v3/v{series}/` is append-only. A methodology
   change bumps `SERIES_VERSION` and publishes a new vintage; it does not rewrite the old one.
4. **Figures are comparable within a series, never across.** v1.0.0, v1.1.0 and v1.2.0 all differ
   in level. Never plot them on one axis.

---

## Changing the methodology

If you genuinely need to change what's measured:

1. Make the change in `scripts/v3_stage8_monthly_refresh.py` (sampling depth, floors) or
   `v3_stage9_generate_dashboard_data.py` (bands, publishing).
2. Bump `SERIES_VERSION` in `v3_stage9_generate_dashboard_data.py`. Bump `MODEL_VERSION` in
   `v3_stage8_monthly_refresh.py` **only if the model itself changed** — v1.2.0 changed the
   sampling while keeping model 1.1.0, which is the point of pinning them separately.
3. Delete `output/v3/severity_baseline.json` **only** if the change shifts levels (it re-derives
   at the new `SERIES_VERSION`).
4. Run. Previous vintages must be untouched — verify with `git status docs/data_v3/v{previous}`.
5. Document the change *and its evidence* in `V3_PLAN.md` and whitepaper §8.

`MIN_SCORED_PER_CELL` and the sampling depth are **coupled**: at v1.1's top-30 sampling a floor
of 15 blanked 28.8% of poster cells; after widening it costs ~2%. Never move one without
re-checking the other.

---

## Layout

```
scripts/                     the 10 scripts the monthly run uses, and nothing else
  v3_tracker_freshness.py    step 0 — per-sub Tier-1/Tier-2 health check
  v3_collect.py              step 1 — Arctic Shift collection, resumable
  reddit_auth.py             Tier-2 fallback (live Reddit OAuth)
  v3_stage0_build.py         step 2 — raw .ndjson.zst -> DuckDB
  v3_account_features.py     step 2 — per-account feature table
  v3_botmarker_composite.py  step 2 — REQUIRED, adds percentile columns
  v3_feature_sanitise.py     step 2 — account_features_model
  v3_stage8_monthly_refresh.py   step 3 — score + prevalence, calls the two below
  v3_stage9_generate_dashboard_data.py   publishes vintages, embeds dashboard data
  v3_build_whitepaper_html.py            renders whitepaper.md -> .html

data/v3/       raw collection + DuckDB corpus
output/v3/     pinned model, its metadata, frozen baseline, prevalence table
docs/          the published site (GitHub Pages)
  data_v3/v1.0.0/ v1.1.0/ v1.2.0/   append-only vintages; flat *.json mirrors the current one
archive/       v1, v2, and the one-time v3 research that produced this model — not run
```

`V3_PLAN.md` is the design record: V1 → V2 → V3 → **v1.0**, i.e. the V3 plan is what produced
the current product. It is history *and* current rationale — read the "🚦 START HERE" block first.

---

## Known limitations

- **Scores shift as the corpus grows — by design.** An account is scored on its whole history,
  because "is this account bot-like" is a property of the account; the monthly number reflects
  *which* accounts were active that month. New evidence therefore improves old estimates — a
  revision, not a bias — and append-only vintages publish that honestly. Don't try to "fix" it by
  windowing feature history; that was checked and rejected (see `V3_PLAN.md`).
- **Month-tail under-sampling** if you run too early — see "When to run".
- **Reliability is not validity.** The v1.2.0 sampling is more self-consistent (split-half 0.81,
  persistence 0.85) but that does not prove it better predicts real bot activity. The n=684 label
  set is account-level and cannot settle a subreddit-level question.
- **`r/DesiVideoMemes`** is confirmed dead (no posts since 2026-03-19); `r/unitedstatesofindia`
  needs the direct-search path permanently. Both are flagged in `subreddits_v3.csv`.
