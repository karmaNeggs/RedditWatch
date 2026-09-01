# RedditWatch

A bot/spam-influence tracker for 45 India-focused subreddits. Scores the accounts behind each subreddit's best-performing content each month against an account-removal model trained and cross-validated on **684 real, live-checked Reddit accounts** (banned / deleted / still active — not a proxy label), and publishes results as a static GitHub Pages dashboard.

**Live site:** https://karmaneggs.github.io/RedditWatch/ · **Dashboard:** [`docs/bot-spam-compass.html`](docs/bot-spam-compass.html) · **Methodology:** [`docs/methodology.html`](docs/methodology.html) · **Whitepaper:** [`docs/v3-research/whitepaper.html`](docs/v3-research/whitepaper.html)

**Current series `v1.2.0` · pinned model `1.1.0` · 45 subreddits · 25 months (2024-08 → 2026-08)**

---

## Running it

The project is complete. Ongoing work is **one incremental run per month**, then publish.

👉 **[`RUNBOOK.md`](RUNBOOK.md) is the operational procedure.** Start there — it has the five
commands, what a correct run looks like, and the rules that keep published numbers stable.

```bash
python3 scripts/v3_tracker_freshness.py                          # 0. per-sub health check
python3 -u scripts/v3_collect.py --only-months YYYY-MM --workers 4   # 1. collect (~20 min)
python3 scripts/v3_stage0_build.py                               # 2. rebuild corpus…
python3 scripts/v3_account_features.py
python3 scripts/v3_botmarker_composite.py                        #    …REQUIRED, see RUNBOOK
python3 scripts/v3_feature_sanitise.py
python3 scripts/v3_stage8_monthly_refresh.py                     # 3. score + publish
```

Run a few days *after* the month ends — top-post karma needs time to settle, or the month's final
days are under-sampled.

---

## What it measures

Prevalence is **influence over each subreddit's best-performing content**, not share of total
activity. Per subreddit-month: take the **top 100 posts by karma**, pull each post's author plus its
**10 highest-scoring and 10 most-recent commenters**, deduplicate into one influencer set, score
every member, and report the **% landing ≥0.7 predicted-removal probability**.

Posters, commenters, and combined are reported as three separate metrics — checked directly, they
carry meaningfully different risk (posters ~22.6% high-risk vs commenters ~14%), so each gets its own
severity bands. Cells with fewer than 15 scored accounts are suppressed rather than published.

**The model:** 10 deduplicated features (from ~67 candidates), tuned XGBoost, repeated 5×10-fold CV
**AUC 0.793 ± 0.054**. Trained on real live-checked account status, not a proxy. Full validation:
[`docs/v3-research/charts/model_analysis.html`](docs/v3-research/charts/model_analysis.html).

---

## How published numbers stay stable

Until v1.1 every monthly refresh silently rewrote the whole published history — one ordinary run
changed **94.5% of 1,076 published subreddit-months** and flipped **24.6% of severity labels**. Four
causes were found and fixed (unconditional retraining, a train/serve imputation skew, an unpinned
random seed, and untiebroken ranking that made runs irreproducible from identical data).

Three rules now hold, enforced in code:

- **The model is pinned.** The monthly path loads `output/v3/final_bot_model_meta.json` and refuses
  to retrain implicitly. Retraining needs an explicit `--retrain` and a version bump.
- **Severity bands are frozen** to a declared window in `output/v3/severity_baseline.json`. Bands
  re-derived per refresh are self-normalizing — 50/20/5% of months land in each band *by
  construction* — so a genuine ecosystem-wide rise would be invisible.
- **Publishing is append-only.** `docs/data_v3/v{series}/` is never rewritten. A methodology change
  bumps `SERIES_VERSION` and publishes a **new vintage beside the old**.

**Figures are comparable within a series, never across.** v1.0.0, v1.1.0 and v1.2.0 all differ in
level — never plot them on one axis.

| series | what changed |
|---|---|
| `v1.0.0` | the originally published series |
| `v1.1.0` | full August; model pinned, medians persisted, seed and ranking made deterministic |
| `v1.2.0` | sampling widened to top-100 posts + 10/10 commenters; min-n floor 5 → 15 |

---

## Layout

```
RUNBOOK.md     the monthly procedure — start here
V3_PLAN.md     design record and rationale (see the "🚦 START HERE" block)
scripts/       the 10 scripts the monthly run uses, and nothing else
data/v3/       raw collection + DuckDB corpus
output/v3/     pinned model + metadata, frozen baseline, prevalence table
docs/          the published site; data_v3/v*/ are the append-only vintages
archive/       v1, v2, and the one-time v3 research — retained, not run
```

**On the naming:** the project went V1 → V2 → V3 → **v1.0**. "V3" is not a superseded branch — the
V3 plan is what *produced* the current product, which is why the live pipeline scripts are named
`v3_*`. `archive/v3-research/` holds only the one-time analysis that led to the model, not the model
itself.

**`archive/` is retained for provenance, not deleted.** V1's 4-component pipeline and V2
(comment-ring detection, calibrated logistic regression, plateaued at ROC-AUC 0.663) both still
exist there and technically still run, but neither is the live methodology and the site links to
neither. Don't extend either without reading `V3_PLAN.md` first.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # Reddit OAuth creds, only needed for the Tier-2 fallback path
```

Collection uses the [Arctic Shift](https://arctic-shift.photon-reddit.com) API and needs no
credentials. `scripts/reddit_auth.py` (live Reddit OAuth) is the Tier-2 fallback for subreddit-months
Arctic Shift hasn't indexed — `v3_tracker_freshness.py` decides which subs need it.

To preview the site locally:

```bash
python3 -m http.server 8080 --directory docs   # then open http://localhost:8080
```

The dashboard embeds its data at build time, so `docs/bot-spam-compass.html` also opens directly
from `file://` with no server.

---

## Known limitations

- **History still moves underneath a series.** `account_features` aggregates each account's whole
  corpus history, so adding a month changes scores for already-published months. Append-only
  publishing contains the symptom, not the cause — and it means a published 2024-09 figure uses
  behavior observed through 2026-08 (look-ahead bias). Fix is point-in-time features. Not done.
- **Reliability is not validity.** v1.2.0's sampling is more self-consistent (split-half 0.81,
  month-to-month persistence 0.85), but that does not prove it better predicts real bot activity.
  The n=684 label set is account-level and cannot settle a subreddit-level question.
- **The model doesn't establish coordination** between accounts, only individual removal risk.
- **Scores are directional, not verdicts.** AUC 0.793 is a real signal, not a confident classifier.

---

**Last updated:** September 2026 · **Subreddits tracked:** 45 · **Series:** v1.2.0
