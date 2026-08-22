# RedditWatch 1.0 — Scoring Bot and Spam-Worker Activity on India-Focused Subreddits

**Status:** this is the current, validated methodology. Two earlier approaches (duplicate-text
detection, then an unvalidated behavioral equation) are summarized in §2 as background — what was
tried, what worked, what didn't, and why the project moved on. Everything from §4 onward is what
RedditWatch 1.0 actually runs: an account-removal model trained and cross-validated against real,
live-checked Reddit account status (banned / deleted / still active), not a proxy label.

- **684 real, live-verified accounts** (450 active, 117 banned, 116 deleted, 1 mod), checked directly by
  the analyst across ten sampling rounds — the label set an earlier duplicate-text screen or hand-built
  equation had no hand in constructing.
- **10 features**, arrived at through five rounds of reduction (8 exact duplicates, 9 near-duplicates,
  backward elimination 49→18, a wall-clock time-denominator bug found and fixed that collapsed 3 more
  into duplicates (→15), then a further backward-elimination pass to 10 to keep the model as light as
  possible — §5, §6b) — down from an original ~67-column candidate pool.
- **Tuned XGBoost, repeated 5×10-fold CV AUC = 0.780 ± 0.035** — clubbed banned+deleted target,
  `account_ordinal` excluded. Full validation: `docs/v3-research/charts/model_analysis.html`.
- **Live dashboard:** `docs/bot-spam-compass.html`. **Monthly refresh, one command:**
  `python3 scripts/v3_stage8_monthly_refresh.py`.

---

## 1. Corpus

- **1,619,492 comments**, full body text, from `commenters_dedup` (collected via `scripts/v3_collect.py`
  against the Arctic Shift API).
- **347,886 distinct commenting accounts**, 45 India-focused subreddits, 2024-08 through 2026-07 (24
  months).
- `account_features` (`scripts/v3_account_features.py` + `scripts/v3_botmarker_composite.py`): one row
  per account, ~70 behavioral/timing/reception/username columns computed from the full comment+post
  history — the source table every model in this project draws from.

## 2. Earlier approaches — what didn't make the cut

Two approaches were tried and set aside before landing on the account-removal model below. Both taught
real lessons that shaped RedditWatch 1.0's methodology; neither is the live scoring method.

**Duplicate-text bot detection.** A SQL screen over `commenters_dedup` for accounts posting the same
body text (≥30 chars) across ≥2 different posts found 385 candidates; LLM batch review classified them
against a further 266-account calibration set, yielding **76 confirmed bots** (templated/referral spam,
ban-evasion sockpuppet chains — e.g. a confirmed `CritFin` → `Critifin` → `criti_fin` chain posting
identical political talking points under three names). An XGBoost classifier on this set reached
**5-fold CV AUC 0.792** — a real result, but narrow in scope: it only catches accounts dumb or lazy
enough to reuse exact text, missing any human-written spam. n=76 is also small (fold AUCs ranged
0.72–0.87). Superseded, not disproven — this species of bot still exists in the corpus, just not the
only one worth catching.

**A behavioral equation for paid spam workers.** The working hypothesis: beyond templated bots, a
second population — human workers paid to promote content (political IT cells, PR firms) — writes
original text but shares a behavioral signature: high activity, karma-farming in a "home" subreddit
while getting downvoted but still posting in an "opposition" one (because they're paid regardless of
reception). A hand-built equation (`(comment_spread + post_spread) × (activity rate)`, AND-gated —
either factor at zero zeroes the score) was calibrated against three hand-picked subreddit groups and
did separate them as predicted — but every attempt to check it against a real labeled set came back at
or below random: **AUC 0.470** against a 20-account LLM-confirmed spam set, and **AUC 0.319** (worse
than random) ranking Phase 1's 76 confirmed bots. Conclusion at the time, still true: the equation and
the duplicate-text screen measure different populations, and neither validates against the other's
labels — but neither is a validated classifier on its own terms either. The equation's only real
evidence was a qualitative group separation that was **never checked against ground truth**, which is
exactly the gap RedditWatch 1.0 closes.

**Why this matters for what follows:** both approaches shared a failure mode — real, defensible-sounding
heuristics that never got checked against actual removed/active outcomes. RedditWatch 1.0's entire
premise is closing that gap first, then building features and model choices on top of a label set that
means something.

## 3. Feature-engineering groundwork

Before the final feature set (§5) was locked in, an exploratory pass characterized every candidate
`account_features` column individually and pairwise: univariate distribution shape per feature, a full
Spearman correlation ranking of every pair, and an unsupervised HDBSCAN segmentation check on the
strongest correlated pairs — `docs/v3-research/eda/index.html` (Account Feature Survey) and
`docs/v3-research/eda/stage2.html` (Bivariate & Segmentation). This is where the discipline of "check
correlation before trusting two differently-named columns are independent" — used repeatedly in §5 and
§6b — started; it's exploratory scaffolding, not a separate validated result, but the redundancy
patterns it surfaced early (several account_features columns visibly moving together) directly
motivated the systematic dedup passes that followed.

---

# RedditWatch 1.0: the account-removal model

This is a different foundation from §2: instead of a proxy label or an unvalidated behavioral
hypothesis, the label is **real, live-checked Reddit account status** — banned, deleted, or still
active — checked directly against live Reddit profiles, not inferred from anything in the corpus.

## 4. Ground truth

**750 candidate accounts** sampled across ten rounds — ranked by whatever candidate scoring function
existed at that point (early karma-churn composites, later the model itself, finally two boundary-region
samples straddling the tier cutoffs), each round excluding every account shown in an earlier round — and
checked one by one against live Reddit. **684 usable**:

| outcome | n |
|---|---|
| still active (`ok`) | 450 |
| banned | 117 |
| deleted | 116 |
| moderator (kept as `ok`-adjacent, excluded from the binary target) | 1 |

**Banned and deleted accounts are combined into one "removed" positive class (n=233).** Checked
directly at every sample-size milestone (n=234→294→354→414→534→584→684): splitting them and treating banned
as the sole positive class consistently underperforms clubbing (§6: AUC 0.780 clubbed vs. 0.645
banned-only). A companion check — deleted-only vs. everyone else — found deleted is actually the
*stronger* standalone signal (0.749 vs. banned-only's 0.645): banned is the noisier target on its own
(often triggered by a single rule-violating incident that aggregate behavioral features can't always
anticipate), while deletion tends to follow a more gradual, cumulative pattern the features can see.
Clubbing wins because the two share enough common signal that combining them gives the model more
positive examples of that shared pattern — not because a strong signal is masking a weak one.

Full labeled set: `output/v3/ground_truth_labels.csv`. A parallel `output/v3/all_shown_accounts.csv`
tracks every account ever sampled, including untagged ones, so later rounds never repeat an account.

## 5. Features

**10 features**, arrived at through five rounds of reduction over the `account_features` pool plus a
few hand-built candidates (a handful of churn/karma-spread metrics, all of which lost to existing
columns measuring the same constructs better and were dropped entirely):

1. **8 exact duplicates** — `account_features`'s `*_1` variants, ρ=1.000 with their base column.
2. **9 near-duplicates**, found via a correlation sweep on the top-20 important features rather than
   assuming different names meant different information: e.g. `n_comments_sample`/`n_gaps`/
   `n_threads_active`/`n_distinct_threads` all ρ≥0.994 with each other; `removal_rate_pctl`/
   `removal_rate` ρ=1.000. Kept the higher-importance member of each cluster.
3. **Backward elimination, 49→18** — drop the single lowest-importance feature, refit, repeat. AUC held
   flat (even ticked up) all the way down to ~17 features before eroding below ~10, confirming most of
   the 49 were redundant, not additive. One manual correction on top: the 18-feature set kept both
   `removal_rate_pctl` and `deleted_later_rate_pctl` (ρ=0.886, above this project's own 0.85 threshold)
   — a direct bivariate check showed `removal_rate_pctl` carries more solo signal (AUC 0.625 vs. 0.590),
   so `deleted_later_rate_pctl` was cut by hand.
4. **A real bug, found during QC, fixed — see §6b** — 3 of those 18 features (`days_since_first_seen`,
   `comments_per_day_since_first_seen`, `karma_per_day_since_first_seen`) turned out to be computed with
   a wall-clock time artifact. Fixing it made all three exact duplicates of already-kept columns, so
   they were pruned the same way as any other duplicate, landing at 15 (CV AUC 0.789 ± 0.041).
5. **A second backward-elimination pass, 15→10**, run to check how far the model could be lightened
   further. AUC held with only a small, real cost below ~12 features (0.789→~0.776–0.778, consistent
   across n=10/11/12, not noise), with clean correlation throughout (max ρ=0.73, well under the 0.85
   threshold). **Final: 10 features, CV AUC 0.780 ± 0.035** — accepted as the right tradeoff for a
   simpler, more interpretable model.

Also excluded on purpose:

- **`account_ordinal`** (account creation order) — a real early signal, but at final scale it doesn't
  even help nominally (0.774 with vs. 0.780 without — §6), and multiple manually-verified samples (up
  to 120 accounts, live-checked) found the model separates high from low risk at least as well without
  it. Not worth the interpretability cost of a feature that reads as "just flags new accounts."
- **Collection-snapshot timing fields** (`last_seen_utc`, `first_seen_utc`, `observed_span_days`) — a
  removed account mechanically stops appearing in the corpus, so these encode a symptom of removal, not
  a behavioral precursor to it.

The final 10, ranked by importance, with the full correlation matrix behind the pruning:
`docs/v3-research/charts/model_analysis.html` §3, `output/v3/final_top20_correlation.csv`.

## 6. Model and validation

XGBoost, tuned via a 20-config random search (`max_depth=5, n_estimators=150, learning_rate=0.1,
min_child_weight=1, subsample=0.9, colsample_bytree=0.7, reg_lambda=1`), `scale_pos_weight` for the
202:381 class imbalance, validated by repeated stratified cross-validation (5 folds × 10 repeats = 50
fold-evaluations), with per-fold imputation (train-fold medians only, no leakage from the held-out fold):

| check | result |
|---|---|
| **Repeated CV AUC (headline number)** | **0.780 ± 0.035** |
| Multifold-averaged out-of-fold AUC (10-repeat average per account) | 0.792 |
| Deleted-only vs. everyone else | 0.749 |
| Banned-only vs. everyone else | 0.645 |
| With `account_ordinal` included | 0.774 (no better) |

**Independent, non-CV confirmation — confirmed-outcome rate by score bucket** (the literal % of
live-verified accounts in each decile that turned out banned/deleted, not a density curve):

| score bucket | % still active |
|---|---|
| 0.0–0.1 | 93.0% |
| 0.4–0.5 | 57.1% |
| 0.9–1.0 | 17.0% |

A clean, mostly-monotonic decline. The same shape shows up in every manually-verified high/mid/low tier
check run across this project, culminating in the largest (n=120): **85% bad in the high tier, 25% in
mid, 2.5% in low**.

Full analysis (target/feature comparison, feature-count elimination curve, importances, ROC,
confirmed-rate-by-bucket): `docs/v3-research/charts/model_analysis.html`.

**Reading this number honestly:** 0.78 is real, validated signal, checked against ground truth the
detection method itself had no hand in constructing, at a scale (n=684, 202 positive) large enough that
the estimate has stopped swinging with each new sampling round. It is not a highly confident classifier
— std of ±0.04 across folds means individual predictions should be read as directional risk, not a
verdict.

## 6b. A calibration bug found and fixed: wall-clock time decay

While QC-checking the subreddit-prevalence dashboard (§8), the 24-month trend showed a smooth,
near-monotonic decline in average prevalence — 20.8% (2024-08) down to 3.4% (2026-07) — with the most
recent month landing entirely in the "Low" severity band. That pattern turned out to be a real bug, not
a real trend.

**Root cause:** three features — `days_since_first_seen`, `comments_per_day_since_first_seen`,
`karma_per_day_since_first_seen` — were computed in `scripts/v3_account_features.py` using
`epoch(now())`, wall-clock "today" at whenever the pipeline last ran, as their time denominator, instead
of the account's actual last-observed activity (`last_seen_utc`). Every refresh, "now" advances for
every account, so these per-day rates mechanically shrank release-over-release — for every account,
including ones banned or inactive long ago — independent of any real behavior change. Because the
subreddit-prevalence pipeline scores each author once per refresh and applies that single static score
to every historical month they appear as a top-30 influencer, this dragged the entire 24-month trend
down uniformly with each re-run: not because the ecosystem got safer, but because the corpus got older.

Two sibling features, `comments_per_day_observed` and `sample_score_per_day_observed`, already used the
correct denominator (`last_seen_utc − first_seen_utc`) and were unaffected — which is what made the bug
easy to isolate: the buggy trio was the odd one out.

**Fix:** changed the three formulas to match the already-correct pattern. This makes them mathematically
identical to already-kept columns (`days_since_first_seen` becomes an exact duplicate of the
already-excluded `observed_span_days`; the two rate features become exact duplicates of
`comments_per_day_observed`/`sample_score_per_day_observed`), so they were pruned rather than kept —
taking the feature count from 18 to 15 (§5). Rebuilt `account_features` and `account_botmarker_composite`,
retrained: **AUC moved from an inflated 0.801 to the correct 0.789** — a small, honest drop, not a
regression; the earlier number was measuring a model that partly worked by exploiting a timestamp
artifact. A further lightening pass (§5 step 5) took the deployed model from 15 to the **final 10
features, AUC 0.780 ± 0.035**.

**Effect on the dashboard trend, before vs. after:**

| | before (buggy) | after (fixed) |
|---|---|---|
| 2024-08 avg prevalence | 15.4%* | 15.4% |
| Trend shape | smooth decay toward ~3% | dip to ~10.4% (mid-2025), recovers to ~13.9% (2026-07) |
| Latest month severity | uniformly Low | mixed, matching historical range |

*(early months are less affected since less wall-clock time had passed at the point the bug's effect
was smallest; the divergence grows with each refresh.)*

**Why this is worth documenting rather than quietly fixing:** it's a concrete illustration of a general
risk in any pipeline that re-scores historical data using present-day features — a static "as of today"
snapshot applied retroactively to a multi-year trend will silently encode the pipeline's own age as
signal unless every time-denominated feature is anchored to the event being scored, not to whenever the
job happened to run. Every rate feature in the final 15 (§5) was re-checked against this specific
failure mode after the fix; none of the survivors share it.

## 7. Population scoring and the activity floor

Scored on every account with **≥10 total contributions** (comments + posts) — 36,762 of 347,886
accounts. Below that floor, several features are too noisy to trust; those accounts are excluded from
scoring entirely, not assigned a default score. Model artifact: `output/v3/final_bot_model.json`. Full
population scores: `output/v3/final_bot_scores.parquet`.

## 8. Subreddit-level prevalence: the top-30-post methodology

Rather than "share of a month's total activity" (dominated by whichever way the bulk of ordinary
commenters leans), prevalence is measured through **influence over each subreddit's best-performing
content** — the accounts Reddit's own ranking already surfaced as consequential that month:

1. Take each subreddit-month's **top 30 posts by karma**.
2. Pull the **poster**, the **top-5 highest-scoring commenters**, and the **top-5 most recent
   commenters** — deduplicated into one "influencer set" per subreddit-month.
3. Score every influencer with the model above (§6), among those meeting the activity floor (§7).
4. Report the **% of scored influencers landing ≥0.7 predicted-removal probability** ("high-risk") per
   subreddit-month, alongside the mean score and coverage (% of the influencer set that could be scored
   at all).

Full 24-month history (2024-08→2026-07). 1,076 subreddit-month rows, 45 subreddits. Data:
`output/v3/subreddit_bot_prevalence_mom.csv`. Severity bands (Low/Moderate/High/Critical) are
percentiles (P50/P80/P95) of the observed prevalence distribution, recalculated fresh at every refresh
— not fixed cutoffs.

## 9. The dashboard

**`docs/bot-spam-compass.html`** — the project's primary artifact: per-subreddit trend (full 24-month
history), a sortable all-45-subreddit leaderboard with sparklines, MoM movers/toppers, and a findings
narrative, all built to the same visual language as the project's earlier V1/V2 dashboards. Fully
self-contained (data embedded at build time, no external fetch) — works opened directly from disk, via
a local server, or on GitHub Pages alike.

**Monthly refresh, one command:** `python3 scripts/v3_stage8_monthly_refresh.py` — retrains on the
current label set, rescoring the full population, rebuilding the subreddit-month prevalence table, and
regenerating the dashboard's embedded data in place. Takes under a minute.

## 10. Outputs

- `output/v3/ground_truth_labels.csv` — the 684-account labeled ground truth.
- `output/v3/all_shown_accounts.csv` — every account ever sampled, used to avoid resampling.
- `output/v3/final_bot_model.json` / `final_bot_model_features.json` — the trained model (10 features).
- `output/v3/final_bot_scores.parquet` — scores for all 36,762 floor-qualifying accounts.
- `output/v3/subreddit_bot_prevalence_mom.csv` — the full 24-month subreddit prevalence table.
- `docs/bot-spam-compass.html` — the dashboard.
- `docs/v3-research/charts/model_analysis.html` — full validation analysis.
- `scripts/v3_stage8_monthly_refresh.py` — the one-command monthly refresh pipeline.

## 11. Limitations

- **n=684 (233 positive) is real progress but still not large** for a 10-feature model — repeated-CV
  fold AUCs vary meaningfully across folds (std ±0.04). Treat scores as directional, not a verdict.
- **"Removed" mixes admin bans and self-deletions**, and §4/§6 show these aren't interchangeable —
  deleted is the stronger standalone signal, banned the noisier one — even though clubbing them
  outperforms either alone. Confirmed at real scale, not a small-sample artifact.
- **This model predicts "will this account get removed," not "is this account a coordinated
  spam/political operation."** Those are correlated but not identical questions. Nothing here
  establishes coordination between accounts — that remains an open thread from the original Phase 2
  hypothesis (§2), not yet revisited with real validation methodology.
- **India-subreddit-specific** — every feature and label comes from 45 India-focused subreddits.
- **The 0.7 high-risk threshold (§8) is a round-number operational choice**, not fit to any external
  criterion. Revisit once the labeled set is large enough to calibrate it against a target
  false-positive rate.
- **The §6b fix addressed the one wall-clock time-denominator bug found during this project's QC pass**,
  not an exhaustive audit of every feature for similar artifacts. The general risk it illustrates —
  present-day features silently encoding pipeline age when applied to historical scoring — is worth
  re-checking whenever a new time-denominated feature is added.
