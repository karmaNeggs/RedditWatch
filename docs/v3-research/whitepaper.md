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
- **Tuned XGBoost, repeated 5×10-fold CV AUC = 0.793 ± 0.054** — clubbed banned+deleted target,
  `account_ordinal` excluded. Full validation: `docs/v3-research/charts/model_analysis.html`.
- **Live dashboard:** `docs/bot-spam-compass.html`. **Monthly refresh, one command:**
  `python3 scripts/v3_stage8_monthly_refresh.py`.

---

## How to read this document

> **In one sentence:** we taught a computer to guess whether a Reddit account is likely to get banned
> or deleted, by showing it hundreds of real accounts whose fate we already checked by hand, then used
> that to answer "which India-focused subreddits currently have the most bot/spam-like accounts behind
> their most-visible posts and comments?"

This is a technical record, written to be checked and reproduced, not a casual explainer — but every
recurring term is defined once, here, so the rest of the document doesn't require outside knowledge.
If a paragraph gets dense later on, it's worth a scroll back to this list rather than a search engine.

- **Feature** — one measurable fact about an account: how many comments it posts per day, how old it
  is, how spread out its karma is across subreddits, and so on. The model looks at 10 of these per
  account (§5).
- **Ground truth** — real, independently-checked outcomes (here: an account's actual live status on
  Reddit) used to test whether a model's guesses are correct. The opposite of a *proxy* — a stand-in
  signal (like "posts the same text twice") that might correlate with the real thing but isn't it.
- **AUC (ROC-AUC)** — a single 0.5-to-1.0 score for how well a model tells two outcomes apart. 0.5 is a
  coin flip; 1.0 is perfect separation; every AUC number in this document is on that scale.
- **Cross-validation (CV)** — the honesty check for AUC: split the labeled accounts into groups, train
  on most of them, test on the group held back, then repeat with a different group held back each time.
  "Repeated 5×10-fold CV" means 5 groups, repeated 10 times with different random splits — 50 separate
  tests averaged together, so one lucky or unlucky split can't skew the number.
- **Correlation (ρ, "rho")** — how tightly two numbers move together, from 0 (unrelated) to 1
  (practically identical). Used throughout §5 to catch features that are secretly measuring the same
  underlying thing under two different names.
- **Backward elimination** — a feature-trimming method: start with every candidate feature, repeatedly
  remove whichever single one contributes least, retrain, and watch whether accuracy holds up.
- **Score bucket / decile** — accounts grouped by their predicted-risk score into 10 equal ranges
  (0.0–0.1, 0.1–0.2, …), used to check that a higher score really does mean more real-world removals,
  not just a number that looks plausible.
- **Prevalence** — the subreddit-level number this project ultimately reports: the % of a subreddit's
  most-visible accounts that month (its top posters and commenters) that the model flags as high-risk.
- **Severity band** — Low / Moderate / High / Critical, computed once from the spread of observed
  prevalence scores over a declared baseline window and then **frozen** (§8b), not recalculated each
  refresh and not a cutoff decided in advance.
- **Poster vs. commenter** — two different roles within a subreddit's influencer set: the person who
  *submitted* a top post, versus the people who *commented* on it. §8 found these carry meaningfully
  different risk, which is why the dashboard reports them separately.

Everything from §4 onward is the live, current methodology. §1–3 are kept as a record of two earlier
approaches that were tried, checked against real evidence, and set aside — worth reading for context on
what didn't work and why, but not what the dashboard runs today.

## 1. Corpus

- **1,684,682 comments**, full body text, from `commenters_dedup` (collected via `scripts/v3_collect.py`
  against the Arctic Shift API).
- **360,431 distinct commenting accounts**, 45 India-focused subreddits, 2024-08 through 2026-08 (25
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

*In short: before a model can learn what a risky account looks like, it needs correct, real-world
examples to learn from. This section is where those examples came from — checked by hand, not guessed.*

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

<!--CHART:target-->

## 5. Features

*In short: more features aren't automatically better — many measure almost the same underlying thing
in different words, and duplicate information doesn't help a model, it just adds noise. This section is
the search for the smallest set of genuinely different, genuinely useful facts about an account.*

**10 features**, arrived at through five rounds of reduction over the `account_features` pool plus a
few hand-built candidates (a handful of churn/karma-spread metrics, all of which lost to existing
columns measuring the same constructs better and were dropped entirely):

1. **8 exact duplicates.** Eight columns turned out to be the same fact stored twice under two
   different names (`account_features`'s `*_1` variants, ρ=1.000 — perfectly identical). Dropped.
2. **9 near-duplicates.** A correlation sweep on the top-20 important features — checking each one
   against every other for hidden overlap, rather than trusting that different names meant different
   information — found 9 more columns measuring almost the same thing under different names. The
   clearest example: how many comments an account has, how many pauses appear in its posting timeline,
   how many discussion threads it's active in, and how many distinct threads it's touched all moved
   together (ρ≥0.994) — four ways of asking "how much does this account post," not four separate facts.
   Similarly, an account's raw removal rate and its percentile rank for removal rate were literally
   identical (ρ=1.000). Kept the single highest-importance member of each overlapping cluster.
3. **Backward elimination, 49→18.** Repeatedly drop whichever single feature contributes least, retrain,
   repeat. Accuracy held flat — even ticked up — all the way down to ~17 features before it started to
   genuinely erode below ~10, confirming most of the original 49 were redundant, not adding anything
   new. One manual correction on top: the resulting 18-feature set kept both an account's removal-rate
   percentile and a near-identical column for "removal rate that happens later, after a delay"
   (ρ=0.886, above this project's own 0.85 overlap threshold) — a direct check showed the percentile
   version carries more standalone signal (0.625 vs. 0.590 accuracy on its own), so the other was cut
   by hand.
4. **A real bug, found during QC, fixed — see §6b.** Three of those 18 features — how long the account
   has existed, how fast it comments, and how fast it earns karma — turned out to be computed against
   the wrong reference point in time, an artifact that silently deflated their values every time the
   data was refreshed. Fixing it made all three exact duplicates of already-kept columns, so they were
   pruned the same way as any other duplicate, landing at 15 (accuracy 0.789 ± 0.041).
5. **A second backward-elimination pass, 15→10**, run to check how much further the model could be
   lightened. Accuracy held with only a small, real cost below ~12 features (0.789→~0.776–0.778,
   consistent across repeated tests — not noise), and every remaining pair of the 10 stayed under the
   0.85 overlap threshold (max ρ=0.73). **Final: 10 features, accuracy 0.780 ± 0.035** (measured on the 24-month corpus this pass ran against; the same model re-validates at 0.793 ± 0.054 on the current 25-month corpus — §6) — accepted as the
   right tradeoff for a simpler, easier-to-reason-about model.

Also excluded on purpose:

- **`account_ordinal`** (account creation order) — a real early signal, but at final scale it doesn't
  even help nominally (0.774 with vs. 0.780 without — §6), and multiple manually-verified samples (up
  to 120 accounts, live-checked) found the model separates high from low risk at least as well without
  it. Not worth the interpretability cost of a feature that reads as "just flags new accounts."
- **Collection-snapshot timing fields** (`last_seen_utc`, `first_seen_utc`, `observed_span_days`) — a
  removed account mechanically stops appearing in the corpus, so these encode a symptom of removal, not
  a behavioral precursor to it.

**What the final 10 actually look at, in order of importance** — each is a genuine fact about an
account's Reddit history, not an abstract statistic:

| what it looks at | column name |
|---|---|
| How many comments it posts per day, averaged over the whole time we've observed it active | `comments_per_day_observed` |
| How many subreddits kept downvoting it on average for months at a time — and it kept posting there anyway | `n_subs_rejected_but_returned` |
| How sparse or thin its overall posting history looks — a high score means we've barely seen this account do anything | `thin_history_score` |
| How many of its own posts actually got any comments back | `n_own_posts_with_comments` |
| How much of its activity happens in subreddits we've flagged as high political-manipulation-incentive (national news and politics) | `n_high_tier` |
| Same, but for medium-incentive subreddits (city/regional politics) | `n_medium_tier` |
| Roughly how much karma it earns per day of activity | `sample_score_per_day_observed` |
| How long its comments tend to be, on average | `mean_body_len` |
| Where it ranks against every other account for how often its comments get taken down | `removal_rate_pctl` |
| Its average reception in the one subreddit that likes it least | `worst_sub_mean_score` |

Full importance ranking (as a chart), and the correlation matrix behind every pruning decision above:
`docs/v3-research/charts/model_analysis.html` §3, `output/v3/final_top20_correlation.csv`.

<!--CHART:elimination-->

## 6. Model and validation

*In short: this is the honesty check. Anyone can claim a model works — this section is the repeated,
held-out testing that backs the claim up, plus a second, independent way of confirming the same thing.*

XGBoost (a well-established, tree-based prediction algorithm — not a novel or exotic choice), tuned via
a 20-config random search (`max_depth=5, n_estimators=150, learning_rate=0.1, min_child_weight=1,
subsample=0.9, colsample_bytree=0.7, reg_lambda=1`), `scale_pos_weight` for the 233:451 class imbalance
(233 removed accounts vs. 451 still-active/moderator), validated by repeated stratified cross-validation
(5 folds × 10 repeats = 50 fold-evaluations), with per-fold imputation (train-fold medians only, no
leakage from the held-out fold):

| check | result |
|---|---|
| **Repeated CV AUC (headline number)** | **0.793 ± 0.054** (v1.1; 0.780 ± 0.035 as published in v1.0) |
| Multifold-averaged out-of-fold AUC (10-repeat average per account) | 0.792 |
| Deleted-only vs. everyone else | 0.749 |
| Banned-only vs. everyone else | 0.645 |
| With `account_ordinal` included | 0.774 (no better) |

<!--CHART:roc-->

**Independent, non-CV confirmation — confirmed-outcome rate by score bucket** (the literal % of
live-verified accounts in each decile that turned out banned/deleted, not a density curve):

| score bucket | % still active |
|---|---|
| 0.0–0.1 | 93.0% |
| 0.4–0.5 | 57.1% |
| 0.9–1.0 | 17.0% |

<!--CHART:bucket-->

A clean, mostly-monotonic decline. The same shape shows up in every manually-verified high/mid/low tier
check run across this project, culminating in the largest (n=120): **85% bad in the high tier, 25% in
mid, 2.5% in low**.

Full analysis (target/feature comparison, feature-count elimination curve, importances, ROC,
confirmed-rate-by-bucket): `docs/v3-research/charts/model_analysis.html`.

**Reading this number honestly:** 0.78 is real, validated signal, checked against ground truth the
detection method itself had no hand in constructing, at a scale (n=684, 233 removed accounts) large
enough that the estimate has stopped swinging with each new sampling round. It is not a highly confident
classifier — a standard deviation of ±0.04 across folds means individual predictions should be read as
directional risk, not a verdict. A useful mental anchor: 0.5 would be a coin flip, 1.0 would be perfect.

## 6b. A calibration bug found and fixed: wall-clock time decay

*In short: while double-checking the subreddit dashboard, a genuine bug was found — a few features were
quietly getting worse at their job every time the corpus was refreshed, for a boring technical reason
that had nothing to do with real-world behavior. Here's what happened and how it was caught and fixed.*

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
features, AUC 0.780 ± 0.035** on the 24-month corpus of the time; re-validated at 0.793 ± 0.054 on the current 25-month corpus.

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

Scored on every account with **≥10 total contributions** (comments + posts) — 38,197 of 360,227
accounts. Below that floor, several features are too noisy to trust; those accounts are excluded from
scoring entirely, not assigned a default score. Model artifact: `output/v3/final_bot_model.json`. Full
population scores: `output/v3/final_bot_scores.parquet`.

## 8. Subreddit-level prevalence: the top-post methodology

*In short: the account-level model (§4–6) scores one account at a time. This section is about turning
that into a subreddit-level number — how do you go from "here's one account's risk score" to "how bad
is r/example this month?" without just averaging every comment ever posted there, most of which nobody
sees?*

Rather than "share of a month's total activity" (dominated by whichever way the bulk of ordinary
commenters leans), prevalence is measured through **influence over each subreddit's best-performing
content** — the accounts Reddit's own ranking already surfaced as consequential that month:

1. Take each subreddit-month's **top 100 posts by karma** (the collector's `role='top'` set; its
   separately-tagged ~20-post counter-sample, drawn from below the top 100 as a control group, is
   excluded).
2. Pull the **poster**, the **10 highest-scoring commenters**, and the **10 most recent commenters**
   on each — deduplicated into one "influencer set" per subreddit-month.
3. Score every influencer with the model above (§6), among those meeting the activity floor (§7).
4. Report the **% of scored influencers landing ≥0.7 predicted-removal probability** ("high-risk") per
   subreddit-month, alongside the mean score and coverage (% of the influencer set that could be scored
   at all).

Cells with **fewer than 15 scored accounts** are suppressed rather than published — below that the
ratio is noise, not a measurement (one cell previously published a figure derived from a single
account, reading 100% in one release and 0% in the next).

**Sampling widened in v1.2.0 (2026-09-01), and why it is not a change of subject.** v1.0/v1.1 used the
top 30 posts and 5+5 commenters — a fraction of what the collector already stored, at no saving, since
the wider data was on disk. Three checks were run before adopting the change. (1) *Same population:*
poster high-risk rate is flat across post rank (22.62% / 22.99% / 22.56% for ranks 1–30 / 31–60 /
61–100), so the added posts carry the same risk as the elite slice. (2) *Better instrument:*
split-half reliability rose **0.68 → 0.81** and month-to-month persistence **0.73 → 0.85** — at the old
sampling roughly a third of the variance in a published figure was measurement noise, and persistence
is independent confirmation since noise cannot persist month over month. (3) *Leaderboard:* mean top-10
overlap between the definitions is 7.0/10, which is what reliability 0.68 predicts — about three
leaderboard slots per month were being decided by noise. The minimum-n floor moved 5→15 in the same
release, and only because widening lifted the poster 10th percentile from 16 to 42; at the old sampling
a floor of 15 would have blanked 28.8% of poster cells. **The floor and the sampling width are
coupled.**

The cost is a level shift: ecosystem mean prevalence falls ~1.5pp (12.72% → 11.26%). That is a
recalibration of the same quantity, not a decline in bot activity — figures are comparable *within* a
series, never across. One limit worth stating plainly: reliability is a property of the sampling, not
of ground truth. These checks show the widened metric is more self-consistent; they do **not** prove it
better predicts real bot activity. The n=684 label set is account-level and cannot settle a
subreddit-level question.

Full 25-month history (2024-08→2026-08). 1,120 subreddit-month rows, 45 subreddits. Data:
`output/v3/subreddit_bot_prevalence_mom.csv`. Severity bands (Low/Moderate/High/Critical) are
percentiles (P50/P80/P95) of the prevalence distribution over a **declared baseline window**, computed
once and frozen in `output/v3/severity_baseline.json` — not recalculated at every refresh (§8b).

**Posters and commenters are reported separately, not just pooled.** They carry meaningfully
different risk: **posters average ~22.6% high-risk vs. commenters' ~14%**, consistently in every
month — the opposite of the intuitive read that top posters skew toward established/organic
contributors. That cross-subreddit, high-volume, opportunistic posting pattern is precisely what the
account-level model's features (activity rate, tier-climbing, thin history) are built to catch.

**A previously recorded explanation for this gap has been checked and withdrawn.** The 2026-08-22
write-up attributed it to the *eliteness* of the top-30 slice — that reaching "top-30-by-karma"
specifically is what repeat karma-farming accounts chase. Measured directly against the corpus
(2026-09-01), poster high-risk rate is **flat across post rank**: 22.62% at ranks 1–30, 22.99% at
31–60, 22.56% at 61–100. The poster/commenter gap is real and replicates; the top-30-specific
mechanism does not. Whatever drives it operates across the whole top-100, not at the very top. Genuine moderator announcements mostly don't even appear here — pinned/informational
posts rarely rank near the top *by karma*, so "top poster" and "mod" overlap less than intuition suggests.
A separate check for volume-driven skew (the concern that periods of elevated overall activity might
shift who clears the top-30 cutoff) found no effect: correlation between a subreddit-month's influencer
volume and its prevalence rate is ~0.05 (noise), and a subreddit's own high-activity months read
essentially the same as its low-activity months (12.6% vs. 13.1%) — the always-exactly-rank-30
selection (not an absolute karma threshold) already self-adjusts for this.

Because posters (~10% of the pooled influencer set by volume) and commenters (~90%) carry different
signal, the combined/pooled metric mostly tracks the larger commenter population — reported alongside,
not replacing, the two separate readings. The dashboard exposes all three (posters / commenters /
combined) with independently-calibrated severity bands for each, since a shared band set would
misrepresent severity for whichever role sits off that band's center (e.g. a 25% poster reading is
unremarkable against a ~21% poster average, but 25% for commenters — whose average sits near 12% — is
Critical).

**A fourth, content-moderation signal was checked and one of four candidates held up.** The question was
whether subreddit-level activity/moderation volume (MoM post-count changes, content removal counts,
ban counts) should factor into the prevalence score. Checked directly (2026-08-23, 1,114
subreddit-months): **comment self-deletion rate — the % of a subreddit's sampled comments the author
deleted themselves — correlates moderately with account risk (Spearman ρ=0.32 vs. combined prevalence)**,
real signal and clearly not a restatement of the same thing (ρ nowhere near 1.0). It's now reported as
its own column, not blended into the risk score — same transparency-over-black-box reasoning as the
poster/commenter split. Three other candidates were checked and dropped: post- and comment-level
*moderator/Reddit* removal rates correlate weakly and in the *wrong* direction (ρ=−0.13, −0.12, +0.06)
— more active moderation reads as *lower* apparent risk, plausibly because it's catching problem content
before its authors accumulate enough karma to become "influencers," not because removal itself signals
risk. "Total posts per month" as a volume trend was ruled out before testing: the `posts` table is
capped at ~120/sub/month by collection design (top-100-by-score + a small counter-sample), so any trend
computed from it would mostly reflect the collection cap, not real subreddit activity. Per-subreddit,
per-month *ban* counts aren't available at all — Reddit's API doesn't expose "banned from r/X on date Y,"
and this project's ground truth is Reddit-wide status as of now, not a per-subreddit ban timeline.

**A second sub-level signal — subscriber growth — was checked and also held up, more weakly.**
Month-over-month % change in the subreddit's subscriber count (a real, uncapped number tracked per
post, unlike the post-count cap above) correlates with account risk at ρ=0.18 vs. combined prevalence
— weaker than comment self-deletion (ρ=0.32) but real, not noise. The shape is informative: it shows up
on the commenter side (ρ=0.18) but barely on the poster side (ρ=0.03) — fast subscriber growth
associates with more risky *commenting* activity, not a different mix of who reaches the top of the
leaderboard. Raw subscriber count alone barely matters (ρ=0.09) — size isn't the signal, the rate of
change is. Added as its own column alongside comment self-deletion rate. A related idea — total or
average comment volume on the top-30 posts, as a "traction" signal — was checked and **not** added: its
raw level is largely redundant with comment self-deletion rate (ρ=0.38 between the two, higher than
either's correlation with account risk), and its month-over-month growth carries essentially no signal
(ρ≈0.00) — a single viral thread can swing a subreddit's monthly comment total either direction,
independent of anything about account risk. "Views" were considered and ruled out immediately: Reddit's
public API doesn't expose post view/impression counts to third-party collection at all.


## 8b. Why published numbers no longer change: frozen vintages

*In short: until v1.1, every monthly refresh silently rewrote the entire published history. A reader
who screenshotted the trend in August and returned in September saw a different past. This section is
what was wrong and what was done about it.*

**The measurement.** Comparing the published series immediately before and after the 2026-08-22
refresh — an ordinary monthly run, no methodology change:

| Effect on the 1,076 already-published subreddit-months | |
|---|---|
| Values that changed at all | **94.5%** (1,017) |
| Mean absolute revision | **1.74 pp** |
| Largest single revision | **18.18 pp** |
| Published severity labels that flipped | **24.6%** (265) |
| Ecosystem trend peak-to-trough | **4.90 pp → 6.34 pp** |

That last row matters most: the headline "dip and recover" shape became **29% more dramatic** without a
single new fact about 2024 or 2025. Part of the story the chart told was produced by recomputation.

**Where it came from.** Re-scoring under frozen-versus-moving bands isolates the cause: values moving
accounted for 24.4% of the label flips, severity bands moving for only 3.4%. Four distinct drivers were
found, all fixed in v1.1:

1. **The model was retrained on every refresh.** New coefficients, new scores for every account, every
   month restated. Retraining now requires an explicit `--retrain` and a `MODEL_VERSION` bump.
2. **Imputation used a moving reference — and was skewed.** Missing features were filled with the
   *labeled set's* median at training time but the *full population's* median at scoring time. Beyond
   drifting each refresh, this is a train/serve skew: the model was trained to read "missing" as one
   value and served another. Medians are now fit once and persisted with the model.
3. **The fit was stochastic.** `subsample`/`colsample_bytree` with no `random_state`, so two retrains
   on identical data produced different trees. Seed pinned.
4. **Ranking was non-deterministic.** Posts and commenters were ranked with `row_number() OVER
   (… ORDER BY score DESC)` and no tiebreaker. Re-running the identical query against the identical
   database pulled between **4 and 22** counter-sample posts into the "top-30" set across five
   consecutive executions, because **99 of 1,120 sub-months (8.8%)** have a score tie exactly at the
   30/31 cutoff. Published figures were not reproducible even from unchanged data. Fixed with a stable
   tiebreaker, plus an explicit `role = 'top'` filter — the collector's ~20-post counter-sample is a
   control group drawn from *below* the top 100 and was never meant to enter the influencer set.

**Severity bands are frozen, and this is not cosmetic.** Bands defined as P50/P80/P95 *of the observed
distribution* are self-normalizing: exactly 50%/20%/5% of subreddit-months land in moderate/high/
critical **by construction, forever**. Had ecosystem-wide bot prevalence doubled, the bands would have
doubled with it and the dashboard would have looked identical — a relative ranking presented to readers
as an absolute severity scale. Bands are now computed once over a declared window and frozen in
`output/v3/severity_baseline.json`, with the window and method published in every month file.

**Publishing is append-only.** A month file, once written under `docs/data_v3/v{SERIES_VERSION}/`, is
never rewritten. Each refresh still recomputes everything (the new month has to come from somewhere),
but only genuinely new months are persisted; recomputed values for already-published months are
discarded. `history.json` is rebuilt from the *published* files rather than the recompute — otherwise
month files would freeze while the trend chart above them kept moving. A deliberate methodology change
does not edit these files: it bumps `SERIES_VERSION` and publishes a new vintage alongside the old, so
a restatement is always visible as a new series rather than a number that quietly changed. Both series
ship: `docs/data_v3/v1.0.0/` is the previously published series, preserved verbatim.

**v1.1 is itself a restatement, and is labelled as one.** Fixing the four drivers above necessarily
moves numbers: against v1.0.0 the mean absolute revision is 1.65 pp and 21.5% of severity labels
change. That is why it ships as its own vintage rather than as a refresh of v1.0.0. Figures are
comparable *within* a series, not across series.

**Why scores still shift between series — and why that is correct.** `account_features` describes an
account over its entire history, so extending the corpus can change an account's score, including for
months already published. This is deliberate, and it is worth being precise about why it is not a flaw.

The account score answers *"is this account bot-like?"* — a property of the **account**, best estimated
from everything known about it. The subreddit-month number answers *"which accounts were active here
that month?"* — a property of the **month**. Two different axes. When an account active in 2024-09 is
later banned, it was already a bad account in 2024-09; we simply hadn't learned it yet. Folding that in
makes the 2024-09 estimate **more** accurate, not less. This is a **revision**, in the sense a
statistical agency revises GDP — not look-ahead bias, which is a flaw specific to claiming you could
have *predicted* something in real time. RedditWatch describes the past; it makes no real-time claim.

Append-only vintages are the honest way to publish that: within a series numbers never move, and a
better estimate ships as a new series beside the old rather than silently overwriting it.

**Checked and rejected (2026-09-03): restricting features to a trailing window.** Windowing was
measured, not assumed. On the same accounts it costs nothing (+0.030 / −0.028 / +0.004 / −0.021 AUC at
6 / 9 / 12 / 18-month windows — all inside the noise band), but it drops **76% of banned/deleted labels
at a 6-month window**: removed accounts stop posting, so a trailing window deletes precisely the
positive class it needs. More fundamentally it would discard real evidence to simulate ignorance,
degrading exactly the historical estimates it claims to protect.


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
- `output/v3/final_bot_scores.parquet` — scores for all 38,197 floor-qualifying accounts.
- `output/v3/subreddit_bot_prevalence_mom.csv` — the full 25-month subreddit prevalence table.
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
