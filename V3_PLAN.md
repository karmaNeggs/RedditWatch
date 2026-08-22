# V3 plan — exposure-weighted inauthenticity scoring

Supersedes `V3_FEATURE_PLAN.md` and `V3_DATA_SOURCES.md`. Those remain valid as
background; where they disagree with this document, this document wins.
`V3_METRIC_CATALOGUE.md` is the full candidate-metric sweep this document's §4
was checked against and updated from (2026-08-05) — kept separate because of
its size, not because it's optional background.

Status marks: ✅ **measured** in this project · 📚 **published**, cited ·
🔬 **hypothesis**, untested · ❌ **unavailable**, confirmed by audit.

---

## 🚦 START HERE (next session) — as of 2026-08-22

**Read this section only.** Everything referenced here has full detail, numbers,
and the reasoning behind every correction in §10.4 — don't re-derive anything
below from scratch, and don't re-litigate anything marked resolved.

### Current primary thread: RedditWatch 1.0 — account-removal model validated against real ground truth — 2026-08-22

**Full writeup: `docs/v3-research/whitepaper.md` §4–11. This is the project's primary bot/spam
scoring methodology** — supersedes the earlier duplicate-text and equation-based approaches for
dashboard/prevalence purposes (both kept as documented background in whitepaper §2, not the live
method). Short version:

- **Real ground truth, not a proxy:** 684 usable accounts (450 active, 117 banned, 116 deleted, 1 mod),
  each checked directly against live Reddit by the analyst across ten sampling rounds — the first
  label set in this project not itself a product of the detection method being validated.
  `output/v3/ground_truth_labels.csv`.
- **Deleted accounts statistically resemble banned accounts, not active ones** — reconfirmed at final
  scale: clubbed (banned+deleted vs ok) AUC 0.780 vs. banned-only 0.645. A companion check found
  deleted-only is actually the *stronger* standalone signal (0.749 vs banned-only's 0.645) — banned is
  the noisier target on its own, not a weak signal riding on deleted's coattails. Combine them; don't
  split them.
- **10 features**, arrived at via five rounds of reduction: 8 exact (ρ=1.000) duplicates of
  `_1`-suffixed variants, 9 near-duplicates (ρ>0.85) found by a correlation sweep on the top-20
  important features, backward elimination 49→18, then **a real wall-clock time-denominator bug found
  during QC and fixed** — 3 of the 18 (`days_since_first_seen`, `comments_per_day_since_first_seen`,
  `karma_per_day_since_first_seen`) were computed against "now" instead of `last_seen_utc`, silently
  decaying every account's score release-over-release as the corpus aged (this is what caused the
  subreddit-prevalence dashboard's MoM trend to look like a fake decline toward "everything's fine
  lately" — see whitepaper §6b). Fixing it collapsed those 3 into duplicates of already-kept columns,
  landing at 15 — then a further backward-elimination pass (user request: keep it to ~10-12 params)
  took it to the final 10, at a small real AUC cost, clean correlation throughout (max ρ=0.73).
  `account_ordinal` (creation-order age proxy) is excluded: it doesn't even help nominally (0.774 with
  vs 0.780 without). Full story: `docs/v3-research/whitepaper.md` §5–6b.
- **Checked and ruled out (2026-08-22): bringing back "spread/variance" features**
  (`score_stddev`, `reception_spread`, `karma_extremeness`, `karma_per_post_extremeness`,
  `reception_spread_pctl`, `net_karma_spread_by_sub`, `churn_ratio` — the Phase 2 equation's original
  intuition) does not help on real ground truth (n=684): each is weak alone (bivariate AUC 0.51–0.61),
  and adding all of them to the 10-feature model makes it *worse* (0.780→0.767 with 5, →0.771 with 7),
  not better — they dilute tree splits without adding real information the kept features don't already
  capture. Don't re-add these without new evidence; this was checked rigorously, not assumed.
- **XGBoost, tuned hyperparameters, repeated 5×10-fold CV AUC = 0.780 ± 0.035** — real, validated, and
  stable across the last several sampling rounds; read scores as directional. Independent confirmation:
  manually-verified high/mid/low tier bad-rates of 85%/25%/2.5% on the largest check (n=120). Full
  analysis (target/feature comparison, feature-count elimination curve, importances, ROC,
  confirmed-rate-by-bucket): `docs/v3-research/charts/model_analysis.html`.
- **Subreddit prevalence** measures influence over each subreddit's **top-30-posts-by-karma** monthly
  (poster + top/latest commenters, deduplicated, scored, % landing ≥0.7) — not share of total monthly
  activity. Full 24-month history (2024-08→2026-07; 2026-08 not yet collected). Trend now shows a
  genuine dip-and-recover shape (~15%→~10%→~14%), not the pre-fix artifact decay toward zero.
  `output/v3/subreddit_bot_prevalence_mom.csv`.
- **Dashboard:** `docs/bot-spam-compass.html` — the project's primary bot/spam artifact, data embedded
  at build time (works from file://, a local server, or GitHub Pages alike, no external fetch).
  **Monthly refresh is one command:** `python3 scripts/v3_stage8_monthly_refresh.py` — retrains,
  rescores, rebuilds prevalence, regenerates the dashboard in place.
- **Not yet done:** this model doesn't establish *coordination* between accounts (only individual
  removal risk) — the coordination angle from the earlier equation-based theory remains open, see the
  coordination-check note below. The labeled set (n=684) could still grow further, but has stopped
  swinging with each new round — this is no longer the most urgent gap.

### Earlier thread: an equation-based reframing, superseded — frozen 2026-08-21

**Full writeup: `docs/v3-research/whitepaper.md` §2.** This supersedes the "explicit next step"
below (scaling 76→~300 bots was never resumed — the project pivoted instead). Short version:

- **Reframing:** Phase 1's 76 bots are all templated/duplicate-text spam — a real but narrow species.
  The analyst's theory: a second, larger population of **paid human spam workers** (political IT
  cells, PR firms) exists, writing original text but sharing a behavioral signature — high activity,
  karma concentrated in a "home" sub while getting downvoted (but still posting, because paid
  regardless) in "opposition" subs. Different detection problem, built and validated independently.
- **Equation:** (high comment or post karma) AND (high karma-reception variance across subreddits) AND
  (high comments or posts per day) — calibrated against 7 named seed accounts (incl. `CritFin`) and
  validated by comparing top-30-per-subreddit accounts across 3 subreddit groups: political/meme
  "spam-dens" (IndiaSpeaks, unitedstatesofindia, indiameme, CricketShitpost, InstaCelebsGossip) vs.
  finance vs. lifestyle comparison groups, with subreddit-level (not account-level) leave-one-out
  z-score outlier removal. **Result: spam-dens run ~1.8–2.2x finance/lifestyle on `karma_extremeness`
  and `reception_spread`, and are the only group with non-zero post-based spread** — replicates,
  not tautological (accounts selected by frequency, not karma/spread), not outlier-driven.
- **Final score:** `(comment_spread band + post_spread band) × (comments/day band + posts/day band)`,
  0–16, **multiplicative — not additive.** A first pass summed all 4 bands (0–8 range); the analyst
  caught this directly ("is this happening? y or N") and it was corrected — the equation was always an
  AND ("high variance" AND "high activity"), and summing let one strong factor compensate for a weak
  other, which a real AND-gate must not allow. The fix zeroed 21,152 of 34,565 floor-qualifying accounts
  (61%) that had scored above 0 under the wrong formula. Thresholds unchanged (comment_spread 20/140,
  post_spread 50/500, comments/day .05/.2, posts/day .01/.1), scored only on accounts with ≥10 sampled
  comments or posts (34,565 of 347,886). Top score-bucket (score 16, n=29) holds 3.2% of karma alone at
  37.76x its account share; top three bands combined (9/12/16, n=236, 0.68% of population) hold 13.6%
  of karma and 18.1% of posts.
- **Critical honest finding: this score does NOT validate as a classifier against either labeled
  set it was checked against** — AUC 0.470 vs. a 20-account LLM-confirmed spam set, AUC 0.319 (worse
  than random) vs. Phase 1's 76 duplicate-text bots — both re-checked against the corrected
  multiplicative formula, same conclusion as the earlier (wrong) additive version. **Conclusion: Phase
  1 and Phase 2 catch different populations** — Phase 1 bots are low-variance/low-effort by construction
  (§6); Phase 2 targets the opposite (high-variance, high-activity). Phase 2's only real evidence is the
  §12 group separation, not a classification metric — treat every score as directional, not validated.
  See whitepaper §18.
- **Subreddit-month rollup (2026-07, 44/45 subs — DesiVideoMemes' collection stopped 2026-03):**
  `output/v3/subreddit_score_2026-07_mult.csv`. Top by % of month's activity from score≥6 accounts:
  unitedstatesofindia (17.2%), BollyBlindsNGossip (13.2%), ISRO (13.1%), IndiaCricket (12.3%).
  Notably **IndiaSpeaks ranks 10th (7.7%)** despite being the strongest §12 account-level signal and
  the source of most seed accounts — the two views measure different things (individual extremity
  vs. share of a month's aggregate activity) and both are real; see whitepaper §16 for why.
- **Not yet done:** no purpose-built "paid spam worker" label set exists to actually validate the
  equation against (the 20-account set was a negative check, not a fitting target). If this thread
  continues, that's the next real step — not scaling Phase 1's bot count further.
- **Two follow-up outputs added 2026-08-21, both documented in whitepaper §19-20:**
  1. **Top-30-post influence rating** (`output/v3/subreddit_top30_rating_2026-07.csv`,
     https://claude.ai/code/artifact/1223e889-72e8-4bcf-ab95-be6ed2306a87): per sub, top 30 posts by
     karma → poster + top-5-by-score commenters + top-5-latest commenters, deduplicated, scored. Ranks
     subs differently from §16 — delhi/unitedstatesofindia/CricketShitpost lead; IndiaSpeaks drops to
     24th. Answers "who's behind the best-performing content," not "share of total activity."
  2. **Full-2026 MoM trend, selectable per subreddit**
     (`output/v3/subreddit_score_2026_mom.csv`, https://claude.ai/code/artifact/d4f5b1f6-d9b0-4658-93fa-b8913cb82313).
- **Gotcha hit while building these:** `sub` is a real pandas `DataFrame` method (row-wise
  subtraction) — `df.sub` (attribute access) silently returns the *method*, not a column literally
  named `sub`, causing a confusing `'function' object has no attribute 'nunique'`-style error with no
  hint that the column name is the problem. Always use `df['sub']` bracket notation for a column named
  `sub` (same risk applies to other pandas-method-shadowing names: `count`, `sum`, `min`, `max`, `mean`,
  `add`, `div`, `mul`, `pow`, `mode`, `size`, etc.).

### Earlier thread: hand-labeled bot-detection methodology (Stage 5-7) — milestone frozen 2026-08-21

**Full writeup, all stats/citations/limitations: `docs/v3-research/whitepaper.md`.** Short version:

- **Ground truth built with zero new API calls.** `commenters_dedup` already holds full comment
  `body` text for all 1.6M collected comments (this was missed for a while — don't re-assume "we need
  to fetch more" before checking that table first). A SQL screen for cross-post duplicate text
  (`scripts/v3_stage5_bot_candidates.py`) plus parallel LLM batch review produced **76 confirmed bots,
  58 suspicious, 516 clean** (`output/v3/{confirmed_bots,suspicious_accounts,clean_accounts}.json`) —
  16.1% hit rate on screened candidates vs. 5.6% on unscreened accounts, so the screen is doing real
  work. Concrete example: `CritFin` → `Critifin` → `criti_fin`, a ban-evasion sockpuppet chain,
  independently reconfirmed by two separate review batches.
- **Method 1 (hand-built composite via bivariate pruning) failed cleanly: AUC 0.474 vs. the labeled
  set — worse than random**, despite every included metric passing an independence check
  (`scripts/v3_stage5_method1_composite.py`). Averaging in ~15 individually-weak metrics dilutes the
  1-2 that actually carry signal.
- **Method 2 (XGBoost) is the one that works: 5-fold CV AUC 0.792**, right at the >0.80 target, not
  cleanly over it — n=76 is small, fold AUCs range 0.72-0.87
  (`scripts/v3_stage6_method2_xgboost.py`). Top features: `median_comment_score`, `score_stddev`,
  `subreddit_entropy`, `karma_per_day_since_first_seen` — notably, `comments_per_day`/`posts_per_day`/
  `removal_rate` (this session's early hypotheses) rank near the *bottom* of 25 features.
- **A simple 4-leaf decision tree is the interpretable version** (0.733 CV AUC): low `score_stddev`
  alone predicts bot; among high-`score_stddev` accounts, only unusually long comments do. This
  *inverts* the "high-variance power users are bots" hypothesis that motivated adding those features —
  most confirmed bots are low-effort templated spam with small, consistent scores, not wild swings.
- **Dormancy-reactivation (a tip from external Reddit mod discussions) tested and found no signal**
  in this corpus (mean percentile 27.4 vs. ~25 baseline) — real negative result, don't rebuild it.
- **Outputs shipped this milestone:** `docs/v3-research/whitepaper.md` (methodology + all stats),
  `docs/v3-research/bot-score-dashboard.html` + `bot-score-mom.json` (MoM dashboard, driven by
  `scripts/v3_stage7_monthly_score.py` — re-run that one script monthly, it retrains Method 2 on
  whatever's in the label-set files and rescoring the full population).
- **Explicit next step, not yet done:** scale the labeled set 76 → ~300 confirmed bots (same
  duplicate-text screen, larger/relaxed candidate pool) and re-check whether Method 2's AUC holds.
  **Important caveat already on record, don't skip it:** an *earlier, separate* verification round in
  this project manually read real subreddit-ranked samples and found the ranking mostly tracked
  meme-culture-vs-discussion-culture, not bot density — that finding predates Method 1/2 above and
  hasn't been re-tested against the new model. Don't present a subreddit ranking from this methodology
  as validated until it is.

### Where things stand, honestly

- **Stage 0–2 (collection, cleaning, account features, EDA, bivariate/segmentation):
  done and stable.** Rebuild scripts in `scripts/v3_stage0_build.py` →
  `v3_account_features.py` → `v3_feature_sanitise.py` → `v3_eda_build.py` →
  `v3_stage2_bivariate.py`. `account_features_model` (347,886 × 73 cols, VIF-pruned,
  volume-normalized, hurdle-split) is the correct feature source for anything
  account-level — never hand-select from raw `account_features`.
- **Stage 3 (account XGBoost per label channel): mature, near its honest
  ceiling, and this is now a settled, trustworthy result.** Five channels
  (`admin_removal`, `self_deletion`, `comment_removed_ambiguous`,
  `automod_filtered`, `moderator_removed`), all landing 0.64–0.80 gated
  rung-4 AUC — inside Kumar et al. 2017's 0.65–0.80 account-level ceiling,
  as the plan predicted from the start. One real feature-engineering win
  (Stage 3b, joining unused post-level data, +0.025 to +0.133) followed by
  three progressively smaller-to-null laps (Tier 1/2/3), each independently
  confirmed genuine — not noise — via a reproduce-twice-and-diff discipline
  that caught and fixed **three separate real bugs** along the way (a
  feature-set bug that had been silently depressing "base" AUC numbers, and
  two instances of a `post_edit_rate`/removal-appeal leak). **Do not launch a
  Tier 4** — four laps in, diminishing-to-zero returns are a measured
  finding, not a guess. Consolidated writeup, all four laps, as a Claude
  Artifact: https://claude.ai/code/artifact/3076ef53-7ff4-4cbc-8a3d-e9acd7e78b9a
  (published before Tier 3 landed — Tier 3's null result isn't on that page).
  Scripts: `v3_stage3_account_model.py` → `_3b_feature_iteration.py` →
  `_3c_tier1_features.py` → `_3d_tier2_features.py` → `_3e_tier3_features.py`.
- **Stage 4 (pair-level model): inconclusive, not tested rigorously enough
  to count as evidence either way.** Built with a View-A/View-B
  anti-circularity split (behavioral features predict a label built only
  from content/identity signals). Primary result (0.616) sits *below* its
  own mandatory baseline (0.653), but the label had only 131 positives (6 in
  the test set after the CV split correctly dropped 117,200 straddling
  pairs) and a manual spot-check found real label-quality problems
  (topically-unrelated accounts scoring as "stylometrically similar").
  **The plan's 0.90+ pair-level claim is still untested, not falsified.**
  If revisited: needs a stronger identity signal than percentile thresholds
  on char-n-gram cosine (which compresses near-ceiling on Reddit-comment-length
  text), and/or a lower co-appearance floor to recover the 8.9M
  single-co-appearance pairs currently excluded entirely.
- **Cross-sample boundary discovery (unsupervised, no removal/deletion
  labels used to construct anything): the most open, most promising, and
  least resolved thread.** User-directed alternative to label-based
  Stage 3, motivated by Stage 3's own finding that removal-based "labels"
  don't agree with each other. Two rounds:
  1. **Univariate** (`scripts/v3_boundary_discovery.py`): 9/64 candidate
     metrics replicate as genuine bimodal splits across two independent
     samples. An incremental AND-rule (not a score — an account must fail
     *every* standout indicator) stabilizes at 5 conditions, flagging a
     replicating ~0.5% of accounts, validated on a third untouched sample.
     But this group's `removal_rate` is *below* population baseline
     (0.84×) — not obviously bot-like by that external check.
  2. **Multivariate** (`scripts/v3_multivariate_kde.py`): PCA (diffuse
     space, 12 components only reach 52% variance) + density-mode
     clustering in the reduced space. Mean-shift and HDBSCAN disagree on
     cluster count (structure isn't settled). But small clusters, found
     independently in each half of the split, show **8–12× removal-rate
     enrichment**, and one specific cluster (cluster 15, n=1,116 in Part1)
     is **18× concentrated** with the univariate round's flagged group —
     two independent methods partially agreeing.

  **RESOLVED 2026-08-20, documented dead end — manual read confirms the
  group is not bot-like.** Pulled real, live comment history (25 most
  recent comments each, via the Arctic Shift API — same source
  `scripts/v3_collect.py` uses) for a seeded random sample of 36 accounts:
  20 from the AND-rule ∩ cluster-15 overlap (the strongest-signal group,
  96 accounts total), 8 AND-rule-only, 8 cluster-15-only. Zero of 900
  comments read as bot-like on any axis checked: no cross-account
  templating (checked every body for exact duplicates across all 36
  accounts — none), no spam/promo/scam links or phrasing (regex swept for
  discount/coupon/DM-me/telegram/bit.ly-style phrasing — the 3 hits were
  all incidental human usage, e.g. "bought virtus with great discounts"),
  no self-templating beyond one account copy-pasting the same 3-sentence
  political opinion into 3 different comments on one thread (a known
  manual troll pattern, not evidence of automation). What the accounts
  actually look like, read directly: young/casual India-focused Redditors
  — heavy giphy-gif reactions, one-word replies, Hinglish slang, genuine
  community-specific banter (exam-prep subs, cricket, Bollywood gossip,
  gaming, fashion) — i.e. the AND-rule + PCA cluster is most likely
  picking up a **low-effort/reactive commenting style**, not coordinated
  or automated behavior. This lines up with the univariate round's own
  external check above (removal_rate 0.84× population, not enriched).
  Reproduction: account-level export added to `v3_multivariate_kde.py`
  (writes `output/v3/flagged_accounts_part1.csv`, all 115,950 Part1
  accounts with `cluster` + `and_rule_flagged` columns — regenerating it
  reproduces the exact same 550/1,116/96 counts, confirmed bit-identical
  against the already-committed `docs/v3-research/eda/*.json` on rerun);
  fetch script and raw pulled comments: `output/v3/flagged_account_samples.json`
  (not committed — real usernames + comment text, kept local).
  **Conclusion: this thread is closed as a negative result, not reopened
  without a new signal — don't re-run this verification on the same
  AND-rule/cluster-15 output expecting a different answer.**

### Next actions, ranked

1. ~~**Open the accounts.**~~ **Done 2026-08-20 — documented dead end, see
   the RESOLVED note above.** 36-account manual read found no bot-like
   behavior; don't redo this on the same output.
2. **Verify the multivariate clusters are the same phenomenon, not
   coincidence.** Part1's and Part2's independently-found enriched clusters
   both run hot on removal_rate — check whether they share the same
   original-feature profile (not just both being removal-heavy by chance).
   Flagged by the fork that built it as the natural next statistical check.
3. ~~**Render the still-orphaned local pages.**~~ **Done 2026-08-20.**
   `stage4.html` already existed (this note was stale); added
   `stage3e.html`, `boundary_discovery.html`, and `multivariate_kde.html`
   in the same house style, linked from `index.html` and every other
   page's nav. (`stage3b`–`3d` remain Artifact-only by design, not local
   pages.)
4. **If revisiting Stage 4**, don't restart from scratch — the null-model
   and co-appearance infrastructure is solid (four real bugs already found
   and fixed there too); the weak link was specifically label
   construction/power, not the modeling pipeline.

### Standing rules, already paid for, don't rediscover

- **Leakage:** `removal_rate`, `deleted_later_rate`, `post_edit_rate` (and
  every `_nonzero`/`_pctl`/`_max` variant) are either hard-excluded from
  model inputs (Stage 3+) or excluded from construction entirely (boundary
  discovery) — reuse the exclusion list in `v3_stage3_account_model.py`
  rather than re-deriving it. `pc_removed_comment_rate`/`pc_tombstone_rate`/
  `pc_bot_comment_rate` are thread-context signals, not the account's own
  record — kept, flagged as a judgment call.
- **Volume-gating, not leave-one-out.** Leave-one-out feature exclusion was
  tried once (Stage 3) and made things worse (turns thin-history accounts'
  features to `NaN`, which XGBoost's missing-value handling then reads as a
  near-perfect label proxy). `n_comments_sample ≥ 10` (and the analogous
  `n_distinct_posts_ctx` gate for post-context features) is the adopted,
  correct fix.
- **Reproduce-twice-and-diff before trusting any new AUC delta.** The true
  noise floor between "identical" reruns is ~0.001–0.003 once the
  feature-set bug above was fixed — verified, not assumed. Any delta near
  that size needs this check before being reported as real.
- **B1's pair-level null-model p-value is a continuous ranking feature
  only** — no binary-significance gate, and none should be added (candidate
  pairs are pre-filtered to already-co-occurring accounts, so p-values skew
  small under the null too, independent of real coordination).
- **`karma_extremeness`/`karma_per_post_extremeness` doc/code mismatch,
  still unfixed:** documented as reporting-only in `v3_feature_sanitise.py`'s
  docstring but actually still in the model-ready column set.
- **Three features still return a residual DEGENERATE Stage-1 verdict**
  (`n_posts_sample`, `n_high_tier`, `n_threads_with_repeat`) — the
  point-mass strip cap doesn't fully catch them, flagged not fixed.
- **Base36 age calibration:** on hold by choice (not abandoned), using the
  free `days_since_first_seen` proxy instead. Script
  (`v3_calibrate_age_sample.py`) works if ever wanted.

---

## 0. What changed, and why V2 is closed

V2 plateaued at ROC-AUC **0.663**. Two experiments run 2026-08-04 established why.

**Experiment B — the label is not the problem.** ✅ Holding the 10 account
features constant and swapping the label moved nothing:

| Label | n | pos | AUC |
|---|---|---|---|
| `gone` (baseline) | 7,872 | 334 | 0.663 |
| suspended only | 7,872 | 76 | 0.662 |
| not_found only | 7,872 | 258 | 0.672 |
| admin-removed ≥1 | 1,956 | 34 | 0.593 |
| admin OR suspended | 1,956 | 55 | 0.623 |
| admin_rate ≥ 50% | 1,956 | 28 | 0.502 |

**Follow-up probes.** ✅ Gradient boosting on the same features scored *worse*
than logistic (0.602 vs 0.663), so logistic was not underfitting — there is no
structure left to extract. Elkan–Noto prior estimation was degenerate
(c = 0.054), which is itself evidence the features barely separate the classes.

**The label channels measure different constructs.** ✅ Of 34 admin-removed post
authors, **zero** were suspended and 31 are still alive. This is not a censoring
artefact: 0 of 35 posts by suspended authors show `[deleted]` re-attribution.

**Note on the PU ceiling argument.** `AUC_pu^max = (1 + β − α)/2` is correct
algebra, but observing 0.663 does *not* prove you sit at that ceiling — it
cannot distinguish "at ceiling with bad labels" from "below ceiling with weak
features". Applied to experiment B it favours the feature reading: purifying to
suspension-only raises the ceiling while observed AUC stayed flat.

**Conclusion:** the 10 account-on-paper features are exhausted. V2 is closed.

### The structural pivot

📚 Kumar et al. (WWW 2017), nine Disqus communities — the closest published
analogue to Reddit — **same data, same features**:

| Question | AUC |
|---|---|
| "Is this account a sockpuppet?" | **0.68** |
| "Are these two accounts the same operator?" | **0.91** |

V2's 0.663 sits on the documented ceiling of the *per-account framing*. The lever
is not better features; it is **changing the unit of analysis**. Corroborated by
TwiBot-22 (1M users, realistic 14% bot rate, best-of-35 detectors F1 **58.7**;
Botometer accuracy **49.9** there despite F1 96–99 on older benchmarks), and by
no published Reddit account-level detector exceeding ~F1 0.70 on a non-leaking
label.

---

## 1. Objective

**Exposure-weighted, not activity-weighted.** The deliverable is not "% of all
activity that is inauthentic". It is:

> For each (subreddit, month): **of what a normal visitor actually sees, how much
> is manufactured** — and is it individual automation, coordination, or a rally —
> with every point of score traceable to named accounts and threads a human can
> open and check.

This reframing matters. Sampling top-100 posts is *biased* as a sample of all
activity, but it is the **correct frame** for exposure-weighted corruption:
top-ranked posts are the population of interest, not a convenience sample. It is
also far cheaper, which is what makes the budget work.

### Three questions, three units, three honest ceilings

| | Question | Unit | Output | Realistic ceiling 📚 |
|---|---|---|---|---|
| **A** | How much of what's seen is inauthentic? | sub-month | prevalence ± CI | quantification, **no AUC** |
| **B** | Is there coordination? | **account-pair** | cluster list + z | **AUC 0.90–0.94** |
| **C** | Was there a rally? | event | burst shape + residual | **no AUC** — no ground truth |

Ceilings by target, for setting expectations honestly 📚:

| Target | Unit | Realistic |
|---|---|---|
| Self-declared / obvious automation | account | F1 0.95–0.99 |
| **Coordination / shared operator** | **pair** | **AUC 0.90–0.94** |
| Predicting platform enforcement | account | 0.75–0.85 (0.65–0.75 without mod-action features) |
| Modern per-account automation | account | 0.65–0.80 |

**The 90%+ target is achievable at the pair level and nowhere else.** Any
per-account "bot score" we publish is capped near 0.70 by the literature, and we
will say so on the methodology page.

---

## 2. Sampling design and budget

### The frame

- **45 subreddits** (`subreddits_v3.csv`), 14 categories, tagged by
  `incentive_tier` ∈ {high, medium, low}.
- **24 months.**
- **Top 100 posts per sub-month** by score — the exposure frame.
- **Counter-sample:** 20 random posts per sub-month, **matched on
  `num_comments`** to the top-100 distribution. Unmatched random sampling makes
  every derived threshold a volume threshold in disguise.
- **Per top post:** the **first 10** and **top 10 by score** commenters.

### Budget

| Bucket | Rows | Source |
|---|---|---|
| Sub-month series | ~1,100 | `/time_series` |
| Posts (100 top + 20 matched random × 45 × 24) | ~130,000 | `/posts/search` full scan, keep top-N |
| Comments (first-10 + top-10, **both top AND counter-sample** × 130,000 posts) | ~1.9–2.3M | `/comments/search` per post |
| Accounts (unique posters + commenters) | ~300–500K | `/users/ids` |

**Correction from the pilot run (2026-08-05):** the original budget line only
counted comment extraction against the 108,000 top posts. The validated pilot
extracted first-10/top-10 commenters from **both** top and counter-sample posts
(130,000 total) — required for Stage 2b (matched top-vs-counter comparison,
where the one real non-tautological lead so far, removed-comment-rate, came
from). Row estimate rebased on the pilot's **measured yield of ~15 commenter
rows/post** (15,806 rows / 1,051 posts), not the assumed 20/post ceiling —
short threads and dedup both pull the real average down.

**Storage:** written as **zstd-compressed NDJSON from the start**. Raw JSON at
this row count is ~2.6 GB and would breach the 3 GB cap; zstd (measured 8–10×)
lands at ~400–600 MB. Parquet + DuckDB for the analysis layer.

### Fetched ≠ stored

The API sorts only by `created_utc`, so top-100-by-score requires scanning every
post in the month. Likewise "top 10 commenters by score" requires the whole
thread. The rule is **compute-then-discard**: fetch the thread, compute the
derived per-post aggregates below, store one post row + 20 comment rows, discard
the rest.

Per-post derived aggregates (stored on the post row, so thread shape survives
without storing the thread):
`n_comments_observed`, `n_unique_commenters`, `first10_arrival_gaps`,
`comment_score_p50/p90/max`, `comment_score_gini`, `max_depth`, `mean_depth`,
`pct_toplevel`, `reply_reciprocity`, `submitter_reply_rate`,
`removed_comment_rate`, `tombstone_rate`, `bot_comment_rate`.

**Estimated wall clock — corrected against pilot measurement (2026-08-05).**
The original "2–3 hours" figure was an unvalidated guess and is **wrong by
roughly an order of magnitude.** The pilot measured 717s for 1,051 posts
(both top and counter-sample, full comment-thread scan, single-threaded,
sequential) = **0.68s/post**. At that rate, 130,000 posts is:

| Mode | Wall clock | Basis |
|---|---|---|
| Single-threaded (measured) | **~24.6h** | Direct pilot measurement — solid ground truth |
| 2 workers | ~12.3h | Proportional extrapolation, **not validated at sustained scale** |
| 4 workers | ~6.2h | Same caveat |
| 8 workers | ~3.1h | Same caveat — also the ceiling the original audit called "verified-safe," but only from a 25-request burst test, not a multi-hour sustained run |

**Plan on ≥6 hours, checkpointed and resumable, running unattended (background
job, not a foreground wait).** Confirm sustainable parallel throughput on the
first ~30 minutes of the real run before committing to a worker count — the
25-request burst test behind the "8 req/s, zero 429s" figure is not the same
claim as "8 req/s sustained for 6+ hours."

---

## 3. Data source reality (audit 2026-08-04) ✅

Everything here was measured live, not read from documentation.

### The `_meta` block — the most valuable find

Undocumented, present on **100%** of content from ~2023-07 onward:

| key | meaning |
|---|---|
| `removal_type` | `moderator`, `deleted`, `automod_filtered`, `reddit`, `content_takedown` |
| `was_deleted_later` | alive at first capture, gone by T+36h ⇒ **original text retained** |
| `was_initially_deleted` | already gone at capture ⇒ tombstone |
| `is_edited` | the only working edit flag (top-level `edited` is ~0%) |
| `retrieved_2nd_on` | deterministic **T+36.0h** re-check |

Records are a merge of two snapshots: **text from +16s, score/removal status from
+36h**. So an item removed between those moments keeps its text *and* carries a
label.

### Removal labelling — three traps

1. **`removed_by_category` is NEVER set on comments** (0/14,627). Comment removal
   detection must use `_meta.removal_type`. On posts it undercounts by 11%; the
   correct label is the **union** of both.
2. **Removed comments retain original text 100%** of the time (661/661). Removed
   **posts** only **13%** — mods act inside the 16-second capture window.
3. A third class, **14.7% of comments**, are unlabelled tombstones identified by
   `author == "[deleted]" AND body == "[removed]" AND collapsed_reason_code == "DELETED"`.

**Rule:** `was_deleted_later == True` ⇒ text is real (382/382 posts, 55/55
comments). Otherwise assume tombstone.

### Hard era boundary at ~2023-07

No `_meta`, no removal labels, no edit flags before it. **Rich modelling starts
2023-07.** With today at 2026-08 that permits ~36 months; our 24-month window
fits comfortably.

### Pagination bug — costs 0.16% of rows silently

`after` is exclusive at **second** granularity, so same-second siblings are
skipped. Verified: second `1752539352` holds two comments; `after=1752539352`
returns neither. **Fix:** cursor on `after = last_created_utc*1000 − 1` (ms) and
dedupe by `id`. Validated exact against `/time_series` ground truth (r/ipl
2025-05-25 → 5,887 rows, row-for-row). Costs ~0.6% duplicate refetch.

### Account age — viable via base36 ✅

There is **no account creation date anywhere in the API**. But decoding
`author_fullname[3:]` from base36 works: lower envelope monotonic in **13/13**
bands, Spearman 0.82, **AUC 0.986** separating pre-2023 from post-2025 accounts.
Calibrate against the lower envelope, not the median.

**Limitation:** `author_fullname` is present on 100% of non-deleted-author content
and **0%** of deleted-author content — no cohort signal for exactly the
population of most interest.

### Dead on arrival — build no features on these ❌

`collapsed_because_crowd_control` (0% — kills the old plan's C8/U25) ·
all award fields (`gilded`, `total_awards_received`, `all_awardings` — 0 nonzero
in 21,691 objects) · `downs` (always 0) · `ups` (byte-identical to `score`) ·
`is_created_from_ads_ui` (always False) · `removal_reason`, `mod_reason_title`,
`banned_by`, `num_reports` (all null) · `active_user_count`, `accounts_active` ·
`edited` top-level (~0%) · comment `depth` (not returned by search — compute from
`parent_id`).

### Staleness ⚠️

- Subreddit objects are **536 days stale** (r/india reports 2.48M subscribers vs
  3.47M actual). **Never** use `subreddits/search.subscribers`. Use
  `/time_series/r/<sub>/subscribers` (monthly to 2018-03, daily on demand).
- User karma aggregates are **344–500 days stale** — usable as coarse priors
  only, never as current-month features.
- `score` is uniformly aged at **T+36h** — consistent across rows, good for
  modelling, but understates late-blooming threads.

### Also unavailable ❌

Score trajectories (two snapshots, one score) · moderator lists / modlogs ·
suspension status · `/posts/search/aggregate` and `/comments/search/aggregate`
(time out on essentially everything — use `/time_series`) ·
`/users/interactions/*` (**hard-blocked on high-volume accounts** — i.e. exactly
the bots of interest).

**Live Reddit is optional, not a dependency.** Arctic Shift covers account age
via base36 (§3, AUC 0.986) and five of the six label channels. The only genuine
gap is suspension status — and at ~100 calls/min, checking 300–500K accounts
would take **80+ hours**, so it was never viable in bulk anyway. V2 also measured
suspension as a weak label (AUC 0.662 vs 0.663 for `gone`).

Two narrow, optional uses remain, both post-hoc:
1. **Calibrate the base36 age proxy** against a few thousand known creation dates
   — turns the lower-envelope estimate into a fitted curve. ~30 min.
2. **t+90d suspension check on the top-risk decile only** (a few thousand
   accounts). This is the only **account-level, externally-generated** signal in
   the design — every Arctic Shift label is content-level — so it is the one
   non-circular validation available.

Neither blocks collection. `scripts/reddit_auth.py` already holds the
credentials; the audit's 403 was a datacenter-IP artefact, not a credential
problem.

---

## 4. Metric catalogue

Every sub-level metric is expressed as a **percentile against the low-incentive
control tier**, never as an absolute.

### 4.1 Sub-month level — "what changed"

From `/time_series` (nearly free) plus per-post `subreddit_subscribers`.

| # | Metric |
|---|---|
| S1 | posts/month, comments/month, MoM delta and 3-month trend |
| S2 | subscriber curve, MoM growth, growth-rate anomalies |
| S3 | comments per post, MoM |
| S4 | sum_score/month, score per post |
| S5 | removed-post rate, removed-comment rate, tombstone rate |
| S6 | moderator intensity (`distinguished`, `stickied`, `locked` rates) — **a confounder, not a signal** |
| S7 | account churn — new authors appearing, prior authors disappearing |
| S8 | Herfindahl over outbound domains |
| S9 | posting-hour profile vs IST expectation |
| S10 | sub age at observation month (creation date is immutable, so valid even though the subreddit object is otherwise 536 days stale — see §3) |
| S11 | rules count and recency of last rule change — `/subreddits/rules`, confirmed working (r/india → 14 rules w/ `created_utc`); a sudden new rule is a mod response to an incident |
| S12 | flair-scheme richness — distinct `link_flair_text` values/month, % of posts carrying flair |
| S13 | cross-post ratio, inbound (% of this sub's posts that are themselves crossposts) and outbound/fan-out (`num_crossposts` sum / post count — how often this sub's content propagates elsewhere) |
| S14 | NSFW / quarantine flag and transitions (`over_18`, `quarantine`) |
| S15 | **sub-month series regime test** 🔬 — apply the Stage-1 GMM/BIC multimodality test (§7) directly to a sub's own monthly post/comment-count series, not just to account features. A sub whose monthly activity clusters into two regimes (baseline vs. spike months) is showing structure account-level tests can't see. This is a genuinely new test, not previously distinct from S1's MoM delta. |

**S6 is load-bearing.** Heavily-moderated subs look cleaner for reasons unrelated
to bots. Any cross-sub comparison ignoring mod intensity is partly ranking mod
staffing. V2 ignored it entirely. **S6 measures moderation *intensity*, not
moderator *count* — no endpoint returns a mod list or headcount (§3), and there
is no viable proxy for the count itself. Keep the two constructs distinct; don't
let S6 stand in for "how many mods does this sub have."**

### 4.2 Post level — "was this pumped?"

| # | Metric | Note |
|---|---|---|
| P1 | `score`, `upvote_ratio`, `num_comments` | raw |
| P2 | **implied vote volume** = `S/(2r−1)` | ✅ verified; **only for r ≥ 0.65** |
| P3 | `contested_share` = `1 − r` | exact, no algebra needed |
| P4 | **`votes / num_comments`** — voted-but-not-discussed | the natural vote-manipulation signature |
| P5 | comment:score ratio vs sub baseline | |
| P6 | score per subscriber | cross-sub comparable |
| P7 | time-to-first-comment | |
| P8 | **first-10 arrival gaps** — tight cluster vs power-law decay | 🔬 |
| P9 | **thread width:depth** — coordinated threads are wide and shallow | 🔬 |
| P10 | reply reciprocity within thread | |
| P11 | outsider share — commenters with no prior history in this sub | 📚 |
| P12 | removed-comment rate within thread | |
| P13 | title-template reuse across posts | 📚 |
| P14 | domain–account concentration | |
| P15 | `link_flair_text` as free topic label | ✅ 88% populated |
| P16 | title length, body length, title:body ratio | ✅ trivial from raw fields — short title / no body is a link/reaction-bait signature |
| P17 | mean thread sentiment/toxicity 🔧 | proxy, not a raw field — see A39. Same Hinglish caveat applies at post level. |
| P18 | edit status and timing | 🔧 use `_meta.is_edited`, **not** top-level `edited` (~0% populated, dead). Timing: edited before the T+16s capture vs. between the two snapshots, via `_meta.was_deleted_later`-style logic |
| P19 | link vs. self post | ✅ `is_self`, raw field, was never promoted to a named metric before |
| P20 | crosspost count and fan-out (which subs) | ✅/🧩 `num_crossposts` is raw; fan-out via same-URL search across subs — was in an earlier draft, reinstated here |
| P21 | posting time vs. the sub's own modal posting hour | 🧩 the "vs. mode" framing generalised from account level (A34–A38) to post level |
| P22 | engagement velocity | 🔧 **proxy only** — score / hours-since-post at the fixed **T+36h** capture (§3). No true trajectory exists (two snapshots, one score); never describe this as real-time velocity. |
| P23 | score per word | 🧩 content-length-normalised engagement efficiency |

**On P2** ✅: the derivation `downs = S(1−r)/(2r−1)` is exact (verified,
`ups − downs == score` on 12,028 posts), but the denominator → 0 as r → 0.5 and
Reddit rounds `upvote_ratio` to 2 decimals. At r=0.55 a 0.01 error moves implied
downvotes by 28%. **The contested-ness you want is just `1−r`, which is free and
exact. What the algebra actually recovers is the vote *denominator*** — use it to
normalise reach, not to measure controversy.

### 4.3 Account level

Note the ceiling: these feed the pair layer, they are **not** the deliverable.

**Provenance.** A1 account age via base36 ✅ · A2 dormancy gap (first activity −
creation; purchased-account signature) · A3 username morphology
(`Adjective_Noun_1234` rate, n-gram entropy) · A4 `author_premium` ·
A5 karma/day, post-karma:comment-karma ratio · A6 flair possession.

⚠️ **A31 raw posting frequency (posts+comments per account-age-day)** — 🧩,
computed from the same `/comments/search?author=` pull already planned for the
timing family below. **V2 had this as `comments_per_day`; it was dropped going
into V3 without a stated reason.** That's a regression, not a considered cut —
reinstated here. A32 comment:post **count** ratio (`n_comments/n_posts`) — 🧩,
distinct from A5's karma-*type* ratio, which the account-level table previously
conflated this with; keep both, they answer different questions.

**Timing** 📚 — the family V2 entirely lacks, and where the discriminative power is.

| # | Metric | Note |
|---|---|---|
| A7 | inter-comment interval entropy | 📚 accuracy 0.848 standalone |
| A8 | **interval quantisation** — gaps within ±2s of 60/300/900s multiples | cron signature, very low organic base rate |
| A9 | **burstiness** | ⚠️ Goh–Barabási `B=(σ−μ)/(σ+μ)` is **length-biased** — use Kim & Jo's finite-size-corrected `A_n`, or you manufacture a volume↔botness correlation |
| A10 | circadian dead hours (24 UTC bins with zero activity) | humans sleep |
| A11 | circadian centroid offset vs IST | |
| A12 | weekday:weekend ratio | |
| A13 | session structure (split at >30min gaps) | 📚 session-position dynamics lift AUC 0.83→0.97 |
| A14 | time-to-arrival on new threads; sustained <120s ⇒ feed monitoring | |
| A15 | activity changepoints, **burst post trends** | |
| A33 | **response latency** — `comment.created_utc − parent.created_utc`, distributional (median/mean) per account | ✅ was in an earlier draft, dropped before the final plan — reinstated. Humans have a floor; scripts don't. |

**Content.** A16 self-similarity of own comments · A17 cross-account
near-duplicates (MinHash/LSH) · A18 type-token ratio · A19 script/language mixing
(Latin/Devanagari, Hinglish) · A20 emoji/punctuation fingerprint (feeds the pair
layer) · A21 edit rate and latency via `_meta.is_edited` · A22 URL rate and domain
concentration **(extend to per-comment link density — URL count / body length —
distinct from the account-level rate; not currently computed)** ·
A39 sentiment/toxicity (mean + volatility) 🔧 · A41 comment-depth tendency
(shallow- vs. deep-commenting bias, from `parent_id` chain-walking — feeds P9).

**A39 sentiment/toxicity** — 🔧 completely absent from the prior version of
this plan (was `C6` in the pre-rewrite draft, silently dropped). No raw field;
computed from `body` text. `nltk` (VADER, lightweight rule-based) and
`transformers` are both confirmed installed in this environment — VADER as the
cheap bulk default, a real classifier if quality matters more than throughput
at census scale. ⚠️ **VADER's lexicon is English-word-based and will degrade on
Hinglish/code-mixed comments** — the same risk already flagged in §9 for
LLM-perplexity detectors. Do not trust it uncalibrated on this corpus.

**Reception** — ⚠️ **weaker than they sound.** 📚 Kumar's community-feedback family
scores **AUC 0.54 alone**; Reddit trolls receive *more* score than normal accounts
(5.7 vs 4.8); LLM-generated Reddit comments draw engagement equal to or higher
than human ones across a 9M-comment study. **Compute as residuals** against
subreddit/hour/thread-age/depth, expect a modest lift, do not build on them.
A23 incoming-reply rate · A24 controversiality rate ✅ (2.1% of comments — real
signal) · A25 score-distribution shape (both tails) · A26 `is_submitter` rate ✅.

**Footprint.** A27 subreddit entropy · A28 share of activity in high-incentive
subs · A29 hobby absence (zero activity in any low-incentive sub) · A30 directed
cross-sub flow (A-then-B; direction distinguishes source from target).

**A34–A38 — the "vs. population mode" family** 🧩. Each is the account's raw
value **expressed against the modal value of the reference population**
(that sub-month's active accounts, or the control tier) — a different feature
than the plain ratio, not a restatement of it:

| # | Metric |
|---|---|
| A34 | karma/account-age-day vs. population mode |
| A35 | comments/account-age-day (A31) vs. population mode |
| A36 | raw posting frequency vs. population mode |
| A37 | reply latency (A33, median) vs. population mode |
| A38 | score-received-per-comment vs. population mode |

### 4.4 Pair level — **the primary detector** 📚

Bipartite **accounts × threads** (also accounts × parent-comment, accounts ×
near-duplicate-text-cluster), validated against a **degree-preserving null**.
This is the rigorous core of the field; everything else is decoration.

| # | Metric |
|---|---|
| B1 | co-appearance count, and excess z vs the null |
| B2 | co-arrival tightness — repeated arrival within Δt on the same thread |
| B3 | text-template sharing rate |
| B4 | stylometric similarity (char n-gram + punctuation + emoji) |
| B5 | registration-cohort adjacency (base36 proximity) |
| B6 | reply reciprocity — do they mainly reply to each other? |
| B7 | temporal correlation of activity series |
| B8 | shared-domain concentration |

**Null model** — three options, increasing rigour 📚:

1. **Hypergeometric SVN** (Tumminello) — exact, simplest, BH-FDR controlled
2. **BiCM + Poisson-Binomial** (Saracco) — `p_rc = x_r y_c/(1 + x_r y_c)`,
   V-motif `V_rr' = Σ_c m_rc m_r'c`, exact survival-function p-values
3. **`backbone` FDSM** — curveball/fastball resampling preserving per-account
   comment counts *and* per-thread sizes exactly. This **is** the
   volume-matched permutation null.

**Coordination interval must be derived from our own corpus.** ⚠️ CooRnet's
`percentile_edge_weight = 0.90` default flags ~10% of edges *whether or not
coordination exists* — a base-rate trap. A 2025 *Scientific Reports* paper shows
transplanted thresholds (10s/5-shares from 2020) miss large amounts of
coordination a year later. **Never inherit a published constant.**

Reference point 📚: Schoch et al. achieved **74% recall at ~1% FPR** on
activity-matched controls using nothing more than a 1-minute co-action window
with ≥10 repetitions.

### 4.5 Rally level 📚

**Key insight: the discriminator is burst *shape*, not size.** Crane & Sornette
relaxation exponents (θ=0.4): exogenous-subcritical ≈1.4, exogenous-critical
≈0.6, endogenous-critical ≈0.2, with endogenous bursts showing power-law
*precursory growth* before the peak.

**The coordinated signature is a burst matching neither class** — tight arrival
cluster then near-silence, no heavy tail — because organic attention always
leaves one.

| # | Metric |
|---|---|
| R1 | changepoint magnitude vs the sub's own baseline (PELT on deseasonalised residuals; CUSUM + BOCPD dual-detector at 0.5 confidence) |
| R2 | Kleinberg burst detection (s=2, γ=1) |
| R3 | relaxation-exponent fit and classification |
| R4 | outsider-influx share |
| R5 | new-account influx (accounts <30d old, sub-relative) |
| R6 | arrival-burst tightness |
| R7 | Hawkes branching ratio `n*` — portable "internal amplification" statistic |
| R8 | **event-conditioned residual** — the metric that matters |

**A nearly-free rally label** 📚: Kumar et al. (WWW 2018) measured moderator
deletion rate **25× higher** during negative mobilisation (0.205 vs 0.008), with
a Reddit-native matched null (matched post: same community, closest in time, no
cross-links; matched user: same activity in past 30 days) giving a **1.6×**
after/before baseline vs **8.8×** for cross-linked threads.

**Event conditioning:** NegBin regression with GDELT lags plus a Reddit-wide
offset, then run burst detection on the **Pearson residuals**, not raw counts.
⚠️ **GDELT DOC 2.0 has a rolling 3-month window** — timelines must be archived
contemporaneously or pulled via BigQuery. CausalImpact/BSTS is stronger but has a
trap: if a campaign hits several of our 45 subs at once, using them as mutual
controls cancels the effect.

---

## 5. Metric sanitisation protocol

Non-optional. Each rule exists because its absence produces a confident wrong
answer.

### 5.1 Concentration metrics are size proxies until corrected 📚

- Raw HHI carries an explicit `1/n` term: `E[HHI] = 1/K + (1−1/K)/n`
- Plug-in Shannon entropy is biased **downward**
- Sample Gini is biased downward by `O(1/n)`

**Fixes:** unbiased Simpson `Σx(x−1)/(n(n−1))`; **Chao–Shen** entropy with
Good–Turing coverage; Deltas `n/(n−1)` Gini correction. Or — universally — report
a **null-model z-score** instead of the raw statistic, which handles any
statistic and any bias.

### 5.2 Transforms: only where they matter 📚

**Skew transforms are pointless for XGBoost.** Tree splits depend only on value
*ordering*, so log1p / Box-Cox / Yeo-Johnson leave the fitted tree identical.
V2's skew-27/41 work matters **only** for z-score composites, PCA/FA, VIF, and
linear baselines — where a skew of 27 genuinely makes a z-score meaningless.

Winsorise at p1/p99 for reporting only, never before tree models. Handle
zero-inflation explicitly (hurdle indicator + magnitude) rather than log1p-ing a
75%-zero column.

### 5.3 Don't hand-build composites for the model 📚

Composite indices impose equal-weight fully-compensatory aggregation, discard the
interactions XGBoost would find, and are **dominated by whichever correlation
cluster has the most members** — a 12-member "volume" family gets 12× the weight
of a singleton. Keep at most one composite as a *reporting artefact*, with
OECD-style uncertainty and sensitivity analysis. Let the model learn interactions.

Prune with VIF **within** evidence families (timing features are collinear with
each other and near-orthogonal to text features — prune within, keep across).

### 5.4 Volume normalisation

Every count metric gets a per-subscriber or per-post denominator, and every
cross-sub comparison carries `log1p(n_posts)` as a covariate. A big sub and a
small sub are not comparable on raw counts.

---

## 6. Labels — per channel, never merged

✅ Measured as **near-disjoint**, so merging them destroys signal.

| Channel | Source | Role |
|---|---|---|
| Confirmed automation | self-declared ("I am a bot", "beep boop", "performed automatically"), `distinguished == moderator` (8.7% ✅) | **seed set** for boundary discovery — benign, so not the target |
| Admin removal | `removed_by_category == 'reddit'` ∪ `_meta.removal_type` | strongest negative-quality signal |
| Automod filtered | `_meta.removal_type == 'automod_filtered'` | separate channel |
| Moderator removal | `_meta.removal_type == 'moderator'` | mostly rule violations — separate, weaker |
| Deletion | `was_deleted_later`, tombstone triple | separate |
| Suspension ⚠️ *optional* | live Reddit at t+90d, **top-risk decile only** | the only account-level, externally-generated signal — but infeasible in bulk (80+ h) and measured weak in V2 |

**Protocol:** one PU model per channel, one class prior per channel, and a **6×6
cross-channel transfer matrix**. **If off-diagonal AUC ≈ 0.5, no single composite
"bot score" is defensible** — and the plan says so publicly rather than shipping
one anyway.

### PU learning 📚

Not-actioned means **unknown**, not negative. Use nnPU risk estimation or
cost-sensitive reweighting with an explicitly estimated class prior (TIcE, BBE,
KM1/KM2, DEDPUL). ⚠️ Elkan–Noto was **degenerate on V2 data** (c=0.054) — do not
assume it will work; check `c` before trusting any prior it reports.

**Evaluating a PU model:** AUC against a PU label is biased. Report PU-corrected
estimates and the recall-at-fixed-FPR against seed sets instead.

---

## 7. Models

**Stage 0 — clean.** Remove sanctioned automation (AutoModerator, self-declared
bots, `distinguished`) from A and B; report separately as A6. Remove
deleted-author rows from modelling (keep for accounting). Compute account age at
**event time**, not collection time.

**Stage 1 — univariate.** Per feature: KDE + **Hartigan dip test** for
unimodality, and a Gaussian mixture with k by BIC. A feature failing unimodality
says two populations exist *before any label is involved*, and the second
component's mass is a label-free prevalence estimate. Check per-month
distributional stability — a feature whose own distribution drifts produces V2's
spurious 13-month climb. **Apply the same test to S15** — a sub's own monthly
post/comment-count series — not only to account features; a sub whose activity
clusters into baseline-vs-spike regimes is structure account-level tests alone
can't surface.

**Stage 2 — bivariate.** Pairwise density grid over the Stage-1 shortlist,
coloured by seed sets. HDBSCAN on the strongest pairs, looking for a satellite
cluster detached from the main mass — with membership probabilities instead of a
hand-drawn line. Interaction screening **requires a stated mechanism in advance**;
V2's `old_x_msgs_per_day` failed because it had none.

**Stage 2 also runs three explicit cross-level segmentation protocols** — not
metrics, comparisons:
1. **Commenter-profile segmentation by P4 (comment:vote ratio) tercile** — do
   accounts commenting on high-P4 posts look different (age, karma/day,
   circadian entropy, …) from those on low-P4 posts?
2. **Same segmentation by P3 (`contested_share`) tercile** — do controversial
   posts draw a different engager population than lopsided ones?
3. **Same segmentation by S15 regime** (baseline vs. spike month) — do
   commenter profiles shift when a sub is in an anomalous month?

None of these are single metrics; they're comparisons of the account-feature
distributions already in §4.3, conditioned on a post- or sub-level split.

**Stage 3 — account model.** XGBoost per label channel. Expect **0.65–0.80** and
say so. Feature selection **inside** the CV fold.

**Stage 4 — pair model.** The primary detector. Features from §4.4, target =
seed-derived same-operator pairs (stylometric + cohort + co-appearance
agreement), validated on held-out seeds. **This is where 0.90 lives.**

**Stage 5 — prevalence.** ⚠️ **Do not average account probabilities.** Two
distinct failures 📚: `E[p̂_CC] = p·tpr + (1−p)·fpr` is systematic and does not
vanish with more data; and the base-rate catastrophe (Botometer flagged ~50% of
the US Congress as bots). BotPercent's temperature-scaling fix still contains an
`argmax` — it is calibrated classify-and-count and fixes only half the problem.

**Correct construction: calibrate → then quantify.** Isotonic (>1000 calibration
points) or Platt, then ACC/PACC/SLD-EM or HDy on top.

**Gate:** Youden's J (`tpr − fpr`) is the ACC denominator. **If J is small,
refuse to publish a prevalence number.**

---

## 8. Validation

### The four-rung ladder — report all four

1. Random CV (optimistic, for reference only)
2. Grouped by account
3. Month-blocked (train early → test late)
4. **Grouped + blocked + purged** — the number that counts

### Mandatory baselines

- **Volume-only baseline: `log1p(n_posts)`.** If the model does not clearly beat
  it on rung 4, **it is a big-subreddit detector**, not a bot detector.
- **Permutation floor.** Every run also executes on volume-preserving permuted
  data, and **both numbers go in the report**. That permuted result is the
  false-positive floor and it costs one extra pipeline run. V2 had none, which is
  why its anomalies were unfalsifiable.
- **Construct-validity check (free, no labels):** the score must rank
  `high > medium > low` incentive tier. If r/ISRO and r/IndianDankMemes score like
  r/IndiaSpeaks, the score is measuring volume or moderation intensity. V2 could
  never run this test because all 25 of its subs were political.

### Leakage register — verify each before fitting

1. **Removal-derived features cannot be used against removal targets.** Hard
   partition. Note this includes *any text feature* when the label is a removal,
   since text is structurally missing for the positive class.
2. `author == '[deleted]'` ↔ suspension, correlated by construction.
3. `retrieved_on − created_utc` correlates with removal timing. Never a feature.
4. **Age at collection vs. age at event** — V2 takes age at collect time, so an
   account's "age" differs across months for reasons unrelated to the account.
5. Account aggregates must exclude the target row (leave-one-out aggregation).
6. Score-derived features against moderation labels — mods act on downvoted
   content, so score partly *causes* the label.
7. Sampling-driven volume leakage — check label rate against activity decile.

📚 Ambroise & McLachlan: selection outside the fold produced near-zero apparent
error where honest error was ~30%. Enforce a **provenance blocklist in CI**.

### Multiple testing

45 subs × 24 months × ~60 metrics ≈ 65,000 tests. At α=0.05 that is ~3,240 false
positives by construction. **BH-FDR across the whole grid**, not per metric.
Under strong dependence BY is available but at this scale `Σ1/j ≈ 11`, making it
~11× more conservative — prefer BH plus effect-size ranking. **At census n,
p-values stop discriminating: rank by effect size.**

### Interpretation

TreeSHAP with `feature_perturbation="interventional"` and family-level
aggregation. ⚠️ Path-dependent TreeSHAP assigns non-zero attribution to features
with **zero** model influence when they are correlated.

---

## 9. What we will not claim

Stated up front, in the UI rather than a methodology footnote.

- **There is no ground truth for coordination.** Seeds are confirmed *automation*
  (benign) and confirmed *platform action* (heterogeneous). Neither is "confirmed
  influence campaign."
- Stage 4 output is "this pair behaves more similarly than the null explains,"
  not "these accounts are the same operator."
- Rally detection has **no AUC** and never will.
- Per-account scores are capped near 0.70 by published evidence. Any 0.90+ we
  report is a **pair-level** number and will be labelled as such.
- ⚠️ Expect LLM/perplexity detectors to false-positive on **Hinglish and Indian
  English**. Botometer's English-vs-German AUC gap (0.90 vs 0.69) is the shape of
  that risk. Do not deploy a text-perplexity detector without a Hinglish
  calibration set.

**Two claims deliberately excluded** as unsafe to cite: an arXiv preprint
reporting AUC 0.977 from account-history features on a 2,432-account balanced
corpus where `verified` alone has Cohen's d = −1.27; and a circulating
"Binghamton 2024, 96% accuracy on Reddit posting rhythm" figure that traces only
to SEO marketing content with no paper behind it.

---

## 10. Sequencing

1. ✅ **Resolve the sub list + sub-month series, done 2026-08-05.** 44/45 resolved
   cleanly against `/time_series` for the 24-month window 2024-08 → 2026-08:
   **3,837,898 posts, 52,949,720 comments** across the resolved subs. Two real
   findings, not artefacts:
   - **`r/unitedstatesofindia`** is a genuine, active sub — `/posts/search`
     returns real live posts — but its **`/time_series` index is empty at every
     precision**, an Arctic Shift indexing gap specific to this sub, not a
     naming problem. Its sub-month counts must be computed client-side from
     `/posts/search` + `/comments/search` rather than `/time_series`.
   - **`r/IndiaTrending` is a ghost sub.** Posts stay healthy (32–700/month
     across the window) but comments collapsed from ~5,270/month (2023) to
     single digits by 2026, despite still carrying 450K+ subscribers —
     subscriber count is a historical residue here, not a current-activity
     signal. Post-level metrics are unaffected; the first-10/top-10 commenter
     layer will be **starved in its 2025–2026 months** and needs a per-cell
     minimum-comment-count flag rather than silently sampling fewer commenters
     than the design assumes.

   Both noted in `subreddits_v3.csv`. Neither blocks collection.
2. ✅ **Pilot: 3 subs × 3 months, done 2026-08-05.** r/IndiaSpeaks (high) ×
   r/IndianStockMarket (medium) × r/ISRO (low), 2026-05/06/07 — 1,051 posts,
   15,806 commenter rows, 5,996 unique accounts. This was **the decision point**,
   and the answer was a qualified go:
   - First pass (engagement-count features: appearance frequency, `pct_toplevel`,
     `pct_first10`) mostly showed sampling-design artefacts, not real structure —
     Zipfian discreteness at small per-account n, not population separation.
   - Second pass, a **450-account behavioral-feature check** (last 50 comments
     each, `fields`-trimmed, ~110s, not full history), found **3 of 7 real
     behavioral features show genuine multimodal structure**: `subreddit_entropy`
     (25% minority), `circadian_dead_hours` (29% minority), `circadian_entropy`
     (14% minority). `burstiness` was flagged by BIC but the two component means
     were nearly identical (0.273 vs 0.299) — **not real**, BIC overfitting noise.
   - The burstiness length-bias risk flagged in A9 was tested directly and did
     **not** materialise in this sample (Spearman vs. `n_gaps` = 0.056, p=0.23)
     — doesn't remove the need for the Kim & Jo correction at full scale, but the
     naive metric wasn't obviously confounded here.
   - Engagement-count features and real behavioral features are **statistically
     independent** (all p > 0.2) — confirms the behavioral layer adds genuinely
     new information rather than restating what appearance counts already showed.
   - One honest null: tier separation held on *footprint* features (subreddit
     entropy, active span) but not on *shape* features (timing, circadian) — and
     with n=1 sub/tier that's still confounded with sub identity, not yet a
     validated tier effect.
   - Reference implementation: `scripts/v3_pilot_collect.py` and
     `scripts/v3_pilot_analyze.py` (post/commenter collection + Stage 1–2
     analysis); `scripts/v3_pilot_behavioral_check.py` (the 450-account
     follow-up). Sub-scoped derived-metric definitions live alongside these, not
     duplicated here.
3. ✅ **Full collection, done 2026-08-06.** 45 subs × 24 months (2024-08 →
   2026-07), 1,080/1,080 cells, zero gaps. **128,374 post rows, 2,312,696
   commenter rows** — both within the §2 budget projections. **178MB**
   compressed on disk (well under the 400–600MB estimate — either zstd beat
   its measured 8–10× on this corpus, or the true mix skewed smaller than the
   ceiling assumption). Total wall clock **~9.4h** across two runs (77min pre-crash
   + 484min resumed — see below) — much closer to the original guess than the pessimistic
   30-minute-checkpoint extrapolation suggested; sustained throughput at
   8 workers held at **~8–10 req/s** for the full run, not just short bursts.
   Reference implementation: `scripts/v3_collect.py`. Data:
   `data/v3/raw/{posts,commenters}/{sub}__{YYYY-MM}.ndjson.zst`, one atomic
   file pair per cell — a cell's existence on disk **is** its checkpoint, so a
   killed run resumes by skipping every completed cell.
   - **Bug found and fixed mid-run:** `matched_random_sample`'s target-quantile
     step divided by `k−1`, which is zero whenever a sub-month has *exactly*
     101 posts (100 top + a single leftover for the counter sample →
     `k=min(20,1)=1`). Crashed the whole process after 77 minutes / 102 cells.
     Checkpointing meant the crash cost only wall-clock time, not re-fetched
     data — restart skipped the 102 done cells and resumed at cell 103.
   - **Post-hoc integrity audit (`scripts/v3_data_qc.py`) found a second,
     quieter defect:** `fetch_all_comments` treats a fully-retry-exhausted
     page request (`get()` → `None`) identically to "no more pages," so a
     transient failure mid-pagination silently truncates that post's thread
     instead of surfacing an error. Detected by comparing
     `n_comments_observed` against `num_comments_reported` — comments/search
     returns removed/deleted comments too (with removal metadata), so a large
     shortfall isn't explained by legitimate removal and is the truncation
     signature. **645 posts (0.50%) across 218 cells** were flagged
     (`num_comments_reported ≥ 10` and `n_comments_observed < 50%` of it),
     scattered across many subs/months with no structural pattern — consistent
     with transient failures, not a permanent per-sub API gap (manually
     re-fetching several flagged posts immediately returned full data).
     Repaired via `scripts/v3_repair_truncated.py`, which re-fetches just the
     flagged posts' threads and patches the two affected cell files in place,
     reusing the same derivation code as the collector
     (`comment_derived_fields_and_rows`, extracted from `process_post` for
     exactly this reuse). **645/645 improved, 0 unchanged** on the repair
     pass; re-running the audit afterward found 0 remaining candidates.
   - **QC also surfaced a real methodological point, not a defect:**
     `n_comments_observed` summed **14.3% higher** than `num_comments_reported`
     across the corpus (14.07M vs 12.31M). `num_comments_reported` is
     Reddit's own T+36h snapshot count (§3) and, like `score`, goes stale —
     threads keep accumulating comments after that snapshot, and our own
     `comments/search` scan (run at collection time, months to years later)
     catches the growth. **`num_comments_reported` should be treated as a
     T+36h velocity snapshot (feeds P22), not a ground-truth total;
     `n_comments_observed` is the more complete count for any metric wanting
     "how big did this thread get."**
   - Known-edge-case subs behaved exactly as §10.1/§11 predicted:
     `DesiVideoMemes` shows full months through 2026-03 then hard zeros
     2026-04 onward (matches the confirmed 2026-03-19 death date);
     `IndiaTrending` shows post counts declining but nonzero through
     2026-07 (32–114/month, matching the "posts survive, comments collapse"
     finding); `unitedstatesofindia` collected cleanly at full volume every
     month via `/posts/search` (never depended on the broken `/time_series`
     index).
   - Everything else audited clean: 0 duplicate `post_id`s within a cell,
     0 negative arrival gaps, 0 out-of-range `upvote_ratio`, `author_fullname`
     present on 99.9%/99.2% of posts/commenters (matches the §3 base36
     viability finding), comment `body` text present on 100% of commenter
     rows (needed for Stage 4 B3/B4 — see §2). One expected-not-a-bug
     artefact: ~30% of commenter rows share a `comment_id` with another row
     in the same cell, because a comment that's simultaneously in a post's
     first-10-arrivals *and* top-10-by-score is deliberately stored twice
     (once per `commenter_tag`) — Stage 0 cleaning must `drop_duplicates` on
     `comment_id` for any analysis that would otherwise double-count a single
     comment's engagement, while keeping the tag-level rows for role-based
     features (P8, first-10 arrival gaps).
4. **Account + pair models**, with the permutation floor from run one. Broken
   into Stage 0–4 per §7; Stage 0 done, Stages 1–4 not yet started:
   - ✅ **Stage 0 (clean + analysis layer), done 2026-08-06.** Built
     `scripts/v3_stage0_build.py`: DuckDB reads the raw `.ndjson.zst` cells
     directly (`read_json_auto(..., union_by_name=true)`, no separate Parquet
     conversion step needed) into a persistent `data/v3/analysis/v3.duckdb`
     (423MB). Adds `is_confirmed_automation_seed` (§6's seed channel:
     self-declared bot phrases, `AutoModerator`/`*bot` author names,
     `distinguished == 'moderator'`), `is_deleted_author`, and
     `account_ordinal` (base36-decoded `author_fullname[3:]`, §3/A1 — a raw
     monotonic-with-creation-order integer, **not yet calibrated to a real
     date**, that's the still-optional item 6 below) as columns on `posts`
     and `commenters`, without dropping any row from the base tables — a
     `commenters_clean` / `posts_clean` view filters on top, so accounting
     stays possible. `commenters_dedup` resolves the ~30% intra-cell
     `comment_id` duplication noted above.
     - **Bug found and fixed:** the first cut of `is_confirmed_automation_seed`
       used `distinguished = 'moderator' OR ...` directly — classic SQL
       three-valued-logic trap, `NULL OR FALSE = NULL` (not `FALSE`), and
       `distinguished` is `NULL` for the overwhelming majority of rows. `NOT
       NULL` is `NULL`, and `WHERE` drops `NULL` silently, so
       `commenters_clean` first came out at **2 rows** instead of ~1.6M.
       Fixed by wrapping the whole boolean expression in `COALESCE(..., FALSE)`.
     - Results: posts 128,374 total, 0.24% automation-seed, 0.09%
       deleted-author, 99.9% `account_ordinal` resolved. Commenters 2,312,696
       raw → **1,619,492 deduped** (30.0% tag-duplicate, exactly matching the
       QC finding) → **1,616,023 in `commenters_clean`** (99.8% of deduped),
       spanning **347,886 distinct accounts**. Tier split (high/medium/low)
       294K/509K/814K rows — plausible given the control tier includes more,
       higher-volume subs.
   - ✅ **Account-level feature table (A1–A41, §4.3), done 2026-08-06.**
     `scripts/v3_account_features.py`, writes `account_features` (347,886
     rows, 30 columns) into `v3.duckdb`.
     - **Load-bearing finding, discovered before building any timing
       feature:** the corpus is a **bipartite sample**, not a per-account
       census — an account only appears when it's in a *sampled* post's
       first-10-arrivals or top-10-by-score. Measured distribution:
       **median 2 comments/account, 48.4% of all 347,886 accounts are
       singletons** (exactly 1 comment in the whole 24-month corpus), only
       **21.0% (72,939) have ≥5**, only ~10% have ≥10. This is not a defect —
       it's the direct empirical reason the plan's own account-vs-pair split
       (§1) is the right call: most accounts simply don't carry enough
       individual signal for a timing feature to mean anything, which is
       exactly why the pair layer (§4.4, co-appearance across accounts) is
       the primary detector rather than an account-level score.
     - **Two tiers, gated on sample size accordingly:** Tier 1 (all 347,886
       accounts) — provenance (`account_ordinal`, uncalibrated), footprint
       (`subreddit_entropy`, tier shares, `hobby_absence`), reception
       (mean/median/stddev comment score, controversiality rate, submitter
       rate, mean depth), username morphology (A3 — regex for Reddit's
       `Adjective-Noun-NNNN`/`Adjective_Noun_NNNN` auto-generated pattern,
       **42.9% match** — visually spot-checked against 30 random matches,
       genuinely the auto-generated format, not regex overmatch, e.g.
       `Ok-Willingness-6039`, `Honest_Lettuce_7181`). Tier 2 (the 72,939
       accounts with ≥5 comments) — A7 interval entropy, A9 burstiness via
       the **Kim–Jo finite-size correction** (§4.3's own warning about the
       naive Goh–Barabási statistic being length-biased — implemented the
       closed-form `A_n` correction, not the naive `B`), A8 interval
       quantization (cron-signature detection). Range check: `burstiness_kimjo`
       came out in `[-0.94, 0.9999...]`, inside the theoretical `[-1,1]` bound
       — not proof of correctness but rules out a broken formula.
     - **Deliberately not built: A10–A13 (circadian dead hours, circadian
       centroid, weekday:weekend ratio, session structure).** The pilot's
       450-account behavioral check (§10.2) found real multimodal structure
       in circadian features — but that used each account's **last 50
       comments via a separate direct API pull**, not this bipartite sample.
       At median n=2 in-corpus, "zero activity in hour H" is a sparsity
       artefact for ~90% of accounts, not a circadian signal, and there's no
       Kim–Jo-style correction for that the way there is for burstiness.
       Doing this properly at scale means a **supplemental full-history pull**
       (same method as the pilot check) for a bounded account subset — not
       yet started, and not faked here with an underpowered proxy.
     - Two SQL bugs hit and fixed while building this (both would have
       silently miscounted, not crashed): (1) an ambiguous `author` column
       reference across a `commenters_clean`/`posts` join, caught immediately
       by DuckDB's binder; (2) `con.register()` doesn't accept a raw Python
       list of dicts (only DataFrame/Arrow/relation/ndarray) — fixed by
       wrapping in `pandas.DataFrame` before registering.
   - ✅ **`removal_rate` / `deleted_later_rate` added as account features,
     done 2026-08-06** — user-prompted, and the single strongest signal found
     in this stage so far. Hypothesis (from direct Reddit browsing experience,
     not the literature): accounts with thin visible history are often ones
     whose content got quietly removed/hidden, not just accounts we happened
     to under-sample. Tested directly against `meta_removal_type` /
     `meta_was_deleted_later` (already collected, just not previously rolled
     up to account level) rather than assumed:

     | times seen in corpus | n accounts | mean removal rate |
     |---|---|---|
     | 1 (singleton) | 168,458 | **8.8%** |
     | 2–4 | 106,489 | 6.0% |
     | 5–9 | 38,638 | 4.7% |
     | 10+ | 34,301 | **3.8%** |

     Clean monotonic gradient, singleton accounts removed **~2.3×** more often
     than 10+-comment accounts. Real signal, not proof of any single cause —
     equally consistent with new-account spam filtering or drive-by
     throwaways as with shadowbanning specifically — but a materially better
     account-level feature than anything Stage 1's automated screen
     surfaced on its own, and it directly connects to the §6 label channels
     (which *are* removal-based), unlike most of the A-family.
   - ⚠️ **Stage 1 (univariate screening) attempted 2026-08-06, output not
     trustworthy, needs redoing properly.** `scripts/v3_stage1_univariate.py`
     ran dip test + GMM/BIC across account_features and flagged **18 of 19
     features "bimodal."** Not credible on its face (pilot found 3 of 7).
     Traced to three compounding pitfalls, two fixed, one still open:
     1. **Zero/point-mass inflation** (§5.2) — `n_posts_sample` etc. are
        >=25–90% a single value; a GMM fit to the raw distribution finds a
        near-zero-variance spike component, producing separation scores in
        the hundreds (693, 4000). **Fixed**: hurdle split, point-mass share
        reported on its own, dip/GMM run only on the remainder.
     2. **Dip test + BIC both break at census n.** Subsampling only the dip
        test to diptest's validated n<=72,000 wasn't enough — BIC's `log(n)`
        penalty means even a trivial fit gain from an extra Gaussian reads
        as "significant" at n in the hundreds of thousands. Generalizes §8's
        "at census n, p-values stop discriminating" to model selection, not
        only significance testing. **Fixed**: both the dip test and the
        GMM/BIC fit run on the same fixed, seeded n=20,000 subsample.
     3. **Still open: GMM/BIC on a heavy-tailed-but-genuinely-unimodal raw
        feature will always "find" extra components.** `mean_comment_score`'s
        raw histogram is a single peak near 0–20 with a smooth monotonic
        decay to 4,625 — no visible second mode — but a Gaussian mixture
        approximates that skew with 2–3 offset components regardless,
        because that actually gets the true underlying data density closer,
        even though there is only one real population. This is *not* the
        same bug as (1); it survives the point-mass fix because there's no
        single dominant value to split off, just smooth skew. Needs a
        **per-feature transform pass** (signed-log for signed heavy-tailed
        features, not just non-negative counts) plus an actual KDE-valley
        visual check before any "bimodal" verdict is trusted — not more
        automated threshold tuning, which would just be tuning until the
        output looks plausible rather than being correct. **Not done.**
     Given §1 already frames account features as feeding the pair layer
     rather than being the deliverable, this is deprioritized relative to
     Stage 4 rather than fixed immediately.
   - ✅ **EDA page, done 2026-08-06.** `scripts/v3_eda_build.py` (data prep,
     DuckDB → summary stats only, nothing raw leaves the database) +
     `scripts/v3_eda_template.html` (page shell) → `docs/v3-research/eda/index.html`,
     git-tracked and rebuildable after any `account_features` change. Per
     feature: raw + transformed histogram (toggle), point-mass flag; a
     Spearman correlation matrix across the 19 non-timing features (rank
     correlation specifically to sidestep re-litigating a transform choice
     per pair — invariant to it by construction); 3 log-scaled 2D density
     panels for pairs called out by findings above. Doubles as the running
     findings log's second surface (condensed version of this section).
   - ✅ **User-directed feature round, done 2026-08-06** — activity span,
     conversation-engagement, and silo-mismatch features, plus two hypothesis
     tests run directly against them.
     - `observed_span_days`, `comments_per_day_observed`,
       `sample_score_per_day_observed` — first_seen/last_seen_utc were
       already computed in Stage 0 but never turned into rate features.
       Explicitly **not** true account age (that needs the base36
       calibration below) — this is rate-of-observed-activity, a different
       and immediately-available quantity.
     - `repeat_engagement_rate` (does an account ever comment more than once
       in the *same* sampled thread) and `own_post_reply_rate` (rolled up
       from the post-level `submitter_reply_rate` already collected, for the
       ~13% of accounts who authored a sampled post) — built to test a
       "bots post-and-leave regardless of whether they're ragebaiting
       (downvoted) or in an aligned/appreciation context (upvoted)"
       hypothesis.
     - **Hypothesis test 1 (null result, kept in the record):** compared
       `mean_comment_score`'s distribution shape (dip test + GMM, the
       corrected Stage-1 methodology) between "post-and-leave"
       (`repeat_engagement_rate=0`) and "engaged" accounts. **Both groups
       show near-identical bimodal structure** (component means barely
       differ, ~70/30 weights in both) — the bimodality looks like a
       generic property of how Reddit scores comments, not something
       specific to disengaged accounts. Only 1.4–2.4% of accounts in either
       group have a negative mean score, so the "ragebait gets downvoted"
       half isn't showing up at volume on this operationalization. Caveats:
       "never double-posts in the same thread" isn't the same claim as
       "ignores replies," and this was tested pooled across all 45 subs,
       which could dilute a sub-tier-specific pattern.
     - `n_subs_rejected_but_returned`, `best_sub_mean_score`,
       `worst_sub_mean_score`, `reception_spread`,
       `shows_silo_mismatch_pattern` — reframed from a rejected first
       version of this idea. The first framing (score variance across subs
       = context-independence = bot-like) was wrong: a human who simply
       never leaves their aligned silo would *also* show low variance, for
       an unrelated reason (restricted footprint, not context-blindness).
       The corrected version needs no political-lean labels on subs at
       all — it only asks whether an account **keeps returning to a sub
       where it's net-downvoted** (`mean_score_in_sub < 0`, active there
       across ≥2 distinct months) **while doing well elsewhere**
       (`best_sub_mean_score > 5`). A human's self-preservation instinct
       predicts avoidance of hostile territory; a script has no such
       feedback loop.
     - **Hypothesis test 2 (positive, volume-controlled):** 2.0% of the
       126,835 multi-sub accounts show this pattern. They run **~2×
       controversiality rate (6.2% vs 3.1%) and ~1.3× removal rate (6.3% vs
       4.8%)** vs. other multi-sub accounts. Checked for the obvious
       confound (accounts active in more subs get more chances to hit this
       pattern by pure volume) by re-running within `n_subs_active` buckets
       (2–3 / 4–6 / 7+) — **the ~2× controversiality gap holds inside every
       bucket**, not just in the pooled comparison. Real signal, not a
       volume artefact — the strongest account-level finding in this stage
       after `removal_rate` itself.
   - **Base36 age calibration: put on hold by user decision, not abandoned
     for cause.** Live Reddit access confirmed working
     (`scripts/reddit_auth.py`, OAuth, 53 req/min — the original audit's 403
     was a datacenter-IP artefact as already suspected, not a credential
     problem) and `scripts/v3_calibrate_age_sample.py` is written
     (stratified sample across the observed `account_ordinal` range, pull
     real `created_utc`, isotonic lower-envelope regression ordinal→date).
     Not run: the user opted for a free alternative instead of ~1,750 live
     lookups — see below.
   - ✅ **Age proxy: `days_since_first_seen`, done 2026-08-06.** User's
     framing: use *today minus first-seen* as the age denominator, and let
     per-account noise cancel out at n=347,886 rather than paying for
     external calibration. Materially better than the `observed_span_days`
     already in the table (`last_seen - first_seen`) for this purpose —
     span is exactly **0 for the 48% of accounts that are singletons**,
     since one data point has no span, whereas `days_since_first_seen` is
     non-zero for every account regardless of how many times we caught
     them (range 1.8–735.5 days, median 421.6, matching the ~24-month
     window). Feeds `comments_per_day_since_first_seen`,
     `posts_per_day_since_first_seen`, `karma_per_day_since_first_seen`.
     Explicitly still a proxy, not true account age — an account first
     spotted last month could have signed up years ago and just not have
     been sampled into a top-100 post before; the docstring in
     `v3_account_features.py` spells out all three age-like quantities now
     in the table (`account_ordinal`, `observed_span_days`,
     `days_since_first_seen`) so they don't get conflated later.
   - ✅ **Risk-colored histograms, done 2026-08-06.** User's ask: color every
     univariate bar by mean `removal_rate` within that bin (min 20
     accounts/bin, else rendered gray — too few to trust) so the EDA page
     shows *where* elevated risk sits in each feature's distribution, not
     just that a feature has structure. First version used the account-level
     p95 of `removal_rate` (50%) as the color-scale max — visually flat,
     because bin *averages* are far more compressed than individual-account
     values. Fixed by calibrating the scale to the actual observed range of
     bin-level means (p95 ≈ 20%) instead. Surfaced a real pattern in the
     process: `account_ordinal` bins run from **3.5% removal (oldest
     accounts) to 13.3% (newest)** — near-monotonic, ~4× — newer accounts in
     this corpus are meaningfully likelier to get removed. Worth its own
     look in Stage 3, not just a EDA-page curiosity.
   - ✅ **Bot-marker composite, done 2026-08-06** (`scripts/v3_botmarker_composite.py`).
     Explicitly an **unsupervised separation check, not a validated
     detector** — no label exists yet. Six theory-motivated markers, each a
     percentile rank (0–100, higher = more suspicious) so they combine
     without hand-tuned weights: `removal_rate`, `deleted_later_rate`,
     `thin_history_score` (inverse of `n_comments_sample` — "not showing any
     history"), `karma_extremeness` (`|mean_comment_score − median| / MAD`
     — catches **both** ragebait-downvoted and appreciation-upvoted
     archetypes, not just one direction), `karma_per_post_extremeness` (same,
     for authored posts — only ~13% of accounts have one), `reception_spread`.
     `botmarker_composite` = mean of whichever marker percentiles are
     non-null (≥3 required). Kept `karma_extremeness` and
     `karma_per_post_extremeness` as two separate markers rather than
     pre-merging them — the composite already blends both for accounts that
     have both, while keeping them separate lets each one's own power be
     inspected, which a pre-merge would have thrown away.
     - **The trustworthy result: top-1%-by-marker profile comparison.**
       Top 1% by `reception_spread` alone (removal_rate 4.6%) and by
       `karma_extremeness` alone (3.3%) both sit *below* the population
       average (7.0%) — in isolation, these two markers are **not**
       removal-aligned; their extremes look more like genuinely popular
       high-variance accounts than bot signatures. Top 1% by
       `botmarker_composite` hits **29.5% removal rate** (~4× population
       average) with `mean_n_comments=11.2` — a *different* population than
       top-1%-by-`removal_rate`-alone (which is mechanically 100% removal
       by construction, and turns out to mostly be one-off singleton
       accounts, `mean_n_comments=1.13`, i.e. "one bad comment that got
       nuked," not a sustained pattern). **The composite finds a more
       diffuse, more consistently-elevated population than any single
       marker's own top tail — genuine evidence combining markers adds
       something, without needing a bimodality test to say so.**
     - ⚠️ **The dip-test/GMM screen on these markers is not trustworthy as
       reported and needs the same skepticism as Stage 1's first pass.** All
       7 (6 markers + composite) came back "real candidate," including
       markers that are themselves percentile ranks — which should be
       exactly uniform for a continuous variable and never show real
       bimodality on their own. Root cause: `percent_rank()` doesn't spread
       out ties, so a percentile rank *of an already zero-inflated variable*
       (e.g. `removal_rate`, 85% mass at 0) is itself still lumpy at the low
       end — the point-mass hurdle catches the *single* largest tie but not
       secondary clumps from other common fractions (1/6, 1/2, 1), which can
       still fool GMM. `removal_rate`'s separation score (19.03) landed right
       at the degenerate-ceiling edge (20) — suspicious on its face. **Not
       fixed. Flagged for next session, not silently trusted.**
   - ✅ **Stage 1 fix + feature sanitisation, done 2026-08-06.** Two threads
     run in parallel.
     - **The `percent_rank()` diagnosis above was wrong; the real bug was in
       the screen, not the ranking function.** `percent_rank()` is
       mathematically correct — the ties it wasn't "spreading" are *real*
       ties, an artefact of small-n rate features on a median-2-comments
       corpus (`removal_rate` ∈ {0, 0.5, 1, 1/3, ...} has genuinely few
       distinct values). The actual defect: the point-mass strip in
       `v3_stage1_univariate.py` only removed the single largest point-mass
       before running dip-test/GMM, so secondary clumps (0.5, 1/6, 1)
       survived and fooled it. Fixed with an iterative strip (every value
       ≥1% of n, capped at 8 values / 50% of n so it doesn't gut wide-range
       count features). Separately fixed the still-open GMM-decorates-skew
       bug (§10.4 above) with a KDE-valley check (`kde_valley_ratio` —
       requires an actual density dip between component means, not just
       dip-test-p + BIC-k agreement) plus `signed_log1p` for signed
       heavy-tailed features (comment/karma scores can be negative).
     - **Before/after, both screens rerun on the fixes:** the raw-feature
       screen (expanded 19→36 features, folding in the previously-unscreened
       "user-directed round" rate features) went from 18/19 "bimodal" → **9
       candidates**, only 2 (`mean_comment_score`, `account_ordinal`) robust
       across two different strip-threshold settings tried — flagged as
       genuinely threshold-sensitive (3 vs. 9 between the two settings) per
       the plan's own instruction not to just tune until output looks
       plausible. `account_ordinal`'s split is likely mostly Reddit's own
       non-uniform historical growth rate, not a behavioral population —
       flagged before anyone cites it. `mean_comment_score` looks like a
       real split: ~7% minority with negative mean score vs. 93% positive.
       The bot-marker composite screen (§10.4 above, "all 7 real candidate")
       went from 7/7 → **1/7** (`karma_extremeness` only) — `removal_rate`
       and `botmarker_composite` no longer show genuine bimodality once
       skew-decoration is filtered out, confirming the suspicion already
       logged above about the 19.03 score sitting at the degenerate ceiling.
       **Still open, not silently fixed:** 3 features (`n_posts_sample`,
       `n_high_tier`, `n_threads_with_repeat`) return an unchanged DEGENERATE
       verdict (sep=693.15, identical across settings) — a residual
       near-zero-variance artefact the strip cap doesn't fully catch.
     - **Stage-3 prep per §5, new script `scripts/v3_feature_sanitise.py` →
       `account_features_model` table (347,886 × 73 cols) in `v3.duckdb`.**
       24 zero-inflation hurdle indicators added. Tier counts
       (`n_high_tier`/`n_medium_tier`/`n_low_tier`, raw counts that confound
       a 2/2 high-tier account with a 2/50 one) replaced with volume-normalized
       shares (`high_tier_share` etc. = count / `n_comments_sample`).
       VIF-pruned *within* evidence families only (§5.3): dropped
       `n_threads_active` (VIF 654.6, near-duplicate of `n_comments_sample` at
       median n=2), `n_subs_active` (VIF 12.4), `best_sub_mean_score`
       (VIF=inf), `mean_comment_score` (65.3), `median_comment_score` (12.8,
       both collinear with `worst_sub_mean_score`/`score_stddev`). Also
       dropped `n_distinct_threads` after verifying it byte-for-byte
       identical to `n_threads_active` across all 347,886 rows, not merely
       correlated. Bot-marker percentile family (`botmarker_composite`,
       `*_pctl` columns) excluded from model inputs per §5.3 — 3 of 4 are
       pure monotonic transforms of a raw column already in the set, so
       redundant for a tree model on top of being philosophically composite.
     - Files: `scripts/v3_stage1_univariate.py` (rewritten),
       `scripts/v3_feature_sanitise.py` (new). `account_features_model` is
       what Stage 3 should read from, not raw `account_features`.
   - ✅ **Stage 2 (bivariate/segmentation), done 2026-08-06.**
     `scripts/v3_stage2_bivariate.py` (read-only DuckDB) +
     `scripts/v3_stage2_template.html` → `docs/v3-research/eda/stage2.html`,
     linked from the main EDA page nav.
     - **Pair selection substituted for the Stage-1 shortlist** (unreliable
       at the time this ran, per the parallel fix above — the two threads
       didn't block each other) — ranked Spearman |ρ| over a 19-feature
       theory-motivated candidate set instead, excluding pairs already shown
       on the main EDA page. Pairs with |ρ|≥0.98 dropped as definitional
       (e.g. `thin_history_score` is a direct inverse of `n_comments_sample`,
       excluded from the candidate set entirely); 0.90–0.98 pairs kept but
       flagged on the page as "shared-construction risk," not a behavioral
       finding. Documented on the page itself, not silently substituted.
     - **HDBSCAN on the top 3 pairs — one real finding, one artefact, one
       cross-check:** `mean_comment_score × score_stddev` shows genuine
       satellite structure (~3.9% of points at 10–19% removal rate vs. 4.8%
       main-mass / 5.2% population mean) — the one result matching the
       plan's literal "satellite cluster detached from the main mass"
       criterion. `removal_rate × deleted_later_rate` clusters landed almost
       exactly on rational fractions (0, 0.25, 0.33, 0.5, 1) — reported as a
       small-n rate-denominator artefact, not behavior, directly reinforcing
       the zero-inflation bug fixed above rather than contradicting it.
       `reception_spread × subreddit_entropy` gave a clean 2-cluster split
       where the *broad*-footprint cluster is safer (4.8% removal) and the
       narrow-footprint cluster riskier (8.0%) — extends the main EDA page's
       existing top-1% `reception_spread` finding to the full population.
     - **Segmentation protocol 1 (P4 tercile):** pooled removal_rate looked
       flat (6.92% high vs. 6.91% low) — a Simpson's-paradox artefact of
       low-P4 skewing toward singleton accounts (elevated baseline risk for
       an unrelated reason). Breaking out by volume bucket shows high-P4
       *consistently* higher removal at every bucket (e.g. 9.3% vs. 8.2% at
       n=1, 4.0% vs. 3.4% at n=10+) — real, confound-checked signal the
       pooled number alone would have hidden, following the same
       within-bucket-recheck discipline as the existing silo-mismatch
       finding above. Controversiality runs the opposite direction (high-P4
       lower), also holding within buckets.
     - **Segmentation protocol 2 (P3 tercile):** clean without needing a
       confound check — high-`contested_share` accounts run ~3× the
       controversiality rate (4.6% vs. 1.5%) and modestly higher removal
       (7.4% vs. 6.8%).
     - **Segmentation protocol 3 (S15 regime):** only 7 of 45 subs show
       genuine 2-component baseline/spike structure at n=24 months — reported
       as exploratory given the small count, not overclaimed. Spike-exposed
       accounts (5.7% of tagged) are *lower* risk (5.3% vs. 6.9%) and skew
       older/broader-footprint/higher-karma — pushes back on a naive "spike
       month = bots" account-composition read, consistent with §4.5's framing
       that the coordinated signature is burst *shape*, not who shows up.
     - Deviations, both caveated on the page: used `interval_entropy`
       (tier-2 only, n≥5 comments) as the closest available proxy for
       "circadian entropy" since true circadian features (A10–13) aren't
       built; used Σ`n_comments_observed` per sub-month (not post count) for
       S15, since post count is capped by the fixed top-100/month sampling
       quota and can't show upside spikes.
   - ✅ **Stage 3 (account XGBoost per label channel), done 2026-08-06.**
     `scripts/v3_stage3_account_model.py` → `docs/v3-research/eda/stage3.html`
     + `stage3_data.json`, linked from the EDA/Stage-2 nav. New deps:
     `xgboost` (needed a system `libomp` via `brew install libomp` — the pip
     wheel's dylib load failed without it), `shap`, and `statsmodels` (was
     silently missing even though `v3_feature_sanitise.py` already imports it
     — a latent break for any fresh environment, now pinned).
     - **Channel set is a data-driven adaptation of §6's literal 6, not a
       literal build.** Verified directly: `meta_removal_type` has different
       granularity in `commenters` (`{deleted, removed, removed by reddit}` —
       `'removed'` doesn't distinguish automod from moderator) vs. `posts`
       (full granularity, but only 55,688/347,886 = 16% of accounts authored
       a sampled post). Built 5 channels: `admin_removal` and
       `self_deletion` (full population, combining post+comment signal),
       `comment_removed_ambiguous` (comment-only, automod/mod
       indistinguishable), `automod_filtered` + `moderator_removed`
       (posts-only, restricted to the 16%). Skipped confirmed-automation
       (seed set, not a target — already excluded from
       `account_features_model`'s source population at Stage 0) and
       suspension (deprioritized elsewhere, alongside the on-hold base36
       calibration). All 5 channels had healthy positive counts (3,204–47,730
       full-population; 4,395–4,970 of the post-author subset) — none needed
       dropping.
     - **Leakage register (§8) applied literally.** Hard-excluded
       `removal_rate`, `deleted_later_rate`, their `_nonzero` hurdle
       indicators, and the 6 reporting-only columns from every channel's
       feature matrix, since `account_features_model` still carried the raw
       removal-rate columns inside its nominally "model-ready" set (no label
       existed yet when the sanitisation pass ran, so it couldn't have
       excluded them for this reason). **Found, not fixed:**
       `v3_feature_sanitise.py`'s docstring claims
       `karma_extremeness`/`karma_per_post_extremeness` are reporting-only,
       but the actual table has no such exclusion — they're still in the
       model-ready set. Treated as ordinary score-derived features here
       (§8 leakage item 6) rather than silently trusting the docstring.
       Score-derived-feature sensitivity (item 6) checked per channel via a
       with/without-score-family rerun (`rung4_no_score` below) rather than a
       silent include/exclude call.
     - **PU learning:** one Elkan-Noto `c` estimate per channel, checked for
       degeneracy (V2's was 0.054). All 5 landed 0.53–0.75 — none degenerate.
     - **Four-rung ladder + a subreddit-blocked check, rung 4 is the number
       that counts:**

       | channel | rung1 (random) | rung4 (grouped+blocked+purged) | sub-blocked | rung4 no-score | volume-only | perm floor | PU `c` |
       |---|---|---|---|---|---|---|---|
       | `admin_removal` | 0.924 | **0.896** | 0.883 | 0.874 | 0.616 | 0.516 | 0.750 |
       | `self_deletion` | 0.803 | **0.778** | 0.781 | 0.765 | 0.642 | 0.601 | 0.640 |
       | `comment_removed_ambiguous` | 0.827 | **0.805** | 0.795 | 0.799 | 0.591 | 0.581 | 0.660 |
       | `automod_filtered` | 0.735 | **0.653** | 0.605 | 0.644 | 0.556 | 0.514 | 0.558 |
       | `moderator_removed` | 0.694 | **0.637** | 0.684 | 0.637 | 0.568 | 0.540 | 0.534 |

       All 5 clear both mandatory baselines (volume-only, permutation floor)
       by a wide margin — none of this is a big-account or a rediscovered-FP
       detector.
     - ⚠️ **`admin_removal`'s 0.896 sits well above the plan's own 0.65–0.80
       ceiling — flagged and investigated, not silently reported.** Ran an
       ablation suspecting temporal-truncation leakage (a banned account's
       observed activity window is mechanically capped by the ban date):
       dropping the whole `provenance_age` feature family barely moved it
       (0.902 without it, *higher* — rules out that specific mechanism);
       `provenance_age` alone only reaches 0.835, so the signal is broadly
       distributed across families, not concentrated in one leaky group.
       Working hypothesis, not proof: `'removed by reddit'` is a more extreme
       construct (platform-wide ban / blatant spam / ban-evasion) than
       Kumar's generic sockpuppet label, so a higher ceiling on this specific
       channel isn't inherently implausible — but this needs scrutiny before
       0.896 gets cited anywhere as validated.
     - ⚠️ **Construct-validity check (predicted score should rank high >
       medium > low incentive tier) holds for 2/5 channels, violated for
       3.** Holds cleanly for `admin_removal` and `moderator_removed`.
       Violated for `self_deletion`, `comment_removed_ambiguous`, and
       `automod_filtered` — all three dip specifically at the *medium* tier
       (high > low > medium), a consistent pattern across 3 independent
       channels, not noise. Flagged, not explained — out of scope for this
       stage.
     - **Cross-channel transfer matrix** (common post-author population,
       n=55,688; diagonal 0.77–0.84): `admin_removal` transfers weakly
       to/from every other channel (0.52–0.63) — the most distinct
       construct, consistent with V2's experiment B finding
       (admin-removal and suspension were near-disjoint, §0). `self_deletion`
       ↔ `comment_removed_ambiguous` transfer unusually well both directions
       (0.72–0.77), suggesting real construct overlap between those two.
     - **Top SHAP families, family-level TreeSHAP aggregation:**
       `admin_removal`/`self_deletion`/`comment_removed_ambiguous` all lead
       with `provenance_age` (top individual features:
       `comments_per_day_since_first_seen`, `account_ordinal`,
       `observed_span_days`) and `reception`; `automod_filtered`/
       `moderator_removed` lead with `reception`/`engagement_pattern`
       (`mean_post_score`, `own_post_reply_rate`) instead — the "who is this
       account" channels and the "was this specific post bad" channels split
       on different feature families, not just different AUCs.
   - ✅ **Stage 3 leakage audit + correction, done 2026-08-06 (same day,
     second pass), user-directed** ("dig in first, need full sanitisation and
     least assumptions before moving fwd" — in response to `admin_removal`'s
     0.896). Same files as Stage 3 (`scripts/v3_stage3_account_model.py`,
     `docs/v3-research/eda/stage3.html`/`stage3_data.json`); original numbers
     kept visible on the page, labeled as superseded, not deleted.
     - **Phase 1 diagnostics.** Singleton share of `admin_removal` positives:
       735/3,204 = 22.9% (their one sampled row *is* the label-defining row).
       Rung-4 AUC restricted to `n_comments_sample ≥ 5` → 0.822; `≥ 10` →
       0.743 (inside 0.65–0.80). Score-corruption sub-hypothesis (removed
       content's `score` distorted by the §3 `_meta` block's +16s/+36h
       snapshot merge) tested directly on the 207 accounts with both removed
       and kept comments: admin-removed comments score *higher* (mean 31.5)
       than the same account's kept comments (mean 19.2) — rejected, the
       opposite of what a depressed-score artefact would predict.
     - **Mechanism confirmed by direct SQL inspection of
       `v3_account_features.py`:** every behavioral aggregate is computed
       over all of an account's sampled rows, with no exclusion for rows
       that themselves satisfy a removal condition. Not inferred from the
       AUC pattern — read directly off the query.
     - **Leave-one-out fix attempted, rejected — made things worse.**
       Rebuilding features per-channel excluding each channel's own
       label-defining rows leaves thin-history accounts with `NaN` features,
       which XGBoost's native missing-value handling then exploits as a
       near-perfect label proxy (a different, worse leak). Numbers that
       exposed this: rung-1 jumped to 0.97–0.98 across the board (nonsense —
       random-CV should never be *that* clean); rung-4 became wildly
       inconsistent by channel (`self_deletion` collapsed to 0.419,
       `automod_filtered` spiked to 0.989) — internal inconsistency was
       itself the tell that the "fix" was a new leak, not a correction. One
       genuine SQL bug caught and fixed en route (the same `NULL OR FALSE =
       NULL` three-valued-logic trap already documented for
       `is_confirmed_automation_seed` in Stage 0) — fixing it didn't rescue
       the approach, the missingness leak remained regardless.
     - **Adopted instead: volume-gating** (`VOLUME_GATE_THRESHOLD = 10` on
       `n_comments_sample`) — dilutes any single row to ≤1/10 of the
       aggregate, introduces no new leak. Full gated-vs-original rung-4:
       `admin_removal` 0.896→**0.743** (in range), `self_deletion`
       0.778→**0.695** (in range), `comment_removed_ambiguous`
       0.805→**0.641** (now *below* range — flagged, not explained further),
       `automod_filtered` 0.653→0.725 and `moderator_removed` 0.637→0.687
       (both barely move). The last two barely moving is itself informative,
       not an oversight: their labels are post-level, not comment-level, so
       this specific leak mechanism structurally doesn't apply to them —
       confirmed separately by dropping their 4 post-derived features, which
       also barely moved their AUC (0.653→0.635, 0.637→0.622). Their
       original numbers look comparatively credible as a result.
     - **§8 leakage-register walk, items not already covered:**
       - Item 2: `admin_removal` ∩ `self_deletion` = 39.9% account overlap;
         `self_deletion` ∩ `comment_removed_ambiguous` = 68.6%. The 5-channel
         set is not as independent as the original framing implied —
         self-deletion plausibly often follows other moderation action.
       - Item 3: confirmed by inspection, no `retrieved_on`-derived column
         exists in `account_features_model`.
       - Item 4: a canary test (username-morphology + `account_ordinal`
         only, nothing else) returned AUC 0.489 — pure noise, ruling out
         collection-time-artifact explanations for the age-like features.
       - Item 6: broadened score-adjacent ablation (score family +
         `controversiality_rate` + `is_submitter_rate` + `mean_depth`)
         barely moved `admin_removal` (0.896→0.887) — not the primary
         driver, the row-inclusion mechanism above is.
     - **Still open, not fixed:** `comment_removed_ambiguous`'s gated number
       (0.641) sits *below* the Kumar floor with no explanation yet.
   - ✅ **Stage 4 (pair-level model), first pass done 2026-08-07 — inconclusive
     result, not a clean pass/fail.** `scripts/v3_stage4_pair_model.py` →
     `docs/v3-research/eda/stage4.html` + `stage4_data.json`, linked from the
     EDA/Stage2/Stage3 nav. Designed with a **View A / View B split**
     specifically because Stage 3 needed a same-day leakage correction — the
     label ("same-operator pair") is constructed only from content/identity
     signals (View B), and the primary model is trained only on
     behavioral/structural signals (View A) to predict it, so the headline
     number can't be the model trivially recovering its own labeling rule.
     - **Four real bugs found and fixed in the null model, none assumed —
       verified directly:**
       1. **Global null miscalibrated.** First cut treated every account's
          thread participation as uniform-random across all 45 subs; real
          accounts are heavily subreddit-clustered. Came back 97.4% of all
          9.67M pairs "BH-FDR significant" — not credible on its face.
          Conditioned the null per-subreddit (partial fix, see bug 3).
       2. **Repeat-comment inflation.** 43,141 (author, post_id) pairs have
          more than one comment; the naive self-join counted comment-pairs,
          not thread co-occurrences, breaking the simple-bipartite
          assumption the hypergeometric null needs. Fixed by deduping to
          distinct (author, post_id) presence first — this is the actual
          reason the candidate pool (n_coappear≥2) shrank from the
          previously-estimated 745,307 to the **corrected 275,643** pairs.
       3. **Ascertainment bias, structural, not "fixable" — resolved by
          changing what the number is used for, not by tuning it away.**
          Candidate pairs are only ever pairs that already co-occur (an
          inner join); at median account degree 1–2, co-occurring at all is
          already rare under the null, so p-values over this pre-filtered
          population skew small regardless of whether real coordination
          exists. Verified with an exact null simulator (each account given
          a truly uniform-random thread-subset of its own real size): mean
          p=0.026, median p=0.005 on **pure null-generated data** — real
          data showed mean p=0.030, median p=0.006, statistically
          indistinguishable from the null simulation. **Resolution: B1's
          z-score/p-value is used as a continuous ranking feature only,
          everywhere in this pipeline — no binary significance gate exists,
          and none should be added** (a threshold here would mostly encode
          the ascertainment artefact, not a coordination signal).
       4. **17GB memory blowup** from a Python-side tuple/dict join at
          9.7M-row scale; a single compound 4-way SQL join also took minutes
          under a bad DuckDB query plan. Fixed by staging into explicit
          `CREATE TEMP TABLE` steps — brought the whole join to ~4 seconds.
     - **B1–B8: 6 built, 2 dropped/thin, neither silently approximated.**
       View A (behavioral/structural, used for the primary result): B1
       (co-appearance z, continuous only per bug 3), B2 (co-arrival
       tightness — threshold is this corpus's own p5 gap, 98s, never an
       imported constant, per §4.4's warning), B7 (temporal activity
       correlation, daily bins, z-scored per account). **B6 (reply
       reciprocity) dropped, not approximated** — checked
       `scripts/v3_collect.py` directly: it computes parent_id-based
       reciprocity at collection time but never persists per-comment
       `parent_id` into any stored table (only the derived boolean
       `parent_is_post` survives), so there is no way to reconstruct which
       specific comment a reply targeted after the fact. View B
       (content/identity, used only for label construction): B3
       (text-template near-duplicate rate — extremely sparse, nonzero for
       only 300/275,643 pairs), B4 (char n-gram TF-IDF cosine, capped at
       max_features=1200/min_df=3 for tractability), B5 (registration-cohort
       adjacency via `account_ordinal`). **B8 (shared-domain) thin** —
       18.4% coverage (post-authors only; `self.*` domains excluded as
       trivially-same-subreddit noise, not a real signal).
     - **Label construction — one design flaw caught and fixed mid-build.**
       First cut required text-template overlap (B3) as a mandatory AND-gate
       on top of style/cohort agreement → **zero label positives** (B3 is
       nonzero for only 0.11% of pairs, so gating everything behind it was
       too strict). Corrected to an OR: strong text-template match (3 pairs)
       **OR** (strong stylometric similarity AND strong cohort adjacency,
       both independently top-percentile — 128 pairs) → **131 total
       positives**, 0.048% prevalence. **Spot-checked 12 sampled positives by
       reading the actual comment text side by side — the style+cohort-only
       positives look like a real limitation, not genuine same-operator
       evidence:** e.g. one pair scored B4=0.93 "stylometric similarity"
       while one account discusses corporate bribery/buybacks and the other
       discusses BMC/asteroid excavation — topically unrelated. Root cause:
       **B4's char n-gram cosine has weak absolute discrimination on
       Reddit-comment-length text** — mean 0.76 across *all* 275,643
       candidate pairs, i.e. compressed near ceiling, usable only as a
       relative rank, not as evidence of genuine stylistic match at the
       thresholds used. This directly degrades the label the primary model
       is trained to predict.
     - **Results, account-disjoint 70/30 split** (117,200/275,643 pairs
       dropped for straddling the train/test boundary — a real, necessary
       cost of disjointness, not a bug):

       | model | AUC | avg precision | recall@1%FPR | n_pos (test) |
       |---|---|---|---|---|
       | **View A only (primary, non-circular)** | **0.616** | 0.00034 | 0.0% | 6 |
       | View B only (reverse check, more circular) | 0.767 | 0.026 | 3.1% | 224 |
       | Combined (most circular — context only) | 1.000 | 1.000 | 100% | 6 |

       Mandatory baseline (raw co-appearance count alone): **0.653**.
       Permutation floor: 0.522.
     - ⚠️ **Honest bottom line: the primary, non-circular View-A result
       (0.616) sits *below* its own mandatory baseline (0.653).** Behavioral
       /structural features alone do not detect this label better than
       simply counting shared threads. The Combined AUC=1.000 is exactly the
       circularity the View-split design exists to expose (the model sees
       the same B3/B4/B5 values used to build the label) and is correctly
       **not** reported as a finding. Given the label-quality issue above
       and only 6 test positives for the primary model, the fairest reading
       is that **this test is underpowered and its label is noisier than
       intended — it does not replicate Kumar's 0.91 pair-level AUC, but it
       also does not test that claim rigorously enough to count as
       falsifying it.** The plan's 0.90+ pair-level claim should be treated
       as **still untested**, not as tested-and-failed.
     - **Not yet evaluated, options for a second pass:** lowering the
       n_coappear≥2 floor to recover some of the 8.9M pairs with exactly one
       co-appearance (currently excluded entirely); finding a stronger or
       external same-operator identity signal than View-B percentile
       thresholds; reporting confidence intervals on a small-n label
       honestly rather than a point AUC; revisiting B4's feature
       resolution (e.g. higher-order n-grams, a stronger stylometric
       embedding) given its near-ceiling compression at comment length.
   - ✅ **Stage 3b: post-context feature iteration, done 2026-08-07 —
     user-directed pivot after Stage 4** ("drop this test and focus on
     feature engineering, feature selection, sanitisation, and multivariate
     or xgb model → feature importance → learn on train-test → iterate").
     New file `scripts/v3_stage3b_feature_iteration.py`; new table
     `account_post_context` in `v3.duckdb` (346,940/347,886 accounts, 99.7%
     coverage); `account_features_model` and `v3_feature_sanitise.py` left
     untouched (additive, not destructive).
     - **Feature engineering.** `posts_clean` already had ~33 computed
       post-level columns (§4.2) that had never been joined up to account
       level. Built mean- (and max-, for the two features that mattered
       most) aggregates of each account's own threads:
       `contested_share`, `comment_score_gini`, `reply_reciprocity`,
       `removed_comment_rate`, `tombstone_rate`, `bot_comment_rate`,
       `submitter_reply_rate`, `upvote_ratio`, `pct_toplevel`, `mean_depth`,
       `num_crossposts`, `log_subscribers`, `n_unique_commenters`,
       `n_comments_observed`, `is_self_rate`, `over18_rate` — prefixed
       `pc_*`. This is a genuinely new signal family: not what an account
       does, but what kind of threads it tends to show up in.
     - **Leakage check, same class of bug already found once in Stage
       3 — checked directly, not assumed safe.** Several `pc_*` features
       are computed from the pool of comments on a post, which includes the
       account's own comment; for an account active on few distinct posts,
       "average removal rate of my threads" can partly re-encode "was my
       own comment removed," the same mechanism (row-inclusion in an
       aggregate) that produced `admin_removal`'s fake 0.896. **Gated on
       `n_distinct_posts_ctx ≥ 5`** (chosen for tractability, not swept
       across settings the way Stage 3's threshold=10 was — a
       lighter-weight check than the original audit, flagged as such).
       **Verified the gain survives a stricter combined gate**
       (`n_comments_sample≥10 AND n_distinct_posts_ctx≥10`):
       `comment_removed_ambiguous` moved 0.641→0.770 under the original
       test-set numbers and 0.644→0.772 under the stricter gate — 
       essentially identical, so the gain is not a gating artefact.
     - **Results, volume-gated rung-4 AUC, base vs. expanded, all 5
       channels:**

       | channel | base | expanded | Δ |
       |---|---|---|---|
       | `admin_removal` | 0.743 | 0.768 | +0.025 |
       | `self_deletion` | 0.695 | 0.755 | +0.060 |
       | `comment_removed_ambiguous` | 0.641 | 0.774 | +0.133 |
       | `automod_filtered` | 0.725 | 0.798 | +0.073 |
       | `moderator_removed` | 0.687 | 0.712 | +0.025 |

       *(Corrected 2026-08-07 — the original run of this script had a
       feature-set bug that silently depressed its own "base" numbers; see
       the variance-root-cause entry below. The gains are real and
       slightly larger than first reported.)*

       All 5 now land inside 0.65–0.80 — `comment_removed_ambiguous`
       specifically moves from *below* the Kumar floor (post-correction) to
       back inside it, via a real added feature rather than by loosening
       the leakage fix that put it below in the first place.
     - **SHAP: `pc_removed_comment_rate`** (mean removal rate of an
       account's own threads — "hangs around threads with heavy removal
       activity") **is the standout new feature** — top feature for
       `self_deletion` (0.47) and for `comment_removed_ambiguous` (0.68, 2.5×
       the next feature). This is a distinct behavioral signal from "my own
       content gets removed" (already captured, and excluded per the
       leakage register, via `removal_rate`). `pc_tombstone_rate`/
       `pc_bot_comment_rate`/`pc_num_crossposts`/`pc_log_subscribers`
       dominate `moderator_removed`'s top 10 (7 of 10 features).
     - **One iteration cycle (Phase E of the brief):** added max-version
       companions (`pc_removed_comment_rate_max`, `pc_tombstone_rate_max`)
       for the two most-affected channels. Small further lift
       (`self_deletion` +0.008, `comment_removed_ambiguous` +0.005) —
       real but marginal, diminishing returns past the mean version.
     - **Two loose ends, flagged not fixed** (a third, the base-number
       discrepancy, was root-caused and fixed the same day — see below):
       1. ~~`shap_family_importance` mis-buckets `pc_*` under "other"~~ —
          **fixed** during the Tier-1 build (`family_fn` override
          parameter added), see the Tier-1 log entry.
       2. Results page `docs/v3-research/eda/stage3b.html` was never
          rendered — `stage3b_data.json` has the data (now regenerated with
          corrected numbers), no render function was written, deprioritized
          in favor of finishing the actual iteration within the available
          time.
   - ✅ **Feature-potential deep dive, done 2026-08-07 — research only, no
     code, no plan-doc edits from the fork itself.** User-directed
     ("i need you to add more to the potential — create an agent to think
     deeply"), cross-referencing `V3_METRIC_CATALOGUE.md` and
     `V3_FEATURE_PLAN.md` against the live schema. **One correction to the
     metric catalogue surfaced in the process:** it marks "response latency"
     (comment↔parent `created_utc` delta) as ✅ buildable on "parent_id
     chains 100% populated" — contradicted by this session's own Stage 4
     finding that per-comment `parent_id` is never persisted to any stored
     table (only the derived `parent_is_post` boolean survives). The
     catalogue entry is stale, or refers to raw collected files never
     checked against the DB — **do not build on it without verifying against
     the raw `.ndjson.zst` cells first.**
     - **Tier 1 — build next** (cheap, grounded in already-collected data,
       low leakage risk): own-history exact-repeat-comment rate (1,520
       accounts already verified to qualify — the much-better-powered
       within-account analogue of Stage 4's cross-account template-sharing,
       which found only 0.11% of pairs had any overlap); comment:post count
       ratio (one division, catalogue flags it as missing and currently
       conflated with the karma ratio already in the table);
       "vs. population mode" family (5 metrics expressed against the modal
       value of an account's tier/sub-month reference group rather than raw
       — catalogue calls this "genuinely new"); link/URL density (88,932
       comments, 5.5%, verified to contain a URL — carries none of the
       row-inclusion leakage risk already fixed twice this session, since
       it's not derived from any label channel); outsider-influx share
       (post-context: share of an account's threads' commenters who are
       new-to-that-sub); title:body length ratio + score-per-word
       (post-context — 65,553/127,961 posts have zero selftext, a
       link/reaction-bait signature the catalogue calls out directly).
     - **Tier 2 — worth trying**, more cost: sentiment/toxicity via VADER
       (flagged as completely absent currently; explicit caveat that VADER
       degrades on Hinglish/code-mixed text — check a sample before
       trusting it uncalibrated); sub-month regime as an actual account
       column, not just Stage 2's prose finding (spike-exposed accounts
       measured *lower* risk, 5.3% vs 6.9%); lightweight co-appearance
       degree/concentration reusing Stage 4's infrastructure rolled up to
       per-account summaries — sidesteps the label-construction problem
       that sank Stage 4, since it needs no same-operator label, just
       description; post-author edit rate (`meta_is_edited`, only 16% of
       accounts have a post at all, compounding with the already-thin
       automod/moderator channels).
     - **Tier 3 — interesting, low-confidence:** domain concentration
       (likely thin, post-author-only); within-thread activity Gini rolled
       to account level (check correlation against already-built
       `pc_n_unique_commenters`/`pc_contested_share` before building, likely
       redundant); flair diversity (94.2% coverage but high-cardinality, no
       clear a priori encoding); hand-built interaction/residual features
       (e.g. `removal_rate` × `pc_bot_comment_rate`) — only worth building
       with a stated mechanism in advance per §7's own rule, not as a
       blanket sweep, since XGBoost finds most simple interactions on its
       own and a sweep just adds multiple-testing cost.
     - **Tier 4 — not worth it, confirmed dead on inspection, don't
       re-attempt:** response latency (parent_id not persisted, see
       correction above); distinguished/flair interaction
       (`distinguished` is 1,616,021 `None` / 2 `admin` in
       `commenters_clean` — moderator-distinguished accounts were already
       excluded as automation seeds at Stage 0, no signal left to mine);
       cross-account stylometric refinement (already attempted in Stage 4,
       confirmed too sparse/compressed-near-ceiling at Reddit-comment
       length — would need embeddings instead of char n-grams to revisit,
       a much bigger lift than anything else on this list); true score
       trajectory (structurally impossible — only 2 snapshots exist,
       T+16s/T+36h, confirmed independently by both the metric catalogue
       and §3); moderator count / client-app source (confirmed absent from
       every endpoint).
   - ⚠️ **Tier 1 feature build, done 2026-08-07 — a wash, not a win, plus a
     recurring-variance flag.** New tables `account_tier1_repeat_url` /
     `account_tier1_post_context`; new script
     `scripts/v3_stage3c_tier1_features.py`; `account_features_model` and
     `account_post_context` left untouched (additive).
     - **The 6 features, as built, with deviations noted:**
       1. `own_repeat_rate`/`has_own_repeat` — exact-duplicate-body rate
          among an account's own comments (`body_len>10`). **1,238 accounts
          qualify, not the 1,520 estimated during ideation** — the gap is
          the `body_len>10` filter excluding short duplicates the original
          count didn't screen out. Minor, not investigated further.
       2. `comment_post_ratio` (+ `has_posts` hurdle) — only defined where
          `n_posts_sample>0`, as specified.
       3. `vs_mode_{karma,comments,posts,score}` — 4 metrics (reply latency
          dropped, confirmed unbuildable) against the histogram-modal value
          of the account's plurality incentive tier. **One forced
          substitution:** `vs_mode_score` was specified against
          `mean_comment_score`, which turned out to already be VIF-pruned
          out of `account_features_model` (Stage 3's own leakage audit)  —
          substituted `worst_sub_mean_score`, the closest surviving
          score-per-comment proxy.
       4. `url_rate` (+ hurdle) — regex URL detection on comment body,
          12% of accounts (41,516) nonzero.
       5. `outsider_influx_share` — post-context, per-(author,sub)
          first-appearance share among a post's commenters.
       6. `title_body_ratio` / `score_per_word` — post-context, from
          `posts_clean`.
     - **Leakage/gating on the two post-derived features (5/6):** reused
       `account_post_context`'s existing `n_distinct_posts_ctx ≥ 5` gate
       rather than inventing a second threshold — both NaN below it. Items
       1/2/4 carry no comparable row-inclusion risk (not post-derived, not
       removal-derived) and ride the existing whole-matrix
       `n_comments_sample≥10` gate.
     - **SHAP family-rollup bug fixed** (see the START HERE headline above
       for detail) — confirmed working: `automod_filtered`'s rollup shows
       `post_context=1.78`, `tier1=0.46` as distinct buckets rather than
       folded into "other."
     - **Results, gated rung-4, this run's own freshly-recomputed base vs.
       +Tier1 vs. the one-iteration trim:**

       | channel | base (this run) | +Tier1 | trimmed (iteration) |
       |---|---|---|---|
       | `admin_removal` | 0.767 | 0.756 | 0.766 |
       | `self_deletion` | 0.754 | 0.752 | 0.754 |
       | `comment_removed_ambiguous` | 0.772 | 0.775 | 0.771 |
       | `automod_filtered` | 0.801 | 0.793 | 0.787 |
       | `moderator_removed` | 0.711 | 0.721 | 0.702 |

       "This run's base" differs from Stage 3b's *originally-cited* numbers
       (0.764/0.750/0.772/0.789/0.720) by 0.001–0.012 — this looked like a
       third occurrence of unexplained rerun variance at the time, but the
       root-cause entry below found the actual explanation: Stage 3b's own
       reported "base" was wrong (a feature-set bug, not noise), and
       Stage 3c's independently-built base was correct all along. See below
       — this run's base is not itself in question.
     - **Iteration cycle:** dropped the two SHAP-inert features
       (`own_repeat_rate`, `url_rate`) on the hypothesis that they were
       diluting the model, and reran. **Hypothesis not supported** —
       `admin_removal`/`self_deletion`/`comment_removed_ambiguous` stayed
       flat, but `automod_filtered` (0.793→0.787) and `moderator_removed`
       (0.721→0.702) got *worse*, ending below even the base number.
       Likeliest explanation: small-n CV instability on the
       post-author-only gated population (11,438 accounts) rather than
       those two features being actively harmful — plausible, not
       confirmed.
     - **Verdict, stated plainly, and re-confirmed after the variance
       root-cause below: real, not noise, and still a wash.** Tier 1 as a
       bundle is not a repeat of Stage 3b's clean win. Some individual
       features look real by SHAP (`score_per_word` top-1 for
       `admin_removal`; `vs_mode_comments`/`vs_mode_posts` top-6 in three
       channels); two look inert (`own_repeat_rate`, `url_rate`). The
       true noise floor turned out to be ~0.001–0.003 (verified via
       bit-identical reruns once the Stage 3b bug was fixed — see below),
       not the ~0.01 feared at the time this run was reported, and this
       run's own base/expanded/trimmed comparison was never affected by
       that bug in the first place (built independently, happened to be
       correct). So Tier 1's up-to-±0.011 deltas are more likely small
       real effects than noise — "wash, not a win" is a genuine reading,
       not a noise-confounded one.
     - **Not done:** no HTML page rendered for `stage3c_data.json` (same
       gap Stage 3b left open, itself still unrendered); the "other" SHAP
       bucket is still non-trivially sized (e.g. 0.59 for
       `comment_removed_ambiguous`) and not fully attributed to a family.
   - ✅ **AUC rerun-variance root-caused and fixed, done 2026-08-07 —
     user-directed** ("Proceed" — in response to the recurring-variance flag
     raised after Tier 1). **A real, reproducible code bug, not sampling
     noise or model non-determinism** — both of those were tested and ruled
     out empirically before finding the actual cause: DuckDB row order from
     an unordered `SELECT * FROM account_features_model` was verified
     stable across repeated calls and fresh process invocations; XGBoost
     (`tree_method='hist'`, fixed `random_state=42`) was verified to
     reproduce bit-identical AUC across separate process runs on identical
     data.
     - **Actual bug**, in `scripts/v3_stage3b_feature_iteration.py`'s
       `run_channel_comparison()`: `base_feats` was built with
       `not c.endswith('_nonzero')`, a blanket filter meant to strip only
       the *new* post-context hurdle columns. `account_features_model`
       already carries ~15 pre-existing `_nonzero`-suffixed hurdle columns
       from Stage 3's original 62-feature set (e.g.
       `high_tier_share_nonzero`) — the blanket filter silently dropped
       those too, so "base" was never actually Stage 3's true 62-column
       set, despite being reported as a same-feature-set comparison. A
       second, compounding bug: two `pc_*_max` columns added during Stage
       3b's own Phase E iteration were never added to `POST_CTX_COLS`, so
       they leaked into "base" as if pre-existing.
     - **Fix:** replaced the blanket suffix filter with an explicit
       `POST_CTX_NONZERO_COLS` list (the 6 genuinely-new hurdle columns
       only) and added the missing `_max` columns to `POST_CTX_COLS`.
       Verified `base_feats` now matches Stage 3's original feature set
       exactly, 62/62, zero diff.
     - **Reproducibility verified, not assumed:** ran the fixed script
       twice consecutively — bit-identical AUC across all channels, all
       rungs, zero field diffs (`admin_removal` gated rung4 both runs:
       `0.7431769654419119`). Determinism is total once the feature-set bug
       is gone — confirming the *true* base numbers are exactly Stage 3's
       original leakage-audit numbers (0.743/0.695/0.641/0.725/0.687), and
       Stage 3b's real expanded numbers are the corrected, slightly higher
       ones now recorded above.
     - **`scripts/v3_stage3c_tier1_features.py` was not affected** — its
       `base_cols` was built independently of 3b's buggy function and, by
       luck, was already correct (confirmed within 0.001–0.003 of the
       newly-fixed 3b numbers, a residual small enough to plausibly be
       feature-selection tie-breaking rather than a further bug — not
       chased to zero, flagged not fixed). This is why Tier 1's own
       base-vs-expanded comparison (§10.4 above) was never corrupted by
       this bug and its verdict stands unchanged.
     - Files: `scripts/v3_stage3b_feature_iteration.py` (bug fix +
       docstring explaining it), `docs/v3-research/eda/stage3b_data.json`
       (regenerated with corrected numbers). `v3_stage3c_tier1_features.py`
       not modified.
   - ⚠️ **Tier 2 feature build, done 2026-08-07 — clean wash, one leak
     caught and fixed before shipping.** New script
     `scripts/v3_stage3d_tier2_features.py`; new tables
     `account_tier2_regime`, `account_tier2_coappear`, `account_tier2_edit`
     in `v3.duckdb`. `V3_PLAN.md` not touched by the fork; nothing
     committed.
     - **Sentiment/toxicity: skipped, not built**, after a required
       pre-check (25-comment VADER spot-check against real corpus text)
       found the corpus heavily Hinglish/code-mixed and URL-heavy, both of
       which VADER flattens to 0.00 — indistinguishable from genuine
       neutrality. Correctly not shipped.
     - **`post_edit_rate` (posts-only, `meta_is_edited`) was a real leak,
       caught mid-build, same pattern as `admin_removal`'s original fake
       0.896:** first run put `automod_filtered` at gated rung4 **0.880**
       (above the Kumar ceiling), with `post_edit_rate` dominating SHAP
       (0.797, 2× the next feature). Direct check: edit rate by
       `meta_removal_type` — None 6.2%, `automod_filtered` 47.3%, `reddit`
       (admin) 60.2%. People edit posts *to fix or appeal a removal*, not
       independently of it — leakage register item 1 (§8), missed by the
       fork's first-pass leakage recheck (which had only checked population
       thinness, not label-derivation — a reminder that "checked for
       leakage" needs to mean checked against every register item, not
       just the ones that come to mind first). Hard-excluded
       `post_edit_rate`/`post_edit_rate_nonzero`, reran, **verified via
       bit-identical base-AUC reproduction across both runs**
       (`admin_removal` base rung4 both runs: `0.7659704492814655`).
     - **Sub-month regime exposure** (`sub_month_spike_share`) —
       operationalizes Stage 2's prose finding (spike-exposed accounts
       measure lower-risk) as an actual account-level column for the first
       time. **Co-appearance degree/concentration**
       (`coappear_degree`/`coappear_hhi`) — reuses Stage 4's co-appearance
       infrastructure rolled up to a per-account summary, deliberately
       sidestepping the same-operator label-construction problem that made
       Stage 4 inconclusive (this needs no label, just description). Both
       directly rechecked against `removal_rate` post-hoc and confirmed
       non-leaky (near-zero correlation).
     - **Results, gated rung4, base→expanded** (regime + co-appearance
       only, edit rate excluded):

       | channel | base | expanded | Δ |
       |---|---|---|---|
       | `admin_removal` | 0.766 | 0.772 | +0.006 |
       | `self_deletion` | 0.754 | 0.760 | +0.006 |
       | `comment_removed_ambiguous` | 0.774 | 0.772 | −0.002 |
       | `automod_filtered` | 0.793 | 0.799 | +0.006 |
       | `moderator_removed` | 0.718 | 0.718 | ~0.000 |

       All 5 stay inside 0.65–0.80. Small, mixed-but-mostly-positive, no
       standout feature — correctly not forced into an iteration cycle
       (the fork's own judgment call: nothing in this run's SHAP results
       justified one, so it reported instead of manufacturing a cycle).
     - **Not done:** no results page rendered for `stage3d_data.json` (now
       the third page in this state, alongside `stage3b`/`stage3c`); the
       "other" SHAP bucket still not investigated.
     - **Three-lap pattern now visible:** Stage 3b (real win, +0.025 to
       +0.133) → Tier 1 (clean wash) → Tier 2 (clean wash, but caught a
       real leak). Diminishing returns from further hand-engineered
       account-level features look real, not assumed — the cheap,
       high-value move (joining already-computed post-level data no one
       had aggregated yet) has been made; what's left is smaller and more
       mixed.
   - ✅ **Tier 3 feature build, done 2026-08-07 — clean null result.** New
     script `scripts/v3_stage3e_tier3_features.py`; new table
     `account_tier3_domain`. `V3_PLAN.md` not touched by the fork; nothing
     committed.
     - **Built:** `domain_hhi` (post-author population, 45,861 accounts,
       Herfindahl over `posts_clean.domain`) — caveated on inspection as
       largely redundant with `subreddit_entropy`, since domain is
       dominated by Reddit's own media hosts or `self.<subreddit>`. Exactly
       two stated-mechanism interaction terms, per §7's rule against
       blanket sweeps: `karma_extremeness_x_reception_spread`
       ("narrow-but-polarizing" signature) and
       `bot_rate_x_coappear_degree` ("broad reach specifically into
       low-quality territory, not just broad reach"). A third candidate,
       `removal_rate × n_comments_sample`, was considered and explicitly
       **rejected** — it would have laundered a hard-excluded
       removal-derived feature back into the model through a side door.
     - **Rejected before a full pipeline build — the check itself was the
       useful output, not a formality:** within-thread activity Gini
       (correlation against `contested_share`/`n_unique_commenters` genuinely
       low, 0.047/−0.050, so not redundant by that measure — but the
       distribution is near-degenerate, median 0.0, p75 0.058, max 0.17,
       and substantively re-derives `repeat_engagement_rate` already in the
       table); flair diversity (1,406 distinct free-text values with heavy
       per-subreddit fragmentation of identical categories —
       `Discussion`/`Discuss`/`Discussions`/`#Discussion 💬` all appear in
       one top-20 alone — no defensible entropy encoding without semantic
       clustering, out of scope here).
     - **Leakage check:** all 3 built features correlate <0.033 (absolute)
       with `removal_rate` — clean.
     - ⚠️ **Same class of leak caught a third time in this session.** First
       run put `admin_removal`'s base AUC at 0.800 — at/above the Kumar
       ceiling, the exact shape of both prior leaks. Cause:
       `post_edit_rate`'s *non-hurdle* column had never actually made it
       into the exclusion list assembled after Tier 2 — only its
       `_nonzero` companion had been added. Fixed (added
       `TIER2_LEAKAGE_EXCLUDE` to the exclusion set), reran, **verified via
       bit-identical reproduction across two runs.**
     - **Result, gated rung4, base→Tier3-expanded, all 5 channels land
       inside the established ±0.003 noise floor:** `admin_removal`
       0.7700→0.7661 (−0.004), `self_deletion` 0.7556→0.7572 (+0.002),
       `comment_removed_ambiguous` 0.7728→0.7732 (+0.0004),
       `automod_filtered` 0.7853→0.7854 (+0.0001), `moderator_removed`
       0.7183→0.7198 (+0.001). None of the 3 built features place in any
       channel's SHAP top-5. **Verdict: null, not noise** — the
       "low-confidence" label this tier carried at ideation time held up
       under the same rigor as the tiers that didn't.
   - ✅ **Cross-sample boundary discovery + conjunctive rule, done
     2026-08-07 — user-directed, methodologically distinct from Stage
     3's supervised approach.** New script `scripts/v3_boundary_discovery.py`,
     new output `docs/v3-research/eda/boundary_discovery_data.json`.
     `V3_PLAN.md` not touched by the fork; nothing committed.
     - **Rationale (user's own words, paraphrased):** rather than trust
       Reddit's moderation actions as ground truth for bot behavior — which
       Stage 3 already showed don't agree with each other — find behavioral
       structure that **replicates across independent samples on its own
       terms**, build a rule from what replicates, and check moderation
       actions only afterward as an outside plausibility read, never as a
       tuning signal.
     - **Split:** 3-way, stratified by (tier bucket × `account_ordinal`
       decile) — not naive-random. Verified balanced: 115,950 / 115,950 /
       115,986 accounts, matching on tier-share/`account_ordinal`/
       `days_since_first_seen` means to within noise. Part 3 held out
       entirely until the final validation step.
     - **Candidate pool:** 64 continuous/count features across
       `account_features_model` + `account_post_context` + all
       Tier1/2/3 tables, joined on `author`. `removal_rate`,
       `deleted_later_rate`, `post_edit_rate` (and every `_nonzero`/
       `_pctl`/`_max` variant) and `botmarker_composite` **excluded
       outright** — not leakage-checked, excluded on principle, since
       these are exactly the external-label signal being avoided.
       `pc_removed_comment_rate`/`pc_tombstone_rate`/`pc_bot_comment_rate`
       kept as thread-context signals (describe the threads an account
       hangs around, not the account's own moderation record) — a
       judgment call, flagged as second-guessable.
     - **Replication screen**, Stage 1's corrected bimodality method
       (point-mass stripping, signed-log transform, real KDE-valley check)
       run independently on Part 1 and Part 2: **9 of 64 candidates
       replicated** with consistent boundary locations in both —
       `low_tier_share`, `subreddit_entropy`, `reception_spread`,
       `account_ordinal`, `n_own_posts_with_comments`, `karma_extremeness`,
       `outsider_influx_share`, `coappear_hhi`, `domain_hhi`. Robustness
       across 5 month/tier slices: 6/9 held in all 5, `account_ordinal` and
       `n_own_posts_with_comments` held in 4/5, `outsider_influx_share`
       (weakest) only 3/5.
     - **Incremental AND-rule, ordered by replication strength — the
       collapse is the finding, not a bug:** flagged share of population
       shrinks 31% → 4.5% → 2.4% → 0.53% → 0.46% across the first 5
       conditions, then **hits exactly 0% at the 6th condition**
       (`coappear_hhi`) and stays at 0 for the remaining 3. The literal
       "require all 9 indicators" reading flags nobody in either sample —
       reported plainly rather than silently swapped for a softer rule.
     - **Chosen cutoff: top 5 conditions (the last point before
       collapse), thresholds fixed from Parts 1+2, applied unchanged to
       the untouched Part 3:** stable ~0.46–0.53% of population in every
       part (Part1=550, Part2=510, Part3=530 accounts) — a real,
       replicating, held-out-validated result, not an artifact of one
       sample.
     - ⚠️ **Plausibility check, done strictly after the rule was
       finalized, not used to tune it: flagged accounts show *lower*
       `removal_rate` than the population (0.0583 vs 0.0696, 0.84×), not
       higher; `deleted_later_rate` roughly flat (0.96×).** This
       behavior-only construction does not point toward what Reddit's own
       moderation flags. Two live readings, **not adjudicated**: (a) a
       real inauthentic pattern moderation systematically misses — the
       whole premise for not trusting removal-as-ground-truth in the
       first place; or (b) a real but different, non-bot phenomenon
       (working guess: narrow-focus, prolific post-authoring power users).
       **Needs a human look at actual flagged accounts before going
       further** — same standard as every other claim in this plan (§1).
   - ✅ **Multivariate KDE follow-up, done 2026-08-07 — user-directed**
     ("bimodal was just a basic suggestion... use KDE and make group
     boundaries in higher dimensions... label them with our suspected bot
     behaviour... see the diffs using PCAs"). New script
     `scripts/v3_multivariate_kde.py`; outputs
     `docs/v3-research/eda/multivariate_kde_data.json` (6.5MB — loadings,
     clusters, enrichment, scatter/KDE-grid data) plus two static PNGs.
     Same 3-way split and same removal-derived exclusion-from-construction
     discipline as the univariate round, carried forward unchanged.
     - **PCA: this behavior space is genuinely diffuse, not a few
       dominant axes.** 12 components (the practical cap for tractable
       KDE at this sample size) reach only 52.4% cumulative variance.
       Named top loadings: PC1 (13.5%) = activity volume/footprint breadth
       (`coappear_degree`, `n_comments_sample`, `subreddit_entropy`); PC2 =
       thread-size/tier visibility; PC3 = post-authoring/self-engagement;
       PC4 = reception quality; PC5 = tier composition/thread health.
     - **Clustering: methods disagree on structure, reported as a real
       disagreement, not smoothed over.** Mean-shift (KDE-mode climbing,
       the literal ask) found 28 (Part1) / 32 (Part2) modes — one dominant
       core (~97% of accounts) plus a long tail of small satellites, several
       n<10. HDBSCAN on the identical space found only 2 clusters and
       called 92.5% of points noise. The satellite-cluster boundaries are
       mean-shift-bandwidth-sensitive — not settled structure.
     - **Replication is mixed, and that nuance matters:** the core cluster
       is extremely stable (Part1↔Part2 centroid distance 0.049; 97.15% of
       held-out Part3 lands in it). The *exact* small satellite clusters do
       **not** match by position between Part1 and Part2 (centroid
       distances 5–25 in PC-space) — but **the phenomenon replicates even
       though the exact partition doesn't**: independently, Part1's
       strongest removal-enriched cluster hit **12.46× baseline** (n=57);
       Part2's strongest hit **9.19× baseline** (n=650) — both samples
       independently produced *some* small, strongly removal-enriched
       subpopulation from behavior-only clustering. Whether Part1's and
       Part2's specific enriched clusters share the same underlying
       original-feature profile (not just coincidentally both
       removal-heavy) is **unverified — flagged as the natural next
       check, not run yet.**
     - **Cross-method triangulation — the most concrete finding.** The
       *previous* round's ~550-account AND-rule group (built by a
       completely different, univariate methodology) is **18× concentrated**
       in this run's cluster 15 relative to that cluster's 0.96% population
       base rate (17.45% of the AND-rule group lands in cluster 15 vs.
       0.96% overall). Two independent, methodologically distinct
       approaches substantially agree on part of the same region — real
       triangulation, though **74% of the AND-rule group still falls in
       the undifferentiated core**, so this is partial overlap, not full
       agreement. Plausible reading: the AND-rule group's earlier
       "removal_rate 0.84× baseline, not bot-like" verdict may have
       been **diluted by pooling** — a genuinely enriched ~17% sub-slice
       (the part landing in cluster 15) may be getting averaged against a
       majority that isn't, rather than the whole group uniformly not
       being bot-like. Not confirmed, worth checking directly.
     - **Bug caught and fixed before reporting:** first pass re-derived the
       previous round's 550-account group using the wrong direction logic
       for 3 of 5 AND-rule features (ignored `direction_hint`, defaulted to
       minority-side for all), producing 768 instead of 550. Fixed,
       reran, confirmed exact match before using it in the overlap check
       above.
     - **Plain verdict (fork's own words):** messier than the univariate
       round in structure (methods disagree on cluster count, small
       clusters don't replicate exactly by position) but more encouraging
       on substance — found subpopulations with genuinely strong (8–12×),
       independently-replicating removal-rate enrichment, which the
       univariate AND-rule's chosen cutoff did not show in aggregate
       (0.84×). Not yet confirmed as *the same* group across samples —
       needs the original-feature-profile match-up next before any
       specific cluster gets treated as a validated finding rather than a
       promising lead.
5. **Rally + GDELT conditioning.** Not yet started.
6. *(Optional)* base36 age calibration + t+90d suspension check on the top-risk
   decile, via `scripts/reddit_auth.py`. Not yet started.
7. **Dashboard**, leading with the traceable claim: "these N accounts, these M
   threads — open them." Not yet started.

---

## 11. Monthly tracker operations — two-tier collection

The historical build (§10.1–10.3) is a one-time pass over 2023-07 → now. Going
forward, the deliverable is a **monthly tracker**, which is a different problem:
each month, for each sub, decide whether Arctic Shift has actually indexed that
month yet — not whether the sub has history — and fall back if not.

### Tier 1 (primary) → Tier 2 (fallback)

- **Tier 1 — Arctic Shift**, same pipeline, same schema, near-zero marginal
  cost. Used whenever the month's data for a sub is present and looks complete.
- **Tier 2 — live Reddit via `scripts/reddit_auth.py` (OAuth already held)**,
  triggered per-sub-month when Tier 1 fails its health check. Reduced sample —
  `/r/{sub}/top.json?t=month` for posts, `/comments.json` for the top/first
  commenters — same downstream schema, degraded coverage rather than a gap in
  the tracker.

### The health check, calibrated 2026-08-05 against real data

A naive "is latest month ≥ some fraction of the trailing-6-month average" check
was tried first and **produced a false positive**: `r/ipl` flagged as a
"comment cliff" (4,174 vs a 6-month trailing average of 118,014) purely because
IPL is a cricket league and its 6-month trailing window happened to span the
season peak. ✅ Confirmed with the full 24-month series: comments run
**260K–343K/month March–May**, **~10–30K the rest of the year**, identically in
2025 and 2026. **A trailing-average check will false-flag every seasonal sub
every off-season, forever, if not fixed.**

**Correct check, two conditions, both required to flag Tier-1 failure:**
1. Latest month's post count is near zero (< ~5% of that **same calendar
   month, prior year** — a YoY same-month baseline, not a trailing average).
2. The zero persists — a direct `/posts/search?subreddit=X&sort=desc&limit=5`
   probe shows no posts within the last N days, independent of the aggregated
   `time_series` index (this is what caught `DesiVideoMemes` below; it's also
   the cross-check that would have caught `unitedstatesofindia`'s indexing gap
   without needing to know about it in advance).

### First run of the check — findings

Run against all 45 subs, 12 complete months (2025-08 → 2026-07):

| Finding | Detail |
|---|---|
| **44/45 actively indexed through 2026-07** | Tier 1 sufficient for the large majority every month |
| **`r/unitedstatesofindia`** | Confirmed genuinely active — Tier-2-style direct search shows **4,253 posts / 26,946 comments** in the latest month alone — but `time_series` (the cheap aggregate path) has **zero coverage at any precision**. This sub needs the direct-search count path **permanently**, not as a fallback; budget it accordingly, it is not a small sub. |
| **`r/DesiVideoMemes`** | **Confirmed genuinely dead** — public, unquarantined, subscriber count still shown (480K), but zero posts since **2026-03-19**, verified by direct search agreeing with `time_series`. Real signal, not an indexing artefact. **Exclude from active monthly collection**; retain in the registry for its historical months only, and re-probe monthly in case it revives. |
| **`r/ipl`** | Seasonal, not stale (see above). Included as a live worked example of why the YoY check exists. |

**Net for the tracker registry:** 43 subs on the standard Tier-1 path, 1
(`unitedstatesofindia`) permanently on the direct-search path, 1
(`DesiVideoMemes`) suspended from active collection pending revival. All
recorded in `subreddits_v3.csv`.

**Operational note:** this health check must run *before* each month's
collection, not after — it's what decides which tier to use, not a post-hoc
audit. Implement it as one function shared between the monthly job and any
future manual re-check, not a one-off script (`tracker_freshness.py` in
`scripts/` is the reference implementation from this run).

---

## Sources

**Account-level detection**
- [TROLLMAGNIFIER: Detecting State-Sponsored Troll Accounts on Reddit](https://arxiv.org/abs/2112.00443)
- [TwiBot-22: Towards Graph-Based Twitter Bot Detection](https://arxiv.org/abs/2206.04564)
- [BLOC: A general language framework for modeling online behavior](https://arxiv.org/abs/2211.00639)
- [Pozzana & Ferrara — Measuring bot and human behavioral dynamics](https://arxiv.org/abs/1802.04286)
- [Kim & Jo — finite-size correction to burstiness](https://arxiv.org/abs/1907.04166)
- [Machine learning-based social media bot detection: literature review](https://link.springer.com/article/10.1007/s13278-022-01020-5)
- [La Cava et al. — LLM-generated content engagement on Reddit (2025)](https://arxiv.org/abs/2503.13905)

**Coordination and sockpuppetry**
- [Kumar et al. — An Army of Me: Sockpuppets in Online Discussion Communities (WWW 2017)](https://arxiv.org/abs/1703.07355)
- [Kumar et al. — Community Interaction and Conflict on the Web (WWW 2018)](https://arxiv.org/abs/1803.03697)
- [CooRnet — coordinated link sharing behaviour](https://coornet.org/) · [A-B-C framework](https://coornet.org/abc.html)
- [CooRTweet — generalised coordinated-action detection](https://github.com/nicolarighetti/CooRTweet)
- [Tumminello et al. — Statistically Validated Networks](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0017994)
- [Saracco et al. — Inferring monopartite projections of bipartite networks (BiCM)](https://arxiv.org/abs/1607.02481)
- [Neal — `backbone` R package (SDSM/FDSM)](https://cran.r-project.org/package=backbone)
- [Schoch et al. — Coordination patterns reveal online political astroturfing](https://www.nature.com/articles/s41467-022-35576-9)
- [Luceri et al. — Unmasking coordinated influence operations](https://arxiv.org/abs/2310.09884)

**Prevalence, rally, methodology**
- [BotPercent: Estimating Bot Populations in Twitter Communities](https://arxiv.org/abs/2302.00381)
- [Crane & Sornette — Robust dynamic classes revealed by measuring response function](https://www.pnas.org/doi/10.1073/pnas.0803685105)
- [Kleinberg — Bursty and Hierarchical Structure in Streams](https://www.cs.cornell.edu/home/kleinber/bhs.pdf)
- [Jain, White & Radivojac — Recovering true classifier performance in PU learning (AAAI 2017)](https://arxiv.org/abs/1702.00518)
- [Saerens, Latinne & Decaestecker — Adjusting the outputs of a classifier (SLD-EM)](https://doi.org/10.1162/089976602753284446)
- [Ambroise & McLachlan — Selection bias in gene extraction (PNAS 2002)](https://www.pnas.org/doi/10.1073/pnas.102102699)
- [Benjamini & Hochberg — Controlling the FDR](https://www.jstor.org/stable/2346101)
- [Chao & Shen — Nonparametric estimation of Shannon's index](https://doi.org/10.1023/A:1026096204727)

**Data sources**
- [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) · [API reference](https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md)
- [Academic Torrents — Reddit dumps 2005-06 → 2023-12](https://academictorrents.com/details/9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4)
- [GDELT DOC 2.0 API](https://api.gdeltproject.org/api/v2/doc/doc)
