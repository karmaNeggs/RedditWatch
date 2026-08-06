# V3 plan — exposure-weighted inauthenticity scoring

Supersedes `V3_FEATURE_PLAN.md` and `V3_DATA_SOURCES.md`. Those remain valid as
background; where they disagree with this document, this document wins.
`V3_METRIC_CATALOGUE.md` is the full candidate-metric sweep this document's §4
was checked against and updated from (2026-08-05) — kept separate because of
its size, not because it's optional background.

Status marks: ✅ **measured** in this project · 📚 **published**, cited ·
🔬 **hypothesis**, untested · ❌ **unavailable**, confirmed by audit.

---

## 🚦 START HERE (next session) — as of 2026-08-06

**Done:** full collection (§10.3, 128,374 posts / 2,312,696 commenter rows,
data committed) → Stage 0 clean + analysis layer (§10.4, `data/v3/analysis/v3.duckdb`,
gitignored, rebuild with `python3 scripts/v3_stage0_build.py`) → account
feature table (§10.4, 347,886 accounts × 40+ columns, rebuild with
`python3 scripts/v3_account_features.py`) → EDA page with risk-colored
histograms + bot-marker composite (rebuild with `python3 scripts/v3_eda_build.py`
→ `docs/v3-research/eda/index.html`, published at
https://claude.ai/code/artifact/26b907c2-1410-4b94-852a-b2cd10febcad) → Stage 1
fix + Stage 2 bivariate/segmentation (§10.4, `docs/v3-research/eda/stage2.html`,
linked from the EDA page nav) → feature sanitisation for Stage 3
(§10.4, `account_features_model` table, 347,886 × 73 cols, rebuild with
`python3 scripts/v3_feature_sanitise.py`) → **Stage 3 account XGBoost per
label channel** (§10.4, `docs/v3-research/eda/stage3.html`, rebuild with
`python3 scripts/v3_stage3_account_model.py`; new deps `xgboost`, `shap`,
`statsmodels`, plus a system `libomp` — all now in `requirements.txt`).

**Previous open thread #1 resolved, not as originally diagnosed — see §10.4
for the full before/after.** The plan's "`percent_rank()` doesn't spread ties"
theory was wrong; `percent_rank()` is correct, the ties are real (small-n rate
features on a median-2-comments corpus). The actual bug was in the GMM/BIC
screen itself (only stripped the single largest point-mass, missed secondary
clumps) plus GMM decorating ordinary skew as false bimodality. Both fixed.
Stage 1 went from 18/19 "bimodal" (not credible) to 9/36 candidates, only 2
robust across parameterizations. Bot-marker screen went from "all 7 real" to
1/7. **Still open:** 3 features (`n_posts_sample`, `n_high_tier`,
`n_threads_with_repeat`) return a residual DEGENERATE verdict the point-mass
strip cap doesn't catch, and the 9-candidate count is itself
threshold-sensitive (3–9 across two reasonable settings) — flagged, not
resolved.

**Stage 3 headline, corrected 2026-08-06 (same day, second pass): `admin_removal`'s
0.896 was leakage, not signal — fixed, and the corrected number is 0.743.**
Root cause **confirmed by direct SQL inspection**, not inferred:
`v3_account_features.py` aggregates every "behavioral" feature over *all* of
an account's sampled rows with no exclusion for rows that themselves define
the label — for a thin-history account (22.9% of `admin_removal` positives
are singletons; median account-wide is 2 comments) the "behavioral" features
being fitted on **are substantially the removed row's own metadata**, not
independent past behavior. Diagnostic sweep: rung-4 AUC restricted to
`n_comments_sample ≥ 5` → 0.822; `≥ 10` → **0.743**, landing exactly inside
Kumar's 0.65–0.80. The specific mechanical sub-hypothesis (removed-content
`score` corrupted by the `_meta` block's +16s/+36h snapshot merge, §3) was
tested directly and **rejected** — admin-removed comments score *higher*
(mean 31.5) than the same account's kept comments (mean 19.2), the opposite
of a depressed-score artefact.

**The literal fix (leave-one-out: rebuild features excluding each channel's
own label rows) was tried, and made things worse — rejected, not adopted.**
It turns thin-history accounts' features into `NaN`, which XGBoost's
missing-value handling then reads as a near-perfect label proxy — a
*different*, worse leak (`self_deletion` rung-4 collapsed to 0.419;
`automod_filtered` spiked to 0.989 — that inconsistency is itself the
tell). **Adopted instead: volume-gating** (`n_comments_sample ≥ 10`, dilutes
any single row to ≤1/10 of the aggregate — no new leak introduced). Gated
rung-4, side by side with the original:

| channel | original rung4 | **gated rung4 (adopted)** |
|---|---|---|
| `admin_removal` | 0.896 | **0.743** ✅ in range |
| `self_deletion` | 0.778 | **0.695** ✅ in range |
| `comment_removed_ambiguous` | 0.805 | **0.641** ⚠️ now *below* range, unexplained |
| `automod_filtered` | 0.653 | 0.725 (barely moves — label is post-level, this leak mechanism doesn't apply) |
| `moderator_removed` | 0.637 | 0.687 (same — post-level label) |

Also surfaced: `admin_removal` ∩ `self_deletion` = 39.9% account overlap,
`self_deletion` ∩ `comment_removed_ambiguous` = 68.6% — the "5 independent
channels" framing needs this caveat; self-deletion plausibly often *follows*
another moderation action rather than being independent of it. Full
before/after table, the rejected-LOO numbers, and the rest of the §8
leakage-register walk (items 2/3/4/6) are in §10.4.

**Bottom line for Stage 4: nothing here points to an account-level result
near pair-model territory.** The 0.90+ claim continues to rest entirely on
Stage 4's pair-level model, not on any Stage 3 channel — which is what §1
said from the start.

**One open thread remains:** Base36 age calibration is on hold, not
abandoned — script (`scripts/v3_calibrate_age_sample.py`) works, live Reddit
access confirmed, just deprioritized in favor of the free
`days_since_first_seen` proxy.

**Not started: Stage 4 (pair-level model — where the plan's 0.90+ AUC claim
actually lives, §1/§4.4) → Stage 5 (prevalence) → rally+GDELT → dashboard.**
Account-level work (everything done so far, Stage 3 included) explicitly
**feeds** the pair layer per §1 — it was never meant to be the deliverable on
its own. Stage 4 is the next big lift.

**Before touching account_features again:** read the docstring at the top of
`scripts/v3_account_features.py` — it explains the bipartite-sampling
constraint (median 2 comments/account) and the three different "age-like"
quantities now in the table, so they don't get re-litigated or conflated.
**Before building on Stage 3:** the leakage exclusion list (`removal_rate`,
`deleted_later_rate`, their hurdle indicators, and the 6 reporting-only
columns) lives in `scripts/v3_stage3_account_model.py` — reuse it rather than
re-deriving, and note `karma_extremeness`/`karma_per_post_extremeness` are
documented as reporting-only in `v3_feature_sanitise.py`'s docstring but are
actually still in the model-ready column set (doc/code mismatch, flagged not
fixed).

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
   - Stage 4 (pair model): not yet started — the next big lift per §1/§7.
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
