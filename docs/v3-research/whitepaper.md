# Scoring Bot-Like Behavior on India-Focused Subreddits: Methodology and Results

**Status:** milestone freeze, 2026-08-21. Corpus-wide account scoring, n=76 hand-labeled bots.
**Next planned step:** scale the labeled set from 76 → ~300 confirmed bots (same duplicate-text
screen, larger candidate pool / relaxed match threshold) and re-check whether Method 2's AUC holds
or improves. Not yet done — this document freezes what's true as of the 76-account label set.

---

## 1. Corpus

- **1,619,492 comments**, full body text, from `commenters_dedup` (collected via `scripts/v3_collect.py`
  against the Arctic Shift API, no re-scraping needed for this work).
- **348,085 distinct commenting accounts**; `account_features` (the account-level feature table used
  throughout) covers **347,886** of them.
- **45 India-focused subreddits**, **24 months** (2024-08 → 2026-07).
- Every account-level metric used below is corpus-wide (aggregated over an account's whole observed
  history in this dataset), not month-specific. See §7 (Limitations).

## 2. Literature and prior tips used

Two kinds of sources fed the candidate feature list and the ground-truth construction:

**Academic (existing project literature, `V3_PLAN.md` §references):**
- Kumar et al., *An Army of Me: Sockpuppets in Online Discussion Communities* (WWW 2017),
  [arXiv:1703.07355](https://arxiv.org/abs/1703.07355) — the account-level ceiling this project has
  repeatedly measured against (AUC 0.65–0.80 on activity+community+linguistic features); also the
  source of the pair-level ceiling (AUC 0.91) that motivated this project's earlier Stage 4 work.
- Kumar et al., *Community Interaction and Conflict on the Web* (WWW 2018),
  [arXiv:1803.03697](https://arxiv.org/abs/1803.03697).
- Schoch et al., *Coordination patterns reveal online political astroturfing*, Nature Communications
  2022, [nature.com/articles/s41467-022-35576-9](https://www.nature.com/articles/s41467-022-35576-9) —
  74% recall at ~1% FPR on ≥10-repetition coordination, the reference point this project's Stage 4 B1
  null-model work was checked against.
- Luceri et al., *Unmasking coordinated influence operations*,
  [arXiv:2310.09884](https://arxiv.org/abs/2310.09884).
- Weber & Neumann, *TROLLMAGNIFIER: Detecting State-Sponsored Troll Accounts on Reddit*,
  [arXiv:2112.00443](https://arxiv.org/abs/2112.00443).
- Feng et al., *BotPercent: Estimating Bot Populations in Twitter Communities*,
  [arXiv:2302.00381](https://arxiv.org/abs/2302.00381).
- Jain, White & Radivojac, *Recovering true classifier performance in positive-unlabeled learning*
  (AAAI 2017), [arXiv:1702.00518](https://arxiv.org/abs/1702.00518) — relevant framing for this
  document's own limitation: our "clean" label is "read and not flagged," not "provably human," which
  is structurally a PU-learning setting, not a clean binary-label one.

**Practitioner (two live Reddit threads, read directly, not summarized secondhand):**
- r/ModSupport, [*Tips on spotting bot/scam accounts*](https://www.reddit.com/r/ModSupport/comments/1ovf843/tips_on_spotting_botscam_accounts/)
  (2026-11) — moderator-perspective tips: karma/account-age/identical-comment screening,
  dormant-then-suddenly-active accounts as a red flag, cross-city/cross-topic repost bots, keyword-based
  automod filtering, "backstory → product mention → rave review" shill narrative arcs, and coordinated
  setup-post-then-reply-accounts patterns.
- r/TheGirlSurvivalGuide, [*Recognizing bot comments*](https://www.reddit.com/r/TheGirlSurvivalGuide/comments/1vs0ril/recognizing_bot_comments/)
  (2026-11) — end-user-perspective tips: generic/insincere always-positive flattery, default
  username patterns, old-dormant-account reactivation, and near-identical replies from different
  accounts on the same thread.

**What transferred and what didn't**, tested directly against this corpus:
- ✅ **Generic templated flattery** ("Wow beautiful picture", "Amazing outfit looking fabulous") —
  independently caught by both the automated duplicate-text screen (§4) and by every LLM review batch,
  with zero prompting toward this specific pattern. Real, and specific to this corpus.
- ✅ **Cross-post/cross-sub duplicate text** — the single tip both threads converge on, and the one
  the whole ground-truth pipeline (§4) is built around. Directly productive: 16.1% hit rate vs. 5.6%
  on unscreened accounts.
- ❌ **Dormant-account reactivation** (old account, long inactive, suddenly posting) — built as a
  feature (`oldness_pctl` × `freshness_pctl` via `account_ordinal` and `days_since_first_seen`,
  see `scripts/` history) and tested against all confirmed bots: **no signal** (mean percentile 27.4
  vs. a ~25 baseline for the min-of-two-uniforms statistic). Plausible explanation: that tip describes
  hijacked/marketplace-sold sleeper accounts running influence campaigns, a different bot species than
  the referral-spam/templated-comment accounts this corpus's screen actually surfaces.

## 3. Ground truth: constructing the labeled set

**Candidate generation (`scripts/v3_stage5_bot_candidates.py`, fully deterministic, no API calls):**
SQL screen over `commenters_dedup` for accounts posting the same body text (≥30 chars, to skip trivial
short reactions) across **≥2 different posts**, excluding official subreddit-bot account name patterns
(`%ModTeam%`, `%AutoModerator%`, etc.). **385 candidates** from the full 1.6M-comment corpus.

**Labeling (LLM batch review, 6 parallel agents, one comment-history file each, no API calls —**
**all from data already local):** each candidate's actual duplicated text (plus surrounding history)
read and classified as `CLEAR_BOT` / `SUSPICIOUS` / `CLEAN`, calibrated against known real examples
from an earlier, separate verification round (36 + 30 + 100 + 100 = 266 accounts, a different sampling
exercise entirely — see `V3_PLAN.md`'s boundary-discovery section for that thread).

| | this round (385 candidates) | earlier round (266 accounts, unscreened) |
|---|---|---|
| CLEAR_BOT | 62 (16.1%) | 15 (5.6%) |
| SUSPICIOUS | 49 | 9 |
| CLEAN | 274 | 242 |

**Combined labeled set: 76 CLEAR_BOT ∪ 58 SUSPICIOUS ∪ 516 CLEAN** (`output/v3/confirmed_bots.json`,
`suspicious_accounts.json`, `clean_accounts.json`). SUSPICIOUS accounts are excluded from both methods'
training — they're genuinely ambiguous on manual read (e.g., a real person restating a strong opinion
twice), not silently folded into either class.

**A concrete example, found and independently reconfirmed by two separate review batches:** the
usernames `CritFin` → `Critifin` → `criti_fin` form a ban-evasion sockpuppet chain — 9 identical
political talking points posted verbatim across all three names, same subreddits
(r/IndiaSpeaks, r/unitedstatesofindia), e.g. *"Our aim should be to have zero crimes. But India already
has low rape rate on per million basis..."* posted under both `CritFin` and `Critifin`.

## 4. Method 1 — hand-built composite (bivariate pruning)

`scripts/v3_stage5_method1_composite.py`. Procedure specified by the analyst:

1. Bivariate Spearman correlation across 31 candidate account-level metrics (the same set already
   computed for the EDA dashboard's heatmap).
2. Greedy pairwise pruning: repeatedly find the most-correlated remaining pair (|ρ| > 0.5), drop
   whichever side has the higher mean |ρ| against everything else still alive (the `findCorrelation`
   algorithm). This is deterministic and auditable — full log in `output/v3/method1_pruning.json`.
   It naturally dropped `thin_history_score` (rank-correlates **−1.000** with `n_comments_sample` — a
   near-exact duplicate, not independent information) and `botmarker_composite` (redundant with
   `removal_rate`, ρ=0.60) **without needing either hard-excluded up front** — the algorithm rediscovers
   what earlier ad-hoc analysis in this project found by hand.
3. **31 → 17 survivors**: `account_ordinal`, `comments_per_day_since_first_seen`, `controversiality_rate`,
   `days_since_first_seen`, `karma_extremeness`, `karma_per_post_extremeness`, `mean_body_len`,
   `mean_depth`, `median_comment_score`, `n_high_tier`, `n_low_tier`, `n_subs_rejected_but_returned`,
   `own_post_reply_rate`, `posts_per_day_since_first_seen`, `removal_rate`, `repeat_engagement_rate`,
   `username_char_entropy`. Each percentile-ranked (direction-flipped so higher = more "flagged") and
   averaged. 10 of the 17 directions are established elsewhere in this project's prior work; 7 are
   judgment calls with no such backing (`days_since_first_seen`, `mean_body_len`, `mean_depth`,
   `median_comment_score`, `n_high_tier`, `n_low_tier`, `repeat_engagement_rate`) — flagged, not hidden.

**Results** (`output/v3/method1_results.json`):

| check | result |
|---|---|
| Target 1a: ρ(composite, `removal_rate`) | 0.274 (component metric — partly circular, not independent validation) |
| Target 1b: ρ(composite, `username_char_entropy`) | −0.160 (component metric, same caveat) |
| **Target 2: AUC vs. 76 marked bots** | **0.474 — no better than random** |
| Top-decile capture | 23 / 76 bots (30%, barely above the ~10% expected by chance at AUC≈0.5) |

**This is the headline negative result of this milestone.** An equal-weighted average of 17
independently-defensible, low-multicollinearity metrics does not automatically produce a working
detector. §5 shows why: the 1–2 metrics that actually separate real bots get diluted by ~15 that carry
near-zero signal on their own.

## 5. Method 2 — XGBoost classifier

`scripts/v3_stage6_method2_xgboost.py`. 25 candidate features (deliberately including ones Method 1's
pruning dropped for redundancy — e.g. `score_stddev`, `reception_spread`, `karma_per_day_since_first_seen`
— a tree-based model isn't harmed by correlated inputs the way an averaged composite is). `max_depth=3,
n_estimators=200, learning_rate=0.05`, `scale_pos_weight` for the 76:516 class imbalance.

| metric | value |
|---|---|
| Train AUC | 1.000 (expected overfit at n=76; not the reported number) |
| Held-out test split (25%, 19 bots) | 0.769 |
| **5-fold CV AUC (headline number)** | **0.792**, folds: [0.718, 0.762, 0.762, 0.867, 0.853] |
| Target (analyst-specified) | > 0.80 |

**Right at target, not cleanly over it** — reported honestly rather than cherry-picking a
higher single-run number seen during exploration (0.804 on an earlier, functionally-identical run;
the ±0.07 spread across folds is the more important number than any single point estimate at this
sample size). A simpler unweighted decision tree (max_depth=2, percentile features only) gets 0.733
mean CV with just 4 leaves — see §6 for why that one is more *interpretable* even though XGBoost scores
slightly higher.

**Feature importances** (top 10 of 25, XGBoost gain-based):

| feature | importance |
|---|---|
| `median_comment_score` | 0.072 |
| `score_stddev` | 0.068 |
| `subreddit_entropy` | 0.063 |
| `karma_per_day_since_first_seen` | 0.059 |
| `n_threads_active` | 0.054 |
| `n_low_tier` | 0.052 |
| `mean_body_len` | 0.050 |
| `n_high_tier` | 0.046 |
| `repeat_engagement_rate` | 0.046 |
| `n_comments_sample` | 0.044 |

Notably **flat** — no single feature dominates (top feature is only 7.2%), unlike the shallow
single-tree version below. `comments_per_day_since_first_seen`, `posts_per_day_since_first_seen`, and
`removal_rate` — the features earlier ad-hoc analysis in this session spent the most time on — rank
near the *bottom* of this list (posts/day 25th of 25, comments/day 21st, removal_rate 12th).

## 6. Why Method 1 fails and Method 2 doesn't: the interpretable version

A max-depth-2 decision tree on the *same* percentile-ranked features Method 1 used (4 leaves, no
raw-value thresholds) scores 0.733 mean CV — worse than XGBoost but far better than the linear
composite, and small enough to read directly:

```
score_stddev ≤ 28th percentile (LOW score variance)
  → BOT (regardless of everything else)

score_stddev > 28th percentile (HIGH score variance)
  → mean_body_len ≤ 92nd percentile (not unusually long)  → CLEAN
  → mean_body_len > 92nd percentile (very long comments)  → BOT
```

This runs opposite to the variance hypothesis that motivated adding `reception_spread`/`score_stddev`
to the candidate list in the first place (*"high-intensity users... take karma from some and bleed
somewhere else"*). The dominant branch is **low** variance, not high: most of the 76 confirmed bots are
low-effort templated spam (referral links, generic flattery, subreddit self-promo) that gets small,
boring, *consistent* scores every time — not wild swings. The high-variance pattern is real but
secondary — it shows up specifically combined with unusually long comments (the copy-pasted
political-essay pattern, e.g. the `CritFin` chain) — not as a standalone signal.

**The core lesson for why Method 1 underperforms Method 2:** averaging in 15 metrics that are each
individually weak (comments/posts-per-day, removal_rate, account_ordinal — all near-zero XGBoost
importance) dilutes the 1–2 metrics that actually carry signal. A tree-based model can *ignore* a weak
feature entirely at every split; a linear average cannot.

## 7. Limitations

- **n=76 positives is small.** CV fold AUCs range 0.72–0.87 — meaningfully unstable. The
  ranked-importance list and even the 0.79-vs-0.80 target comparison should be read as directional, not
  final, until the labeled set grows (planned next step: 76 → ~300, same screening method, see the
  status line at the top of this document).
- **Corpus-wide, not month-specific scores.** The MoM dashboard (§8) rolls up a corpus-wide account
  score by which accounts were *active* in a given month, comment-count-weighted — it is not a
  month-specific re-scoring of behavior. A subreddit's month-over-month trend line reflects shifts in
  *which accounts posted that month*, not changes in any individual account's behavior.
- **"Clean" ≠ "provably human."** `clean_accounts.json` means "read by an LLM reviewer and not flagged
  as bot-like" — a positive-unlabeled setting (cf. Jain, White & Radivojac, §2), not a verified-negative
  one. Some fraction of "clean" accounts are certainly undetected bots; this inflates the false-negative
  rate in a direction we can't currently measure.
- **India-subreddit-specific.** Every feature, threshold, and the entire labeled set come from 45
  India-focused subreddits. Nothing here has been checked against other communities.
- **Subreddit-level ranking is separately, directly falsified as a bot-density signal.** An earlier,
  independent verification round in this project (documented in `V3_PLAN.md`) manually read real
  comment samples from the highest- and lowest-ranked subreddits under several earlier composite
  variants and found **no consistent difference in bot presence** between them — the ranking mostly
  tracked meme-culture-vs-discussion-culture subreddit style, not bot density. That finding predates
  Method 1/Method 2 above and has not yet been re-tested against the current model; treat any
  subreddit-level ranking from this methodology as unverified until it is.
- **Judgment-call feature directions** (§4) are asserted, not derived — 7 of Method 1's 17 metrics have
  no prior finding in this project backing their "higher/lower = more suspicious" direction.

## 8. Outputs

- `scripts/v3_stage5_bot_candidates.py` → `output/v3/bot_candidates.csv` — candidate generation.
- `scripts/v3_stage5_method1_composite.py` → `output/v3/method1_{pruning,results}.json` — Method 1.
- `scripts/v3_stage6_method2_xgboost.py` → `output/v3/method2_results.json` — Method 2.
- `scripts/v3_stage7_monthly_score.py` → `docs/v3-research/bot-score-mom.json` +
  `docs/v3-research/bot-score-dashboard.html` — the MoM dashboard. **Re-run monthly** (or after any
  relabeling) to refresh; it retrains Method 2 on the current label set and rescoring the full
  347,886-account population, so it stays in sync automatically as the labeled set grows.
- `output/v3/{confirmed_bots,suspicious_accounts,clean_accounts}.json` — the labeled ground truth
  itself, the one artifact everything else depends on.
