# V3 plan — final pass

Companion to `V3_DATA_SOURCES.md` (what data is reachable). This is the plan:
objective, method, metric catalogue, and modelling protocol. Written as a
clean sheet — V2's analysis is assumed dumped, not migrated.

Status marks used throughout:
✅ **verified** in live responses · ◻️ **needs check** (check stated) ·
🔬 **derived hypothesis** (no availability risk, signal claim untested) ·
📚 **published method** (grounded in literature, cited at the end).

---

## 0. Objective

**The deliverable is a subreddit-month verdict.** Everything at account and
post level exists to serve that. Stated precisely:

> For each (subreddit, month), estimate **what share of activity is
> inauthentic**, **how confident we are**, and **what kind of inauthenticity**
> it is — with every point of score traceable to named accounts and threads a
> human can open and check.

Three sub-objectives, deliberately separated because they have different
evidence standards and get confused otherwise:

| | Question | Output | Evidence standard |
|---|---|---|---|
| **A** | How much of this sub-month is automated/inauthentic? | % of activity | calibrated prevalence estimate |
| **B** | Is there *coordination* here, beyond individual bad accounts? | cluster list + z | permutation null |
| **C** | Was there a *rally* — a sudden mobilisation event? | event list + magnitude | changepoint vs. sub's own baseline |

A, B and C are different phenomena. A single bot farm posting steadily is A
without C. A genuine news event brings C without A or B. A brigade is B and C
without much A. V2 collapses all three into one number, which is a large part
of why it can't say what it found.

**Method commitment (your framing, and it's the right one):** learn the
boundary from the data first, then classify. Not: assume a label, fit, hope.
§3 makes this concrete.

---

## 1. What the literature actually does

Searched rather than assumed. Four families, all applicable, all with things
worth stealing.

**Seed-and-expand — the closest match to this project.**
[TROLLMAGNIFIER](https://arxiv.org/abs/2112.00443) (Reddit specifically) starts
from 335 Reddit-confirmed Russian troll accounts and expands to 1,248
candidates by learning what seeds have in common: interacting with each other,
similar topics, **temporal synchronisation**, and — notably —
**creation on identical dates**. 66% of detections independently validated;
300% growth over the seed set. 📚

Two things to take from this. First, it validates the registration-cohort idea
(U1 below) as a *used, published* signal rather than a hunch. Second, its
structure is exactly what you described: find a group, learn its shape, assign
probability to accounts near that shape.

**Coordinated-behaviour networks.**
[CooRnet / CLSB](https://coornet.org/) builds co-action networks: entities
repeatedly sharing the same URL within an anomalously short window.
The key methodological move is that **the coordination interval is estimated
from the data**, not hand-set — it's the time threshold that is unusually
short relative to the whole dataset's distribution. [CooRTweet](https://github.com/nicolarighetti/CooRTweet)
generalises this to arbitrary shared actions. 📚

Take: don't hand-pick "co-appearance within 60 seconds." Derive the threshold
from the corpus's own inter-arrival distribution. This directly answers "how
tight is suspicious?" without a magic number.

**Community-level prevalence.**
[BotPercent](https://arxiv.org/abs/2302.00381) addresses exactly sub-objective
A — estimating bot *populations in communities* rather than classifying
individuals. Its central finding is the one to internalise: **naively averaging
per-account bot scores gives a biased community estimate.** Prevalence needs
explicit calibration. Reported bot shares vary enormously by domain (~10–20%
typical, 45–60% in some contested topics), so an absolute number without a
calibrated baseline is meaningless. 📚

**Temporal / linguistic account features.** The Reddit-specific literature
converges on burstiness of inter-comment gaps and network connectivity across
subreddits, plus lexical repetition and identical-title posting. Broadly what
§2b/§2c below encode. 📚

**Also relevant, from the practitioner side:** Reddit's own vote-manipulation
stack reportedly uses timing-entropy burst detection, co-voter clustering, and
a Contributor Quality Score. The co-voter graph isn't reachable — no vote data,
as established — but timing-entropy bursts are computable from comment
timestamps, and it's a useful signal that the platform itself weights timing
entropy heavily.

---

## 2. What the last pass missed

Re-read against the above, five real gaps. The first two are serious.

### 2.1 No control subreddits — the biggest miss

All 25 tracked subs are Indian, most political or news-adjacent. **There is no
organic baseline in the corpus at all.** Every "anomaly" is measured against
other Indian political subs, so the design cannot distinguish

> "Indian political Reddit is unusually manipulated"

from

> "this is what Reddit looks like everywhere, and we only ever look here."

That's a load-bearing weakness in a project whose entire claim is about a
specific community. Fix: add a control tier and collect it identically.

- **Topic controls** — non-political Indian subs (r/IndianFood, r/india_tourism,
  hobby subs). Same population, different stakes.
- **Geographic controls** — comparable national subs (r/unitedkingdom,
  r/canada, r/brasil, r/philippines). Similar structure, different polity.
- **Scale controls** — large generic subs (r/AskReddit, r/mildlyinteresting)
  to calibrate what volume alone does to every metric.

Every sub-level metric in §4 becomes a *percentile against the control
distribution* rather than an absolute. This is the change that makes the
numbers mean something, and it costs only collection time.

### 2.2 Benign bots will dominate every automation metric

Not hypothetical — visible in the live interaction test I ran during the
sources research. The top co-interaction partners for a normal r/Cricket user
were `CricketMatchBot` (433), `AutoModerator` (91), `CricinfoBot` (38),
`sansa-bot` (27), `NextGenBot` (17). Any naive co-appearance or automation
metric surfaces these first, and "% bot activity" that counts AutoModerator is
not measuring what the dashboard claims.

Needs an explicit **sanctioned-automation tier**, excluded from A and B but
reported separately:

- Reddit's new **"APP" label** for sanctioned automated accounts, introduced
  ~March 2026. ◻️ **Check:** is it exposed via API/archive, or UI-only? If
  exposed, it's a free high-precision automation label.
- Self-declared bots — regex on comment bodies for *"I am a bot"*,
  *"beep boop"*, *"performed automatically"*, and the standard footer
  patterns. 🔬 High precision, near-zero cost.
- Name morphology (`*bot`, `*Bot`, `auto*`) as a weak prior only, never alone.

**And this is also a free positive seed set.** Self-declared bots are
*confirmed automation* — exactly the seed TROLLMAGNIFIER needs. They're benign
automation rather than malicious, so they can't be the training target
directly, but they are ground truth for "what does automated timing/text
actually look like," which is precisely what §3's boundary discovery needs.
That's a genuinely valuable asset the last pass walked straight past.

### 2.3 The label problem is Positive-Unlabeled, not "weak labels"

V2 treats "suspended" vs "not suspended" as a binary. It isn't. Suspended ⇒
almost certainly bad. Not-suspended ⇒ **unknown** — could be good, could be
undetected, could be caught next month. Fitting standard logistic regression on
that systematically underestimates the positive class and mis-calibrates every
probability.

This is a named problem with named solutions: **PU learning** 📚 — either
sample-selection (identify reliable negatives first) or cost-sensitive
(reweight the unlabeled pool by estimated class prior). It changes the
loss function, not just the interpretation. Worth doing properly; it likely
matters more than any individual feature.

### 2.4 Event conditioning

`analysis.py` has an event-calendar overlay, but the plan didn't carry it
forward. Without conditioning on real-world events, elections, cricket matches,
and crises all look like coordination — sudden influx, tight timing, topic
concentration, outsider accounts. **Every sub-objective-C claim must be
evaluated against an event calendar**, and GDELT (verified working in
`V3_DATA_SOURCES.md`) supplies it automatically rather than by hand.

### 2.5 Statistical hygiene at census scale

Two failure modes that only appear once n goes from 35k to 5.6M rows:

- **Everything becomes significant.** At census n, trivial differences produce
  p < 0.001. Report **effect sizes** and rank by them; p-values stop
  discriminating.
- **Multiple testing.** 25 subs × 14 months × ~60 metrics ≈ 21,000 tests. At
  α=0.05 that's ~1,050 false positives by construction — more than enough to
  populate a dashboard entirely with noise. **Benjamini-Hochberg FDR control
  across the whole grid**, not per-metric, is mandatory.

Neither is optional, and both are invisible until they've already produced a
confident wrong answer.

---

## 3. The method: discovery first, then classification

Your framing — *learn bot behaviour from uni/bivariate structure, find a
separation boundary, then assign probability to accounts beyond it* — is
methodologically sound and better-suited to this problem than supervised-first.
Here's how I'd make it rigorous rather than eyeballed.

### Stage 0 — Clean the population

Remove sanctioned automation (§2.2). Remove deleted-author rows from the
modelling population (keep for accounting). Fix age-at-event vs.
age-at-collection (§6 leakage register). Nothing below is trustworthy until
this is done.

### Stage 1 — Univariate: find the multimodal features

For each candidate metric, don't just check separation against a label — check
whether **the feature's own distribution is multimodal**. That's the empirical
question behind "is there a distinct population here at all?"

- Kernel density estimate + **dip test** (Hartigan) for unimodality. A feature
  that fails unimodality is telling you two populations exist *before any
  label is involved*.
- **Gaussian mixture** with k selected by BIC. If a 2-component fit beats
  1-component decisively, the second component's mass is a direct,
  label-free estimate of the anomalous population's size. 📚
- Skew/kurtosis → pre-declare transform (don't pick post hoc; V2 found skews
  of 27 and 41, this will recur).
- **Temporal stability** — per-month distributions. A feature whose own
  distribution drifts produces V2's spurious 13-month climb regardless of how
  good it looks cross-sectionally.

Output: a shortlist of features with genuine population structure, plus a
first label-free prevalence estimate from the mixture masses.

### Stage 2 — Bivariate: find the separation boundary

This is the stage you described, made systematic rather than a scatter hunt.

- Full pairwise scatter/density grid over the Stage-1 shortlist, **coloured by
  the seed sets** (self-declared bots; admin-removed accounts). If seeds
  concentrate in a region, that region is the boundary — and it was found by
  the data, not assumed.
- Spearman correlation + **VIF within evidence families** (§4 groups are
  deliberate: timing features will be collinear with each other and near-
  orthogonal to text features — prune within, keep across).
- **2-D mixture / density-based clustering** (HDBSCAN) on the strongest pairs.
  Look for a satellite cluster detached from the main mass. That's the visual
  you're after, and HDBSCAN gives it a membership probability instead of a
  hand-drawn line.
- Targeted interaction screening with a mechanism required in advance — not
  exhaustive. Candidates: U1×co-appearance (registration adjacency + shared
  threads), U10×U14 (always-on + instant arrival), U2×U5 (dormancy + karma
  step). V2's `old_x_msgs_per_day` failed because it had no mechanism; the
  lesson is to require a story, not to abandon interactions.

### Stage 3 — Propagate probability from the boundary

Once a region is identified, assign every account a probability of belonging to
it. Three estimators, agreement between them being the confidence signal:

1. **Mixture posterior** — P(anomalous component | features). Fully
   unsupervised; no label assumed.
2. **Seed-expansion** 📚 — TROLLMAGNIFIER-style. Learn seed behaviour, score
   the population by similarity, expand. Validate by held-out seeds.
3. **PU classifier** (§2.3) — supervised, but with the right loss.

Report the ensemble, and flag disagreement rather than hiding it behind a mean.
An account all three agree on is a materially stronger claim than one flagged
by the most permissive.

### Stage 4 — Coordination, which is not a classification problem

Co-appearance excess, timing synchrony, text reuse — evaluated against a
**volume-matched permutation null** (preserve each account's activity count and
each thread's size; shuffle assignment; recompute; repeat). Coordination
interval derived from the corpus's own inter-arrival distribution, CooRnet-style,
not hand-set. 📚

No AUC here — there are no coordination labels. Say so on the methodology page
rather than implying validation the design cannot provide.

### Stage 5 — Roll up to sub level, with calibration

Per BotPercent: **do not average account probabilities.** Calibrate against
the control tier (§2.1), estimate prevalence with explicit uncertainty, and
publish an interval rather than a point.

---

## 4. Sub-level metric catalogue

The deliverable layer. Organised by sub-objective. Every metric is expressed
**as a percentile against the control tier**, not as an absolute.

### 4A — Prevalence: "how much of this is inauthentic?"

| # | Metric | Status |
|---|---|---|
| A1 | **% of comments** from accounts above the boundary (Stage 3 ensemble) | 🔬 |
| A2 | **% of posts** from same — reported separately; posting and commenting are different economies | 🔬 |
| A3 | **% of *score* received** by flagged accounts — attention share, not activity share. Often the more alarming number. | 🔬 |
| A4 | % of *top-decile-scoring* content from flagged accounts — concentration at the visible end | 🔬 |
| A5 | Prevalence with CI, mixture-derived (label-free cross-check on A1) | 🔬 |
| A6 | Sanctioned-automation share, reported **separately** (§2.2) | ◻️ |
| A7 | Admin-removal rate (`removed_by_category == 'reddit'`) — platform's own verdict | ✅ |
| A8 | Comment removal rate (measured baseline ~19%) | ✅ |
| A9 | Suspension rate of the month's active accounts, checked at t+90d | ✅ |

A3 is the one I'd lead the dashboard with. "12% of accounts produced 47% of
the upvoted content" is a sharper and more checkable claim than a share of
activity.

### 4B — Coordination: "is there structure beyond bad individuals?"

| # | Metric | Status |
|---|---|---|
| B1 | **Co-appearance excess z** vs. volume-matched null | 🔬 📚 |
| B2 | Number and size of clusters passing FDR-controlled significance | 🔬 |
| B3 | **Registration-cohort concentration** — U1 aggregated | ◻️ 📚 |
| B4 | Reply-graph modularity; density and isolation of detected communities | 🔬 |
| B5 | Cross-account text-template reuse rate | 🔬 📚 |
| B6 | **Derived coordination interval** — the sub's own "anomalously tight" threshold. A sub whose threshold is far tighter than controls is itself a signal. | 🔬 📚 |
| B7 | Reply reciprocity concentration — small sets replying mainly to each other | 🔬 |
| B8 | **Directed cross-sub flow** — accounts appearing in A *then* B. Direction distinguishes source from target; V2's symmetric overlap cannot. | 🔬 |
| B9 | Domain–account bipartite concentration (a domain posted only by a few accounts) | 🔬 |
| B10 | Stylometric linkage — accounts whose writing fingerprints match | 🔬 |

B10 is the sockpuppet-detection angle the last pass missed entirely. Character
n-gram + punctuation + emoji profiles link alt accounts of the same operator
independently of timing or co-appearance — a third orthogonal channel.

### 4C — Rally: "was there a mobilisation event?"

Entirely absent from V2, and the metric family you specifically asked for.

| # | Metric | Status |
|---|---|---|
| C1 | **Changepoint detection** on daily activity vs. the sub's own baseline | 🔬 |
| C2 | **Outsider-influx share** — % of thread commenters with no prior history in this sub. Literature notes 30/40 first-time commenters is not organic growth. | 🔬 📚 |
| C3 | New-account influx — share of activity from accounts <30d old, sub-relative | 🔬 |
| C4 | **Arrival-burst tightness** — inter-arrival distribution of the first N commenters. Coordinated: tight cluster then silence. Organic: power-law decay. | 🔬 |
| C5 | Topic concentration spike (from `link_flair_text`, free) | ✅ |
| C6 | Sentiment/stance polarisation spike | 🔬 |
| C7 | `controversiality` rate spike | ✅ |
| C8 | `collapsed_because_crowd_control` rate — Reddit's own brigading heuristic | ◻️ |
| C9 | **Event-conditioned residual** — rally magnitude *after* removing what GDELT-observable news explains. This is the metric that matters. | 🔬 |

C9 is the whole point of C. Raw spikes are mostly real news. The residual —
mobilisation *unexplained* by observable events — is the interesting quantity,
and it's the honest version of "high rally."

### 4D — Structure & confounders

| # | Metric | Status |
|---|---|---|
| D1 | Gini over account **activity** (V2 Ginis *scores*, mostly a popularity artifact) | 🔬 |
| D2 | Account churn / turnover MoM | 🔬 |
| D3 | Diurnal profile vs. IST expectation | 🔬 |
| D4 | Outbound-domain Herfindahl | ✅ |
| D5 | **Mod intensity** (`distinguished`/`stickied`/`locked`/removal rates) | ✅ |
| D6 | Sub growth rate, `subreddit_subscribers` normaliser | ✅ |
| D7 | Score:comment decoupling — high score, no discussion | ✅ |

**D5 is a confounder, not a signal.** Heavily-moderated subs look cleaner for
reasons unrelated to bots. Any cross-sub comparison ignoring mod intensity is
partly ranking mod staffing. V2 ignores it entirely.

---

## 5. Account and post metrics (inputs to §4)

Condensed — these feed the sub level, they aren't the deliverable.

**Provenance.** U1 registration-cohort adjacency (◻️ 📚 — decode `t2_`
`author_fullname`, count corpus accounts at near-adjacent IDs; validated as a
signal by TROLLMAGNIFIER's "created on the same exact day") · U2 dormancy gap ·
U3 username morphology (default `Adjective_Noun_1234` rate, n-gram entropy) ·
U4 `author_premium` · U5 karma accrual shape · U6 flair possession.

**Timing.** U7 interval entropy · **U8 interval quantisation** (gaps mod
60/300/900s — cron signature, near-binary tell, very low organic base rate) ·
U9 burstiness B=(σ−μ)/(σ+μ) 📚 · U10 circadian dead-hours · U11 circadian
centroid vs. IST · U12 weekday:weekend · U13 session structure ·
**U14 time-to-arrival on new threads** (consistent sub-2-min ⇒ programmatic
feed monitoring) · U15 activity changepoints.

**Content.** U16 self-similarity · U17 cross-account near-duplicates ·
U18 type-token ratio · **U19 script/language mixing** (Latin/Devanagari,
Hinglish markers — domain-specific and discriminative here) · U20 emoji/
punctuation fingerprint (also feeds B10) · U21 edit rate and latency ·
U22 reply:toplevel ratio · U23 URL rate and domain concentration.

**Reception** — harder for an operator to control than their own behaviour,
which is what makes it valuable. U24 `controversiality` rate ·
U25 crowd-control collapse rate ◻️ · U26 score distribution shape (both tails:
consistently-zero *and* suspiciously-uniform-positive) · U27 awards received ·
**U28 incoming-reply rate** (bots get talked *past*, not *to* — low incoming
replies at high volume is a strong tell).

**Footprint.** U29 subreddit entropy · U30 absence of unrelated hobby subs ·
U31 political concentration · U32 sub-adoption lag.

**Post level.** P1 time-to-first-comment · **P2 early-arrival shape** ·
P3 implied downvotes from `score`×`upvote_ratio` (trivially derivable, never
computed by V2) · P4 comment:score ratio vs. baseline · P5 score per
subscriber · **P6 thread width:depth** (coordinated threads are wide and
shallow — many top-level drops, no conversation) · P7 in-thread reciprocity ·
P8 outsider share · P9 domain–account concentration · P10 title-template reuse
· P11 crosspost fan-out timing · P12 `is_created_from_ads_ui` ◻️ ·
P13 edited-after-scoring · P14 `link_flair_text` as free topic label.

---

## 6. Labels and the leakage register

**Label tiers**, kept separate rather than merged:

| Tier | Source | Cost | Role |
|---|---|---|---|
| Confirmed automation | self-declared bots, "APP" label | free | seed set for boundary discovery — benign, so not the target |
| Strong negative-quality | `removed_by_category == 'reddit'` (admin, ~4%) | free | primary PU positive |
| Moderate | comment `[removed]` (~19%), suspension at t+90d | free / 1 call | secondary targets |
| Weak | moderator removal (~57%) | free | mostly rule violations — separate target, not merged |

**Leakage register — verify each before fitting.**

1. Removal-derived features cannot be used against removal targets. Hard
   partition.
2. `author == '[deleted]'` ↔ suspension, correlated by construction.
3. `retrieved_on − created_utc` (capture lag) correlates with removal timing.
   Never a feature.
4. **Age at collection vs. age at event.** V2 takes age from `about.json` at
   collect time, so an account's "age" differs across months for reasons
   unrelated to the account. Compute at each event's `created_utc`. This one
   is subtle and probably already biasing V2's temporal comparisons.
5. Sampling-driven volume leakage — check label rate against activity decile.
6. Score-derived features against moderation labels: mods act on downvoted
   content, so score partly *causes* the label.
7. Negative-class contamination — the PU problem (§2.3). Reflect in wording,
   not just in the metric.

**Validation:** grouped CV by account *and* month-blocked (train early → test
late). Report both; the month-blocked number is the one that matters for a
system making forward claims.

---

## 7. Drop from V2

Not migrate — drop. Each compensates for a defect the census removes; carrying
it forward reimports the assumption.

Six legacy heuristic components (never individually validated) · `decay_slope`
(invented to survive top-N truncation) · month-relative decile thresholding
(a patch for population drift — model drift directly) · `score_cv` (documented
sampling artifact) · fixed severity bands (recalibrated per run against a
moving population, so not comparable across time) · `calibrated_weights` /
`pca_weights` in `findings.json` (already flagged vestigial) ·
`analyze_network()` as written (set arithmetic, not network analysis).

---

## 8. Verification checklist

All short, none needing the full census. Ordered by how much they'd change the
plan if they fail.

1. **`author_fullname` monotonicity**, 2023–2026 — decode ~2,000 IDs against
   known ages. Gates U1/B3, the highest-upside items. Reddit changed ID
   allocation ~2021, so this is genuinely uncertain. *~20 min.*
2. **Comment-body retention for removed comments** — original text or
   `[removed]` tombstone? Determines whether text features exist for the
   positive class *at all*. I expect this to be a real problem. *~10 min.*
3. **`author_fullname` on suspended accounts** — scrubbed? If so U1 is
   unavailable exactly where it matters. *~10 min.*
4. **"APP" label exposure** via API/archive — free automation label if yes. *~10 min.*
5. **Self-declared-bot regex yield** — how many seeds does it actually
   produce? *~15 min.*
6. **`collapsed_because_crowd_control` density** — usable or ~always null?
   Gates U25/C8. *~10 min.*
7. **Admin-removal rate at scale** — the 4% was one 100-post slice. *~15 min.*
8. **`is_created_from_ads_ui` base rate** — probably ~0. *~5 min.*

---

## 9. Sequencing

1. **Verification checklist** (§8) — ~1.5h, and items 1–2 can invalidate large
   parts of §4/§5.
2. **Control-tier selection** (§2.1) — a decision, not a computation, but
   everything downstream is expressed relative to it, so it must come first.
3. **Single-sub census pilot** — r/india + one control, 3 months. Run Stages
   1–2 and look at the actual scatter. This is where you find out whether a
   separation boundary visibly exists.
4. **Boundary + propagation** (Stages 3–4) on the pilot.
5. **Full census**, then §4 at scale with FDR control.
6. **GDELT event conditioning** for C9.

Step 3 is the decision point. If the univariate/bivariate work shows no
population structure, that is a finding worth having early and cheaply — and it
would mean the honest product is an anomaly-surfacing tool, not a detector.

---

## 10. Honest read

What improves: prevalence estimation gets a real denominator and a control
baseline; coordination gets structural evidence with a null model; rally gets
separated from prevalence and conditioned on real events; the label problem
gets the right math instead of a weak binary.

What doesn't: **there is still no ground truth for coordination.** The seed
sets are confirmed *automation* (benign) and confirmed *platform action*
(heterogeneous) — neither is "confirmed influence campaign." Stage 3's output
is "this account resembles a population that behaves anomalously," and Stage
4's is "this cluster is statistically unusual." Neither is proof, and the
distinction has to live in the UI rather than in a methodology footnote.

The realistic ceiling for "proving coordinated influence" is still ~6/10 — but
the *investigative* value goes high, because every number now names accounts
and threads someone can open. That traceability, more than any AUC, is what
would make this defensible.

---

## Sources

- [TROLLMAGNIFIER: Detecting State-Sponsored Troll Accounts on Reddit](https://arxiv.org/abs/2112.00443)
- [BotPercent: Estimating Bot Populations in Twitter Communities](https://arxiv.org/abs/2302.00381)
- [CooRnet — coordinated link sharing behaviour](https://coornet.org/) · [A-B-C framework](https://coornet.org/abc.html)
- [CooRTweet — generalised coordinated-action detection](https://github.com/nicolarighetti/CooRTweet)
- [Unsupervised detection of coordinated information operations in the wild](https://arxiv.org/pdf/2401.06205)
- [Exposing Cross-Platform Coordinated Inauthentic Activity, 2024 US Election](https://arxiv.org/html/2410.22716v2)
- [Machine learning-based social media bot detection: literature review](https://link.springer.com/article/10.1007/s13278-022-01020-5)
- [Understanding Longitudinal Behaviors of Toxic Accounts on Reddit](https://arxiv.org/pdf/2209.02533)
- [Coarse-to-fine label propagation for semi-supervised bot detection](https://link.springer.com/article/10.1007/s11276-024-03821-2)
- [Uncertainty-aware Pseudo-label Selection for Positive-Unlabeled Learning](https://arxiv.org/pdf/2201.13192)
- [Reddit to require human verification, label automated accounts (TechCrunch, Mar 2026)](https://techcrunch.com/2026/03/25/reddit-bots-new-human-verification-requirements)
