# V3 metric catalogue — full candidate pool

Companion to `V3_PLAN.md`. That document's §4/§5 are the *curated* metric
list — what survived my own judgment when I first wrote it. This document is
the wider net: every candidate metric across all four levels, tagged by
whether it's directly available, needs a proxy, needs composition from
multiple pieces, or is genuinely unrecoverable. Curation (which of these
actually separate a population) happens later, in Stage 1–2 of §7 — that's
the explicit point of this document: don't let my judgment filter the
candidate pool before the data gets a chance to.

**Availability tags**
✅ **Direct** — a raw field, confirmed populated in the audit, no work needed
🔧 **Proxy** — the exact thing isn't available; an available substitute
approximates the same construct
🧩 **Composite** — built by combining ≥2 available metrics/fields
❌ **Excluded** — no viable proxy exists; confirmed absent, not just unbuilt

**Temporal tag** — since the deliverable is a 12-month + ongoing tracker, every
viable metric gets a **T** marker if a trend variant applies: MoM delta,
trailing baseline, and — per the r/ipl lesson (§11 of the plan) — a
**YoY same-month baseline**, never a naive trailing average alone for anything
that might be seasonal.

---

## 1. Subreddit level

| Metric | Tag | Source / method | Temporal |
|---|---|---|---|
| Subscriber count | ✅ | `/time_series/r/<sub>/subscribers` | T |
| Creation date | ✅ | subreddit object `created_utc` — immutable once set, so valid despite the object's 536-day staleness on *other* fields (subscriber count etc.) | — |
| Sub age at observation month | 🧩 | observation month − creation date | T |
| NSFW flag | ✅ | `over_18` | — |
| Quarantined flag | ✅ | `quarantine` | T (rare but real transitions) |
| Posting frequency (posts, comments/month) | ✅ | `/time_series` (S1 in plan) | T |
| Moderator count | ❌ | no endpoint anywhere returns a mod list. **No viable proxy for the count itself.** | — |
| Moderator *intensity* (substitute construct) | 🔧 | `distinguished`/`stickied`/`locked` rates from posts+comments — proxies moderation *activity level*, not headcount. Already S6 in the plan; the distinction from "count" needs to stay explicit so it's never read as the same thing. | T |
| Rules — count | 🧩 | `/subreddits/rules`, confirmed working (r/india → 14 rules w/ `created_utc`) | T (rule additions are events) |
| Rules — recency of last change | 🧩 | max `created_utc` across a sub's rules — a sudden new rule is itself a signal (mod response to an incident) | T |
| Flair-scheme richness | 🧩 | distinct `link_flair_text` values used per sub-month, and % of posts carrying flair — aggregated up from post level | T |
| Cross-post ratio (inbound) | 🧩 | % of a sub's own posts that are themselves crossposts (`crosspost_parent_id` populated) | T |
| Cross-post ratio (outbound / fan-out) | 🧩 | sum of `num_crossposts` across a sub's posts / post count — how often this sub's content propagates elsewhere | T |
| MoM post/comment count fluctuation | ✅ | S1, raw MoM delta | T |
| **Sub-month series bimodality / regime-switching** | 🧩 *(new protocol, not just a metric)* | Apply the Stage-1 GMM/BIC test **to a sub's own monthly count series**, not just to account features. A sub whose monthly activity clusters into two regimes (baseline vs. spike months) is showing structure the account-level tests can't see. This is the literal thing you asked about — "fluctuation... bimodal distribution tells there may be a boundary" — and it wasn't in the plan as a distinct test before now. | T, by construction |

---

## 2. Post level

| Metric | Tag | Source / method | Temporal |
|---|---|---|---|
| Title length | ✅ | `len(title)` — was in the throwaway pilot script, never promoted to the written plan | — |
| Body length | ✅ | `len(selftext)` — same gap | — |
| Title:body length ratio | 🧩 | short punchy title + no body is a link/reaction-bait signature | — |
| Score | ✅ | P1 | T |
| Upvote ratio | ✅ | P1 | T |
| Awards | ❌ | field is dead platform-wide (Reddit retired awards ~2023; confirmed 0 non-zero across 21,691 sampled objects). **Not a data gap — a real absence. No proxy needed or possible.** | — |
| Flair | ✅ | P15 | — |
| Link vs. self post | ✅ | `is_self` — raw field exists, was never promoted to a named metric | — |
| Domain | ✅ | P14, aggregated | T |
| Posting time (raw) | ✅ | `created_utc` → hour/weekday | — |
| **Posting time vs. the sub's own modal posting hour** | 🧩 | this post's hour compared to that sub-month's modal/mean posting hour — the "vs mode" framing generalized from account level to post level | T |
| Edit status | 🔧 | `_meta.is_edited` — **not** the top-level `edited` field, which is ~0% populated (confirmed dead) | T |
| Edit timing | 🧩 | edited before the T+16s capture vs. between the two snapshots (`_meta.was_deleted_later`-style logic) — tells you whether an edit happened before or after the post accrued most of its score | — |
| Crossposts (count) | ✅ | `num_crossposts`, raw field captured, never made a named metric | T |
| Crosspost fan-out (which subs) | 🧩 | same-URL search across subs — was in the earlier draft catalogue, dropped from the final plan. Reinstating. | T |
| Deletion status | ✅ | §6 labels — used as target, not currently as a standalone engagement metric | T |
| **Engagement velocity** | 🔧 | score / hours-since-post, computed **at the fixed T+36h capture** — this is a proxy, not a true trajectory. No score history exists (confirmed: two snapshots, one score). Must be labelled a snapshot-velocity proxy, never described as real-time velocity. | — |
| Score per subscriber | ✅ | P6 | T |
| Score per word | 🧩 | content-length-normalized engagement efficiency | — |
| Comment:vote ratio | ✅ | P4 | T |
| Implied vote volume / contested share | ✅ | P2/P3 | T |

---

## 3. Comment level

| Metric | Tag | Source / method | Temporal |
|---|---|---|---|
| Depth | 🔧 | not directly returned by `/comments/search` (confirmed); reconstruct by walking `parent_id` chains, which are 100% populated | — |
| Parent type (top-level vs. reply) | ✅ | `parent_id` prefix `t3_`/`t1_` | — |
| Score | ✅ | raw field | — |
| Awards | ❌ | dead field, same as posts | — |
| Edit / deletion | ✅ | `_meta.removal_type` + `was_deleted_later` + the unlabelled-tombstone triple (`author=[deleted]`, `body=[removed]`, `collapsed_reason_code=DELETED`) — already well covered in §3 of the plan | T |
| **Response latency** | ✅ | `comment.created_utc − parent.created_utc`, per comment. Was in the earlier draft, **dropped from the final plan**. Reinstating — this is a genuinely strong candidate (a floor exists for humans; scripts don't have one). | — |
| **Sentiment / toxicity** | 🔧 | **Completely absent from the current plan** — was `C6` in the old plan, silently dropped in the rewrite. No raw field; must be computed from `body` text. Two real options, both confirmed available in this environment: `nltk` (VADER, lightweight, rule-based) as the cheap bulk default, or `transformers` (a real classifier) if quality matters more than throughput at scale. **Caveat that must travel with this metric**: VADER's lexicon is English-word-based and will degrade on Hinglish/code-mixed text — the same risk already flagged in the plan (§9) for LLM-perplexity detectors. Don't trust it uncalibrated on this corpus. | T |
| Repeated text | ✅ | A16, A17, P13, B3 — already well covered | T |
| **Link density** | 🧩 | URL count in `body` / body length (or / word count) — distinct from A22's account-level URL *rate*; this is the per-comment normalized version, not currently in the plan | — |

---

## 4. Account level

| Metric | Tag | Source / method | Temporal |
|---|---|---|---|
| Age | ✅ | base36 proxy on `author_fullname` (AUC 0.986, already validated) | — |
| Karma (post/comment) | ✅ | A5 | T |
| **Posting frequency (raw count, not karma)** | 🧩 | comments+posts per account-age-day, from the same `/comments/search?author=` pull already planned for timing features. **V2 had `comments_per_day`; it was dropped going into V3.** This is a straight regression, not a new idea — needs reinstating. | T |
| Active hours | ✅ | A10/A11 | T |
| Subreddit diversity | ✅ | A27 | T |
| **Comment:post count ratio** | 🧩 | `n_comments / n_posts` from collected history — **distinct from A5's karma-type ratio**, which the current plan conflates this with. Needs its own line. | T |
| Language consistency | ✅ | A19, partial | — |
| Client/app source | ❌ | never exposed by any endpoint, confirmed. No proxy exists — nothing in the data hints at posting client. Genuinely excluded. | — |
| Suspension/deletion status | ✅ | §6, used as label | T |

### The "vs. mode" family — genuinely new, not currently in the plan

Every ratio below is the account's raw value **expressed against the modal value of the reference population** (that sub-month's active accounts, or the control tier) — not just the raw ratio itself. This is a different feature than the plain version and needs to be listed separately:

| Metric | Tag |
|---|---|
| Karma/account-age-day vs. population mode | 🧩 |
| Comments/account-age-day vs. population mode | 🧩 |
| Raw posting frequency vs. population mode | 🧩 |
| Reply latency (median) vs. population mode | 🧩 |
| Score-received-per-comment vs. population mode | 🧩 |

---

## 5. Cross-level composites — the explicit analysis protocol you asked about

This is the piece that was genuinely missing as a *method*, not just as a metric: segmenting **who** engages, conditioned on a **post-level** regime, rather than treating account and post features as separate pools.

| Protocol | What it does |
|---|---|
| **Commenter-profile segmentation by comment:vote-ratio tercile** | Split posts into top/bottom tercile of P4 (comment:vote ratio). Compare the *account-level* feature distributions (age, karma/day, circadian entropy, etc.) of their commenters between terciles. This is literally what you described as missing in the Y/N round. |
| **Commenter-profile segmentation by contested_share tercile** | Same split, using P3 (up:down ratio) instead of comment:vote ratio — tests whether *controversial* posts draw a different engager population than *lopsided* ones. |
| **Commenter-profile segmentation by sub-month regime** | Using the new sub-level bimodality test (§1 above): do commenter profiles differ between a sub's "baseline" months and its "spike" months? |

These are Stage-2 bivariate protocol additions, not single metrics — they belong in §7 of `V3_PLAN.md` as explicit steps, not in the metric table.

---

## 6. Temporal layer — applies to every viable metric above

For each metric tagged ✅/🔧/🧩 with a **T**, generate:

1. Point-in-month value
2. MoM delta
3. Trailing-N-month baseline (N=3 or 6)
4. **YoY same-month baseline** — the default comparison for anything with seasonal risk, per the r/ipl lesson in §11 of the plan. Trailing average alone is not safe until seasonality is ruled out per sub.
5. Where the sub-level bimodality test (§1) applies: regime label (baseline/spike) for that month

---

## 7. Explicitly excluded — confirmed absent, not just unbuilt

| Item | Why |
|---|---|
| Moderator count/list | No endpoint returns it. Mod *activity rate* (§1) is a proxy for intensity, not count — keep that distinction honest, don't conflate them. |
| Awards | Dead platform-wide since ~2023, confirmed 0 non-zero across every sample. Not a collection gap. |
| Client/app source | Never exposed by any endpoint. |
| Exact account creation date | Base36 proxy substitutes (AUC 0.986, already validated) — this is the one exclusion with a strong proxy, listed here only for completeness. |
| True score trajectory / vote velocity | Only two snapshots exist (T+16s, T+36h). The engagement-velocity proxy above approximates it at a single fixed age; it is not a curve. |

---

## Summary

- **13 items** confirmed missing from `V3_PLAN.md` against your original four-list spec (per the Y/N audit) — all present in this catalogue now, either directly or via a labelled proxy/composite.
- **3 new cross-level analysis protocols** (§5) that were never going to appear as "metrics" because they're comparison methods, not fields — these need to go into §7 of the plan, not §4.
- **1 new class of feature** — the "vs. population mode" family (§4) — generalized from your original karma/day and comments/day framing to five metrics, not just the two you named.
- **1 new sub-level test** (§1, sub-month bimodality) — directly answers your "fluctuation... bimodal distribution" question, and it wasn't a distinct test anywhere in the plan before this pass.
- **4 genuine exclusions** (§7), each with a stated reason, none of them silent.

Next: fold the surviving items back into `V3_PLAN.md` §4/§5/§7, and flag the account-level `comments_per_day` drop explicitly as a regression from V2 in §7 of the plan (Drop from V2 lists what was correctly cut; this one should not have been cut).
