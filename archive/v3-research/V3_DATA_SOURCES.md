# V3 data sources — user-level features and network

Research note, 2026-07-31. Every endpoint below was tested live against
post IDs and account names already in `data/v2/`, not read off documentation.
Numbers in this doc are measured, not estimated, unless labelled otherwise.

**Bottom line:** Arctic Shift (`arctic-shift.photon-reddit.com`) covers
essentially every gap. It is free, unauthenticated, indexes Reddit through
today, supports date-bounded queries, and — critically — is keyed by the same
`post_id` / `author` strings already sitting in our CSVs. Nothing needs to be
re-collected to use it. The 14 months of existing data become the join key.

---

## 1. Why the current collection can't answer user-level or network questions

Not a criticism of the design — these are consequences of the endpoints
`collect_data_v2.py` uses, and they cap what any model fitted on this corpus
can achieve.

| Limit | Where it comes from | Consequence |
|---|---|---|
| 30 comments/post, depth ≤2, top-10 posts only | `fetch_comments()` | Measured 10.0 comment rows per sampled post. 8,642 of 12,125 posts have **zero** comment data. |
| No comment `body` stored | `COMM_COLS` | No linguistic analysis possible at all — repetition, templating, near-duplicate replies. |
| No `parent_id` stored | `COMM_COLS` | No reply graph. "Network" is currently set-overlap, not a graph. |
| Account features = `about.json` only | `fetch_user()` | Age, karma, verified email. Nothing about *behaviour*. |
| `n_posts`/`n_comments` are sample-local | acknowledged in `anomaly_detection.py:160` | The model's activity features describe our top-40 slice, not the account. |
| One score snapshot per post | `collected_utc` | No diffusion curve. Can't distinguish fast organic from fast coordinated. |
| No `url`/`domain`/`selftext`/flair | `POST_COLS` | Can't trace a story across subreddits. |

The AUC ≈ 0.66 is the honest ceiling of a model built on account age, karma
ratios, and a sampled activity proxy. The features that would actually
separate coordinated from organic behaviour — timing regularity, co-appearance,
reply structure, text reuse — are not in the corpus yet.

---

## 2. Tested inventory

Base: `https://arctic-shift.photon-reddit.com/api`. No key, no OAuth.
Measured ~266 req/min sustained with no throttling applied (Reddit's own OAuth
ceiling is 100/min). It's a free community service — I'd still throttle to
~60/min out of courtesy.

### Working, verified

| Endpoint | Cap | Verified result |
|---|---|---|
| `/posts/ids?ids=` | 500 IDs/req | Full post object — `selftext`, `url`, `domain`, `link_flair_text`, `crosspost_parent_id`, `removed_by_category`, `upvote_ratio`, `num_crossposts`. |
| `/comments/tree?link_id={post_id}` | 25,000 | **20.3× more comment rows than we hold today** (see §3). Includes `body`, `parent_id`, `controversiality`, `is_submitter`. |
| `/comments/search?author=&after=&before=` | 100/page | Full site-wide comment history per account, date-bounded, with body. |
| `/posts/search?url=&url_exact=true` | 100 | Every Reddit post sharing a URL — the cross-subreddit propagation trace. |
| `/users/interactions/users?author=&after=&before=` | — | Ranked co-interaction counts. **Date filtering confirmed working.** |
| `/users/interactions/users/list?author=` | — | Individual timestamped, typed edges (`"commented under users post"`). |
| `/users/interactions/subreddits?author=&after=&before=` | — | Subreddit footprint per account per window. |

### Not usable

- **`/*/search/aggregate`** — returns `422 "Timeout. Maybe slow down a bit"`
  consistently, including after a 45s cooldown on a single-month window. Would
  have been the cheap way to get true per-account activity counts. Compute the
  same thing client-side by paging `/comments/search` instead.

### Freshness caveat, matters for the monthly run

Arctic Shift indexes to the current day (confirmed: r/india posts from
2026-07-31 present), but **scores on fresh content are stubs** — the three
posts I pulled from today all read `score: 1`. Their docs say ~36h to full
update. So:

> Arctic Shift for structure, history, text, and graph.
> Live Reddit API for current-month scores.

Don't swap `collect_data_v2.py`'s score collection over to it.

### Other sources considered

- **PullPush.io** — Pushshift-schema drop-in, but documented coverage gaps
  after 2023, 30 req/min, frequent outages. Keep as fallback only.
- **Academic Torrents monthly dumps** — the same project's bulk export.
  Hundreds of GB, weeks of lag. Right answer if this ever needs a frozen,
  citable corpus; wrong answer for a monthly pipeline.
- **Reddit native API** — still needed for live scores and account status.
  Worth noting `reddit_auth.py:82` throttles to 54 req/min against an assumed
  60/min ceiling; the current documented OAuth limit is **100/min averaged over
  10 minutes**. Roughly a 2× speedup on collection for a one-line change,
  though I'd verify against the `X-Ratelimit-*` response headers before relying
  on it rather than trusting a blog post.

---

## 3. Backward compatibility — measured

The corpus: **14 months, 12,125 posts, 31,075 unique accounts, 34,963 comment
rows.** All of it enrichable in place.

Comment-tree backfill, 12 randomly sampled posts spanning 2025-09 → 2026-07:

```
post_id     month     num_cmts   ours  arctic  uniq_authors
1qchyle     2026-01        161     11     166           136
1uda9i0     2026-06        198     10     201           131
1nwe845     2025-10        160     10     169            61
1umdev2     2026-07        255     11     266           177
1v65vu8     2026-07        400     11     411           231
1nruq10     2025-09        459     10     482           260
...
12 posts | ours 109 rows -> arctic 2,210 rows (20.3x)
```

Two things worth reading off that table. Trees are **complete**, not sampled —
`arctic` tracks `num_cmts` closely. And where `arctic` *exceeds* `num_cmts`
(482 vs 459), it's returning deleted/removed comments Reddit no longer counts.
Deleted-comment rate is itself a signal we currently can't see.

Extrapolated backfill cost at a polite 60 req/min:

| Job | Requests | Wall time |
|---|---|---|
| Re-fetch all 12,125 posts (batch 500) | 25 | seconds |
| Comment trees for all 12,125 posts | 12,125 | ~3.4 h |
| Interaction graph, 31,075 accounts × 2 | 62,150 | ~17 h |
| Full comment history, 31k accounts | ~150,000+ | ~40 h |

The first three are a weekend. Checkpoint them the way
`collect_data_v2.py` already does and they're resumable. The fourth I'd scope
to accounts that actually matter — top risk-decile plus anything appearing in
≥2 subreddits, maybe 3–5k accounts — rather than all 31k.

---

## 4. What this unlocks — user level

Currently 10 features, all essentially "who is this account on paper." These
are "what does this account *do*," which is where coordination actually shows.

**From `/comments/search?author=` (full history, date-bounded):**

- **Inter-post interval entropy.** Humans are bursty and irregular. Scripted
  accounts cluster around a scheduler. Low entropy on a long history is one of
  the strongest single bot signals in the literature and we have zero coverage
  of it today.
- **Circadian fingerprint.** 24-bin UTC hour histogram. A human sleeps —
  typically 6+ near-dead hours. Sustained 24/7 activity is anomalous. Doubles
  as a crude timezone check: an account posting exclusively in IST daylight vs.
  one whose activity centres on a different timezone while posting only to
  Indian subs.
- **True site-wide activity rate.** Directly fixes the caveat at
  `anomaly_detection.py:160` — `comments_per_day` becomes real rather than a
  sample-local proxy.
- **Subreddit entropy and topical coherence.** From
  `/users/interactions/subreddits`. Single-topic accounts with no unrelated
  hobby subs behave differently from real users.
- **Account trajectory.** First-activity date vs. account-creation date — the
  dormancy gap. Aged accounts bought and reactivated show a long silence then a
  sudden onset. This is a classic purchased-account signature and it's a
  two-column calculation once we have history.

**From comment bodies (new):**

- Self-similarity — mean pairwise similarity of an account's own comments.
- Cross-account template reuse — near-duplicate comments across *different*
  accounts within a window. Currently we compute near-duplicate *titles* only,
  and `analyze_network()` explicitly excludes even that from scoring.
- Length/vocabulary distribution, copypasta detection.

**A better label — the highest-leverage item here.** The model is fit against
account suspension, "a weak but real proxy." `removed_by_category` is populated
in Arctic Shift post objects. Sampling 100 r/india posts: 72 `moderator`, 2
`reddit`, 2 `deleted`. That `reddit` category is an **admin-level action —
Reddit's own anti-spam classifier firing.** It's a far more direct label for
"this is a bot" than "this account is gone for any reason including quitting."
Combining suspension with admin-removal, and treating moderator-removal as a
separate weaker signal, is more likely to move AUC than any feature above.
Worth testing before building anything else.

---

## 5. What this unlocks — network

`analyze_network()` today is Jaccard title overlap, cross-sub account overlap,
and a Gini on scores. That's set arithmetic, not network analysis — no graph is
ever constructed. With `parent_id` from comment trees and the interactions
endpoints, real structure becomes available:

- **Directed reply graph** per sub-month. Nodes = accounts, edges = A replied
  to B. Then the standard toolkit: degree distribution, reciprocity,
  clustering coefficient, connected components, community detection.
- **Bipartite co-appearance with proper significance.** `analyze_cooccurrence()`
  counts repeat pairs, but against no null model. With full trees, the right
  test is: given each account's posting volume and each thread's size, how
  often would this pair co-appear by chance? Coordinated clusters are the pairs
  whose co-appearance is many standard deviations above that null. This is the
  difference between "these two accounts both like r/india" and "these two
  accounts appear together far more than volume explains."
- **Rapid-response clusters.** Accounts that repeatedly arrive within N seconds
  of each other on the same thread. Needs full trees with timestamps —
  buildable retroactively, and probably the most direct coordination evidence
  available without vote data.
- **Reply-target concentration.** An account whose replies are disproportionately
  aimed at one other account, or one small set — amplification and brigading
  both leave this trace.

`/users/interactions/users?author=X&after=&before=` returns this pre-aggregated
per account, so the month-scoped ego network is one request. Confirmed working
with date bounds.

---

## 6. On the propagation-graph proposal

The framing — "which *stage* of diffusion was unusual" rather than "was this
manipulated" — is right, and it's a genuine improvement over a single composite
score. But the specific chain has stages that are not observable, and the
distinction matters before anyone builds toward it.

| Stage | Observable? | How |
|---|---|---|
| News published | ✅ | GDELT DOC 2.0 API — free, no key, verified working with date windows and `sourcecountry:india`. |
| First Reddit post | ✅ | `/posts/search?url=&url_exact=true&sort=asc`. |
| **First 100 voters** | ❌ **Never** | Reddit has never exposed voter identity through any API, and won't. Not a gap to engineer around — the stage cannot be built. |
| Early commenters | ✅ | Comment trees, sorted by `created_utc`. Retroactive. |
| Cross-subreddit spread | ✅ | Same URL search + `crosspost_parent_id`. |
| **Recommendation boost** | ❌ | No endpoint exposes ranking or feed placement. Only inferable as a residual — "growth unexplained by observed comment/crosspost activity" — which is an assumption, not a measurement. |
| Mainstream pickup | ✅ | GDELT, timestamped, comparable against the Reddit timeline. |
| Secondary Reddit wave | ✅ | Same URL search, later time window. |

So five of eight stages are directly measurable and four of those are
measurable **retroactively on the existing corpus**. That's a real system. It
just shouldn't be described as reconstructing vote propagation, because the
vote layer is exactly the part that's invisible.

Working proof of concept, run against a post already in `data/v2/`:

```
STORY : "No Left govt left in India for first time in 50 years"
URL   : deccanherald.com/elections/kerala-assembly-elections-results-2026...

  +0.00h  r/india          score=2638  cmts=208  u/godblessthegays
  +1.45h  r/IndiaSpeaks    score=15    cmts=1    u/hrpanjwani
  +4.31h  r/worldnews      score=5     cmts=3    u/LookNoRook
```

Three sub-second requests, no auth, and the shape of that trace is already
interpretable — one dominant post, weak secondary pickup, which is what organic
looks like. The baseline distribution of these shapes across thousands of
stories is what makes an individual one anomalous.

One caveat on scope: **37 of 400** sampled posts were external-link posts. The
propagation graph only applies to link posts, so it's a lens on roughly a tenth
of the corpus, not a replacement for the current per-subreddit score.

**The one thing that cannot be backfilled: score trajectory.** Early vote
velocity — the single best proxy for the unobservable voter layer — requires
polling a post repeatedly while it's live. Arctic Shift keeps one snapshot;
so do we. If diffusion curves matter, a new collection mode is needed: poll
`/r/{sub}/new` every ~15 min, then re-poll each post at fixed ages (+10m, +1h,
+6h, +24h, +72h). Cheap, but only ever works forward. Worth starting soon
precisely because it's the one thing that gets more valuable the earlier it
begins.

---

## 7. On the 3/10 for "proving coordinated influence"

That score is fair, and the enrichment above will move it — but not to 9, and
it's worth being clear about why, since the README's honesty about AUC 0.66 is
one of this project's better features and shouldn't be undermined by V3
optimism.

The ceiling isn't features. It's that **coordination has no ground truth
available to us.** There is no labelled set of "these 400 accounts were a
coordinated campaign" for Indian subreddits. Suspension and admin-removal are
proxies — better ones once combined, still proxies. Any model here is
calibrated against "Reddit eventually acted on this account," which correlates
with coordination but is not it.

What the enrichment realistically buys:

- **Coordination detection: 3 → 6.** Co-appearance significance against a null
  model, timing correlation, and text reuse are *direct* structural evidence,
  not account-level circumstantial correlates. That's a real category change.
  It won't reach "proof" because behavioural evidence never does.
- **Anomaly detection: 8 → 8.5.** Already the system's strength.
- **Investigative usefulness: 8 → 9.** The biggest practical gain. Right now
  the dashboard says "r/X scored 63 this month." With a reply graph it can say
  "these 14 accounts formed an unusually dense cluster across these 3 threads,
  here are the threads." That is something a person can actually check.

The evaluation questions in the review are the right ones and several are now
testable. Two are worth prioritising because they're falsifiable and cheap:

1. **Does the anomaly score predict later mainstream coverage?** GDELT + our
   existing 14 months. This is a genuine forward-looking test in the same
   spirit as `backtest_predictive.py`, and a null result is publishable — it
   would mean the score tracks Reddit-internal dynamics rather than news
   salience, which is useful to know either way.
2. **Are the same accounts repeatedly involved?** Directly answerable from the
   existing corpus once account histories are joined across months. Currently
   unanswered, and cheap.

On cross-platform: Bluesky is free and open (AT Protocol, no auth for public
search) and worth a look. Telegram public channels are accessible. X is ~$200/mo
and hostile. Facebook is effectively closed outside Meta's academic programme.
I'd treat all of it as a separate project — the Reddit-side work above is
higher value per unit effort and doesn't depend on it.

---

## 8. Suggested sequencing

Ordered by value per unit of work, and structured so each step is independently
useful if the next never happens.

1. **Test the label first.** Pull `removed_by_category` for the existing 12,125
   posts (25 requests, seconds). Refit with admin-removal folded into the label
   via `--refit-only`. If AUC doesn't move, that's worth knowing before
   investing 20 hours of collection. This project's own bar — an unambiguous
   backtest win before shipping — applies here.
2. **Post re-fetch.** 25 requests. Adds `url`, `domain`, `selftext`, flair,
   `crosspost_parent_id`, `num_crossposts`, true `upvote_ratio` to all 14
   months. Enables §6 immediately.
3. **Comment-tree backfill.** ~3.4h, checkpointed. 20× the comment data,
   `parent_id` and `body` for the first time. Unblocks everything in §5.
4. **Interaction graph** for top-risk + multi-sub accounts. Build the reply
   graph, implement co-appearance significance against a volume-matched null.
5. **Score-trajectory poller.** New forward-only collection mode. Start early;
   it accrues.
6. **GDELT join** and the two evaluation tests above.

Steps 1–3 are the ones I'd actually commit to now. They're bounded, they're
retroactive, and step 1 might reshape the priority of everything after it.

---

## 9. The stronger option: stop sampling, take a census

Everything above treats the existing collection shape as fixed and enriches it.
That's the conservative read. The more useful question is whether the old
schema should constrain the new score at all — and once the volume is measured,
the answer is clearly no.

### The current corpus is a 0.45% sample

Measured via `/posts/search` pagination over one week, extrapolated:

| Subreddit | Actual posts/month | We keep | Coverage |
|---|---|---|---|
| r/india | ~8,950 | 40 | **0.45%** |
| r/IndiaSpeaks | ~1,875 | 40 | **2.1%** |

Comments are starker. One day of r/india: **2,141 comments from 958 unique
authors** — roughly 64,000 comments/month in one subreddit. The entire 14-month
corpus across all 25 subreddits currently holds 34,963 comment rows. We hold
less comment data for 25 subreddits over 14 months than r/india generates in
three weeks.

### A full census is a laptop job

Pagination measured at **339 comments/sec** — 22 pages in 6 seconds. Across 25
subs × 14 months, assuming the average sub runs ~25% of r/india's volume:

```
~5.6M comment rows  |  ~4.6h collection  |  ~160x current comment coverage
~780k posts         |                    |  ~65x current post coverage
```

That is not a research-infrastructure problem. It's an afternoon and a few
hundred MB of parquet.

**One practical caveat:** pulling 5.6M rows through a free community API is
impolite regardless of whether the rate limiter allows it. The right approach
is Academic Torrents dumps for the historical bulk — that is precisely what
they exist for — and the API for incremental monthly collection. Same data,
same project, appropriate channel.

### What a census fixes

The README and code comments document a long series of careful workarounds.
Reading them together, they are almost all patches for one root cause: the
sample is score-selected and tiny.

| Existing workaround | Why it exists | Under a census |
|---|---|---|
| Month-relative risk threshold | Fixed threshold produced a spurious 13-month climb | Population statistics are directly computable — no relative hack needed |
| `decay_slope` replacing `score_cv` | Top-N truncation destroys score variance | Full score distribution is observable; measure it directly |
| `/new.json` random supplement | Survivorship bias; only works forward | Census has no survivorship bias, and applies retroactively |
| `n_posts` is 75% zero | Being a top-40 author is rare | Real per-account posting counts |
| `comments_per_day` is "a narrower proxy, not a real site-wide rate" (`anomaly_detection.py:160`) | Sample-local activity | Actual activity rates |

Each of those was the right call given the data. None of them is necessary
with a census. That's the real argument for the rebuild — not "more data is
better," but "a specific list of known-acknowledged compromises all resolve at
once."

### Labels stop being scarce

This is the part that actually moves AUC. Current labelling is live
suspension-checking — one API call per account, hours per thousand, which is
why `scale_weak_labels.py` is checkpointed and run rarely.

Census data carries labels **for free, at population scale**. Measured on
real slices:

- Posts: 4% `removed_by_category == 'reddit'` — an *admin* action, i.e.
  Reddit's own anti-spam system firing. ~57% moderator-removed.
- Comments: **19% `[removed]`**, plus 20% with `author == '[deleted]'`.

At census scale that's on the order of a million removal events across the
corpus, at zero marginal API cost. Suspension-checking remains valuable as an
independent label, but it stops being the only one — and a multi-label fit
(admin removal, mod removal, suspension, deletion) against real population
features is a fundamentally better-posed problem than one weak binary proxy
against ten paper features.

### What the score could become

The current score is a share: *% of activity from accounts in this month's top
risk decile*. It's relative by construction, has no null model, and inherits
whatever bias the sample carries.

With a census, a better-posed score is available — two layers:

**Layer 1 — account risk.** Same logistic idea, but fitted on full-history
behavioural features (interval entropy, circadian coverage, dormancy gap,
subreddit entropy, text self-similarity) against the multi-label target above,
on a representative population sample rather than a top-40-visible one.

**Layer 2 — coordination structure.** This is the genuinely new part, and it's
only possible with complete threads. For each (sub, month), compute
co-appearance excess, reply-timing synchrony, and cross-account text reuse —
each measured **against a volume-matched permutation null**. Given each
account's actual comment volume and each thread's actual size, how often would
this cluster co-occur by chance? Coordinated groups are the ones sitting many
standard deviations above that.

The published score then becomes: *the share of this sub-month's total activity
occurring inside clusters that pass a significance threshold against the null.*

Both numerator and denominator mean something. The denominator is real total
activity, not a sampled slice. The numerator is a significance test, not a
hand-set decile. And it's **auditable in a way the current score is not** —
every point of score traces back to specific accounts in specific threads that
a person can open and judge.

### What it does not fix

Worth stating plainly, because the temptation with a 160× data increase is to
assume everything scales with it.

A census fixes sampling bias (completely), feature quality (dramatically),
label quantity (dramatically), and makes null models possible at all. It does
**not** create ground truth. There is still no labelled set of "these accounts
were a coordinated campaign." Removal and suspension remain proxies — better,
more numerous, multi-signal proxies, still proxies. The "proving coordinated
influence" score moves because structural co-appearance evidence is
categorically stronger than account-level circumstantial correlates, not
because the corpus got bigger.

### Engineering cost

Real, but modest. At ~5.6M rows the pipeline stops being an in-memory pandas
job — parquet on disk with DuckDB over it handles this size without
complaint, and the permutation nulls in Layer 2 are the actual compute cost,
not the I/O. The per-sub-month structure of the existing code survives; the
storage layer underneath it changes.

### Honest sequencing given this

The §8 plan is still correct as *low-risk incremental* work, and step 1 (the
label test) is worth doing regardless — it's 25 requests and it informs
everything. But if the goal is a materially better score rather than a better-
instrumented version of the current one, the census is the higher-value path
and the two are somewhat independent. I'd suggest:

1. Label test on existing data (25 requests, minutes) — cheap, informative.
2. **One-subreddit census pilot** — pull r/india complete for 3 months
   (~30 min), rebuild Layer 1 features and the Layer 2 null model on it, and
   compare against the current score for the same sub-months. This is the
   decision point: if the census score disagrees with the current score, look
   at which one the evidence supports before committing 5 hours and a rewrite.
3. Full census + V3 scoring, if the pilot holds up.

Step 2 is the one that answers this question properly, and it costs an
afternoon.

---

## Sources

- [Arctic Shift API reference](https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md)
- [Arctic Shift project](https://github.com/ArthurHeitmann/arctic_shift)
- [Best Pushshift Alternatives 2026 — PullPush, Arctic Shift, Dumps](https://www.redditapis.com/blogs/best-pushshift-alternatives-2026)
- [Pushshift Alternative 2026 (PullPush, Arctic Shift) — THINKPOL](https://think-pol.com/pushshift-alternative)
- [How to Get Historical Reddit Data After Pushshift (2026)](https://www.xpoz.ai/blog/tutorials/how-to-get-historical-reddit-data-after-pushshift/)
- [Reddit API in 2026: Pricing, Rate Limits & What Works](https://www.socialcrawl.dev/blog/reddit-data-api-2026)
- [GDELT DOC 2.0 API](https://api.gdeltproject.org/api/v2/doc/doc)
