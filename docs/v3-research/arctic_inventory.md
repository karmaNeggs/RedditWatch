# Arctic Shift API — Empirical Capability Audit

**Base URL:** `https://arctic-shift.photon-reddit.com/api`
**Audit date:** 2026-08-04 · **Method:** live `python3`/urllib calls, no API key, ~1 req/s politeness except in the dedicated throughput tests.
**Sample sizes:** 7,064 posts + 14,627 comments (unbiased *census* of 5 subs, live era) · 2,167 posts + 2,152 comments (stratified 2016→2026, era comparison) · 343 subreddit objects · 1,194 user objects · 43 Indian subreddits' full monthly volume series.

> **Headline:** Arctic Shift is a live, near-real-time mirror (current to the minute of this audit) and is materially *richer* than a plain Pushshift clone — it carries an undocumented `_meta` block with removal/edit provenance, a `/time_series` endpoint with historical subscriber curves, and account-level karma aggregates. Three things will bite you: (1) the **`after` cursor is exclusive at second granularity** and silently drops rows; (2) **`removed_by_category` is never set on comments** — removal detection needs `_meta.removal_type`; (3) **the entire rich-metadata layer only exists from ~2023-07 onward**.

---

## 1. Endpoint reference

Verified by direct call. "auto" = the `limit=auto` mode, which returned **651–668 rows/page** consistently (server picks by load).

| Endpoint | Status | Required | Key optional params | Page cap | Sort | Date filter | Notes |
|---|---|---|---|---|---|---|---|
| `/posts/search` | ✅ | — (in practice `subreddit` or `author`) | `author`, `subreddit`, `title`, `selftext`, `query`, `url`, `url_exact`, `link_flair_text`, `author_flair_text`, `over_18`, `spoiler`, `crosspost_parent_id`, `md2html`, `fields` | `limit` 1–100, or `auto` → **~660** | `asc`/`desc` on `created_utc` only | ✅ `after`/`before` | Main bulk path for posts |
| `/comments/search` | ✅ | — | `author`, `subreddit`, `body`, `link_id`, `parent_id`, `md2html`, `fields` | 1–100 or `auto` → **~660** | `asc`/`desc` | ✅ | **Main bulk path.** `parent_id=` empty ⇒ top-level only |
| `/posts/ids` | ✅ | `ids` | `md2html`, `fields` | **500 ids** (100 verified OK) | n/a | n/a | Returns same stored record as search |
| `/comments/ids` | ✅ | `ids` | same | 500 | n/a | n/a | |
| `/subreddits/ids` | ✅ | `ids` | | 500 | n/a | n/a | Takes bare `2qh1q` |
| `/users/ids` | ✅ | `ids` (`t2_…`) | | 100/call verified | n/a | n/a | **299/300 resolve.** Returns `_meta` karma aggregates |
| `/comments/tree` | ✅ | `link_id` | `parent_id`, `limit` (1–25000), `start_breadth`, `start_depth`, `md2html` | 25000 | n/a | n/a | `start_breadth=0&start_depth=0&limit=25000` ⇒ **zero `more` stubs** |
| `/subreddits/search` | ✅ | — | `subreddit`, `subreddit_prefix`, `min_subscribers`, `max_subscribers`, `over18`, `sort_type` (`created_utc`/`subscribers`/`subreddit`) | **1–1000** | `asc`/`desc` | ✅ on sub creation | ⚠️ `subreddit` does **not** accept a comma list (`must be 2-30 characters`) — one call per sub |
| `/subreddits/rules` | ✅ | `subreddits` | | 1000 | n/a | n/a | r/india → 14 rules w/ `created_utc`, `kind`, `description` |
| `/subreddits/wikis` | ✅ | `subreddit` or `paths` | `limit` ≤100 | 100 | n/a | n/a | Returns `content`, `revision_author`, `revision_date` |
| `/subreddits/wikis/list` | ✅ | `subreddit` | | | n/a | n/a | r/india → 20 paths |
| `/users/search` | ✅ | — | `author`, `author_prefix`, `min_num_posts`, `min_num_comments`, `min_karma`, `active_since`, `sort_type` (`author`/`total_karma`) | 1–1000 | `asc`/`desc` | ✅ `active_since` | ⚠️ single `author` only, no comma list. `author_prefix` **timed out** |
| `/users/interactions/users` | ⚠️ | `author` | `subreddit`, `after`, `before`, `min_count`, `limit` | no limit | | ✅ | 1.1–3.4 s. **Heavy accounts hard-blocked** |
| `/users/interactions/users/list` | ⚠️ | `author` | same | | | ✅ | Per-interaction rows w/ `interaction_type` |
| `/users/interactions/subreddits` | ⚠️ | `author` | `weight_posts`, `weight_comments`, `min_count` | | | ✅ | Returns `{subreddit, count}` |
| `/users/aggregate_flairs` | ⚠️ | `author` | | | | | Blocked for large accounts |
| `/short_links` | ✅ | `paths` | | 1000 | | | Resolves `redd.it/xxx` |
| **`/time_series`** | ✅ | `key`, `precision` | `after`, `before` | — | date asc | ✅ | **Undocumented gem — see §3d** |
| `/posts/search/aggregate` | ❌ | `aggregate`, `frequency` | `min_count`, `limit`, `sort` | | | ✅ | **Effectively unusable — see below** |
| `/comments/search/aggregate` | ❌ | same | | | | ✅ | same |

**404 (do not exist):** `/users/posts`, `/posts/tree`, `/subreddits/moderators`, `/users/interactions`, `/info`, `/status`, `/meta`.

### Aggregate endpoints are effectively dead
Every `aggregate=author` / `aggregate=subreddit` call returned `422 {"error":"Timeout. Maybe slow down a bit"}` — including a **single-day** window on r/india, a 1-week window on the much smaller r/bollywood, with and without `min_count`. `aggregate=created_utc&frequency=month` succeeded exactly once (13.07 s) and then began timing out too. **Do not design around these.** Use `/time_series` for counts and compute author aggregations client-side.

### Pagination mechanics (measured)
- `after` and `before` are both **exclusive**, comparing against `created_utc`.
- Accepts epoch **seconds or milliseconds**, ISO-8601, and relative offsets (`7d`, `1month` — all verified).
- There is **no offset/page token**. The only cursor is `after` + `sort=asc`.
- ⚠️ **Second-granularity cursors lose rows.** In r/india, 8 of 539 distinct seconds in one day held ≥2 comments. Setting `after=<last created_utc>` skips those siblings — verified directly:
  - second `1752539352` holds `n36d6rz` and `n36d6sh`; `after=1752539352` returns neither.
  - `after=1752539352000-1` (ms) returns **both**.
- **Fix, verified:** cursor on `after = last_created_utc*1000 - 1` and de-duplicate by `id`. On r/ipl 2025-05-25 this returned **5,887 comments — exactly matching** the `/time_series` ground truth. The naive second cursor lost 85/53,789 rows (0.16%) on r/developersIndia July 2025.
- Cost of the fix: ~0.6% duplicate rows re-fetched (88 dupes over 53,795 fetched).

### Field selection
`fields` accepts only a documented whitelist. `fields=controversiality` → `400 'controversiality' is not a valid field`. Allowed: `author, author_fullname, author_flair_text, created_utc, distinguished, id, retrieved_on, subreddit, subreddit_id, score` (+ posts: `crosspost_parent, link_flair_text, num_comments, over_18, post_hint, selftext, spoiler, title, url`; + comments: `body, link_id, parent_id`). **`_meta` is not selectable — requesting `fields` drops the removal metadata.** Using `fields` cut payload 25.5 MB → 5.6 MB (4.5×) and raised throughput 1,158 → 1,735 rows/s, so it is worth it only for passes that don't need removal labels.

### Archive coverage
r/india comments: **oldest 2008-02-17**, **newest 2026-08-04 18:47** — i.e. current to the minute of this audit.

---

## 2. Field inventory

Populated-rates below are from the **live-era census** (5 subs × 1 week posts / 2 days comments, Jul 2025). A `legacy` column contrasts the pre-2023 backfill sample. Full machine-generated tables (every field, all 4 object types plus every `_meta` sub-object) are in **`as/inv_final.md`**; the operationally important rows are reproduced here.

### 2.1 POST (t3) — 121 fields, n=7,064

**Always usable (≥99%):** `id`, `name`, `permalink`, `author`, `subreddit`, `subreddit_id`, `created_utc`, `title`, `score`, `ups`, `upvote_ratio`, `num_comments`, `num_crossposts`, `over_18`, `spoiler`, `locked`, `stickied`, `pinned`, `is_self`, `is_video`, `is_original_content`, `is_robot_indexable`, `media_only`, `contest_mode`, `archived`, `hidden`, `no_follow`, `send_replies`, `total_awards_received`, `gilded`, `subreddit_subscribers`, `retrieved_on`, `url` (99.1%), `domain` (98.8%).

| field | type | non-empty | legacy (pre-2023) | notes |
|---|---|---|---|---|
| `author_fullname` | str | **98.6%** | 94.4% | `t2_8h15cqoc`. **100.0% when `author != "[deleted]"`** (6,967/6,967) |
| `author_premium` | bool | 98.6% | 85.8% | only 19/7,064 True (0.3%) |
| `author_flair_text` | str | 16.1% | 16.1% | genuinely sparse — most users have no flair |
| `author_flair_template_id` | str | 15.1% | 15.1% | |
| `author_patreon_flair` | bool | 98.6% | 94.4% | all False |
| `link_flair_text` | str | 88.0% | 88.0% | |
| `selftext` | str | 59.9% non-tomb | — | **see §3a** |
| `removed_by_category` | str | **46.3%** | 50.4% | values: `moderator`, `reddit`, `automod_filtered`, `deleted`, `content_takedown`, `author` |
| `distinguished` | str | **0.014%** | 0.0% | 1 / 7,064. Effectively dead on posts |
| `edited` | bool/int | **0.06%** | 0.06% | 4 / 7,064 — **use `_meta.is_edited` instead** |
| `crosspost_parent` | str | 3.6% | 3.6% | |
| `post_hint` | str | 45.8% | 45.8% | |
| `media` | dict | 11.9% | 11.9% | |
| `is_created_from_ads_ui` | bool | 100% | 90.5% | **always False** — 0/7,064 True |
| `subreddit_subscribers` | int | 100% | 100% | live count *at post time* — see §3d |
| `downs` | int | 100% | 74% | **always 0** — see §3c |
| `gildings` / `all_awardings` | dict/list | 0.0% / 0.0% | 0.4% / 0.7% | **dead** — see §3c |

**ALWAYS null/empty on posts (23 fields — do not build features on these):**
`approved_at_utc`, `approved_by`, `awarders`, `banned_at_utc`, `banned_by`, `call_to_action`, `category`, `content_categories`, `discussion_type`, `likes`, `mod_note`, `mod_reason_by`, **`mod_reason_title`**, `mod_reports`, `num_reports`, **`removal_reason`**, `removed_by`, `report_reasons`, `top_awarded_type`, `treatment_tags`, `user_reports`, `view_count`, `distinguished` (0.014%, effectively null).

### 2.2 COMMENT (t1) — 75 fields, n=14,627

| field | type | non-empty | legacy | notes |
|---|---|---|---|---|
| `body` | str | 100% key-present | | 14.7% are `[removed]`/`[deleted]` tombstones — §3a |
| `author_fullname` | str | **84.2%** | 88.5% | **98.6% when `author != "[deleted]"`**; 0% when deleted |
| `controversiality` | int | 100% | 100% | **2.1% are 1** (312/14,627) — real, usable signal |
| `is_submitter` | bool | 100% | 100% | **8.9% True** — usable |
| `distinguished` | str | **8.7%** | 4.9% | 1,278 `moderator` — usable (AutoModerator detection) |
| `collapsed` | bool | 100% | 100% | **24.7% True** |
| `collapsed_reason_code` | str | **15.8%** | — | only value `DELETED`; **exactly identifies the unlabeled-tombstone class** (§3a) |
| `collapsed_reason` | str | 0.4% | 0.4% | near-dead |
| `collapsed_because_crowd_control` | null | **0.0%** | 0.0% | **DEAD** |
| `parent_id` | str | 100% | 100% | 55.5% `t3_` (top-level), 44.5% `t1_` |
| `link_id` | str | 100% | 100% | |
| `edited` | bool | **0.0%** | 0.0% | 0/14,627 — **use `_meta.is_edited`** |
| `removed_by_category` | — | **0.0%** | 0.0% | **NEVER SET ON COMMENTS** — critical, see §3a |
| `score`/`ups` | int | 100% | 100% | identical in 14,627/14,627 |
| `downs` | int | 100% | 74.3% | always 0 |
| `depth` | — | **absent** | absent | not returned by `/comments/search` |
| `total_awards_received`/`gilded` | int | 0 nonzero | 0 | **dead** |

**ALWAYS null/empty on comments (21 fields):**
`approved_at_utc`, `approved_by`, `associated_award`, `awarders`, `banned_at_utc`, `banned_by`, **`collapsed_because_crowd_control`**, `comment_type`, `likes`, `mod_note`, `mod_reason_by`, **`mod_reason_title`**, `mod_reports`, `num_reports`, **`removal_reason`**, `replies`, `report_reasons`, `top_awarded_type`, `treatment_tags`, `unrepliable_reason`, `user_reports`.

### 2.3 `_meta` — the undocumented provenance block (**the most valuable find**)

Present on **100%** of live-era (≥2023-07) posts and comments; **0%** before ~2023-04.

| key | posts | comments | meaning (empirically derived) |
|---|---|---|---|
| `retrieved_2nd_on` | 100% | 100% | second-pass timestamp. **`retrieved_2nd_on − created_utc` = 36.0 h** (p10 36.0, median 36.0/36.3, p90 36.4/40.1) — a deterministic T+36h re-check |
| `removal_type` | 7.0% | 4.5% | `moderator`, `deleted`, `automod_filtered`, `reddit`, `removed`, `removed by reddit`, `content_takedown` |
| `was_deleted_later` | 5.6% | 2.0% | alive at first capture, gone by T+36h ⇒ **original text is retained** |
| `was_initially_deleted` | 1.6% | 0.4% | already gone at first capture ⇒ text is a tombstone |
| `is_edited` | 6.4% | 0.6% | **the only working edit flag** (top-level `edited` is ~0%) |
| `edited_title` | 0.07% | — | e.g. `"[ Removed by moderator ]"` |

Also `_meta` on **subreddits**: `earliest_post`, `earliest_comment`, `num_posts`, `num_comments` (lifetime totals) + `*_updated_at`.

### 2.4 SUBREDDIT (t5) — 102 fields, n=343

Populated: `display_name`, `id`, `created_utc`, `subscribers`, `over18`, `quarantine`, `subreddit_type`, `lang`, `title`, `public_description`, `description`, `submission_type`, `restrict_posting`, `restrict_commenting`, `wiki_enabled`, `spoilers_enabled`, `allow_images`/`allow_videos`/`allow_polls`, `advertiser_category`, `community_icon`, `header_title`, `submit_text`, `free_form_reports`, `retrieved_on`.

**ALWAYS null (8):** **`accounts_active`**, **`active_user_count`**, `is_enrolled_in_new_modmail`, `user_can_flair_in_sr`, `user_flair_background_color`, `user_flair_css_class`, `user_flair_text`, `user_sr_flair_enabled`.

⚠️ **The subreddit object is 536 days stale.** All 43 Indian subs have `retrieved_on` = 2025-02-14/15. r/india reports `subscribers = 2,482,127`, while `/time_series` gives **3,470,901** for 2026-08. **Never use `subreddits/search.subscribers` as a current figure.**

### 2.5 USER (t2) — only 3 top-level fields

`/users/ids` returns just `{author, id, _meta}`. **There is no `created_utc`, no `is_suspended`, no `verified`, no `has_verified_email`, no profile description.**

`_meta` (n=299 sampled):

| key | non-null | note |
|---|---|---|
| `earliest_post_at` / `num_posts` / `post_karma` / `total_karma` | **100%** | |
| `earliest_comment_at` / `num_comments` / `comment_karma` | **47.5%** | absent for accounts first active after the comment-stats cutoff |
| `post_stats_updated_at` | 100% | median 2025-08-26 → **~344 days stale** |
| `comment_stats_updated_at` | 47.5% | 2025-03-24 → **~500 days stale** |

**⚠️ Account karma/activity aggregates are 1–1.5 years out of date and cannot support current-month account features.**

---

## 3. Critical questions — answers with evidence

### 3a. Is removed text retained, or tombstoned? **Partly retained, and the mechanism is knowable per-row.**

Arctic Shift stores a **merge of two snapshots**: text from the **first capture (median +16 s)** and score/removal status from the **second pass (+36.0 h)**. So an item removed *between* those two moments keeps its original text *and* carries a removal label.

**Unbiased census, 5 subs, Jul 2025:**

| | posts (self-only, n=4,543) | comments (n=14,627) |
|---|---|---|
| labelled removed/deleted | **3,660 (80.6%)** | 661 (4.5%) |
| …with original text recovered | **476 (13.0%)** | **661 (100.0%)** |
| unlabelled tombstones (`[removed]`/`[deleted]`, no label) | — | **2,145 (14.7%)** |

Breakdown by *when* removal happened (posts):

| label | timing | REAL text | TOMBSTONE |
|---|---|---|---|
| `moderator` | at capture | 8 | **2,548** |
| `moderator` | later (`was_deleted_later`) | **192** | 0 |
| `deleted` | at capture | 0 | 57 |
| `deleted` | later | **190** | 0 |
| `automod_filtered` | at capture | 58 | 206 |
| `reddit` | at capture | 22 | 337 |

**The rule is clean: `was_deleted_later == True` ⇒ text is real (382/382 posts, 55/55 comments). Otherwise the item was already `[removed]` when Arctic Shift first saw it and the text is gone forever.** Moderator/automod action in these subs is near-instantaneous (< 16 s), which is why post recovery is only 13%.

**Two traps:**
1. **`removed_by_category` is NEVER populated on comments (0/14,627).** Removal detection on comments *must* use `_meta.removal_type`. On posts, `removed_by_category` alone catches 3,274 but the union with `_meta.removal_type` is 3,660 — **it under-counts removals by 11%**.
2. **The 14.7% unlabelled comment tombstones** all have `author == "[deleted]"` *and* `body == "[removed]"` *and* `collapsed_reason_code == "DELETED"` — use that triple as the detector; they carry no `removal_type`.

**Era dependence (r/india + IndiaSpeaks + bollywood, 3-day windows):**

| era | posts `_meta` | posts %removed | posts %text recovered | comments %removed | comments %recovered | comment tombstones, unlabelled |
|---|---|---|---|---|---|---|
| 2016-03 | 0% | 0.0% | — | 0.0% | — | 10.3% |
| 2019-03 | 0% | 0.0% | — | 0.0% | — | 9.4% |
| 2021-03 | 0% | 62.9% | 3.7% | 0.0% | — | 12.1% |
| 2023-03 | 0% | 74.2% | 0.0% | 0.0% | — | 11.4% |
| 2024-09 | **100%** | 89.3% | 10.1% | 5.4% | **100%** | 24.0% |
| 2025-07 | 92% | 92.7% | 13.9% | 4.2% | **100%** | 23.5% |
| 2026-06 | 100% | 89.7% | 14.0% | 3.6% | **100%** | 12.8% |

**Bottom line for the positive class:** text features for removed **comments** exist and are complete (100% of labelled removals) — but only from **2023-07** onward. For removed **posts** you get text on ~13% of them. Before 2023-07 there are **no removal labels at all**, so the positive class is unlabelled and its text is largely `[removed]` with no way to tell removal from deletion.

### 3b. `author_fullname` on deleted authors, and base36 monotonicity

**(i) Presence:** `author_fullname` is present in **100.0% of posts** and **98.6% of comments** where `author != "[deleted]"`, and in **0 of 97 posts / 0 of 2,145 comments** where the author *is* `[deleted]`. It is a perfect proxy for "author still exists". **Once an account is deleted or suspended, its `t2_` id is unrecoverable from content** — so cohort features are impossible for exactly the population you most want to profile.

**(ii) Monotonicity: YES, confirmed.** Arctic Shift exposes no account `created_utc`, so I used `min(earliest_post_at, earliest_comment_at)` from `/users/ids._meta` (n=1,194) as an upper-bounded proxy — an account can lurk for years, so this is noisy upward but never downward. Therefore the **lower envelope** of earliest-activity per id band bounds true creation date.

| band | example id | log10(b36) | n | **MIN earliest activity** | median |
|---|---|---|---|---|---|
| 0 | `1zo4` | 4.97 | 85 | **2007-04-07** | 2017-10-01 |
| 2 | `9hd9f3n6` | 11.87 | 85 | **2021-01-04** | 2022-09-06 |
| 4 | `totqzrcq` | 12.37 | 85 | **2022-10-27** | 2023-10-14 |
| 6 | `y8rplmzwt` | 13.98 | 85 | **2024-04-14** | 2024-11-26 |
| 8 | `1hgveiaws6` | 14.18 | 85 | **2025-01-21** | 2025-03-23 |
| 10 | `1ovwotvrmh` | 14.23 | 85 | **2025-05-07** | 2025-06-09 |
| 12 | `1spzhy8797` | 14.26 | 85 | **2025-07-02** | 2025-07-07 |
| 13 | `1t3lv6eajv` | 14.26 | 89 | **2025-07-08** | 2025-07-09 |

**Lower envelope is monotonic in 13/13 consecutive transitions.** Spearman(base36, earliest-activity) = **0.8216** (the shortfall from 1.0 is lurk-time noise, not id disorder). **AUC for separating pre-2023 from post-2025-07 accounts = 0.9855.**

**Verdict: the registration-cohort signal is viable.** Decode `author_fullname[3:]` base-36 and calibrate against the lower envelope. Note id *length* alone is a coarse era proxy (len 4–5 → 2007–2018; len 8 → median 2022-08; len 9 → 2024-05; len 10 → 2025-06).

### 3c. `downs`, `ups` vs `score`, and awards

- **`downs` is always 0.** 7,064/7,064 posts and 14,627/14,627 comments in the live census; the entire distribution is `{0: n}`. In legacy data it is 0 or absent. **Dead field.**
- **`ups` is a byte-identical copy of `score`** — equal in **7,064/7,064** and **14,627/14,627** cases. **Carry only one.**
- **Awards are entirely dead:** `gilded > 0` in **0** of 21,691 objects; `total_awards_received > 0` in **0**; `all_awardings` non-empty in **0**. (Reddit retired awards in 2023; the legacy 2019–2022 sample shows 0.7%/0.1%.) `gildings` is 0.4%/9.0% in legacy only.
- **`score` itself is genuine and mature**, contrary to first appearance. Median is 1 (most content in these subs really is ignored or removed), but the tail is real: posts **max 5,183 / p99 818 / p90 32 / mean 35.7**; comments **max 1,253 / p99 116 / mean 7.1**. `upvote_ratio` has 77 distinct values from 0.10 upward. Because the score comes from the fixed **T+36h** pass, it is *consistently* aged across every row — good for modelling, but it is **not** a final score and will understate late-blooming threads.

### 3d. Subreddit-level metadata and time-series

**Snapshot fields** (see §2.4): subscriber count, `created_utc`, `over18`, `quarantine`, `subreddit_type`, descriptions, posting restrictions, plus `/subreddits/rules` (14 rules for r/india with per-rule `created_utc`) and `/subreddits/wikis`. **`active_user_count` and `accounts_active` are always null** — no online-user metric. The snapshot itself is **536 days stale**.

**✅ Historical time-series DOES exist — `/api/time_series`.** Precision `year|quarter|month|week|day|hour|minute`, with `after`/`before`. Keys:
`global/{posts,comments}/{count,sum_score,sum_retrieved_after_seconds}`, `r/<sub>/{posts,comments}/{count,sum_score}`, and **`r/<sub>/subscribers`**.

Measured coverage — sub-second responses, monthly since 2018-03, gapless daily:

| sub | months | from | to |
|---|---|---|---|
| r/india | 63 | 2018-03 → 117,792 | 2026-08 → **3,470,901** |
| r/bollywood | 63 | 2018-03 → 8,179 | 2026-08 → 1,406,667 |
| r/ipl | 61 | 2018-03 → 444 | 2026-08 → 838,951 |
| r/developersIndia | 50 | 2020-04 → 3,036 | 2026-08 → 1,577,228 |

Daily precision verified: 61 consecutive daily points for r/india Jun–Jul 2026. Values are floats (interpolated/averaged).

**Second, independent subscriber series:** every post carries `subreddit_subscribers` **at post time** (100% populated). Over one week of r/india that traced 3,276,377 → 3,281,775 (+5,398) across 2,605 posts. Non-monotonic at fine grain (Reddit's own counter fluctuates), but it gives free high-resolution growth data and a cross-check on `/time_series`.

**`/time_series` is also the correct way to get post/comment counts**, since the aggregate endpoints time out. It is exact: r/ipl 2025-05-25 → 5,887, matching a full paginated pull row-for-row.

### 3e. Comment-tree completeness

With `limit=25000&start_breadth=0&start_depth=0`, **`/comments/tree` returned zero `more` stubs on every post tested** and consistently returned **more** comments than `num_comments`, including removed/deleted ones.

| post | sub | `num_comments` | tree | `more` stubs | tombstones in tree |
|---|---|---|---|---|---|
| 1ly4hmw | india | 558 | **770** | 0 | 65 |
| 1lxae42 | bollywood | 400 | **413** | 0 | 124 |
| 1lul0s5 | india | 340 | **391** | 0 | 76 |
| 1ltpucr | IndiaSpeaks | 300 | **323** | 0 | 146 |
| 1lvpef6 | developersIndia | 295 | **299** | 0 | 22 |
| 1lwnf3i | bollywood | 267 | **359** | 0 | 82 |
| 1ltl9ng | india | 10 | **15** | 0 | 0 |
| 1ltlbfn | india | 30 | **34** | 0 | 7 |

Tree ≥ `num_comments` always, because `num_comments` is the **T+36h** count while the tree includes everything ever archived plus tombstones. **The tree is complete; `num_comments` is the undercount.**

**But you don't need the tree.** `/comments/search` returns `parent_id` (100%, 55.5% `t3_`/44.5% `t1_`) and `link_id` (100%), so the full tree is reconstructable offline. And the tree endpoint is **~7× slower per comment** (179 comments/s vs 1,158–1,735 rows/s). Note `depth` is *not* returned by search — compute it from `parent_id`. **Recommendation: bulk-collect via `/comments/search`, rebuild trees locally, and reserve `/comments/tree` for spot checks.**

### 3f. Score trajectory over time? **No — exactly two snapshots, and only one carries a score.**

The stored record has `retrieved_on` (+16 s median) and `_meta.retrieved_2nd_on` (**+36.0 h**, p10 36.0 → p90 36.4). The full set of `_meta` keys ever observed across 21,691 objects is `retrieved_2nd_on, removal_type, is_edited, was_deleted_later, was_initially_deleted, edited_title` — **no score history of any kind**. Repeated fetches return an identical stored value (20/20 identical scores; `/posts/ids` and `/posts/search` agree byte-for-byte).

**Available instead:**
- A **fixed-age score at T+36h**, which is at least uniform across all rows.
- Subreddit-level `sum_score` series via `/time_series` at any precision.
- If per-post trajectories are genuinely required, you must poll live Reddit yourself going forward — the archive cannot supply them retroactively.

---

## 4. Throughput and collection-time estimates

### Measured (r/india, 2025-07-07 → 07-14, 13,588 comments, single-threaded)

| mode | rows | pages | wall | **rows/s** | **req/s** | payload | median latency |
|---|---|---|---|---|---|---|---|
| `limit=auto`, full objects, no sleep | 13,588 | 30 | 11.7 s | **1,158** | 2.56 | 25.5 MB | 0.36 s |
| `limit=auto`, `fields` subset, no sleep | 13,588 | 31 | 7.8 s | **1,735** | 3.96 | 5.6 MB | 0.24 s |
| `limit=auto`, full objects, 0.35 s sleep | 13,588 | 27 | 17.2 s | 789 | 1.57 | 25.5 MB | 0.29 s |
| `/comments/tree` (10 posts, 404 comments) | 404 | 10 | 2.3 s | **179** | 4.42 | | |

**Rate limits:** a 25-request burst with zero sleep completed in 3.1 s = **8.17 req/s with 25/25 HTTP 200 and no `429`**. No rate-limit headers were returned. `limit=auto` yielded 651–668 rows/page and was stable across repeats. Throughput is bandwidth-bound, not request-bound — hence `fields` giving a 1.5× speed-up.

### Volume assumptions (measured, not assumed)

Pulled from `/time_series` for the 43 Indian subs that resolved (of 45 attempted; `r/unitedstatesofindia` returned 0, and 2 names did not resolve):

- **July 2025: 162,810 posts + 2,356,120 comments = 2,518,930 rows**
- **12 months (2025-08 → 2026-07): 1,838,717 posts + 26,214,513 comments = 28,053,230 rows**

Largest contributors/month: r/indiasocial 397,629 · r/Cricket 364,723 · r/JEENEETards 173,659 · r/delhi 140,257 · r/BollyBlindsNGossip 118,735 · r/Btechtards 107,791 · r/AskIndia 105,488.

Overhead: +0.6% duplicate rows from the ms-cursor overlap; ~1 request per 660 rows.

### Wall-clock estimates (single-threaded, sequential)

| scope | rows | @1,158 rows/s (full) | @1,735 rows/s (`fields`) | @789 rows/s (polite) |
|---|---|---|---|---|
| **1 month, 43 subs** | 2.52 M | **36 min** | 24 min | 53 min |
| **12 months, 43 subs** | 28.05 M | **6.7 h** | 4.5 h | 9.9 h |

**Requests:** ~3,800 for one month, ~42,500 for twelve — trivially within budget at 8 req/s.

**Storage (measured 1.88 KB/row full JSON, 0.41 KB/row with `fields`):**
- 1 month: **4.7 GB** raw / 1.0 GB slim
- 12 months: **52.6 GB** raw / 11.5 GB slim
- With zstd on NDJSON expect ~8–10× reduction → ~5–7 GB for the full year.

**Practical recommendation:** run the full-object path (you need `_meta`, which `fields` strips) at ~2 concurrent workers with ~0.2 s sleep. That lands near 8 req/s in aggregate — the verified-safe ceiling — and puts **12 months of all 45 subs at roughly 3–4 hours**. This is a very comfortable budget; there is no need to compromise on scope.

**Do NOT** use `/comments/tree` for bulk (28 M comments at 179/s = **43 hours**), and do not use `/users/interactions/*` at scale (1–3.4 s per account, hard-blocked on heavy accounts).

---

## 5. Alternative sources

| source | status | coverage | verdict |
|---|---|---|---|
| **Arctic Shift API** | ✅ live | 2008-02 → **now** (current to the minute) | Primary. Only source with `_meta` provenance |
| **Arctic Shift monthly dumps** | ✅ | via GitHub `download_links.md` + releases; `.zst`/`.zst_blocks`/NDJSON | **Recommended for the 12-month backfill.** Same schema as the API; avoids 42,500 HTTP requests. Also a [per-subreddit/user download tool](https://arctic-shift.photon-reddit.com/download-tool) (HTTP 200, live) |
| **PullPush.io** | ⚠️ works, **stale** | Most recent r/india comment: **2025-05-19** (~15 months behind) | `/reddit/search/{comment,submission}/` return 70/105 fields incl. `_meta`. `/reddit/{comment,submission}/ids/` → **404**. Useful only as a cross-check for pre-2025-05 data; **cannot fill the current-month gap** |
| **Reddit public JSON** (`.json`, `about.json`) | ❌ **blocked** | — | **403 'Blocked'** on all of `www.reddit.com` and `old.reddit.com`, with both a bot UA and a browser UA. Reddit blocks datacenter IPs. Would need OAuth credentials or a residential egress. This is the **only** route to live scores, `num_comments`, subscriber counts, and account `created_utc`/suspension status |
| **Academic Torrents Pushshift dumps** | ✅ | 2005-06 → 2023-12 ([full](https://academictorrents.com/details/9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4)), and a [per-subreddit split](https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10) covering 2005-06 → 2023-12 | Identical NDJSON schema. Best option for the **pre-2023-07 era** where the API has no `_meta`. Note it inherits the same "no removal labels" limitation |

**Gaps that no free source fills:** live score trajectories, account creation dates, account suspension status, moderator lists, and current subscriber counts (except via Arctic Shift's own `/time_series`).

Sources: [Academic Torrents — Reddit 2005-06 to 2023-12](https://academictorrents.com/details/9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4) · [Academic Torrents — per-subreddit](https://academictorrents.com/details/56aa49f9653ba545f48df2e33679f014d2829c10) · [Pushshift Alternatives 2026](https://rawneed.com/guides/pushshift-alternatives/) · [Arctic Shift GitHub](https://github.com/ArthurHeitmann/arctic_shift)

---

## 6. NOT available — build no features on these

**Fields that exist in the response but are always null/empty**
- Posts (23): `approved_at_utc`, `approved_by`, `awarders`, `banned_at_utc`, **`banned_by`**, `call_to_action`, `category`, `content_categories`, `discussion_type`, `likes`, `mod_note`, `mod_reason_by`, **`mod_reason_title`**, `mod_reports`, `num_reports`, **`removal_reason`**, `removed_by`, `report_reasons`, `top_awarded_type`, `treatment_tags`, `user_reports`, `view_count`; plus `distinguished` (1/7,064).
- Comments (21): same mod-tooling set plus **`collapsed_because_crowd_control`**, `associated_award`, `comment_type`, `replies`, `unrepliable_reason`.
- Subreddits (8): **`active_user_count`**, **`accounts_active`**, and the `user_flair_*` set.
- Everywhere: **`downs`** (always 0), **`gilded`/`total_awards_received`/`all_awardings`** (always 0/empty), **`is_created_from_ads_ui`** (always False), **`ups`** (duplicate of `score`).
- Comments: **`removed_by_category`** — never set. **`depth`** — not returned by search.
- Posts/comments: **`edited`** — ~0% true; the real signal is `_meta.is_edited`.

**Capabilities that do not exist**
1. **No account creation date, suspension flag, verified-email flag, or profile metadata.** Account age must be inferred from base36 `author_fullname` (works — AUC 0.986).
2. **No `author_fullname` on deleted/suspended-author content** (0/2,242 cases) — no cohort signal for the deleted population.
3. **No score trajectory.** Two snapshots; only the T+36h one has a score.
4. **No moderator lists, modlogs, mod-action attribution, or report counts.**
5. **No removal metadata whatsoever before ~2023-07**, and no `_meta` at all before ~2023-04.
6. **Original text of content removed within ~16 s is permanently gone** — 87% of removed posts.
7. **Aggregate endpoints (`/posts/search/aggregate`, `/comments/search/aggregate`) time out** on essentially every author/subreddit grouping.
8. **`/users/interactions/*` and `/users/aggregate_flairs` are hard-blocked for high-volume accounts** (`"This user is currently not supported (too much data)"`) — i.e. exactly the bots you want to profile.
9. **User karma/activity aggregates are 344–500 days stale.**
10. **Subreddit snapshot objects are 536 days stale** — use `/time_series` for subscribers.
11. **No sorting except `created_utc` asc/desc**, no offset pagination, no full-text search without an `author`/`subreddit`/`link_id` anchor.
12. **`fields` strips `_meta`** — you cannot have both the bandwidth saving and the removal labels.
13. **No live Reddit fallback from this environment** — `www.reddit.com` returns 403.

---

## 7. Implementation checklist

1. **Paginate with `after = last_created_utc*1000 - 1`, `sort=asc`, `limit=auto`; de-duplicate by `id`.** Anything else silently loses ~0.16% of rows. Validate every sub-month against `/time_series/r/<sub>/comments/count` — it matched exactly in testing.
2. **Removal label = `removed_by_category` ∪ `_meta.removal_type`.** On comments the first is always null. On posts, `removed_by_category` alone misses 11%.
3. **Text-availability flag = `_meta.was_deleted_later`.** True ⇒ real text; otherwise assume tombstone.
4. **Third class for comments:** `author == "[deleted]" AND body == "[removed]" AND collapsed_reason_code == "DELETED"` (14.7% of comments) — removed, no label, no text.
5. **Do not request `fields`** on the main pass; you need `_meta`.
6. **Account age:** base-36 decode `author_fullname[3:]`, calibrate on the §3b lower envelope.
7. **Subscribers:** `/time_series/r/<sub>/subscribers` (daily available), plus per-post `subreddit_subscribers`. Never the subreddit object.
8. **Restrict rich modelling to ≥ 2023-07.** Earlier data has no removal labels, no edit flags, and no `_meta`.
9. **Consider the monthly dumps** instead of 42,500 API calls for the 12-month backfill.
