#!/usr/bin/env python3
"""V3 account-level feature table (V3_PLAN.md Sec 4.3), built on top of
Stage 0's commenters_clean / posts tables in data/v3/analysis/v3.duckdb.

Important constraint found while building this: our corpus is a BIPARTITE
SAMPLE (an account only appears when it's in a sampled post's first-10 or
top-10-by-score), not a per-account census. Measured distribution:
median 2 comments/account, 48% of the 347,886 accounts are singletons
(exactly 1 comment in the whole 24-month corpus), only ~10% have >=10.
This directly explains -- and is explained by -- the plan's own account-vs-
pair framing (Sec 1): most accounts simply don't carry enough individual
signal for timing-family features, which is exactly why the pair layer
(Sec 4.4, bipartite co-appearance) is the primary detector, not an
account-level score.

Two tiers, gated on sample size so a feature computed from 2 data points
doesn't get treated as if it meant something:
  Tier 1 (all accounts, n_comments>=1): provenance, footprint, reception,
    username morphology -- these don't need a dense time series.
  Tier 2 (n_comments>=5): interval entropy, Kim-Jo finite-size-corrected
    burstiness (A9), interval quantization (A8, cron-signature detection).

NOT built here, and why: circadian/session/weekday features (A10-A13).
The pilot's behavioral check (scripts/v3_pilot_behavioral_check.py) found
real multimodal structure in these on 450 accounts -- but using each
account's last-50-comments via a SEPARATE direct API pull, not this
bipartite sample. At median n=2 in-corpus, "zero activity in hour H" is a
sparsity artefact for ~90% of accounts, not a circadian signal. Doing this
properly at scale means a supplemental full-history pull (like the pilot's)
for a bounded account subset -- flagged as follow-up work, not faked here
with an underpowered proxy.

Three different "age-like" quantities, do not conflate them:
  account_ordinal        base36-decoded author_fullname. Monotonic with true
                          signup order but an uncalibrated integer, not a
                          date -- live-Reddit calibration exists as a script
                          (v3_calibrate_age_sample.py) but is on hold; not
                          worth ~1,750 live-Reddit lookups when the proxy
                          below is free and, at n=347,886, noise-cancels.
  observed_span_days     last_seen_utc - first_seen_utc, i.e. how long we
                          kept re-observing them. Exactly 0 for the 48% of
                          accounts that are singletons -- not usable as an
                          age proxy for half the population.
  days_since_first_seen  now() - first_seen_utc. The one actually used for
                          the karma/comments/posts "per age" rate features
                          below -- non-zero even for singletons, and a
                          single noisy per-account estimate but, per the
                          user's framing, cancels out in aggregate at this
                          n. Still not true account age: an account first
                          spotted last month could have signed up years
                          ago, we just hadn't sampled them into a top-100
                          post before."""
import math
import os
import re
import time
from collections import Counter

import duckdb
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')

TIER2_MIN_COMMENTS = 5

DEFAULT_USERNAME_RE = re.compile(r'^[A-Z][a-z]+[-_]?[A-Z][a-z0-9]+[-_]?\d{2,6}$')
DIGIT_SUFFIX_RE = re.compile(r'(\d+)$')


def username_morphology(u):
    if not u:
        return None, None, None
    is_default_pattern = bool(DEFAULT_USERNAME_RE.match(u))
    n = len(u)
    counts = Counter(u)
    char_entropy = -sum((c / n) * math.log2(c / n) for c in counts.values()) if n else 0.0
    m = DIGIT_SUFFIX_RE.search(u)
    digit_suffix_len = len(m.group(1)) if m else 0
    return is_default_pattern, char_entropy, digit_suffix_len


def interval_entropy_normalized(gaps, n_bins=10):
    gaps = [g for g in gaps if g > 0]
    if len(gaps) < 2:
        return None
    lo, hi = min(gaps), max(gaps)
    if lo == hi:
        return 0.0
    log_lo, log_hi = math.log10(lo), math.log10(hi + 1)
    width = (log_hi - log_lo) / n_bins
    bins = [0] * n_bins
    for g in gaps:
        idx = min(n_bins - 1, int((math.log10(g) - log_lo) / width))
        bins[idx] += 1
    total = sum(bins)
    ent = -sum((c / total) * math.log2(c / total) for c in bins if c > 0)
    return ent / math.log2(n_bins)


def kim_jo_burstiness(gaps):
    n = len(gaps)
    if n < 3:
        return None
    mu = sum(gaps) / n
    if mu == 0:
        return None
    var = sum((g - mu) ** 2 for g in gaps) / n
    sigma = math.sqrt(var)
    if sigma == 0:
        return -1.0
    r = sigma / mu
    sqrt_np1, sqrt_nm1 = math.sqrt(n + 1), math.sqrt(n - 1)
    denom = (sqrt_np1 - 2) * r + sqrt_nm1
    if denom == 0:
        return None
    return (sqrt_np1 * r - sqrt_nm1) / denom


def quantization_rate(gaps, targets=(60, 300, 900), tol=2):
    if not gaps:
        return None
    hits = 0
    for g in gaps:
        if g <= 0:
            continue
        for t in targets:
            nearest = round(g / t) * t
            if nearest > 0 and abs(g - nearest) <= tol:
                hits += 1
                break
    return hits / len(gaps)


def build_tier2(con):
    rows = con.execute(f"""
        SELECT author, list(created_utc ORDER BY created_utc) AS ts
        FROM commenters_clean
        GROUP BY author
        HAVING count(*) >= {TIER2_MIN_COMMENTS}
    """).fetchall()

    out = []
    for author, ts in rows:
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        out.append({
            'author': author,
            'interval_entropy': interval_entropy_normalized(gaps),
            'burstiness_kimjo': kim_jo_burstiness(gaps),
            'interval_quantization_rate': quantization_rate(gaps),
            'n_gaps': len(gaps),
        })
    return out


def main():
    t0 = time.time()
    con = duckdb.connect(DB_PATH)

    print('Tier 1: provenance / footprint / reception (all accounts)...', flush=True)
    con.execute("""
        CREATE OR REPLACE TABLE account_comment_stats AS
        SELECT
            author,
            count(*) AS n_comments_sample,
            count(DISTINCT sub) AS n_subs_active,
            count(DISTINCT post_id) AS n_threads_active,
            min(created_utc) AS first_seen_utc,
            max(created_utc) AS last_seen_utc,
            avg(score) AS mean_comment_score,
            median(score) AS median_comment_score,
            stddev_samp(score) AS score_stddev,
            avg(CASE WHEN controversiality=1 THEN 1.0 ELSE 0.0 END) AS controversiality_rate,
            avg(CASE WHEN is_submitter THEN 1.0 ELSE 0.0 END) AS is_submitter_rate,
            avg(depth) AS mean_depth,
            avg(body_len) AS mean_body_len,
            any_value(account_ordinal) AS account_ordinal,
            bool_or(is_confirmed_automation_seed) AS ever_automation_seed,
            avg(CASE WHEN meta_removal_type IS NOT NULL THEN 1.0 ELSE 0.0 END) AS removal_rate,
            avg(CASE WHEN meta_was_deleted_later THEN 1.0 ELSE 0.0 END) AS deleted_later_rate
        FROM commenters_clean
        GROUP BY author
    """)

    con.execute("""
        CREATE OR REPLACE TABLE account_post_stats AS
        SELECT author, count(*) AS n_posts_sample, avg(score) AS mean_post_score
        FROM posts
        WHERE NOT is_deleted_author AND NOT is_confirmed_automation_seed
        GROUP BY author
    """)

    print('Tier 1: conversation engagement (repeat-in-thread + own-post reply rate)...', flush=True)
    con.execute("""
        CREATE OR REPLACE TABLE account_thread_repeat AS
        WITH per_thread AS (
            SELECT author, post_id, count(*) AS n_in_thread
            FROM commenters_clean GROUP BY author, post_id
        )
        SELECT author,
               count(*) AS n_distinct_threads,
               sum(CASE WHEN n_in_thread > 1 THEN 1 ELSE 0 END) AS n_threads_with_repeat,
               avg(CASE WHEN n_in_thread > 1 THEN 1.0 ELSE 0.0 END) AS repeat_engagement_rate
        FROM per_thread GROUP BY author
    """)
    con.execute("""
        CREATE OR REPLACE TABLE account_own_post_engagement AS
        SELECT author,
               avg(submitter_reply_rate) AS own_post_reply_rate,
               count(*) AS n_own_posts_with_comments
        FROM posts
        WHERE NOT is_deleted_author AND NOT is_confirmed_automation_seed AND n_comments_observed > 0
        GROUP BY author
    """)

    print('Tier 1: silo-mismatch pattern (rejected-but-returns, no lean labels needed)...', flush=True)
    con.execute("""
        CREATE OR REPLACE TABLE account_sub_reception AS
        SELECT author, sub,
               avg(score) AS mean_score_in_sub,
               count(*) AS n_comments_in_sub,
               count(DISTINCT month) AS n_months_in_sub
        FROM commenters_clean
        GROUP BY author, sub
    """)
    con.execute("""
        CREATE OR REPLACE TABLE account_silo_pattern AS
        SELECT author,
               count(*) FILTER (WHERE mean_score_in_sub < 0 AND n_months_in_sub >= 2) AS n_subs_rejected_but_returned,
               max(mean_score_in_sub) AS best_sub_mean_score,
               min(mean_score_in_sub) AS worst_sub_mean_score,
               (max(mean_score_in_sub) - min(mean_score_in_sub)) AS reception_spread
        FROM account_sub_reception
        GROUP BY author
    """)

    con.execute("""
        CREATE OR REPLACE TABLE account_sub_footprint AS
        SELECT
            c.author AS author,
            count(*) FILTER (WHERE incentive_tier='high') AS n_high_tier,
            count(*) FILTER (WHERE incentive_tier='medium') AS n_medium_tier,
            count(*) FILTER (WHERE incentive_tier='low') AS n_low_tier,
            count(DISTINCT c.sub) FILTER (WHERE incentive_tier='low') AS n_low_tier_subs
        FROM commenters_clean c JOIN posts p ON c.post_id=p.post_id AND c.sub=p.sub AND c.month=p.month
        GROUP BY c.author
    """)

    print('Tier 1: subreddit entropy (Shannon, per account)...', flush=True)
    con.execute("""
        CREATE OR REPLACE TABLE account_sub_entropy AS
        WITH sub_counts AS (
            SELECT author, sub, count(*) AS n FROM commenters_clean GROUP BY author, sub
        ), totals AS (
            SELECT author, sum(n) AS total FROM sub_counts GROUP BY author
        )
        SELECT sc.author,
               -sum((sc.n::DOUBLE / t.total) * ln(sc.n::DOUBLE / t.total)) / ln(2) AS subreddit_entropy
        FROM sub_counts sc JOIN totals t USING (author)
        GROUP BY sc.author
    """)

    print('Tier 1: username morphology (all distinct authors)...', flush=True)
    authors = [r[0] for r in con.execute('SELECT DISTINCT author FROM commenters_clean').fetchall()]
    morph_rows = []
    for a in authors:
        is_default, ent, digit_len = username_morphology(a)
        morph_rows.append({'author': a, 'username_is_default_pattern': is_default,
                            'username_char_entropy': ent, 'username_digit_suffix_len': digit_len})
    morph_df = pd.DataFrame(morph_rows)
    con.register('morph_df', morph_df)
    con.execute("CREATE OR REPLACE TABLE account_username_morphology AS SELECT * FROM morph_df")

    print(f'Tier 2: timing (n_comments >= {TIER2_MIN_COMMENTS} only)...', flush=True)
    tier2_rows = build_tier2(con)
    tier2_df = pd.DataFrame(tier2_rows) if tier2_rows else pd.DataFrame(
        columns=['author', 'interval_entropy', 'burstiness_kimjo', 'interval_quantization_rate', 'n_gaps'])
    con.register('tier2_df', tier2_df)
    con.execute("CREATE OR REPLACE TABLE account_timing_features AS SELECT * FROM tier2_df")

    print('Assembling account_features...', flush=True)
    con.execute("""
        CREATE OR REPLACE TABLE account_features AS
        SELECT
            cs.author,
            cs.n_comments_sample,
            COALESCE(ps.n_posts_sample, 0) AS n_posts_sample, ps.mean_post_score,
            cs.n_subs_active, cs.n_threads_active,
            cs.first_seen_utc, cs.last_seen_utc,
            cs.mean_comment_score, cs.median_comment_score, cs.score_stddev,
            cs.controversiality_rate, cs.is_submitter_rate, cs.mean_depth, cs.mean_body_len,
            cs.account_ordinal, cs.ever_automation_seed,
            cs.removal_rate, cs.deleted_later_rate,
            sf.n_high_tier, sf.n_medium_tier, sf.n_low_tier, sf.n_low_tier_subs,
            (sf.n_low_tier_subs = 0) AS hobby_absence,
            se.subreddit_entropy,
            um.username_is_default_pattern, um.username_char_entropy, um.username_digit_suffix_len,
            tf.interval_entropy, tf.burstiness_kimjo, tf.interval_quantization_rate, tf.n_gaps,
            (tf.n_gaps IS NOT NULL) AS has_timing_features,
            GREATEST((cs.last_seen_utc - cs.first_seen_utc) / 86400.0, 0) AS observed_span_days,
            cs.n_comments_sample / GREATEST((cs.last_seen_utc - cs.first_seen_utc) / 86400.0, 1) AS comments_per_day_observed,
            (cs.mean_comment_score * cs.n_comments_sample) / GREATEST((cs.last_seen_utc - cs.first_seen_utc) / 86400.0, 1) AS sample_score_per_day_observed,
            (epoch(now()) - cs.first_seen_utc) / 86400.0 AS days_since_first_seen,
            cs.n_comments_sample / GREATEST((epoch(now()) - cs.first_seen_utc) / 86400.0, 1) AS comments_per_day_since_first_seen,
            COALESCE(ps.n_posts_sample, 0) / GREATEST((epoch(now()) - cs.first_seen_utc) / 86400.0, 1) AS posts_per_day_since_first_seen,
            (cs.mean_comment_score * cs.n_comments_sample) / GREATEST((epoch(now()) - cs.first_seen_utc) / 86400.0, 1) AS karma_per_day_since_first_seen,
            tr.repeat_engagement_rate, tr.n_distinct_threads, tr.n_threads_with_repeat,
            ope.own_post_reply_rate, ope.n_own_posts_with_comments,
            sp.n_subs_rejected_but_returned, sp.best_sub_mean_score, sp.worst_sub_mean_score, sp.reception_spread,
            (sp.n_subs_rejected_but_returned >= 1 AND sp.best_sub_mean_score > 5) AS shows_silo_mismatch_pattern
        FROM account_comment_stats cs
        LEFT JOIN account_post_stats ps USING (author)
        LEFT JOIN account_sub_footprint sf USING (author)
        LEFT JOIN account_sub_entropy se USING (author)
        LEFT JOIN account_username_morphology um USING (author)
        LEFT JOIN account_timing_features tf USING (author)
        LEFT JOIN account_thread_repeat tr USING (author)
        LEFT JOIN account_own_post_engagement ope USING (author)
        LEFT JOIN account_silo_pattern sp USING (author)
    """)

    # ---- report ----
    n = con.execute('SELECT count(*) FROM account_features').fetchone()[0]
    n_timing = con.execute('SELECT count(*) FROM account_features WHERE has_timing_features').fetchone()[0]
    n_default_user = con.execute('SELECT count(*) FROM account_features WHERE username_is_default_pattern').fetchone()[0]
    n_hobby_absent = con.execute('SELECT count(*) FROM account_features WHERE hobby_absence').fetchone()[0]
    print('\n=== ACCOUNT FEATURE TABLE REPORT ===')
    print(f'accounts: {n}')
    print(f'  with Tier-2 timing features (n_comments>={TIER2_MIN_COMMENTS}): {n_timing} ({100*n_timing/n:.1f}%)')
    print(f'  default-pattern username (Adjective_Noun_1234-style): {n_default_user} ({100*n_default_user/n:.1f}%)')
    print(f'  hobby_absence (zero activity in any low-incentive sub): {n_hobby_absent} ({100*n_hobby_absent/n:.1f}%)')
    print('\nburstiness_kimjo distribution (Tier-2 accounts):')
    print(con.execute('SELECT min(burstiness_kimjo), quantile_cont(burstiness_kimjo,0.25), median(burstiness_kimjo), quantile_cont(burstiness_kimjo,0.75), max(burstiness_kimjo) FROM account_features WHERE has_timing_features').fetchdf())
    print('\nsubreddit_entropy distribution (all accounts):')
    print(con.execute('SELECT min(subreddit_entropy), median(subreddit_entropy), max(subreddit_entropy) FROM account_features').fetchdf())
    print('\nremoval_rate by n_comments_sample bucket (sanity check on the low-n=suspicious hypothesis):')
    print(con.execute("""
        SELECT CASE WHEN n_comments_sample=1 THEN '1 (singleton)'
                    WHEN n_comments_sample BETWEEN 2 AND 4 THEN '2-4'
                    WHEN n_comments_sample BETWEEN 5 AND 9 THEN '5-9'
                    ELSE '10+' END AS bucket,
               count(*) AS n_accounts, avg(removal_rate) AS mean_removal_rate
        FROM account_features GROUP BY bucket ORDER BY bucket
    """).fetchdf())

    con.close()
    print(f'\nDONE in {time.time()-t0:.1f}s.')


if __name__ == '__main__':
    main()
