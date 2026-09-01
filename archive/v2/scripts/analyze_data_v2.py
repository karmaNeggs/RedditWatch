#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Scoring Engine
Reads data/v2/posts_YYYY-MM.csv + data/v2/commenters_YYYY-MM.csv
Writes output/v2/analysis_YYYY-MM.json

Usage:
  python3 scripts/analyze_data_v2.py                  # latest data
  python3 scripts/analyze_data_v2.py --month 2026-01  # specific month
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / 'data' / 'v2'
OUTPUT_DIR  = ROOT / 'output' / 'v2'
FINDINGS    = ROOT / 'reports' / 'findings.json'
RISK_SCORES = OUTPUT_DIR / 'account_risk_scores.csv'

NEW_ACCOUNT_DAYS = 90   # accounts younger than this are "new"
KPD_SUSPICIOUS   = 200  # lowered from 500 — analysis shows 500 was too conservative
KPD_VERY_SUSP    = 1000

# final_score used to be a weighted blend of 6 hand-tuned heuristic components
# (see git history — analyze_accounts/analyze_comment_ring/analyze_engagement/
# analyze_temporal/analyze_distribution/analyze_network below). None of those
# per-component weights were ever individually checked against real evidence —
# only bundled candidate schemes were (referee_weights.py, retired). final_score
# is now driven entirely by score_accounts.py's validated account-risk model
# (see analyze_account_risk / RISK_SCORES below); the six analyzers still run
# and their output is still published per-sub, but purely as diagnostic detail
# explaining *why* a sub's score moved — not as inputs to the score itself.


def _load_severity_bands():
    """Percentile-based bands from score_accounts.py, derived from the actual
    observed pct_high_risk distribution — falls back to the legacy fixed
    20/40/70 thresholds (meaningless on this 0-100 scale, but a safe default)
    if score_accounts.py hasn't been run yet."""
    defaults = {'moderate': 20.0, 'high': 40.0, 'critical': 70.0}
    if FINDINGS.exists():
        try:
            with open(FINDINGS) as f:
                d = json.load(f)
            bands = d.get('severity_bands', {})
            if all(k in bands for k in ('moderate', 'high', 'critical')):
                return bands
        except Exception:
            pass
    print("  Using default severity bands (run score_accounts.py to calibrate real ones)")
    return defaults


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(month: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if month:
        posts_f = DATA_DIR / f'posts_{month}.csv'
        comms_f = DATA_DIR / f'commenters_{month}.csv'
    else:
        posts_f = DATA_DIR / 'posts_latest.csv'
        comms_f = DATA_DIR / 'commenters_latest.csv'

    if not posts_f.exists():
        raise FileNotFoundError(f'{posts_f} not found. Run collect_data_v2.py first.')

    posts = pd.read_csv(posts_f)
    comms = pd.read_csv(comms_f)

    # sample_type only exists in months collected after the Phase 3 survivorship-
    # bias fix (collect_data_v2.py's fetch_posts_random) — months collected
    # before that are entirely 'top' posts, so backfill the column rather than
    # branch on its presence everywhere downstream.
    if 'sample_type' not in posts.columns:
        posts['sample_type'] = 'top'

    # Derived columns
    posts['kpd']         = (posts['author_link_karma'] + posts['author_comment_karma']) / posts['author_account_age_days'].clip(lower=1)
    posts['link_ratio']  = posts['author_link_karma'] / (posts['author_comment_karma'].clip(lower=0) + 1)
    posts['post_age_h']  = (posts['collected_utc'] - posts['created_utc']) / 3600
    posts['score_per_h'] = posts['score'] / posts['post_age_h'].clip(lower=1)
    posts['is_new_acct'] = posts['author_account_age_days'] < NEW_ACCOUNT_DAYS

    comms['kpd']         = (comms['author_link_karma'] + comms['author_comment_karma']) / comms['author_account_age_days'].clip(lower=1)
    comms['link_ratio']  = comms['author_link_karma'] / (comms['author_comment_karma'].clip(lower=0) + 1)
    comms['is_new_acct'] = comms['author_account_age_days'] < NEW_ACCOUNT_DAYS

    return posts, comms


# ── Component 1: Account Signals (30%) ───────────────────────────────────────
# Poster + commenter account health: kpd, link_ratio, new accounts, unverified

def analyze_accounts(posts: pd.DataFrame, comms: pd.DataFrame) -> dict:
    result = {}
    for sub in posts['subreddit'].unique():
        p = posts[posts['subreddit'] == sub].dropna(subset=['kpd'])
        c = comms[(comms['subreddit'] == sub) & comms['in_top10']].dropna(subset=['kpd'])

        # --- Poster signals ---
        n_post = len(p)
        if n_post == 0:
            poster_score = 0.0
        else:
            susp_mask  = (p['kpd'] > KPD_SUSPICIOUS) | ((p['author_account_age_days'] < 90) & (p['kpd'] > 100))
            vsusp_mask = p['kpd'] > KPD_VERY_SUSP
            susp_pct   = susp_mask.sum() / n_post * 100
            vsusp_pct  = vsusp_mask.sum() / n_post * 100
            # High link_ratio: >10 means karma-farmer (posts links, doesn't discuss)
            high_lr_pct = (p['link_ratio'] > 10).sum() / n_post * 100
            # New accounts (<90 days)
            new_pct = p['is_new_acct'].mean() * 100
            poster_score = min(susp_pct * 1.0 + vsusp_pct * 0.3 + high_lr_pct * 0.5 + new_pct * 0.4, 100)

        # --- Commenter signals (top10 commenters) ---
        n_comm = len(c)
        if n_comm == 0:
            commenter_score = 0.0
            unverified_pct  = 0.0
            new_comm_pct    = 0.0
        else:
            susp_c = (c['kpd'] > KPD_SUSPICIOUS) | ((c['author_account_age_days'] < 90) & (c['kpd'] > 100))
            vsusp_c = c['kpd'] > KPD_VERY_SUSP
            susp_cpct  = susp_c.sum() / n_comm * 100
            vsusp_cpct = vsusp_c.sum() / n_comm * 100
            high_lr_cpct = (c['link_ratio'] > 10).sum() / n_comm * 100
            new_comm_pct  = c['is_new_acct'].mean() * 100
            # Unverified email — treat None as unknown (ignore), False as suspect
            known_verified = c['author_verified_email'].dropna()
            unverified_pct = (known_verified == False).sum() / len(known_verified) * 100 if len(known_verified) > 0 else 0
            commenter_score = min(susp_cpct * 1.0 + vsusp_cpct * 0.3 + high_lr_cpct * 0.5
                                  + new_comm_pct * 0.4 + unverified_pct * 0.3, 100)

        # Commenter given 60% weight (user requested), poster 40%
        if n_comm > 0:
            account_score = poster_score * 0.40 + commenter_score * 0.60
        else:
            account_score = poster_score

        result[sub] = {
            'account_score':      round(float(account_score), 1),
            'poster_score':       round(float(poster_score), 1),
            'commenter_score':    round(float(commenter_score), 1),
            'new_poster_pct':     round(float(p['is_new_acct'].mean() * 100) if n_post else 0, 1),
            'new_commenter_pct':  round(float(new_comm_pct), 1),
            'unverified_comm_pct': round(float(unverified_pct), 1),
            'n_posters':          int(n_post),
            'n_top_commenters':   int(n_comm),
        }
    return result


# ── Component 2: Comment Ring Detection (25%) ─────────────────────────────────
# First-vs-top overlap, early timing burst, commenter recurrence, self-amplification

def analyze_comment_ring(posts: pd.DataFrame, comms: pd.DataFrame) -> dict:
    result = {}
    for sub in posts['subreddit'].unique():
        sub_posts = posts[posts['subreddit'] == sub]
        sub_comms = comms[comms['subreddit'] == sub]

        # -- First-vs-top overlap: same accounts in first5 AND top10 --------
        # High overlap = ring promotes early comments to top
        overlaps = []
        for pid in sub_posts['post_id']:
            pc = sub_comms[sub_comms['post_id'] == pid]
            top_authors   = set(pc[pc['in_top10']]['author'])
            first_authors = set(pc[pc['in_first5']]['author'])
            if first_authors:
                overlap = len(top_authors & first_authors) / len(first_authors)
                overlaps.append(overlap)
        overlap_rate = float(np.mean(overlaps)) if overlaps else 0.0

        # -- Early burst: std of first-5 timestamps (low std = tight cluster) --
        # Normalise: burst_score = max(0, 1 - std_minutes/30) * 100
        burst_scores = []
        for pid in sub_posts['post_id']:
            pc = sub_comms[(sub_comms['post_id'] == pid) & sub_comms['in_first5']]
            ts = pc['comment_created_utc'].dropna()
            if len(ts) >= 3:
                post_ts = sub_posts.loc[sub_posts['post_id'] == pid, 'created_utc'].values[0]
                deltas_min = (ts - post_ts) / 60
                std_min = float(deltas_min.std())
                burst_scores.append(max(0.0, 1.0 - std_min / 30.0) * 100)
        burst_score = float(np.mean(burst_scores)) if burst_scores else 0.0

        # Avg time to first comment (minutes)
        ttfc_list = []
        for pid in sub_posts['post_id']:
            pc = sub_comms[(sub_comms['post_id'] == pid) & sub_comms['in_first5']]
            ts = pc['comment_created_utc'].dropna()
            if len(ts) > 0:
                post_ts = sub_posts.loc[sub_posts['post_id'] == pid, 'created_utc'].values[0]
                ttfc_list.append(float((ts.min() - post_ts) / 60))
        avg_ttfc_min = round(float(np.mean(ttfc_list)), 1) if ttfc_list else None

        # -- Commenter recurrence: accounts in 3+ posts of same sub -----------
        all_comms_sub = sub_comms[sub_comms['in_top10']]
        if len(all_comms_sub) > 0:
            recur = all_comms_sub.groupby('author')['post_id'].nunique()
            recurring_authors = recur[recur >= 3]
            posts_with_recur  = all_comms_sub[all_comms_sub['author'].isin(recurring_authors.index)]['post_id'].nunique()
            recurrence_rate   = posts_with_recur / len(sub_posts) if len(sub_posts) > 0 else 0.0
        else:
            recurrence_rate = 0.0

        # -- Self-amplification: post author in first5 comments ---------------
        self_amp_posts = 0
        for pid in sub_posts['post_id']:
            post_author = sub_posts.loc[sub_posts['post_id'] == pid, 'author'].values[0]
            pc = sub_comms[(sub_comms['post_id'] == pid) & sub_comms['in_first5']]
            if pc['is_submitter'].any() or (post_author in pc['author'].values):
                self_amp_posts += 1
        self_amp_rate = self_amp_posts / len(sub_posts) if len(sub_posts) > 0 else 0.0

        # Fast time-to-first-comment: % posts where first comment < 5 min
        fast_ttfc_pct = 0.0
        if ttfc_list:
            fast_ttfc_pct = sum(1 for t in ttfc_list if t < 5) / len(ttfc_list) * 100

        # Ring score — overlap_rate removed (proven organic, r≈0.075 with final)
        # burst_score is the strongest ring signal; fast_ttfc and recurrence are secondary
        ring_score = min(
            burst_score     * 0.60 +   # up to 60 pts: coordinated tight timing
            fast_ttfc_pct   * 0.25 +   # up to 25 pts: sub-5-min first comments
            recurrence_rate * 10  +    # up to 10 pts: same commenters across posts
            self_amp_rate   * 5,       # up to  5 pts: poster in early comments
            100
        )

        result[sub] = {
            'ring_score':          round(float(ring_score), 1),
            'burst_score':         round(float(burst_score), 1),
            'fast_ttfc_pct':       round(float(fast_ttfc_pct), 1),
            'avg_ttfc_minutes':    avg_ttfc_min,
            'recurrence_rate':     round(float(recurrence_rate), 3),
            'self_amp_rate':       round(float(self_amp_rate), 3),
            'overlap_rate':        round(float(overlap_rate), 3),  # retained for reference, not scored
        }
    return result


# ── Component 3: Engagement Structure (20%) ───────────────────────────────────
# Score-comment correlation, upvote ratio variance, awards, score velocity

def analyze_engagement(posts: pd.DataFrame) -> dict:
    result = {}
    for sub in posts['subreddit'].unique():
        p = posts[posts['subreddit'] == sub]
        if len(p) < 5:
            continue

        corr          = float(p['score'].corr(p['num_comments']))
        ratio_std     = float(p['upvote_ratio'].std())
        avg_awards    = float(p['total_awards'].mean())
        ucr           = float(p['score'].mean() / max(p['num_comments'].mean(), 1))
        # Simulacra: high score, near-zero comments — pure upvote manipulation (finding F9)
        simulacra_rate = float(((p['score'] > 500) & (p['num_comments'] < 5)).mean() * 100)

        # Low correlation = upvotes without discussion (threshold 0.30)
        corr_pts  = max(0, min((0.30 - corr) / 0.30 * 50, 50)) if corr < 0.30 else 0
        # Uniform upvote ratios (threshold std < 0.025)
        ratio_pts = max(0, min((0.025 - ratio_std) / 0.025 * 30, 30)) if ratio_std < 0.025 else 0
        # High UCR (upvote-to-comment; ceiling at 30)
        ucr_pts   = min((ucr / 30) * 40, 40)
        # Awards: systematic gifting (>0.5 avg awards/post is suspicious)
        award_pts = min(avg_awards / 0.5 * 10, 10)
        # Simulacra rate: >10% of posts with this pattern is suspicious
        simulacra_pts = min(simulacra_rate / 10 * 20, 20)

        eng_score = min(corr_pts + ratio_pts + ucr_pts + award_pts + simulacra_pts, 100)

        result[sub] = {
            'engagement_score':   round(float(eng_score), 1),
            'score_comment_corr': round(corr, 3),
            'upvote_ratio_std':   round(ratio_std, 4),
            'ucr':                round(ucr, 1),
            'avg_awards':         round(avg_awards, 2),
            'simulacra_rate':     round(simulacra_rate, 1),
        }
    return result


# ── Component 4: Temporal Patterns (15%) ─────────────────────────────────────
# Post interval regularity + hour concentration + entropy

def analyze_temporal(posts: pd.DataFrame) -> dict:
    df = posts.copy()
    df['hour'] = pd.to_datetime(df['created_utc'], unit='s').dt.hour
    result = {}

    for sub in df['subreddit'].unique():
        p = df[df['subreddit'] == sub].sort_values('created_utc')

        # Interval regularity
        if len(p) >= 4:
            ivs  = p['created_utc'].diff().dropna().values.astype(float)
            mean_iv = float(np.mean(ivs))
            iv_cv   = float(np.std(ivs) / mean_iv) if mean_iv > 0 else 0.0
        else:
            iv_cv, mean_iv = 1.0, 0.0

        # Hour concentration
        hc        = p['hour'].value_counts()
        top3_conc = float(hc.nlargest(3).sum() / len(p) * 100)
        probs     = hc / len(p)
        entropy   = float(-np.sum(probs * np.log2(probs + 1e-10)))

        conc_pts  = min(top3_conc * 1.0, 60)
        entr_pts  = max(0, (1.0 - entropy / np.log2(24)) * 40)
        reg_pts   = max(0, min((0.85 - iv_cv) / 0.85 * 30, 30))  # regularity bonus

        temporal_score = min(conc_pts * 0.70 + entr_pts * 0.70 + reg_pts, 100)

        result[sub] = {
            'temporal_score':     round(float(temporal_score), 1),
            'interval_cv':        round(iv_cv, 3),
            'top3_concentration': round(top3_conc, 1),
            'entropy':            round(entropy, 2),
            'peak_hour_utc':      int(hc.idxmax()),
        }
    return result


# ── Component 6 (observational — not yet weighted into final_score): Network/Text ──
# Near-duplicate titles, cross-sub account overlap, vote concentration (Gini).
# Computed and exposed here so it's visible in the output, but the production
# ensemble intentionally does not fold it in yet — adding a 6th component changes
# every historical score and needs a recalibration pass + review first, not a
# silent change. See reports/findings.json component_signals for the 5 that are live.

def _gini(values) -> float:
    arr = np.sort(np.asarray([v for v in values if v is not None and v >= 0], dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _title_words(title) -> set:
    return set(str(title).lower().split())



# Empirically calibrated from the full 13-month/25-sub corpus (n=325 sub-months),
# same percentile-calibration convention as the KPD/decay_slope thresholds.
# near_dupe_rate excluded from network_score: median 0, p95 0, max 2.4% across the
# entire corpus — the Jaccard>0.6 near-dupe detector essentially never fires at this
# sample size, so it would be dead weight in the composite. Kept for reference only.
_NETWORK_CROSS_SUB_P95 = 13.1
_NETWORK_GINI_P95      = 54.2


def analyze_network(posts: pd.DataFrame, comms: pd.DataFrame) -> dict:
    result = {}
    for sub in posts['subreddit'].unique():
        p = posts[posts['subreddit'] == sub]

        # -- Near-duplicate titles: Jaccard word-overlap > 0.6 between any post pair --
        titles = [_title_words(t) for t in p['title']]
        dupe_pairs, total_pairs = 0, 0
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                a, b = titles[i], titles[j]
                if not a or not b:
                    continue
                total_pairs += 1
                jac = len(a & b) / len(a | b)
                if jac > 0.6:
                    dupe_pairs += 1
        near_dupe_rate = (dupe_pairs / total_pairs * 100) if total_pairs > 0 else 0.0

        # -- Cross-sub account overlap: % of this sub's active accounts also active
        #    (poster or commenter) in >=1 other subreddit this same month --
        sub_posters    = set(p['author'].dropna())
        sub_commenters = set(comms[comms['subreddit'] == sub]['author'].dropna())
        sub_accounts   = sub_posters | sub_commenters

        other_posters    = set(posts[posts['subreddit'] != sub]['author'].dropna())
        other_commenters = set(comms[comms['subreddit'] != sub]['author'].dropna())
        other_accounts   = other_posters | other_commenters

        cross_sub_rate = (len(sub_accounts & other_accounts) / len(sub_accounts) * 100) if sub_accounts else 0.0

        # -- Vote concentration: Gini of post scores within the sub-month --
        gini_score = _gini(p['score'].tolist()) * 100

        # -- Composite (excludes near_dupe_rate — see module note above) ------
        cross_pts = min(cross_sub_rate / _NETWORK_CROSS_SUB_P95 * 50, 50)
        gini_pts  = min(gini_score / _NETWORK_GINI_P95 * 50, 50)
        network_score = min(cross_pts + gini_pts, 100)

        result[sub] = {
            'network_score':   round(float(network_score), 1),
            'near_dupe_rate':  round(float(near_dupe_rate), 1),  # reference only, not scored
            'cross_sub_rate':  round(float(cross_sub_rate), 1),
            'gini_score':      round(float(gini_score), 1),
            'n_title_pairs':   int(total_pairs),
        }
    return result


# ── Component 5: Vote Distribution (10%) ──────────────────────────────────────
# Score CV, comment depth distribution (shallow = bot-like)

def analyze_distribution(posts: pd.DataFrame, comms: pd.DataFrame) -> dict:
    """
    score_cv (raw score variance across the sub's top-40) used to drive this
    component, but collection sorts by top.json score and caps at 40/month —
    every subreddit's top-N slice is variance-truncated by that ranking-then-cutting
    process itself, bot activity aside, so a low score_cv doesn't distinguish
    "this sub's top posts are uniform because of coordination" from "this is what
    any top-40 slice looks like". Kept below for reference, not scored.

    Replaced with decay_slope: fit log(score) ~ log(rank) across the sub-month's
    top-N (a power-law/Zipfian decay is the organic shape — one or two breakout
    posts, long tail). An unnaturally *flat* slope (several posts pinned near-equal
    near the top) is a shape signature of coordinated amplification that survives
    the top-N truncation, since it's about the curve's shape, not its raw variance.
    Thresholds calibrated from the actual observed slope distribution across the
    full 13-month/25-sub corpus (n=304 sub-months): p10=-0.93 (steep/organic) to
    max=-0.22 (flattest observed) — same percentile-calibration approach as the
    KPD threshold.
    """
    result = {}
    for sub in posts['subreddit'].unique():
        p = posts[posts['subreddit'] == sub]
        c = comms[(comms['subreddit'] == sub) & comms['in_top10']]

        score_mean = p['score'].mean()
        score_cv   = p['score'].std() / score_mean if score_mean > 0 else 0
        comm_mean  = p['num_comments'].mean()
        comm_cv    = p['num_comments'].std() / comm_mean if comm_mean > 0 else 0
        comm_uniformity  = max(0, min((0.8 - comm_cv)  / 0.8 * 30, 30))

        # -- Rank-decay shape (replaces score_cv-based scoring) --------------
        scores_sorted = p.sort_values('score', ascending=False)['score'].values
        scores_sorted = scores_sorted[scores_sorted > 0]
        decay_slope, decay_r2 = None, None
        decay_pts = 0.0
        if len(scores_sorted) >= 8:
            ranks = np.arange(1, len(scores_sorted) + 1)
            log_r, log_s = np.log(ranks), np.log(scores_sorted)
            slope, intercept = np.polyfit(log_r, log_s, 1)
            resid  = log_s - (slope * log_r + intercept)
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((log_s - log_s.mean()) ** 2))
            decay_slope = float(slope)
            decay_r2    = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            # p10=-0.93 (steep, organic) .. max=-0.22 (flattest observed = most suspicious)
            decay_pts = max(0.0, min((decay_slope - (-0.93)) / ((-0.22) - (-0.93)) * 50, 50))

        # Comment depth: low avg depth = all bot-like direct replies
        if len(c) > 0:
            avg_depth = float(c['comment_depth'].mean())
            depth_pts = max(0, min((0.5 - avg_depth) / 0.5 * 20, 20))
        else:
            avg_depth, depth_pts = 0.0, 0.0

        dist_score = min(decay_pts + comm_uniformity + depth_pts, 100)

        result[sub] = {
            'distribution_score': round(float(dist_score), 1),
            'decay_slope':        round(decay_slope, 3) if decay_slope is not None else None,
            'decay_r2':           round(decay_r2, 3) if decay_r2 is not None else None,
            'comments_cv':        round(float(comm_cv), 3),
            'avg_comment_depth':  round(avg_depth, 2),
            'score_cv':           round(float(score_cv), 3),  # reference only, not scored — see docstring
        }
    return result


# ── Account-risk rollup (drives final_score) ───────────────────────────────────
# Production score from score_accounts.py's validated model, rolled up to this
# month's (subreddit, month) activity. See module docstring at top of file for
# why this replaced the 6-component weighted blend.
#
# Runs on posts_top ONLY (see main()) — deliberately, not the fuller pool
# Phase 3 collects. The random supplement's job is calibration/diagnostic
# input (see analyze_baseline_comparison below), not part of what gets
# reported: top posts are what the subreddit's audience actually sees, so
# that's the population final_score should describe. Scoring the full pool
# would also make every month before/after the Phase 3 collection change
# incomparable — mixing "more accounts counted" into a trend line alongside
# real month-to-month change.
#
# risk_month (account, risk_score, high_risk) is computed ONCE in main() via
# anomaly_detection.month_relative_high_risk and threaded into every function
# below that needs it — "high risk" is month-relative (top decile of THIS
# month's active accounts), not a fixed global population threshold. See
# that function's docstring for why: a global threshold quietly tracks
# population composition drift (the active population trending younger over
# 2025-06→2026-07) rather than actual relative risk.

def analyze_account_risk(posts: pd.DataFrame, comms: pd.DataFrame, risk_month: pd.DataFrame) -> dict:
    if risk_month is None:
        print(f"  WARNING: {RISK_SCORES} not found — run score_accounts.py first. "
              f"final_score will be 0 for all subs this run.")
        return {sub: {'final_score': 0.0, 'pct_high_risk_activity': None, 'n_activity_rows': 0}
                for sub in posts['subreddit'].unique()}

    from anomaly_detection import rollup_to_subreddit
    rollup = rollup_to_subreddit(posts, comms, risk_month, score_col='high_risk')
    rollup = rollup.set_index('subreddit')

    result = {}
    for sub in posts['subreddit'].unique():
        if sub in rollup.index:
            row = rollup.loc[sub]
            pct = float(row['avg_score']) * 100
            result[sub] = {
                'final_score':             round(pct, 1),
                'pct_high_risk_activity':  round(pct, 1),
                'n_activity_rows':         int(row['n_activity_rows']),
            }
        else:
            result[sub] = {'final_score': 0.0, 'pct_high_risk_activity': None, 'n_activity_rows': 0}
    return result


# ── Baseline comparison (diagnostic, calibration-only — not scored) ───────────
# Uses the Phase 3 random supplement (posts_random) purely to answer "does
# reaching top status correlate with high-risk posters, relative to an
# unfiltered baseline?" — NOT to expand what final_score counts. Only
# meaningful for months with a random supplement (Phase 3 onward); returns
# None for earlier months rather than a misleading 0.

def analyze_baseline_comparison(posts: pd.DataFrame, risk_month: pd.DataFrame) -> dict:
    if risk_month is None:
        return {}
    p = posts.merge(risk_month[['account', 'high_risk']], left_on='author', right_on='account', how='left')

    result = {}
    for sub in posts['subreddit'].unique():
        sub_p = p[p['subreddit'] == sub]
        top_p    = sub_p[sub_p['sample_type'] == 'top']
        random_p = sub_p[sub_p['sample_type'] == 'random']

        if len(top_p) == 0 or len(random_p) < 5:  # too few random posts for a stable baseline
            result[sub] = {'top_poster_high_risk_pct': None, 'baseline_poster_high_risk_pct': None,
                            'top_vs_baseline_risk_ratio': None, 'n_random_sampled': int(len(random_p))}
            continue

        top_rate    = float(top_p['high_risk'].mean() * 100)
        random_rate = float(random_p['high_risk'].mean() * 100)
        ratio = round(top_rate / random_rate, 2) if random_rate > 0 else None

        result[sub] = {
            'top_poster_high_risk_pct':      round(top_rate, 1),
            'baseline_poster_high_risk_pct': round(random_rate, 1),
            'top_vs_baseline_risk_ratio':    ratio,
            'n_random_sampled':              int(len(random_p)),
        }
    return result


# ── Post-level coordination (diagnostic — activity BY vs. SUPPORTED BY bots) ──
# Ports V1's astroturf_density (archive/v1/scripts/analyze_data.py — dropped
# in the V2 rebuild) onto our validated high_risk flag instead of V1's old
# unvalidated kpd>500 heuristic. Answers a different question than
# final_score: not "how much activity comes from high-risk accounts" but
# "how many POSTS have a high-risk footprint, and where". commenter_only_risk
# is the "organic-looking post, suspicious support" case specifically —
# activity SUPPORTED BY bot-like accounts, not BY them.
#
# Denominator is posts that actually got comment-sampled (COMMENT_SAMPLE=10/
# month in collect_data_v2.py), not all 40 posts — V1 could use "all posts"
# as the denominator because it fetched commenters for every post; V2 only
# comment-samples the top 10, so using all 40 here would silently deflate
# every percentage by counting un-sampled posts as automatic "no".

def analyze_post_coordination(posts: pd.DataFrame, comms: pd.DataFrame, risk_month: pd.DataFrame) -> dict:
    if risk_month is None:
        return {}
    high_risk_set = set(risk_month[risk_month['high_risk'] == 1]['account'])

    result = {}
    for sub in posts['subreddit'].unique():
        sub_posts = posts[posts['subreddit'] == sub].set_index('post_id')
        sub_comms = comms[(comms['subreddit'] == sub) & comms['in_top10']]
        sampled_pids = sub_comms['post_id'].unique()

        n = len(sampled_pids)
        if n == 0:
            continue

        fully_coord = poster_only = commenter_only = any_bad_commenter = 0
        for pid in sampled_pids:
            if pid not in sub_posts.index:
                continue
            poster_bad = sub_posts.loc[pid, 'author'] in high_risk_set

            commenters = set(sub_comms[sub_comms['post_id'] == pid]['author'])
            bad_commenters = commenters & high_risk_set
            commenter_majority_bad = bool(commenters) and len(bad_commenters) / len(commenters) >= 0.5

            if bad_commenters:
                any_bad_commenter += 1
            if poster_bad and commenter_majority_bad:
                fully_coord += 1
            elif poster_bad:
                poster_only += 1
            elif commenter_majority_bad:
                commenter_only += 1

        result[sub] = {
            'pct_posts_fully_coordinated':       round(fully_coord / n * 100, 1),
            'pct_posts_poster_only_risk':         round(poster_only / n * 100, 1),
            'pct_posts_commenter_only_risk':      round(commenter_only / n * 100, 1),  # "supported by bots"
            'pct_posts_any_high_risk_commenter':  round(any_bad_commenter / n * 100, 1),
            'n_posts_sampled_for_coordination':   n,
        }
    return result


# ── Co-occurrence network (diagnostic) ────────────────────────────────────────
# Do the SAME pairs of top-10 commenters keep showing up together across
# DIFFERENT posts in a sub-month? Coordinated-behavior literature treats
# repeated co-occurrence on shared targets as a stronger network signal than
# simple activity overlap (our existing cross_sub_rate/recurrence_rate),
# since real rings act together repeatedly, not just independently-prolific
# accounts that happen to both be active a lot.

def analyze_cooccurrence(posts: pd.DataFrame, comms: pd.DataFrame) -> dict:
    from itertools import combinations
    from collections import Counter

    result = {}
    for sub in comms['subreddit'].unique():
        sub_comms = comms[(comms['subreddit'] == sub) & comms['in_top10']]
        post_groups = sub_comms.groupby('post_id')['author'].apply(set)
        n_posts = len(post_groups)
        if n_posts < 2:
            result[sub] = {'repeat_pair_rate': 0.0, 'n_repeat_pairs': 0, 'max_pair_cooccurrence': 0}
            continue

        pair_counts = Counter()
        for authors in post_groups:
            for pair in combinations(sorted(authors), 2):
                pair_counts[pair] += 1

        repeat_pairs = {pair for pair, c in pair_counts.items() if c >= 2}
        posts_with_repeat_pair = sum(
            1 for authors in post_groups
            if any(pair in repeat_pairs for pair in combinations(sorted(authors), 2))
        )

        result[sub] = {
            'repeat_pair_rate':      round(posts_with_repeat_pair / n_posts * 100, 1),
            'n_repeat_pairs':        len(repeat_pairs),
            'max_pair_cooccurrence': max(pair_counts.values()) if pair_counts else 0,
        }
    return result


# ── Unified scoring ───────────────────────────────────────────────────────────

SEVERITY_BANDS = _load_severity_bands()

def severity(score: float) -> str:
    if score >= SEVERITY_BANDS['critical']: return 'CRITICAL'
    if score >= SEVERITY_BANDS['high']:     return 'HIGH'
    if score >= SEVERITY_BANDS['moderate']: return 'MODERATE'
    return 'LOW'

COMPONENT_SCORE_KEY = {
    'account':      'account_score',
    'ring':         'ring_score',
    'engagement':   'engagement_score',
    'temporal':     'temporal_score',
    'distribution': 'distribution_score',
    'network':      'network_score',
}

def calculate_scores(risk, acct, ring, eng, temp, dist, net, coord, cooc, baseline) -> dict:
    """final_score comes entirely from `risk` (score_accounts.py's validated
    account-risk rollup, top-sample only). acct/ring/eng/temp/dist/net/coord/
    cooc/baseline are all diagnostic detail attached to each row, not weighted
    into it, so the site/narrative layer can explain *why* a sub's score moved
    without those components silently double-counting into the number itself."""
    components = {'account': acct, 'ring': ring, 'engagement': eng,
                  'temporal': temp, 'distribution': dist, 'network': net}

    scores = {}
    for sub, r in risk.items():
        row = {
            'final_score':            r['final_score'],
            'pct_high_risk_activity': r['pct_high_risk_activity'],
            'n_activity_rows':        r['n_activity_rows'],
        }
        for name, key in COMPONENT_SCORE_KEY.items():
            if sub in components[name]:
                row[key] = components[name][sub][key]
        if sub in coord:
            row['pct_posts_fully_coordinated']  = coord[sub]['pct_posts_fully_coordinated']
            row['pct_posts_commenter_only_risk'] = coord[sub]['pct_posts_commenter_only_risk']
        if sub in cooc:
            row['repeat_pair_rate'] = cooc[sub]['repeat_pair_rate']
        if sub in baseline:
            row['top_vs_baseline_risk_ratio'] = baseline[sub]['top_vs_baseline_risk_ratio']
        scores[sub] = row
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--month', default=None)
    return p.parse_args()


def main():
    args  = parse_args()
    month = args.month

    print("\n" + "=" * 70)
    print("REDDIT BOT ANALYSIS V2 — SCORING ENGINE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    posts, comms = load_data(month)
    detected_month = month or posts['collection_month'].iloc[0]
    n_random = int((posts['sample_type'] == 'random').sum())
    print(f"\nData: {len(posts)} posts ({n_random} supplementary non-top)  {len(comms)} comment-rows  "
          f"month={detected_month}  subs={posts['subreddit'].nunique()}")

    # final_score and the six diagnostic components all run on posts_top only —
    # top posts are what the subreddit's audience actually sees, and it's what
    # every month before Phase 3 already was, so this keeps the whole trend
    # line comparable. The six components additionally have magic-number
    # thresholds (decay_slope percentiles, KPD cutoffs, Gini percentile)
    # calibrated on this exact distribution — mixing in random posts there
    # would silently invalidate those thresholds regardless. The random
    # supplement (posts_random) is calibration-only input to
    # analyze_baseline_comparison — never part of what's reported.
    posts_top    = posts[posts['sample_type'] == 'top']
    posts_random = posts[posts['sample_type'] == 'random']

    # "High risk" computed once, relative to THIS month's active accounts —
    # not a fixed global population threshold. See month_relative_high_risk's
    # docstring (anomaly_detection.py) for why. Threaded into every function
    # below that needs it so they all use the exact same yardstick.
    risk_month = None
    if RISK_SCORES.exists():
        from anomaly_detection import month_relative_high_risk
        risk_scores = pd.read_csv(RISK_SCORES)[['account', 'risk_score']]
        risk_month, risk_threshold = month_relative_high_risk(posts_top, comms, risk_scores)
        print(f"  Month-relative high-risk threshold: {risk_threshold:.1f} "
              f"(top 10% of this month's active accounts)")

    print("\nRunning components…")
    risk     = analyze_account_risk(posts_top, comms, risk_month);   print("  Account-risk rollup done (drives final_score; top-sample only)")
    baseline = analyze_baseline_comparison(posts, risk_month);        print("  Baseline comparison done (diagnostic only, calibration use of random supplement)")
    acct  = analyze_accounts(posts_top, comms);      print("  Account signals done (diagnostic only, top-sample only)")
    ring  = analyze_comment_ring(posts_top, comms);  print("  Comment ring detection done (diagnostic only, top-sample only)")
    eng   = analyze_engagement(posts_top);           print("  Engagement structure done (diagnostic only, top-sample only)")
    temp  = analyze_temporal(posts_top);             print("  Temporal patterns done (diagnostic only, top-sample only)")
    dist  = analyze_distribution(posts_top, comms);  print("  Vote distribution done (diagnostic only, top-sample only)")
    net   = analyze_network(posts_top, comms);       print("  Network/text signals done (diagnostic only, top-sample only)")
    coord = analyze_post_coordination(posts, comms, risk_month); print("  Post-level coordination done (diagnostic only)")
    cooc  = analyze_cooccurrence(posts, comms);      print("  Co-occurrence network done (diagnostic only)")

    scores = calculate_scores(risk, acct, ring, eng, temp, dist, net, coord, cooc, baseline)

    print("\n" + "=" * 70)
    print("BOT ACTIVITY RANKINGS (V2)")
    print("=" * 70)
    for rank, (sub, sc) in enumerate(
        sorted(scores.items(), key=lambda x: x[1]['final_score'], reverse=True), 1
    ):
        sev = severity(sc['final_score'])
        print(f"\n#{rank} r/{sub:<25} Score: {sc['final_score']:5.1f}/100  [{sev}]  "
              f"({sc['n_activity_rows']} activity rows)")
        print(f"     [diagnostic] Acct:{sc.get('account_score', 0):5.1f}  Ring:{sc.get('ring_score', 0):5.1f}  "
              f"Eng:{sc.get('engagement_score', 0):5.1f}  Temp:{sc.get('temporal_score', 0):5.1f}  "
              f"Dist:{sc.get('distribution_score', 0):5.1f}")
        print(f"     [diagnostic] FullyCoord:{sc.get('pct_posts_fully_coordinated', 0):5.1f}%  "
              f"CommenterOnlyRisk:{sc.get('pct_posts_commenter_only_risk', 0):5.1f}%  "
              f"RepeatPairRate:{sc.get('repeat_pair_rate', 0):5.1f}%")

    output = {
        'version':          2,
        'analysis_date':    datetime.now().isoformat(),
        'month':            detected_month,
        'severity_bands':   SEVERITY_BANDS,
        'unified_scores':   scores,
        'account_risk_analysis': risk,
        'account_analysis': acct,
        'ring_analysis':    ring,
        'engagement_analysis': eng,
        'temporal_analysis':   temp,
        'distribution_analysis': dist,
        'network_analysis':    net,
        'coordination_analysis':  coord,
        'cooccurrence_analysis':  cooc,
        'baseline_comparison':    baseline,
    }

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = OUTPUT_DIR / f'analysis_{detected_month}_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(OUTPUT_DIR / 'analysis_latest.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved: {out_path}")
    print("=" * 70 + "\n")
    return str(out_path)


if __name__ == '__main__':
    main()
