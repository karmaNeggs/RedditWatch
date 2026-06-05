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

NEW_ACCOUNT_DAYS = 90   # accounts younger than this are "new"
KPD_SUSPICIOUS   = 500  # karma/day threshold (top ~10% observed)
KPD_VERY_SUSP    = 2000 # karma/day threshold (top ~5% observed)


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

        # -- Ring score --------------------------------------------------------
        ring_score = min(
            overlap_rate   * 40   +   # up to 40 pts: same accounts both early & promoted
            burst_score    * 0.30 +   # up to 30 pts: tight timing cluster
            recurrence_rate * 20  +   # up to 20 pts: same commenters across many posts
            self_amp_rate  * 10,      # up to 10 pts: poster also in early comments
            100
        )

        result[sub] = {
            'ring_score':          round(float(ring_score), 1),
            'overlap_rate':        round(float(overlap_rate), 3),
            'burst_score':         round(float(burst_score), 1),
            'avg_ttfc_minutes':    avg_ttfc_min,
            'recurrence_rate':     round(float(recurrence_rate), 3),
            'self_amp_rate':       round(float(self_amp_rate), 3),
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

        # Low correlation = upvotes without discussion (threshold 0.30)
        corr_pts  = max(0, min((0.30 - corr) / 0.30 * 50, 50)) if corr < 0.30 else 0
        # Uniform upvote ratios (threshold std < 0.025)
        ratio_pts = max(0, min((0.025 - ratio_std) / 0.025 * 30, 30)) if ratio_std < 0.025 else 0
        # High UCR (upvote-to-comment; ceiling at 30)
        ucr_pts   = min((ucr / 30) * 40, 40)
        # Awards: systematic gifting (>0.5 avg awards/post is suspicious)
        award_pts = min(avg_awards / 0.5 * 10, 10)

        eng_score = min(corr_pts + ratio_pts + ucr_pts + award_pts, 100)

        result[sub] = {
            'engagement_score':   round(float(eng_score), 1),
            'score_comment_corr': round(corr, 3),
            'upvote_ratio_std':   round(ratio_std, 4),
            'ucr':                round(ucr, 1),
            'avg_awards':         round(avg_awards, 2),
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


# ── Component 5: Vote Distribution (10%) ──────────────────────────────────────
# Score CV, comment depth distribution (shallow = bot-like)

def analyze_distribution(posts: pd.DataFrame, comms: pd.DataFrame) -> dict:
    result = {}
    for sub in posts['subreddit'].unique():
        p = posts[posts['subreddit'] == sub]
        c = comms[(comms['subreddit'] == sub) & comms['in_top10']]

        score_mean = p['score'].mean()
        score_cv   = p['score'].std() / score_mean if score_mean > 0 else 0
        comm_mean  = p['num_comments'].mean()
        comm_cv    = p['num_comments'].std() / comm_mean if comm_mean > 0 else 0

        score_uniformity = max(0, min((0.8 - score_cv) / 0.8 * 70, 70))
        comm_uniformity  = max(0, min((0.8 - comm_cv)  / 0.8 * 30, 30))

        # Comment depth: low avg depth = all bot-like direct replies
        if len(c) > 0:
            avg_depth = float(c['comment_depth'].mean())
            depth_pts = max(0, min((0.5 - avg_depth) / 0.5 * 20, 20))
        else:
            avg_depth, depth_pts = 0.0, 0.0

        dist_score = min(score_uniformity + comm_uniformity + depth_pts, 100)

        result[sub] = {
            'distribution_score': round(float(dist_score), 1),
            'score_cv':           round(float(score_cv), 3),
            'comments_cv':        round(float(comm_cv), 3),
            'avg_comment_depth':  round(avg_depth, 2),
        }
    return result


# ── Unified scoring ───────────────────────────────────────────────────────────

WEIGHTS = {'account': 0.30, 'ring': 0.25, 'engagement': 0.20,
           'temporal': 0.15, 'distribution': 0.10}

def severity(score: float) -> str:
    if score >= 70: return 'CRITICAL'
    if score >= 40: return 'HIGH'
    if score >= 20: return 'MODERATE'
    return 'LOW'

def calculate_scores(acct, ring, eng, temp, dist) -> dict:
    scores = {}
    for sub in acct:
        if not all(sub in d for d in [ring, eng, temp, dist]):
            continue
        final = (
            acct[sub]['account_score']      * WEIGHTS['account']      +
            ring[sub]['ring_score']          * WEIGHTS['ring']         +
            eng[sub]['engagement_score']     * WEIGHTS['engagement']   +
            temp[sub]['temporal_score']      * WEIGHTS['temporal']     +
            dist[sub]['distribution_score']  * WEIGHTS['distribution']
        )
        scores[sub] = {
            'final_score':        round(float(final), 1),
            'account_score':      acct[sub]['account_score'],
            'ring_score':         ring[sub]['ring_score'],
            'engagement_score':   eng[sub]['engagement_score'],
            'temporal_score':     temp[sub]['temporal_score'],
            'distribution_score': dist[sub]['distribution_score'],
        }
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
    print(f"\nData: {len(posts)} posts  {len(comms)} comment-rows  "
          f"month={detected_month}  subs={posts['subreddit'].nunique()}")

    print("\nRunning components…")
    acct = analyze_accounts(posts, comms);    print("  Account signals done")
    ring = analyze_comment_ring(posts, comms); print("  Comment ring detection done")
    eng  = analyze_engagement(posts);          print("  Engagement structure done")
    temp = analyze_temporal(posts);            print("  Temporal patterns done")
    dist = analyze_distribution(posts, comms); print("  Vote distribution done")

    scores = calculate_scores(acct, ring, eng, temp, dist)

    print("\n" + "=" * 70)
    print("BOT ACTIVITY RANKINGS (V2)")
    print("=" * 70)
    for rank, (sub, sc) in enumerate(
        sorted(scores.items(), key=lambda x: x[1]['final_score'], reverse=True), 1
    ):
        sev = severity(sc['final_score'])
        print(f"\n#{rank} r/{sub:<25} Score: {sc['final_score']:5.1f}/100  [{sev}]")
        print(f"     Acct:{sc['account_score']:5.1f}  Ring:{sc['ring_score']:5.1f}  "
              f"Eng:{sc['engagement_score']:5.1f}  Temp:{sc['temporal_score']:5.1f}  "
              f"Dist:{sc['distribution_score']:5.1f}")

    output = {
        'version':          2,
        'analysis_date':    datetime.now().isoformat(),
        'month':            detected_month,
        'weights':          WEIGHTS,
        'unified_scores':   scores,
        'account_analysis': acct,
        'ring_analysis':    ring,
        'engagement_analysis': eng,
        'temporal_analysis':   temp,
        'distribution_analysis': dist,
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
