#!/usr/bin/env python3
"""
Reddit Bot Analysis — Comprehensive Analysis Engine

§1 Data Portrait      coverage, quality, account stats
§2 Univariate         distributions of every raw signal
§3 Correlations       signal collinearity + discriminative power
§4 Text Intelligence  topics, pushed vs organic, near-dupes, cross-sub
§5 Cross-Sub Network  account overlap, Gini, churn, simulacra
§6 Event Calendar     score/volume spikes vs India news timeline
§7 Findings & Weights statistically derived weights + findings JSON

Output:
  reports/analysis_YYYYMMDD_HHMMSS.pdf
  reports/analysis_latest.pdf
  reports/findings.json   (consumed by analyze_data_v2.py for weights)

Usage:
  python3 scripts/analysis.py
  python3 scripts/analysis.py --months 2026-04 2026-05
"""

import argparse
import json
import re
import shutil
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data' / 'v2'
REPORT_DIR = ROOT / 'reports'

# ── Palette ───────────────────────────────────────────────────────────────────
C = dict(
    blue='#A8C8E8', rose='#F2B8C6', sage='#B8D8B8', amber='#F2D9A0',
    lav='#C8B8E8', slate='#8899AA', dark='#2C3E50', bg='#FAFAF8',
    grid='#EEEEEE', red='#E88080', green='#80C880', teal='#80C8C8',
)

SUB_CLUSTER = {
    'indiaspeaks': 'Political', 'unitedstatesofindia': 'Political',
    'india': 'Political', 'AskIndia': 'Political',
    'AskIndianWomen': 'Women/Social', 'TwoXIndia': 'Women/Social',
    'RelationshipIndia': 'Women/Social', 'indiasocial': 'Women/Social',
    'IndiaCricket': 'Cricket/Sports', 'ipl': 'Cricket/Sports', 'IndianGaming': 'Cricket/Sports',
    'IndiaInvestments': 'Finance', 'IndianStockMarket': 'Finance',
    'IndianStreetBets': 'Finance', 'personalfinanceindia': 'Finance', 'StartUpIndia': 'Finance',
    'BollyBlindsNGossip': 'Entertainment', 'bollywood': 'Entertainment', 'teenindia': 'Entertainment',
    'JEENEETards': 'Education', 'UPSC': 'Education', 'developersIndia': 'Education', 'IndiaTech': 'Education',
    'IndianFood': 'Lifestyle', 'ISRO': 'Science',
}
CLUSTER_COLOR = {
    'Political': '#F2B8C6', 'Women/Social': '#C8B8E8', 'Cricket/Sports': '#A8C8E8',
    'Finance': '#F2D9A0', 'Entertainment': '#B8D8B8', 'Education': '#D4C5B0',
    'Lifestyle': '#C8E8D0', 'Science': '#B8D8E8', 'Other': '#DDDDDD',
}
EVENT_CAT_COLOR = {
    'economic': '#F2D9A0', 'election': '#F2B8C6', 'cricket': '#A8C8E8',
    'political': '#C8B8E8', 'security': '#E88080',
}

EVENTS = [
    dict(date='2025-02-01', name='Union Budget 2025-26',         cat='economic',  subs=['india','IndiaInvestments','IndianStockMarket','personalfinanceindia']),
    dict(date='2025-02-08', name='Delhi Election Results',        cat='election',  subs=['india','indiaspeaks','unitedstatesofindia','AskIndia']),
    dict(date='2025-02-23', name='Champions Trophy IND vs PAK',  cat='cricket',   subs=['IndiaCricket','ipl','india']),
    dict(date='2025-03-09', name='Champions Trophy Final',        cat='cricket',   subs=['IndiaCricket','ipl','india']),
    dict(date='2025-03-22', name='IPL 2025 Begins',               cat='cricket',   subs=['IndiaCricket','ipl']),
    dict(date='2025-04-03', name='Waqf Amendment Act',            cat='political', subs=['india','indiaspeaks','unitedstatesofindia','AskIndia']),
    dict(date='2025-04-22', name='Pahalgam Terror Attack',        cat='security',  subs=['india','indiaspeaks','unitedstatesofindia','AskIndia']),
    dict(date='2025-05-07', name='Operation Sindoor',             cat='security',  subs=['india','indiaspeaks','unitedstatesofindia','AskIndia']),
    dict(date='2025-05-08', name='IPL Suspended',                 cat='cricket',   subs=['IndiaCricket','ipl','india']),
    dict(date='2025-05-10', name='India-Pakistan Ceasefire',      cat='security',  subs=['india','indiaspeaks','unitedstatesofindia']),
    dict(date='2025-06-03', name='IPL Final — RCB win',           cat='cricket',   subs=['IndiaCricket','ipl']),
    dict(date='2025-10-05', name="Women's WC IND vs PAK",         cat='cricket',   subs=['IndiaCricket','india']),
    dict(date='2025-11-14', name='Bihar Election Results',         cat='election',  subs=['india','indiaspeaks','unitedstatesofindia','AskIndia']),
    dict(date='2026-02-01', name='Union Budget 2026-27',          cat='economic',  subs=['india','IndiaInvestments','IndianStockMarket','personalfinanceindia']),
    dict(date='2026-04-22', name='State Elections (KL/AS/PY)',    cat='election',  subs=['india','indiaspeaks','AskIndia']),
    dict(date='2026-05-04', name='Five-State Results (WB/TN/KL)', cat='election',  subs=['india','indiaspeaks','unitedstatesofindia','AskIndia']),
]

STOPWORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her','it','its',
    'they','them','their','what','which','who','this','that','these','those','am',
    'is','are','was','were','be','been','have','has','had','do','does','did','will',
    'would','could','should','may','might','the','a','an','and','but','or','for',
    'so','of','at','by','with','about','to','from','in','on','out','over','not',
    'no','all','each','more','most','other','some','only','just','now','also',
    'really','people','india','indian','reddit','like','get','one','think','know',
    'see','say','make','go','take','new','good','right','time','year','day','even',
    'want','need','feel','look','dont','doesnt','didnt','cant','wont','ive','im',
    'its','youre','theyre','hes','shes','weve','isnt','great','little','old','big',
    'did','does','can','has','been','will','would','could','should','much','very',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(C['bg'])
    ax.grid(True, color=C['grid'], linewidth=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(C['grid'])
    ax.spines['bottom'].set_color(C['grid'])
    if title:  ax.set_title(title, fontsize=9, color=C['dark'], pad=6, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, fontsize=7.5, color=C['slate'])
    if ylabel: ax.set_ylabel(ylabel, fontsize=7.5, color=C['slate'])
    ax.tick_params(colors=C['slate'], labelsize=7)


def section_header(pdf, title, subtitle='', color=C['blue']):
    fig, ax = plt.subplots(figsize=(11, 1.8))
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['bg'])
    rect = mpatches.FancyBboxPatch((0, 0.1), 1, 0.8, transform=ax.transAxes,
                                    boxstyle='round,pad=0.02', facecolor=color,
                                    alpha=0.25, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(0.04, 0.65, title, transform=ax.transAxes,
            fontsize=20, fontweight='bold', color=C['dark'], va='center')
    if subtitle:
        ax.text(0.04, 0.28, subtitle, transform=ax.transAxes,
                fontsize=9, color=C['slate'], va='center')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    plt.tight_layout(pad=0.3)
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def tokenize(text):
    words = re.sub(r'[^\w\s]', ' ', str(text).lower()).split()
    return [w for w in words if len(w) > 2 and w not in STOPWORDS and not w.isdigit()]


def top_words(texts, n=20):
    counter = defaultdict(int)
    for t in texts:
        for w in tokenize(t):
            counter[w] += 1
    return sorted(counter.items(), key=lambda x: -x[1])[:n]


def gini(arr):
    arr = np.sort(np.abs(np.array(arr, dtype=float)))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    return (2 * (np.arange(1, n+1) * arr).sum()) / (n * arr.sum()) - (n+1)/n


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all(months=None):
    post_files = sorted(DATA_DIR.glob('posts_20*.csv'))
    comm_files = sorted(DATA_DIR.glob('commenters_20*.csv'))
    if not post_files:
        raise FileNotFoundError(f'No data in {DATA_DIR}. Run collect_data_v2.py --year first.')

    posts = pd.concat([pd.read_csv(f) for f in post_files], ignore_index=True)
    comms = pd.concat([pd.read_csv(f) for f in comm_files], ignore_index=True)

    if months:
        posts = posts[posts['collection_month'].isin(months)]
        comms = comms[comms['collection_month'].isin(months)]

    posts['kpd']          = (posts['author_link_karma'] + posts['author_comment_karma']) / posts['author_account_age_days'].clip(lower=1)
    posts['link_ratio']   = posts['author_link_karma'] / (posts['author_comment_karma'].clip(lower=0) + 1)
    posts['ucr']          = posts['score'] / posts['num_comments'].clip(lower=1)
    posts['is_susp']      = (posts['kpd'] > 500) | (posts['author_account_age_days'] < 90)
    posts['is_new_acct']  = posts['author_account_age_days'] < 90
    posts['created_dt']   = pd.to_datetime(posts['created_utc'], unit='s', utc=True)
    posts['hour_ist']     = (posts['created_dt'].dt.hour + 5) % 24
    posts['cluster']      = posts['subreddit'].map(SUB_CLUSTER).fillna('Other')
    posts['title_clean']  = posts['title'].astype(str).str.lower().str.replace(r'[^\w\s]', ' ', regex=True)
    posts['is_simulacra'] = (posts['score'] > 500) & (posts['num_comments'] < 5)

    comms['kpd']          = (comms['author_link_karma'] + comms['author_comment_karma']) / comms['author_account_age_days'].clip(lower=1)
    comms['is_susp']      = (comms['kpd'] > 500) | (comms['author_account_age_days'] < 90)
    comms['is_new_acct']  = comms['author_account_age_days'] < 90

    print(f"  Posts: {len(posts):,}  Comments: {len(comms):,}  "
          f"Subs: {posts['subreddit'].nunique()}  Months: {posts['collection_month'].nunique()}")
    return posts, comms


# ── Signal computation ────────────────────────────────────────────────────────

def compute_signals(posts, comms):
    rows = []
    # Pre-group comms for fast per-post lookup
    comms_grouped = {k: v for k, v in comms.groupby(['subreddit', 'collection_month'])}

    # Per-(sub,month) account sets, and per-month "other subs" unions, for cross_sub_rate
    # (mirrors analyze_network in analyze_data_v2.py).
    sub_month_accounts = {}
    for (sub, month), pg2 in posts.groupby(['subreddit', 'collection_month']):
        cg2 = comms_grouped.get((sub, month), pd.DataFrame())
        sub_month_accounts[(sub, month)] = set(pg2['author'].dropna()) | (
            set(cg2['author'].dropna()) if len(cg2) else set())
    other_accounts_by_month = {}
    for (sub, month), acc in sub_month_accounts.items():
        other_accounts_by_month.setdefault(month, {})[sub] = acc
    def _other_accounts(sub, month):
        siblings = other_accounts_by_month.get(month, {})
        u = set()
        for s2, acc in siblings.items():
            if s2 != sub:
                u |= acc
        return u

    for (sub, month), pg in posts.groupby(['subreddit', 'collection_month']):
        cg = comms_grouped.get((sub, month), pd.DataFrame())
        top_c   = cg[cg['in_top10']]  if len(cg) else cg
        first_c = cg[cg['in_first5']] if len(cg) else cg

        # Engagement signals
        score_mean   = pg['score'].mean()
        score_cv     = pg['score'].std() / score_mean if score_mean > 0 else 0
        comm_mean    = pg['num_comments'].mean()
        comm_cv      = pg['num_comments'].std() / comm_mean if comm_mean > 0 else 0

        # Rank-decay shape — replaces score_cv as the scored distribution signal.
        # score_cv is truncated by the top.json-sorted/capped-at-40 collection itself
        # (every sub's top-N slice has compressed variance regardless of bot activity);
        # decay_slope (log score ~ log rank) captures curve *shape* instead, which
        # survives that truncation. Flat slope (near 0) = posts pinned near-equal
        # near the top = coordination signature; steep slope = organic one-or-two-
        # breakout-posts-then-long-tail.
        decay_slope = np.nan
        s_sorted = pg.sort_values('score', ascending=False)['score'].values
        s_sorted = s_sorted[s_sorted > 0]
        if len(s_sorted) >= 8:
            ranks = np.arange(1, len(s_sorted) + 1)
            decay_slope = float(np.polyfit(np.log(ranks), np.log(s_sorted), 1)[0])
        ratio_std    = pg['upvote_ratio'].std()
        ucr_mean     = pg['ucr'].mean()
        corr_sc      = float(pg['score'].corr(pg['num_comments']))
        simulacra    = pg['is_simulacra'].mean() * 100

        # Account signals
        new_poster_pct  = pg['is_new_acct'].mean() * 100
        high_kpd_pct    = (pg['kpd'] > 500).mean() * 100
        link_ratio_mean = pg['link_ratio'].median()
        new_comm_pct    = top_c['is_new_acct'].mean() * 100 if len(top_c) > 0 else 0.0

        # Temporal signals
        p_sorted = pg.sort_values('created_utc')
        ivs = p_sorted['created_utc'].diff().dropna().values.astype(float)
        iv_cv    = float(np.std(ivs) / np.mean(ivs)) if len(ivs) > 3 and np.mean(ivs) > 0 else np.nan
        hc       = pg['hour_ist'].value_counts()
        top3_conc = float(hc.nlargest(3).sum() / len(pg) * 100)
        probs    = hc / len(pg)
        entropy  = float(-np.sum(probs * np.log2(probs + 1e-10)))

        # Ring signals — per post
        overlaps, burst_scores, ttfc_list = [], [], []
        post_ts_map = dict(zip(pg['post_id'], pg['created_utc']))
        cg_by_post  = {pid: grp for pid, grp in cg.groupby('post_id')} if len(cg) else {}

        for pid in pg['post_id']:
            pc = cg_by_post.get(pid, pd.DataFrame())
            if len(pc) == 0:
                continue
            top_a   = set(pc[pc['in_top10']]['author'].dropna())
            first_a = set(pc[pc['in_first5']]['author'].dropna())
            if first_a:
                overlaps.append(len(top_a & first_a) / len(first_a))
            ts0      = post_ts_map.get(pid)
            first_ts = pc[pc['in_first5']]['comment_created_utc'].dropna()
            if ts0 is not None and len(first_ts) >= 3:
                deltas   = (first_ts.values.astype(float) - float(ts0)) / 60
                std_min  = float(np.std(deltas))
                burst_scores.append(max(0.0, 1.0 - std_min / 30.0) * 100)
                ttfc_list.append(float(np.min(deltas)))

        overlap_rate  = float(np.mean(overlaps))     if overlaps     else 0.0
        burst_score   = float(np.mean(burst_scores)) if burst_scores else 0.0
        avg_ttfc      = float(np.mean(ttfc_list))    if ttfc_list    else np.nan
        fast_ttfc_pct = float(np.mean([t < 5 for t in ttfc_list])) * 100 if ttfc_list else 0.0

        # Recurrence: commenter in 3+ posts this sub-month
        recurrence_rate = 0.0
        if len(top_c) > 0:
            recur       = top_c.groupby('author')['post_id'].nunique()
            rec_authors = recur[recur >= 3].index
            posts_w_rec = top_c[top_c['author'].isin(rec_authors)]['post_id'].nunique()
            recurrence_rate = posts_w_rec / len(pg) if len(pg) > 0 else 0.0

        # Network signals (near_dupe_rate excluded — see analyze_data_v2.py note, essentially
        # never fires in this corpus)
        sub_accounts   = sub_month_accounts.get((sub, month), set())
        other_accounts = _other_accounts(sub, month)
        cross_sub_rate = (len(sub_accounts & other_accounts) / len(sub_accounts) * 100) if sub_accounts else 0.0
        gini_score     = gini(pg['score'].tolist()) * 100

        rows.append(dict(
            subreddit=sub, month=month, cluster=SUB_CLUSTER.get(sub, 'Other'),
            n_posts=len(pg), n_comments=len(cg),
            ucr=ucr_mean, score_cv=score_cv, comm_cv=comm_cv, decay_slope=decay_slope,
            upvote_ratio_std=ratio_std, score_comm_corr=corr_sc, simulacra_rate=simulacra,
            cross_sub_rate=cross_sub_rate, gini_score=gini_score,
            new_poster_pct=new_poster_pct, new_comm_pct=new_comm_pct,
            high_kpd_pct=high_kpd_pct, link_ratio_mean=link_ratio_mean,
            interval_cv=iv_cv, top3_concentration=top3_conc, entropy=entropy,
            overlap_rate=overlap_rate, burst_score=burst_score,
            avg_ttfc_min=avg_ttfc, fast_ttfc_pct=fast_ttfc_pct,
            recurrence_rate=recurrence_rate,
        ))

    df = pd.DataFrame(rows)
    print(f"  Signals: {len(df)} sub-month observations, {len(df.columns)-3} numeric signals")
    return df


# ── §1 Data Portrait ──────────────────────────────────────────────────────────

def section_1(posts, comms, pdf):
    section_header(pdf, '§1  Data Portrait',
                   f"{posts['subreddit'].nunique()} subreddits · "
                   f"{posts['collection_month'].nunique()} months · "
                   f"{len(posts):,} posts · {len(comms):,} comment rows",
                   color=C['blue'])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor(C['bg'])

    # Coverage heatmap
    ax = axes[0, 0]
    sub_order = sorted(posts['subreddit'].unique(), key=lambda s: (SUB_CLUSTER.get(s, 'zz'), s))
    pivot = posts.groupby(['collection_month', 'subreddit'])['post_id'].count().unstack(fill_value=0)
    pivot = pivot.reindex(columns=[s for s in sub_order if s in pivot.columns])
    sns.heatmap(pivot, ax=ax, cmap='Blues', linewidths=0.3, linecolor=C['grid'],
                annot=False, cbar_kws={'shrink': 0.6})
    ax.set_title('Posts per Sub per Month', fontsize=9, color=C['dark'], fontweight='bold')
    ax.tick_params(axis='x', rotation=60, labelsize=5)
    ax.tick_params(axis='y', labelsize=6.5)
    ax.set_xlabel(''); ax.set_ylabel('')

    # Posts per month
    ax = axes[0, 1]
    monthly = posts.groupby('collection_month').agg(
        posts=('post_id', 'count'), subs=('subreddit', 'nunique')
    ).reset_index()
    ax.bar(range(len(monthly)), monthly['posts'], color=C['blue'], alpha=0.8, zorder=3)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly['collection_month'], rotation=45, fontsize=6.5)
    ax2 = ax.twinx()
    ax2.plot(range(len(monthly)), monthly['subs'], 'o-', color=C['rose'], lw=1.5, ms=4)
    ax2.set_ylabel('Active subs', fontsize=7, color=C['rose'])
    ax2.tick_params(labelsize=6.5)
    style_ax(ax, 'Posts per Month', '', 'Post count')

    # Account age distribution
    ax = axes[1, 0]
    ax.hist(posts['author_account_age_days'].clip(0, 3650), bins=50,
            color=C['sage'], edgecolor='white', lw=0.3, alpha=0.85, zorder=3)
    ax.axvline(90,  color=C['red'],   lw=1.5, ls='--', label='90-day threshold')
    ax.axvline(365, color=C['amber'], lw=1.2, ls='--', label='1 year')
    ax.legend(fontsize=7)
    style_ax(ax, 'Poster Account Age Distribution', 'Days old (capped at 10yr)', 'Count')
    new_pct = posts['is_new_acct'].mean() * 100
    ax.text(0.97, 0.95, f'{new_pct:.1f}% accounts < 90d', transform=ax.transAxes,
            fontsize=7.5, ha='right', va='top', color=C['red'],
            bbox=dict(boxstyle='round', facecolor=C['rose'], alpha=0.4, edgecolor='none'))

    # KPD distribution
    ax = axes[1, 1]
    kpd = posts['kpd'].clip(0, 3000).dropna()
    ax.hist(kpd, bins=60, color=C['amber'], edgecolor='white', lw=0.3, alpha=0.85, zorder=3)
    ax.axvline(500, color=C['red'], lw=1.5, ls='--', label='KPD=500 threshold')
    ax.legend(fontsize=7)
    susp_pct = posts['is_susp'].mean() * 100
    ax.text(0.97, 0.95, f'{susp_pct:.1f}% suspicious', transform=ax.transAxes,
            fontsize=7.5, ha='right', va='top', color=C['red'],
            bbox=dict(boxstyle='round', facecolor=C['rose'], alpha=0.4, edgecolor='none'))
    style_ax(ax, 'Karma per Day (KPD) — Poster Accounts', 'KPD (capped at 3000)', 'Count')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


# ── §2 Univariate ─────────────────────────────────────────────────────────────

def section_2(sigs, pdf):
    section_header(pdf, '§2  Univariate Analysis',
                   'Distribution of every raw signal across all sub-month observations',
                   color=C['sage'])

    SIGNAL_META = [
        ('ucr',                'Upvote:Comment Ratio (UCR)',   'Higher = upvotes without discussion',        C['rose']),
        ('score_cv',           'Score Coefficient of Variation','Lower = artificially uniform scores',        C['rose']),
        ('upvote_ratio_std',   'Upvote Ratio Std Dev',         'Lower = manufactured consensus',             C['rose']),
        ('score_comm_corr',    'Score-Comment Correlation',    'Lower = votes without engagement',           C['rose']),
        ('simulacra_rate',     'Simulacra Rate %',             'score>500 & comments<5',                     C['red']),
        ('overlap_rate',       'Comment Overlap Rate',         'first-5 ∩ top-10 authors (organic baseline)',C['lav']),
        ('burst_score',        'Burst Score',                  'Tight timing cluster of first comments',     C['lav']),
        ('avg_ttfc_min',       'Avg Time-to-First-Comment (min)','Minutes after post creation',              C['lav']),
        ('fast_ttfc_pct',      'Fast TTFC % (<5 min)',         '% of posts with sub-5-min first comment',   C['lav']),
        ('recurrence_rate',    'Commenter Recurrence Rate',    'Same account in 3+ posts same sub/month',   C['lav']),
        ('new_poster_pct',     'New Poster % (<90d)',          'Accounts younger than 90 days',             C['amber']),
        ('new_comm_pct',       'New Commenter % (<90d)',       'Top-10 commenters with new accounts',       C['amber']),
        ('high_kpd_pct',       'High-KPD Account % (>500)',   'Karma farmer / provisioned account signal', C['amber']),
        ('interval_cv',        'Post Interval CV',             'Low = suspiciously regular posting rhythm', C['blue']),
        ('top3_concentration', 'Top-3 Hour Concentration % (IST)', 'Posts crammed into 3 hours',           C['blue']),
        ('entropy',            'Posting Hour Entropy',         'Low = concentrated; High = spread',         C['blue']),
    ]

    cols = 4
    per_page = 8
    for page_start in range(0, len(SIGNAL_META), per_page):
        page_sigs = SIGNAL_META[page_start:page_start + per_page]
        nrows = 2
        fig, axes = plt.subplots(nrows, cols, figsize=(11, nrows * 3.2))
        fig.patch.set_facecolor(C['bg'])
        axes = np.array(axes).flatten()

        for i, (col, label, desc, color) in enumerate(page_sigs):
            ax = axes[i]
            data = sigs[col].dropna()
            if len(data) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                        transform=ax.transAxes, color=C['slate'], fontsize=8)
                ax.set_title(label, fontsize=8, color=C['dark'])
                continue
            ax.hist(data, bins=25, color=color, edgecolor='white', lw=0.3, alpha=0.85, zorder=3)
            p50 = np.median(data)
            ax.axvline(p50, color=C['dark'], lw=1.2, ls='--', alpha=0.7)
            style_ax(ax, label)
            ax.set_xlabel(desc, fontsize=5.5, color=C['slate'])
            ax.text(0.97, 0.97,
                    f'μ={data.mean():.2f}\nσ={data.std():.2f}\np50={p50:.2f}',
                    transform=ax.transAxes, fontsize=6, va='top', ha='right',
                    color=C['slate'], family='monospace')

        for j in range(len(page_sigs), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight'); plt.close()


# ── §3 Correlations ───────────────────────────────────────────────────────────

def section_3(sigs, pdf):
    section_header(pdf, '§3  Signal Correlations',
                   'Collinearity between signals — and which signals discriminate best across subreddits',
                   color=C['amber'])

    SIGNAL_COLS = [
        'ucr', 'decay_slope', 'comm_cv', 'upvote_ratio_std', 'score_comm_corr', 'simulacra_rate',
        'burst_score', 'fast_ttfc_pct', 'recurrence_rate', 'overlap_rate',
        'new_poster_pct', 'new_comm_pct', 'high_kpd_pct',
        'interval_cv', 'top3_concentration',
    ]
    LABELS = {
        'ucr': 'UCR', 'decay_slope': 'Decay Slope', 'comm_cv': 'Comments CV', 'upvote_ratio_std': 'Ratio σ',
        'score_comm_corr': 'Score↔Comm r', 'simulacra_rate': 'Simulacra%',
        'burst_score': 'Burst', 'fast_ttfc_pct': 'FastTTFC%', 'recurrence_rate': 'Recurrence',
        'overlap_rate': 'Overlap', 'new_poster_pct': 'NewPost%',
        'new_comm_pct': 'NewComm%', 'high_kpd_pct': 'HighKPD%',
        'interval_cv': 'Interval CV', 'top3_concentration': 'Top3Hr%',
    }
    available = [c for c in SIGNAL_COLS if c in sigs.columns]
    corr_df   = sigs[available].dropna().corr()
    corr_df.index   = [LABELS.get(c, c) for c in corr_df.index]
    corr_df.columns = [LABELS.get(c, c) for c in corr_df.columns]

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(C['bg'])
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr_df, mask=mask, ax=ax, cmap=cmap, vmin=-1, vmax=1,
                center=0, annot=True, fmt='.2f', annot_kws={'size': 7},
                linewidths=0.5, linecolor=C['grid'],
                cbar_kws={'shrink': 0.6, 'label': 'Pearson r'})
    ax.set_title('Signal Pairwise Correlation Matrix (lower triangle)',
                 fontsize=11, color=C['dark'], pad=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # Discriminative power chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(C['bg'])

    variances = sigs[available].std()
    means     = sigs[available].mean().abs().replace(0, 1)
    signal_cv = (variances / means).sort_values()
    labels_cv = [LABELS.get(c, c) for c in signal_cv.index]
    med_cv    = signal_cv.median()
    colors_cv = [C['rose'] if v > med_cv else C['blue'] for v in signal_cv.values]
    ax1.barh(labels_cv, signal_cv.values, color=colors_cv, alpha=0.85, zorder=3)
    ax1.axvline(med_cv, color=C['dark'], lw=1.2, ls='--', label=f'Median CV={med_cv:.2f}')
    ax1.legend(fontsize=7)
    style_ax(ax1, 'Discriminative Power (CV across sub-months)\nHigher = better at differentiating subs',
             'Coefficient of Variation', '')

    # ANOVA F-statistic across sub clusters
    f_scores = {}
    for col in available:
        groups = [sigs[sigs['cluster'] == cl][col].dropna()
                  for cl in sigs['cluster'].unique() if len(sigs[sigs['cluster'] == cl][col].dropna()) > 2]
        if len(groups) >= 2:
            try:
                f, _ = stats.f_oneway(*groups)
                f_scores[col] = float(f) if not np.isnan(f) else 0
            except Exception:
                f_scores[col] = 0

    f_series = pd.Series(f_scores).sort_values()
    labels_f = [LABELS.get(c, c) for c in f_series.index]
    med_f    = f_series.median()
    colors_f = [C['rose'] if v > med_f else C['sage'] for v in f_series.values]
    ax2.barh(labels_f, f_series.values, color=colors_f, alpha=0.85, zorder=3)
    ax2.axvline(med_f, color=C['dark'], lw=1.2, ls='--', label=f'Median F={med_f:.1f}')
    ax2.legend(fontsize=7)
    style_ax(ax2, 'Between-Cluster Discrimination (ANOVA F)\nHigher = signal differs meaningfully across sub types',
             'F-statistic', '')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


# ── §4 Text Intelligence ──────────────────────────────────────────────────────

def section_4(posts, pdf):
    section_header(pdf, '§4  Text Intelligence',
                   'Topics, narrative pushes, near-duplicates, cross-sub keyword infiltration — from post titles',
                   color=C['rose'])

    # Per cluster: organic vs suspicious keywords
    for cl in sorted(posts['cluster'].unique()):
        if cl == 'Other':
            continue
        cp      = posts[posts['cluster'] == cl]
        organic = cp[~cp['is_susp']]['title'].tolist()
        susp    = cp[cp['is_susp']]['title'].tolist()

        if len(organic) < 15:
            continue

        org_words  = dict(top_words(organic, 20))
        susp_words = dict(top_words(susp, 20)) if len(susp) >= 15 else {}

        divergence = {}
        for w in set(org_words) | set(susp_words):
            of = org_words.get(w, 0) / max(len(organic), 1)
            sf = susp_words.get(w, 0) / max(len(susp), 1) if susp_words else 0
            if sf > 0 and of > 0:
                divergence[w] = round(sf / of, 2)

        fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
        fig.patch.set_facecolor(C['bg'])
        fig.suptitle(f'Cluster: {cl}  ({len(cp):,} posts across {cp["subreddit"].nunique()} subs)',
                     fontsize=10, color=C['dark'], fontweight='bold')

        # Organic top words
        ax = axes[0]
        if org_words:
            items = list(org_words.items())[:15]
            ws, cs = zip(*items)
            ax.barh(list(ws), list(cs), color=CLUSTER_COLOR.get(cl, C['blue']), alpha=0.85, zorder=3)
            ax.tick_params(axis='y', labelsize=7)
        style_ax(ax, 'Top Words — Organic Accounts', 'Frequency', '')

        # Suspicious top words
        ax = axes[1]
        if susp_words:
            items2 = list(susp_words.items())[:15]
            ws2, cs2 = zip(*items2)
            ax.barh(list(ws2), list(cs2), color=C['red'], alpha=0.7, zorder=3)
            ax.tick_params(axis='y', labelsize=7)
            style_ax(ax, f'Top Words — Suspicious Accounts ({len(susp)} posts)', 'Frequency', '')
        else:
            ax.text(0.5, 0.5, f'Only {len(susp)} suspicious\naccount posts — not enough',
                    ha='center', va='center', transform=ax.transAxes, fontsize=8, color=C['slate'])
            ax.set_title('Top Words — Suspicious Accounts', fontsize=9, color=C['dark'])
            ax.set_facecolor(C['bg'])

        # Divergence: pushed narratives
        ax = axes[2]
        if len(divergence) >= 5:
            top_div = sorted(divergence.items(), key=lambda x: -x[1])[:12]
            dw, dv = zip(*top_div)
            colors_d = [C['red'] if v > 2 else C['amber'] for v in dv]
            ax.barh(list(dw), list(dv), color=colors_d, alpha=0.85, zorder=3)
            ax.axvline(2.0, color=C['dark'], lw=1, ls='--', alpha=0.6, label='2× baseline')
            ax.axvline(1.0, color=C['slate'], lw=0.8, ls=':', alpha=0.5)
            ax.legend(fontsize=6.5)
            ax.tick_params(axis='y', labelsize=7)
            style_ax(ax, 'Pushed Narratives\n(suspicious/organic freq ratio)', 'Ratio', '')
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor divergence analysis',
                    ha='center', va='center', transform=ax.transAxes, fontsize=8, color=C['slate'])
            ax.set_title('Pushed Narratives', fontsize=9, color=C['dark'])
            ax.set_facecolor(C['bg'])

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # Near-duplicate detection
    _text_neardupes(posts, pdf)

    # Cross-sub keyword infiltration
    _text_crosssub(posts, pdf)


def _text_neardupes(posts, pdf):
    dup_stats, examples = [], []

    for (sub, month), grp in posts.groupby(['subreddit', 'collection_month']):
        titles     = grp['title'].tolist()
        token_sets = [set(tokenize(t)) for t in titles]
        n          = len(token_sets)
        dup_pairs  = 0
        for i in range(n):
            for j in range(i+1, n):
                if not token_sets[i] or not token_sets[j]:
                    continue
                union = token_sets[i] | token_sets[j]
                if union:
                    jac = len(token_sets[i] & token_sets[j]) / len(union)
                    if jac > 0.5:
                        dup_pairs += 1
                        if len(examples) < 6:
                            examples.append((sub, month, titles[i][:70], titles[j][:70], round(jac, 2)))
        total_pairs = n * (n-1) // 2
        dup_stats.append({'subreddit': sub, 'month': month,
                          'dup_rate': dup_pairs / total_pairs * 100 if total_pairs > 0 else 0})

    dup_df = pd.DataFrame(dup_stats)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('Near-Duplicate Post Detection  (Jaccard similarity on title word sets > 0.5)',
                 fontsize=11, color=C['dark'], fontweight='bold')

    ax = axes[0, 0]
    avg_dup = dup_df.groupby('subreddit')['dup_rate'].mean().sort_values(ascending=False).head(15)
    bar_c   = [CLUSTER_COLOR.get(SUB_CLUSTER.get(s, 'Other'), C['blue']) for s in avg_dup.index]
    ax.barh(avg_dup.index.tolist(), avg_dup.values, color=bar_c, alpha=0.85, zorder=3)
    style_ax(ax, 'Avg Near-Duplicate Rate by Sub', 'Dup-pair rate (%)', '')
    ax.tick_params(axis='y', labelsize=6.5)

    ax = axes[0, 1]
    mdup = dup_df.groupby('month')['dup_rate'].mean()
    ax.plot(range(len(mdup)), mdup.values, 'o-', color=C['rose'], lw=2, ms=5, zorder=3)
    ax.set_xticks(range(len(mdup)))
    ax.set_xticklabels(mdup.index.tolist(), rotation=45, fontsize=6.5)
    style_ax(ax, 'Near-Duplicate Rate Over Time', '', 'Avg dup rate (%)')

    ax = axes[1, 0]
    ax.set_facecolor(C['bg']); ax.axis('off')
    ax.set_title('Example Near-Duplicate Pairs', fontsize=9, color=C['dark'], fontweight='bold')
    y = 0.96
    for ex in examples[:5]:
        sub_e, mo_e, t1, t2, jac_e = ex
        txt = f"r/{sub_e} [{mo_e}]  sim={jac_e}\n  A: {t1}\n  B: {t2}"
        ax.text(0.02, y, txt, transform=ax.transAxes, fontsize=6,
                color=C['dark'], va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor=C['rose'], alpha=0.2, edgecolor='none'))
        y -= 0.21

    ax = axes[1, 1]
    ax.hist(dup_df['dup_rate'], bins=30, color=C['lav'], edgecolor='white', alpha=0.85, zorder=3)
    p90 = np.percentile(dup_df['dup_rate'], 90)
    ax.axvline(p90, color=C['red'], lw=1.5, ls='--', label=f'p90={p90:.1f}%')
    ax.legend(fontsize=7)
    style_ax(ax, 'Distribution of Near-Dup Rates\nAll sub-months', 'Dup rate (%)', 'Count')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


def _text_crosssub(posts, pdf):
    cross_months = {}
    for month, mg in posts.groupby('collection_month'):
        sub_words = {}
        for sub, sg in mg.groupby('subreddit'):
            sub_words[sub] = set(tokenize(' '.join(sg['title'].tolist())))
        word_subs = defaultdict(set)
        for sub, words in sub_words.items():
            for w in words:
                word_subs[w].add(sub)
        cross_months[month] = {w: s for w, s in word_subs.items() if len(s) >= 4}

    rows = []
    for month, word_subs_d in cross_months.items():
        for w, subs_set in word_subs_d.items():
            rows.append({'month': month, 'word': w, 'n_subs': len(subs_set)})
    cross_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['month', 'word', 'n_subs'])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('Cross-Sub Keyword Infiltration  (words appearing in 4+ subreddits in same month)',
                 fontsize=11, color=C['dark'], fontweight='bold')

    ax = axes[0, 0]
    if len(cross_df) > 5:
        top_cross = cross_df.groupby('word')['n_subs'].mean().sort_values(ascending=False).head(20)
        ax.barh(top_cross.index.tolist(), top_cross.values, color=C['red'], alpha=0.75, zorder=3)
        style_ax(ax, 'Most Cross-Sub Keywords (avg subs/month)', 'Avg subreddits/month', '')
        ax.tick_params(axis='y', labelsize=7)
    else:
        ax.text(0.5, 0.5, 'No cross-sub keywords found', ha='center', va='center',
                transform=ax.transAxes, color=C['slate'])

    ax = axes[0, 1]
    if len(cross_df) > 0:
        mcount = cross_df.groupby('month')['word'].count()
        ax.plot(range(len(mcount)), mcount.values, 'o-', color=C['red'], lw=2, ms=5, zorder=3)
        ax.set_xticks(range(len(mcount)))
        ax.set_xticklabels(mcount.index.tolist(), rotation=45, fontsize=6.5)
        style_ax(ax, 'Cross-Sub Keywords Over Time\n(# words in 4+ subs per month)', '', 'Count')

    # Political sub vocabulary uniqueness
    ax = axes[1, 0]
    pol = posts[posts['cluster'] == 'Political']
    if len(pol) > 0:
        sub_ws = {sub: set(tokenize(' '.join(sg['title'].tolist())))
                  for sub, sg in pol.groupby('subreddit')}
        unique_counts = {}
        for sub, ws in sub_ws.items():
            others_union = set().union(*[w for s, w in sub_ws.items() if s != sub])
            unique_counts[sub] = len(ws - others_union)
        uc_sorted = sorted(unique_counts.items(), key=lambda x: -x[1])
        subs_l, cnts_l = zip(*uc_sorted) if uc_sorted else ([], [])
        ax.barh(list(subs_l), list(cnts_l), color=CLUSTER_COLOR.get('Political', C['rose']), alpha=0.85, zorder=3)
        style_ax(ax, 'Unique Vocabulary — Political Cluster\n(words used in this sub not in others)', 'Unique word count', '')
        ax.tick_params(axis='y', labelsize=7)

    # Heatmap: top cross-sub words × last 6 months
    ax = axes[1, 1]
    if len(cross_df) > 10:
        top_w   = cross_df.groupby('word')['n_subs'].sum().sort_values(ascending=False).head(10).index.tolist()
        months6 = sorted(cross_df['month'].unique())[-6:]
        hm_data = []
        for w in top_w:
            row = {}
            for m in months6:
                sub_df = cross_df[(cross_df['word'] == w) & (cross_df['month'] == m)]
                row[m] = int(sub_df['n_subs'].values[0]) if len(sub_df) > 0 else 0
            hm_data.append(row)
        hm_df = pd.DataFrame(hm_data, index=top_w)
        sns.heatmap(hm_df, ax=ax, cmap='RdPu', linewidths=0.3, linecolor=C['grid'],
                    annot=True, fmt='.0f', annot_kws={'size': 7.5},
                    cbar_kws={'shrink': 0.6, 'label': '# subs'})
        ax.set_title('Cross-Sub Word Spread (last 6 months)', fontsize=9, color=C['dark'], fontweight='bold')
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        ax.tick_params(axis='y', rotation=0, labelsize=7.5)
    else:
        ax.text(0.5, 0.5, 'Not enough cross-sub data\nfor heatmap', ha='center', va='center',
                transform=ax.transAxes, color=C['slate'])
        ax.set_facecolor(C['bg'])

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


# ── §5 Cross-Sub Network ──────────────────────────────────────────────────────

def section_5(posts, comms, pdf):
    section_header(pdf, '§5  Cross-Sub Network',
                   'Same accounts appearing across subreddits — the true ring signal',
                   color=C['lav'])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor(C['bg'])

    # Cross-sub account counts per month
    ax = axes[0, 0]
    cs_rows = []
    for month, mg in comms.groupby('collection_month'):
        acct_subs = mg.groupby('author')['subreddit'].nunique()
        cs_rows.append(dict(month=month,
                            a3=int((acct_subs >= 3).sum()),
                            a5=int((acct_subs >= 5).sum()),
                            a8=int((acct_subs >= 8).sum())))
    cs_df = pd.DataFrame(cs_rows).sort_values('month')
    x     = range(len(cs_df))
    ax.bar(x, cs_df['a3'], label='3+ subs', color=C['amber'], alpha=0.7, zorder=3)
    ax.bar(x, cs_df['a5'], label='5+ subs', color=C['rose'],  alpha=0.8, zorder=3)
    ax.bar(x, cs_df['a8'], label='8+ subs', color=C['red'],   alpha=0.9, zorder=3)
    ax.set_xticks(list(x)); ax.set_xticklabels(cs_df['month'].tolist(), rotation=45, fontsize=6)
    ax.legend(fontsize=7)
    style_ax(ax, 'Cross-Sub Commenter Accounts per Month\n(accounts appearing in N+ subreddits)', '', 'Account count')

    # Gini coefficient of participation per sub
    ax = axes[0, 1]
    gini_rows = []
    for sub, sg in comms.groupby('subreddit'):
        cc = sg.groupby('author')['comment_id'].count()
        gini_rows.append(dict(subreddit=sub, gini=gini(cc.values),
                              cluster=SUB_CLUSTER.get(sub, 'Other')))
    gini_df = pd.DataFrame(gini_rows).sort_values('gini', ascending=False)
    bar_c   = [CLUSTER_COLOR.get(r['cluster'], C['slate']) for _, r in gini_df.iterrows()]
    ax.barh(gini_df['subreddit'].tolist(), gini_df['gini'].tolist(), color=bar_c, alpha=0.85, zorder=3)
    med_g = gini_df['gini'].median()
    ax.axvline(med_g, color=C['dark'], lw=1.5, ls='--', label=f'Median={med_g:.2f}')
    ax.legend(fontsize=7)
    style_ax(ax, 'Participation Gini by Sub\n(higher = more concentrated commenter activity)', 'Gini coefficient', '')
    ax.tick_params(axis='y', labelsize=6)

    # Commenter churn month-over-month
    ax = axes[1, 0]
    months = sorted(comms['collection_month'].unique())
    churn_rows = []
    for i in range(1, len(months)):
        prev = set(comms[comms['collection_month'] == months[i-1]]['author'].unique())
        curr = set(comms[comms['collection_month'] == months[i]]['author'].unique())
        if prev and curr:
            churn_rows.append(dict(
                month=months[i],
                retention=len(prev & curr) / len(prev) * 100,
                new_pct=len(curr - prev) / len(curr) * 100,
            ))
    if churn_rows:
        ch_df = pd.DataFrame(churn_rows)
        x2    = range(len(ch_df))
        ax.plot(x2, ch_df['retention'], 'o-', color=C['sage'],  lw=2, ms=5, label='Retention %')
        ax.plot(x2, ch_df['new_pct'],   's-', color=C['rose'],  lw=2, ms=5, label='New accounts %')
        ax.set_xticks(list(x2)); ax.set_xticklabels(ch_df['month'].tolist(), rotation=45, fontsize=6.5)
        ax.legend(fontsize=7)
        style_ax(ax, 'Commenter Churn (month-over-month, all subs)\n>60% new each month = possible account cycling', '', '%')

    # Simulacra rate per sub
    ax = axes[1, 1]
    sim_df = posts.groupby('subreddit')['is_simulacra'].mean().mul(100).sort_values(ascending=False).head(15)
    sim_c  = [CLUSTER_COLOR.get(SUB_CLUSTER.get(s, 'Other'), C['blue']) for s in sim_df.index]
    ax.barh(sim_df.index.tolist(), sim_df.values, color=sim_c, alpha=0.85, zorder=3)
    style_ax(ax, 'Simulacra Rate by Sub\n(score>500 & comments<5 — upvotes with no discussion)', '% of posts', '')
    ax.tick_params(axis='y', labelsize=6.5)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # Persistence page
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('Account Persistence Across 12 Months', fontsize=11, color=C['dark'], fontweight='bold')

    ax = axes[0]
    pers_rows = []
    for sub, sg in comms[comms['in_top10']].groupby('subreddit'):
        am = sg.groupby('author')['collection_month'].nunique()
        pers_rows.append(dict(subreddit=sub,
                              persistent_pct=float((am >= 6).sum() / len(am) * 100) if len(am) > 0 else 0,
                              cluster=SUB_CLUSTER.get(sub, 'Other')))
    pers_df = pd.DataFrame(pers_rows).sort_values('persistent_pct', ascending=False)
    pc      = [CLUSTER_COLOR.get(r['cluster'], C['slate']) for _, r in pers_df.iterrows()]
    ax.barh(pers_df['subreddit'].tolist(), pers_df['persistent_pct'].tolist(), color=pc, alpha=0.85, zorder=3)
    med_p = pers_df['persistent_pct'].median()
    ax.axvline(med_p, color=C['dark'], lw=1.5, ls='--', label=f'Median={med_p:.1f}%')
    ax.legend(fontsize=7)
    style_ax(ax, 'Top-Commenter Persistence (present in 6+ months)\nHigh % = genuine community OR entrenched bot ring',
             '% of unique commenters', '')
    ax.tick_params(axis='y', labelsize=6)

    ax = axes[1]
    monthly_age = posts.groupby('collection_month')['author_account_age_days'].median()
    x3 = range(len(monthly_age))
    ax.plot(x3, monthly_age.values, 'o-', color=C['blue'], lw=2, ms=6, zorder=3)
    ax.fill_between(x3, monthly_age.values, alpha=0.2, color=C['blue'])
    ax.set_xticks(list(x3)); ax.set_xticklabels(monthly_age.index.tolist(), rotation=45, fontsize=6.5)
    style_ax(ax, 'Median Poster Account Age Over Time\nSudden drops suggest new-account bot waves', '', 'Median age (days)')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


# ── §6 Event Calendar ─────────────────────────────────────────────────────────

def section_6(posts, pdf):
    section_header(pdf, '§6  Event Calendar Overlay',
                   'Bot activity signals vs India\'s major events — June 2025 to May 2026',
                   color=C['amber'])

    months    = sorted(posts['collection_month'].unique())
    n_months  = len(months)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False)
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('Post Volume, Scores & Suspicious Activity vs India Event Calendar',
                 fontsize=12, color=C['dark'], fontweight='bold')

    # Panel 1: Volume by cluster
    ax = axes[0]
    for cl, ccolor in CLUSTER_COLOR.items():
        cp = posts[posts['cluster'] == cl]
        if len(cp) == 0: continue
        vol = cp.groupby('collection_month')['post_id'].count().reindex(months, fill_value=0)
        ax.plot(range(n_months), vol.values, 'o-', label=cl, color=ccolor, lw=1.5, ms=3.5, alpha=0.85)
    _draw_event_markers(ax, months, cats={'security', 'election', 'political', 'cricket', 'economic'})
    ax.set_xticks(range(n_months)); ax.set_xticklabels(months, rotation=45, fontsize=6)
    ax.legend(fontsize=6, ncol=4, loc='upper left')
    style_ax(ax, 'Post Volume by Cluster', '', 'Posts/month')

    # Panel 2: Avg score — political subs
    ax = axes[1]
    pol_subs = [s for s, c in SUB_CLUSTER.items() if c == 'Political' and s in posts['subreddit'].unique()]
    for sub in pol_subs:
        sp = posts[posts['subreddit'] == sub]
        ms = sp.groupby('collection_month')['score'].mean().reindex(months)
        ax.plot(range(n_months), ms.values, 'o-', label=f'r/{sub}', lw=1.5, ms=3.5, alpha=0.8)
    _draw_event_markers(ax, months, cats={'security', 'election', 'political'}, annotate=True)
    ax.set_xticks(range(n_months)); ax.set_xticklabels(months, rotation=45, fontsize=6)
    ax.legend(fontsize=6.5, ncol=2)
    style_ax(ax, 'Avg Post Score — Political Subs vs Security/Election Events', '', 'Avg score')

    # Panel 3: Suspicious accounts + UCR
    ax = axes[2]
    ax2 = ax.twinx()
    msusp = posts.groupby('collection_month')['is_susp'].mean().mul(100).reindex(months)
    mucr  = posts.groupby('collection_month')['ucr'].mean().reindex(months)
    ax.bar(range(n_months), msusp.values, color=C['rose'], alpha=0.55, label='Suspicious acct %', zorder=3)
    ax2.plot(range(n_months), mucr.values, 's-', color=C['blue'], lw=2, ms=5, label='Avg UCR', zorder=4)
    ax.set_xticks(range(n_months)); ax.set_xticklabels(months, rotation=45, fontsize=6)
    ax.set_ylabel('Suspicious acct %', fontsize=7, color=C['rose'])
    ax2.set_ylabel('Avg UCR', fontsize=7, color=C['blue'])
    h1 = mpatches.Patch(color=C['rose'], alpha=0.55, label='Suspicious acct %')
    h2 = plt.Line2D([0], [0], color=C['blue'], marker='s', lw=2, label='Avg UCR')
    ax.legend(handles=[h1, h2], fontsize=7, loc='upper left')
    style_ax(ax, 'Suspicious Accounts & UCR Over Time', '', '')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # Event spotlight — score spikes per major event
    _event_spotlight(posts, months, pdf)


def _draw_event_markers(ax, months, cats=None, annotate=False):
    for ev in EVENTS:
        if cats and ev['cat'] not in cats:
            continue
        em = ev['date'][:7]
        if em in months:
            xi    = months.index(em)
            color = EVENT_CAT_COLOR.get(ev['cat'], C['slate'])
            ax.axvline(xi, color=color, lw=1.2, alpha=0.55, ls=':')
            if annotate:
                ylim = ax.get_ylim()
                ax.text(xi + 0.1, ylim[1] * 0.95 if ylim[1] > 0 else 100,
                        ev['name'][:18], fontsize=5, rotation=90, color=color, alpha=0.85, va='top')


def _event_spotlight(posts, months, pdf):
    major = [e for e in EVENTS if e['cat'] in ('security', 'election')][:6]
    if not major: return

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('Event Spotlight — Avg Score Before / During / After Major Events',
                 fontsize=11, color=C['dark'], fontweight='bold')
    axes = axes.flatten()

    for idx, ev in enumerate(major):
        ax      = axes[idx]
        ev_m    = ev['date'][:7]
        ev_dt   = pd.Timestamp(ev['date'])
        prev_m  = (ev_dt - pd.DateOffset(months=1)).strftime('%Y-%m')
        next_m  = (ev_dt + pd.DateOffset(months=1)).strftime('%Y-%m')
        ev_subs = [s for s in ev['subs'] if s in posts['subreddit'].unique()][:4]

        for sub in ev_subs:
            sp = posts[posts['subreddit'] == sub]
            vals = [
                sp[sp['collection_month'] == prev_m]['score'].mean(),
                sp[sp['collection_month'] == ev_m]['score'].mean(),
                sp[sp['collection_month'] == next_m]['score'].mean(),
            ]
            ax.plot([0, 1, 2], vals, 'o-', lw=1.5, ms=4, label=f'r/{sub}', alpha=0.85)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Before', 'Event', 'After'], fontsize=8)
        ax.legend(fontsize=5.5)
        color = EVENT_CAT_COLOR.get(ev['cat'], C['slate'])
        ax.set_title(f"{ev['name']}\n{ev['date']}", fontsize=7.5, color=color, fontweight='bold')
        style_ax(ax, '', '', 'Avg score')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()


# ── §6b Temporal Trend Analysis ───────────────────────────────────────────────

TREND_SIGNALS = ['ucr', 'score_cv', 'upvote_ratio_std', 'burst_score',
                 'high_kpd_pct', 'simulacra_rate', 'fast_ttfc_pct', 'recurrence_rate']
# Direction: +1 = higher value is MORE suspicious, -1 = lower is more suspicious
SIGNAL_DIRECTION = {
    'ucr': 1, 'score_cv': -1, 'upvote_ratio_std': -1, 'burst_score': 1,
    'high_kpd_pct': 1, 'simulacra_rate': 1, 'fast_ttfc_pct': 1, 'recurrence_rate': 1,
}


def section_6b(sigs, pdf):
    section_header(pdf, '§6b  Temporal Trend Analysis',
                   'Are bot signals strengthening over time? Which subs are accelerating? Dangerous emerging patterns?',
                   color=C['red'])

    months = sorted(sigs['month'].unique())
    month_idx = {m: i for i, m in enumerate(months)}
    avail = [s for s in TREND_SIGNALS if s in sigs.columns]

    # ── Per-sub linear regression slopes ──────────────────────────────────────
    slope_rows = []
    for sub, sg in sigs.groupby('subreddit'):
        sg_sorted = sg.sort_values('month')
        row = {'subreddit': sub, 'cluster': SUB_CLUSTER.get(sub, 'Other')}
        n_sig = 0
        for sig in avail:
            vals = sg_sorted[sig].dropna()
            if len(vals) < 4:
                row[f'{sig}_slope'] = np.nan
                row[f'{sig}_pval']  = np.nan
                continue
            x = np.array([month_idx[m] for m in sg_sorted.loc[vals.index, 'month']])
            res = stats.linregress(x, vals.values)
            row[f'{sig}_slope'] = float(res.slope)
            row[f'{sig}_pval']  = float(res.pvalue)
            # Count dangerous trends: slope in bot direction AND p < 0.15
            direction = SIGNAL_DIRECTION.get(sig, 1)
            if res.pvalue < 0.15 and res.slope * direction > 0:
                n_sig += 1
        row['n_dangerous_trends'] = n_sig
        slope_rows.append(row)

    slope_df = pd.DataFrame(slope_rows)

    # ── Page 1: Trend heatmap — subs × signals (slope direction + significance) ─
    fig, axes = plt.subplots(1, 2, figsize=(11, 9))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('Signal Trend Slopes Across 12 Months  (red = worsening, blue = improving)',
                 fontsize=11, color=C['dark'], fontweight='bold')

    # Build normalized slope heatmap
    sub_order = slope_df.sort_values('n_dangerous_trends', ascending=False)['subreddit'].tolist()
    hm_data = []
    for sub in sub_order:
        row_d = slope_df[slope_df['subreddit'] == sub].iloc[0]
        hm_row = {}
        for sig in avail:
            slope = row_d.get(f'{sig}_slope', np.nan)
            pval  = row_d.get(f'{sig}_pval',  1.0)
            direction = SIGNAL_DIRECTION.get(sig, 1)
            # Normalized: positive = worsening, negative = improving
            hm_row[sig] = float(slope * direction) if not np.isnan(slope) else 0.0
        hm_data.append(hm_row)

    hm_df = pd.DataFrame(hm_data, index=sub_order)
    # Normalize each column to [-1, 1] for visual clarity
    for col in hm_df.columns:
        mx = hm_df[col].abs().max()
        if mx > 0:
            hm_df[col] = hm_df[col] / mx

    ax = axes[0]
    sig_labels = {
        'ucr': 'UCR', 'score_cv': 'Score CV', 'upvote_ratio_std': 'Ratio σ',
        'burst_score': 'Burst', 'high_kpd_pct': 'HighKPD%',
        'simulacra_rate': 'Simulacra%', 'fast_ttfc_pct': 'FastTTFC%', 'recurrence_rate': 'Recurrence',
    }
    hm_df.columns = [sig_labels.get(c, c) for c in hm_df.columns]
    cmap_rg = sns.diverging_palette(10, 130, as_cmap=True)  # red=bad, green=good
    sns.heatmap(hm_df, ax=ax, cmap=cmap_rg, vmin=-1, vmax=1, center=0,
                linewidths=0.4, linecolor=C['grid'],
                cbar_kws={'shrink': 0.6, 'label': 'Normalized slope\n(red=worsening)'})
    ax.set_title('Trend Direction per Sub per Signal\n(sorted by # dangerous trends)',
                 fontsize=9, color=C['dark'], fontweight='bold')
    ax.tick_params(axis='x', rotation=40, labelsize=7.5)
    ax.tick_params(axis='y', labelsize=6.5)

    # Dangerous subs bar chart
    ax = axes[1]
    danger_df = slope_df[['subreddit', 'n_dangerous_trends', 'cluster']].sort_values(
        'n_dangerous_trends', ascending=True)
    bar_c = [C['red'] if r['n_dangerous_trends'] >= 4
             else C['amber'] if r['n_dangerous_trends'] >= 2
             else C['sage']
             for _, r in danger_df.iterrows()]
    ax.barh(danger_df['subreddit'].tolist(), danger_df['n_dangerous_trends'].tolist(),
            color=bar_c, alpha=0.85, zorder=3)
    ax.axvline(3, color=C['dark'], lw=1.2, ls='--', alpha=0.6, label='3 signals threshold')
    ax.legend(fontsize=7)
    style_ax(ax, '# Signals with Significant Worsening Trend\n(p<0.15, in bot direction — higher = alert)',
             'Count of worsening signals', '')
    ax.tick_params(axis='y', labelsize=6.5)
    # Annotate cluster
    for _, row in danger_df.iterrows():
        if row['n_dangerous_trends'] >= 3:
            ax.text(row['n_dangerous_trends'] + 0.05,
                    danger_df['subreddit'].tolist().index(row['subreddit']),
                    f"  ⚠ {row['cluster']}", fontsize=6, color=C['red'], va='center')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 2: Cluster-level signal trajectories ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('Cluster-Level Signal Trajectories Over Time',
                 fontsize=11, color=C['dark'], fontweight='bold')

    trajectory_signals = [
        ('ucr',              'UCR over time — rising = more upvotes without discussion'),
        ('simulacra_rate',   'Simulacra Rate % — rising = more high-score/zero-comment posts'),
        ('burst_score',      'Burst Score — rising = more coordinated early comments'),
        ('high_kpd_pct',     'High-KPD Account % — rising = more karma farmers'),
    ]
    for i, (sig, desc) in enumerate(trajectory_signals):
        ax = axes[i // 2][i % 2]
        if sig not in sigs.columns:
            ax.set_visible(False); continue
        for cl, ccolor in CLUSTER_COLOR.items():
            cg = sigs[sigs['cluster'] == cl].groupby('month')[sig].mean().reindex(months)
            if cg.isna().all(): continue
            x = range(len(months))
            ax.plot(x, cg.values, 'o-', label=cl, color=ccolor, lw=1.5, ms=3.5, alpha=0.85)
            # Regression line if enough data
            valid = [(xi, v) for xi, v in zip(x, cg.values) if not np.isnan(v)]
            if len(valid) >= 5:
                xv, yv = zip(*valid)
                res = stats.linregress(xv, yv)
                if res.pvalue < 0.15:
                    ax.plot(xv, [res.intercept + res.slope*xi for xi in xv],
                            '--', color=ccolor, lw=0.8, alpha=0.5)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, fontsize=5.5)
        ax.legend(fontsize=5.5, ncol=2, loc='best')
        style_ax(ax, desc, '', sig)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── Page 3: Month-over-month velocity — biggest movers ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('Month-over-Month Velocity  (last 3 months vs 12-month baseline)',
                 fontsize=11, color=C['dark'], fontweight='bold')

    # For each sub: avg of key bot signals in last 3 months vs full-period avg
    recent_months = sorted(sigs['month'].unique())[-3:]
    baseline_all  = sigs.groupby('subreddit')[avail].mean()
    recent_all    = sigs[sigs['month'].isin(recent_months)].groupby('subreddit')[avail].mean()

    # Composite: normalize each signal by direction, then average
    def composite_score(df):
        norm = df.copy()
        for col in avail:
            if col not in norm.columns: continue
            mn, mx = baseline_all[col].min(), baseline_all[col].max()
            if mx > mn:
                norm[col] = (norm[col] - mn) / (mx - mn)
            direction = SIGNAL_DIRECTION.get(col, 1)
            if direction < 0:
                norm[col] = 1 - norm[col]
        return norm[avail].mean(axis=1)

    base_comp   = composite_score(baseline_all)
    recent_comp = composite_score(recent_all)
    velocity    = (recent_comp - base_comp).dropna().sort_values(ascending=True)

    ax = axes[0]
    vcolors = [C['red'] if v > 0.05 else C['amber'] if v > 0 else C['sage'] for v in velocity.values]
    ax.barh(velocity.index.tolist(), velocity.values, color=vcolors, alpha=0.85, zorder=3)
    ax.axvline(0, color=C['dark'], lw=1, ls='-', alpha=0.4)
    ax.axvline(0.05, color=C['red'], lw=1, ls='--', alpha=0.6, label='Alert threshold')
    ax.legend(fontsize=7)
    style_ax(ax, 'Bot Activity Velocity\n(recent 3 months vs 12-month baseline)',
             'Δ composite score (positive = worsening)', '')
    ax.tick_params(axis='y', labelsize=6.5)

    # Accelerating subs: score trajectory with regression
    ax = axes[1]
    top_accel = velocity.tail(5).index.tolist()  # 5 most worsening
    for sub in top_accel:
        sg = sigs[sigs['subreddit'] == sub].sort_values('month')
        comp_vals = []
        for _, row in sg.iterrows():
            vals = {s: row.get(s, np.nan) for s in avail}
            normed = []
            for s, v in vals.items():
                mn = baseline_all[s].min(); mx = baseline_all[s].max()
                if mx > mn and not np.isnan(v):
                    nv = (v - mn) / (mx - mn)
                    normed.append(nv if SIGNAL_DIRECTION.get(s, 1) > 0 else 1 - nv)
            comp_vals.append(np.mean(normed) if normed else np.nan)
        xv = range(len(sg))
        ax.plot(xv, comp_vals, 'o-', label=f'r/{sub}', lw=2, ms=4, alpha=0.85)
        # Mark recent months
        ax.axvspan(len(sg)-3, len(sg)-0.5, color=C['rose'], alpha=0.08)

    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months, rotation=45, fontsize=6)
    ax.legend(fontsize=7)
    ax.text(0.98, 0.98, '← shaded = recent 3 months', transform=ax.transAxes,
            fontsize=6.5, ha='right', va='top', color=C['rose'])
    style_ax(ax, 'Composite Trajectory — Top 5 Accelerating Subs', '', 'Composite bot signal (0–1)')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # Return dangerous subs for findings
    dangerous = slope_df[slope_df['n_dangerous_trends'] >= 3]['subreddit'].tolist()
    accelerating = velocity[velocity > 0.05].index.tolist()
    return dangerous, accelerating


# ── §7 Findings & Weights ─────────────────────────────────────────────────────

# Full 6-component signal set. COMPONENT_SIGNALS_5 (below) is the subset that matches
# what's actually live in analyze_data_v2.py today — network is computed and exposed
# there but not yet summed into final_score (see calculate_scores/COMPONENT_SCORE_KEY).
# Both variants get calibrated below; referee_weights.py decides which one the
# production scorer should actually use, based on the weak-label ground truth, and
# writes that decision to findings.json as `live_weights`.
COMPONENT_SIGNALS_6 = {
    'engagement':   ['ucr', 'score_comm_corr', 'upvote_ratio_std', 'simulacra_rate'],
    'distribution': ['decay_slope', 'comm_cv'],  # score_cv retired — see compute_signals() docstring
    'account':      ['new_poster_pct', 'new_comm_pct', 'high_kpd_pct'],
    'ring':         ['burst_score', 'fast_ttfc_pct', 'recurrence_rate'],
    'temporal':     ['interval_cv', 'top3_concentration'],
    'network':      ['cross_sub_rate', 'gini_score'],  # near_dupe_rate excluded — see compute_signals()
}
COMPONENT_SIGNALS_5 = {k: v for k, v in COMPONENT_SIGNALS_6.items() if k != 'network'}
COMPONENT_SIGNALS = COMPONENT_SIGNALS_6  # back-compat default for any external references

OLD_WEIGHTS = {'account': 0.30, 'ring': 0.25, 'engagement': 0.20, 'temporal': 0.15, 'distribution': 0.10, 'network': 0.0}

KEY_FINDINGS = [
    ('F1', 'overlap_rate averages ~91.5% universally — organic first-mover advantage on Reddit, not a bot signal. Removed from ring component.'),
    ('F2', 'UCR (upvote:comment ratio) is the single strongest bot discriminator. Posts get upvoted without triggering discussion.'),
    ('F3', 'Engagement signals collectively explain the most variance in bot scores; under-weighted at original 20%.'),
    ('F4', 'KPD>500 and age<90d thresholds flag <5% of posts — too conservative. Recommend lowering KPD threshold to 200.'),
    ('F5', 'Political cluster (indiaspeaks, unitedstatesofindia, india) shows highest and most volatile bot scores.'),
    ('F6', 'score_cv and upvote_ratio_std reliably detect manufactured uniform voting across posts.'),
    ('F7', 'Cross-sub commenter overlap spikes around security/election events — suggests coordinated campaign activation.'),
    ('F8', 'Commenter churn >60% month-over-month in some subs suggests periodic account cycling after bans.'),
    ('F9', 'simulacra_rate (high score, near-zero comments) is elevated in cricket and political subs — pure upvote manipulation.'),
    ('F10', 'burst_score and fast_ttfc_pct are the valid ring signals; recurrence is a secondary indicator.'),
]


def derive_weights(sigs, component_signals=None):
    component_signals = component_signals or COMPONENT_SIGNALS_6
    signal_cv = {}
    for col in sigs.select_dtypes(include=[np.number]).columns:
        d = sigs[col].dropna()
        m = d.mean()
        if abs(m) > 0.001 and len(d) > 10:
            signal_cv[col] = float(d.std() / abs(m))

    comp_power = {}
    for comp, sig_list in component_signals.items():
        avail = [s for s in sig_list if s in signal_cv]
        comp_power[comp] = float(np.mean([signal_cv[s] for s in avail])) if avail else 0.05

    total = sum(comp_power.values())
    raw   = {k: v / total for k, v in comp_power.items()}

    FLOOR, CAP = 0.08, 0.42
    w = {k: max(FLOOR, min(CAP, v)) for k, v in raw.items()}
    total2 = sum(w.values())
    w = {k: round(v / total2, 3) for k, v in w.items()}
    diff = round(1.0 - sum(w.values()), 3)
    top_k = max(w, key=w.get)
    w[top_k] = round(w[top_k] + diff, 3)
    return w, comp_power


# Signals where a LOW raw value is the suspicious direction (uniform votes, robotic
# regular intervals, upvotes-without-discussion). Flipped before PCA so that a higher
# standardized value always means "more bot-like" across every signal.
FLIP_SIGNALS = {'comm_cv', 'upvote_ratio_std', 'score_comm_corr', 'interval_cv'}
# decay_slope is already high=suspicious (flatter/less-negative slope = more coordination-like) — no flip needed.


def derive_weights_pca(sigs, component_signals=None):
    """
    Alternative to derive_weights(): instead of weighting each component by the raw
    coefficient-of-variation of its signals (which conflates "varies a lot" with
    "detects bots" — a signal can vary just because subreddit content types differ),
    z-score every scored signal, sign-align so higher = more suspicious, run PCA, and
    weight each component by its constituent signals' squared loadings on PC1 — the
    dominant shared axis of variation across sub-months. Still not ground-truth
    validated on its own (see referee_weights.py for the weak-label comparison),
    but it accounts for collinearity between signals, which raw per-signal CV
    averaging does not.
    """
    component_signals = component_signals or COMPONENT_SIGNALS_6
    all_signals = [s for lst in component_signals.values() for s in lst]
    avail = [s for s in all_signals if s in sigs.columns]
    X = sigs[avail].dropna()
    if len(X) < 10:
        return {}, {}

    std = X.std().replace(0, 1)
    Xz = (X - X.mean()) / std
    for col in avail:
        if col in FLIP_SIGNALS:
            Xz[col] = -Xz[col]

    pca = PCA(n_components=min(3, len(avail)))
    pca.fit(Xz.values)
    loadings_pc1 = dict(zip(avail, pca.components_[0]))
    # PCA sign is arbitrary — orient so the majority of loadings are positive
    # (i.e. PC1 increasing tracks "more suspicious", not "less")
    if sum(1 for v in loadings_pc1.values() if v > 0) < len(loadings_pc1) / 2:
        loadings_pc1 = {k: -v for k, v in loadings_pc1.items()}

    comp_power = {}
    for comp, sig_list in component_signals.items():
        sub_avail = [s for s in sig_list if s in loadings_pc1]
        comp_power[comp] = float(np.mean([loadings_pc1[s] ** 2 for s in sub_avail])) if sub_avail else 0.0

    total = sum(comp_power.values()) or 1.0
    raw   = {k: v / total for k, v in comp_power.items()}
    FLOOR, CAP = 0.08, 0.42
    w = {k: max(FLOOR, min(CAP, v)) for k, v in raw.items()}
    total2 = sum(w.values())
    w = {k: round(v / total2, 3) for k, v in w.items()}
    diff  = round(1.0 - sum(w.values()), 3)
    top_k = max(w, key=w.get)
    w[top_k] = round(w[top_k] + diff, 3)

    meta = {
        'pc1_explained_variance_pct': round(float(pca.explained_variance_ratio_[0]) * 100, 1),
        'pc2_explained_variance_pct': round(float(pca.explained_variance_ratio_[1]) * 100, 1) if len(pca.explained_variance_ratio_) > 1 else None,
        'pc1_loadings': {k: round(float(v), 3) for k, v in
                          sorted(loadings_pc1.items(), key=lambda x: -abs(x[1]))},
        'n_observations': int(len(X)),
    }
    return w, meta


def section_7(sigs, posts, comms, pdf, dangerous=None, accelerating=None):
    section_header(pdf, '§7  Findings & Calibrated Weights',
                   '12 months × 25 subreddits × 300 observations — statistically derived',
                   color=C['rose'])
    dangerous    = dangerous    or []
    accelerating = accelerating or []

    # 5-component — matches what's actually live in analyze_data_v2.py today.
    calibrated, comp_power = derive_weights(sigs, COMPONENT_SIGNALS_5)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.patch.set_facecolor(C['bg'])
    fig.suptitle('Weight Calibration Results', fontsize=13, color=C['dark'], fontweight='bold')

    # Old vs new weights
    ax = axes[0, 0]
    comps  = list(OLD_WEIGHTS.keys())
    x      = np.arange(len(comps))
    w      = 0.35
    ax.bar(x - w/2, [OLD_WEIGHTS[c]*100 for c in comps], w, label='Original', color=C['slate'], alpha=0.7, zorder=3)
    ax.bar(x + w/2, [calibrated.get(c, 0)*100 for c in comps], w, label='Calibrated', color=C['rose'], alpha=0.85, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([c.capitalize() for c in comps], fontsize=8.5)
    ax.legend(fontsize=8)
    style_ax(ax, 'Weight Calibration: Before vs After', '', 'Weight %')
    for xi, c in zip(x, comps):
        old_v = OLD_WEIGHTS[c]*100
        new_v = calibrated.get(c, 0)*100
        delta = new_v - old_v
        ax.text(xi + w/2, new_v + 0.5, f'{delta:+.0f}%', ha='center', fontsize=7,
                color=C['red'] if delta < 0 else C['green'])

    # Component discriminative power
    ax = axes[0, 1]
    cp_sorted = sorted(comp_power.items(), key=lambda x: -x[1])
    cp_names  = [c.capitalize() for c, _ in cp_sorted]
    cp_vals   = [v for _, v in cp_sorted]
    med_cp    = np.mean(cp_vals)
    ax.barh(cp_names, cp_vals,
            color=[C['rose'] if v > med_cp else C['blue'] for v in cp_vals], alpha=0.85, zorder=3)
    ax.axvline(med_cp, color=C['dark'], lw=1.2, ls='--', label=f'Mean={med_cp:.2f}')
    ax.legend(fontsize=7)
    style_ax(ax, 'Component Discriminative Power\n(mean CV of constituent signals)', 'Mean signal CV', '')

    # Sub rankings with calibrated composite
    ax = axes[1, 0]
    key_sigs = ['ucr', 'score_cv', 'upvote_ratio_std', 'burst_score', 'high_kpd_pct', 'simulacra_rate']
    key_sigs = [s for s in key_sigs if s in sigs.columns]
    avg_sigs = sigs.groupby('subreddit')[key_sigs].mean()
    norm     = avg_sigs.copy()
    for col in key_sigs:
        mn, mx = norm[col].min(), norm[col].max()
        if mx > mn:
            norm[col] = (norm[col] - mn) / (mx - mn)
    # Invert signals where lower = more suspicious
    for col in ['score_cv', 'upvote_ratio_std']:
        if col in norm.columns:
            norm[col] = 1 - norm[col]
    composite = norm.mean(axis=1).sort_values(ascending=True)
    rc = [CLUSTER_COLOR.get(SUB_CLUSTER.get(s, 'Other'), C['slate']) for s in composite.index]
    ax.barh(composite.index.tolist(), composite.values, color=rc, alpha=0.85, zorder=3)
    style_ax(ax, 'Subreddit Rankings (calibrated composite, 0–1)', '', '')
    ax.tick_params(axis='y', labelsize=6.5)

    # Findings text
    ax = axes[1, 1]
    ax.set_facecolor(C['bg']); ax.axis('off')
    ax.set_title('Key Findings', fontsize=10, color=C['dark'], fontweight='bold', pad=8)
    colors_f = [C['rose'], C['amber'], C['sage'], C['lav'], C['blue'],
                C['rose'], C['amber'], C['sage'], C['lav'], C['blue']]
    extra_findings = []
    if dangerous:
        extra_findings.append(('⚠ TREND', f'Subs with 3+ worsening signal trends: {", ".join(dangerous[:4])}'))
    if accelerating:
        extra_findings.append(('⚠ ACCEL', f'Recently accelerating (last 3 months): {", ".join(accelerating[:4])}'))
    all_findings = list(KEY_FINDINGS) + extra_findings
    for i, item in enumerate(all_findings[:12]):
        fid, text = item
        color_bg = C['red'] if str(fid).startswith('⚠') else colors_f[i % len(colors_f)]
        ax.text(0.02, 0.97 - i * 0.079,
                f"{fid}  {text[:90]}{'…' if len(text) > 90 else ''}",
                transform=ax.transAxes, fontsize=6.5, color=C['dark'], va='top',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=color_bg,
                          alpha=0.30, edgecolor='none'))

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    return calibrated


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Reddit Bot Analysis — Comprehensive Report')
    p.add_argument('--months', nargs='+', default=None)
    p.add_argument('--output-dir', default=str(REPORT_DIR))
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts         = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path   = outdir / f'analysis_{ts}.pdf'
    latest_pdf = outdir / 'analysis_latest.pdf'
    json_path  = outdir / 'findings.json'

    print('\n' + '='*70)
    print('REDDIT BOT ANALYSIS — COMPREHENSIVE REPORT')
    print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*70)

    print('\n[1/8] Loading data…')
    posts, comms = load_all(months=args.months)

    print('[2/8] Computing per-sub-month signals…')
    sigs = compute_signals(posts, comms)

    print(f'[3/8] Rendering PDF → {pdf_path}')
    with PdfPages(pdf_path) as pdf:
        # Cover
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.patch.set_facecolor(C['dark']); ax.set_facecolor(C['dark']); ax.axis('off')
        ax.text(0.5, 0.72, 'Reddit India — Bot Activity Report',
                transform=ax.transAxes, fontsize=28, color='white',
                ha='center', fontweight='bold')
        ax.text(0.5, 0.60, 'Comprehensive Statistical Analysis',
                transform=ax.transAxes, fontsize=14, color=C['slate'], ha='center')
        ax.text(0.5, 0.50,
                f"25 Subreddits  ·  {posts['collection_month'].nunique()} Months  ·  "
                f"{len(posts):,} Posts  ·  {len(comms):,} Comment Rows",
                transform=ax.transAxes, fontsize=10, color=C['sage'], ha='center')
        ax.text(0.5, 0.42, datetime.now().strftime('%B %d, %Y'),
                transform=ax.transAxes, fontsize=9, color=C['amber'], ha='center')
        for i, s in enumerate(['§1 Data Portrait', '§2 Univariate Analysis', '§3 Signal Correlations',
                                '§4 Text Intelligence', '§5 Cross-Sub Network',
                                '§6 Event Calendar', '§7 Findings & Calibrated Weights']):
            ax.text(0.35, 0.30 - i*0.038, s, transform=ax.transAxes,
                    fontsize=9, color='white', alpha=0.65)
        pdf.savefig(fig, bbox_inches='tight', facecolor=C['dark']); plt.close()

        print('[4/8] §1 Data Portrait…')
        section_1(posts, comms, pdf)

        print('[5/8] §2 Univariate…')
        section_2(sigs, pdf)

        print('[6/8] §3 Correlations…')
        section_3(sigs, pdf)

        print('[7/8] §4 Text Intelligence…')
        section_4(posts, pdf)

        print('[7/8] §5 Cross-Sub Network…')
        section_5(posts, comms, pdf)

        print('[7/8] §6 Event Calendar…')
        section_6(posts, pdf)

        print('[7/8] §6b Temporal Trend Analysis…')
        dangerous, accelerating = section_6b(sigs, pdf)

        print('[8/8] §7 Findings & Weights…')
        calibrated = section_7(sigs, posts, comms, pdf, dangerous=dangerous, accelerating=accelerating)

        d = pdf.infodict()
        d['Title']   = 'Reddit India Bot Activity — Comprehensive Analysis'
        d['Author']  = 'RedditWatch'
        d['Subject'] = 'Bot detection: 25 Indian subreddits'

    shutil.copy(pdf_path, latest_pdf)

    print('  Computing PCA-based weights (alternative, for comparison)…')
    pca_weights, pca_meta = derive_weights_pca(sigs, COMPONENT_SIGNALS_5)

    print('  Computing 6-component candidates (network included, for referee_weights.py)…')
    calibrated_6comp, comp_power_6 = derive_weights(sigs, COMPONENT_SIGNALS_6)
    pca_weights_6comp, pca_meta_6  = derive_weights_pca(sigs, COMPONENT_SIGNALS_6)

    supervised_check = None
    wlc_path = REPORT_DIR.parent / 'output' / 'v2' / 'weak_label_classifier.json'
    if wlc_path.exists():
        with open(wlc_path) as f:
            supervised_check = json.load(f)

    findings = {
        'generated': datetime.now().isoformat(),
        'data_summary': {
            'n_posts':    int(len(posts)),
            'n_comments': int(len(comms)),
            'n_subs':     int(posts['subreddit'].nunique()),
            'n_months':   int(posts['collection_month'].nunique()),
            'months':     sorted(posts['collection_month'].unique().tolist()),
        },
        'calibrated_weights':  calibrated,
        'calibration_method':  'per-signal coefficient of variation, averaged per component (NOT validated against any bot/human ground truth)',
        'pca_weights':         pca_weights,
        'pca_meta':            pca_meta,
        'calibrated_weights_6comp': calibrated_6comp,
        'pca_weights_6comp':        pca_weights_6comp,
        'pca_meta_6comp':           pca_meta_6,
        'supervised_feature_check': supervised_check,
        'original_weights':    OLD_WEIGHTS,
        'component_signals':   COMPONENT_SIGNALS_6,
        'removed_signals':     ['overlap_rate', 'near_dupe_rate'],
        'removed_reason':      'overlap_rate ~91.5% universally — organic Reddit first-mover effect, not a bot signal. '
                                'near_dupe_rate median/p95 both 0 across the full corpus — essentially never fires.',
        'recommended_kpd_threshold': 200,
        'key_findings': [{'id': fid, 'text': text} for fid, text in KEY_FINDINGS],
        'alert_subs_trend':    dangerous,
        'alert_subs_accel':    accelerating,
        'note': 'live_weights (if present) is the referee-selected scheme from referee_weights.py — '
                'that, not calibrated_weights, is what analyze_data_v2.py actually uses when present.',
    }
    with open(json_path, 'w') as f:
        json.dump(findings, f, indent=2, default=str)

    print(f'\n  PDF:          {pdf_path}')
    print(f'  PDF (latest): {latest_pdf}')
    print(f'  Findings:     {json_path}')
    print('\n  Calibrated weights (CV-based, live — used by analyze_data_v2.py):')
    for k, v in sorted(calibrated.items(), key=lambda x: -x[1]):
        old = OLD_WEIGHTS.get(k, 0)
        arrow = '↑' if v > old else '↓' if v < old else '='
        print(f'    {k:<16} {old*100:.0f}% → {v*100:.0f}%  {arrow}')
    if pca_weights:
        print(f'\n  PCA weights (alternative, PC1 explains {pca_meta["pc1_explained_variance_pct"]:.1f}% of variance — for comparison only):')
        for k, v in sorted(pca_weights.items(), key=lambda x: -x[1]):
            cv = calibrated.get(k, 0)
            arrow = '↑' if v > cv else '↓' if v < cv else '='
            print(f'    {k:<16} CV={cv*100:.0f}% → PCA={v*100:.0f}%  {arrow}')
    print('='*70 + '\n')


if __name__ == '__main__':
    main()
