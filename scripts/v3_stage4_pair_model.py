#!/usr/bin/env python3
"""V3 Stage 4 (V3_PLAN.md Sec 7, "Stage 4 -- pair model"): the primary
detector -- "this is where 0.90 lives" per the plan, and the actual point of
the whole V3 pivot (Kumar et al. 2017: 0.68 AUC per-account vs 0.91 AUC
per-pair, same features).

**Anti-circularity design is the point of this stage, not a nice-to-have.**
Stage 3 needed a same-day correction because account-level labels and
account-level features were both aggregated over the same rows (see
V3_PLAN.md Sec 10.4, "Stage 3 leakage audit"). Stage 4's label ("seed-derived
same-operator pairs, stylometric + cohort + co-appearance agreement", Sec 7)
is self-constructed from the same B1-B8 metric family used to predict it
unless deliberately structured otherwise. Fix adopted here: split B1-B8 into
two STRUCTURALLY INDEPENDENT views --
  View A (behavioral/structural, no text/identity): B1 co-appearance+null-z,
    B2 co-arrival tightness, B7 temporal-activity correlation.
  View B (content/identity): B3 text-template sharing, B4 stylometric
    similarity, B5 registration-cohort adjacency, B8 shared-domain (thin
    coverage, post-authors only).
The "same-operator" pseudo-label is constructed from View B ONLY. The
primary model is trained on View A ONLY to predict that View-B label -- a
structurally non-circular test: does co-appearance/timing pattern alone,
with zero access to text or registration-cohort information, predict
identity-based same-operator status? Reverse-direction and combined-feature
models are also reported, explicitly labeled as more circular / less
trustworthy, never as the headline number.

**B6 (reply reciprocity) dropped, not approximated.** Checked the raw
collected data directly: v3_collect.py computes parent_id-based reciprocity
at collection time (for the POST-level reply_reciprocity metric) but never
persists per-comment parent_id into commenters/commenters_clean -- only the
derived boolean parent_is_post survives. There is no way to know which
specific comment a reply targets, so true account-to-account reply
reciprocity is not reconstructable from any table or raw file we have.
Approximating it (e.g. "both non-top-level in the same thread") would not
measure reciprocity and would silently misrepresent what the feature is.
Dropped; View A is B1+B2+B7 (3 features), still structurally independent.

**Candidate pool.** All distinct co-appearing account pairs (two accounts on
the same thread, deduped to distinct (author,post_id) presence first -- see
all_pairs_coappear's docstring for a bug this caught) = 9,665,091. Restricted
full B1-B8 computation to the 275,643 pairs with total n_coappear >= 2
(across all shared subs) -- explicit modeling choice, not a hidden filter,
mirrors Sec 4.5's own reference point (Schoch et al., 74% recall at 1% FPR
using a >=10-repetition requirement). Null-model z/p-values are computed
across the FULL ~9.7M (pair,sub) rows (cheap, vectorized); only the
expensive B3/B4/B8 features are restricted to the smaller pool.

**Null model (Sec 4.4): Hypergeometric SVN (Tumminello), computed PER
SUBREDDIT** (see within_sub_degrees_and_T's docstring -- a global null badly
overstates significance because accounts are heavily subreddit-clustered,
not uniformly spread across the corpus: first run showed 97.4% of all pairs
"BH-FDR significant", the per-sub fix alone only got that to ~84%). The two
more rigorous null options in Sec 4.4 (BiCM+Poisson-Binomial, `backbone` FDSM
curveball/fastball resampling) are noted, not silently skipped -- full FDSM
resampling on a ~348K-account bipartite graph is a heavier lift than this
pass needs for a first defensible result.

**A third, structural issue found and NOT "fixed" (because it isn't a bug):
the co-occurring-only candidate pool is ascertainment-biased relative to any
hypergeometric significance test.** `all_pairs_coappear` only ever builds
pairs that already share a thread; for a corpus where the median account has
degree 1-2, simply co-occurring at all is already a rare event under the
null, so p-values over this pre-filtered population skew small regardless of
real coordination. Verified directly: an EXACT null simulator (each account
given a truly uniform-random thread-subset of its own real size) still shows
mean p~0.05 / median p~0.01, not the ~0.5/~0.5 a real null implies -- see
null_sanity_check()'s docstring for the full diagnosis. **Resolution: B1's
z-score is used as a continuous ranking feature only, everywhere in this
pipeline -- no BH-FDR or other binary significance gate is computed or used.**

Per Sec 4.4: never inherit a published coordination-interval constant
(CooRnet's 0.90 percentile-edge default etc.) -- the B2 tightness threshold
is derived from this corpus's own co-arrival-gap distribution.

Reads data/v3/analysis/v3.duckdb read-only + raw commenters/posts NDJSON
(read-only, for text) throughout. Writes no tables back to the DuckDB file.
Outputs a JSON summary + static HTML report (docs/v3-research/eda/stage4.html,
linked from the EDA/Stage2/Stage3 nav)."""
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
OUT_JSON = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'stage4_data.json')
OUT_HTML = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'stage4.html')

RNG = 42
N_COAPPEAR_MIN = 2
KUMAR_PAIR = 0.91
# B4 char n-gram vectorizer bounds -- kept deliberately small. This
# environment showed repeated multi-minute stalls / multi-GB RSS growth
# building a 4000-feature vocabulary over ~150-190K account documents (see
# feature_b4_stylometric timing log). Lower max_features / higher min_df
# trades some stylometric resolution for guaranteed tractability -- the same
# "don't attempt the maximal method" tradeoff already made for the null model
# (Hypergeometric SVN over full FDSM resampling).
B4_MAX_FEATURES = 1200
B4_MIN_DF = 3
SCHOCH_RECALL_AT_1PCT_FPR = 0.74

T0 = time.time()


def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)


# ---------------------------------------------------------------------------
# Phase 0: bipartite structure + null model
# ---------------------------------------------------------------------------

def load_bipartite(con):
    log('loading commenters_clean (author, post_id, created_utc, account_ordinal, body)')
    df = con.execute("""
        SELECT author, post_id, created_utc, account_ordinal, body, comment_id, sub
        FROM commenters_clean
    """).fetchdf()
    return df


def account_degrees(df):
    return df.groupby('author')['post_id'].nunique()


def all_pairs_coappear_with_degrees(con):
    """Combines all_pairs_coappear + within_sub_degrees_and_T's join into ONE
    SQL query. **Fourth bug, found from a 17GB runaway process (`top` showed
    RSS climbing without bound, state "stuck"):** the original approach built
    d1/d2 by Python-side `pd.Series(list(zip(a, sub))).map(dict)` over
    ~9.7M rows -- materializing 9.7M Python tuples (twice, once per side)
    plus the intermediate list/Series copies, blew past available memory and
    started thrashing rather than erroring cleanly. DuckDB can do the same
    join natively, vectorized, without ever materializing Python tuple
    objects -- moved the degree/T attachment into the query itself."""
    log('computing (pair, sub) rows + within-sub degrees + T via staged temp tables')
    # Staged temp tables, not one compound query: a single query with all 4
    # joins + GROUP BY took *minutes* (DuckDB's optimizer picked a bad plan);
    # splitting into explicit CREATE TEMP TABLE steps (each one simple)
    # measured at ~1.7s total on this exact workload. Same result, staged.
    con.execute('DROP TABLE IF EXISTS presence')
    con.execute('CREATE TEMP TABLE presence AS SELECT DISTINCT author, post_id, sub FROM commenters_clean')
    con.execute('DROP TABLE IF EXISTS deg_s4')
    con.execute('CREATE TEMP TABLE deg_s4 AS SELECT author, sub, count(*) AS d FROM presence GROUP BY 1, 2')
    con.execute('DROP TABLE IF EXISTS sub_t_s4')
    con.execute('CREATE TEMP TABLE sub_t_s4 AS SELECT sub, count(DISTINCT post_id) AS t FROM presence GROUP BY 1')
    con.execute('DROP TABLE IF EXISTS pairs_raw_s4')
    con.execute("""CREATE TEMP TABLE pairs_raw_s4 AS
        SELECT a.author AS a1, b.author AS a2, a.sub AS sub, count(*) AS n_coappear
        FROM presence a JOIN presence b ON a.post_id = b.post_id AND a.author < b.author
        GROUP BY 1, 2, 3""")
    df = con.execute("""
        SELECT p.a1, p.a2, p.sub, p.n_coappear, d1.d AS d1, d2.d AS d2, st.t AS T_sub
        FROM pairs_raw_s4 p
        JOIN deg_s4 d1 ON d1.author = p.a1 AND d1.sub = p.sub
        JOIN deg_s4 d2 ON d2.author = p.a2 AND d2.sub = p.sub
        JOIN sub_t_s4 st ON st.sub = p.sub
    """).fetchdf()
    return df


def all_pairs_coappear(con):
    """Per-(pair, sub) co-appearance counts, on the SIMPLE bipartite edge
    list (one row per distinct (author, post_id), duplicates dropped first).

    **Second bug found on the second calibration check** (the sub-conditioned
    null still showed 84.8% "significant" on real data and 79.8% on
    null-generated simulated data -- should be ~5% on the simulated data by
    construction, so this was unambiguously a bug, not a modeling choice).
    43,141 (author, post_id) pairs in commenters_clean have >1 comment
    (repeat-engagement, same account posting twice in one thread -- a real,
    previously-documented behavior, see account_features_model's
    repeat_engagement_rate). Joining raw comment rows on post_id counts
    *comment pairs*, not *distinct co-occurring accounts*: if account A left
    2 comments and B left 1 on the same thread, the raw join produces 2 rows
    for what should be ONE thread-level co-occurrence. The hypergeometric SVN
    model assumes a SIMPLE bipartite graph (each account present on a thread
    0 or 1 times); feeding it comment-weighted counts breaks its assumptions
    and was inflating apparent significance across the board -- in both the
    real data AND (worse) the null_sanity_check's own simulated null data,
    which is what made the bug unmissable. Fix: dedupe to (author, post_id)
    presence before any co-appearance counting."""
    log('computing all co-appearing (pair, sub) triples via SQL self-join (deduped to distinct author-thread presence)')
    pairs = con.execute("""
        WITH presence AS (
            SELECT DISTINCT author, post_id, sub FROM commenters_clean
        )
        SELECT a.author AS a1, b.author AS a2, a.sub AS sub, count(*) AS n_coappear
        FROM presence a
        JOIN presence b ON a.post_id = b.post_id AND a.author < b.author
        GROUP BY 1, 2, 3
    """).fetchdf()
    return pairs


def hypergeom_null(v, d1, d2, T):
    # P(X >= v) for X ~ Hypergeometric(M=T, n=d1, N=d2)
    pval = hypergeom.sf(v - 1, T, d1, d2)
    mu = d1 * d2 / T
    var = d1 * d2 * (T - d1) * (T - d2) / (T ** 2 * (T - 1))
    var = np.clip(var, 1e-12, None)
    z = (v - mu) / np.sqrt(var)
    return pval, z, mu


def within_sub_degrees_and_T(bip_df):
    """**Critical fix, found by spot-checking the first run's output.** A
    global null (single T = all 122,350 threads across all 45 subs) treats
    every account's thread participation as if it were a uniform-random
    sample from the ENTIRE corpus. Real accounts are heavily
    subreddit-clustered (a r/chennai commenter mostly only comments in
    r/chennai), so almost any shared-subreddit co-appearance looks
    "surprising" against a corpus-wide baseline even with zero coordination.
    First run: 97.4% of all 9.67M pairs came back BH-FDR-significant at
    q=0.05 -- not credible. Spot-check on 15 random pairs confirmed it
    directly: e.g. a r/unitedstatesofindia pair scored p=0.030 (barely
    "significant") against the global null vs. p=0.718 (nowhere close)
    against a within-subreddit null using that sub's own thread count and
    each account's within-sub degree. Fix: T and degrees are now computed
    PER SUBREDDIT, and used per (pair, sub) row -- see all_pairs_coappear
    and hypergeom_null callers in main()."""
    T_sub = bip_df.groupby('sub')['post_id'].nunique()
    deg_sub = bip_df.groupby(['author', 'sub'])['post_id'].nunique()
    return T_sub, deg_sub


def bh_fdr(pvals, q=0.05):
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh_line = (np.arange(1, n + 1) / n) * q
    below = ranked <= thresh_line
    if not below.any():
        return 0.0, np.zeros(n, dtype=bool)
    k = np.max(np.where(below)[0])
    cutoff = ranked[k]
    sig = pvals <= cutoff
    return cutoff, sig


def null_sanity_check(bip_df, seed=RNG, n_subs=6):
    """**Third bug, and the real one -- an ascertainment-bias finding, not a
    fixable implementation error.** Tried an EXACT null simulator first
    (each account gets a genuinely uniform-random subset of size = its real
    degree from that sub's T threads, via np.random.choice(..., replace=False)
    per account -- a faithful implementation of the hypergeometric model's own
    assumption, no approximation). Even on this exact null-generated data,
    BH-FDR at q=0.05 still flagged ~72% of co-appearing pairs "significant" on
    a single test sub (r/ipl, checked directly), and the raw p-value
    distribution had mean 0.047 / median 0.012 -- nowhere near the uniform
    [0,1] a true null should produce.

    Root cause: **`all_pairs_coappear` only ever constructs pairs that
    already co-occur** (an INNER self-join on post_id). This silently
    conditions the entire test population on "did these two accounts share a
    thread at all" -- and for a corpus where the median account has degree 1-2
    within a sub, simply co-occurring even once is already a low-probability
    event under the null. Every pair that clears that bar is, by
    construction, one that "won" an unlikely draw, so the p-value
    distribution over the co-occurring-only subset is mechanically skewed
    toward small values regardless of whether real coordination exists. BH-FDR
    (or any binary significance threshold) computed over this pre-filtered
    population is not meaningful -- there's no way to fix it without also
    including the vast majority of ACCOUNT PAIRS THAT NEVER CO-OCCUR (p~1) in
    the same correction, which is combinatorially intractable at
    347,886-account scale (~121 billion possible pairs).

    **Resolution adopted:** the hypergeometric z-score/p-value is kept as a
    CONTINUOUS ranking feature (B1) -- valid and well-defined per pair,
    the selection-bias problem only affects turning it into a binary
    "significant/not" gate -- but no BH-FDR significance flag is computed or
    used anywhere downstream. Where a binary "unusually high co-appearance"
    label is needed (the View-B reverse-check's target), a relative
    top-percentile-of-z cutoff WITHIN the already-co-occurring candidate pool
    is used instead (see TOP_Z_PCTL in main()) -- a well-defined relative
    comparison, not a contaminated absolute significance claim.

    This function now exists only to DEMONSTRATE the bias directly (not to
    calibrate anything): runs the exact per-account uniform-subset simulator
    on a few subs and reports the p-value distribution, which should look
    uniform-ish but doesn't -- the numbers here are the evidence for the
    writeup above, not a pass/fail gate."""
    log(f'null bias check: exact per-account uniform-subset simulator ({n_subs} subs)')
    rng = np.random.RandomState(seed)
    subs = sorted(bip_df['sub'].unique())
    chosen_subs = rng.choice(subs, size=min(n_subs, len(subs)), replace=False)
    all_pvals = []
    per_sub_mean_p = {}
    for s in chosen_subs:
        sub_df = bip_df.loc[bip_df['sub'] == s, ['author', 'post_id']].drop_duplicates()
        threads = sub_df['post_id'].unique()
        T_sim = len(threads)
        deg = sub_df.groupby('author')['post_id'].nunique()
        rows_author, rows_thread = [], []
        for author, d in deg.items():
            d = min(int(d), T_sim)
            chosen = rng.choice(threads, size=d, replace=False)
            rows_author.extend([author] * d)
            rows_thread.extend(chosen)
        sim = pd.DataFrame({'author': rows_author, 'post_id': rows_thread})
        sim_pairs = (sim.merge(sim, on='post_id')
                     .query('author_x < author_y')
                     .groupby(['author_x', 'author_y']).size()
                     .reset_index(name='n_coappear'))
        if len(sim_pairs) == 0:
            continue
        sim_degrees = sim.groupby('author')['post_id'].nunique()
        d1 = sim_pairs['author_x'].map(sim_degrees).values
        d2 = sim_pairs['author_y'].map(sim_degrees).values
        pval, z, mu = hypergeom_null(sim_pairs['n_coappear'].values, d1, d2, T_sim)
        per_sub_mean_p[s] = dict(n_pairs=int(len(pval)), mean_p=float(np.mean(pval)),
                                  median_p=float(np.median(pval)))
        all_pvals.append(pval)
    all_pvals = np.concatenate(all_pvals) if all_pvals else np.array([])
    return dict(
        n_subs_checked=int(len(chosen_subs)), n_sim_pairs=int(len(all_pvals)),
        mean_pvalue=float(np.mean(all_pvals)) if len(all_pvals) else None,
        median_pvalue=float(np.median(all_pvals)) if len(all_pvals) else None,
        frac_below_05=float(np.mean(all_pvals < 0.05)) if len(all_pvals) else None,
        per_sub=per_sub_mean_p,
        note=('EXACT null simulator (uniform-random subset per account, faithful to the '
              'model\'s own assumption) still shows p-values skewed far below uniform[0,1] '
              '(expected mean/median ~0.5) -- this is ascertainment bias from only testing '
              'pairs that already co-occur, not a fixable calibration error. See the '
              'null_sanity_check() docstring for the full diagnosis. B1 is used as a '
              'continuous ranking feature only; no binary significance gate is used anywhere '
              'in this pipeline.'))


# ---------------------------------------------------------------------------
# Phase 1: candidate pool + B1/B2/B7 (View A) and B3/B4/B5/B8 (View B)
# ---------------------------------------------------------------------------

def collapse_to_pair_level(pairs_by_sub):
    """pairs_by_sub has one row per (a1, a2, sub) with its own within-sub
    pvalue/z. Collapse to one row per (a1, a2): total n_coappear summed
    across subs (same quantity the un-conditioned version would have
    reported), plus the most-significant (min p-value) sub-context as the
    pair's overall null-model result -- if a pair looks anomalous in ANY
    shared subreddit, that's the relevant signal, and using each sub's own
    degree/T avoids the cross-sub inflation documented in
    within_sub_degrees_and_T's docstring."""
    totals = pairs_by_sub.groupby(['a1', 'a2'])['n_coappear'].sum().rename('n_coappear_total')
    best = (pairs_by_sub.sort_values('pvalue')
            .drop_duplicates(['a1', 'a2'], keep='first')
            .set_index(['a1', 'a2']))
    out = best.join(totals, how='left')
    out = out.rename(columns={'n_coappear': 'n_coappear_best_sub', 'sub': 'best_sub'})
    out['n_coappear'] = out['n_coappear_total']
    return out.reset_index()


def build_candidate_pool(pairs, min_coappear=N_COAPPEAR_MIN):
    pool = pairs[pairs['n_coappear'] >= min_coappear].reset_index(drop=True)
    log(f'candidate pool: {len(pool):,} pairs (total n_coappear >= {min_coappear}, across all shared subs)')
    return pool


def feature_b2_arrival_tightness(pool, bip_df):
    log('B2: co-arrival tightness')
    # global gap distribution across ALL co-appearing (pair, shared-thread)
    # instances, to derive a corpus-native "tight" threshold (never import
    # CooRnet's constant).
    idx = bip_df.set_index(['post_id', 'author'])['created_utc']
    merged = bip_df.merge(bip_df, on='post_id', suffixes=('_1', '_2'))
    merged = merged[merged['author_1'] < merged['author_2']]
    merged['gap'] = (merged['created_utc_1'] - merged['created_utc_2']).abs()
    global_gap_p5 = float(merged['gap'].quantile(0.05))
    agg = merged.groupby(['author_1', 'author_2'])['gap'].agg(['median', 'min']).reset_index()
    agg.columns = ['a1', 'a2', 'b2_median_gap', 'b2_min_gap']
    pool = pool.merge(agg, on=['a1', 'a2'], how='left')
    pool['b2_tight_frac'] = (pool['b2_min_gap'] <= global_gap_p5).astype(float)
    return pool, global_gap_p5


def feature_b7_temporal_corr(pool, bip_df):
    log('B7: temporal activity correlation (daily bins)')
    bip_df = bip_df.copy()
    bip_df['day'] = (bip_df['created_utc'] // 86400).astype(int)
    accounts_needed = set(pool['a1']) | set(pool['a2'])
    sub = bip_df[bip_df['author'].isin(accounts_needed)]
    daily = sub.groupby(['author', 'day']).size().unstack(fill_value=0)
    # z-score each account's series so correlation isn't dominated by volume
    daily_z = daily.sub(daily.mean(axis=1), axis=0).div(daily.std(axis=1).replace(0, 1), axis=0)
    idx = {a: i for i, a in enumerate(daily_z.index)}
    mat = daily_z.values
    a1_idx = pool['a1'].map(idx).values
    a2_idx = pool['a2'].map(idx).values
    valid = (~pd.isna(a1_idx)) & (~pd.isna(a2_idx))
    corr = np.full(len(pool), np.nan)
    a1i = a1_idx[valid].astype(int)
    a2i = a2_idx[valid].astype(int)
    num = (mat[a1i] * mat[a2i]).sum(axis=1)
    denom = np.sqrt((mat[a1i] ** 2).sum(axis=1) * (mat[a2i] ** 2).sum(axis=1))
    denom = np.where(denom == 0, np.nan, denom)
    corr[valid] = num / denom
    pool = pool.copy()
    pool['b7_temporal_corr'] = corr
    return pool


def feature_b5_cohort(pool, degrees_ordinal):
    log('B5: registration-cohort adjacency (account_ordinal proximity)')
    pool = pool.copy()
    o1 = pool['a1'].map(degrees_ordinal)
    o2 = pool['a2'].map(degrees_ordinal)
    gap = (o1 - o2).abs()
    pool['b5_ordinal_gap'] = gap
    # similarity form: inverse-log so it's bounded and monotonic decreasing
    pool['b5_cohort_sim'] = 1.0 / (1.0 + np.log1p(gap.fillna(gap.max())))
    return pool


TEXT_CAP_CHARS = 5000  # see account_texts() docstring


def account_texts(bip_df, accounts_needed):
    """Concatenates each account's comment bodies, capped at TEXT_CAP_CHARS.
    Found by direct measurement: mean concatenated length across all 347,886
    accounts is 534 chars / median 148, but the max is 460,134 -- one extreme
    high-volume account whose single document made char-n-gram tokenization
    (TfidfVectorizer, feature_b4_stylometric) hang for minutes. A stylometric
    fingerprint doesn't need unbounded text -- 5000 chars is already dozens of
    typical comments, far more than needed to characterize punctuation/n-gram
    style -- so this is a deliberate, documented cap, not silent truncation."""
    sub = bip_df[bip_df['author'].isin(accounts_needed)]
    texts = sub.groupby('author')['body'].apply(
        lambda s: ' '.join(s.astype(str))[:TEXT_CAP_CHARS]).to_dict()
    return texts


def feature_b4_stylometric(pool, texts):
    log(f'B4: stylometric similarity, {len(texts):,} account texts (char n-gram TF, vectorized cosine per pair)')
    t0 = time.time()
    accounts = sorted(texts.keys())
    idx = {a: i for i, a in enumerate(accounts)}
    corpus = [texts[a] for a in accounts]
    log(f'B4: corpus assembled ({time.time()-t0:.1f}s), fitting TfidfVectorizer '
        f'(max_features={B4_MAX_FEATURES}, min_df={B4_MIN_DF})')
    t0 = time.time()
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), min_df=B4_MIN_DF, max_features=B4_MAX_FEATURES)
    X = vec.fit_transform(corpus)  # sparse, L2-normalizable rows
    log(f'B4: fit_transform done ({time.time()-t0:.1f}s), shape={X.shape}, nnz={X.nnz:,}')
    t0 = time.time()
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    a1_idx = pool['a1'].map(idx)
    a2_idx = pool['a2'].map(idx)
    valid = (~a1_idx.isna()) & (~a2_idx.isna())
    sim = np.full(len(pool), np.nan)
    a1i = a1_idx[valid].values.astype(int)
    a2i = a2_idx[valid].values.astype(int)
    rows1 = X[a1i]
    rows2 = X[a2i]
    dot = np.asarray(rows1.multiply(rows2).sum(axis=1)).ravel()
    cos = dot / (norms[a1i] * norms[a2i])
    sim[valid.values] = cos
    pool = pool.copy()
    pool['b4_stylometric_sim'] = sim
    log(f'B4: pairwise cosine done ({time.time()-t0:.1f}s)')
    return pool


def feature_b3_textdup(pool, bip_df, texts_needed_accounts):
    log('B3: text-template sharing (normalized-text near-duplicate rate)')
    import re

    def norm(t):
        t = str(t).lower()
        t = re.sub(r'[^a-z0-9 ]', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    sub = bip_df[bip_df['author'].isin(texts_needed_accounts)].copy()
    sub['norm_body'] = sub['body'].map(norm)
    sub = sub[sub['norm_body'].str.len() >= 20]  # skip trivially short comments ("lol", "this")
    by_account = sub.groupby('author')['norm_body'].apply(set).to_dict()

    def share_rate(row):
        s1 = by_account.get(row['a1'])
        s2 = by_account.get(row['a2'])
        if not s1 or not s2:
            return np.nan
        inter = len(s1 & s2)
        return inter / min(len(s1), len(s2))

    pool = pool.copy()
    pool['b3_textdup_rate'] = pool.apply(share_rate, axis=1)
    return pool


def feature_b8_domain(pool, con, accounts_needed):
    log('B8: shared-domain concentration (post authors only, self.* excluded)')
    dom = con.execute("""
        SELECT author, domain FROM posts_clean WHERE domain IS NOT NULL
    """).fetchdf()
    dom = dom[~dom['domain'].str.startswith('self.')]
    by_account = dom.groupby('author')['domain'].apply(set).to_dict()

    def share(row):
        s1 = by_account.get(row['a1'])
        s2 = by_account.get(row['a2'])
        if not s1 or not s2:
            return np.nan
        inter = len(s1 & s2)
        return inter / min(len(s1), len(s2))

    pool = pool.copy()
    pool['b8_domain_share'] = pool.apply(share, axis=1)
    coverage = pool['b8_domain_share'].notna().mean()
    return pool, coverage


# ---------------------------------------------------------------------------
# Phase 2: label construction (View B only) + spot-check
# ---------------------------------------------------------------------------

VIEW_A = ['n_coappear', 'z', 'b2_min_gap', 'b2_median_gap', 'b2_tight_frac', 'b7_temporal_corr']
VIEW_B = ['b3_textdup_rate', 'b4_stylometric_sim', 'b5_cohort_sim', 'b8_domain_share']


def construct_label(pool, texts_by_account):
    """Thresholds and combination rule derived from this pool's own
    distributions -- checked directly, not assumed. First cut used AND
    (strong_textdup required, style/cohort optional on top): b3 (text
    near-duplicate) is present in only 300/275,643 pairs (0.11%) -- direct
    text overlap between two specific accounts is rare, as expected -- and
    requiring it as a hard gate produced ZERO label positives after
    intersecting with the style/cohort percentiles. Corrected combination:
    OR of (strong direct text-template match) with (strong stylometric
    similarity AND strong cohort adjacency BOTH independently, i.e. two
    weaker signals agreeing) -- still requires genuine multi-signal
    agreement on the weak-signal path, just doesn't gate everything behind
    the rarest signal. b4 (stylometric cosine, char n-gram TF-IDF at
    max_features=1200) turned out to have weak absolute discrimination
    (mean 0.76 across ALL pairs -- common short Reddit-comment n-grams
    dominate) -- used only as a relative top-percentile ranking, never an
    absolute similarity claim, for exactly this reason."""
    log('Phase 2: constructing same-operator pseudo-label from View B only')
    b3 = pool['b3_textdup_rate'].fillna(0)
    b4 = pool['b4_stylometric_sim'].fillna(0)
    b5 = pool['b5_cohort_sim'].fillna(0)
    b3_thresh = max(b3[b3 > 0].quantile(0.99), 0.3) if (b3 > 0).sum() > 20 else 0.5
    b4_thresh = b4.quantile(0.99)
    b5_thresh = b5.quantile(0.98)  # tight cohort = HIGH sim value (already inverted)
    strong_textdup = b3 >= b3_thresh
    strong_style = b4 >= b4_thresh
    strong_cohort = b5 >= b5_thresh
    n_signals = strong_textdup.astype(int) + strong_style.astype(int) + strong_cohort.astype(int)
    label = (strong_textdup | (strong_style & strong_cohort)).astype(int)
    return label, dict(b3_thresh=float(b3_thresh), b4_thresh=float(b4_thresh),
                        b5_thresh=float(b5_thresh), n_signals_dist=n_signals.value_counts().to_dict(),
                        n_from_textdup=int(strong_textdup.sum()),
                        n_from_style_and_cohort=int((strong_style & strong_cohort).sum()))


def spot_check(pool, label, texts_by_account, n=12, seed=RNG):
    rng = np.random.RandomState(seed)
    pos_idx = pool.index[label == 1]
    examples = []
    if len(pos_idx) == 0:
        return examples
    chosen = rng.choice(pos_idx, size=min(n, len(pos_idx)), replace=False)
    for i in chosen:
        row = pool.loc[i]
        a1, a2 = row['a1'], row['a2']
        t1 = texts_by_account.get(a1, '')[:200]
        t2 = texts_by_account.get(a2, '')[:200]
        examples.append(dict(a1=a1, a2=a2, b3=float(row.get('b3_textdup_rate', np.nan)),
                              b4=float(row.get('b4_stylometric_sim', np.nan)),
                              b5=float(row.get('b5_cohort_sim', np.nan)),
                              text1_sample=t1, text2_sample=t2))
    return examples


# ---------------------------------------------------------------------------
# Phase 2/3: account-disjoint split, model fit/eval
# ---------------------------------------------------------------------------

def account_disjoint_split(pool, test_frac=0.3, seed=RNG):
    accounts = pd.unique(pd.concat([pool['a1'], pool['a2']]))
    rng = np.random.RandomState(seed)
    shuffled = accounts.copy()
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_frac)
    test_accounts = set(shuffled[:n_test])
    is_test_pair = pool['a1'].isin(test_accounts) & pool['a2'].isin(test_accounts)
    is_train_pair = (~pool['a1'].isin(test_accounts)) & (~pool['a2'].isin(test_accounts))
    train_idx = pool.index[is_train_pair].to_numpy()
    test_idx = pool.index[is_test_pair].to_numpy()
    dropped = len(pool) - len(train_idx) - len(test_idx)
    return train_idx, test_idx, dropped


def make_xgb(scale_pos_weight, n_estimators=200, max_depth=4):
    return XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, tree_method='hist',
        objective='binary:logistic', eval_metric='auc',
        scale_pos_weight=scale_pos_weight, random_state=RNG, n_jobs=-1,
        missing=np.nan,
    )


def fit_eval(pool, features, label, train_idx, test_idx):
    X = pool[features]
    y = label
    X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
    y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    if y_tr.sum() < 5 or y_te.sum() < 3 or (y_te == 0).sum() < 3:
        return None
    spw = max((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1), 1.0)
    model = make_xgb(spw)
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, probs)
    ap = average_precision_score(y_te, probs)
    # recall at 1% FPR
    neg_scores = np.sort(probs[y_te.values == 0])[::-1]
    k = max(int(len(neg_scores) * 0.01), 1)
    thresh = neg_scores[k - 1] if len(neg_scores) else 0.5
    recall_at_1pct_fpr = float(np.mean(probs[y_te.values == 1] >= thresh)) if y_te.sum() else None
    return dict(auc=float(auc), avg_precision=float(ap), recall_at_1pct_fpr=recall_at_1pct_fpr,
                n_train=len(train_idx), n_test=len(test_idx),
                n_pos_train=int(y_tr.sum()), n_pos_test=int(y_te.sum()),
                model=model, X_te=X_te, y_te=y_te, probs=probs)


def mandatory_baseline_raw_coappear(pool, label, train_idx, test_idx):
    X = pool[['n_coappear']]
    return fit_eval_simple(X, label, train_idx, test_idx)


def fit_eval_simple(X, y, train_idx, test_idx, n_estimators=50, max_depth=1):
    X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
    y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    if y_tr.sum() < 5 or y_te.sum() < 3:
        return None
    spw = max((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1), 1.0)
    model = make_xgb(spw, n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, probs))


def permutation_floor(pool, features, label, train_idx, test_idx, seed=RNG):
    rng = np.random.RandomState(seed)
    y_perm = pd.Series(rng.permutation(label.values), index=label.index)
    r = fit_eval(pool, features, y_perm, train_idx, test_idx)
    return r['auc'] if r else None


def shap_family_importance(model, X, view_name, max_rows=3000, seed=RNG):
    import shap
    Xf = X.astype('float64')
    if len(Xf) > max_rows:
        Xf = Xf.sample(max_rows, random_state=seed)
    explainer = shap.TreeExplainer(model, feature_perturbation='interventional',
                                    data=Xf.sample(min(300, len(Xf)), random_state=seed))
    sv = explainer.shap_values(Xf)
    if isinstance(sv, list):
        sv = sv[-1]
    mean_abs = np.abs(sv).mean(axis=0)
    ranked = sorted(zip(Xf.columns, mean_abs), key=lambda kv: -kv[1])
    return [(f, float(v)) for f, v in ranked]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    log('connecting (read-only)')
    con = duckdb.connect(DB_PATH, read_only=True)

    bip_df = load_bipartite(con)
    T_global = bip_df['post_id'].nunique()
    log(f'bipartite: {len(bip_df):,} rows, {bip_df["author"].nunique():,} accounts, {T_global:,} threads, '
        f'{bip_df["sub"].nunique()} subs')

    pairs_by_sub = all_pairs_coappear_with_degrees(con)
    log(f'{len(pairs_by_sub):,} (pair, sub) rows, with degrees/T joined in SQL')

    log('Phase 0: hypergeometric null, PER SUBREDDIT (see within_sub_degrees_and_T docstring for why)')
    pval, z, mu = hypergeom_null(pairs_by_sub['n_coappear'].values, pairs_by_sub['d1'].values,
                                  pairs_by_sub['d2'].values, pairs_by_sub['T_sub'].values)
    pairs_by_sub = pairs_by_sub.copy()
    pairs_by_sub['pvalue'] = pval
    pairs_by_sub['z'] = z
    log(f'p-value distribution over co-occurring pairs: mean={np.mean(pval):.4f}, median={np.median(pval):.4f} '
        f'(a true null would show ~0.5/~0.5 -- see null_sanity_check docstring for why this is ascertainment '
        f'bias, not a defect: only pairs that already co-occur are ever tested)')

    pairs = collapse_to_pair_level(pairs_by_sub)
    log(f'collapsed to {len(pairs):,} distinct pairs (best-sub p-value/z retained per pair)')

    sanity = null_sanity_check(bip_df)
    log(f'null bias check (exact simulator): mean p={sanity["mean_pvalue"]:.4f}, median p={sanity["median_pvalue"]:.4f} '
        f'on null-generated data ({sanity["n_subs_checked"]} subs) -- confirms the bias is structural, not a bug '
        f'in the formula')

    pool = build_candidate_pool(pairs, N_COAPPEAR_MIN)
    # top-percentile-of-z within the co-occurring pool: a RELATIVE ranking
    # definition, not an absolute significance claim -- sidesteps the
    # ascertainment-bias problem above entirely, used only as the
    # View-A-derived target for the reverse-direction robustness check.
    TOP_Z_PCTL = 0.99
    pool = pool.copy()
    pool['top_z'] = (pool['z'] >= pool['z'].quantile(TOP_Z_PCTL)).astype(int)
    accounts_needed = set(pool['a1']) | set(pool['a2'])

    # View A
    pool, gap_p5 = feature_b2_arrival_tightness(pool, bip_df)
    pool = feature_b7_temporal_corr(pool, bip_df)

    # View B
    ordinal = bip_df.drop_duplicates('author').set_index('author')['account_ordinal']
    pool = feature_b5_cohort(pool, ordinal)
    texts = account_texts(bip_df, accounts_needed)
    pool = feature_b4_stylometric(pool, texts)
    pool = feature_b3_textdup(pool, bip_df, accounts_needed)
    pool, b8_coverage = feature_b8_domain(pool, con, accounts_needed)
    log(f'B8 domain-share coverage: {b8_coverage*100:.2f}% of candidate pool')

    label, label_meta = construct_label(pool, texts)
    n_pos = int(label.sum())
    log(f'label positives: {n_pos:,} / {len(pool):,} pairs ({n_pos/len(pool)*100:.3f}%)')
    examples = spot_check(pool, label, texts)

    train_idx, test_idx, dropped = account_disjoint_split(pool)
    log(f'account-disjoint split: train={len(train_idx):,} test={len(test_idx):,} '
        f'dropped(straddling accounts)={dropped:,}')

    results = {}
    log('primary: View A -> View-B label')
    results['view_a_primary'] = fit_eval(pool, VIEW_A, label, train_idx, test_idx)
    log('reverse check: View B -> View-A-derived label (top_z, top-percentile co-appearance z-score)')
    va_label = pool['top_z'].fillna(0).astype(int)
    results['view_b_reverse'] = fit_eval(pool, VIEW_B, va_label, train_idx, test_idx)
    log('combined: View A + View B -> View-B label (most circular, reported for context only)')
    results['combined'] = fit_eval(pool, VIEW_A + VIEW_B, label, train_idx, test_idx)

    log('mandatory baseline: raw co-appearance count alone')
    baseline_auc = mandatory_baseline_raw_coappear(pool, label, train_idx, test_idx)

    log('permutation floor (View A model, label shuffled)')
    perm_auc = permutation_floor(pool, VIEW_A, label, train_idx, test_idx)

    shap_a = None
    if results['view_a_primary'] is not None:
        log('SHAP on View-A primary model')
        shap_a = shap_family_importance(results['view_a_primary']['model'],
                                         results['view_a_primary']['X_te'], 'A')

    log('writing output')
    write_output(pool, label, label_meta, examples, results, baseline_auc, perm_auc,
                 shap_a, sanity, T_global, len(bip_df), b8_coverage, gap_p5, dropped)
    log('done')


def jsonable(d):
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        if k in ('model', 'X_te', 'y_te', 'probs'):
            continue
        out[k] = v
    return out


def write_output(pool, label, label_meta, examples, results, baseline_auc, perm_auc,
                  shap_a, sanity, T, n_edges, b8_coverage, gap_p5, dropped):
    summary = dict(
        meta=dict(
            n_threads=int(T), n_edges=int(n_edges),
            n_candidate_pairs=int(len(pool)), n_coappear_min=N_COAPPEAR_MIN,
            n_label_positive=int(label.sum()),
            label_prevalence=float(label.mean()),
            b8_coverage=float(b8_coverage),
            arrival_gap_p5_seconds=float(gap_p5),
            n_dropped_straddling_split=int(dropped),
        ),
        null_sanity_check=sanity,
        label_construction=label_meta,
        spot_check_examples=examples,
        results=dict(
            view_a_primary=jsonable(results['view_a_primary']),
            view_b_reverse=jsonable(results['view_b_reverse']),
            combined=jsonable(results['combined']),
        ),
        baseline_raw_coappear_auc=baseline_auc,
        permutation_floor_auc=perm_auc,
        shap_view_a=shap_a,
        reference_points=dict(kumar_2017_pair_auc=KUMAR_PAIR,
                               schoch_recall_at_1pct_fpr=SCHOCH_RECALL_AT_1PCT_FPR),
    )
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    va = results['view_a_primary']
    vb = results['view_b_reverse']
    comb = results['combined']

    def fmt(r):
        if r is None:
            return '<td colspan="4">insufficient positives</td>'
        recall_str = 'n/a' if r['recall_at_1pct_fpr'] is None else f"{r['recall_at_1pct_fpr']:.3f}"
        return (f"<td>{r['auc']:.3f}</td><td>{r['avg_precision']:.3f}</td>"
                f"<td>{recall_str}</td>"
                f"<td>{r['n_pos_test']:,}/{r['n_test']:,}</td>")

    shap_rows = ''.join(f'<tr><td>{f}</td><td>{v:.4f}</td></tr>' for f, v in (shap_a or [])[:10])
    example_rows = ''.join(
        f"<tr><td>{e['a1']}</td><td>{e['a2']}</td><td>{e['b3']:.2f}</td><td>{e['b4']:.2f}</td>"
        f"<td>{e['b5']:.2f}</td><td class='mono'>{e['text1_sample']}</td><td class='mono'>{e['text2_sample']}</td></tr>"
        for e in examples)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>V3 Stage 4 — Pair-Level Model</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85rem; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
th {{ background: #f0f0f0; }}
td:first-child, th:first-child {{ text-align: left; }}
.muted {{ color: #666; font-size: 0.9rem; }}
.mono {{ font-family: monospace; font-size: 0.75rem; max-width: 260px; text-align: left; }}
h2 {{ border-bottom: 2px solid #ddd; padding-bottom: 4px; margin-top: 2.5rem; }}
.callout {{ background: #fff8e1; border-left: 4px solid #f5a623; padding: 0.8rem 1rem; margin: 1rem 0; }}
</style></head><body>
<p><a href="index.html">&larr; EDA</a> | <a href="stage2.html">Stage 2</a> | <a href="stage3.html">Stage 3</a> | Stage 4</p>
<h1>V3 Stage 4 — Pair-Level Model (the primary detector)</h1>
<p class="muted">V3_PLAN.md &sect;7: "Features from &sect;4.4, target = seed-derived same-operator pairs
... This is where 0.90 lives." Kumar et al. 2017: 0.68 AUC per-account vs. <b>0.91 AUC per-pair</b>, same
features — this stage is the actual point of the V3 pivot away from account-level modeling (Stage 3).</p>

<div class="callout"><b>Anti-circularity design.</b> Stage 3 needed a same-day correction because
account-level labels and features were aggregated over the same rows (see V3_PLAN.md &sect;10.4). Here, the
8 pairwise metrics (B1&ndash;B8) are split into two <b>structurally independent views</b>: View A
(behavioral/structural — co-appearance, arrival timing, activity correlation) and View B
(content/identity — text-template sharing, stylometric similarity, registration-cohort adjacency, shared
domains). The "same-operator" label is built <b>only from View B</b>. The primary model is trained
<b>only on View A</b> to predict that label — a non-circular test of whether behavioral pattern alone,
with zero access to text or identity information, predicts identity-confirmed same-operator status.</p>
<p><b>B6 (reply reciprocity) was dropped, not approximated</b> — the raw per-comment <code>parent_id</code>
needed to know which specific comment a reply targets was never persisted past collection time (only the
derived <code>parent_is_post</code> boolean survives in every table checked). View A is B1+B2+B7.</div>

<h2>Scale</h2>
<table>
<tr><th>quantity</th><th>value</th></tr>
<tr><td>threads (T)</td><td>{T:,}</td></tr>
<tr><td>commenter-thread edges</td><td>{n_edges:,}</td></tr>
<tr><td>candidate pairs (n_coappear&ge;{N_COAPPEAR_MIN})</td><td>{len(pool):,}</td></tr>
<tr><td>same-operator label positives</td><td>{int(label.sum()):,} ({label.mean()*100:.3f}%)</td></tr>
<tr><td>B8 domain-share feature coverage</td><td>{b8_coverage*100:.2f}% of candidate pool</td></tr>
<tr><td>B2 corpus-derived "tight" arrival gap (p5)</td><td>{gap_p5:.0f}s</td></tr>
<tr><td>pairs dropped by account-disjoint split (straddling accounts)</td><td>{dropped:,}</td></tr>
</table>

<h2>Null model: an ascertainment-bias finding, not a calibration pass/fail</h2>
<div class="callout">{sanity['note']}</div>
<p><b>Exact-simulator result</b> ({sanity['n_subs_checked']} subs, {sanity['n_sim_pairs']:,} simulated
co-occurring pairs, data generated under the TRUE null by construction): mean p-value =
{f"{sanity['mean_pvalue']:.4f}" if sanity['mean_pvalue'] is not None else 'n/a'}, median =
{f"{sanity['median_pvalue']:.4f}" if sanity['median_pvalue'] is not None else 'n/a'}
(a true null should show ~0.5 / ~0.5). {f"{sanity['frac_below_05']*100:.1f}%" if sanity['frac_below_05'] is not None else 'n/a'}
of even these null-generated pairs would cross a naive p&lt;0.05 threshold. <b>Because of this, B1's
z-score/p-value is used only as a continuous ranking feature throughout this pipeline — no binary
significance gate is computed or used anywhere.</b> Where a binary "unusually high co-appearance" split was
needed (the View-B reverse-check target below), a top-percentile-of-z cutoff <em>within</em> the
already-co-occurring candidate pool is used instead — a relative comparison, not a contaminated absolute
claim.</p>

<h2>Label construction (View B only)</h2>
<p class="muted">Thresholds are this pool's own percentiles, not imported constants. <b>Combination rule is
OR, not AND</b> — first cut required text-template overlap as a mandatory gate and produced ZERO label
positives (direct text overlap exists in only 300/275,643 = 0.11% of pairs — rare, as expected — gating
everything behind it was too strict; see construct_label()'s docstring). Corrected: text-template overlap
&ge; {label_meta['b3_thresh']:.2f} ({label_meta['n_from_textdup']} pairs) <b>OR</b> (stylometric similarity
&ge; p99 <b>AND</b> cohort adjacency &ge; p98 both independently, {label_meta['n_from_style_and_cohort']}
pairs). Stylometric similarity (char n-gram TF-IDF cosine) has weak absolute discrimination here — mean
0.76 across ALL candidate pairs, common short-comment n-grams dominate at max_features=1200 — so it is used
only as a relative top-percentile rank, never an absolute similarity claim. Signal-count distribution:
{label_meta['n_signals_dist']}.</p>
<h3>Spot-check: sampled labeled-positive pairs</h3>
<table>
<tr><th>account 1</th><th>account 2</th><th>B3</th><th>B4</th><th>B5</th><th>text sample 1</th><th>text sample 2</th></tr>
{example_rows}
</table>

<h2>Primary result: View A &rarr; View-B label (account-disjoint holdout)</h2>
<table>
<tr><th>model</th><th>AUC</th><th>avg precision</th><th>recall@1%FPR</th><th>pos/total (test)</th></tr>
<tr><td><b>View A only (primary, non-circular)</b></td>{fmt(va)}</tr>
<tr><td>View B only (reverse check, predicting View-A-derived null-significance label)</td>{fmt(vb)}</tr>
<tr><td>Combined A+B (context only — most circular, do not cite as primary)</td>{fmt(comb)}</tr>
</table>
<p class="muted">Mandatory baseline (raw co-appearance count alone, no null-adjustment, no other features):
AUC = {f'{baseline_auc:.3f}' if baseline_auc is not None else 'n/a'}. Permutation floor (View-A model,
label shuffled): AUC = {f'{perm_auc:.3f}' if perm_auc is not None else 'n/a'}. The primary View-A number
must clearly beat both to mean anything beyond "these accounts are simply active in the same threads a
lot."</p>
<p class="muted">Reference points: Kumar et al. 2017 pair-level AUC = {KUMAR_PAIR}; Schoch et al. recall at
1% FPR (&ge;10-repetition coordination) = {SCHOCH_RECALL_AT_1PCT_FPR*100:.0f}%. Compared honestly, not
force-fit — see the primary AUC/recall above.</p>

<h2>TreeSHAP — View A primary model (top features)</h2>
<table><tr><th>feature</th><th>mean |SHAP|</th></tr>{shap_rows}</table>

</body></html>"""
    with open(OUT_HTML, 'w') as f:
        f.write(html)


if __name__ == '__main__':
    main()
