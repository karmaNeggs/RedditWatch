#!/usr/bin/env python3
"""V3 bot-marker composite (V3_PLAN.md Sec 10.4): does combining our
theory-motivated markers concentrate suspicious signal better than any one
alone? No label exists yet (that's Stage 3/4's job) -- this is an
UNSUPERVISED separation check, not a validated detector. Treat it as a
prioritization tool for which accounts to look at first, and as prep work
for Stage 3, not as a bot score.

Marker list, from the session's theoretical discussion of what bot behavior
should look like, each mapped to an existing account_features column and a
"higher = more suspicious" direction:
  removal_rate              content getting removed/hidden
  deleted_later_rate        companion signal (alive at capture, gone later)
  thin_history_score        inverse of n_comments_sample -- "not showing any
                             history" -- percentile rank, flipped so low
                             activity -> high suspicion
  karma_extremeness         |mean_comment_score - population median| / MAD --
                             BOTH extreme-low (ragebait, downvoted) and
                             extreme-high (appreciation-bot, upvoted) are
                             "disproportionate", per the session's discussion
                             of two different bot archetypes with opposite
                             valence but the same underlying pattern
  karma_per_post_extremeness   same idea, for authored posts (n_posts_sample>0
                             only, ~13% of accounts -- most will be null)
  reception_spread           best-sub-minus-worst-sub reception gap --
                             "extreme vote behaviour on diff subs"

Method: each marker -> percentile rank (0-100, higher=more suspicious) so
they're comparable without hand-tuned weights. Composite = mean of whichever
marker percentiles are non-null for that account (needs >=3 to get a score,
else null -- most accounts lack karma_per_post_extremeness entirely).
Reuses the Stage-1 screening machinery unchanged (point-mass hurdle,
seeded n=20,000 subsample, separation-score guard) so the composite gets
exactly the same scrutiny as everything upstream, not a looser bar."""
import os
import sys

import duckdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_stage1_univariate import screen_feature, verdict_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')

MARKERS = ['removal_rate', 'deleted_later_rate', 'thin_history_score',
           'karma_extremeness', 'karma_per_post_extremeness', 'reception_spread']


def build_markers(con):
    print('Computing marker percentile ranks + composite...', flush=True)
    con.execute("""
        CREATE OR REPLACE TABLE account_botmarkers AS
        WITH base AS (
            SELECT *,
                -n_comments_sample AS thin_history_raw,
                abs(mean_comment_score - (SELECT median(mean_comment_score) FROM account_features))
                    / NULLIF((SELECT median(abs(mean_comment_score - (SELECT median(mean_comment_score) FROM account_features))) FROM account_features), 0)
                    AS karma_extremeness_raw,
                CASE WHEN mean_post_score IS NOT NULL THEN
                    abs(mean_post_score - (SELECT median(mean_post_score) FROM account_features WHERE mean_post_score IS NOT NULL))
                    / NULLIF((SELECT median(abs(mean_post_score - (SELECT median(mean_post_score) FROM account_features WHERE mean_post_score IS NOT NULL))) FROM account_features WHERE mean_post_score IS NOT NULL), 0)
                END AS karma_per_post_extremeness_raw
            FROM account_features
        )
        SELECT
            author,
            100 * percent_rank() OVER (ORDER BY removal_rate) AS removal_rate_pctl,
            100 * percent_rank() OVER (ORDER BY deleted_later_rate) AS deleted_later_rate_pctl,
            100 * percent_rank() OVER (ORDER BY thin_history_raw) AS thin_history_score,
            100 * percent_rank() OVER (ORDER BY karma_extremeness_raw) AS karma_extremeness,
            CASE WHEN karma_per_post_extremeness_raw IS NOT NULL
                 THEN 100 * percent_rank() OVER (PARTITION BY (karma_per_post_extremeness_raw IS NOT NULL) ORDER BY karma_per_post_extremeness_raw)
            END AS karma_per_post_extremeness,
            100 * percent_rank() OVER (ORDER BY reception_spread) AS reception_spread_pctl
        FROM base
    """)
    # composite: mean of non-null marker percentiles, require >=3
    cols_pctl = ['removal_rate_pctl', 'deleted_later_rate_pctl', 'thin_history_score',
                 'karma_extremeness', 'karma_per_post_extremeness', 'reception_spread_pctl']
    coalesce_list = ', '.join(cols_pctl)
    con.execute(f"""
        CREATE OR REPLACE TABLE account_botmarkers AS
        SELECT *,
            list_avg(list_filter([{coalesce_list}], x -> x IS NOT NULL)) AS botmarker_composite,
            len(list_filter([{coalesce_list}], x -> x IS NOT NULL)) AS n_markers_available
        FROM account_botmarkers
    """)
    con.execute("""
        CREATE OR REPLACE TABLE account_features AS
        SELECT af.*, bm.removal_rate_pctl, bm.deleted_later_rate_pctl, bm.thin_history_score,
               bm.karma_extremeness, bm.karma_per_post_extremeness, bm.reception_spread_pctl,
               CASE WHEN bm.n_markers_available >= 3 THEN bm.botmarker_composite END AS botmarker_composite,
               bm.n_markers_available
        FROM account_features af JOIN account_botmarkers bm USING (author)
    """)


def main():
    con = duckdb.connect(DB_PATH)
    build_markers(con)

    print('\n=== MARKER SEPARATION (unsupervised -- no labels yet, see docstring) ===\n')
    print(f'{"marker":<28} {"n":>8} {"dip_p":>8} {"k":>3} {"minor%":>8} {"sep":>8}  verdict')
    rows = []
    for feature in ['removal_rate', 'deleted_later_rate', 'thin_history_score',
                     'karma_extremeness', 'karma_per_post_extremeness', 'reception_spread',
                     'botmarker_composite']:
        r = screen_feature(con, feature, False, False)
        v = verdict_for(r)
        rows.append((feature, r, v))
        if r.get('skipped'):
            print(f'{feature:<28} {r["n"]:>8}  -- {v}')
            continue
        minority_pct = f'{100*r["minority_mass"]:.1f}%' if r['minority_mass'] is not None else '--'
        sep = f'{r["separation_score"]:.2f}' if r['separation_score'] is not None else '--'
        print(f'{feature:<28} {r["n"]:>8} {r["dip_p"]:>8.4f} {r["bic_k"]:>3} {minority_pct:>8} {sep:>8}  {v}')

    print('\n=== TOP-1%-BY-COMPOSITE vs TOP-1%-BY-EACH-MARKER-ALONE: profile comparison ===')
    print('(does the composite\'s top tail look more concentrated than any single marker\'s?)\n')
    profile_cols = 'removal_rate, controversiality_rate, deleted_later_rate, n_comments_sample'
    for by in ['botmarker_composite', 'removal_rate', 'reception_spread', 'karma_extremeness']:
        df = con.execute(f"""
            SELECT count(*) n,
                   avg(removal_rate) removal_rate, avg(controversiality_rate) controversiality,
                   avg(deleted_later_rate) deleted_later, avg(n_comments_sample) mean_n_comments
            FROM account_features
            WHERE "{by}" >= (SELECT quantile_cont("{by}", 0.99) FROM account_features WHERE "{by}" IS NOT NULL)
        """).fetchdf()
        print(f'top 1% by {by}:')
        print('  ' + df.to_string(index=False).replace('\n', '\n  '))

    con.close()
    print('\nDONE.')


if __name__ == '__main__':
    main()
