#!/usr/bin/env python3
"""V3 Stage 5a: generate bot-candidate accounts from the local corpus for manual/LLM
labeling -- the deterministic half of building the ground-truth set used by Stage 5b
(composite, Method 1) and Stage 6 (XGBoost, Method 2).

Motivated by two external mod/user discussions on spotting bot accounts (see
docs/v3-research/bot-detection-literature.md for the full tip extraction):
  - r/ModSupport "Tips on spotting bot/scam accounts" (2026-11)
  - r/TheGirlSurvivalGuide "Recognizing bot comments" (2026-11)
Both independently converge on the same strongest tell available in this corpus:
an account posting near-identical text across multiple threads/subreddits. That
tell is what this script screens for, over the full local corpus -- no Arctic
Shift API calls needed, since scripts/v3_collect.py already stored full comment
`body` text in commenters_dedup for every collected comment (1.6M rows, 348K
distinct authors).

A naive random sample of accounts has a ~5.6% confirmed-bot hit rate on manual
read (measured, not assumed -- see the two earlier rounds of 266 and then this
screen's candidates). Screening by cross-post text duplication first raised
that to 16.1% (62/385) on manual read, a 2.9x improvement in reviewer-time
efficiency for the same labeling budget.

Output: output/v3/bot_candidates.csv (author, n_dup_bodies, total_repeat_comments,
max_subs_for_one_body), for a human/LLM reviewer to read real comment text and
classify as CLEAR_BOT / SUSPICIOUS / CLEAN. The actual labels produced by that
review are checked in as output/v3/{confirmed_bots,suspicious_accounts,
clean_accounts}.json -- this script only rebuilds the candidate pool, it does
not re-label anything.
"""
import os

import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
OUT_CSV = os.path.join(ROOT, 'output', 'v3', 'bot_candidates.csv')

MIN_BODY_LEN = 30  # below this, "duplicate" is just common short reactions ("lol", "same")
EXCLUDE_PATTERNS = ['%ModTeam%', '%AutoModerator%', '%autopostremover%', '%bot%']


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    exclude_sql = ' AND '.join(f"author NOT ILIKE '{p}'" for p in EXCLUDE_PATTERNS)
    df = con.execute(f'''
        WITH dup AS (
            SELECT author, body, count(*) AS n_repeats, count(DISTINCT post_id) AS n_distinct_posts,
                   count(DISTINCT sub) AS n_distinct_subs
            FROM commenters_dedup
            WHERE length(body) >= {MIN_BODY_LEN}
            GROUP BY author, body
            HAVING count(DISTINCT post_id) >= 2
        )
        SELECT author, count(*) AS n_dup_bodies, sum(n_repeats) AS total_repeat_comments,
               max(n_distinct_subs) AS max_subs_for_one_body
        FROM dup
        WHERE {exclude_sql}
        GROUP BY author
        ORDER BY total_repeat_comments DESC
    ''').fetchdf()
    print(f'{len(df)} candidate accounts with cross-post duplicate text (>={MIN_BODY_LEN} chars, official-bot names excluded)')
    df.to_csv(OUT_CSV, index=False)
    print(f'wrote {OUT_CSV}')


if __name__ == '__main__':
    main()
