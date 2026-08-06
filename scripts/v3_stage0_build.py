#!/usr/bin/env python3
"""V3 Stage 0: analysis layer + cleaning (V3_PLAN.md Sec 2, Sec 7).

Builds a persistent DuckDB file over the raw per-cell zstd-NDJSON, adding the
Stage-0 cleaning columns without discarding any rows (nothing is dropped from
the base tables -- "clean" views filter on top, so accounting stays possible):

  - is_confirmed_automation_seed: the plan's "Confirmed automation" seed
    channel (Sec 6) -- self-declared bot phrases, AutoModerator / *bot author
    names, distinguished == 'moderator'. Benign seed set, not the modeling
    target.
  - is_deleted_author: author in (NULL, '[deleted]').
  - account_ordinal: base36-decoded author_fullname[3:] (Sec 3, A1) -- a raw,
    monotonic-with-creation-order integer. NOT yet calibrated to a real date
    (that's the optional Sec 10 item 6); usable directly for percentile/
    relative-age comparisons.

Also resolves the ~30% intra-cell duplicate comment_id issue found in QC (a
comment that's both a first-10-arrival and a top-10-by-score is stored twice,
once per tag, by design) via a commenters_dedup view, so engagement-counting
analyses don't double-count while role-tagged analyses (P8 etc.) still can via
the raw table.

Re-run whenever data/v3/raw gains new cells (monthly tracker) -- CREATE OR
REPLACE TABLE means this script is idempotent, not incremental; fine at this
data volume (rebuild takes seconds)."""
import os
import time

import duckdb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'v3', 'raw')
ANALYSIS_DIR = os.path.join(ROOT, 'data', 'v3', 'analysis')
DB_PATH = os.path.join(ANALYSIS_DIR, 'v3.duckdb')

os.makedirs(ANALYSIS_DIR, exist_ok=True)

BOT_PHRASES = ['i am a bot', 'beep boop', 'performed automatically',
               'this action was performed automatically']


def base36_decode(s):
    if s is None:
        return None
    try:
        return int(s, 36)
    except (ValueError, TypeError):
        return None


def main():
    t0 = time.time()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = duckdb.connect(DB_PATH)
    con.create_function('base36_decode', base36_decode, ['VARCHAR'], 'BIGINT')

    posts_glob = os.path.join(RAW, 'posts', '*.ndjson.zst')
    comm_glob = os.path.join(RAW, 'commenters', '*.ndjson.zst')

    print('Building posts table...', flush=True)
    con.execute(f"""
        CREATE OR REPLACE TABLE posts AS
        SELECT *,
            COALESCE(author = 'AutoModerator' OR author ILIKE '%bot', FALSE) AS is_confirmed_automation_seed,
            COALESCE(author IS NULL OR author = '[deleted]', FALSE) AS is_deleted_author,
            CASE WHEN author_fullname LIKE 't2\\_%' ESCAPE '\\'
                 THEN base36_decode(substr(author_fullname, 4)) END AS account_ordinal
        FROM read_json_auto('{posts_glob}', union_by_name=true)
    """)

    print('Building commenters table...', flush=True)
    bot_body_clause = ' OR '.join(f"lower(body) LIKE '%{p}%'" for p in BOT_PHRASES)
    con.execute(f"""
        CREATE OR REPLACE TABLE commenters AS
        SELECT *,
            COALESCE(distinguished = 'moderator'
             OR author = 'AutoModerator'
             OR author ILIKE '%bot'
             OR ({bot_body_clause}), FALSE) AS is_confirmed_automation_seed,
            COALESCE(author IS NULL OR author = '[deleted]', FALSE) AS is_deleted_author,
            CASE WHEN author_fullname LIKE 't2\\_%' ESCAPE '\\'
                 THEN base36_decode(substr(author_fullname, 4)) END AS account_ordinal
        FROM read_json_auto('{comm_glob}', union_by_name=true)
    """)

    print('Building dedup / clean views...', flush=True)
    con.execute("""
        CREATE OR REPLACE VIEW commenter_tags AS
        SELECT comment_id, list(DISTINCT commenter_tag) AS tags, count(*) AS tag_count
        FROM commenters
        GROUP BY comment_id
    """)
    con.execute("""
        CREATE OR REPLACE VIEW commenters_dedup AS
        SELECT c.* EXCLUDE (commenter_tag), t.tags, t.tag_count
        FROM commenters c
        JOIN commenter_tags t USING (comment_id)
        QUALIFY ROW_NUMBER() OVER (PARTITION BY c.comment_id ORDER BY c.commenter_tag) = 1
    """)
    con.execute("""
        CREATE OR REPLACE VIEW commenters_clean AS
        SELECT * FROM commenters_dedup
        WHERE NOT is_confirmed_automation_seed AND NOT is_deleted_author
    """)
    con.execute("""
        CREATE OR REPLACE VIEW posts_clean AS
        SELECT * FROM posts
        WHERE NOT is_confirmed_automation_seed AND NOT is_deleted_author
    """)

    # ---- validation report ----
    print('\n=== STAGE 0 BUILD REPORT ===')
    n_posts = con.execute('SELECT count(*) FROM posts').fetchone()[0]
    n_posts_auto = con.execute('SELECT count(*) FROM posts WHERE is_confirmed_automation_seed').fetchone()[0]
    n_posts_del = con.execute('SELECT count(*) FROM posts WHERE is_deleted_author').fetchone()[0]
    n_posts_ord = con.execute('SELECT count(*) FROM posts WHERE account_ordinal IS NOT NULL').fetchone()[0]
    print(f'posts: {n_posts} total | automation_seed={n_posts_auto} ({100*n_posts_auto/n_posts:.2f}%) | '
          f'deleted_author={n_posts_del} ({100*n_posts_del/n_posts:.2f}%) | '
          f'account_ordinal resolved={n_posts_ord} ({100*n_posts_ord/n_posts:.1f}%)')

    n_comm = con.execute('SELECT count(*) FROM commenters').fetchone()[0]
    n_comm_dedup = con.execute('SELECT count(*) FROM commenters_dedup').fetchone()[0]
    n_comm_auto = con.execute('SELECT count(*) FROM commenters_dedup WHERE is_confirmed_automation_seed').fetchone()[0]
    n_comm_del = con.execute('SELECT count(*) FROM commenters_dedup WHERE is_deleted_author').fetchone()[0]
    n_comm_clean = con.execute('SELECT count(*) FROM commenters_clean').fetchone()[0]
    n_comm_ord = con.execute('SELECT count(*) FROM commenters_dedup WHERE account_ordinal IS NOT NULL').fetchone()[0]
    print(f'commenters: {n_comm} raw rows -> {n_comm_dedup} deduped comments '
          f'({100*(n_comm-n_comm_dedup)/n_comm:.1f}% were first10/top10 tag duplicates)')
    print(f'  of {n_comm_dedup} deduped: automation_seed={n_comm_auto} ({100*n_comm_auto/n_comm_dedup:.2f}%) | '
          f'deleted_author={n_comm_del} ({100*n_comm_del/n_comm_dedup:.2f}%) | '
          f'account_ordinal resolved={n_comm_ord} ({100*n_comm_ord/n_comm_dedup:.1f}%)')
    print(f'  commenters_clean (modeling set): {n_comm_clean} rows '
          f'({100*n_comm_clean/n_comm_dedup:.1f}% of deduped)')

    print('\nsanity: distinct accounts in commenters_clean:',
          con.execute('SELECT count(DISTINCT author) FROM commenters_clean').fetchone()[0])
    print('sanity: tag_count distribution in commenters_dedup:',
          dict(con.execute('SELECT tag_count, count(*) FROM commenters_dedup GROUP BY tag_count ORDER BY 1').fetchall()))

    con.close()
    print(f'\nDONE in {time.time()-t0:.1f}s. DB -> {DB_PATH}')


if __name__ == '__main__':
    main()
