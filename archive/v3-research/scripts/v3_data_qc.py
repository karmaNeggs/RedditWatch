#!/usr/bin/env python3
"""V3 collection QC: streaming integrity + sanity checks over data/v3/raw/.
Doesn't hold the corpus in memory -- one file at a time, accumulator counters
only. Re-runnable after any future incremental collection (monthly tracker).
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, 'data', 'v3', 'raw')


def iter_rows(path):
    dctx = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        with dctx.stream_reader(f) as reader:
            buf = b''
            chunk = reader.read(1 << 20)
            while chunk:
                buf += chunk
                lines = buf.split(b'\n')
                buf = lines.pop()
                for line in lines:
                    if line:
                        yield json.loads(line)
                chunk = reader.read(1 << 20)
            if buf.strip():
                yield json.loads(buf)


def pct(n, d):
    return f'{100*n/d:.1f}%' if d else 'n/a'


def main():
    post_files = sorted(glob.glob(os.path.join(RAW, 'posts', '*.ndjson.zst')))
    comm_files = sorted(glob.glob(os.path.join(RAW, 'commenters', '*.ndjson.zst')))
    print(f'{len(post_files)} post files, {len(comm_files)} commenter files\n')

    # ---- posts ----
    n_posts = 0
    role_counts = Counter()
    tier_counts = Counter()
    sub_post_counts = Counter()
    month_post_counts = Counter()
    null_fields = Counter()
    dup_ids_within_cell = 0
    meta_removal_present = 0
    meta_was_deleted_later_present = 0
    author_fullname_present = 0
    score_out_of_range = 0
    negative_gap = 0
    n_comments_reported_sum = 0
    n_comments_observed_sum = 0
    zero_comment_reported_but_observed_gt0 = 0
    reply_recip_present = 0
    max_depth_present = 0
    empty_cells = []

    FIELDS_CHECK = ['score', 'upvote_ratio', 'created_utc', 'author_fullname',
                     'n_comments_observed', 'comment_score_gini', 'max_depth',
                     'reply_reciprocity']

    for pf in post_files:
        basename = os.path.basename(pf).replace('.ndjson.zst', '')
        sub, month = basename.rsplit('__', 1)
        seen_ids = set()
        cell_n = 0
        for row in iter_rows(pf):
            n_posts += 1
            cell_n += 1
            role_counts[row.get('role')] += 1
            tier_counts[row.get('incentive_tier')] += 1
            sub_post_counts[sub] += 1
            month_post_counts[month] += 1
            pid = row.get('post_id')
            if pid in seen_ids:
                dup_ids_within_cell += 1
            seen_ids.add(pid)
            for fld in FIELDS_CHECK:
                if row.get(fld) is None:
                    null_fields[fld] += 1
            if row.get('meta_removal_type') is not None:
                meta_removal_present += 1
            if row.get('meta_was_deleted_later') is not None:
                meta_was_deleted_later_present += 1
            if row.get('author_fullname'):
                author_fullname_present += 1
            r = row.get('upvote_ratio')
            if r is not None and not (0.0 <= r <= 1.0):
                score_out_of_range += 1
            gaps = row.get('first10_arrival_gaps') or []
            if any(g < 0 for g in gaps):
                negative_gap += 1
            nc_rep = row.get('num_comments_reported') or 0
            nc_obs = row.get('n_comments_observed') or 0
            n_comments_reported_sum += nc_rep
            n_comments_observed_sum += nc_obs
            if nc_rep == 0 and nc_obs > 0:
                zero_comment_reported_but_observed_gt0 += 1
            if row.get('reply_reciprocity') is not None:
                reply_recip_present += 1
            if row.get('max_depth') is not None:
                max_depth_present += 1
        if cell_n == 0:
            empty_cells.append(basename)

    print('=== POSTS ===')
    print(f'total post rows: {n_posts}')
    print(f'role split: {dict(role_counts)}')
    print(f'tier split: {dict(tier_counts)}')
    print(f'duplicate post_id within a cell: {dup_ids_within_cell}')
    print(f'empty cells (0 posts): {len(empty_cells)}')
    if empty_cells:
        print(f'  sample: {empty_cells[:15]}')
    print(f'null rates: ' + ', '.join(f'{k}={pct(v, n_posts)}' for k, v in null_fields.items()))
    print(f'meta_removal_type present: {pct(meta_removal_present, n_posts)}')
    print(f'meta_was_deleted_later present (non-null): {pct(meta_was_deleted_later_present, n_posts)}')
    print(f'author_fullname present: {pct(author_fullname_present, n_posts)}')
    print(f'upvote_ratio out of [0,1]: {score_out_of_range}')
    print(f'negative first10 arrival gaps: {negative_gap}')
    print(f'reply_reciprocity non-null: {pct(reply_recip_present, n_posts)}')
    print(f'max_depth non-null: {pct(max_depth_present, n_posts)}')
    print(f'sum num_comments_reported: {n_comments_reported_sum}')
    print(f'sum n_comments_observed:   {n_comments_observed_sum}  '
          f'(ratio observed/reported = {n_comments_observed_sum/max(1,n_comments_reported_sum):.3f})')
    print(f'posts with reported=0 but observed>0 (should be ~0, API lag): {zero_comment_reported_but_observed_gt0}')

    # sub-level and month-level post-count spread (sanity: no sub/month wildly short)
    print('\nposts per sub (min/median/max):')
    vals = sorted(sub_post_counts.values())
    print(f'  min={vals[0]} median={vals[len(vals)//2]} max={vals[-1]}  '
          f'(45 subs x 24mo, full cell=120 -> max possible 2880)')
    low_subs = [(s, c) for s, c in sub_post_counts.items() if c < 500]
    print(f'  subs with <500 total post rows across 24mo: {low_subs}')

    print('\nposts per month (min/median/max):')
    vals = sorted(month_post_counts.values())
    print(f'  min={vals[0]} median={vals[len(vals)//2]} max={vals[-1]}')

    # ---- known edge-case subs ----
    print('\n=== KNOWN EDGE CASES ===')
    for sub in ['DesiVideoMemes', 'IndiaTrending', 'unitedstatesofindia']:
        by_month = {}
        for pf in post_files:
            basename = os.path.basename(pf).replace('.ndjson.zst', '')
            s, month = basename.rsplit('__', 1)
            if s != sub:
                continue
            n = sum(1 for _ in iter_rows(pf))
            by_month[month] = n
        months_sorted = sorted(by_month)
        print(f'{sub}: {[by_month[m] for m in months_sorted[-6:]]} (last 6 months, {months_sorted[-6:]})')

    # ---- commenters ----
    print('\n=== COMMENTERS ===')
    n_comm = 0
    body_present = 0
    body_empty_not_deleted = 0
    tag_counts = Counter()
    author_fullname_present_c = 0
    depth_present = 0
    dup_comment_ids = 0
    score_present = 0
    for cf in comm_files:
        seen = set()
        for row in iter_rows(cf):
            n_comm += 1
            cid = row.get('comment_id')
            if cid in seen:
                dup_comment_ids += 1
            seen.add(cid)
            tag_counts[row.get('commenter_tag')] += 1
            body = row.get('body')
            if body:
                body_present += 1
            elif row.get('author') not in ('[deleted]', None):
                body_empty_not_deleted += 1
            if row.get('author_fullname'):
                author_fullname_present_c += 1
            if row.get('depth') is not None:
                depth_present += 1
            if row.get('score') is not None:
                score_present += 1

    print(f'total commenter rows: {n_comm}')
    print(f'tag split: {dict(tag_counts)}')
    print(f'duplicate comment_id within a cell: {dup_comment_ids}')
    print(f'body text present: {pct(body_present, n_comm)}')
    print(f'body empty but author not [deleted] (unexpected gap): {pct(body_empty_not_deleted, n_comm)}')
    print(f'author_fullname present: {pct(author_fullname_present_c, n_comm)}')
    print(f'depth present: {pct(depth_present, n_comm)}')
    print(f'score present: {pct(score_present, n_comm)}')

    print('\nQC complete.')


if __name__ == '__main__':
    main()
