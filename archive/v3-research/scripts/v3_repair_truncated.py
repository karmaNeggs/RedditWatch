#!/usr/bin/env python3
"""Repair pass for the V3 full collection: some posts' comment threads got
silently truncated when a comments/search page exhausted all 5 retries mid-
pagination (fetch_all_comments treats a failed page the same as "no more
pages"). Detected post-hoc by comparing n_comments_observed against
num_comments_reported (Arctic Shift's T+36h snapshot count) -- a large
shortfall is the truncation signature, since removed/deleted comments are
still returned by comments/search (with removal metadata), so a shortfall
this large isn't explained by legitimate removal.

Re-fetches only the flagged posts' comment threads and patches the affected
cell files in place (read -> patch -> atomic rewrite), reusing the exact
same derivation logic as the original collector (comment_derived_fields_and_rows)
so a repaired post is indistinguishable from one collected cleanly the first time."""
import argparse
import glob
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_collect import (fetch_all_comments, comment_derived_fields_and_rows,
                         write_ndjson_zst, post_path, commenter_path, Stats)

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


def find_candidates(min_reported, max_ratio):
    post_files = sorted(glob.glob(os.path.join(RAW, 'posts', '*.ndjson.zst')))
    by_cell = defaultdict(list)  # (sub, month) -> [post_id, ...]
    n_total = 0
    for pf in post_files:
        for row in iter_rows(pf):
            n_total += 1
            rep = row.get('num_comments_reported') or 0
            obs = row.get('n_comments_observed') or 0
            if rep >= min_reported and obs < max_ratio * rep:
                by_cell[(row['sub'], row['month'])].append(row['post_id'])
    return by_cell, n_total


def repair_cell(sub, month_label, target_ids, stats, workers):
    pp = post_path(sub, month_label, RAW)
    cp = commenter_path(sub, month_label, RAW)
    posts = list(iter_rows(pp))
    commenters = list(iter_rows(cp))
    target_set = set(target_ids)

    role_by_id = {p['post_id']: p.get('role', 'top') for p in posts if p['post_id'] in target_set}

    results = {}

    def refetch(pid):
        comments = fetch_all_comments(pid, f't3_{pid}', stats)
        fields, rows = comment_derived_fields_and_rows(
            comments, sub, month_label, pid, role_by_id.get(pid, 'top'))
        return pid, fields, rows

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(refetch, pid) for pid in target_ids]
        for fut in as_completed(futs):
            pid, fields, rows = fut.result()
            results[pid] = (fields, rows)

    before_after = []
    for p in posts:
        pid = p['post_id']
        if pid in results:
            fields, _ = results[pid]
            before_after.append((pid, p.get('n_comments_observed'), fields['n_comments_observed']))
            p.update(fields)

    commenters = [c for c in commenters if c['post_id'] not in target_set]
    for pid, (_, rows) in results.items():
        commenters.extend(rows)

    write_ndjson_zst(pp, posts)
    write_ndjson_zst(cp, commenters)
    return before_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-reported', type=int, default=10)
    ap.add_argument('--max-ratio', type=float, default=0.5,
                     help='flag if n_comments_observed < max_ratio * num_comments_reported')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    print('Scanning for truncation candidates...', flush=True)
    by_cell, n_total = find_candidates(args.min_reported, args.max_ratio)
    n_candidates = sum(len(v) for v in by_cell.values())
    print(f'{n_candidates} candidate posts across {len(by_cell)} cells (of {n_total} total posts, '
          f'{100*n_candidates/n_total:.3f}%)', flush=True)

    if args.dry_run:
        for (sub, month), ids in sorted(by_cell.items()):
            print(f'  {sub}/{month}: {len(ids)} posts')
        return

    stats = Stats()
    total_improved, total_still_bad, total_worse = 0, 0, 0
    for i, ((sub, month), ids) in enumerate(sorted(by_cell.items()), 1):
        t0 = time.time()
        ba = repair_cell(sub, month, ids, stats, args.workers)
        improved = sum(1 for (_, before, after) in ba if after > (before or 0))
        still_bad = sum(1 for (_, before, after) in ba if after <= (before or 0))
        total_improved += improved
        total_still_bad += still_bad
        print(f'[{i}/{len(by_cell)}] {sub}/{month}: {len(ids)} posts, '
              f'{improved} improved, {still_bad} unchanged/worse, {time.time()-t0:.0f}s', flush=True)

    print(f'\nDONE. improved={total_improved} unchanged_or_worse={total_still_bad}')
    print(stats.line(len(by_cell)))


if __name__ == '__main__':
    main()
