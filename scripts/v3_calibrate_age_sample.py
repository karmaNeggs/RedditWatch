#!/usr/bin/env python3
"""V3 base36 age calibration, sampling step (V3_PLAN.md Sec 10 item 6a).

account_ordinal (base36-decoded author_fullname) is monotonic with account
creation order (Sec 3, AUC 0.986 pre-2023 vs post-2025), but uncalibrated --
it's an ordinal, not a date. This pulls real created_utc for a stratified
sample of accounts via live Reddit (scripts/reddit_auth.py, OAuth, ~53
req/min) so a later step can fit ordinal -> creation-date.

Stratified by percentile of the OBSERVED account_ordinal distribution (not
uniform over the raw integer range, which would be dominated by whichever
band happens to be numerically wide) so the calibration curve has real
support across the whole span our corpus actually touches.

Checkpointed: writes to OUT incrementally, one line per resolved account,
so a killed run resumes by just skipping already-covered accounts."""
import json
import os
import sys
import time

import duckdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reddit_auth import get_json, print_auth_status, USING_OAUTH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
OUT = os.path.join(ROOT, 'data', 'v3', 'analysis', 'age_calibration_sample.ndjson')

N_STRATA = 25
PER_STRATUM = 70  # ~1750 total


def already_done():
    seen = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                if line.strip():
                    try:
                        seen.add(json.loads(line)['author'])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return seen


def main():
    print_auth_status()
    con = duckdb.connect(DB_PATH, read_only=True)

    df = con.execute("""
        SELECT author, account_ordinal FROM account_features
        WHERE account_ordinal IS NOT NULL
    """).fetchdf()
    con.close()

    df = df.sort_values('account_ordinal').reset_index(drop=True)
    n = len(df)
    strata_edges = np.linspace(0, n, N_STRATA + 1).astype(int)

    done = already_done()
    print(f'{n} candidate accounts, {N_STRATA} strata x {PER_STRATUM} = target ~{N_STRATA*PER_STRATUM}. '
          f'{len(done)} already resolved.', flush=True)

    rng = np.random.RandomState(7)
    targets = []
    for i in range(N_STRATA):
        lo, hi = strata_edges[i], strata_edges[i + 1]
        stratum = df.iloc[lo:hi]
        pick = stratum.sample(min(PER_STRATUM, len(stratum)), random_state=rng)
        targets.extend(pick.to_dict('records'))

    targets = [t for t in targets if t['author'] not in done]
    print(f'{len(targets)} lookups remaining this run.', flush=True)

    base = 'https://oauth.reddit.com' if USING_OAUTH else 'https://www.reddit.com'
    t0 = time.time()
    n_ok, n_gone, n_err = 0, 0, 0
    with open(OUT, 'a') as f:
        for i, t in enumerate(targets, 1):
            author = t['author']
            d = get_json(f'{base}/user/{author}/about.json')
            row = {'author': author, 'account_ordinal': t['account_ordinal']}
            if d and 'data' in d and d['data'].get('created_utc'):
                row['created_utc'] = d['data']['created_utc']
                row['status'] = 'ok'
                n_ok += 1
            elif d is None:
                row['status'] = 'gone'  # 404 -- suspended/deleted/shadowbanned
                n_gone += 1
            else:
                row['status'] = 'error'
                n_err += 1
            f.write(json.dumps(row) + '\n')
            f.flush()
            if i % 100 == 0:
                elapsed = time.time() - t0
                print(f'  [{elapsed/60:.1f}m] {i}/{len(targets)}  ok={n_ok} gone={n_gone} err={n_err}  '
                      f'({i/elapsed:.2f}/s)', flush=True)

    print(f'\nDONE. ok={n_ok} gone={n_gone} err={n_err}  -> {OUT}')


if __name__ == '__main__':
    main()
