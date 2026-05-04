"""
Shared Reddit OAuth session + rate limiter for all scripts.

Reads REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT from .env
in the repo root. Falls back to anonymous (User-Agent rotation) if not set.

OAuth gives 60 req/min vs ~10 req/min anonymous.
The global throttle enforces ≤54 req/min so we never hit the ceiling.
"""

import os
import time
import threading
import requests
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── load .env manually (no extra dependency) ──────────────────────────────────
def _load_env():
    env_file = ROOT / '.env'
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, val = line.partition('=')
        os.environ.setdefault(key.strip(), val.strip())

_load_env()

CLIENT_ID     = os.environ.get('REDDIT_CLIENT_ID', '').strip()
CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET', '').strip()
USER_AGENT    = os.environ.get('REDDIT_USER_AGENT', 'RedditBotWatch/1.0').strip()

USING_OAUTH = bool(CLIENT_ID and CLIENT_SECRET)

# ── OAuth token (expires in 1h; auto-refreshed) ───────────────────────────────
_token       = None
_token_expiry = 0
_token_lock  = threading.Lock()

def _get_token():
    global _token, _token_expiry
    with _token_lock:
        if _token and time.time() < _token_expiry - 60:
            return _token
        resp = requests.post(
            'https://www.reddit.com/api/v1/access_token',
            auth=(CLIENT_ID, CLIENT_SECRET),
            data={'grant_type': 'client_credentials'},
            headers={'User-Agent': USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        _token = data['access_token']
        _token_expiry = time.time() + data.get('expires_in', 3600)
        return _token

# ── build session ─────────────────────────────────────────────────────────────
import random

_ANON_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
]

def get_headers():
    """Return request headers — OAuth bearer if configured, else rotating UA."""
    if USING_OAUTH:
        return {
            'Authorization': f'Bearer {_get_token()}',
            'User-Agent': USER_AGENT,
        }
    return {'User-Agent': random.choice(_ANON_AGENTS)}

# ── global rate limiter ───────────────────────────────────────────────────────
# OAuth = 60 req/min; we target 54 (MIN_INTERVAL=1.12s).
# Anonymous = ~10 req/min; we target 8 (MIN_INTERVAL=7.5s).
MIN_INTERVAL = 1.12 if USING_OAUTH else 7.5

_throttle_lock = threading.Lock()
_last_sent     = [0.0]

def throttle():
    """Block until it's safe to fire the next request."""
    with _throttle_lock:
        wait = MIN_INTERVAL - (time.time() - _last_sent[0])
        if wait > 0:
            time.sleep(wait)
        _last_sent[0] = time.time()

# ── single fetch helper used by all scripts ───────────────────────────────────
def get_json(url, retries=6):
    for attempt in range(retries):
        throttle()
        try:
            r = requests.get(url, headers=get_headers(), timeout=20)
            if r.status_code == 429:
                pause = 15 * (2 ** attempt)
                print(f"    Rate limited, waiting {pause}s…")
                time.sleep(pause)
                continue
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(5)
    return None

# ── startup banner ────────────────────────────────────────────────────────────
def print_auth_status():
    if USING_OAUTH:
        print(f"  Auth: OAuth  ({int(60/MIN_INTERVAL)} req/min, user-agent: {USER_AGENT})")
    else:
        print("  Auth: Anonymous (set REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET in .env for 6× faster runs)")
