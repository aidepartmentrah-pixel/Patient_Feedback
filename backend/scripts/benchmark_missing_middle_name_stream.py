#!/usr/bin/env python3
"""
Timing comparison: the old blocking guess-mode search endpoint vs the new
SSE streaming one, against a running stack (dev by default). Quantifies
the perceived-speed win: streaming should deliver its first event in
roughly one candidate's latency (well under 1s), while total completion
time for both is expected to be statistically the same (~13-15s for the
default 32-candidate set) -- proving faster perceived response without
doing any less actual work.

Not part of CI -- hits a live/dev server over HTTPS with a self-signed
cert. Run manually.

Usage (from repository root or backend/):
    python backend/scripts/benchmark_missing_middle_name_stream.py [base_url] [first_name] [last_name]

Defaults to the dev stack and the عباس / زهرالدين case used to diagnose
the original slowness complaint.
"""

from __future__ import annotations

import sys
import json
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://localhost:8112"
FIRST_NAME = sys.argv[2] if len(sys.argv) > 2 else "عباس"
LAST_NAME = sys.argv[3] if len(sys.argv) > 3 else "زهرالدين"
USERNAME = "complaint_supervisor"
PASSWORD = "5bb5a339"


def login() -> requests.Session:
    session = requests.Session()
    session.verify = False
    resp = session.post(f"{BASE_URL}/api/auth/login", json={"username": USERNAME, "password": PASSWORD})
    resp.raise_for_status()
    return session


def bench_blocking(session: requests.Session) -> dict:
    t0 = time.monotonic()
    resp = session.get(
        f"{BASE_URL}/api/records/search/patients/missing-middle-name",
        params={"first_name": FIRST_NAME, "last_name": LAST_NAME, "limit": 100},
    )
    resp.raise_for_status()
    time_to_completion = time.monotonic() - t0
    data = resp.json()
    return {
        "time_to_first_event": time_to_completion,  # blocking call: nothing arrives before the whole thing
        "time_to_completion": time_to_completion,
        "count": data.get("count"),
        "tried": data.get("tried"),
    }


def bench_streaming(session: requests.Session) -> dict:
    t0 = time.monotonic()
    first_event_time = None
    match_count = 0
    event_type = None
    final = None

    with session.get(
        f"{BASE_URL}/api/records/search/patients/missing-middle-name/stream",
        params={"first_name": FIRST_NAME, "last_name": LAST_NAME, "limit": 100},
        stream=True,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("event:"):
                event_type = raw_line[len("event:"):].strip()
                if first_event_time is None:
                    first_event_time = time.monotonic() - t0
                continue
            if not raw_line.startswith("data:"):
                continue
            payload = json.loads(raw_line[len("data:"):].strip())
            if event_type == "match":
                match_count += 1
            elif event_type == "done":
                final = payload
                break

    time_to_completion = time.monotonic() - t0
    return {
        "time_to_first_event": first_event_time,
        "time_to_completion": time_to_completion,
        "match_events": match_count,
        "count": final.get("count") if final else None,
        "tried": final.get("tried") if final else None,
    }


if __name__ == "__main__":
    print(f"base_url={BASE_URL}  first_name={FIRST_NAME}  last_name={LAST_NAME}\n")

    session = login()

    blocking = bench_blocking(session)
    print("Blocking endpoint:")
    print(f"  time_to_first_event:  {blocking['time_to_first_event']:.3f}s  (== time_to_completion, nothing streams)")
    print(f"  time_to_completion:   {blocking['time_to_completion']:.3f}s")
    print(f"  count={blocking['count']}  tried={blocking['tried']}\n")

    streaming = bench_streaming(session)
    print("Streaming endpoint:")
    print(f"  time_to_first_event:  {streaming['time_to_first_event']:.3f}s")
    print(f"  time_to_completion:   {streaming['time_to_completion']:.3f}s")
    print(f"  match_events={streaming['match_events']}  count={streaming['count']}  tried={streaming['tried']}\n")

    print(
        f"Perceived-speed win: first feedback arrives "
        f"{blocking['time_to_completion'] / max(streaming['time_to_first_event'], 0.001):.0f}x sooner; "
        f"total work done is unchanged (tried={blocking['tried']} vs {streaming['tried']})."
    )
