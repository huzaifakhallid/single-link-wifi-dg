"""
Literature search on OpenAlex (free substitute for Scopus / Web of Science).

Two queries, both restricted to 2021-01-01 .. 2026-12-31 and to journal articles, conference papers and preprints:
  Q_broad   WiFi/CSI sensing for gesture or activity recognition         -> corpus for bibliometrics (trends, keywords)
  Q_domain  Q_broad AND cross-domain / domain generalization / adaptation -> corpus for the taxonomy and review

Each run saves the raw API records (JSON Lines), a flat CSV and a log with the exact query strings, the run date and
the hit counts, so the search can be reported and repeated.

  python survey/search_openalex.py
"""
import csv
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "search"
API = "https://api.openalex.org/works"
UA = "single-link-wifi-survey (research script)"
# OpenAlex requires a free API key (https://openalex.org/settings/api). Set it in the environment, never in this file:
#   PowerShell:  $env:OPENALEX_API_KEY = "..."      bash:  export OPENALEX_API_KEY=...
API_KEY = os.environ.get("OPENALEX_API_KEY", "")

WIFI = '("WiFi" OR "Wi-Fi" OR "WLAN" OR "channel state information" OR "CSI")'
TASK = '("gesture recognition" OR "activity recognition" OR "gesture sensing" OR "action recognition" OR "human sensing")'
DOMAIN = ('("cross-domain" OR "domain generalization" OR "domain generalisation" OR "domain adaptation" OR "domain-independent" '
          'OR "domain invariant" OR "zero-effort" OR "cross-orientation" OR "cross-location" OR "cross-user" OR "few-shot" '
          'OR "single link" OR "single-link")')
QUERIES = {
    "Q_broad": f"{WIFI} AND {TASK}",
    "Q_domain": f"{WIFI} AND {TASK} AND {DOMAIN}",
}
FILTERS = "from_publication_date:2021-01-01,to_publication_date:2026-12-31,type:article|preprint|conference-paper|review"


def fetch(query: str):
    params = {"filter": f"title_and_abstract.search:{query},{FILTERS}", "per-page": 200, "cursor": "*", "api_key": API_KEY}
    rows, total = [], None
    while True:
        url = API + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
            d = json.load(r)
        total = d["meta"]["count"]
        rows += d["results"]
        nxt = d["meta"].get("next_cursor")
        if not nxt or not d["results"]:
            break
        params["cursor"] = nxt
        time.sleep(0.2)
    assert len(rows) == total, f"retrieved {len(rows)} of {total}"
    return rows


def flat(w):
    src = (w.get("primary_location") or {}).get("source") or {}
    return {
        "openalex_id": w["id"].rsplit("/", 1)[-1], "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "title": w.get("display_name") or "", "year": w.get("publication_year"), "type": w.get("type"),
        "venue": src.get("display_name") or "", "venue_type": src.get("type") or "",
        "authors": "; ".join(a["author"]["display_name"] for a in w.get("authorships", [])),
        "cited_by": w.get("cited_by_count", 0),
        "keywords": "; ".join(k["display_name"] for k in w.get("keywords", [])),
        "topic": ((w.get("primary_topic") or {}).get("display_name") or ""),
    }


def main():
    if not API_KEY:
        sys.exit("OPENALEX_API_KEY is not set. Get a free key at https://openalex.org/settings/api and set it first.")
    OUT.mkdir(parents=True, exist_ok=True)
    run = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log = {"database": "OpenAlex (api.openalex.org)", "run": run, "filters": FILTERS, "queries": {}}
    for name, q in QUERIES.items():
        rows = fetch(q)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for w in rows:
                f.write(json.dumps(w) + "\n")
        flat_rows = [flat(w) for w in rows]
        with open(OUT / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
            wr.writeheader()
            wr.writerows(flat_rows)
        log["queries"][name] = {"string": q, "hits": len(rows)}
        print(f"[{name}] {len(rows)} records")
    json.dump(log, open(OUT / "search_log.json", "w"), indent=2)


if __name__ == "__main__":
    sys.exit(main())
