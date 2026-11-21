"""
Merges the database searches into one deduplicated title/abstract screening sheet and writes the PRISMA identification counts.

Sources
  OpenAlex  focused query (Q_domain.jsonl)                                   - API, see search_openalex.py
  IEEE      IEEE Xplore export (ieee_xplore.csv)                             - manual export, query in search_log.json
  ACM       ACM DL full-text search: count only (bulk export needs a subscription). ACM-published records enter through
            OpenAlex (DOI prefix 10.1145); papers missed this way are recovered by snowballing.
Deduplication: same DOI, else same normalised title. Peer-reviewed records are kept over preprints, then more-cited ones.
Outputs: search/screening.csv (with empty R1/R2 decision columns) and search/prisma_identification.json.

  python survey/make_screening_sheet.py
"""
import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent / "search"
YEARS = (2021, 2026)


def abstract(w):
    inv = w.get("abstract_inverted_index") or {}
    return " ".join(word for _, word in sorted((i, word) for word, idx in inv.items() for i in idx))


oa = []
for line in open(D / "Q_domain.jsonl", encoding="utf-8"):
    w = json.loads(line)
    src = (w.get("primary_location") or {}).get("source") or {}
    oa.append({"source_db": "OpenAlex", "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
               "title": w.get("display_name") or "", "year": w.get("publication_year"), "type": w.get("type"),
               "venue": src.get("display_name") or "", "cited_by": w.get("cited_by_count", 0), "abstract": abstract(w)})
oa = pd.DataFrame(oa)

ie = pd.read_csv(D / "ieee_xplore.csv")
ie = pd.DataFrame({"source_db": "IEEE Xplore", "doi": ie["DOI"].fillna("").str.lower(), "title": ie["Document Title"],
                   "year": ie["Publication Year"], "venue": ie["Publication Title"],
                   "type": ie["Document Identifier"].str.contains("Conference", case=False, na=False).map(
                       {True: "conference-paper", False: "article"}),
                   "cited_by": ie["Article Citation Count"].fillna(0), "abstract": ie["Abstract"].fillna("")})

log = json.load(open(D / "search_log.json"))
ident = {"OpenAlex": len(oa), "IEEE Xplore": len(ie), "ACM Digital Library (full text, count only)":
         log["manual_searches"]["ACM Digital Library"]["hits"]}

df = pd.concat([oa, ie], ignore_index=True)
n_all = len(df)
out_of_range = ~df.year.between(*YEARS)
df = df[~out_of_range]
df["norm_title"] = df.title.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
df["peer_reviewed"] = df.type.isin(["article", "conference-paper", "review"])
df = df.sort_values(["peer_reviewed", "cited_by"], ascending=False)
found_in = df.groupby("norm_title").source_db.agg(lambda s: "; ".join(sorted(set(s))))
before = len(df)
df = df[~(df.doi.ne("") & df.doi.duplicated())]
df = df[~df.norm_title.duplicated()]
df["found_in"] = df.norm_title.map(found_in)
df = df.sort_values(["year", "title"]).drop(columns=["norm_title", "source_db"]).reset_index(drop=True)
df.insert(0, "rec_id", [f"R{i + 1:03d}" for i in range(len(df))])
for col in ("R1_decision", "R1_reason", "R2_decision", "R2_reason", "final_decision"):
    df[col] = ""
df.to_csv(D / "screening.csv", index=False, encoding="utf-8-sig")

prisma = {"identified": ident, "records_from_exports": n_all, "removed_outside_2021_2026": int(out_of_range.sum()),
          "duplicates_removed": before - len(df), "records_screened": len(df),
          "found_in_both_OpenAlex_and_IEEE": int(df.found_in.str.contains(";").sum()),
          "acm_published_records_via_openalex": int(df.doi.str.startswith("10.1145/").sum())}
json.dump(prisma, open(D / "prisma_identification.json", "w"), indent=2)
print(json.dumps(prisma, indent=2))
