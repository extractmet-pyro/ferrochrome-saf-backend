"""Map site tag names to canonical twin tags (CSV: site_tag,twin_tag,scale,offset)."""
from __future__ import annotations

import csv
import io

from .synthetic_gateway import TAGS

CANON = {t.tag for t in TAGS}


def parse_mapping(text: str) -> dict:
    m = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row["twin_tag"] not in CANON:
            raise ValueError(f"unknown twin tag {row['twin_tag']}")
        m[row["site_tag"]] = (row["twin_tag"], float(row.get("scale") or 1), float(row.get("offset") or 0))
    return m


def apply(df, mapping: dict):
    out = df.copy()
    ren = {}
    for site, (twin, sc, off) in mapping.items():
        if site in out.columns:
            out[site] = out[site] * sc + off
            ren[site] = twin
    return out.rename(columns=ren)
