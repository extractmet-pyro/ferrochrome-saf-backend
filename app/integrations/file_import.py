"""Historical telemetry / lab import (CSV, Parquet, Excel) with strict validation.

Rules: explicit `timestamp` column (ISO-8601, stored UTC), monotonic non-duplicated times, size
and file-type limits, numeric tags only, no silent discarding: every rejected row is counted
and reported."""
from __future__ import annotations

import io

import pandas as pd

ALLOWED = {".csv", ".parquet", ".xlsx"}


class ImportError_(ValueError):
    pass


def read_table(name: str, data: bytes, max_mb: float) -> pd.DataFrame:
    ext = name[name.rfind("."):].lower() if "." in name else ""
    if ext not in ALLOWED:
        raise ImportError_(f"file type {ext} not allowed ({sorted(ALLOWED)})")
    if len(data) > max_mb * 1e6:
        raise ImportError_(f"file exceeds {max_mb} MB")
    if ext == ".csv":
        return pd.read_csv(io.BytesIO(data))
    if ext == ".parquet":
        return pd.read_parquet(io.BytesIO(data))
    return pd.read_excel(io.BytesIO(data))


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if "timestamp" not in df.columns:
        raise ImportError_("missing 'timestamp' column")
    rep = {"rows_in": len(df)}
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    bad_ts = int(ts.isna().sum())
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"])
    dup = int(df["timestamp"].duplicated().sum())
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    nonnum = [c for c in df.columns if c != "timestamp" and not pd.api.types.is_numeric_dtype(df[c])]
    rep.update({"invalid_timestamps_rejected": bad_ts, "duplicate_timestamps_rejected": dup,
                "non_numeric_columns_ignored": nonnum, "rows_out": len(df),
                "span_utc": [df["timestamp"].min().isoformat(), df["timestamp"].max().isoformat()] if len(df) else None})
    df = df.drop(columns=nonnum)
    return df, rep
