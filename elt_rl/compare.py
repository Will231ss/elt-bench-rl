"""Column-level comparison of a produced data model against the gold output.

Adapted from ELT-Bench's evaluation/eva_stage2.py (`check_corretness` with the
"typed" rule and `sort_by_keys`) so local grading agrees with the official
evaluator: rows are sorted by the unique key, every gold column must exist
(case-insensitive, extra columns allowed), integer columns must match exactly,
fractional columns within 1% and to the gold column's precision, and text is
compared after strip + lowercase.

The one addition is that the result is per column, which the reward uses for
partial credit.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd


def _sort_key(col: pd.Series) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    if col.notna().sum() > 0 and int(num.notna().sum()) == int(col.notna().sum()):
        return num
    return col.astype("string").str.strip().str.lower()


def sort_by_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    lower = {c.lower(): c for c in df.columns}
    resolved = [lower[k.lower()] for k in keys if k.lower() in lower]
    ordered = resolved + [c for c in df.columns if c not in resolved]
    aux = pd.DataFrame({c: _sort_key(df[c]) for c in ordered})
    idx = aux.sort_values(by=ordered, kind="mergesort", na_position="last").index
    return df.loc[idx].reset_index(drop=True)


def _numeric(s: pd.Series) -> pd.Series | None:
    out = pd.to_numeric(s, errors="coerce")
    return out if out.isna().sum() == s.isna().sum() else None


def _decimals(v) -> int:
    try:
        exp = Decimal(str(v)).normalize().as_tuple().exponent
    except InvalidOperation:
        return 0
    return max(0, -exp) if isinstance(exp, int) else 0


def column_matches(gold: pd.Series, out: pd.Series, rtol: float = 1e-2) -> bool:
    if len(gold) != len(out):
        return False
    gold = gold.reset_index(drop=True)
    out = out.reset_index(drop=True)
    if not gold.isna().equals(out.isna()):
        return False
    mask = gold.notna()
    g_num, o_num = _numeric(gold), _numeric(out)
    if g_num is not None and o_num is not None:
        g, o = g_num[mask].to_numpy(float), o_num[mask].to_numpy(float)
        if len(g) == 0:
            return True
        if np.all(np.mod(g, 1) == 0):  # integer column: exact (float noise only)
            return bool(np.all(np.abs(g - o) <= 1e-9 + 1e-9 * np.abs(o)))
        places = max(1, max(_decimals(v) for v in gold[mask]))
        tol = np.maximum(rtol * np.abs(g), 10.0 ** -places * 1.000001)
        return bool(np.all(np.abs(g - o) <= tol))
    g_txt = gold[mask].astype(str).str.strip().str.lower()
    o_txt = out[mask].astype(str).str.strip().str.lower()
    return bool((g_txt.to_numpy() == o_txt.to_numpy()).all())


def compare_model(gold: pd.DataFrame, out: pd.DataFrame, keys: list[str]) -> dict:
    """Return {"match", "matched", "unmatched", "missed", "row_count_ok"}."""
    gold = sort_by_keys(gold, keys)
    out = sort_by_keys(out, keys)
    out_cols = {c.lower(): c for c in out.columns}
    matched, unmatched, missed = [], [], []
    for col in gold.columns:
        actual = out_cols.get(col.lower())
        if actual is None:
            missed.append(col)
        elif column_matches(gold[col], out[actual]):
            matched.append(col)
        else:
            unmatched.append(col)
    return {
        "match": not unmatched and not missed,
        "matched": matched,
        "unmatched": unmatched,
        "missed": missed,
        "row_count_ok": len(gold) == len(out),
    }
