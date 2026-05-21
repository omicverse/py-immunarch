"""Repertoire filtering — :func:`repFilter` and query helpers.

Faithful port of immunarch's ``filters.R``: filter an :class:`ImmunData`
by metadata, by repertoire-level statistics, or by clonotype-level data.
"""
from __future__ import annotations

from collections import OrderedDict

import pandas as pd

from .io import IMMCOL, ImmunData

__all__ = [
    "repFilter",
    "include",
    "exclude",
    "lessthan",
    "morethan",
    "interval",
]


# --- query constructors ---------------------------------------------------
def include(*args):
    """Keep rows whose value matches one of ``args``."""
    if not args:
        raise ValueError("include() expects at least 1 argument.")
    return ("include",) + tuple(args)


def exclude(*args):
    """Drop rows whose value matches one of ``args``."""
    if not args:
        raise ValueError("exclude() expects at least 1 argument.")
    return ("exclude",) + tuple(args)


def lessthan(value):
    """Keep rows/samples with a numeric value strictly less than ``value``."""
    return ("lessthan", value)


def morethan(value):
    """Keep rows/samples with a numeric value strictly above ``value``."""
    return ("morethan", value)


def interval(frm, to):
    """Keep rows/samples with a value in ``[frm, to)``."""
    return ("interval", frm, to)


# --------------------------------------------------------------------------
def _filter_table(table: pd.DataFrame, column: str, qtype: str,
                  qargs, match: str) -> pd.DataFrame:
    """Apply one query to a DataFrame (metadata or a repertoire)."""
    series = table[column]
    if qtype == "include":
        if match == "exact":
            mask = series.isin(qargs)
        elif match == "startswith":
            mask = series.astype(str).str.startswith(tuple(qargs))
        else:  # substring
            mask = series.astype(str).apply(
                lambda s: any(a in s for a in qargs))
        return table[mask]
    if qtype == "exclude":
        if match == "exact":
            mask = ~series.isin(qargs)
        elif match == "startswith":
            mask = ~series.astype(str).str.startswith(tuple(qargs))
        else:
            mask = ~series.astype(str).apply(
                lambda s: any(a in s for a in qargs))
        return table[mask]
    if qtype == "lessthan":
        return table[pd.to_numeric(series) < float(qargs[0])]
    if qtype == "morethan":
        return table[pd.to_numeric(series) > float(qargs[0])]
    if qtype == "interval":
        num = pd.to_numeric(series)
        return table[(num >= float(qargs[0])) & (num < float(qargs[1]))]
    raise ValueError(f"Unknown query type {qtype!r}.")


def _filter_by_meta(data: ImmunData, query: dict, match: str) -> ImmunData:
    meta = data.meta.copy()
    for col, q in query.items():
        if col not in meta.columns:
            raise ValueError(f"Column {col!r} not found in metadata.")
        meta = _filter_table(meta, col, q[0], q[1:], match)
    keep = list(meta["Sample"])
    new_data = OrderedDict((s, data.data[s]) for s in keep if s in data.data)
    return ImmunData(new_data, meta.reset_index(drop=True))


def _filter_by_repertoire(data: ImmunData, query: dict) -> ImmunData:
    keep = OrderedDict(data.data)
    for key, q in query.items():
        qtype = q[0]
        qargs = q[1:]
        if key == "n_clonotypes":
            counts = {s: len(df) for s, df in keep.items()}
        elif key == "n_clones":
            counts = {s: float(df[IMMCOL.count].sum())
                      for s, df in keep.items()}
        else:
            raise ValueError(
                f"Bad by.repertoire key {key!r}; use 'n_clonotypes' "
                f"or 'n_clones'.")
        if qtype == "lessthan":
            keep = OrderedDict((s, keep[s]) for s in keep
                               if counts[s] < float(qargs[0]))
        elif qtype == "morethan":
            keep = OrderedDict((s, keep[s]) for s in keep
                               if counts[s] > float(qargs[0]))
        elif qtype == "interval":
            keep = OrderedDict((s, keep[s]) for s in keep
                               if float(qargs[0]) <= counts[s]
                               < float(qargs[1]))
    meta = data.meta[data.meta["Sample"].isin(keep.keys())]
    return ImmunData(keep, meta.reset_index(drop=True))


def _filter_by_clonotype(data: ImmunData, query: dict,
                         match: str) -> ImmunData:
    keep = OrderedDict()
    for s, df in data.data.items():
        sub = df
        for col, q in query.items():
            if col not in sub.columns:
                raise ValueError(
                    f"Column {col!r} not found in sample {s!r}.")
            sub = _filter_table(sub, col, q[0], q[1:], match)
        if len(sub) > 0:
            keep[s] = sub.reset_index(drop=True)
    meta = data.meta[data.meta["Sample"].isin(keep.keys())]
    return ImmunData(keep, meta.reset_index(drop=True))


def repFilter(data: ImmunData, method: str = "by.clonotype",
              query=None, match: str = "exact") -> ImmunData:
    """Filter an immune dataset by metadata, repertoire stats or clonotypes.

    Parameters
    ----------
    data
        An :class:`ImmunData`.
    method
        ``"by.meta"``, ``"by.repertoire"`` (alias ``"by.rep"``) or
        ``"by.clonotype"`` (alias ``"by.cl"``).
    query
        A ``{column: query}`` dict where each query is built with
        :func:`include`, :func:`exclude`, :func:`lessthan`,
        :func:`morethan` or :func:`interval`.
    match
        Matching mode for include/exclude: ``"exact"``, ``"startswith"``
        or ``"substring"``.

    Returns
    -------
    ImmunData
        A new filtered dataset.
    """
    if not isinstance(data, ImmunData):
        raise TypeError("repFilter expects an ImmunData object.")
    if query is None:
        query = {IMMCOL.cdr3aa: exclude("partial", "out_of_frame")}
    if not query:
        raise ValueError("query must be a non-empty named mapping.")
    if match not in ("exact", "startswith", "substring"):
        raise ValueError(f"Unknown matching method {match!r}.")

    m = method.lower()
    if m == "by.meta":
        return _filter_by_meta(data, query, match)
    if m in ("by.repertoire", "by.rep"):
        return _filter_by_repertoire(data, query)
    if m in ("by.clonotype", "by.cl"):
        return _filter_by_clonotype(data, query, match)
    raise ValueError(f"Unknown method {method!r}.")
