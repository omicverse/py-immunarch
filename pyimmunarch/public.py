"""Public-clonotype repertoires — :func:`pubRep` and friends.

Faithful port of immunarch's ``public.R``: building the public-repertoire
table, filtering it by metadata groups, applying transformations and
computing public-clonotype incidence statistics.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from .io import IMMCOL, ImmunData
from .utils import as_repertoire_list, coding
from .utils import process_col_argument as _proc

__all__ = [
    "pubRep",
    "public_matrix",
    "pubRepStatistics",
    "pubRepFilter",
    "pubRepApply",
]


def _quant_column(quant: str) -> str:
    q = quant.lower()
    if q in ("count",):
        return IMMCOL.count
    if q in ("prop", "proportion", "freq"):
        return IMMCOL.prop
    return IMMCOL.count


def pubRep(data, col: str = "aa+v", quant: str = "count",
           coding_only: bool = True, min_samples: int = 1,
           max_samples=None) -> pd.DataFrame:
    """Build a public-clonotype repertoire across samples.

    Parameters
    ----------
    data
        :class:`ImmunData` or list/dict of repertoires.
    col
        ``.col`` string defining the clonotype key, e.g. ``"aa+v"``.
    quant
        ``"count"`` or ``"prop"`` — the per-sample quantity to record.
    coding_only
        If ``True``, drop non-coding sequences first.
    min_samples, max_samples
        Keep only clonotypes present in this many samples.

    Returns
    -------
    pandas.DataFrame
        Columns: the clonotype key column(s), ``Samples`` (incidence) and
        one column per sample.  Sorted by descending incidence.
    """
    reps = as_repertoire_list(data)
    if coding_only:
        reps = coding(reps)

    cols = _proc(col)
    qcol = _quant_column(quant)

    merged = None
    sample_names = []
    for name, df in reps.items():
        sub = df[cols + [qcol]].copy()
        agg = sub.groupby(cols, dropna=False, sort=False)[qcol].sum()
        t = agg.reset_index()
        t = t.rename(columns={qcol: name})
        sample_names.append(name)
        merged = t if merged is None else merged.merge(t, on=cols,
                                                       how="outer")

    sample_block = merged[sample_names]
    merged.insert(len(cols), "Samples",
                  sample_block.notna().sum(axis=1).astype(int))

    if max_samples is None:
        max_samples = int(merged["Samples"].max())
    merged = merged[(merged["Samples"] >= min_samples)
                    & (merged["Samples"] <= max_samples)]
    merged = merged.sort_values("Samples", ascending=False,
                                kind="stable").reset_index(drop=True)
    merged.attrs["immunr"] = "public_repertoire"
    merged.attrs["key_cols"] = cols
    return merged


def public_matrix(pr: pd.DataFrame) -> pd.DataFrame:
    """Extract the per-sample count/proportion sub-matrix of a pubRep table."""
    idx = list(pr.columns).index("Samples") + 1
    return pr.iloc[:, idx:].copy()


def pubRepStatistics(pr: pd.DataFrame) -> pd.DataFrame:
    """Incidence statistics: how many clonotypes are shared by each group.

    Returns a table of ``Group`` (the set of samples sharing a clonotype,
    joined by ``&``) and ``Count`` (number of such clonotypes), restricted
    to clonotypes present in more than one sample.
    """
    if pr.attrs.get("immunr") != "public_repertoire":
        raise ValueError("pubRepStatistics expects the output of pubRep().")
    sample_names = list(public_matrix(pr).columns)
    mat = pr[sample_names]
    groups = []
    for i, row in mat.iterrows():
        present = [s for s in sample_names if not pd.isna(row[s])]
        if len(present) > 1:
            groups.append("&".join(present))
    vc = pd.Series(groups).value_counts()
    out = pd.DataFrame({"Group": vc.index, "Count": vc.values})
    out.attrs["immunr"] = "public_statistics"
    return out


def pubRepFilter(pr: pd.DataFrame, meta: pd.DataFrame, by: dict,
                 min_samples: int = 1) -> pd.DataFrame:
    """Filter a public repertoire to samples belonging to a metadata group.

    Parameters
    ----------
    pr
        A :func:`pubRep` output.
    meta
        Sample metadata with a ``Sample`` column.
    by
        Mapping of metadata column -> required value, e.g.
        ``{"Status": "MS"}``.
    min_samples
        Drop clonotypes below this incidence after filtering.
    """
    sample_names = list(public_matrix(pr).columns)
    sel = None
    for k, v in by.items():
        grp = set(meta.loc[meta[k] == v, "Sample"])
        sel = grp if sel is None else (sel & grp)
    keep = [s for s in sample_names if s in (sel or set())]
    if not keep:
        raise ValueError("No samples matched the .by filter.")

    idx = list(pr.columns).index("Samples")
    front = list(pr.columns[: idx + 1])
    new = pr[front + keep].copy()
    new["Samples"] = new[keep].notna().sum(axis=1).astype(int)
    new = new[new["Samples"] >= min_samples].reset_index(drop=True)
    new.attrs["immunr"] = "public_repertoire"
    return new


def pubRepApply(pr1: pd.DataFrame, pr2: pd.DataFrame, fun=None) -> pd.DataFrame:
    """Join two public repertoires and apply ``fun`` to paired frequencies.

    For each shared clonotype the mean per-sample quantity of each input is
    computed; ``fun`` (default ``log10(x)/log10(y)``) is then applied to the
    ``(Quant.x, Quant.y)`` pair.
    """
    if fun is None:
        def fun(x, y):
            return np.log10(x) / np.log10(y)

    idx1 = list(pr1.columns).index("Samples")
    key_cols = list(pr1.columns[:idx1])

    q1 = public_matrix(pr1).mean(axis=1, skipna=True)
    q2 = public_matrix(pr2).mean(axis=1, skipna=True)
    a = pr1.iloc[:, : idx1 + 1].copy()
    a["Quant"] = q1.values
    b = pr2.iloc[:, : list(pr2.columns).index("Samples") + 1].copy()
    b["Quant"] = q2.values

    merged = a.merge(b, on=key_cols, how="inner", suffixes=(".x", ".y"))
    merged["Samples"] = merged["Samples.x"] + merged["Samples.y"]
    merged = merged.drop(columns=["Samples.x", "Samples.y"])
    merged["Result"] = [fun(x, y) for x, y in
                        zip(merged["Quant.x"], merged["Quant.y"])]
    merged.attrs["immunr"] = "public_repertoire_apply"
    return merged
