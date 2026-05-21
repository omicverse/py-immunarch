"""Clonal-space analysis — :func:`repClonality`.

Faithful port of immunarch's ``clonality.R``: clonal proportion, clonal
space homeostasis, top-N proportion and rare-clone proportion.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from .io import IMMCOL
from .utils import as_repertoire_list

__all__ = ["repClonality"]


def _signif(x: float, digits: int = 3) -> float:
    """Round to ``digits`` significant figures (R's ``signif``)."""
    if x == 0 or not np.isfinite(x):
        return x
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + (digits - 1))


# --------------------------------------------------------------------------
def _clonal_proportion(counts: np.ndarray, perc: float = 10.0) -> dict:
    """How many clonotypes occupy ``perc`` % of the repertoire."""
    perc = min(perc, 100.0)
    col = np.sort(counts)[::-1]
    col_sum = col.sum()
    prop = 0.0
    n = 0
    while prop < col_sum * (perc / 100.0):
        prop += col[n]
        n += 1
    return {
        "Clones": n,
        "Percentage": 100 * _signif(prop / col_sum, 3),
        "Clonal.count.prop": n / len(col),
    }


def _clonal_space_homeostasis(prop: np.ndarray, clone_types) -> np.ndarray:
    """Proportion of repertoire in each clonal-abundance bin."""
    p = prop / prop.sum()
    bounds = [0.0] + [v for _, v in clone_types]
    out = np.zeros(len(clone_types))
    for i in range(1, len(bounds)):
        lo, hi = bounds[i - 1], bounds[i]
        sel = (p > lo) & (p <= hi)
        out[i - 1] = p[sel].sum()
    return out


def _top_proportion(prop: np.ndarray, head) -> np.ndarray:
    """Cumulative proportion of the top-``h`` clonotypes for each ``h``."""
    p = np.sort(prop)[::-1]
    total = p.sum()
    return np.array([p[:h].sum() / total for h in head])


def _rare_proportion(counts: np.ndarray, bound) -> np.ndarray:
    """Proportion of clones with abundance <= each bound (last = inf)."""
    bounds = list(bound) + [np.inf]
    return np.array([counts[counts <= b].sum() / counts.sum()
                     for b in bounds])


# --------------------------------------------------------------------------
def repClonality(data, method: str = "clonal.prop", perc: float = 10.0,
                  clone_types=None, head=None, bound=None) -> pd.DataFrame:
    """Estimate clonal-space structure of immune repertoires.

    Parameters
    ----------
    data
        :class:`ImmunData`, list/dict of repertoires, or single repertoire.
    method
        ``"clonal.prop"``, ``"homeo"``, ``"top"`` or ``"rare"`` (``"tail"``
        is an alias of ``"rare"``).
    perc
        Percentage threshold for ``"clonal.prop"``.
    clone_types
        Named bin upper-bounds for ``"homeo"`` (default
        Rare/Small/Medium/Large/Hyperexpanded).
    head
        Top-N cut-offs for ``"top"``.
    bound
        Abundance thresholds for ``"rare"``.

    Returns
    -------
    pandas.DataFrame
        Per-sample clonality statistics (samples as the index).
    """
    reps = as_repertoire_list(data)
    method = method.lower()
    if method == "tail":
        method = "rare"

    if clone_types is None:
        clone_types = [("Rare", 1e-5), ("Small", 1e-4), ("Medium", 1e-3),
                       ("Large", 1e-2), ("Hyperexpanded", 1.0)]
    if head is None:
        head = [10, 100, 1000, 3000, 10000, 30000, 100000]
    if bound is None:
        bound = [1, 3, 10, 30, 100]

    names = list(reps.keys())

    if method == "clonal.prop":
        rows = OrderedDict()
        for name, df in reps.items():
            rows[name] = _clonal_proportion(
                df[IMMCOL.count].to_numpy(dtype=float), perc)
        out = pd.DataFrame(rows).T
        out.attrs["immunr"] = "clonal_prop"
        return out

    if method == "homeo":
        bins = [0.0] + [v for _, v in clone_types]
        colnames = [
            f"{nm} ({_fmt(bins[i])} < X <= {_fmt(bins[i + 1])})"
            for i, (nm, _) in enumerate(clone_types)
        ]
        mat = np.zeros((len(names), len(clone_types)))
        for i, (_, df) in enumerate(reps.items()):
            mat[i] = _clonal_space_homeostasis(
                df[IMMCOL.count].to_numpy(dtype=float), clone_types)
        out = pd.DataFrame(mat, index=names, columns=colnames)
        out.attrs["immunr"] = "homeo"
        return out

    if method == "top":
        mat = np.zeros((len(names), len(head)))
        for i, (_, df) in enumerate(reps.items()):
            mat[i] = _top_proportion(
                df[IMMCOL.prop].to_numpy(dtype=float), head)
        out = pd.DataFrame(mat, index=names, columns=[str(h) for h in head])
        out.attrs["immunr"] = "top_prop"
        return out

    if method == "rare":
        cols = [str(b) for b in bound] + ["MAX"]
        mat = np.zeros((len(names), len(bound) + 1))
        for i, (_, df) in enumerate(reps.items()):
            mat[i] = _rare_proportion(
                df[IMMCOL.count].to_numpy(dtype=float), bound)
        out = pd.DataFrame(mat, index=names, columns=cols)
        out.attrs["immunr"] = "rare_prop"
        return out

    raise ValueError(f"Unknown method {method!r}. Use 'clonal.prop', "
                     f"'homeo', 'top' or 'rare'.")


def _fmt(v: float) -> str:
    """Format a bin bound the way R prints it (e.g. ``1e-05``, ``1``)."""
    if v == int(v):
        return str(int(v))
    # R prints small numbers in scientific notation
    s = np.format_float_scientific(v, trim="-", exp_digits=2)
    s = s.replace("e+", "e+").replace("e-0", "e-0")
    return s
