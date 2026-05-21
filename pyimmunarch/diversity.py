"""Diversity estimators — :func:`repDiversity`.

Faithful port of immunarch's ``diversity.R``: Chao1, Hill numbers, true
diversity, Gini-Simpson, inverse Simpson, the Gini coefficient, d50/dXX
and rarefaction (with bootstrap-style extrapolation).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import norm

from .io import IMMCOL
from .utils import as_repertoire_list, check_distribution, process_col_argument

__all__ = ["repDiversity"]


# --------------------------------------------------------------------------
def _clone_counts(df: pd.DataFrame, cols) -> np.ndarray:
    """Group by clonotype columns and sum Clones (immunarch's ``vec``)."""
    g = df.groupby(list(cols), dropna=False, sort=True)[IMMCOL.count].sum()
    return g.to_numpy(dtype=float)


# --------------------------------------------------------------------------
def chao1(counts: np.ndarray) -> dict:
    """Chao1 nonparametric species-richness estimator.

    Returns Estimator, SD and the two 95 % confidence-interval bounds,
    exactly as immunarch's ``chao1``.
    """
    counts = np.asarray(counts, dtype=float)
    vals, freq = np.unique(counts, return_counts=True)
    table = dict(zip(vals, freq))
    n = counts.sum()
    D = len(counts)
    f1 = table.get(1.0)
    f2 = table.get(2.0)
    z = norm.ppf(0.975)

    if f1 is None and f2 is None:
        e = D
        uniq = np.unique(counts)
        v = (sum(np.sum(counts == i) * (np.exp(-i) - np.exp(-2 * i))
                 for i in uniq)
             - (sum(i * np.exp(-i) * np.sum(counts == i)
                    for i in uniq)) ** 2 / n)
        P = sum(np.sum(counts == i) * np.exp(-i) / D for i in uniq)
        lo = max(D, D / (1 - P) - z * np.sqrt(v) / (1 - P))
        hi = D / (1 - P) + z * np.sqrt(v) / (1 - P)
    elif f2 is None:
        f1 = float(f1)
        e = D + f1 * (f1 - 1) / 2 * (n - 1) / n
        v = ((n - 1) / n * f1 * (f1 - 1) / 2
             + ((n - 1) / n) ** 2 * f1 * (2 * f1 - 1) ** 2 / 4
             - ((n - 1) / n) ** 2 * f1 ** 4 / 4 / e)
        t = e - D
        K = np.exp(z * np.sqrt(np.log(1 + v / t ** 2)))
        lo = D + t / K
        hi = D + t * K
    else:
        f1 = float(f1)
        f2 = float(f2)
        const = (n - 1) / n
        e = D + f1 ** 2 / (2 * f2) * const
        f12 = f1 / f2
        v = f2 * (const * f12 ** 2 / 2 + const ** 2 * f12 ** 3
                  + const ** 2 * f12 ** 4 / 4)
        t = e - D
        K = np.exp(z * np.sqrt(np.log(1 + v / t ** 2)))
        lo = D + t / K
        hi = D + t * K

    return {"Estimator": e, "SD": np.sqrt(v),
            "Conf.95.lo": lo, "Conf.95.hi": hi}


# --------------------------------------------------------------------------
def diversity_eco(data: np.ndarray, q: float = 5.0) -> float:
    """True diversity (effective number of types) of order ``q``."""
    p = check_distribution(data, do_norm=None, laplace=0.0)
    if q == 0:
        return float(len(p))
    if q == 1:
        return float(np.exp(-np.sum(p * np.log(p))))
    if q > 1:
        return float(1.0 / (np.sum(p ** q) ** (1.0 / (q - 1))))
    return float("nan")


def hill_numbers(data: np.ndarray, min_q: int = 1,
                 max_q: int = 6) -> "OrderedDict[int, float]":
    """Hill numbers q = ``min_q`` .. ``max_q``."""
    p = check_distribution(data, do_norm=True, laplace=0.0)
    if min_q < 0:
        min_q = 0
    return OrderedDict((q, diversity_eco(p, q))
                       for q in range(min_q, max_q + 1))


def gini_coef(data: np.ndarray) -> float:
    """Gini coefficient of inequality (0 = even, 1 = maximally uneven)."""
    p = np.sort(check_distribution(data, do_norm=True, laplace=0.0))
    n = len(p)
    idx = np.arange(1, n + 1)
    return float(1.0 / n * (n + 1 - 2 * np.sum((n + 1 - idx) * p) / p.sum()))


def gini_simpson(data: np.ndarray) -> float:
    """Gini-Simpson index (probability of interspecific encounter)."""
    p = check_distribution(data, do_norm=True, laplace=0.0)
    return float(1.0 - np.sum(p ** 2))


def inverse_simpson(data: np.ndarray) -> float:
    """Inverse Simpson index (effective number of types)."""
    p = check_distribution(data, do_norm=True, laplace=0.0)
    return float(1.0 / np.sum(p ** 2))


def dXX(counts: np.ndarray, perc: float = 50.0) -> dict:
    """dXX index: minimum #clonotypes covering ``perc`` % of all reads."""
    col = np.sort(counts)[::-1]
    col_sum = col.sum()
    prop = 0.0
    n = 0
    while prop < col_sum * (perc / 100.0):
        prop += col[n]
        n += 1
    from .clonality import _signif
    return {"Clones": n,
            "Percentage": 100 * _signif(prop / col_sum, 3)}


# --------------------------------------------------------------------------
def _rarefaction(rep_counts: "OrderedDict[str, np.ndarray]",
                 step=None, quantile=(0.025, 0.975),
                 extrapolation=None, norm_curve: bool = True) -> pd.DataFrame:
    """Rarefaction / extrapolation curves (immunarch ``rarefaction``)."""
    z = norm.ppf(0.975)
    sums = {k: float(np.sum(v)) for k, v in rep_counts.items()}

    if step is None:
        step = min(sums.values()) // 50
    step = max(1, int(step))
    if extrapolation is None:
        extrapolation = max(sums.values()) * 20

    def alpha(n, Xi, m):
        return (1 - m / n) ** Xi

    frames = []
    for name, vec in rep_counts.items():
        Sobs = len(vec)
        Sest = chao1(vec)["Estimator"]
        if np.isnan(Sest):
            Sest = Sobs
        n = float(np.sum(vec))
        sizes = list(np.arange(step, n + 1e-9, step))
        vals, freq = np.unique(vec, return_counts=True)
        counts_tab = dict(zip(vals, freq))
        freqs = np.array(sorted(counts_tab.keys()))
        cnts = np.array([counts_tab[f] for f in freqs], dtype=float)

        muc_rows = []
        for sz in sizes:
            alphas = np.array([alpha(n, k, sz) for k in freqs])
            Sind = np.sum((1 - alphas) * cnts)
            if Sest == Sobs:
                SD = 0.0
            else:
                SD = np.sqrt(np.sum((1 - alphas) ** 2 * cnts)
                             - Sind ** 2 / Sest)
            t = Sind - Sobs
            if t != 0:
                K = np.exp(z * np.sqrt(np.log(1 + (SD / t) ** 2)))
                lo = Sobs + t * K
                hi = Sobs + t / K
            else:
                lo = hi = Sind
            muc_rows.append([sz, lo, Sind, hi])
        muc = np.array(muc_rows, dtype=float)
        types = ["interpolation"] * len(muc)

        if extrapolation > 0 and len(sizes) > 0:
            ex_sizes = list(np.arange(sizes[-1] + step, extrapolation + 1e-9,
                                      step))
            f0 = Sest - Sobs
            f1 = counts_tab.get(1.0)
            ex_rows = []
            for sz in ex_sizes:
                if f1 is None or f0 == 0:
                    Sind = Sobs
                else:
                    Sind = Sobs + f0 * (1 - np.exp(-(sz - n) / n * f1 / f0))
                ex_rows.append([sz, Sind, Sind, Sind])
            if ex_rows:
                muc = np.vstack([muc, np.array(ex_rows, dtype=float)])
                types += ["extrapolation"] * len(ex_rows)

        df = pd.DataFrame(muc, columns=["Size", f"Q{quantile[0]}",
                                        "Mean", f"Q{quantile[1]}"])
        df["Sample"] = name
        df["Type"] = types
        if norm_curve:
            df["Size"] = df["Size"] / n
            df[f"Q{quantile[0]}"] = df[f"Q{quantile[0]}"] / Sobs
            df["Mean"] = df["Mean"] / Sobs
            df[f"Q{quantile[1]}"] = df[f"Q{quantile[1]}"] / Sobs
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.attrs["immunr"] = "rarefaction"
    return out


# --------------------------------------------------------------------------
def repDiversity(data, method: str = "chao1", col: str = "aa",
                 max_q: int = 6, min_q: int = 1, q: float = 5.0,
                 perc: float = 50.0, step=None, quantile=(0.025, 0.975),
                 extrapolation=None, norm: bool = True) -> pd.DataFrame:
    """Estimate immune-repertoire diversity.

    Parameters
    ----------
    data
        :class:`ImmunData`, list/dict of repertoires, or single repertoire.
    method
        ``"chao1"``, ``"hill"``, ``"div"``, ``"gini.simp"``, ``"inv.simp"``,
        ``"gini"``, ``"d50"``, ``"dxx"`` or ``"raref"`` (aka
        ``"rarefaction"``).
    col
        ``.col`` string, e.g. ``"aa"`` or ``"aa+v"``.
    max_q, min_q
        Hill-number range for ``method="hill"``.
    q
        Order parameter for ``method="div"``.
    perc
        Percentage for ``method="dxx"`` (``"d50"`` always uses 50).
    step, quantile, extrapolation, norm
        Rarefaction parameters.

    Returns
    -------
    pandas.DataFrame
        Per-sample diversity statistics.
    """
    method = method.lower()
    if method == "rarefaction":
        method = "raref"
    cols = process_col_argument(col)
    reps = as_repertoire_list(data)
    counts = OrderedDict((name, _clone_counts(df, cols))
                         for name, df in reps.items())

    if method == "raref":
        return _rarefaction(counts, step=step, quantile=quantile,
                            extrapolation=extrapolation, norm_curve=norm)

    if method == "chao1":
        rows = OrderedDict((name, chao1(v)) for name, v in counts.items())
        out = pd.DataFrame(rows).T
        out = out[["Estimator", "SD", "Conf.95.lo", "Conf.95.hi"]]
        out.attrs["immunr"] = "chao1"
        return out

    if method == "hill":
        recs = []
        for name, v in counts.items():
            for qq, val in hill_numbers(v, min_q, max_q).items():
                recs.append((name, qq, val))
        out = pd.DataFrame(recs, columns=["Sample", "Q", "Value"])
        out.attrs["immunr"] = "hill"
        return out

    if method == "div":
        recs = [(name, diversity_eco(v, q)) for name, v in counts.items()]
        out = pd.DataFrame(recs, columns=["Sample", "Value"])
        out.attrs["immunr"] = "div"
        return out

    if method == "gini.simp":
        recs = [(name, gini_simpson(v)) for name, v in counts.items()]
        out = pd.DataFrame(recs, columns=["Sample", "Value"])
        out.attrs["immunr"] = "ginisimp"
        return out

    if method == "inv.simp":
        recs = [(name, inverse_simpson(v)) for name, v in counts.items()]
        out = pd.DataFrame(recs, columns=["Sample", "Value"])
        out.attrs["immunr"] = "invsimp"
        return out

    if method == "gini":
        recs = [(name, gini_coef(v)) for name, v in counts.items()]
        out = pd.DataFrame(recs, columns=["Sample", "Value"])
        out.attrs["immunr"] = "gini"
        return out

    if method in ("d50", "dxx"):
        pp = 50.0 if method == "d50" else perc
        rows = OrderedDict((name, dXX(v, pp)) for name, v in counts.items())
        out = pd.DataFrame(rows).T
        out.attrs["immunr"] = "dxx"
        return out

    raise ValueError(f"Unknown method {method!r}.")
