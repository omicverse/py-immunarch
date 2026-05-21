"""Repertoire down-/re-sampling — :func:`repSample`.

Faithful port of immunarch's ``sample.R``: bootstrap an immune repertoire
to a fixed number of reads or clonotypes.  Three resampling schemes:

* ``"downsample"`` — sample ``.n`` *reads* without replacement, i.e. expand
  every clonotype into one index per read, draw ``.n`` of them and re-count
  (immunarch's ``downsample_col`` / C++ ``fill_vec`` + ``fill_reads``);
* ``"resample"``   — draw ``.n`` reads with a single multinomial draw over
  the clonotype proportions (immunarch's ``resample_col`` / ``rmultinom``);
* ``"sample"``     — sample ``.n`` *clonotypes* (rows) without replacement,
  optionally weighting the draw by clone abundance.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd

from .io import IMMCOL, ImmunData
from .utils import as_repertoire_list

__all__ = ["repSample"]


# --------------------------------------------------------------------------
def _downsample_col(counts: np.ndarray, n: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Sample ``n`` reads without replacement; return new per-clonotype counts.

    Replicates immunarch's ``downsample_col``: build a vector with one entry
    per read holding that read's clonotype index, sample ``n`` of those
    indices without replacement, then tally reads per clonotype.
    """
    total = int(counts.sum())
    if n > total:
        raise ValueError(
            f"Cannot downsample to {n} reads: repertoire has only {total}."
        )
    read_indices = np.repeat(np.arange(len(counts)), counts.astype(int))
    chosen = rng.choice(read_indices, size=n, replace=False)
    new = np.zeros(len(counts), dtype=np.int64)
    idx, cnt = np.unique(chosen, return_counts=True)
    new[idx] = cnt
    return new


def _resample_col(counts: np.ndarray, n: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Multinomial resample of ``n`` reads over the clonotype proportions."""
    p = counts.astype(float)
    p = p / p.sum()
    return rng.multinomial(n, p)


# --------------------------------------------------------------------------
def _downsample_one(df: pd.DataFrame, n: int,
                    rng: np.random.Generator) -> pd.DataFrame:
    counts = df[IMMCOL.count].to_numpy(dtype=float)
    new_col = _downsample_col(counts, n, rng)
    out = df.loc[new_col > 0].copy()
    new_col = new_col[new_col > 0]
    out[IMMCOL.count] = new_col
    out[IMMCOL.prop] = new_col / new_col.sum()
    return out.sort_values(IMMCOL.prop, ascending=False,
                           kind="stable").reset_index(drop=True)


def _resample_one(df: pd.DataFrame, n: int,
                  rng: np.random.Generator) -> pd.DataFrame:
    counts = df[IMMCOL.count].to_numpy(dtype=float)
    new_col = _resample_col(counts, n, rng)
    out = df.loc[new_col != 0].copy()
    new_col = new_col[new_col != 0]
    out[IMMCOL.count] = new_col
    out[IMMCOL.prop] = new_col / new_col.sum()
    return out.sort_values(IMMCOL.prop, ascending=False,
                           kind="stable").reset_index(drop=True)


def _sample_one(df: pd.DataFrame, n: int, prob: bool,
                rng: np.random.Generator) -> pd.DataFrame:
    if n > len(df):
        raise ValueError(
            f"Cannot sample {n} clonotypes: repertoire has only {len(df)}."
        )
    weights = None
    if prob:
        weights = df[IMMCOL.count].to_numpy(dtype=float)
        weights = weights / weights.sum()
    idx = rng.choice(len(df), size=n, replace=False, p=weights)
    out = df.iloc[idx].copy()
    cnt = out[IMMCOL.count].to_numpy(dtype=float)
    out[IMMCOL.prop] = cnt / cnt.sum()
    return out.sort_values(IMMCOL.prop, ascending=False,
                           kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------
def repSample(data, method: str = "downsample", n: Optional[int] = None,
              prob: bool = True, seed: Optional[int] = None):
    """Down-sample / resample immune repertoires to a fixed depth.

    Parameters
    ----------
    data
        :class:`ImmunData`, list/dict of repertoires or a single repertoire
        :class:`pandas.DataFrame`.
    method
        ``"downsample"`` — sample ``n`` reads without replacement;
        ``"resample"``  — multinomial draw of ``n`` reads;
        ``"sample"``    — sample ``n`` clonotypes (rows) without replacement.
    n
        Target depth.  When ``None`` it is auto-chosen as immunarch does:
        for ``"downsample"`` / ``"resample"`` the smallest total read count
        across repertoires, for ``"sample"`` the smallest clonotype count
        (and, for a lone repertoire, defaults to ``1000``).
    prob
        For ``method="sample"`` only: weight the clonotype draw by clone
        abundance (``True``, immunarch's default) or sample uniformly.
    seed
        Optional RNG seed for reproducibility.

    Returns
    -------
    Same container type as ``data`` with the resampled repertoires.
    """
    method = method.lower()
    if method not in ("downsample", "resample", "sample"):
        raise ValueError(
            "Wrong resampling method. Use 'downsample', 'resample' or "
            "'sample'."
        )
    rng = np.random.default_rng(seed)

    single = not isinstance(data, (ImmunData, dict, list, tuple))
    reps = as_repertoire_list(data)
    names = list(reps.keys())

    if n is None:
        if single or len(reps) == 1:
            if method == "sample":
                n = int(min(len(df) for df in reps.values()))
            else:
                n = 1000
        elif method in ("downsample", "resample"):
            n = int(min(df[IMMCOL.count].sum() for df in reps.values()))
        else:  # sample
            n = int(min(len(df) for df in reps.values()))

    out = OrderedDict()
    for name, df in reps.items():
        if method == "downsample":
            out[name] = _downsample_one(df, n, rng)
        elif method == "resample":
            out[name] = _resample_one(df, n, rng)
        else:
            out[name] = _sample_one(df, n, prob, rng)

    if single:
        return out[names[0]]
    if isinstance(data, ImmunData):
        return ImmunData(out, data.meta.copy())
    if isinstance(data, dict):
        return out
    return list(out.values())
