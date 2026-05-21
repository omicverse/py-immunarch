"""Exploratory repertoire statistics — :func:`repExplore`.

Faithful port of immunarch's ``explore.R``: clonotype counts, repertoire
volume, CDR3 length distribution and clonal-count distribution.
"""
from __future__ import annotations

import pandas as pd

from .io import IMMCOL, ImmunData
from .utils import as_repertoire_list, coding

__all__ = ["repExplore"]


def repExplore(data, method: str = "volume", col: str = "nt",
               coding_only: bool = True) -> pd.DataFrame:
    """Compute basic repertoire statistics.

    Parameters
    ----------
    data
        An :class:`ImmunData`, a list/dict of repertoires, or a single
        repertoire DataFrame.
    method
        One of:

        * ``"volume"`` — number of unique clonotypes per sample.
        * ``"clones"`` — total number of clones (cells/reads) per sample.
        * ``"count"``  — distribution of clonotype abundances.
        * ``"len"``    — CDR3 sequence length distribution.
    col
        ``"nt"`` or ``"aa"`` — which CDR3 column to use for ``method="len"``.
    coding_only
        If ``True`` (default), only coding sequences are analysed.

    Returns
    -------
    pandas.DataFrame
        Long-format result.  ``"volume"``/``"clones"`` -> one row per sample;
        ``"count"``/``"len"`` -> one row per (sample, value) pair.
    """
    reps = as_repertoire_list(data)
    if coding_only:
        reps = coding(reps)

    method = method.lower()

    if method == "volume":
        rows = [(name, len(df)) for name, df in reps.items()]
        out = pd.DataFrame(rows, columns=["Sample", "Volume"])
        out.attrs["immunr"] = "exp_vol"
        return out

    if method in ("clones", "clone"):
        rows = [(name, float(df[IMMCOL.count].sum()))
                for name, df in reps.items()]
        out = pd.DataFrame(rows, columns=["Sample", "Clones"])
        out.attrs["immunr"] = "exp_clones"
        return out

    if method == "count":
        frames = []
        for name, df in reps.items():
            vc = df[IMMCOL.count].value_counts().sort_index()
            frames.append(pd.DataFrame({
                "Sample": name,
                "Clone.num": vc.index.astype(float),
                "Clonotypes": vc.values.astype(int),
            }))
        out = pd.concat(frames, ignore_index=True)
        out.attrs["immunr"] = "exp_count"
        return out

    if method in ("len", "lens", "length"):
        seq_col = IMMCOL.cdr3nt if col.lower() in ("nt", "nuc") \
            else IMMCOL.cdr3aa
        frames = []
        for name, df in reps.items():
            lengths = df[seq_col].astype(str).str.len()
            vc = lengths.value_counts().sort_index()
            frames.append(pd.DataFrame({
                "Sample": name,
                "Length": vc.index.astype(float),
                "Count": vc.values.astype(int),
            }))
        out = pd.concat(frames, ignore_index=True)
        out.attrs["immunr"] = "exp_len"
        return out

    raise ValueError(f"Unknown method {method!r}. Use 'volume', 'clones', "
                     f"'count' or 'len'.")
