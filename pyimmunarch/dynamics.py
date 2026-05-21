"""Clonotype tracking — :func:`trackClonotypes`.

Faithful port of immunarch's ``dynamics.R``: follow specified clonotypes'
abundance across samples / time points.
"""
from __future__ import annotations

import pandas as pd

from .io import IMMCOL
from .utils import as_repertoire_list
from .utils import process_col_argument as _proc

__all__ = ["trackClonotypes"]


def trackClonotypes(data, which=(1, 15), col: str = "aa",
                    norm: bool = True) -> pd.DataFrame:
    """Track the abundance of clonotypes across repertoires.

    Parameters
    ----------
    data
        :class:`ImmunData` or list/dict of >= 2 repertoires.
    which
        Selection of clonotypes to track:

        * ``(sample, n)`` — the ``n`` most abundant clonotypes of the named
          / indexed sample.  An integer ``sample`` is **1-based** (to match
          immunarch's ``list(1, n)`` convention);
        * a sequence of strings — track those exact sequences;
        * a DataFrame — its rows define the clonotype keys.
    col
        ``.col`` string, e.g. ``"aa"`` or ``"nt+v"`` (used for the tuple
        forms of ``which``).
    norm
        If ``True``, record ``Proportion`` instead of ``Clones``.

    Returns
    -------
    pandas.DataFrame
        The clonotype key column(s) followed by one column per repertoire.
    """
    reps = as_repertoire_list(data)
    if len(reps) < 2:
        raise ValueError("Need at least 2 repertoires to track clonotypes.")
    names = list(reps.keys())

    is_select = (
        isinstance(which, (list, tuple)) and len(which) == 2
        and (isinstance(which[0], int) or which[0] in reps)
        and isinstance(which[1], int)
    )

    if isinstance(which, pd.DataFrame):
        target = which.drop_duplicates().reset_index(drop=True)
        cols = list(target.columns)
    elif is_select:
        cols = _proc(col)
        sample = which[0]
        n = which[1]
        # integer sample index is 1-based, matching immunarch's list(1, n)
        df = reps[sample] if isinstance(sample, str) \
            else list(reps.values())[sample - 1]
        target = (df.sort_values(IMMCOL.count, ascending=False,
                                 kind="stable")[cols]
                  .head(n).drop_duplicates().reset_index(drop=True))
    else:
        cols = _proc(col)
        seqs = list(pd.unique(pd.Series(list(which))))
        target = pd.DataFrame({cols[0]: seqs})

    result = target.copy()
    for name, df in reps.items():
        sub = df[cols + [IMMCOL.count]].copy()
        sub = sub.rename(columns={IMMCOL.count: "Count"})
        if norm:
            tot = sub["Count"].sum()
            sub["Count"] = sub["Count"] / tot if tot else sub["Count"]
        agg = sub.groupby(cols, dropna=False, sort=False)["Count"].sum()
        agg = agg.reset_index().rename(columns={"Count": name})
        result = result.merge(agg, on=cols, how="left")

    for name in names:
        result[name] = result[name].fillna(0)
    result.attrs["immunr"] = "dynamics"
    return result
