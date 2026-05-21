"""Sequence distance & clustering — :func:`seqDist`, :func:`seqCluster`.

Faithful port of immunarch's ``seqDist.R`` / ``seqCluster.R``: compute
pairwise string-distance matrices between CDR3 sequences (grouped by V/J
gene and/or sequence length) and cluster sequences within those groups by
connected components of a distance-thresholded graph.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from .io import ImmunData

__all__ = ["seqDist", "seqCluster", "hamming_dist", "levenshtein_dist"]


# --------------------------------------------------------------------------
def hamming_dist(a: str, b: str) -> int:
    """Hamming distance; ``inf`` for unequal-length strings (R's behaviour)."""
    if len(a) != len(b):
        return np.iinfo(np.int64).max
    return sum(x != y for x, y in zip(a, b))


def levenshtein_dist(a: str, b: str) -> int:
    """Levenshtein (edit) distance between two strings."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


_DIST_FUNS = {
    "hamming": hamming_dist,
    "lv": levenshtein_dist,
    "levenshtein": levenshtein_dist,
    "osa": levenshtein_dist,
}


def _dist_matrix(seqs, fn) -> pd.DataFrame:
    """Symmetric pairwise distance matrix over the *unique* sequences."""
    uniq = list(dict.fromkeys(seqs))
    n = len(uniq)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = fn(uniq[i], uniq[j])
            mat[i, j] = mat[j, i] = d
    return pd.DataFrame(mat, index=uniq, columns=uniq)


def _first_gene(value) -> str:
    """immunarch's ``add_column_with_first_gene``: keep gene before , ( *."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    s = str(value)
    for sep in (",", "(", "*"):
        s = s.split(sep)[0]
    return s


# --------------------------------------------------------------------------
def seqDist(data, col: str = "CDR3.nt", method: str = "hamming",
            group_by=("V.name", "J.name"), group_by_seqLength: bool = True,
            trim_genes: bool = True) -> "OrderedDict":
    """Compute pairwise sequence-distance matrices, grouped by gene/length.

    Parameters
    ----------
    data
        :class:`ImmunData`, list/dict of repertoires or a single repertoire.
    col
        Full column name to compute distances on (e.g. ``"CDR3.nt"``).
    method
        ``"hamming"`` (default) or ``"lv"`` / ``"levenshtein"``.
    group_by
        Column name(s) used to partition sequences before distances are
        computed; ``None`` / ``np.nan`` disables gene grouping.
    group_by_seqLength
        Also split each group by sequence length (so the resulting blocks
        contain only equal-length sequences — required for ``"hamming"``).
    trim_genes
        Reduce gene strings to the first gene before grouping.

    Returns
    -------
    OrderedDict
        ``{sample: {group_key: distance_DataFrame}}``.  ``group_key`` is a
        tuple of the grouping-column values (and the sequence length when
        ``group_by_seqLength`` is set).  Carries ``.attrs`` describing the
        ``col`` / ``group_by`` / ``group_by_length`` / ``trimmed`` settings.
    """
    if col in ("aa", "nt", "v", "j", "aa+v"):
        raise ValueError("Please provide a full column name (e.g. 'CDR3.nt').")

    gb = list(group_by) if isinstance(group_by, (list, tuple)) else (
        [] if group_by is None
        or (isinstance(group_by, float) and np.isnan(group_by))
        else [group_by])

    fn = _DIST_FUNS.get(str(method).lower())
    if fn is None:
        raise ValueError(
            f"Unknown .method {method!r}. Use 'hamming' or 'lv'."
        )

    if isinstance(data, ImmunData):
        reps = data.data
    elif isinstance(data, dict):
        reps = OrderedDict(data)
    elif isinstance(data, (list, tuple)):
        reps = OrderedDict((f"Sample{i + 1}", d)
                           for i, d in enumerate(data))
    else:
        reps = OrderedDict([("Sample", data)])

    result: "OrderedDict[str, OrderedDict]" = OrderedDict()
    for name, df in reps.items():
        if col not in df.columns:
            raise ValueError(f"There is no {col} column in data!")
        work = df[[col] + [c for c in gb if c in df.columns]].copy()
        work = work[work[col].notna()]
        for c in gb:
            if c not in df.columns:
                raise ValueError(f"Column {c!r} is missing in data!")
            if trim_genes:
                work[c] = work[c].map(_first_gene)
        key_cols = list(gb)
        if group_by_seqLength:
            work["__len__"] = work[col].astype(str).str.len()
            key_cols = key_cols + ["__len__"]

        groups: "OrderedDict" = OrderedDict()
        if key_cols:
            for gkey, sub in work.groupby(key_cols, sort=False,
                                          dropna=False):
                gkey = (gkey,) if not isinstance(gkey, tuple) else gkey
                groups[gkey] = _dist_matrix(list(sub[col].astype(str)), fn)
        else:
            groups[("all",)] = _dist_matrix(
                list(work[col].astype(str)), fn)
        result[name] = groups

    result.attrs = {  # type: ignore[attr-defined]
        "col": col, "group_by": gb,
        "group_by_length": group_by_seqLength, "trimmed": trim_genes,
    }
    return result


# --------------------------------------------------------------------------
def _connected_components(labels, dmat: pd.DataFrame,
                          threshold: float) -> "dict":
    """Connected components of the graph of pairs with distance <= threshold."""
    n = len(labels)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    arr = dmat.to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i, j] <= threshold:
                union(i, j)
    comp: "dict[int, int]" = {}
    membership = {}
    for i in range(n):
        r = find(i)
        if r not in comp:
            comp[r] = len(comp) + 1
        membership[labels[i]] = comp[r]
    return membership


def seqCluster(data, dist, perc_similarity=None, nt_similarity=None,
               fixed_threshold: float = 10):
    """Cluster sequences within distance groups by connected components.

    Parameters
    ----------
    data
        The repertoires the distances were computed on (same container as
        passed to :func:`seqDist`).
    dist
        Output of :func:`seqDist`.
    perc_similarity
        Distance threshold expressed as a *fraction* of sequence length:
        the per-group cut-off is ``length * (1 - perc_similarity)``.
    nt_similarity
        Distance threshold expressed as a number of nucleotides: the
        per-group cut-off is ``length / nt_similarity``.
    fixed_threshold
        A constant distance threshold (the immunarch default, ``10``).

    Exactly one of ``perc_similarity`` / ``nt_similarity`` /
    ``fixed_threshold`` should be supplied.

    Returns
    -------
    Same container type as ``data``, with a ``Cluster`` column added to
    every repertoire.
    """
    chosen = [perc_similarity is not None, nt_similarity is not None,
              fixed_threshold is not None]
    if sum(chosen) == 0:
        raise ValueError(
            "Provide .perc_similarity, .nt_similarity or .fixed_threshold."
        )
    if perc_similarity is not None and nt_similarity is not None:
        raise ValueError(
            "Provide only one of .perc_similarity / .nt_similarity / "
            ".fixed_threshold."
        )
    if perc_similarity is not None:
        def thr(length):
            return length * (1.0 - perc_similarity)
    elif nt_similarity is not None:
        def thr(length):
            return length / nt_similarity
    else:
        def thr(length):
            return fixed_threshold

    attrs = getattr(dist, "attrs", {})
    col = attrs.get("col", "CDR3.nt")
    grouping_cols = attrs.get("group_by", [])
    trimmed = attrs.get("trimmed", True)

    if isinstance(data, ImmunData):
        reps = data.data
    elif isinstance(data, dict):
        reps = OrderedDict(data)
    elif isinstance(data, (list, tuple)):
        reps = OrderedDict((f"Sample{i + 1}", d)
                           for i, d in enumerate(data))
    else:
        reps = OrderedDict([("Sample", data)])

    out = OrderedDict()
    for name, df in reps.items():
        groups = dist.get(name, {})
        seq_to_cluster = {}
        for gkey, dmat in groups.items():
            labels = list(dmat.index)
            length = len(labels[0]) if labels else 0
            t = thr(length)
            proto = "/".join(str(v) for v in gkey)
            if len(labels) == 1:
                seq_to_cluster[labels[0]] = f"{proto}_cluster_1"
                continue
            membership = _connected_components(labels, dmat, t)
            for seq, cid in membership.items():
                seq_to_cluster[seq] = f"{proto}_cluster_{cid}"

        new = df.copy()
        keycol = df[col].astype(str)
        new["Cluster"] = keycol.map(seq_to_cluster)
        out[name] = new.reset_index(drop=True)

    if not isinstance(data, (ImmunData, dict, list, tuple)):
        return out[next(iter(out))]
    if isinstance(data, ImmunData):
        return ImmunData(out, data.meta.copy())
    if isinstance(data, dict):
        return out
    return list(out.values())
