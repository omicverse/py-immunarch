"""Repertoire overlap — :func:`repOverlap`, :func:`repOverlapAnalysis`.

Faithful port of immunarch's ``overlap.R`` / ``overlap_analysis.R``: the
public/overlap/jaccard/tversky/cosine/morisita coefficients plus the
MDS / hierarchical-clustering post-analysis on the overlap matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import IMMCOL
from .utils import as_repertoire_list, process_col_argument

__all__ = ["repOverlap", "repOverlapAnalysis"]


# --------------------------------------------------------------------------
class _Rep:
    """Pre-extracted clonotype keys of a repertoire (immunarch ``select``).

    ``nrow`` is the raw row count (with duplicate clonotype keys, exactly
    as immunarch's ``select(.col)`` keeps them) while ``keys`` is the
    *distinct* key set used by ``dplyr::intersect`` / ``setdiff``.
    """

    __slots__ = ("nrow", "keys")

    def __init__(self, df: pd.DataFrame, cols):
        sub = df[list(cols)].astype(str)
        tuples = list(sub.itertuples(index=False, name=None))
        self.nrow = len(tuples)
        self.keys = set(tuples)


def _key_set(df: pd.DataFrame, cols) -> _Rep:
    return _Rep(df, cols)


def _key_counts(df: pd.DataFrame, cols):
    """Per-clonotype quantity tables (raw rows + grouped sums).

    Returns a tuple ``(raw_df, grouped_dict)``: ``raw_df`` keeps the
    duplicate-key rows that immunarch's ``cosine`` full-join consumes, and
    ``grouped_dict`` maps key -> summed ``Clones`` for ``morisita``.
    """
    sub = df[list(cols) + [IMMCOL.count]].copy()
    for c in cols:
        sub[c] = sub[c].astype(str)
    g = sub.groupby(list(cols), sort=False)[IMMCOL.count].sum()
    if len(cols) == 1:
        grouped = {(str(k),): float(v) for k, v in g.items()}
    else:
        grouped = {tuple(map(str, k)): float(v) for k, v in g.items()}
    return sub, grouped, list(cols)


# --- pairwise coefficients ------------------------------------------------
def _public(a: _Rep, b: _Rep):
    return len(a.keys & b.keys)


def _overlap_coef(a: _Rep, b: _Rep):
    return len(a.keys & b.keys) / min(a.nrow, b.nrow)


def _jaccard(a: _Rep, b: _Rep):
    inter = len(a.keys & b.keys)
    return inter / (a.nrow + b.nrow - inter)


def _tversky(a: _Rep, b: _Rep, alpha=0.5, beta=0.5):
    inter = len(a.keys & b.keys)
    return inter / (alpha * len(a.keys - b.keys)
                    + beta * len(b.keys - a.keys) + inter)


def _cosine(ra, rb):
    """Cosine similarity replicating immunarch's full-join behaviour.

    immunarch joins the two per-clonotype tables (which still carry
    duplicate keys) with a *full outer join*; duplicate keys therefore
    expand into a cartesian product before the dot product is taken.
    """
    raw_a, _, cols = ra
    raw_b, _, _ = rb
    merged = raw_a.merge(raw_b, on=cols, how="outer",
                         suffixes=("_a", "_b"))
    x = merged[f"{IMMCOL.count}_a"].fillna(0.0).to_numpy(dtype=float)
    y = merged[f"{IMMCOL.count}_b"].fillna(0.0).to_numpy(dtype=float)
    denom = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom else 0.0


def _morisita(ra, rb):
    """Morisita-Horn overlap on the grouped (distinct-key) count tables."""
    _, ca, _ = ra
    _, cb, _ = rb
    keys = set(ca) | set(cb)
    x = np.array([ca.get(k, 0.0) for k in keys])
    y = np.array([cb.get(k, 0.0) for k in keys])
    X = x.sum()
    Y = y.sum()
    num = 2 * np.sum(x * y)
    den = (np.sum(x ** 2) / X ** 2 + np.sum(y ** 2) / Y ** 2) * X * Y
    return float(num / den) if den else 0.0


# --------------------------------------------------------------------------
def _apply_symm(items, fn, diag=np.nan):
    """Symmetric pairwise matrix M[i,j] = fn(items[i], items[j])."""
    n = len(items)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j and diag is np.nan:
                mat[i, j] = np.nan
            else:
                mat[i, j] = fn(items[i], items[j])
            mat[j, i] = mat[i, j]
    return mat


# --------------------------------------------------------------------------
def repOverlap(data, method: str = "public", col: str = "aa",
               a: float = 0.5, b: float = 0.5) -> pd.DataFrame:
    """Compute pairwise repertoire overlap.

    Parameters
    ----------
    data
        :class:`ImmunData`, list/dict of repertoires.
    method
        ``"public"`` (= ``"shared"``), ``"overlap"``, ``"jaccard"``,
        ``"tversky"``, ``"cosine"`` or ``"morisita"``.
    col
        ``.col`` string, e.g. ``"aa"`` or ``"aa+v"``.
    a, b
        Alpha / beta parameters for the Tversky index.

    Returns
    -------
    pandas.DataFrame
        Square symmetric overlap matrix (``NaN`` on the diagonal).
    """
    reps = as_repertoire_list(data)
    names = list(reps.keys())
    cols = process_col_argument(col)
    method = method.lower()

    if method in ("cosine", "morisita"):
        items = [_key_counts(df, cols) for df in reps.values()]
        fn = _cosine if method == "cosine" else _morisita
    else:
        items = [_key_set(df, cols) for df in reps.values()]
        if method in ("public", "shared"):
            fn = _public
        elif method == "overlap":
            fn = _overlap_coef
        elif method == "jaccard":
            fn = _jaccard
        elif method == "tversky":
            fn = lambda x, y: _tversky(x, y, a, b)  # noqa: E731
        else:
            raise ValueError(f"Unknown method {method!r}.")

    mat = _apply_symm(items, fn)
    out = pd.DataFrame(mat, index=names, columns=names)
    out.attrs["immunr"] = "ov_matrix"
    out.attrs["method"] = method
    return out


# --------------------------------------------------------------------------
def repOverlapAnalysis(data, method: str = "mds+hclust",
                       k: int = 2) -> dict:
    """Post-analysis of an overlap matrix: MDS / clustering.

    Parameters
    ----------
    data
        An overlap matrix (output of :func:`repOverlap`) or any square
        distance/similarity matrix.
    method
        ``"mds"``, ``"tsne"`` optionally chained with ``"+hclust"`` or
        ``"+kmeans"``, e.g. ``"mds+hclust"``.
    k
        Number of clusters for the clustering step.

    Returns
    -------
    dict
        With keys ``"coords"`` (the embedding) and, when a clustering step
        is requested, ``"clusters"`` (cluster labels per sample).
    """
    steps = method.split("+")
    mat = np.asarray(data, dtype=float)
    # symmetric similarity -> distance: use (max - x) with 0 diagonal
    dmat = mat.copy()
    np.fill_diagonal(dmat, 0.0)
    finite = np.isfinite(dmat)
    if finite.any():
        mx = np.nanmax(dmat[finite])
        dmat = np.where(np.isnan(dmat), mx, mx - dmat)
        np.fill_diagonal(dmat, 0.0)
    else:
        dmat = np.nan_to_num(dmat)

    names = (list(data.index) if isinstance(data, pd.DataFrame)
             else [f"S{i + 1}" for i in range(mat.shape[0])])

    result: dict = {}
    if steps[0] in ("mds", "tsne"):
        from sklearn.manifold import MDS, TSNE
        if steps[0] == "mds":
            emb = MDS(n_components=2, dissimilarity="precomputed",
                      random_state=42, normalized_stress="auto")
            coords = emb.fit_transform(dmat)
        else:
            perp = max(1, min(5, mat.shape[0] - 1))
            emb = TSNE(n_components=2, metric="precomputed", init="random",
                       perplexity=perp, random_state=42)
            coords = emb.fit_transform(dmat)
        result["coords"] = pd.DataFrame(coords, index=names,
                                        columns=["DimI", "DimII"])
    else:
        raise ValueError("First method step must be 'mds' or 'tsne'.")

    if len(steps) > 1:
        clu = steps[-1]
        X = result["coords"].to_numpy()
        if clu == "hclust":
            from scipy.cluster.hierarchy import fcluster, linkage
            Z = linkage(X, method="complete")
            labels = fcluster(Z, t=k, criterion="maxclust")
        elif clu == "kmeans":
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=k, n_init=10,
                            random_state=42).fit_predict(X)
        else:
            raise ValueError(f"Unknown clustering step {clu!r}.")
        result["clusters"] = pd.Series(labels, index=names, name="Cluster")

    return result
