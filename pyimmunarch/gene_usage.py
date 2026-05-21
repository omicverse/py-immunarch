"""V/D/J gene-segment usage — :func:`geneUsage`, :func:`geneUsageAnalysis`.

Faithful port of immunarch's ``gene_usage.R`` / ``gene_usage_analysis.R``.
"""
from __future__ import annotations

import re
from collections import OrderedDict

import numpy as np
import pandas as pd

from .io import IMMCOL
from .utils import as_repertoire_list

__all__ = ["geneUsage", "geneUsageAnalysis"]


# --------------------------------------------------------------------------
def _return_segments(s: str) -> str:
    """Strip allele suffixes (``*01``) — immunarch's ``return_segments``."""
    return re.sub(r"\*[0-9]+", "", s)


def _return_families(s: str) -> str:
    """Strip allele + sub-family suffixes — ``return_families``."""
    return re.sub(r"\-[0-9]+", "", _return_segments(s))


def _strip_alleles(val) -> str:
    """immunarch's ``add_column_without_alleles`` for a single value."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return val
    s = str(val).replace(", ", ",")
    s = re.sub(r"([*][0-9]*)([(][0-9]*[.,]*[0-9]*[)])", "", s)
    s = s.replace(",", ", ")
    parts = []
    for x in s.split(", "):
        if x not in parts:
            parts.append(x)
    s = ", ".join(parts)
    s = re.sub(r"[*][0-9]*", "", s)
    return s


def _gene_column(gene: str) -> str:
    """Resolve a ``.gene`` string to the V/D/J data column name."""
    which = gene.split(".")[1] if "." in gene else gene
    letter = which[3:4].lower()
    return {"v": IMMCOL.v, "d": IMMCOL.d, "j": IMMCOL.j}[letter]


# --------------------------------------------------------------------------
def _gene_usage_one(df: pd.DataFrame, gene: str, quant, ambig: str,
                    gene_type: str, norm: bool) -> pd.DataFrame:
    """Single-repertoire gene usage table (columns ``Names`` + count)."""
    gene_col = _gene_column(gene)
    series = df[gene_col].map(_strip_alleles)

    if ambig == "inc":
        if gene_type == "segment":
            series = series.map(lambda s: _return_segments(s)
                                if isinstance(s, str) else s)
        elif gene_type == "family":
            series = series.map(lambda s: _return_families(s)
                                if isinstance(s, str) else s)

    work = pd.DataFrame({"Gene": series, "Quant": df[IMMCOL.count].values})
    work = work.dropna(subset=["Gene"])

    if quant is None:
        agg = work.groupby("Gene", sort=True).size()
    else:
        agg = work.groupby("Gene", sort=True)["Quant"].sum()

    res = pd.DataFrame({"Names": agg.index, IMMCOL.count: agg.values})

    if ambig == "exc":
        res = res[~res["Names"].str.contains(",", na=False)]
    elif ambig == "maj":
        res["Names"] = res["Names"].str.split(",").str[0]
        res = res.groupby("Names", as_index=False, sort=True)[
            IMMCOL.count].sum()

    # NOTE: immunarch's geneUsage does *not* pad with the reference gene
    # list — its padding guard ``length(.gene %in% res) < length(.gene)``
    # always evaluates to FALSE (a boolean vector's length equals the
    # number of reference genes).  We faithfully reproduce this: only the
    # genes actually observed in the repertoire appear in the output.
    res = res.sort_values("Names", kind="stable").reset_index(drop=True)
    if norm:
        total = res[IMMCOL.count].sum()
        if total > 0:
            res[IMMCOL.count] = res[IMMCOL.count] / total
    return res


def geneUsage(data, gene: str = "hs.trbv", quant=None,
              ambig: str = "inc", gene_type: str = "segment",
              norm: bool = False) -> pd.DataFrame:
    """Compute V/D/J gene-segment usage.

    Parameters
    ----------
    data
        :class:`ImmunData`, list/dict of repertoires, or a single one.
    gene
        Species + gene string, e.g. ``"hs.trbv"``, ``"hs.trbj"``.
    quant
        ``None`` counts each clonotype once; ``"count"`` weights genes by
        the ``Clones`` abundance of their clonotypes.
    ambig
        Ambiguous-assignment handling: ``"inc"`` (include), ``"exc"``
        (exclude) or ``"maj"`` (take first / major gene).
    gene_type
        ``"segment"``, ``"allele"`` or ``"family"``.
    norm
        If ``True``, return per-sample proportions instead of counts.

    Returns
    -------
    pandas.DataFrame
        Rows = gene segments (column ``Names``), columns = samples.
    """
    reps = as_repertoire_list(data)
    if quant == "count":
        quant_arg = "count"
    elif quant in (None, "id"):
        quant_arg = None
    else:
        quant_arg = quant

    if len(reps) == 1:
        name = next(iter(reps))
        res = _gene_usage_one(reps[name], gene, quant_arg, ambig,
                              gene_type, norm)
        res.columns = ["Names", name]
        res.attrs["immunr"] = "gene_usage"
        return res

    tables = OrderedDict()
    for name, df in reps.items():
        t = _gene_usage_one(df, gene, quant_arg, ambig, gene_type, norm)
        t.columns = ["Names", name]
        tables[name] = t

    merged = None
    for name, t in tables.items():
        merged = t if merged is None else merged.merge(t, on="Names",
                                                       how="outer")
    merged = merged.sort_values("Names", kind="stable").reset_index(drop=True)
    merged.attrs["immunr"] = "gene_usage"
    return merged


# --------------------------------------------------------------------------
def _laplace_fill(mat: np.ndarray, val: float) -> np.ndarray:
    out = mat.copy().astype(float)
    out[np.isnan(out)] = val
    return out


def _js_div(p, q, base=2):
    """Jensen-Shannon divergence between two distributions."""
    from .utils import check_distribution
    p = check_distribution(p, do_norm=None, laplace=1e-12)
    q = check_distribution(q, do_norm=None, laplace=1e-12)
    m = (p + q) / 2.0

    def kl(a, b):
        return np.sum(np.log(a / b) / np.log(base) * a)

    return 0.5 * (kl(p, m) + kl(q, m))


def geneUsageAnalysis(data: pd.DataFrame, method: str = "js",
                      cor: str = "pearson", base: float = 2.0) -> pd.DataFrame:
    """Post-analysis of a gene-usage table: similarity / divergence matrices.

    Parameters
    ----------
    data
        Output of :func:`geneUsage` (column ``Names`` + per-sample columns).
    method
        ``"js"`` (Jensen-Shannon divergence), ``"cor"`` (correlation),
        ``"cosine"``, ``"pca"`` or ``"mds"``.
    cor
        Correlation type for ``method="cor"``.
    base
        Logarithm base for ``method="js"``.

    Returns
    -------
    pandas.DataFrame
        Square sample-by-sample matrix (or PCA/MDS coordinates).
    """
    samples = [c for c in data.columns if c != "Names"]
    mat = data[samples].to_numpy(dtype=float)
    mat = _laplace_fill(mat, 1e-12)
    n = len(samples)

    if method == "js":
        out = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    out[i, j] = np.nan
                else:
                    out[i, j] = _js_div(mat[:, i], mat[:, j], base)
                out[j, i] = out[i, j]
        return pd.DataFrame(out, index=samples, columns=samples)

    if method == "cor":
        df = pd.DataFrame(mat, columns=samples)
        out = df.corr(method=cor).to_numpy()
        np.fill_diagonal(out, np.nan)
        return pd.DataFrame(out, index=samples, columns=samples)

    if method == "cosine":
        out = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    out[i, j] = np.nan
                else:
                    x, y = mat[:, i], mat[:, j]
                    denom = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))
                    out[i, j] = np.sum(x * y) / denom if denom else 0.0
                out[j, i] = out[i, j]
        return pd.DataFrame(out, index=samples, columns=samples)

    if method == "pca":
        from sklearn.decomposition import PCA
        X = mat.T
        coords = PCA(n_components=min(2, n - 1, X.shape[1])).fit_transform(X)
        cols = [f"PC{i + 1}" for i in range(coords.shape[1])]
        return pd.DataFrame(coords, index=samples, columns=cols)

    if method == "mds":
        from sklearn.manifold import MDS
        X = mat.T
        coords = MDS(n_components=2, random_state=42,
                     normalized_stress="auto").fit_transform(X)
        return pd.DataFrame(coords, index=samples,
                            columns=["DimI", "DimII"])

    raise ValueError(f"Unknown method {method!r}.")
