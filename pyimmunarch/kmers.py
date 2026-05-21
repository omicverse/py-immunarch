"""K-mer analysis — :func:`getKmers`, :func:`kmer_profile`, :func:`spectratype`.

Faithful port of immunarch's ``kmers.R`` and ``spectratyping.R``.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from .io import IMMCOL
from .utils import as_repertoire_list, coding
from .utils import switch_type as _switch

__all__ = ["getKmers", "split_to_kmers", "kmer_profile", "spectratype"]


# 20 standard amino acids + stop (`*`) + frame-shift (`~`); used by
# kmer_profile for the row order (immunarch sorts unique AA_TABLE values).
_AA20 = list("ACDEFGHIKLMNPQRSTVWY")
_AA_ROWS = sorted(_AA20)  # rows after dropping `*` and `~`


def split_to_kmers(seqs, k: int) -> pd.DataFrame:
    """Split a vector of sequences into k-mers and count occurrences.

    Mirrors immunarch's ``split_to_kmers``: every length-``k`` substring at
    every starting position is counted; substrings shorter than ``k`` (from
    the sequence ends) are dropped.
    """
    seqs = [str(s) for s in seqs if s is not None
            and not (isinstance(s, float) and np.isnan(s))]
    max_len = max((len(s) for s in seqs), default=0)
    counter: "OrderedDict[str, int]" = OrderedDict()
    for i in range(max_len - k + 1):
        for s in seqs:
            sub = s[i:i + k]
            if len(sub) == k:
                counter[sub] = counter.get(sub, 0) + 1
    items = sorted(counter.items())
    out = pd.DataFrame(items, columns=["Kmer", "Count"]) if items \
        else pd.DataFrame({"Kmer": [], "Count": []})
    out["Count"] = out["Count"].astype(int)
    out.attrs["immunr"] = "kmers"
    return out


def getKmers(data, k: int, col: str = "aa",
             coding_only: bool = True) -> pd.DataFrame:
    """Compute k-mer occurrence tables for one or more repertoires.

    Parameters
    ----------
    data
        :class:`ImmunData`, list/dict of repertoires, or a single one.
    k
        K-mer length.
    col
        ``"aa"`` (CDR3 amino acids) or ``"nt"`` (CDR3 nucleotides).
    coding_only
        Drop non-coding sequences first if ``True``.

    Returns
    -------
    pandas.DataFrame
        Column ``Kmer`` plus one count column per sample.
    """
    seq_col = _switch(col)
    if coding_only:
        data = coding(data)
    reps = as_repertoire_list(data)

    if len(reps) == 1:
        name = next(iter(reps))
        res = split_to_kmers(reps[name][seq_col].tolist(), k)
        res.attrs["immunr"] = "kmer_table"
        return res

    merged = None
    for name, df in reps.items():
        t = split_to_kmers(df[seq_col].tolist(), k).rename(
            columns={"Count": name})
        merged = t if merged is None else merged.merge(t, on="Kmer",
                                                       how="outer")
    merged.attrs["immunr"] = "kmer_table"
    return merged


def kmer_profile(data, method: str = "freq",
                 remove_stop: bool = True) -> pd.DataFrame:
    """Per-position amino-acid profile of a set of equal-length k-mers.

    Parameters
    ----------
    data
        A :func:`getKmers` output (single sample) or a list of k-mer strings.
    method
        ``"freq"`` (counts / PFM), ``"prob"`` (PPM), ``"wei"`` (PWM,
        log2 of count/n_rows) or ``"self"`` (self-information).
    remove_stop
        Drop k-mers containing ``*`` / ``~`` first.

    Returns
    -------
    pandas.DataFrame
        Amino-acid (rows) by position (columns ``V1``, ``V2``, ...).
    """
    if method not in ("freq", "prob", "wei", "self"):
        raise ValueError("method must be 'freq', 'prob', 'wei' or 'self'.")

    if isinstance(data, pd.DataFrame):
        if data.shape[1] > 2:
            raise ValueError("kmer_profile supports a single-sample table.")
        seqs = list(data["Kmer"])
        cnts = list(data["Count"])
    else:
        seqs = [str(s) for s in data]
        cnts = [1] * len(seqs)

    k = len(seqs[0]) if seqs else 0
    if remove_stop:
        keep = [i for i, s in enumerate(seqs)
                if "*" not in s and "~" not in s]
        seqs = [seqs[i] for i in keep]
        cnts = [cnts[i] for i in keep]

    res = np.zeros((len(_AA_ROWS), k))
    row_idx = {aa: i for i, aa in enumerate(_AA_ROWS)}
    for pos in range(k):
        for s, c in zip(seqs, cnts):
            aa = s[pos]
            if aa in row_idx:
                res[row_idx[aa], pos] += c
        col = res[:, pos]
        if method == "prob":
            res[:, pos] = col / col.sum() if col.sum() else col
        elif method == "wei":
            with np.errstate(divide="ignore"):
                res[:, pos] = np.log2(col / len(_AA_ROWS))
        elif method == "self":
            p = col / col.sum() if col.sum() else col
            with np.errstate(divide="ignore", invalid="ignore"):
                res[:, pos] = -p * np.log2(p)

    out = pd.DataFrame(res, index=_AA_ROWS,
                       columns=[f"V{i + 1}" for i in range(k)])
    out.attrs["immunr"] = f"kmer_profile_{method}"
    return out


# --------------------------------------------------------------------------
def spectratype(data, quant: str = "id", col: str = "nt") -> pd.DataFrame:
    """CDR3-length spectratype of a single repertoire.

    Parameters
    ----------
    data
        A single repertoire DataFrame.
    quant
        ``"id"`` counts each clonotype once; ``"count"`` uses ``Clones``.
    col
        Sequence column: ``"nt"`` or ``"aa"`` (optionally ``"+v"``/``"+j"``).

    Returns
    -------
    pandas.DataFrame
        ``Length`` + ``Val`` (counts), optionally split by gene segment.
    """
    if isinstance(data, dict) or hasattr(data, "data"):
        raise ValueError("spectratype expects a single repertoire.")

    col = col.replace("nuc", "nt")
    has_nt = "nt" in col
    has_aa = "aa" in col
    if has_nt and has_aa:
        raise ValueError("Provide only one sequence column: 'nt' or 'aa'.")
    if not has_nt and not has_aa:
        raise ValueError("Provide one sequence column: 'nt' or 'aa'.")
    has_v = "v" in col
    has_j = "j" in col
    if has_v and has_j:
        raise ValueError("Provide only one gene column: 'v' or 'j'.")

    df = data.copy()
    seq_col = IMMCOL.cdr3nt if has_nt else IMMCOL.cdr3aa
    df = df[df[seq_col].notna()]
    df["Length"] = df[seq_col].astype(str).str.len()
    qcol = "Spec.count" if quant == "id" else IMMCOL.count
    if quant == "id":
        df[qcol] = 1

    if has_v or has_j:
        gene_col = IMMCOL.v if has_v else IMMCOL.j
        agg = df.groupby(["Length", gene_col], dropna=False,
                         sort=False)[qcol].sum().reset_index()
        agg = agg.rename(columns={qcol: "Val", gene_col: "Gene"})
        agg = agg.sort_values("Val", ascending=False,
                              kind="stable").reset_index(drop=True)
        agg.attrs["immunr"] = "spectr"
        return agg

    agg = df.groupby("Length", sort=False)[qcol].sum().reset_index()
    agg = agg.rename(columns={qcol: "Val"})
    agg.attrs["immunr"] = "spectr_nogene"
    return agg
