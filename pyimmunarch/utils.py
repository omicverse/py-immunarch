"""Shared helpers — column resolution, distribution checks, translation.

These faithfully reproduce immunarch's internal utilities from ``tools.R``
and ``preprocessing.R``: ``process_col_argument``, ``check_distribution``,
``coding``/``noncoding``/``inframes``/``outofframes`` and ``bunch_translate``.
"""
from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from .io import IMMCOL, ImmunData

__all__ = [
    "process_col_argument",
    "check_distribution",
    "coding",
    "noncoding",
    "inframes",
    "outofframes",
    "bunch_translate",
    "top",
    "as_repertoire_list",
]


# --------------------------------------------------------------------------
_TYPE_MAP = {
    "nt": IMMCOL.cdr3nt,
    "nuc": IMMCOL.cdr3nt,
    "aa": IMMCOL.cdr3aa,
    "v": IMMCOL.v,
    "j": IMMCOL.j,
}
_TYPE_ORDER = {"nt": 1, "aa": 2, "v": 3, "j": 4}


def switch_type(t: str) -> str:
    """Resolve a single column identifier (``nt``/``aa``/``v``/``j``)."""
    key = t.lower()
    if key not in _TYPE_MAP:
        raise ValueError(
            f"Unknown column identifier {t!r}. Use 'nt', 'aa', 'v' or 'j'."
        )
    return _TYPE_MAP[key]


def process_col_argument(col: str) -> List[str]:
    """Resolve a ``.col`` string like ``"aa+v"`` to real column names.

    Mirrors immunarch's ``process_col_argument``: identifiers are sorted in
    canonical order (nt, aa, v, j) regardless of the order given.
    """
    parts = col.replace("nuc", "nt").split("+")
    parts = [p.strip() for p in parts if p.strip()]
    parts = sorted(parts, key=lambda p: _TYPE_ORDER.get(p.lower(), 99))
    return [switch_type(p) for p in parts]


# --------------------------------------------------------------------------
def check_distribution(data: Sequence[float], do_norm=None,
                       laplace: float = 1.0, na_val: float = 0.0
                       ) -> np.ndarray:
    """Normalise a numeric vector into a probability distribution.

    Faithful port of immunarch's ``check_distribution`` (``tools.R``):

    * ``do_norm is None`` — normalise only if the sum is not already 1,
      applying the Laplace pseudocount **twice** (matching the R bug/quirk
      ``prop.table(.data + .laplace)`` applied to ``.data + .laplace``).
    * ``do_norm is True`` — always normalise with one Laplace pseudocount.
    * ``do_norm is False`` — return the input untouched.
    """
    arr = np.asarray(data, dtype=float)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)

    if do_norm is None:
        arr = np.where(np.isnan(arr), na_val, arr)
        if arr.sum() != 1:
            arr = arr + laplace
            arr = arr + laplace
            arr = arr / arr.sum()
    elif do_norm:
        arr = np.where(np.isnan(arr), na_val, arr)
        arr = arr + laplace
        arr = arr / arr.sum()
    return arr


# --------------------------------------------------------------------------
def _filter_one(df: pd.DataFrame, pattern: str, invert: bool) -> pd.DataFrame:
    """Drop NA CDR3.aa rows then keep/remove rows matching a regex."""
    aa = df[IMMCOL.cdr3aa]
    df = df[aa.notna()].copy()
    mask = df[IMMCOL.cdr3aa].astype(str).str.contains(pattern, regex=True,
                                                      na=False)
    if invert:
        mask = ~mask
    return df[mask].reset_index(drop=True)


def _apply_each(data, fn):
    """Apply ``fn`` to a repertoire, list/dict of them, or an ImmunData."""
    if isinstance(data, ImmunData):
        out = data.copy()
        for k in out.data:
            out.data[k] = fn(out.data[k])
        return out
    if isinstance(data, dict):
        return {k: fn(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [fn(v) for v in data]
    return fn(data)


def coding(data):
    """Keep only coding clonotypes (no stop codons / frame shifts)."""
    return _apply_each(data, lambda d: _filter_one(d, r"[\*, ~]", invert=True))


def noncoding(data):
    """Keep only non-coding clonotypes (with stop codons / frame shifts)."""
    return _apply_each(data, lambda d: _filter_one(d, r"[\*, ~]", invert=False))


def inframes(data):
    """Keep only in-frame clonotypes (drop sequences with frame shifts)."""
    return _apply_each(data, lambda d: _filter_one(d, r"[~]", invert=True))


def outofframes(data):
    """Keep only out-of-frame clonotypes (frame-shifted sequences)."""
    return _apply_each(data, lambda d: _filter_one(d, r"[~]", invert=False))


# --------------------------------------------------------------------------
def top(data, n: int = 10):
    """Return the ``n`` most abundant clonotypes of each repertoire."""
    def _one(df):
        return df.nlargest(n, IMMCOL.count, keep="all").reset_index(drop=True)
    return _apply_each(data, _one)


# --------------------------------------------------------------------------
_GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def bunch_translate(seq, two_way: bool = True, ignore_n: bool = False):
    """Translate nucleotide CDR3 sequences to amino acids.

    Port of immunarch's ``bunch_translate``: with ``two_way=True`` translates
    from both ends (MiXCR-style), padding the centre with ``N`` codons; codons
    containing ``N`` map to ``~``.  Sequences with ``N`` return ``None`` when
    ``ignore_n=False``.
    """
    single = isinstance(seq, str)
    seqs = [seq] if single else list(seq)
    out = []
    for y in seqs:
        if y is None or (isinstance(y, float) and np.isnan(y)):
            out.append(None)
            continue
        y = str(y).upper()
        if not ignore_n and "N" in y:
            out.append(None)
            continue
        ny = len(y)
        ny3 = ny // 3
        if two_way:
            tmp = "NNN" if ny % 3 != 0 else ""
            left = y[: 3 * ((ny3 // 2) + (ny % 2))]
            right = y[3 * ((ny3 // 2) + (ny3 % 2)) + (ny % 3):]
            y = left + tmp + right
        codons = [y[i:i + 3] for i in range(0, len(y) - 2, 3)]
        aa = "".join(_GENETIC_CODE.get(c, "~") for c in codons)
        out.append(aa)
    return out[0] if single else out


# --------------------------------------------------------------------------
def as_repertoire_list(data) -> "dict[str, pd.DataFrame]":
    """Coerce input into an ordered ``{name: DataFrame}`` mapping."""
    from collections import OrderedDict
    if isinstance(data, ImmunData):
        return data.data
    if isinstance(data, dict):
        return OrderedDict(data)
    if isinstance(data, (list, tuple)):
        return OrderedDict((f"Sample{i + 1}", d)
                           for i, d in enumerate(data))
    return OrderedDict([("Sample", data)])
