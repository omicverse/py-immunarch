"""Information theory + CDR3 amino-acid analysis.

* Information measures — :func:`entropy`, :func:`kl_div`, :func:`js_div`,
  :func:`cross_entropy` (port of immunarch's ``info_theory.R``).
* Position-wise CDR3 amino-acid frequency / property profiles
  (:func:`cdr3_aa_profile`).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from .io import IMMCOL
from .utils import check_distribution

__all__ = [
    "entropy",
    "kl_div",
    "js_div",
    "cross_entropy",
    "cdr3_aa_profile",
    "AA_PROPERTIES",
]


# --------------------------------------------------------------------------
def entropy(data, base: float = 2.0, norm: bool = False,
            do_norm=None, laplace: float = 1e-12) -> float:
    """Shannon entropy of a distribution."""
    p = check_distribution(data, do_norm, laplace)
    res = -np.sum(p * np.log(p) / np.log(base))
    if norm:
        return float(res / (np.log(len(p)) / np.log(base)))
    return float(res)


def kl_div(alpha, beta, base: float = 2.0,
           do_norm=None, laplace: float = 1e-12) -> float:
    """Kullback-Leibler divergence ``D(alpha || beta)``."""
    a = check_distribution(alpha, do_norm, laplace)
    b = check_distribution(beta, do_norm, laplace)
    return float(np.sum(np.log(a / b) / np.log(base) * a))


def js_div(alpha, beta, base: float = 2.0, do_norm=None,
           laplace: float = 1e-12, norm_entropy: bool = False) -> float:
    """Jensen-Shannon divergence between two distributions."""
    a = check_distribution(alpha, do_norm, laplace)
    b = check_distribution(beta, do_norm, laplace)
    if norm_entropy:
        nrm = 0.5 * (entropy(a, base, False, do_norm, laplace)
                     + entropy(b, base, False, do_norm, laplace))
    else:
        nrm = 1.0
    m = (a + b) / 2.0
    return float(0.5 * (kl_div(a, m, base, False)
                        + kl_div(b, m, base, False)) / nrm)


def cross_entropy(alpha, beta, base: float = 2.0, do_norm=None,
                  laplace: float = 1e-12) -> float:
    """Cross-entropy ``H(alpha, beta)``."""
    a = check_distribution(alpha, do_norm, laplace)
    b = check_distribution(beta, do_norm, laplace)
    return float(-np.sum(np.log(b) / np.log(base) * a))


# --------------------------------------------------------------------------
# Kidera-style amino-acid physico-chemical properties (subset used for the
# CDR3 property profile).  Values are standard hydropathy / volume / charge.
AA_PROPERTIES = {
    # Kyte-Doolittle hydropathy
    "hydropathy": {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    },
    # residue volume (A^3)
    "volume": {
        "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
        "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
        "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
        "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
    },
    # net charge at neutral pH
    "charge": {
        "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
        "Q": 0, "E": -1, "G": 0, "H": 0.5, "I": 0,
        "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
        "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
    },
}

_AA20 = sorted("ACDEFGHIKLMNPQRSTVWY")


def cdr3_aa_profile(data, property: str = None, max_len: int = None,
                    align: str = "left") -> pd.DataFrame:
    """Position-wise CDR3 amino-acid frequency or property profile.

    Parameters
    ----------
    data
        A single repertoire DataFrame (uses its ``CDR3.aa`` column).
    property
        ``None`` -> return per-position amino-acid frequencies (20 rows).
        Otherwise one of :data:`AA_PROPERTIES` (``"hydropathy"``,
        ``"volume"``, ``"charge"``) -> return the mean property value per
        position (1 row).
    max_len
        Number of positions to profile.  Defaults to the longest CDR3.
    align
        ``"left"`` or ``"right"`` — how sequences of differing lengths are
        aligned to the position grid.

    Returns
    -------
    pandas.DataFrame
        Rows = amino acids (or the property name), columns = positions.
    """
    seqs = [str(s) for s in data[IMMCOL.cdr3aa].dropna()]
    seqs = [s for s in seqs if "*" not in s and "~" not in s]
    if not seqs:
        return pd.DataFrame()
    if max_len is None:
        max_len = max(len(s) for s in seqs)

    def pos_char(s, pos):
        if align == "left":
            return s[pos] if pos < len(s) else None
        # right alignment
        idx = pos - (max_len - len(s))
        return s[idx] if 0 <= idx < len(s) else None

    cols = [f"P{p + 1}" for p in range(max_len)]

    if property is None:
        mat = np.zeros((len(_AA20), max_len))
        row_idx = {aa: i for i, aa in enumerate(_AA20)}
        for s in seqs:
            for p in range(max_len):
                c = pos_char(s, p)
                if c in row_idx:
                    mat[row_idx[c], p] += 1
        # column-normalise to frequencies
        colsum = mat.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = np.where(colsum > 0, mat / colsum, 0.0)
        return pd.DataFrame(mat, index=_AA20, columns=cols)

    if property not in AA_PROPERTIES:
        raise ValueError(f"Unknown property {property!r}. Options: "
                         f"{sorted(AA_PROPERTIES)}")
    table = AA_PROPERTIES[property]
    means = np.full(max_len, np.nan)
    for p in range(max_len):
        vals = []
        for s in seqs:
            c = pos_char(s, p)
            if c in table:
                vals.append(table[c])
        if vals:
            means[p] = np.mean(vals)
    return pd.DataFrame([means], index=[property], columns=cols)
