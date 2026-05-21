"""Repertoire data I/O — :func:`repLoad`, the immunarch data model and parsers.

A *repertoire* is a :class:`pandas.DataFrame` with the immunarch-standard
columns (see :data:`IMMCOL`).  An *immune dataset* (the structure returned
by :func:`repLoad`) is an :class:`ImmunData` carrying

* ``.data`` — an ordered ``dict`` mapping sample name -> repertoire DataFrame;
* ``.meta`` — a metadata :class:`pandas.DataFrame` with one row per sample.

Supported input formats (``.format`` argument of :func:`repLoad`):

* ``"airr"``      — AIRR rearrangement TSV.
* ``"immunarch"`` — immunarch tabular TSV / CSV (the standard column names).
* ``"mixcr"``     — MiXCR clones export.
* ``"vdjtools"``  — VDJtools tabular export.
* ``"10x"``       — 10x Genomics ``filtered_contig_annotations.csv``.
* ``None``        — auto-detect from the header.
"""
from __future__ import annotations

import gzip
import io as _io
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd

__all__ = [
    "IMMCOL",
    "ImmunData",
    "repLoad",
    "load_example_immdata",
]


# --------------------------------------------------------------------------
# Standard immunarch column names (immunr_data_format.R / IMMCOL).
# --------------------------------------------------------------------------
class _IMMCOL:
    count = "Clones"
    prop = "Proportion"
    cdr3nt = "CDR3.nt"
    cdr3aa = "CDR3.aa"
    v = "V.name"
    d = "D.name"
    j = "J.name"
    ve = "V.end"
    ds = "D.start"
    de = "D.end"
    js = "J.start"
    vnj = "VJ.ins"
    vnd = "VD.ins"
    dnj = "DJ.ins"
    seq = "Sequence"
    order = [
        count, prop, cdr3nt, cdr3aa, v, d, j,
        ve, ds, de, js, vnj, vnd, dnj, seq,
    ]


IMMCOL = _IMMCOL()


# --------------------------------------------------------------------------
class ImmunData:
    """An immunarch immune dataset: repertoires + sample metadata.

    Mirrors the ``immdata`` list (``$data`` + ``$meta``) used throughout
    immunarch.  Access repertoires with :meth:`__getitem__` (by name or
    integer index), iterate sample names with :meth:`samples`.
    """

    def __init__(self, data: "OrderedDict[str, pd.DataFrame]",
                 meta: Optional[pd.DataFrame] = None):
        self.data: "OrderedDict[str, pd.DataFrame]" = OrderedDict(data)
        if meta is None:
            meta = pd.DataFrame({"Sample": list(self.data.keys())})
        self.meta = meta.reset_index(drop=True)

    # -- container protocol -------------------------------------------------
    def samples(self) -> List[str]:
        """Ordered list of sample names."""
        return list(self.data.keys())

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data.values())

    def __getitem__(self, key: Union[str, int]) -> pd.DataFrame:
        if isinstance(key, int):
            return list(self.data.values())[key]
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = len(self.data)
        sizes = [len(df) for df in self.data.values()]
        return (f"ImmunData({n} repertoires, "
                f"{sum(sizes)} clonotypes total)")

    def copy(self) -> "ImmunData":
        return ImmunData(
            OrderedDict((k, v.copy()) for k, v in self.data.items()),
            self.meta.copy(),
        )


# --------------------------------------------------------------------------
def _open_text(path: str):
    """Open a possibly gzip-compressed text file."""
    if str(path).endswith(".gz"):
        return _io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _read_table(path: str, sep: Optional[str] = None) -> pd.DataFrame:
    """Read a delimited table, auto-sniffing the separator if needed."""
    with _open_text(path) as fh:
        first = fh.readline()
    if sep is None:
        sep = "\t" if "\t" in first else ","
    return pd.read_csv(path, sep=sep, comment="#",
                       compression="gzip" if str(path).endswith(".gz")
                       else "infer")


# --------------------------------------------------------------------------
def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns, fill missing standard columns, compute Proportion."""
    df = df.copy()
    if IMMCOL.count in df.columns:
        total = df[IMMCOL.count].sum()
        if total > 0 and (IMMCOL.prop not in df.columns
                          or df[IMMCOL.prop].isna().all()):
            df[IMMCOL.prop] = df[IMMCOL.count] / total
    for col in IMMCOL.order:
        if col not in df.columns:
            df[col] = pd.NA
    extra = [c for c in df.columns if c not in IMMCOL.order]
    df = df[IMMCOL.order + extra]
    if IMMCOL.count in df.columns:
        df = df.sort_values(IMMCOL.count, ascending=False,
                            kind="stable").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Format parsers.
# --------------------------------------------------------------------------
_AIRR_RENAME = {
    "duplicate_count": IMMCOL.count,
    "junction": IMMCOL.cdr3nt,
    "junction_aa": IMMCOL.cdr3aa,
    "cdr3": IMMCOL.cdr3nt,
    "cdr3_aa": IMMCOL.cdr3aa,
    "v_call": IMMCOL.v,
    "d_call": IMMCOL.d,
    "j_call": IMMCOL.j,
    "v_germline_end": IMMCOL.ve,
    "d_germline_start": IMMCOL.ds,
    "d_germline_end": IMMCOL.de,
    "j_germline_start": IMMCOL.js,
    "np1_length": IMMCOL.vnd,
    "np2_length": IMMCOL.dnj,
    "sequence": IMMCOL.seq,
    "consensus_count": IMMCOL.count,
}


def parse_airr(path: str) -> pd.DataFrame:
    """Parse an AIRR rearrangement TSV into an immunarch repertoire."""
    df = _read_table(path, sep="\t")
    df = df.rename(columns={k: v for k, v in _AIRR_RENAME.items()
                            if k in df.columns})
    if IMMCOL.count not in df.columns:
        df[IMMCOL.count] = 1
    if IMMCOL.seq in df.columns:
        df[IMMCOL.seq] = df[IMMCOL.seq].astype(str).str.replace("N", "",
                                                                regex=False)
    return _postprocess(df)


def parse_immunarch(path: str) -> pd.DataFrame:
    """Parse a native immunarch tabular file (TSV or CSV)."""
    df = _read_table(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",", comment="#")
    return _postprocess(df)


_MIXCR_RENAME = {
    "cloneCount": IMMCOL.count,
    "Clone count": IMMCOL.count,
    "readCount": IMMCOL.count,
    "cloneFraction": IMMCOL.prop,
    "Clone fraction": IMMCOL.prop,
    "nSeqCDR3": IMMCOL.cdr3nt,
    "N. Seq. CDR3": IMMCOL.cdr3nt,
    "aaSeqCDR3": IMMCOL.cdr3aa,
    "AA. Seq. CDR3": IMMCOL.cdr3aa,
    "allVHitsWithScore": IMMCOL.v,
    "All V hits": IMMCOL.v,
    "allDHitsWithScore": IMMCOL.d,
    "All D hits": IMMCOL.d,
    "allJHitsWithScore": IMMCOL.j,
    "All J hits": IMMCOL.j,
}


def _strip_mixcr_score(val):
    """MiXCR gene hits look like ``TRBV5-1*00(1234.5)`` — keep first gene."""
    if pd.isna(val):
        return val
    s = str(val).split(",")[0]
    s = s.split("(")[0].split("*")[0]
    return s


def parse_mixcr(path: str) -> pd.DataFrame:
    """Parse a MiXCR clones export into an immunarch repertoire."""
    df = _read_table(path, sep="\t")
    df = df.rename(columns={k: v for k, v in _MIXCR_RENAME.items()
                            if k in df.columns})
    for col in (IMMCOL.v, IMMCOL.d, IMMCOL.j):
        if col in df.columns:
            df[col] = df[col].map(_strip_mixcr_score)
    return _postprocess(df)


_VDJTOOLS_RENAME = {
    "count": IMMCOL.count,
    "freq": IMMCOL.prop,
    "cdr3nt": IMMCOL.cdr3nt,
    "cdr3aa": IMMCOL.cdr3aa,
    "v": IMMCOL.v,
    "d": IMMCOL.d,
    "j": IMMCOL.j,
    "VEnd": IMMCOL.ve,
    "DStart": IMMCOL.ds,
    "DEnd": IMMCOL.de,
    "JStart": IMMCOL.js,
}


def parse_vdjtools(path: str) -> pd.DataFrame:
    """Parse a VDJtools tabular export into an immunarch repertoire."""
    df = _read_table(path, sep="\t")
    df = df.rename(columns={k: v for k, v in _VDJTOOLS_RENAME.items()
                            if k in df.columns})
    return _postprocess(df)


_10X_RENAME = {
    "umis": IMMCOL.count,
    "reads": IMMCOL.count,
    "cdr3_nt": IMMCOL.cdr3nt,
    "cdr3": IMMCOL.cdr3aa,
    "v_gene": IMMCOL.v,
    "d_gene": IMMCOL.d,
    "j_gene": IMMCOL.j,
}


def parse_10x(path: str) -> pd.DataFrame:
    """Parse a 10x Genomics contig annotation CSV into a repertoire."""
    df = _read_table(path, sep=",")
    df = df.rename(columns={k: v for k, v in _10X_RENAME.items()
                            if k in df.columns})
    return _postprocess(df)


_PARSERS = {
    "airr": parse_airr,
    "immunarch": parse_immunarch,
    "mixcr": parse_mixcr,
    "vdjtools": parse_vdjtools,
    "10x": parse_10x,
}


def _detect_format(path: str) -> str:
    """Sniff the file format from the header columns."""
    header = ""
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                header = line
                break
    cols = set(header.replace(",", "\t").split("\t"))
    if {"junction", "v_call", "sequence_id"} & cols:
        return "airr"
    if {"cloneCount", "nSeqCDR3", "Clone count", "N. Seq. CDR3"} & cols:
        return "mixcr"
    if {"cdr3nt", "cdr3aa"} & cols and "count" in cols:
        return "vdjtools"
    if {"umis", "cdr3_nt", "barcode"} & cols:
        return "10x"
    return "immunarch"


# --------------------------------------------------------------------------
def _load_metadata(meta_path: str) -> pd.DataFrame:
    """Load a metadata table (must contain a ``Sample`` column)."""
    df = _read_table(meta_path)
    if "Sample" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Sample"})
    df["Sample"] = df["Sample"].astype(str)
    return df


def repLoad(path: Union[str, Iterable[str]],
            format: Optional[str] = None,
            metadata: Optional[str] = None) -> ImmunData:
    """Load immune repertoire file(s) into an :class:`ImmunData` dataset.

    Parameters
    ----------
    path
        A single repertoire file, a directory containing repertoire files
        (optionally with a ``metadata.txt``), or a list of file paths.
    format
        One of ``"airr"``, ``"immunarch"``, ``"mixcr"``, ``"vdjtools"``,
        ``"10x"``.  ``None`` (default) auto-detects per file.
    metadata
        Path to an explicit metadata table.  If ``None`` and ``path`` is a
        directory, a ``metadata.txt``/``metadata.tsv`` inside it is used.

    Returns
    -------
    ImmunData
    """
    files: List[str] = []
    meta_path = metadata

    if isinstance(path, (list, tuple)):
        files = list(path)
    elif os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            full = os.path.join(path, fn)
            low = fn.lower()
            if low in ("metadata.txt", "metadata.tsv", "metadata.csv"):
                if meta_path is None:
                    meta_path = full
            elif low.endswith((".tsv", ".csv", ".txt", ".tsv.gz",
                               ".csv.gz", ".txt.gz")):
                files.append(full)
    else:
        files = [path]

    if not files:
        raise FileNotFoundError(f"No repertoire files found at {path!r}")

    data: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for fp in files:
        name = os.path.basename(fp)
        for suf in (".tsv.gz", ".csv.gz", ".txt.gz", ".tsv", ".csv", ".txt"):
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        fmt = format or _detect_format(fp)
        parser = _PARSERS.get(fmt)
        if parser is None:
            raise ValueError(f"Unknown format {fmt!r}. Supported: "
                             f"{sorted(_PARSERS)}")
        data[name] = parser(fp)

    if meta_path is not None and os.path.exists(meta_path):
        meta = _load_metadata(meta_path)
        # keep only samples that were loaded, preserve metadata order
        meta = meta[meta["Sample"].isin(data.keys())].reset_index(drop=True)
    else:
        meta = pd.DataFrame({"Sample": list(data.keys())})

    return ImmunData(data, meta)


# --------------------------------------------------------------------------
def load_example_immdata() -> ImmunData:
    """Load the bundled immunarch example dataset (``immdata``).

    The canonical immunarch ``immdata`` — a 12-sample TCR-beta cohort
    (6 ``MS`` / 6 healthy ``C``) — exported from the R package and shipped
    inside py-immunarch as a parquet, so the function is self-contained
    and needs no R install. Falls back to a small synthetic TCR cohort
    only if the bundled file is missing.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "_data")
    pq = os.path.join(data_dir, "immdata.parquet")
    if os.path.isfile(pq):
        alldf = pd.read_parquet(pq)
        meta = pd.read_parquet(os.path.join(data_dir, "immdata_meta.parquet"))
        data: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        for sample, sub in alldf.groupby("Sample", sort=False):
            data[str(sample)] = (
                sub.drop(columns=["Sample"]).reset_index(drop=True)
            )
        return ImmunData(data, meta)
    return _synthetic_immdata()  # pragma: no cover - defensive fallback


def _synthetic_immdata(n_samples: int = 4, seed: int = 0) -> ImmunData:
    """Generate a small synthetic TCR cohort (fallback example dataset)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    vgenes = [f"TRBV{i}-1" for i in range(1, 11)]
    jgenes = [f"TRBJ{i}-1" for i in range(1, 7)]
    data: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for s in range(n_samples):
        n = rng.integers(200, 400)
        seqs = ["".join(rng.choice(aas, rng.integers(10, 18)))
                for _ in range(n)]
        counts = rng.integers(1, 50, n)
        nt = ["".join(rng.choice(list("ACGT"), len(x) * 3)) for x in seqs]
        df = pd.DataFrame({
            IMMCOL.count: counts,
            IMMCOL.cdr3nt: nt,
            IMMCOL.cdr3aa: seqs,
            IMMCOL.v: rng.choice(vgenes, n),
            IMMCOL.d: pd.NA,
            IMMCOL.j: rng.choice(jgenes, n),
        })
        data[f"Sample{s + 1}"] = _postprocess(df)
    meta = pd.DataFrame({
        "Sample": list(data.keys()),
        "Status": ["C", "C", "MS", "MS"][:n_samples],
    })
    return ImmunData(data, meta)
