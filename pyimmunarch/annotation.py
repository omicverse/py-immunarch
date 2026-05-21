"""Immunogenicity-database annotation — :func:`dbLoad`, :func:`dbAnnotate`.

Faithful port of immunarch's ``annotate.R``: load a public immunogenicity
database (VDJdb / McPAS-TCR / PIRD-TBAdb tabular export) and annotate the
clonotypes of a dataset by matching them against database records on the
CDR3 amino-acid sequence and, optionally, the V / J genes.
"""
from __future__ import annotations

import pandas as pd

from .dynamics import trackClonotypes
from .io import ImmunData

__all__ = ["dbLoad", "dbAnnotate"]


# --------------------------------------------------------------------------
def dbLoad(path: str, db: str, species=None, chain=None,
           pathology=None) -> pd.DataFrame:
    """Load an immunogenicity annotation database.

    Parameters
    ----------
    path
        Path to a tabular database file (TSV / CSV, optionally gzipped).
    db
        Database format: ``"vdjdb"``, ``"vdjdb-search"``, ``"mcpas"``
        (alias ``"mcpas-tcr"``) or ``"pird"`` (alias ``"tbadb"``).
    species
        Optional species filter (e.g. ``"HomoSapiens"``); a string or a
        list of strings.  ``None`` keeps every species.
    chain
        Optional chain filter (e.g. ``"TRB"``).
    pathology
        Optional antigen-species / pathology filter.

    Returns
    -------
    pandas.DataFrame
        The (filtered) database with harmonised ``Species`` / ``Chain`` /
        ``Pathology`` columns added, ready for :func:`dbAnnotate`.
    """
    db = str(db).lower()
    if db == "mcpas-tcr":
        db = "mcpas"
    if db == "tbadb":
        db = "pird"
    if db not in ("vdjdb", "vdjdb-search", "mcpas", "pird"):
        raise ValueError(
            "Unknown .db argument. Use 'vdjdb', 'vdjdb-search', 'mcpas' "
            "or 'pird'."
        )

    sep = "," if db == "mcpas" else "\t"
    dbf = pd.read_csv(path, sep=sep, comment="#",
                      compression="gzip" if str(path).endswith(".gz")
                      else "infer")

    if db == "vdjdb":
        if "species" in dbf.columns:
            dbf["Species"] = dbf["species"]
        if "gene" in dbf.columns:
            dbf["Chain"] = dbf["gene"]
        if "antigen.species" in dbf.columns:
            dbf["Pathology"] = dbf["antigen.species"]
    elif db == "vdjdb-search":
        if "Gene" in dbf.columns:
            dbf["Chain"] = dbf["Gene"]
        if "Epitope species" in dbf.columns:
            dbf["Pathology"] = dbf["Epitope species"]
    elif db == "mcpas":
        dbf["Chain"] = "TRB"
        if "Pathology" not in dbf.columns and "Category" in dbf.columns:
            dbf["Pathology"] = dbf["Category"]
    elif db == "pird":
        if "Species" not in dbf.columns and "species" in dbf.columns:
            dbf["Species"] = dbf["species"]
        if "Chain" not in dbf.columns and "Locus" in dbf.columns:
            dbf["Chain"] = dbf["Locus"]
        if "Pathology" not in dbf.columns and "Disease.name" in dbf.columns:
            dbf["Pathology"] = dbf["Disease.name"]

    if species is not None and "Species" in dbf.columns:
        wanted = [species] if isinstance(species, str) else list(species)
        avail = set(dbf["Species"].dropna().unique())
        missing = [s for s in wanted if s not in avail]
        if missing:
            raise ValueError(
                f"Species {missing} not found in the database. Available: "
                f"{sorted(avail)}"
            )
        dbf = dbf[dbf["Species"].isin(wanted)]
    if chain is not None and "Chain" in dbf.columns:
        wanted = [chain] if isinstance(chain, str) else list(chain)
        avail = set(dbf["Chain"].dropna().unique())
        missing = [c for c in wanted if c not in avail]
        if missing:
            raise ValueError(
                f"Chain {missing} not found in the database. Available: "
                f"{sorted(avail)}"
            )
        dbf = dbf[dbf["Chain"].isin(wanted)]
    if pathology is not None and "Pathology" in dbf.columns:
        wanted = [pathology] if isinstance(pathology, str) else list(pathology)
        avail = set(dbf["Pathology"].dropna().unique())
        missing = [p for p in wanted if p not in avail]
        if missing:
            raise ValueError(
                f"Pathology {missing} not found in the database. Available: "
                f"{sorted(avail)}"
            )
        dbf = dbf[dbf["Pathology"].isin(wanted)]

    return dbf.reset_index(drop=True)


# --------------------------------------------------------------------------
def dbAnnotate(data, db: pd.DataFrame, data_col, db_col) -> pd.DataFrame:
    """Annotate repertoire clonotypes against an immunogenicity database.

    For every database record the matching clonotypes are located in each
    repertoire and counted, exactly as immunarch's ``dbAnnotate`` does via
    :func:`trackClonotypes`.

    Parameters
    ----------
    data
        :class:`ImmunData`, list/dict of repertoires or a single repertoire.
    db
        A database table (e.g. the output of :func:`dbLoad`).
    data_col
        Column name(s) in the repertoires used as the match key
        (e.g. ``"CDR3.aa"`` or ``["CDR3.aa", "V.name"]``).
    db_col
        The matching column name(s) in ``db``; same length as ``data_col``.

    Returns
    -------
    pandas.DataFrame
        One row per matched database clonotype: the key column(s), a
        ``Samples`` incidence count and one column of clone counts per
        repertoire.  Sorted by descending incidence.
    """
    data_col = [data_col] if isinstance(data_col, str) else list(data_col)
    db_col = [db_col] if isinstance(db_col, str) else list(db_col)
    if len(data_col) != len(db_col):
        raise ValueError(
            "Number of columns in .data.col and .db.col doesn't match!"
        )

    if isinstance(data, ImmunData):
        reps = data.data
    elif isinstance(data, dict):
        reps = data
    elif isinstance(data, (list, tuple)):
        reps = {f"Sample{i + 1}": d for i, d in enumerate(data)}
    else:
        reps = {"Sample": data}

    first = next(iter(reps.values()))
    missing = [c for c in data_col if c not in first.columns]
    if missing:
        raise ValueError(
            f"Can't find column(s) {missing} in the input data!"
        )
    missing = [c for c in db_col if c not in db.columns]
    if missing:
        raise ValueError(
            f"Can't find column(s) {missing} in the database!"
        )

    # build the clonotype-key table from the database, renamed to data cols
    key = db[db_col].copy()
    key.columns = data_col
    key = key.dropna().drop_duplicates().reset_index(drop=True)

    ann = trackClonotypes(reps, which=key, col="aa", norm=False)

    sample_cols = [c for c in ann.columns if c not in data_col]
    ann["Samples"] = (ann[sample_cols] > 0).sum(axis=1).astype(int)
    ann = ann[ann["Samples"] > 0]
    ann = ann.sort_values("Samples", ascending=False,
                          kind="stable").reset_index(drop=True)
    ann = ann[data_col + ["Samples"] + sample_cols]
    ann.attrs["immunr"] = "db_annotation"
    return ann
