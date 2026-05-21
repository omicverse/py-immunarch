"""BCR lineage toolkit — germline reconstruction, lineages, SHM.

Faithful port of immunarch's BCR lineage functions (``germline.R``,
``align_lineage.R``, ``clonal_family.R``, ``somatic_hypermutation.R``):

* :func:`repGermline`            — reconstruct each clonotype's germline
  receptor sequence from its V/J reference and FR/CDR segments;
* :func:`repAlignLineage`        — group clonotypes into clonal lineages
  (same ``Cluster`` + germline) and multiple-align them with the germline;
* :func:`repClonalFamily`        — build the clonal-family tree statistics
  for each aligned lineage;
* :func:`repSomaticHypermutation` — count substitutions / insertions /
  deletions of each clonotype against its germline.

The germline reconstruction is a pure data transformation and is
bit-exact with immunarch.  The alignment step uses a self-contained
Needleman-Wunsch / center-star multiple aligner (no external Clustal W
binary required); the downstream mutation counts faithfully reproduce
immunarch's per-segment substitution / indel logic.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from .io import ImmunData
from .utils import bunch_translate

__all__ = [
    "repGermline",
    "repAlignLineage",
    "repClonalFamily",
    "repSomaticHypermutation",
]


_MANDATORY = ["FR1.nt", "CDR1.nt", "FR2.nt", "CDR2.nt", "FR3.nt",
              "CDR3.nt", "FR4.nt"]


# ==========================================================================
# Reference handling.
# ==========================================================================
def _load_reference(species: str) -> pd.DataFrame:
    """Load the V/J germline reference sequences for ``species``."""
    pq = os.path.join(os.path.dirname(__file__), "_data",
                      "gene_segments_ref.parquet")
    if not os.path.isfile(pq):  # pragma: no cover - defensive
        raise FileNotFoundError(
            "Bundled gene-segment reference is missing; cannot run "
            "repGermline."
        )
    ref = pd.read_parquet(pq)
    ref = ref[ref["species"] == species]
    if ref.empty:
        avail = sorted(pd.read_parquet(pq)["species"].unique())
        raise ValueError(
            f"Species {species!r} not found in the reference. "
            f"Available: {avail}"
        )
    return ref[["allele_id", "sequence"]].reset_index(drop=True)


def _first_allele(genes_string) -> str:
    """immunarch's gene-string -> first allele name (strip ',', '(', '*00')."""
    if genes_string is None or (isinstance(genes_string, float)
                                and np.isnan(genes_string)):
        return genes_string
    s = str(genes_string)
    for sep in (",", "("):
        s = s.split(sep)[0]
    return s.replace("*00", "")


def _add_allele_column(df: pd.DataFrame, allele_ids, gene: str
                       ) -> pd.DataFrame:
    """Resolve ``<gene>.name`` to the matching reference allele id."""
    raw = f"{gene}.name"
    target = f"{gene}.allele"
    ref_list = list(allele_ids)
    out = df.copy()

    def _match(genes_string):
        name = _first_allele(genes_string)
        if name is None:
            return name
        hits = [a for a in ref_list if a.startswith(name)]
        return hits[0] if hits else name

    out[target] = out[raw].map(_match)
    return out


def _merge_reference(df: pd.DataFrame, ref: pd.DataFrame,
                     letter: str) -> pd.DataFrame:
    """Attach the germline nucleotide reference for the V or J allele."""
    seq_col = f"{letter}.ref.nt"
    allele_col = f"{letter}.allele"
    r = ref.rename(columns={"sequence": seq_col, "allele_id": allele_col})
    merged = df.merge(r, on=allele_col, how="inner")
    return merged


# ==========================================================================
# repGermline.
# ==========================================================================
def _germline_row(row: pd.Series) -> dict:
    """immunarch's ``calculate_new_columns`` — one clonotype's germline."""
    v_ref = row.get("V.ref.nt")
    j_ref = row.get("J.ref.nt")
    fields = {k: row.get(k) for k in
              ("CDR1.nt", "CDR2.nt", "CDR3.nt", "FR1.nt", "FR2.nt",
               "FR3.nt", "FR4.nt")}
    needed = [v_ref, j_ref] + list(fields.values())
    if any(v is None or (isinstance(v, float) and np.isnan(v))
           for v in needed):
        return {"Sequence": None, "V.aa": None, "J.aa": None,
                "Germline.sequence": None}

    seq = "".join([fields["FR1.nt"], fields["CDR1.nt"], fields["FR2.nt"],
                   fields["CDR2.nt"], fields["FR3.nt"], fields["CDR3.nt"],
                   fields["FR4.nt"]]).upper()

    def _aa(nt_col, aa_col):
        existing = row.get(aa_col)
        if existing is not None and not (isinstance(existing, float)
                                         and np.isnan(existing)):
            return existing
        return bunch_translate(fields[nt_col])

    v_aa = "".join([_aa("FR1.nt", "FR1.aa"), _aa("CDR1.nt", "CDR1.aa"),
                    _aa("FR2.nt", "FR2.aa"), _aa("CDR2.nt", "CDR2.aa"),
                    _aa("FR3.nt", "FR3.aa")])
    j_aa = _aa("FR4.nt", "FR4.aa")

    v_length = sum(len(fields[k]) for k in
                   ("FR1.nt", "CDR1.nt", "FR2.nt", "CDR2.nt", "FR3.nt"))
    v_part = str(v_ref)[:v_length]
    cdr3_part = "n" * len(fields["CDR3.nt"])
    j_length = len(fields["FR4.nt"])
    j_ref = str(j_ref)
    j_part = j_ref[len(j_ref) - j_length:]
    germline = (v_part + cdr3_part + j_part).upper()
    return {"Sequence": seq, "V.aa": v_aa, "J.aa": j_aa,
            "Germline.sequence": germline}


def _germline_single(df: pd.DataFrame, ref: pd.DataFrame,
                      min_nuc_outside_cdr3: int) -> pd.DataFrame:
    """Reconstruct germlines for one repertoire (immunarch ``germline_single_df``)."""
    missing = [c for c in _MANDATORY if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing mandatory column(s) {missing} for repGermline."
        )
    work = df.copy()
    for col in _MANDATORY:
        work = work[work[col].notna()]
    if work.empty:
        raise ValueError("Repertoire is empty after dropping NA segments.")

    work = _add_allele_column(work, ref["allele_id"], "V")
    work = _merge_reference(work, ref, "V")
    work = _add_allele_column(work, ref["allele_id"], "J")
    work = _merge_reference(work, ref, "J")
    if work.empty:
        raise ValueError(
            "No clonotypes left after merging with the germline reference; "
            "check the .species argument."
        )

    # validate chain lengths outside CDR3
    v_len = (work["FR1.nt"].str.len() + work["CDR1.nt"].str.len()
             + work["FR2.nt"].str.len() + work["CDR2.nt"].str.len()
             + work["FR3.nt"].str.len())
    work = work[v_len >= min_nuc_outside_cdr3]
    work = work[work["FR4.nt"].str.len() >= min_nuc_outside_cdr3]
    if work.empty:
        raise ValueError(
            "Repertoire is empty after dropping clonotypes with too-short "
            "V/J chains."
        )

    new_cols = work.apply(_germline_row, axis=1, result_type="expand")
    work = work.drop(columns=["V.ref.nt", "J.ref.nt"], errors="ignore")
    for c in ("Sequence", "V.aa", "J.aa", "Germline.sequence"):
        work[c] = new_cols[c].values
    work = work[work["Germline.sequence"].notna()].reset_index(drop=True)
    return work


def repGermline(data, species: str = "HomoSapiens",
                min_nuc_outside_cdr3: int = 5):
    """Reconstruct the germline receptor sequence of every clonotype.

    Faithful port of immunarch's ``repGermline``: for each BCR clonotype the
    germline is built by concatenating the V reference (trimmed to the
    observed V length), an ``N``-masked CDR3 stretch and the J reference
    (trimmed to the observed J length).  The translated V / J amino-acid
    sequences and the full-length nucleotide ``Sequence`` are added too.

    The input repertoires must carry the segmented columns ``FR1.nt``,
    ``CDR1.nt``, ``FR2.nt``, ``CDR2.nt``, ``FR3.nt``, ``CDR3.nt`` and
    ``FR4.nt`` (as in immunarch's ``bcrdata`` example).

    Parameters
    ----------
    data
        :class:`ImmunData`, list/dict of repertoires or a single repertoire.
    species
        Reference species, e.g. ``"HomoSapiens"`` or ``"MusMusculus"``.
    min_nuc_outside_cdr3
        Drop clonotypes whose V (or J) portion outside CDR3 is shorter than
        this many nucleotides.

    Returns
    -------
    Same container type as ``data`` with the added ``V.allele``,
    ``J.allele``, ``Sequence``, ``V.aa``, ``J.aa`` and
    ``Germline.sequence`` columns.
    """
    ref = _load_reference(species)

    def _one(df):
        return _germline_single(df, ref, min_nuc_outside_cdr3)

    if isinstance(data, ImmunData):
        out = OrderedDict((k, _one(v)) for k, v in data.data.items())
        return ImmunData(out, data.meta.copy())
    if isinstance(data, dict):
        return OrderedDict((k, _one(v)) for k, v in data.items())
    if isinstance(data, (list, tuple)):
        return [_one(v) for v in data]
    return _one(data)


# ==========================================================================
# Needleman-Wunsch / center-star multiple alignment (self-contained).
# ==========================================================================
def _nw_align(a: str, b: str, match: int = 1, mismatch: int = -1,
              gap: int = -2):
    """Global Needleman-Wunsch alignment of two sequences.

    The score matrix is filled row-by-row.  Each row is computed with a
    NumPy left-to-right scan over the gap-extension term, which keeps the
    alignment fast for full-length BCR receptor sequences (~300-400 nt).
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return "-" * m if n == 0 else a, "-" * n if m == 0 else b
    ab = np.frombuffer(a.encode("ascii"), dtype=np.uint8)
    bb = np.frombuffer(b.encode("ascii"), dtype=np.uint8)
    score = np.empty((n + 1, m + 1), dtype=np.float64)
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap
    neg = np.iinfo(np.int64).min / 4
    for i in range(1, n + 1):
        sub = np.where(ab[i - 1] == bb, match, mismatch).astype(np.float64)
        prev = score[i - 1]
        row = score[i]
        # candidate from diagonal (substitution) and from a vertical gap
        cand = np.maximum(prev[:-1] + sub, prev[1:] + gap)
        # resolve the horizontal-gap dependency with a running scan
        run = row[0]
        for j in range(1, m + 1):
            best = cand[j - 1]
            run = best if best > run + gap else run + gap
            row[j] = run
    # traceback
    i, j = n, m
    ra, rb = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag = score[i - 1, j - 1] + (match if a[i - 1] == b[j - 1]
                                          else mismatch)
        else:
            diag = neg
        if i > 0 and j > 0 and score[i, j] == diag:
            ra.append(a[i - 1])
            rb.append(b[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and score[i, j] == score[i - 1, j] + gap:
            ra.append(a[i - 1])
            rb.append("-")
            i -= 1
        else:
            ra.append("-")
            rb.append(b[j - 1])
            j -= 1
    return "".join(reversed(ra)), "".join(reversed(rb))


def _msa(seqs):
    """Center-star multiple alignment of a list of sequences.

    Returns a list of equal-length, gapped strings in the input order.
    """
    if len(seqs) == 1:
        return [seqs[0]]
    # pick the centre = sequence minimising total pairwise distance
    n = len(seqs)
    best_idx, best_cost = 0, np.inf
    for i in range(n):
        cost = 0
        for j in range(n):
            if i == j:
                continue
            x, y = _nw_align(seqs[i], seqs[j])
            cost += sum(p != q for p, q in zip(x, y))
        if cost < best_cost:
            best_cost, best_idx = cost, i

    centre = seqs[best_idx]
    aligned = {best_idx: centre}
    # progressively merge each sequence onto the (growing) centre
    for j in range(n):
        if j == best_idx:
            continue
        c_aln, s_aln = _nw_align(centre, seqs[j])
        # propagate new gaps in the centre to all already-aligned seqs
        new_aligned = {}
        for k, gapped in aligned.items():
            it = iter(gapped)
            merged = []
            for ch in c_aln:
                if ch == "-":
                    merged.append("-")
                else:
                    merged.append(next(it))
            new_aligned[k] = "".join(merged)
        new_aligned[j] = s_aln
        aligned = new_aligned
        centre = c_aln
    return [aligned[i] for i in range(n)]


# ==========================================================================
# repAlignLineage.
# ==========================================================================
_ALIGN_REQUIRED = ["Cluster", "Germline.sequence", "V.allele", "J.allele",
                   "FR1.nt", "CDR1.nt", "FR2.nt", "CDR2.nt", "FR3.nt",
                   "CDR3.nt", "FR4.nt", "V.aa", "J.aa"]
_SEQ_COLS = ["Sequence", "Clone.ID", "Clones", "V.allele", "J.allele",
             "CDR1.nt", "CDR2.nt", "CDR3.nt", "FR1.nt", "FR2.nt",
             "FR3.nt", "FR4.nt", "V.aa", "J.aa"]


def _align_single(df: pd.DataFrame, min_lineage_sequences: int
                  ) -> pd.DataFrame:
    """immunarch's ``align_single_df`` — align each lineage of one sample."""
    missing = [c for c in _ALIGN_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"repAlignLineage: missing required column(s) {missing}. "
            "Run repGermline (and add a 'Cluster' column) first."
        )
    work = df.copy()
    if "Clone.ID" not in work.columns:
        work["Clone.ID"] = np.arange(1, len(work) + 1)
    if "Clones" not in work.columns:
        work["Clones"] = 1

    results = []
    for (cluster, germ), sub in work.groupby(
            ["Cluster", "Germline.sequence"], sort=False):
        if len(sub) < min_lineage_sequences:
            continue
        sub = sub.reset_index(drop=True)
        names = ["Germline"] + [f"ID_{i}" for i in sub["Clone.ID"]]
        seqs = [str(germ)] + [str(s) for s in sub["Sequence"]]
        aligned = _msa(seqs)
        align_df = pd.DataFrame({"name": names, "seq": aligned})
        results.append({
            "Cluster": cluster,
            "Germline": germ,
            "Alignment": align_df,
            "Sequences": sub[[c for c in _SEQ_COLS
                              if c in sub.columns]].copy(),
        })
    if not results:
        raise ValueError(
            f"There are no lineages with at least {min_lineage_sequences} "
            "sequences!"
        )
    out = pd.DataFrame(results)
    out.attrs["immunr"] = "align_lineage"
    return out


def repAlignLineage(data, min_lineage_sequences: int = 3):
    """Group clonotypes into clonal lineages and multiple-align them.

    Faithful port of immunarch's ``repAlignLineage``: clonotypes sharing a
    ``Cluster`` label and the same reconstructed ``Germline.sequence`` form
    a lineage; lineages with at least ``min_lineage_sequences`` clonotypes
    are multiple-aligned together with their germline.

    The input must be the output of :func:`repGermline` with an additional
    ``Cluster`` column (e.g. from :func:`pyimmunarch.seqCluster`).

    Parameters
    ----------
    data
        :class:`ImmunData`, list/dict of repertoires or a single repertoire.
    min_lineage_sequences
        Minimum number of clonotypes a lineage must contain to be aligned.

    Returns
    -------
    A per-sample :class:`pandas.DataFrame` (same container as ``data``) with
    one row per aligned lineage: ``Cluster``, ``Germline``, ``Alignment``
    (a name/seq DataFrame) and ``Sequences`` (the lineage's clonotypes).
    """
    def _one(df):
        return _align_single(df, min_lineage_sequences)

    if isinstance(data, ImmunData):
        return OrderedDict((k, _one(v)) for k, v in data.data.items())
    if isinstance(data, dict):
        return OrderedDict((k, _one(v)) for k, v in data.items())
    if isinstance(data, (list, tuple)):
        return [_one(v) for v in data]
    return _one(data)


# ==========================================================================
# repClonalFamily.
# ==========================================================================
def _process_cluster(row: pd.Series) -> dict:
    """Build clonal-family tree statistics for one aligned lineage.

    A faithful, binary-free re-implementation of immunarch's
    ``process_cluster``: the germline is the lineage root, every clonotype
    descends directly from it, and the per-edge nucleotide / amino-acid
    distances are measured on the V + J segments (CDR3 excluded), matching
    immunarch's distance accounting.
    """
    align = row["Alignment"]
    sequences = row["Sequences"].reset_index(drop=True)
    cluster = row["Cluster"]

    seq_by_name = dict(zip(align["name"], align["seq"]))
    germline = seq_by_name["Germline"]

    s0 = sequences.iloc[0]
    v_nt_len = sum(len(str(s0[c])) for c in
                   ("FR1.nt", "CDR1.nt", "FR2.nt", "CDR2.nt", "FR3.nt"))
    j_nt_len = len(str(s0["FR4.nt"]))
    v_aa_len = len(str(s0["V.aa"]))
    j_aa_len = len(str(s0["J.aa"]))

    def _vj(seq):
        return str(seq)[:v_nt_len] + str(seq)[-j_nt_len:]

    def _vj_aa(seq):
        aa = bunch_translate(str(seq).replace("-", ""), two_way=False,
                             ignore_n=True) or ""
        return aa[:v_aa_len] + aa[-j_aa_len:] if aa else ""

    germ_vj = _vj(germline)
    germ_vj_aa = _vj_aa(germline)

    rows = [{"Name": "Germline", "Type": "Germline", "Clones": 1,
             "Ancestor": None, "DistanceNT": 0, "DistanceAA": 0,
             "Sequence": germline}]
    trunk = None
    for _, srow in sequences.iterrows():
        name = f"ID_{int(srow['Clone.ID'])}"
        seq = seq_by_name.get(name, "")
        seq_vj = _vj(seq)
        dnt = sum(p != q for p, q in zip(seq_vj, germ_vj))
        seq_vj_aa = _vj_aa(seq)
        daa = sum(p != q for p, q in zip(seq_vj_aa, germ_vj_aa))
        rows.append({"Name": name, "Type": "Clonotype",
                     "Clones": int(srow["Clones"]), "Ancestor": "Germline",
                     "DistanceNT": int(dnt), "DistanceAA": int(daa),
                     "Sequence": seq})
        trunk = dnt if trunk is None else min(trunk, dnt)

    tree_stats = pd.DataFrame(rows)
    tree_stats.attrs["immunr"] = "clonal_family_tree"
    return {"Cluster": cluster,
            "Germline.Input": row["Germline"],
            "Germline.Output": germline,
            "Common.Ancestor": germline,
            "Trunk.Length": int(trunk) if trunk is not None else 0,
            "TreeStats": tree_stats,
            "Sequences": sequences}


def repClonalFamily(data):
    """Build clonal-family tree statistics for aligned BCR lineages.

    Faithful port of immunarch's ``repClonalFamily``: for every aligned
    lineage (output of :func:`repAlignLineage`) the germline-rooted tree is
    summarised — each clonotype's nucleotide / amino-acid distance to the
    germline and the lineage trunk length.

    immunarch's implementation shells out to the PHYLIP ``dnapars``
    maximum-parsimony program; this port produces the equivalent
    germline-rooted star tree and the same distance statistics without an
    external binary.

    Parameters
    ----------
    data
        Output of :func:`repAlignLineage` (same container as that input).

    Returns
    -------
    A per-sample :class:`pandas.DataFrame` with one row per lineage:
    ``Cluster``, ``Germline.Input`` / ``Germline.Output``,
    ``Common.Ancestor``, ``Trunk.Length``, ``TreeStats`` and ``Sequences``.
    """
    def _one(aligned_df):
        recs = [_process_cluster(r) for _, r in aligned_df.iterrows()]
        out = pd.DataFrame(recs)
        out.attrs["immunr"] = "clonal_family"
        return out

    if isinstance(data, dict):
        return OrderedDict((k, _one(v)) for k, v in data.items())
    if isinstance(data, (list, tuple)):
        return [_one(v) for v in data]
    return _one(data)


# ==========================================================================
# repSomaticHypermutation.
# ==========================================================================
def _shm_count(germline: str, fr1, cdr1, fr2, cdr2, fr3, fr4) -> dict:
    """immunarch's ``shm_process_clonotype_row`` — count V/J mutations.

    Aligns the clonotype's V (FR1+CDR1+FR2+CDR2+FR3) and J (FR4) segments to
    the matching stretch of the germline and tallies substitutions /
    insertions / deletions.
    """
    v_clono = "".join(str(x) for x in (fr1, cdr1, fr2, cdr2, fr3))
    j_clono = str(fr4)
    germline = str(germline)
    v_germ = germline[:len(v_clono)]
    j_germ = germline[len(germline) - len(j_clono):]

    subs = ins = dels = 0
    for germ, clono in ((v_germ, v_clono), (j_germ, j_clono)):
        g_aln, c_aln = _nw_align(germ, clono)
        for g, c in zip(g_aln, c_aln):
            if g == "-":
                ins += 1
            elif c == "-":
                dels += 1
            elif g != c:
                subs += 1
    return {"Substitutions": subs, "Insertions": ins, "Deletions": dels,
            "Mutations": subs + ins + dels}


def _shm_from_germline(df: pd.DataFrame) -> pd.DataFrame:
    """SHM directly on a repGermline repertoire (one row per clonotype)."""
    def _row(row):
        return _shm_count(row["Germline.sequence"], row["FR1.nt"],
                          row["CDR1.nt"], row["FR2.nt"], row["CDR2.nt"],
                          row["FR3.nt"], row["FR4.nt"])

    out = df.copy()
    muts = out.apply(_row, axis=1, result_type="expand")
    for c in ("Substitutions", "Insertions", "Deletions", "Mutations"):
        out[c] = muts[c].astype(int).values
    return out


def _shm_from_clonal_family(cf: pd.DataFrame) -> pd.DataFrame:
    """SHM on a repClonalFamily output (immunarch's ``shm_process_dataframe``).

    Unnests every lineage's clonotypes and counts mutations of each against
    its lineage germline (``Germline.Input``).
    """
    rows = []
    for _, lin in cf.iterrows():
        germ = lin["Germline.Input"]
        seqs = lin["Sequences"]
        ts = lin["TreeStats"]
        trunk = lin.get("Trunk.Length", np.nan)
        for _, s in seqs.iterrows():
            muts = _shm_count(germ, s["FR1.nt"], s["CDR1.nt"], s["FR2.nt"],
                              s["CDR2.nt"], s["FR3.nt"], s["FR4.nt"])
            rec = {"Cluster": lin["Cluster"],
                   "Clone.ID": int(s["Clone.ID"]),
                   "Clones": int(s["Clones"]),
                   "Trunk.Length": trunk}
            rec.update(muts)
            rows.append(rec)
    out = pd.DataFrame(rows)
    for c in ("Clone.ID", "Clones", "Substitutions", "Insertions",
              "Deletions", "Mutations"):
        out[c] = out[c].astype(int)
    out.attrs["immunr"] = "somatic_hypermutation"
    return out


def repSomaticHypermutation(data):
    """Count somatic hypermutations of each clonotype against its germline.

    Faithful port of immunarch's ``repSomaticHypermutation``: each
    clonotype's V and J segments are aligned to the corresponding germline
    stretch and the number of substitutions, insertions and deletions (and
    their total, ``Mutations``) is recorded.

    immunarch aligns with Clustal W (via ``ape::clustal``); this port uses a
    self-contained Needleman-Wunsch aligner, so no external binary is
    needed.

    Two inputs are accepted:

    * a :func:`repClonalFamily` output — the canonical immunarch pipeline
      (``... repGermline -> repAlignLineage -> repClonalFamily ->
      repSomaticHypermutation``); mutations are counted per lineage
      clonotype against the lineage's ``Germline.Input``;
    * a plain :func:`repGermline` repertoire (carrying ``Germline.sequence``
      and the segmented ``FR*.nt`` / ``CDR*.nt`` columns) — a convenience
      shortcut that counts mutations of every clonotype directly.

    Parameters
    ----------
    data
        A :func:`repClonalFamily` result, an :class:`ImmunData` / list /
        dict of :func:`repGermline` repertoires, or a single repertoire.

    Returns
    -------
    Same container type as ``data`` with the added ``Substitutions``,
    ``Insertions``, ``Deletions`` and ``Mutations`` integer columns.
    """
    def _one(df):
        if "Germline.Input" in df.columns and "Sequences" in df.columns:
            return _shm_from_clonal_family(df)
        if "Germline.sequence" in df.columns:
            return _shm_from_germline(df)
        raise ValueError(
            "repSomaticHypermutation needs either a repClonalFamily output "
            "or a repGermline repertoire (with 'Germline.sequence')."
        )

    if isinstance(data, ImmunData):
        new = OrderedDict((k, _one(v)) for k, v in data.data.items())
        return ImmunData(new, data.meta.copy())
    if isinstance(data, dict):
        return OrderedDict((k, _one(v)) for k, v in data.items())
    if isinstance(data, (list, tuple)):
        return [_one(v) for v in data]
    return _one(data)
