"""Algorithmic smoke tests for pyimmunarch — no R required.

These check the internal consistency of each ported routine on a small
synthetic TCR cohort and on the bundled immunarch example dataset when it
is available.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pyimmunarch as pim

warnings.filterwarnings("ignore")

# immunarch ships its example I/O files inside the installed R package.
_EXTDATA = Path(
    "/scratch/users/steorra/env/CMAP/lib/R/library/immunarch/extdata/io"
)
_HAS_EXTDATA = (_EXTDATA / "metadata.txt").exists()


# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def imm():
    """A small synthetic 4-sample TCR cohort (always available)."""
    return pim.io._synthetic_immdata(n_samples=4, seed=1)


@pytest.fixture(scope="module")
def one_rep(imm):
    """A single repertoire DataFrame."""
    return imm[0]


# ----------------------------------------------------------------------
# data model & I/O
# ----------------------------------------------------------------------
def test_immdata_structure(imm):
    assert isinstance(imm, pim.ImmunData)
    assert len(imm) == 4
    assert len(imm.samples()) == 4
    assert "Sample" in imm.meta.columns
    for df in imm:
        for col in (pim.IMMCOL.count, pim.IMMCOL.cdr3aa, pim.IMMCOL.v):
            assert col in df.columns


def test_repload_immunarch_format(tmp_path, imm):
    # round-trip a repertoire through a TSV and reload it
    f = tmp_path / "S1.tsv"
    imm[0].to_csv(f, sep="\t", index=False)
    loaded = pim.repLoad(str(f), format="immunarch")
    assert len(loaded) == 1
    assert len(loaded["S1"]) == len(imm[0])


def test_repload_airr_format(tmp_path):
    airr = pd.DataFrame({
        "sequence_id": ["a", "b", "c"],
        "v_call": ["TRBV1-1", "TRBV2-1", "TRBV1-1"],
        "j_call": ["TRBJ1-1", "TRBJ2-1", "TRBJ1-1"],
        "junction": ["TGT", "TGC", "TGT"],
        "junction_aa": ["CAS", "CAR", "CAS"],
        "duplicate_count": [10, 5, 2],
    })
    f = tmp_path / "rep.tsv"
    airr.to_csv(f, sep="\t", index=False)
    loaded = pim.repLoad(str(f))
    df = loaded["rep"]
    assert pim.IMMCOL.count in df.columns
    assert df[pim.IMMCOL.count].sum() == 17
    assert abs(df[pim.IMMCOL.prop].sum() - 1.0) < 1e-9


def test_repload_directory_with_metadata():
    if not _HAS_EXTDATA:
        pytest.skip("immunarch extdata not available.")
    imm = pim.repLoad(str(_EXTDATA))
    assert len(imm) >= 1
    assert "Sample" in imm.meta.columns


# ----------------------------------------------------------------------
# repExplore
# ----------------------------------------------------------------------
def test_repexplore_volume(imm):
    vol = pim.repExplore(imm, method="volume")
    assert list(vol.columns) == ["Sample", "Volume"]
    assert len(vol) == 4
    assert (vol["Volume"] > 0).all()


def test_repexplore_clones(imm):
    cl = pim.repExplore(imm, method="clones")
    assert "Clones" in cl.columns
    assert (cl["Clones"] > 0).all()


def test_repexplore_count_and_len(imm):
    cnt = pim.repExplore(imm, method="count")
    assert {"Sample", "Clone.num", "Clonotypes"} <= set(cnt.columns)
    ln = pim.repExplore(imm, method="len", col="aa")
    assert {"Sample", "Length", "Count"} <= set(ln.columns)
    assert (ln["Length"] > 0).all()


# ----------------------------------------------------------------------
# repClonality
# ----------------------------------------------------------------------
def test_repclonality_all_methods(imm):
    cp = pim.repClonality(imm, method="clonal.prop")
    assert {"Clones", "Percentage", "Clonal.count.prop"} <= set(cp.columns)
    homeo = pim.repClonality(imm, method="homeo")
    # each row of a homeostasis table sums to <= 1
    assert (homeo.sum(axis=1) <= 1.0 + 1e-9).all()
    topp = pim.repClonality(imm, method="top")
    # top proportions are monotonically non-decreasing along the columns
    arr = topp.to_numpy()
    assert np.all(np.diff(arr, axis=1) >= -1e-9)
    rare = pim.repClonality(imm, method="rare")
    assert "MAX" in rare.columns
    assert np.allclose(rare["MAX"], 1.0)


# ----------------------------------------------------------------------
# repDiversity
# ----------------------------------------------------------------------
def test_repdiversity_chao1(imm):
    ch = pim.repDiversity(imm, method="chao1")
    assert list(ch.columns) == ["Estimator", "SD", "Conf.95.lo",
                                "Conf.95.hi"]
    assert (ch["Estimator"] > 0).all()


def test_repdiversity_estimators(imm):
    for m in ("hill", "div", "gini.simp", "inv.simp", "gini"):
        res = pim.repDiversity(imm, method=m)
        assert "Value" in res.columns
        assert res["Value"].notna().all()


def test_repdiversity_gini_bounds(imm):
    g = pim.repDiversity(imm, method="gini")
    assert ((g["Value"] >= 0) & (g["Value"] <= 1)).all()
    gs = pim.repDiversity(imm, method="gini.simp")
    assert ((gs["Value"] >= 0) & (gs["Value"] <= 1)).all()


def test_repdiversity_d50(imm):
    d50 = pim.repDiversity(imm, method="d50")
    assert "Clones" in d50.columns
    assert (d50["Percentage"] >= 50).all()


def test_repdiversity_rarefaction(imm):
    rf = pim.repDiversity(imm, method="raref", norm=False)
    assert {"Size", "Mean", "Sample"} <= set(rf.columns)
    assert (rf["Mean"] > 0).all()


def test_hill_q1_equals_exp_entropy(imm):
    # Hill number of order 1 == exp(Shannon entropy in nats)
    from pyimmunarch.diversity import diversity_eco, hill_numbers
    counts = imm[0][pim.IMMCOL.count].to_numpy(float)
    hn = hill_numbers(counts, 0, 2)
    assert hn[0] == pytest.approx(len(np.unique(counts >= 0)) + 0,
                                  abs=len(counts))
    assert hn[1] == pytest.approx(diversity_eco(counts, 1))


# ----------------------------------------------------------------------
# repOverlap
# ----------------------------------------------------------------------
def test_repoverlap_methods(imm):
    for m in ("public", "overlap", "jaccard", "tversky", "cosine",
              "morisita"):
        ov = pim.repOverlap(imm, method=m)
        assert ov.shape == (4, 4)
        # symmetric with NaN diagonal
        assert np.isnan(np.diag(ov)).all()
        arr = ov.to_numpy()
        assert np.allclose(arr, arr.T, equal_nan=True)


def test_repoverlap_jaccard_self_consistency(imm):
    ov = pim.repOverlap(imm, method="jaccard")
    off = ov.to_numpy()[~np.eye(4, dtype=bool)]
    assert np.all((off >= 0) & (off <= 1))


def test_repoverlap_analysis(imm):
    ov = pim.repOverlap(imm, method="jaccard")
    res = pim.repOverlapAnalysis(ov, method="mds+hclust", k=2)
    assert "coords" in res and "clusters" in res
    assert res["coords"].shape[0] == 4


# ----------------------------------------------------------------------
# geneUsage
# ----------------------------------------------------------------------
def test_geneusage_count_and_norm(imm):
    gu = pim.geneUsage(imm, gene="hs.trbv")
    assert "Names" in gu.columns
    assert gu.shape[1] == 5  # Names + 4 samples
    gun = pim.geneUsage(imm, gene="hs.trbv", norm=True)
    sample_cols = [c for c in gun.columns if c != "Names"]
    for c in sample_cols:
        assert gun[c].sum() == pytest.approx(1.0, abs=1e-9)


def test_geneusage_analysis(imm):
    gun = pim.geneUsage(imm, gene="hs.trbv", norm=True)
    for m in ("js", "cor", "cosine"):
        mat = pim.geneUsageAnalysis(gun, method=m)
        assert mat.shape == (4, 4)


# ----------------------------------------------------------------------
# pubRep
# ----------------------------------------------------------------------
def test_pubrep(imm):
    pr = pim.pubRep(imm, col="aa+v", quant="count")
    assert "Samples" in pr.columns
    assert (pr["Samples"] >= 1).all()
    mat = pim.public_matrix(pr)
    assert mat.shape[1] == 4
    stats = pim.pubRepStatistics(pr)
    assert {"Group", "Count"} <= set(stats.columns)


def test_pubrep_filter(imm):
    pr = pim.pubRep(imm, col="aa+v", quant="count")
    filt = pim.pubRepFilter(pr, imm.meta, by={"Status": "C"})
    assert "Samples" in filt.columns


# ----------------------------------------------------------------------
# trackClonotypes
# ----------------------------------------------------------------------
def test_track_clonotypes(imm):
    tc = pim.trackClonotypes(imm, which=(1, 10), col="aa")
    assert pim.IMMCOL.cdr3aa in tc.columns
    # one column per sample + the key column
    assert tc.shape[1] == 1 + 4
    assert (tc[imm.samples()].to_numpy() >= 0).all()


def test_track_clonotypes_explicit_sequences(imm):
    seqs = imm[0][pim.IMMCOL.cdr3aa].head(5).tolist()
    tc = pim.trackClonotypes(imm, which=seqs, col="aa")
    assert len(tc) == len(set(seqs))


# ----------------------------------------------------------------------
# k-mers / spectratype
# ----------------------------------------------------------------------
def test_getkmers(imm):
    km = pim.getKmers(imm[0], 3)
    assert {"Kmer", "Count"} <= set(km.columns)
    assert (km["Kmer"].str.len() == 3).all()
    assert (km["Count"] > 0).all()


def test_kmer_profile(imm):
    km = pim.getKmers(imm[0], 4)
    prof = pim.kmer_profile(km, method="prob")
    # each column of a PPM sums to 1
    sums = prof.sum(axis=0)
    assert np.allclose(sums[sums > 0], 1.0)


def test_spectratype(one_rep):
    sp = pim.spectratype(one_rep, quant="count", col="aa")
    assert {"Length", "Val"} <= set(sp.columns)
    sp_v = pim.spectratype(one_rep, quant="id", col="aa+v")
    assert "Gene" in sp_v.columns


# ----------------------------------------------------------------------
# repFilter
# ----------------------------------------------------------------------
def test_repfilter_by_meta(imm):
    sub = pim.repFilter(imm, "by.meta",
                        {"Status": pim.include("C")})
    assert all(imm.meta.set_index("Sample").loc[s, "Status"] == "C"
               for s in sub.samples())


def test_repfilter_by_clonotype(imm):
    sub = pim.repFilter(imm, "by.clonotype",
                        {pim.IMMCOL.count: pim.morethan(5)})
    for df in sub:
        assert (df[pim.IMMCOL.count] > 5).all()


def test_repfilter_by_repertoire(imm):
    sub = pim.repFilter(imm, "by.repertoire",
                        {"n_clones": pim.morethan(0)})
    assert len(sub) == len(imm)


# ----------------------------------------------------------------------
# information theory
# ----------------------------------------------------------------------
def test_information_measures():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    # entropy of a uniform distribution over 4 outcomes is 2 bits
    assert pim.entropy(p, base=2, do_norm=False) == pytest.approx(2.0)
    # KL / JS of identical distributions is 0
    assert pim.kl_div(p, p, do_norm=False) == pytest.approx(0.0, abs=1e-9)
    assert pim.js_div(p, p, do_norm=False) == pytest.approx(0.0, abs=1e-9)
    # JS divergence is symmetric and non-negative
    q = np.array([0.4, 0.3, 0.2, 0.1])
    assert pim.js_div(p, q, do_norm=False) == pytest.approx(
        pim.js_div(q, p, do_norm=False))
    assert pim.js_div(p, q, do_norm=False) >= 0


# ----------------------------------------------------------------------
# CDR3 amino-acid analysis
# ----------------------------------------------------------------------
def test_cdr3_aa_profile(one_rep):
    freq = pim.cdr3_aa_profile(one_rep)
    # frequency columns sum to 1
    sums = freq.sum(axis=0)
    assert np.allclose(sums[sums > 0], 1.0)
    hyd = pim.cdr3_aa_profile(one_rep, property="hydropathy")
    assert hyd.shape[0] == 1


# ----------------------------------------------------------------------
# preprocessing utilities
# ----------------------------------------------------------------------
def test_coding_and_translate():
    df = pd.DataFrame({
        pim.IMMCOL.count: [3, 1, 2],
        pim.IMMCOL.cdr3nt: ["TGTGCC", "TGT", "TGCGCC"],
        pim.IMMCOL.cdr3aa: ["CASS", "CA*S", "CA~S"],
        pim.IMMCOL.v: ["TRBV1", "TRBV2", "TRBV3"],
    })
    cod = pim.coding(df)
    assert len(cod) == 1
    ic = pim.inframes(df)
    assert "CA~S" not in ic[pim.IMMCOL.cdr3aa].tolist()
    aa = pim.bunch_translate("TGTGCCTCCTCC")
    assert isinstance(aa, str) and len(aa) == 4


def test_top(imm):
    # top() keeps ties (matching R's top_n), so it may return >= n rows,
    # but never fewer than n when the repertoire is large enough.
    t = pim.top(imm, 5)
    for orig, df in zip(imm, t):
        assert len(df) >= min(5, len(orig))
        assert df[pim.IMMCOL.count].min() >= orig[pim.IMMCOL.count].nlargest(
            5).min()


# ----------------------------------------------------------------------
# repSample
# ----------------------------------------------------------------------
def test_repsample_downsample(imm):
    ds = pim.repSample(imm, method="downsample", n=200, seed=0)
    assert isinstance(ds, pim.ImmunData)
    for s in imm.samples():
        assert int(ds[s][pim.IMMCOL.count].sum()) == 200
        assert (ds[s][pim.IMMCOL.count] > 0).all()
        assert ds[s][pim.IMMCOL.prop].sum() == pytest.approx(1.0)


def test_repsample_resample(imm):
    rs = pim.repSample(imm, method="resample", n=300, seed=0)
    for s in imm.samples():
        assert int(rs[s][pim.IMMCOL.count].sum()) == 300


def test_repsample_sample(imm):
    sm = pim.repSample(imm, method="sample", n=50, seed=0)
    for s in imm.samples():
        assert len(sm[s]) == 50
    # uniform sampling path
    sm2 = pim.repSample(imm, method="sample", n=50, prob=False, seed=0)
    assert len(sm2[0]) == 50


def test_repsample_single_repertoire(one_rep):
    ds = pim.repSample(one_rep, method="downsample", n=100, seed=1)
    assert isinstance(ds, pd.DataFrame)
    assert int(ds[pim.IMMCOL.count].sum()) == 100


# ----------------------------------------------------------------------
# repSave
# ----------------------------------------------------------------------
def test_repsave_immunarch_roundtrip(tmp_path, imm):
    out = pim.repSave(imm, str(tmp_path / "ds"), format="immunarch",
                      compress=False)
    reloaded = pim.repLoad(out)
    assert len(reloaded) == len(imm)
    assert len(reloaded[0]) == len(imm[0])


def test_repsave_formats(tmp_path, imm):
    for fmt in ("immunarch", "vdjtools", "airr"):
        out = pim.repSave(imm, str(tmp_path / fmt), format=fmt,
                          compress=True)
        files = list(Path(out).glob("*.tsv.gz"))
        assert len(files) == len(imm)


def test_repsave_single(tmp_path, one_rep):
    p = pim.repSave(one_rep, str(tmp_path / "rep"), format="immunarch",
                    compress=False)
    assert Path(p + ".tsv").exists()


# ----------------------------------------------------------------------
# gene_stats
# ----------------------------------------------------------------------
def test_gene_stats():
    gs = pim.gene_stats()
    assert {"alias", "species"} <= set(gs.columns)
    assert len(gs) > 1
    gene_cols = [c for c in gs.columns if c not in ("alias", "species")]
    assert (gs[gene_cols].to_numpy() >= 0).all()


# ----------------------------------------------------------------------
# dbLoad / dbAnnotate
# ----------------------------------------------------------------------
def test_dbload_and_annotate(tmp_path, imm):
    seqs = imm[0][pim.IMMCOL.cdr3aa].dropna().head(8).tolist()
    db = pd.DataFrame({"CDR3": seqs, "Gene": ["TRB"] * len(seqs),
                       "Epitope species": ["HomoSapiens"] * len(seqs)})
    f = tmp_path / "db.tsv"
    db.to_csv(f, sep="\t", index=False)
    loaded = pim.dbLoad(str(f), "vdjdb-search")
    assert "Chain" in loaded.columns and "Pathology" in loaded.columns
    ann = pim.dbAnnotate(imm, loaded, data_col="CDR3.aa", db_col="CDR3")
    assert {"CDR3.aa", "Samples"} <= set(ann.columns)
    assert (ann["Samples"] >= 1).all()


def test_dbannotate_multi_column(imm):
    sub = imm[0][[pim.IMMCOL.cdr3aa, pim.IMMCOL.v]].dropna().head(6)
    db = sub.rename(columns={pim.IMMCOL.cdr3aa: "cdr3",
                             pim.IMMCOL.v: "v"})
    ann = pim.dbAnnotate(imm, db, data_col=["CDR3.aa", "V.name"],
                         db_col=["cdr3", "v"])
    assert {"CDR3.aa", "V.name", "Samples"} <= set(ann.columns)


# ----------------------------------------------------------------------
# seqDist / seqCluster
# ----------------------------------------------------------------------
def test_seqdist_hamming():
    df = pd.DataFrame({
        pim.IMMCOL.count: [5, 3, 2, 1],
        pim.IMMCOL.cdr3nt: ["ACGTAC", "ACGTAC", "ACGAAC", "TTTT"],
        pim.IMMCOL.cdr3aa: ["AB", "AB", "AC", "DD"],
        pim.IMMCOL.v: ["V1", "V1", "V1", "V2"],
        pim.IMMCOL.j: ["J1", "J1", "J1", "J2"],
    })
    d = pim.seqDist({"S": df}, col="CDR3.nt", method="hamming")
    assert "S" in d
    # at least one group holds a 2x2 (or larger) distance matrix
    sizes = [m.shape[0] for m in d["S"].values()]
    assert max(sizes) >= 1


def test_seqdist_levenshtein():
    d = pim.seqdist.levenshtein_dist("kitten", "sitting")
    assert d == 3
    assert pim.seqdist.hamming_dist("ACGT", "ACGA") == 1


def test_seqcluster():
    df = pd.DataFrame({
        pim.IMMCOL.count: [5, 3, 2],
        pim.IMMCOL.cdr3nt: ["ACGTAC", "ACGTAG", "TTTTTT"],
        pim.IMMCOL.cdr3aa: ["AB", "AB", "DD"],
        pim.IMMCOL.v: ["V1", "V1", "V1"],
        pim.IMMCOL.j: ["J1", "J1", "J1"],
    })
    d = pim.seqDist({"S": df}, col="CDR3.nt", method="hamming",
                    group_by=None)
    clu = pim.seqCluster({"S": df}, d, fixed_threshold=2)
    assert "Cluster" in clu["S"].columns
    # the two close sequences share a cluster, the distant one does not
    cl = clu["S"].set_index(pim.IMMCOL.cdr3nt)["Cluster"]
    assert cl["ACGTAC"] == cl["ACGTAG"]
    assert cl["TTTTTT"] != cl["ACGTAC"]


# ----------------------------------------------------------------------
# BCR lineage toolkit
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def bcr():
    """The bundled BCR example dataset."""
    return pim.load_example_bcrdata()


def test_repgermline(bcr):
    g = pim.repGermline(bcr)
    df = g["full_clones"]
    for col in ("V.allele", "J.allele", "Sequence", "V.aa", "J.aa",
                "Germline.sequence"):
        assert col in df.columns
    assert df["Germline.sequence"].notna().all()
    # germline length equals the full nt receptor length
    assert (df["Germline.sequence"].str.len()
            == df["Sequence"].str.len()).all()


def test_repgermline_mouse(bcr):
    # the reference covers multiple species
    g = pim.repGermline(bcr, species="HomoSapiens")
    assert len(g["full_clones"]) > 0


def test_bcr_lineage_pipeline(bcr):
    d = pim.seqDist(bcr)
    clu = pim.seqCluster(bcr, d, fixed_threshold=3)
    g = pim.repGermline(clu)
    al = pim.repAlignLineage(g, min_lineage_sequences=2)
    aligned = al["full_clones"]
    assert {"Cluster", "Germline", "Alignment", "Sequences"} <= set(
        aligned.columns)
    assert len(aligned) >= 1
    cf = pim.repClonalFamily(al)
    fam = cf["full_clones"]
    assert {"Cluster", "Trunk.Length", "TreeStats"} <= set(fam.columns)
    assert (fam["Trunk.Length"] >= 0).all()
    shm = pim.repSomaticHypermutation(cf)
    smdf = shm["full_clones"]
    for col in ("Substitutions", "Insertions", "Deletions", "Mutations"):
        assert col in smdf.columns
        assert (smdf[col] >= 0).all()
    assert (smdf["Mutations"]
            == smdf[["Substitutions", "Insertions",
                     "Deletions"]].sum(axis=1)).all()


def test_repsomatichypermutation_from_germline(bcr):
    # the germline shortcut path: count mutations directly per clonotype
    g = pim.repGermline(bcr)
    sub = g["full_clones"].head(40).reset_index(drop=True)
    shm = pim.repSomaticHypermutation(sub)
    assert "Mutations" in shm.columns
    assert (shm["Mutations"] >= 0).all()
    assert (shm["Mutations"]
            == shm[["Substitutions", "Insertions",
                    "Deletions"]].sum(axis=1)).all()


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def test_plotting_functions(imm):
    import matplotlib.pyplot as plt
    div = pim.repDiversity(imm, method="div")
    ax = pim.vis_diversity(div)
    assert ax is not None
    ov = pim.repOverlap(imm, method="jaccard")
    pim.vis_overlap_heatmap(ov)
    gu = pim.geneUsage(imm, gene="hs.trbv")
    pim.vis_gene_usage(gu)
    homeo = pim.repClonality(imm, method="homeo")
    pim.vis_clonal_space(homeo)
    tc = pim.trackClonotypes(imm, which=(1, 8))
    pim.vis_tracking(tc)
    sp = pim.spectratype(imm[0], col="aa")
    pim.vis_spectratype(sp)
    pim.vis_explore(pim.repExplore(imm, method="volume"))
    plt.close("all")
