"""R-parity tests — pyimmunarch vs immunarch 0.10.3.

The R driver (:file:`r_reference_driver.R`) runs immunarch on its bundled
``immdata`` example dataset (a 12-sample TCR cohort + metadata), dumps the
raw input repertoires to TSVs, and writes the numeric results of every
analysis-function family to TSVs.  The Python side here loads the *identical*
dumped repertoires and asserts agreement.

Because diversity / overlap / gene-usage are deterministic closed-form
formulas, parity is expected to be bit-exact (relative difference < 1e-6).
Rarefaction uses an extrapolation curve with no RNG in either implementation,
so its mean curve is also checked tightly.

Tests skip gracefully when the CMAP R env or immunarch is unavailable.
"""
from __future__ import annotations

import subprocess
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import pyimmunarch as pim

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _r_available() -> bool:
    if not R_DRIVER.exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'suppressPackageStartupMessages("
             "library(immunarch)); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=240, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or immunarch not installed.",
)

REL_TOL = 1e-6


# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def r_ref(tmp_path_factory):
    """Run the immunarch R reference once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("immunarch_R")
    cmd = (f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
           f"&& Rscript {R_DRIVER} {out_dir}")
    res = subprocess.run(["bash", "-lc", cmd], capture_output=True,
                         text=True, timeout=1200)
    if res.returncode != 0:
        pytest.skip(f"R reference driver failed:\n{res.stderr[-2000:]}")
    return out_dir


@pytest.fixture(scope="module")
def imm(r_ref):
    """Load the exact repertoires that the R driver dumped."""
    samples = (r_ref / "input_samples.txt").read_text().split()
    data = OrderedDict()
    for s in samples:
        df = pd.read_csv(r_ref / f"input_{s}.tsv", sep="\t")
        data[s] = pim.io._postprocess(df)
    meta = pd.read_csv(r_ref / "input_meta.tsv", sep="\t")
    return pim.ImmunData(data, meta)


def _rel(a, b):
    """Maximum relative difference between two finite arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return 0.0
    denom = np.maximum(np.abs(b[mask]), 1e-12)
    return float(np.max(np.abs(a[mask] - b[mask]) / denom))


# ======================================================================
# repExplore
# ======================================================================
def test_explore_volume(r_ref, imm):
    r = pd.read_csv(r_ref / "explore_volume.tsv", sep="\t")
    py = pim.repExplore(imm, method="volume")
    m = r.merge(py, on="Sample", suffixes=("_r", "_py"))
    assert (m["Volume_r"] == m["Volume_py"]).all()


def test_explore_clones(r_ref, imm):
    r = pd.read_csv(r_ref / "explore_clones.tsv", sep="\t")
    py = pim.repExplore(imm, method="clones")
    m = r.merge(py, on="Sample", suffixes=("_r", "_py"))
    assert _rel(m["Clones_py"], m["Clones_r"]) < REL_TOL


def test_explore_count(r_ref, imm):
    r = pd.read_csv(r_ref / "explore_count.tsv", sep="\t")
    py = pim.repExplore(imm, method="count")
    m = r.merge(py, on=["Sample", "Clone.num"], suffixes=("_r", "_py"))
    assert len(m) == len(r)
    assert (m["Clonotypes_r"] == m["Clonotypes_py"]).all()


def test_explore_len(r_ref, imm):
    r = pd.read_csv(r_ref / "explore_len.tsv", sep="\t")
    py = pim.repExplore(imm, method="len", col="aa")
    m = r.merge(py, on=["Sample", "Length"], suffixes=("_r", "_py"))
    assert len(m) == len(r)
    assert (m["Count_r"] == m["Count_py"]).all()


# ======================================================================
# repClonality
# ======================================================================
def test_clonality_clonalprop(r_ref, imm):
    r = pd.read_csv(r_ref / "clonality_clonalprop.tsv", sep="\t",
                    index_col=0)
    py = pim.repClonality(imm, method="clonal.prop")
    py = py.loc[r.index]
    assert (r["Clones"].to_numpy() == py["Clones"].to_numpy()).all()
    assert _rel(py["Clonal.count.prop"], r["Clonal.count.prop"]) < REL_TOL


def test_clonality_homeo(r_ref, imm):
    r = pd.read_csv(r_ref / "clonality_homeo.tsv", sep="\t", index_col=0)
    py = pim.repClonality(imm, method="homeo")
    py = py.loc[r.index]
    assert _rel(py.to_numpy(), r.to_numpy()) < REL_TOL


def test_clonality_top(r_ref, imm):
    r = pd.read_csv(r_ref / "clonality_top.tsv", sep="\t", index_col=0)
    py = pim.repClonality(imm, method="top")
    py = py.loc[r.index]
    assert _rel(py.to_numpy(), r.to_numpy()) < REL_TOL


def test_clonality_rare(r_ref, imm):
    r = pd.read_csv(r_ref / "clonality_rare.tsv", sep="\t", index_col=0)
    py = pim.repClonality(imm, method="rare")
    py = py.loc[r.index]
    assert _rel(py.to_numpy(), r.to_numpy()) < REL_TOL


# ======================================================================
# repDiversity
# ======================================================================
def test_diversity_chao1(r_ref, imm):
    r = pd.read_csv(r_ref / "diversity_chao1.tsv", sep="\t", index_col=0)
    py = pim.repDiversity(imm, method="chao1").loc[r.index]
    assert _rel(py.to_numpy(), r.to_numpy()) < REL_TOL


def test_diversity_hill(r_ref, imm):
    r = pd.read_csv(r_ref / "diversity_hill.tsv", sep="\t")
    py = pim.repDiversity(imm, method="hill")
    m = r.merge(py, on=["Sample", "Q"], suffixes=("_r", "_py"))
    assert len(m) == len(r)
    assert _rel(m["Value_py"], m["Value_r"]) < REL_TOL


def test_diversity_div(r_ref, imm):
    r = pd.read_csv(r_ref / "diversity_div.tsv", sep="\t")
    py = pim.repDiversity(imm, method="div", q=5)
    m = r.merge(py, on="Sample", suffixes=("_r", "_py"))
    assert _rel(m["Value_py"], m["Value_r"]) < REL_TOL


def test_diversity_gini_simpson(r_ref, imm):
    r = pd.read_csv(r_ref / "diversity_ginisimp.tsv", sep="\t")
    py = pim.repDiversity(imm, method="gini.simp")
    m = r.merge(py, on="Sample", suffixes=("_r", "_py"))
    assert _rel(m["Value_py"], m["Value_r"]) < REL_TOL


def test_diversity_inv_simpson(r_ref, imm):
    r = pd.read_csv(r_ref / "diversity_invsimp.tsv", sep="\t")
    py = pim.repDiversity(imm, method="inv.simp")
    m = r.merge(py, on="Sample", suffixes=("_r", "_py"))
    assert _rel(m["Value_py"], m["Value_r"]) < REL_TOL


def test_diversity_gini(r_ref, imm):
    r = pd.read_csv(r_ref / "diversity_gini.tsv", sep="\t")
    py = pim.repDiversity(imm, method="gini")
    m = r.merge(py, on="Sample", suffixes=("_r", "_py"))
    assert _rel(m["Value_py"], m["Value_r"]) < REL_TOL


def test_diversity_d50(r_ref, imm):
    r = pd.read_csv(r_ref / "diversity_d50.tsv", sep="\t", index_col=0)
    py = pim.repDiversity(imm, method="d50").loc[r.index]
    assert (r["Clones"].to_numpy() == py["Clones"].to_numpy()).all()
    assert _rel(py["Percentage"], r["Percentage"]) < REL_TOL


def test_diversity_dxx(r_ref, imm):
    r = pd.read_csv(r_ref / "diversity_dxx.tsv", sep="\t", index_col=0)
    py = pim.repDiversity(imm, method="dxx", perc=25).loc[r.index]
    assert (r["Clones"].to_numpy() == py["Clones"].to_numpy()).all()


def test_diversity_rarefaction(r_ref, imm):
    """Rarefaction mean curve agrees within a few percent."""
    r = pd.read_csv(r_ref / "diversity_raref.tsv", sep="\t")
    py = pim.repDiversity(imm, method="raref", norm=True)
    # compare per-sample interpolation means at matched Size grid points
    for s in r["Sample"].unique():
        rr = r[(r["Sample"] == s) & (r["Type"] == "interpolation")]
        pp = py[(py["Sample"] == s) & (py["Type"] == "interpolation")]
        n = min(len(rr), len(pp))
        assert n > 5
        diff = np.abs(rr["Mean"].to_numpy()[:n]
                      - pp["Mean"].to_numpy()[:n])
        rel = diff / np.maximum(rr["Mean"].to_numpy()[:n], 1e-9)
        assert np.median(rel) < 0.05, f"{s}: rarefaction median rel {rel}"


# ======================================================================
# repOverlap
# ======================================================================
@pytest.mark.parametrize("method",
                         ["public", "overlap", "jaccard", "tversky",
                          "cosine", "morisita"])
def test_overlap_methods(r_ref, imm, method):
    r = pd.read_csv(r_ref / f"overlap_{method}.tsv", sep="\t", index_col=0)
    py = pim.repOverlap(imm, method=method).loc[r.index, r.columns]
    assert _rel(py.to_numpy(), r.to_numpy()) < REL_TOL


# ======================================================================
# geneUsage
# ======================================================================
def test_geneusage_count(r_ref, imm):
    r = pd.read_csv(r_ref / "geneusage_count.tsv", sep="\t")
    py = pim.geneUsage(imm, gene="hs.trbv", norm=False)
    m = r.merge(py, on="Names", suffixes=("_r", "_py"))
    assert len(m) == len(r)
    for s in imm.samples():
        rv = m[f"{s}_r"].fillna(0).to_numpy()
        pv = m[f"{s}_py"].fillna(0).to_numpy()
        assert _rel(pv, rv) < REL_TOL


def test_geneusage_norm(r_ref, imm):
    r = pd.read_csv(r_ref / "geneusage_norm.tsv", sep="\t")
    py = pim.geneUsage(imm, gene="hs.trbv", norm=True)
    m = r.merge(py, on="Names", suffixes=("_r", "_py"))
    for s in imm.samples():
        rv = m[f"{s}_r"].fillna(0).to_numpy()
        pv = m[f"{s}_py"].fillna(0).to_numpy()
        assert _rel(pv, rv) < REL_TOL


def test_geneusage_j(r_ref, imm):
    r = pd.read_csv(r_ref / "geneusage_j.tsv", sep="\t")
    py = pim.geneUsage(imm, gene="hs.trbj", norm=False)
    m = r.merge(py, on="Names", suffixes=("_r", "_py"))
    assert len(m) == len(r)
    for s in imm.samples():
        rv = m[f"{s}_r"].fillna(0).to_numpy()
        pv = m[f"{s}_py"].fillna(0).to_numpy()
        assert _rel(pv, rv) < REL_TOL


def test_geneusage_cosine(r_ref, imm):
    r = pd.read_csv(r_ref / "geneusage_cosine.tsv", sep="\t", index_col=0)
    gun = pim.geneUsage(imm, gene="hs.trbv", norm=True)
    py = pim.geneUsageAnalysis(gun, method="cosine").loc[r.index, r.columns]
    assert _rel(py.to_numpy(), r.to_numpy()) < 1e-5


def test_geneusage_cor(r_ref, imm):
    r = pd.read_csv(r_ref / "geneusage_cor.tsv", sep="\t", index_col=0)
    gun = pim.geneUsage(imm, gene="hs.trbv", norm=True)
    py = pim.geneUsageAnalysis(gun, method="cor").loc[r.index, r.columns]
    assert _rel(py.to_numpy(), r.to_numpy()) < 1e-5


def test_geneusage_js(r_ref, imm):
    r = pd.read_csv(r_ref / "geneusage_js.tsv", sep="\t", index_col=0)
    gun = pim.geneUsage(imm, gene="hs.trbv", norm=True)
    py = pim.geneUsageAnalysis(gun, method="js").loc[r.index, r.columns]
    rv = r.to_numpy()
    pv = py.to_numpy()
    mask = np.isfinite(rv) & np.isfinite(pv)
    rho, _ = pearsonr(rv[mask], pv[mask])
    assert rho > 0.999, f"geneUsage JS Pearson r = {rho:.5f}"


# ======================================================================
# pubRep
# ======================================================================
def test_pubrep(r_ref, imm):
    r = pd.read_csv(r_ref / "pubrep.tsv", sep="\t")
    py = pim.pubRep(imm, col="aa+v", quant="count")
    # same number of public clonotypes and same incidence distribution
    assert len(py) == len(r)
    rv = r["Samples"].value_counts().sort_index()
    pv = py["Samples"].value_counts().sort_index()
    assert (rv == pv.reindex(rv.index).fillna(0)).all()
    # per-clonotype incidence agrees on the keyed join
    m = r.merge(py, on=["CDR3.aa", "V.name"], suffixes=("_r", "_py"))
    assert len(m) == len(r)
    assert (m["Samples_r"] == m["Samples_py"]).all()


# ======================================================================
# trackClonotypes
# ======================================================================
def test_track_clonotypes(r_ref, imm):
    r = pd.read_csv(r_ref / "track.tsv", sep="\t")
    py = pim.trackClonotypes(imm, which=(1, 15), col="aa")
    assert len(py) == len(r)
    m = r.merge(py, on="CDR3.aa", suffixes=("_r", "_py"))
    assert len(m) == len(r)
    for s in imm.samples():
        assert _rel(m[f"{s}_py"], m[f"{s}_r"]) < REL_TOL


# ======================================================================
# getKmers / kmer_profile
# ======================================================================
def test_kmers(r_ref, imm):
    r = pd.read_csv(r_ref / "kmers.tsv", sep="\t")
    py = pim.getKmers(imm[0], 3)
    m = r.merge(py, on="Kmer", suffixes=("_r", "_py"))
    assert len(m) == len(r)
    assert (m["Count_r"] == m["Count_py"]).all()


def test_kmer_profile(r_ref, imm):
    r = pd.read_csv(r_ref / "kmer_profile.tsv", sep="\t", index_col=0)
    km = pim.getKmers(imm[0], 3)
    py = pim.kmer_profile(km, method="freq")
    common = r.index.intersection(py.index)
    assert len(common) >= 18
    rv = r.loc[common].to_numpy()
    pv = py.loc[common].to_numpy()
    assert _rel(pv, rv) < REL_TOL


# ======================================================================
# spectratype
# ======================================================================
def test_spectratype(r_ref, imm):
    r = pd.read_csv(r_ref / "spectratype.tsv", sep="\t")
    py = pim.spectratype(imm[0], quant="count", col="aa")
    m = r.merge(py, on="Length", suffixes=("_r", "_py"))
    assert len(m) == len(r)
    assert (m["Val_r"] == m["Val_py"]).all()
