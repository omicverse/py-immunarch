"""Head-to-head speed + accuracy benchmark: R immunarch vs pyimmunarch.

Runs the core repertoire-analysis pipeline -- repExplore, repClonality,
repDiversity, repOverlap, geneUsage -- on immunarch's bundled ``immdata``
example dataset (a 12-sample TCR cohort), so both languages analyse the
identical input that the R reference driver dumps to TSVs.

Reports, per function family:

* wall-clock time (Python via ``time.perf_counter``; R via ``Rscript``).
* accuracy of the Python output vs R: Pearson r / max relative difference
  for the diversity, overlap and gene-usage numbers.

Usage::

    python examples/benchmark.py --runs 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import pyimmunarch as pim

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
WORK = HERE / "compare_out"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"
R_DRIVER = HERE.parent / "tests" / "r_reference_driver.R"


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(a[mask] - b[mask])
                        / np.maximum(np.abs(b[mask]), 1e-12)))


def load_immdata(r_out: Path) -> pim.ImmunData:
    """Load the exact repertoires dumped by the R reference driver."""
    samples = (r_out / "input_samples.txt").read_text().split()
    data = OrderedDict()
    for s in samples:
        df = pd.read_csv(r_out / f"input_{s}.tsv", sep="\t")
        data[s] = pim.io._postprocess(df)
    meta = pd.read_csv(r_out / "input_meta.tsv", sep="\t")
    return pim.ImmunData(data, meta)


def run_python(imm: pim.ImmunData, runs: int):
    """Run the pyimmunarch pipeline ``runs`` times; return mean time + out."""
    elapsed = []
    out = {}
    for _ in range(runs):
        t0 = time.perf_counter()
        out["volume"] = pim.repExplore(imm, method="volume")
        out["clonality"] = pim.repClonality(imm, method="homeo")
        out["chao1"] = pim.repDiversity(imm, method="chao1")
        out["div"] = pim.repDiversity(imm, method="div", q=5)
        out["gini"] = pim.repDiversity(imm, method="gini")
        out["overlap"] = pim.repOverlap(imm, method="jaccard")
        out["geneusage"] = pim.geneUsage(imm, gene="hs.trbv", norm=True)
        elapsed.append(time.perf_counter() - t0)
    return float(np.mean(elapsed)), out


def run_R(out_dir: Path) -> float:
    """Run the immunarch R reference driver once; return wall time."""
    cmd = (f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
           f"&& Rscript {R_DRIVER} {out_dir}")
    t0 = time.perf_counter()
    res = subprocess.run(["bash", "-lc", cmd], capture_output=True,
                         text=True)
    if res.returncode != 0:
        raise RuntimeError(f"R driver failed:\n{res.stderr[-2000:]}")
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    r_out = WORK / "R_out"

    print("Dataset: immunarch bundled immdata (12-sample TCR cohort)")

    print("\n--- R pipeline (single run) ---")
    r_time = run_R(r_out)
    print(f"  R immunarch total   {r_time * 1000:9.1f} ms "
          f"(incl. Rscript startup)")

    imm = load_immdata(r_out)
    n_clono = sum(len(df) for df in imm)
    print(f"  loaded {len(imm)} repertoires, {n_clono} clonotypes")

    print(f"\n--- Python pipeline (mean of {args.runs} runs) ---")
    py_time, out = run_python(imm, args.runs)
    print(f"  pyimmunarch total   {py_time * 1000:9.1f} ms")
    print(f"\nSpeedup (R / Python): {r_time / py_time:.2f}x")

    # ----- accuracy --------------------------------------------------------
    print("\n--- Accuracy (Python vs R) ---")
    summary = {
        "n_repertoires": len(imm),
        "n_clonotypes": n_clono,
        "py_time_ms": py_time * 1000,
        "r_time_ms": r_time * 1000,
        "speedup": r_time / py_time,
    }

    r_vol = pd.read_csv(r_out / "explore_volume.tsv", sep="\t")
    m = r_vol.merge(out["volume"], on="Sample", suffixes=("_r", "_py"))
    vol_ok = bool((m["Volume_r"] == m["Volume_py"]).all())
    print(f"  repExplore volume       exact match = {vol_ok}")
    summary["volume_exact"] = vol_ok

    r_chao = pd.read_csv(r_out / "diversity_chao1.tsv", sep="\t",
                         index_col=0)
    py_chao = out["chao1"].loc[r_chao.index]
    chao_r = _pearson(py_chao.to_numpy().ravel(),
                      r_chao.to_numpy().ravel())
    chao_rel = _rel(py_chao.to_numpy(), r_chao.to_numpy())
    print(f"  repDiversity chao1      Pearson r   = {chao_r:.6f}  "
          f"max rel diff = {chao_rel:.2e}")
    summary["chao1_pearson_r"] = chao_r
    summary["chao1_max_rel"] = chao_rel

    r_div = pd.read_csv(r_out / "diversity_div.tsv", sep="\t")
    m = r_div.merge(out["div"], on="Sample", suffixes=("_r", "_py"))
    div_rel = _rel(m["Value_py"], m["Value_r"])
    print(f"  repDiversity div        max rel diff = {div_rel:.2e}")
    summary["div_max_rel"] = div_rel

    r_gini = pd.read_csv(r_out / "diversity_gini.tsv", sep="\t")
    m = r_gini.merge(out["gini"], on="Sample", suffixes=("_r", "_py"))
    gini_rel = _rel(m["Value_py"], m["Value_r"])
    print(f"  repDiversity gini       max rel diff = {gini_rel:.2e}")
    summary["gini_max_rel"] = gini_rel

    r_ov = pd.read_csv(r_out / "overlap_jaccard.tsv", sep="\t",
                       index_col=0)
    py_ov = out["overlap"].loc[r_ov.index, r_ov.columns]
    ov_r = _pearson(py_ov.to_numpy().ravel(), r_ov.to_numpy().ravel())
    ov_rel = _rel(py_ov.to_numpy(), r_ov.to_numpy())
    print(f"  repOverlap jaccard      Pearson r   = {ov_r:.6f}  "
          f"max rel diff = {ov_rel:.2e}")
    summary["overlap_pearson_r"] = ov_r
    summary["overlap_max_rel"] = ov_rel

    r_gu = pd.read_csv(r_out / "geneusage_norm.tsv", sep="\t")
    m = r_gu.merge(out["geneusage"], on="Names", suffixes=("_r", "_py"))
    gu_vals_r, gu_vals_py = [], []
    for s in imm.samples():
        gu_vals_r.extend(m[f"{s}_r"].fillna(0).tolist())
        gu_vals_py.extend(m[f"{s}_py"].fillna(0).tolist())
    gu_r = _pearson(gu_vals_py, gu_vals_r)
    gu_rel = _rel(gu_vals_py, gu_vals_r)
    print(f"  geneUsage (norm)        Pearson r   = {gu_r:.6f}  "
          f"max rel diff = {gu_rel:.2e}")
    summary["geneusage_pearson_r"] = gu_r
    summary["geneusage_max_rel"] = gu_rel

    (WORK / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFull report -> {WORK / 'summary.json'}")


if __name__ == "__main__":
    main()
