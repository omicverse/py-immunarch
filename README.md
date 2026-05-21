# py-immunarch

Pure-Python port of the R/CRAN package
**[immunarch](https://cran.r-project.org/package=immunarch)** — bulk and
single-cell immune-repertoire (TCR/BCR AIRR-seq) analytics, by Vadim I.
Nazarov and the ImmunoMind team.

`pyimmunarch` is a standalone, dependency-light re-implementation of
immunarch's **repertoire-analysis core**: data import, exploratory
statistics, clonality, diversity estimators, repertoire overlap, gene
usage, public clonotypes, clonotype tracking, k-mer analysis, filtering
and the standard repertoire plots. It does **not** require R or `rpy2`.

| | |
|---|---|
| PyPI / import name | `pyimmunarch` |
| License | Apache 2.0 (same as upstream immunarch) |
| Upstream | immunarch 0.10.3 |
| Numerical parity | bit-exact vs immunarch (max rel diff ~1e-13) |

## Install

```bash
pip install pyimmunarch          # once published
# or, from a checkout:
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`.

## Data model

`pyimmunarch` follows immunarch's data model: an immune dataset is a list
of per-sample repertoire `DataFrame`s plus a sample-metadata table. The
container class is `ImmunData` (mirroring R's `immdata` list, with
`.data` and `.meta`). Repertoire columns use the immunarch standard
(`Clones`, `Proportion`, `CDR3.nt`, `CDR3.aa`, `V.name`, `D.name`,
`J.name`, ...).

## Quick start

```python
import pyimmunarch as pim

# 1. load repertoire files (AIRR / immunarch / MiXCR / VDJtools / 10x,
#    auto-detected) or the bundled example dataset
imm = pim.repLoad("my_repertoires/")        # directory with metadata.txt
imm = pim.load_example_immdata()            # bundled example TCR cohort

# 2. exploratory statistics
pim.repExplore(imm, method="volume")        # unique clonotypes per sample
pim.repExplore(imm, method="len", col="aa") # CDR3 length distribution

# 3. clonal-space analysis
pim.repClonality(imm, method="homeo")       # clonal space homeostasis
pim.repClonality(imm, method="top")         # top-N clonal proportion

# 4. diversity estimators
pim.repDiversity(imm, method="chao1")       # Chao1 richness
pim.repDiversity(imm, method="hill")        # Hill numbers
pim.repDiversity(imm, method="div", q=5)    # true diversity
pim.repDiversity(imm, method="gini")        # Gini coefficient
pim.repDiversity(imm, method="raref")       # rarefaction curve

# 5. repertoire overlap
ov = pim.repOverlap(imm, method="jaccard")  # public/overlap/jaccard/...
pim.repOverlapAnalysis(ov, method="mds+hclust")

# 6. gene usage
gu = pim.geneUsage(imm, gene="hs.trbv", norm=True)
pim.geneUsageAnalysis(gu, method="js")

# 7. public clonotypes & tracking
pr = pim.pubRep(imm, col="aa+v")
tc = pim.trackClonotypes(imm, which=(1, 15), col="aa")

# 8. k-mers, filtering, plotting
km  = pim.getKmers(imm[0], 3)
sub = pim.repFilter(imm, "by.meta", {"Status": pim.include("MS")})
pim.vis_overlap_heatmap(ov)
```

## Ported API

| immunarch family | `pyimmunarch` |
|---|---|
| I/O | `repLoad`, `load_example_immdata`, `ImmunData`, `IMMCOL` |
| Exploratory | `repExplore` (`volume` / `count` / `len` / `clones`) |
| Clonality | `repClonality` (`clonal.prop` / `homeo` / `top` / `rare`) |
| Diversity | `repDiversity` (`chao1`, `hill`, `div`, `gini.simp`, `inv.simp`, `gini`, `d50`, `dxx`, `raref`) |
| Overlap | `repOverlap` (`public` / `overlap` / `jaccard` / `tversky` / `cosine` / `morisita`), `repOverlapAnalysis` |
| Gene usage | `geneUsage`, `geneUsageAnalysis` |
| Public clonotypes | `pubRep`, `public_matrix`, `pubRepStatistics`, `pubRepFilter`, `pubRepApply` |
| Dynamics | `trackClonotypes` |
| K-mers | `getKmers`, `split_to_kmers`, `kmer_profile`, `spectratype` |
| Filtering | `repFilter`, `include`, `exclude`, `lessthan`, `morethan`, `interval` |
| Information theory | `entropy`, `kl_div`, `js_div`, `cross_entropy` |
| CDR3 analysis | `cdr3_aa_profile` |
| Preprocessing | `coding`, `noncoding`, `inframes`, `outofframes`, `top`, `bunch_translate` |
| Plotting | `vis_diversity`, `vis_overlap_heatmap`, `vis_gene_usage`, `vis_clonal_space`, `vis_tracking`, `vis_spectratype`, `vis_explore` |

## R parity

`pyimmunarch` is validated against **immunarch 0.10.3** on immunarch's own
bundled `immdata` example dataset (a 12-sample TCR cohort). The diversity,
overlap and gene-usage numbers are deterministic closed-form formulas, so
agreement is **bit-exact** (maximum relative difference ~1e-13). The
test suite (`tests/test_r_parity.py`) re-runs immunarch via `Rscript` and
asserts `rel-diff < 1e-6` for every function family; it skips gracefully
when R / immunarch is unavailable.

On the example dataset the Python pipeline runs ~60x faster than the R
pipeline (which includes `Rscript` startup).

```bash
python examples/benchmark.py --runs 3
pytest tests/ -q
```

See `examples/compare_R_vs_Python.ipynb` for a full R-vs-Python
comparison (timing, accuracy table, scatter plots, diversity bar and
overlap heatmap).

## License

Apache License 2.0 — the same license as the upstream immunarch package.
See `LICENSE`.
