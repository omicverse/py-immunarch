"""pyimmunarch: Pure-Python port of the R package immunarch.

A standalone, dependency-light Python re-implementation of the bulk +
single-cell immune-repertoire analytics toolkit ``immunarch`` (Nazarov
*et al.*, ImmunoMind).  It covers immunarch's repertoire-analysis API for
TCR/BCR AIRR-seq data, working on the immunarch data model: a list of
per-sample repertoire DataFrames plus sample metadata.

Core data model
---------------
* :class:`ImmunData`  — an immune dataset (``.data`` repertoires +
  ``.meta`` metadata), mirroring R's ``immdata`` list.
* :data:`IMMCOL`      — the standard immunarch column names.

I/O
---
* :func:`repLoad`             — load repertoire file(s) (AIRR / immunarch /
  MiXCR / VDJtools / 10x; auto-detected).
* :func:`load_example_immdata` — the bundled example TCR cohort.

Exploratory statistics
----------------------
* :func:`repExplore`   — clone counts, volume, CDR3 length / count
  distributions.

Clonal-space analysis
---------------------
* :func:`repClonality` — clonal proportion, homeostasis, top-N, rare.

Diversity
---------
* :func:`repDiversity` — Chao1, Hill numbers, true diversity, Gini-Simpson,
  inverse Simpson, Gini, d50/dXX, rarefaction.

Overlap
-------
* :func:`repOverlap`         — public / overlap / jaccard / tversky /
  cosine / morisita matrices.
* :func:`repOverlapAnalysis` — MDS / clustering on the overlap matrix.

Gene usage
----------
* :func:`geneUsage`          — V/D/J segment usage tables.
* :func:`geneUsageAnalysis`  — JS / correlation / cosine / PCA / MDS.

Public clonotypes
-----------------
* :func:`pubRep`, :func:`public_matrix`, :func:`pubRepStatistics`,
  :func:`pubRepFilter`, :func:`pubRepApply`.

Dynamics, k-mers, filtering
---------------------------
* :func:`trackClonotypes`    — abundance of clonotypes across samples.
* :func:`getKmers`, :func:`split_to_kmers`, :func:`kmer_profile`,
  :func:`spectratype`.
* :func:`repFilter` (+ :func:`include` / :func:`exclude` / :func:`lessthan`
  / :func:`morethan` / :func:`interval`).

Information theory & CDR3 analysis
----------------------------------
* :func:`entropy`, :func:`kl_div`, :func:`js_div`, :func:`cross_entropy`.
* :func:`cdr3_aa_profile` — position-wise AA frequency / property profile.

Plotting
--------
* :func:`vis_diversity`, :func:`vis_overlap_heatmap`, :func:`vis_gene_usage`,
  :func:`vis_clonal_space`, :func:`vis_tracking`, :func:`vis_spectratype`,
  :func:`vis_explore`.

Utilities
---------
* :func:`coding`, :func:`noncoding`, :func:`inframes`, :func:`outofframes`,
  :func:`top`, :func:`bunch_translate`.

Quick-start
-----------
>>> import pyimmunarch as pim
>>> imm = pim.load_example_immdata()
>>> pim.repExplore(imm, method="volume")
>>> pim.repDiversity(imm, method="chao1")
>>> ov = pim.repOverlap(imm, method="jaccard")
>>> gu = pim.geneUsage(imm, gene="hs.trbv", norm=True)
"""
from __future__ import annotations

from .analysis import (
    AA_PROPERTIES,
    cdr3_aa_profile,
    cross_entropy,
    entropy,
    js_div,
    kl_div,
)
from .clonality import repClonality
from .diversity import repDiversity
from .dynamics import trackClonotypes
from .explore import repExplore
from .filters import (
    exclude,
    include,
    interval,
    lessthan,
    morethan,
    repFilter,
)
from .gene_segments import GENE_SEGMENTS, get_genes
from .gene_usage import geneUsage, geneUsageAnalysis
from .io import IMMCOL, ImmunData, load_example_immdata, repLoad
from .kmers import getKmers, kmer_profile, spectratype, split_to_kmers
from .overlap import repOverlap, repOverlapAnalysis
from .plotting import (
    vis_clonal_space,
    vis_diversity,
    vis_explore,
    vis_gene_usage,
    vis_overlap_heatmap,
    vis_spectratype,
    vis_tracking,
)
from .public import (
    public_matrix,
    pubRep,
    pubRepApply,
    pubRepFilter,
    pubRepStatistics,
)
from .utils import (
    bunch_translate,
    coding,
    inframes,
    noncoding,
    outofframes,
    top,
)

__version__ = "0.1.0"

__all__ = [
    # data model
    "ImmunData",
    "IMMCOL",
    # I/O
    "repLoad",
    "load_example_immdata",
    # exploratory
    "repExplore",
    # clonality
    "repClonality",
    # diversity
    "repDiversity",
    # overlap
    "repOverlap",
    "repOverlapAnalysis",
    # gene usage
    "geneUsage",
    "geneUsageAnalysis",
    "GENE_SEGMENTS",
    "get_genes",
    # public clonotypes
    "pubRep",
    "public_matrix",
    "pubRepStatistics",
    "pubRepFilter",
    "pubRepApply",
    # dynamics
    "trackClonotypes",
    # k-mers
    "getKmers",
    "split_to_kmers",
    "kmer_profile",
    "spectratype",
    # filtering
    "repFilter",
    "include",
    "exclude",
    "lessthan",
    "morethan",
    "interval",
    # information theory & CDR3 analysis
    "entropy",
    "kl_div",
    "js_div",
    "cross_entropy",
    "cdr3_aa_profile",
    "AA_PROPERTIES",
    # preprocessing utilities
    "coding",
    "noncoding",
    "inframes",
    "outofframes",
    "top",
    "bunch_translate",
    # plotting
    "vis_diversity",
    "vis_overlap_heatmap",
    "vis_gene_usage",
    "vis_clonal_space",
    "vis_tracking",
    "vis_spectratype",
    "vis_explore",
]
