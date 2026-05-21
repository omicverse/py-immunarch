"""Matplotlib visualisations — the ``vis``-style repertoire plots.

A pragmatic, dependency-light re-implementation of immunarch's ``vis()``
family: diversity bars, overlap heatmaps, gene-usage bars, clonal-space
stacked bars, clonotype-tracking curves and CDR3 spectratypes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "vis_diversity",
    "vis_overlap_heatmap",
    "vis_gene_usage",
    "vis_clonal_space",
    "vis_tracking",
    "vis_spectratype",
    "vis_explore",
]


def _ax(ax):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    return ax


# --------------------------------------------------------------------------
def vis_diversity(div: pd.DataFrame, ax=None, title: str = "Diversity"):
    """Bar plot of a per-sample diversity table (``Sample`` + ``Value``)."""
    ax = _ax(ax)
    if "Value" in div.columns and "Sample" in div.columns:
        x = div["Sample"].astype(str)
        y = div["Value"]
    elif "Estimator" in div.columns:  # chao1
        x = div.index.astype(str)
        y = div["Estimator"]
    else:
        x = div.index.astype(str)
        y = div.iloc[:, 0]
    ax.bar(range(len(x)), y, color="#2c7fb8")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.figure.tight_layout()
    return ax


def vis_overlap_heatmap(ov: pd.DataFrame, ax=None,
                        title: str = "Repertoire overlap",
                        cmap: str = "viridis", annot: bool = True):
    """Heatmap of a repertoire-overlap matrix."""
    ax = _ax(ax)
    mat = ov.to_numpy(dtype=float)
    im = ax.imshow(mat, cmap=cmap)
    ax.set_xticks(range(ov.shape[1]))
    ax.set_xticklabels(ov.columns, rotation=45, ha="right")
    ax.set_yticks(range(ov.shape[0]))
    ax.set_yticklabels(ov.index)
    if annot:
        finite = mat[np.isfinite(mat)]
        thr = (finite.max() + finite.min()) / 2 if finite.size else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < thr else "black",
                            fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.figure.tight_layout()
    return ax


def vis_gene_usage(gu: pd.DataFrame, ax=None, top: int = 20,
                   title: str = "Gene usage"):
    """Grouped bar plot of a gene-usage table (``Names`` + samples)."""
    ax = _ax(ax)
    samples = [c for c in gu.columns if c != "Names"]
    sub = gu.copy()
    sub["_tot"] = sub[samples].fillna(0).sum(axis=1)
    sub = sub.nlargest(top, "_tot")
    x = np.arange(len(sub))
    w = 0.8 / max(1, len(samples))
    for k, s in enumerate(samples):
        ax.bar(x + k * w, sub[s].fillna(0), width=w, label=s)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(sub["Names"], rotation=90, fontsize=7)
    ax.set_ylabel("Usage")
    ax.set_title(title)
    if len(samples) <= 12:
        ax.legend(fontsize=7, ncol=2)
    ax.figure.tight_layout()
    return ax


def vis_clonal_space(clon: pd.DataFrame, ax=None,
                     title: str = "Clonal space"):
    """Stacked bar plot of a clonal-space (homeo/top/rare) table."""
    ax = _ax(ax)
    samples = clon.index.astype(str)
    bottom = np.zeros(len(samples))
    cmap = __import__("matplotlib").cm.get_cmap("Spectral",
                                                clon.shape[1])
    for k, col in enumerate(clon.columns):
        vals = clon[col].to_numpy(dtype=float)
        ax.bar(range(len(samples)), vals, bottom=bottom,
               label=str(col), color=cmap(k))
        bottom += vals
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels(samples, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.figure.tight_layout()
    return ax


def vis_tracking(track: pd.DataFrame, ax=None,
                 title: str = "Clonotype tracking"):
    """Line plot of clonotype abundance across samples (trackClonotypes)."""
    ax = _ax(ax)
    key_cols = [c for c in track.columns
                if c in ("CDR3.aa", "CDR3.nt", "V.name", "J.name")]
    sample_cols = [c for c in track.columns if c not in key_cols]
    for _, row in track.iterrows():
        ax.plot(range(len(sample_cols)),
                row[sample_cols].to_numpy(dtype=float),
                marker="o", markersize=3, alpha=0.6)
    ax.set_xticks(range(len(sample_cols)))
    ax.set_xticklabels(sample_cols, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.figure.tight_layout()
    return ax


def vis_spectratype(sp: pd.DataFrame, ax=None,
                    title: str = "Spectratype"):
    """Bar plot of a CDR3-length spectratype."""
    ax = _ax(ax)
    if "Gene" in sp.columns:
        piv = sp.pivot_table(index="Length", columns="Gene", values="Val",
                             aggfunc="sum", fill_value=0).sort_index()
        bottom = np.zeros(len(piv))
        for col in piv.columns:
            ax.bar(piv.index, piv[col], bottom=bottom, label=str(col))
            bottom += piv[col].to_numpy()
        ax.legend(fontsize=6, ncol=2)
    else:
        sp = sp.sort_values("Length")
        ax.bar(sp["Length"], sp["Val"], color="#41b6c4")
    ax.set_xlabel("CDR3 length")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.figure.tight_layout()
    return ax


def vis_explore(exp: pd.DataFrame, ax=None, title: str = None):
    """Bar plot of a repExplore volume / clones table."""
    ax = _ax(ax)
    value_col = ("Volume" if "Volume" in exp.columns
                 else "Clones" if "Clones" in exp.columns else None)
    if value_col is None:
        raise ValueError("vis_explore needs a 'volume' or 'clones' table.")
    ax.bar(exp["Sample"].astype(str), exp[value_col], color="#7fcdbb")
    ax.set_ylabel(value_col)
    ax.set_title(title or f"Repertoire {value_col.lower()}")
    ax.tick_params(axis="x", rotation=45)
    ax.figure.tight_layout()
    return ax
