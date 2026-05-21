#!/usr/bin/env Rscript
# R-parity reference driver for py-immunarch.
#
# Runs immunarch 0.10.3 on its bundled `immdata` example dataset
# (a small TCR cohort with 12 samples + metadata) and writes the numeric
# results of every analysis function family to TSVs, so the Python port
# can be checked for bit-exact agreement.
#
# Usage:  Rscript r_reference_driver.R <output_dir>

suppressPackageStartupMessages(library(immunarch))

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "."
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

op <- function(name) file.path(out_dir, name)

data(immdata)
imm <- immdata$data
meta <- immdata$meta

# --- Dump the raw input so Python analyses the identical data --------------
for (s in names(imm)) {
  write.table(imm[[s]], op(paste0("input_", s, ".tsv")),
    sep = "\t", quote = FALSE, row.names = FALSE, na = "NA"
  )
}
write.table(meta, op("input_meta.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE, na = "NA"
)
writeLines(names(imm), op("input_samples.txt"))

wt <- function(obj, name) {
  write.table(obj, op(name), sep = "\t", quote = FALSE,
    row.names = TRUE, col.names = NA, na = "NA"
  )
}
wtdf <- function(df, name) {
  write.table(df, op(name), sep = "\t", quote = FALSE,
    row.names = FALSE, na = "NA"
  )
}

# ====================== repExplore =========================================
wtdf(repExplore(imm, .method = "volume"), "explore_volume.tsv")
wtdf(repExplore(imm, .method = "clones"), "explore_clones.tsv")
wtdf(repExplore(imm, .method = "count"), "explore_count.tsv")
wtdf(repExplore(imm, .method = "len", .col = "aa"), "explore_len.tsv")

# ====================== repClonality =======================================
wt(repClonality(imm, .method = "clonal.prop"), "clonality_clonalprop.tsv")
wt(repClonality(imm, .method = "homeo"), "clonality_homeo.tsv")
wt(repClonality(imm, .method = "top"), "clonality_top.tsv")
wt(repClonality(imm, .method = "rare"), "clonality_rare.tsv")

# ====================== repDiversity =======================================
wt(repDiversity(imm, .method = "chao1"), "diversity_chao1.tsv")
wtdf(repDiversity(imm, .method = "hill"), "diversity_hill.tsv")
wtdf(repDiversity(imm, .method = "div", .q = 5), "diversity_div.tsv")
wtdf(repDiversity(imm, .method = "gini.simp"), "diversity_ginisimp.tsv")
wtdf(repDiversity(imm, .method = "inv.simp"), "diversity_invsimp.tsv")
gini <- repDiversity(imm, .method = "gini")
gini_df <- data.frame(Sample = rownames(gini), Value = as.numeric(gini[, 1]))
wtdf(gini_df, "diversity_gini.tsv")
wt(repDiversity(imm, .method = "d50"), "diversity_d50.tsv")
wt(repDiversity(imm, .method = "dxx", .perc = 25), "diversity_dxx.tsv")

# rarefaction: write the curve (RNG-dependent, compared only loosely)
raref <- repDiversity(imm, .method = "raref", .verbose = FALSE)
wtdf(as.data.frame(raref), "diversity_raref.tsv")

# ====================== repOverlap =========================================
for (m in c("public", "overlap", "jaccard", "tversky", "cosine", "morisita")) {
  ov <- repOverlap(imm, .method = m, .verbose = FALSE)
  wt(as.matrix(ov), paste0("overlap_", m, ".tsv"))
}

# ====================== geneUsage ==========================================
gu <- geneUsage(imm, .gene = "hs.trbv", .norm = FALSE)
wtdf(as.data.frame(gu), "geneusage_count.tsv")
gun <- geneUsage(imm, .gene = "hs.trbv", .norm = TRUE)
wtdf(as.data.frame(gun), "geneusage_norm.tsv")
guj <- geneUsage(imm, .gene = "hs.trbj", .norm = FALSE)
wtdf(as.data.frame(guj), "geneusage_j.tsv")
# gene usage analysis: cosine / correlation matrices on the usage table
gu_cos <- geneUsageAnalysis(gun, "cosine", .verbose = FALSE)
wt(as.matrix(gu_cos), "geneusage_cosine.tsv")
gu_cor <- geneUsageAnalysis(gun, "cor", .verbose = FALSE)
wt(as.matrix(gu_cor), "geneusage_cor.tsv")
gu_js <- geneUsageAnalysis(gun, "js", .verbose = FALSE)
wt(as.matrix(gu_js), "geneusage_js.tsv")

# ====================== pubRep =============================================
pr <- pubRep(imm, .col = "aa+v", .quant = "count", .verbose = FALSE)
wtdf(as.data.frame(pr), "pubrep.tsv")

# ====================== trackClonotypes ====================================
tc <- trackClonotypes(imm, list(1, 15), .col = "aa")
wtdf(as.data.frame(tc), "track.tsv")

# ====================== getKmers ===========================================
km <- getKmers(imm[[1]], 3)
wtdf(as.data.frame(km), "kmers.tsv")
kmp <- kmer_profile(km, .method = "freq")
wt(as.matrix(kmp), "kmer_profile.tsv")

# ====================== spectratype =======================================
sp <- spectratype(imm[[1]], .quant = "count", .col = "aa")
wtdf(as.data.frame(sp), "spectratype.tsv")

# ====================== repSample =========================================
# Deterministic invariants: downsample / resample preserve the total read
# count, sample keeps an exact number of clonotypes. The RNG differs across
# languages so only the read totals / row counts are bit-exact.
set.seed(42)
ds <- repSample(imm, .method = "downsample", .n = 500)
dsdf <- data.frame(
  Sample = names(ds),
  Reads = sapply(ds, function(d) sum(d$Clones)),
  Rows = sapply(ds, nrow)
)
wtdf(dsdf, "sample_downsample.tsv")

set.seed(42)
rs <- repSample(imm, .method = "resample", .n = 500)
rsdf <- data.frame(
  Sample = names(rs),
  Reads = sapply(rs, function(d) sum(d$Clones))
)
wtdf(rsdf, "sample_resample.tsv")

set.seed(42)
sm <- repSample(imm, .method = "sample", .n = 100)
smdf <- data.frame(
  Sample = names(sm),
  Rows = sapply(sm, nrow)
)
wtdf(smdf, "sample_sample.tsv")

# ====================== gene_stats ========================================
gst <- gene_stats()
wtdf(as.data.frame(gst), "gene_stats.tsv")

# ====================== dbAnnotate ========================================
# Build a small annotation database from real clonotypes of sample 1 and
# annotate the whole cohort against it.
db_seqs <- head(na.omit(imm[[1]]$CDR3.aa), 12)
db <- data.frame(CDR3 = db_seqs, stringsAsFactors = FALSE)
ann <- dbAnnotate(imm, db, .data.col = "CDR3.aa", .db.col = "CDR3")
wtdf(as.data.frame(ann), "db_annotate.tsv")

# ====================== seqDist ===========================================
# Pairwise CDR3.nt Hamming distances on a BCR repertoire, grouped by
# V/J gene + sequence length (the immunarch default).
data(bcrdata)
bcr <- bcrdata$data
sd <- seqDist(bcr, .col = "CDR3.nt", .method = "hamming")
sd_groups <- sd[[1]]
# flatten every distance matrix into long form (SeqA, SeqB, dist), keyed by
# the sequence pair so it can be joined regardless of group ordering.
sd_rows <- list()
for (mat in sd_groups) {
  labs <- attr(mat, "Labels")
  if (length(labs) < 2) next
  m <- as.matrix(mat)
  for (i in seq_len(nrow(m))) {
    for (j in seq_len(ncol(m))) {
      if (j > i) {
        sd_rows[[length(sd_rows) + 1]] <- data.frame(
          SeqA = labs[i], SeqB = labs[j],
          Dist = m[i, j], stringsAsFactors = FALSE
        )
      }
    }
  }
}
sd_long <- do.call(rbind, sd_rows)
wtdf(sd_long, "seqdist.tsv")

# ====================== repGermline ======================================
# Reconstruct BCR germlines; deterministic and bit-exact.
gml <- repGermline(bcr, .threads = 1)$full_clones
gml_out <- gml[, c("Clone.ID", "V.allele", "J.allele",
                   "Germline.sequence", "V.aa", "J.aa", "Sequence")]
gml_out <- gml_out[order(gml_out$Clone.ID), ]
wtdf(gml_out, "germline.tsv")

cat("R reference driver done.\n")
