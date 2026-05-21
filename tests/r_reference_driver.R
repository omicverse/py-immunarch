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

cat("R reference driver done.\n")
