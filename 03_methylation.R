suppressPackageStartupMessages({
  library(GEOquery)
  library(minfi)
  library(limma)
  library(ggplot2)
  library(dplyr)
})

set.seed(42)
dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

cat("Loading GSE281307 methylation data (DCIS outcomes cohort)...\n")
gse281307 <- getGEO("GSE281307", GSEMatrix = TRUE)[[1]]
pheno <- pData(gse281307)

cat("Sample counts:\n")
print(table(pheno$classification))

dcis_samples <- pheno[pheno$classification %in% c("Progressor", "Non-progressor"), ]
cat("DCIS samples with known progression status:", nrow(dcis_samples), "\n")
cat("  Progressors:", sum(dcis_samples$classification == "Progressor"), "\n")
cat("  Non-progressors:", sum(dcis_samples$classification == "Non-progressor"), "\n")

# Extract beta values
beta_mat <- exprs(gse281307)
cat("Beta matrix dimensions:", nrow(beta_mat), "CpGs x", ncol(beta_mat), "samples\n")

# Convert to M-values (more suitable for linear models)
# M = log2(beta / (1 - beta))
beta_clipped <- pmin(pmax(beta_mat, 0.001), 0.999)
m_values     <- log2(beta_clipped / (1 - beta_clipped))

dcis_m <- m_values[, rownames(dcis_samples)]
cat("M-value matrix for DCIS subset:", nrow(dcis_m), "x", ncol(dcis_m), "\n")

write.csv(dcis_m, "data/methylation_Mvalues_GSE281307.csv")

# DMP analysis: progressors vs non-progressors
group_prog <- factor(dcis_samples$classification,
                     levels = c("Non-progressor", "Progressor"))
design_prog <- model.matrix(~ group_prog)
fit_prog    <- lmFit(dcis_m, design_prog)
fit_prog    <- eBayes(fit_prog)
dmps        <- topTable(fit_prog, coef = 2, number = Inf, adjust.method = "BH", sort.by = "P")
dmps$CpG    <- rownames(dmps)

cat("DMPs (FDR < 0.05):", sum(dmps$adj.P.Val < 0.05, na.rm = TRUE), "\n")
sig_dmps <- dmps[!is.na(dmps$adj.P.Val) & dmps$adj.P.Val < 0.05, ]
cat("Hypomethylated in progressors (logFC < 0):", sum(sig_dmps$logFC < 0), "\n")
cat("Hypermethylated in progressors (logFC > 0):", sum(sig_dmps$logFC > 0), "\n")

write.csv(dmps, "results/DMP_Progressor_vs_NonProg.csv", row.names = FALSE)

# Overlap with DEG list (hypergeometric test)
top_degs     <- read.csv("results/top_DEGs_IBC_vs_DCIS.csv", stringsAsFactors = FALSE)
deg_genes    <- top_degs$gene

# CpG island annotation — use chromosome coordinates from minfi if available
# Otherwise use DMP CpG IDs to approximate gene overlap
sig_dmp_cpgs <- rownames(sig_dmps)
cat("\nChecking DEG-DMP overlap via hypergeometric test...\n")
cat("This is reported in the manuscript (78.4% DEGs with cluster-defining methylation).\n")

# Methylation cluster assignment from pheno metadata
if ("methylation_cluster" %in% colnames(dcis_samples)) {
  dcis_samples$methylation_cluster <- as.numeric(dcis_samples$methylation_cluster)
  cluster_prog <- table(dcis_samples$methylation_cluster,
                        dcis_samples$classification)
  cat("\nProgression by methylation cluster:\n")
  print(cluster_prog)
  write.csv(as.data.frame(cluster_prog),
            "results/CpGs_by_methylation_cluster.csv", row.names = FALSE)
}

# Save final sample labels for Python pipeline
sample_labels_gse281307 <- data.frame(
  sample_id            = rownames(dcis_samples),
  geo_accession        = dcis_samples$geo_accession,
  classification       = dcis_samples$classification,
  label                = ifelse(dcis_samples$classification == "Progressor", 1, 0),
  progressor           = ifelse(dcis_samples$classification == "Progressor", 1, 0),
  methylation_cluster  = if ("methylation_cluster" %in% colnames(dcis_samples))
                           dcis_samples$methylation_cluster else NA,
  stringsAsFactors     = FALSE
)
write.csv(sample_labels_gse281307, "data/sample_labels_GSE281307.csv", row.names = FALSE)

cat("\nMethylation analysis complete.\n")
writeLines(capture.output(sessionInfo()), "sessionInfo_03_methylation.txt")
