suppressPackageStartupMessages({
  library(GEOquery)
  library(limma)
  library(ggplot2)
  library(ggrepel)
  library(pheatmap)
  library(dplyr)
})

set.seed(42)
dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

cat("Downloading GSE59246 (discovery cohort)...\n")
gse59246 <- getGEO("GSE59246", GSEMatrix = TRUE, getGPL = TRUE)[[1]]
pheno59246 <- pData(gse59246)

disease_col <- grep("disease.state|disease state",
                    colnames(pheno59246), ignore.case = TRUE, value = TRUE)[1]
pheno59246$disease_state <- trimws(gsub("disease state: ", "",
                                         pheno59246[[disease_col]], ignore.case = TRUE))
print(table(pheno59246$disease_state))

keep <- pheno59246$disease_state %in% c("Invasive breast cancer (IBC)",
                                          "Ductal carcinoma in situ (DCIS)")
gse_sub   <- gse59246[, keep]
pheno_sub <- pheno59246[keep, ]
cat("Samples retained:", sum(keep), "\n")

expr <- exprs(gse_sub)
feat <- fData(gse_sub)
gene_col <- grep("gene.symbol|Symbol", colnames(feat), ignore.case = TRUE, value = TRUE)[1]
feat$gene_symbol <- feat[[gene_col]]

valid <- !is.na(feat$gene_symbol) & feat$gene_symbol != "" & feat$gene_symbol != "---"
expr  <- expr[valid, ]
feat  <- feat[valid, ]

feat$row_mean <- rowMeans(expr)
feat$probe_id <- rownames(feat)
best_probes   <- feat %>%
  group_by(gene_symbol) %>%
  slice_max(row_mean, n = 1, with_ties = FALSE) %>%
  pull(probe_id)

expr <- expr[best_probes, ]
rownames(expr) <- feat$gene_symbol[match(best_probes, feat$probe_id)]
cat("Expression matrix:", nrow(expr), "genes x", ncol(expr), "samples\n")

write.csv(expr, "data/gene_expression.csv")
write.csv(data.frame(sample_id = colnames(expr),
                     disease_state = pheno_sub$disease_state,
                     stringsAsFactors = FALSE),
          "data/pheno_GSE59246_clean.csv", row.names = FALSE)

group <- factor(ifelse(grepl("IBC|Invasive", pheno_sub$disease_state, ignore.case = TRUE),
                       "IBC", "DCIS"), levels = c("DCIS", "IBC"))
design <- model.matrix(~ group)
colnames(design) <- c("Intercept", "IBC_vs_DCIS")

fit <- lmFit(expr, design)
fit <- eBayes(fit)
res <- topTable(fit, coef = "IBC_vs_DCIS", number = Inf, adjust.method = "BH", sort.by = "P")
res$gene <- rownames(res)

sig  <- res[res$adj.P.Val < 0.05, ]
top  <- sig[abs(sig$logFC) > 1, ]
cat("DEGs (FDR<0.05):", nrow(sig), "| with |logFC|>1:", nrow(top), "\n")
cat("Upregulated:", sum(top$logFC > 0), "| Downregulated:", sum(top$logFC < 0), "\n")

write.csv(res, "results/DEG_IBC_vs_DCIS_GSE59246.csv", row.names = FALSE)
write.csv(top, "results/top_DEGs_IBC_vs_DCIS.csv", row.names = FALSE)

for (cohort in c("GSE26304", "GSE33692")) {
  cat("\nDownloading", cohort, "...\n")
  gse <- getGEO(cohort, GSEMatrix = TRUE, getGPL = TRUE)[[1]]
  ph  <- pData(gse)
  tc  <- grep("type|tissue|histol", colnames(ph), ignore.case = TRUE, value = TRUE)[1]
  ph$sample_type <- trimws(ph[[tc]])
  print(table(ph$sample_type))

  k   <- grepl("IDC|invasive|DCIS", ph$sample_type, ignore.case = TRUE) &
         !grepl("stroma|normal|mixed", ph$sample_type, ignore.case = TRUE)
  g2  <- gse[, k]
  p2  <- ph[k, ]

  ex2  <- exprs(g2)
  fe2  <- fData(g2)
  gc2  <- grep("gene.symbol|Symbol", colnames(fe2), ignore.case = TRUE, value = TRUE)[1]
  fe2$gene_symbol <- fe2[[gc2]]
  v2   <- !is.na(fe2$gene_symbol) & fe2$gene_symbol != "" & fe2$gene_symbol != "---"
  ex2  <- ex2[v2, ]
  fe2  <- fe2[v2, ]
  fe2$row_mean <- rowMeans(ex2)
  fe2$probe_id <- rownames(fe2)
  bp2  <- fe2 %>% group_by(gene_symbol) %>%
    slice_max(row_mean, n = 1, with_ties = FALSE) %>% pull(probe_id)
  ex2  <- ex2[bp2, ]
  rownames(ex2) <- fe2$gene_symbol[match(bp2, fe2$probe_id)]

  gr2  <- factor(ifelse(grepl("IDC|invasive", p2$sample_type, ignore.case = TRUE),
                        "IDC", "DCIS"), levels = c("DCIS", "IDC"))
  des2 <- model.matrix(~ gr2)
  f2   <- eBayes(lmFit(ex2, des2))
  r2   <- topTable(f2, coef = 2, number = Inf, sort.by = "P")
  r2$gene <- rownames(r2)

  write.csv(r2, paste0("results/DEG_", cohort, ".csv"), row.names = FALSE)
  cat(cohort, ": n =", ncol(ex2), "\n")
}

cat("\nDEG analysis complete.\n")
writeLines(capture.output(sessionInfo()), "sessionInfo_01_DEG.txt")
