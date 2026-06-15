suppressPackageStartupMessages({
  library(GEOquery)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(rstatix)
})

set.seed(42)
dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

# GSE87517: sorted CD45+CD3+ T-cell RNA-seq data
# Normal n=19, DCIS n=11, IDC n=11
cat("Downloading GSE87517 (immune microenvironment)...\n")
gse87517 <- getGEO("GSE87517", GSEMatrix = TRUE, getGPL = TRUE)[[1]]
pheno     <- pData(gse87517)

type_col <- grep("disease|status|group|type",
                 colnames(pheno), ignore.case = TRUE, value = TRUE)[1]
pheno$group <- trimws(pheno[[type_col]])
print(table(pheno$group))

keep  <- grepl("Normal|DCIS|IDC|invasive", pheno$group, ignore.case = TRUE)
gse_s <- gse87517[, keep]
ph_s  <- pheno[keep, ]
ph_s$group <- dplyr::case_when(
  grepl("Normal|normal", ph_s$group) ~ "Normal",
  grepl("DCIS|dcis",     ph_s$group) ~ "DCIS",
  grepl("IDC|invasive",  ph_s$group) ~ "IDC",
  TRUE ~ ph_s$group
)
ph_s$group <- factor(ph_s$group, levels = c("Normal", "DCIS", "IDC"))
cat("Samples: Normal =", sum(ph_s$group == "Normal"),
    "| DCIS =", sum(ph_s$group == "DCIS"),
    "| IDC =", sum(ph_s$group == "IDC"), "\n")

expr <- exprs(gse_s)
feat <- fData(gse_s)
gc   <- grep("gene.symbol|Symbol", colnames(feat), ignore.case = TRUE, value = TRUE)[1]
feat$gene_symbol <- feat[[gc]]
valid <- !is.na(feat$gene_symbol) & feat$gene_symbol != "" & feat$gene_symbol != "---"
expr  <- expr[valid, ]
feat  <- feat[valid, ]
feat$row_mean <- rowMeans(expr)
feat$probe_id <- rownames(feat)
bp    <- feat %>% group_by(gene_symbol) %>%
  slice_max(row_mean, n = 1, with_ties = FALSE) %>% pull(probe_id)
expr  <- expr[bp, ]
rownames(expr) <- feat$gene_symbol[match(bp, feat$probe_id)]

write.csv(expr, "data/expr_GSE87517_normalised.csv")

# 8 immune checkpoint and effector markers
markers <- c("CTLA4", "FOXP3", "PDCD1", "LAG3", "TIGIT", "GZMB", "NKG7", "IFNG")
markers_present <- intersect(markers, rownames(expr))
cat("Markers available:", paste(markers_present, collapse = ", "), "\n")

expr_markers <- t(expr[markers_present, ])
df_long <- as.data.frame(expr_markers)
df_long$sample_id <- rownames(df_long)
df_long$group     <- ph_s$group
df_long <- pivot_longer(df_long, cols = all_of(markers_present),
                        names_to = "marker", values_to = "expression")

# Wilcoxon tests: IDC vs DCIS and IDC vs Normal
# All p-values are nominal (unadjusted) — exploratory analysis on small n
# Interpret with caution; confirm in larger independent cohorts
wilcox_results <- df_long %>%
  group_by(marker) %>%
  do({
    idc   <- .$expression[.$group == "IDC"]
    dcis  <- .$expression[.$group == "DCIS"]
    norm  <- .$expression[.$group == "Normal"]
    data.frame(
      IDC_vs_DCIS   = wilcox.test(idc, dcis)$p.value,
      IDC_vs_Normal = wilcox.test(idc, norm)$p.value
    )
  }) %>%
  ungroup()

cat("\nWilcoxon test results (nominal p-values, unadjusted, exploratory):\n")
print(wilcox_results)
write.csv(wilcox_results, "results/immune_marker_wilcox_tests.csv", row.names = FALSE)

# Figure 3: boxplot of immune checkpoint markers by disease stage
p <- ggplot(df_long, aes(x = group, y = expression, fill = group)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7, width = 0.6) +
  geom_jitter(width = 0.15, size = 1.2, alpha = 0.6) +
  facet_wrap(~ marker, scales = "free_y", ncol = 4) +
  scale_fill_manual(values = c("Normal" = "#95C8C0", "DCIS" = "#4878D0", "IDC" = "#E8554E")) +
  labs(x = NULL, y = "Expression", fill = NULL,
       title = "Immune checkpoint markers in sorted T-cells (GSE87517)",
       subtitle = "Normal n=19, DCIS n=11, IDC n=11 | nominal p-values, exploratory") +
  theme_classic(base_size = 11) +
  theme(legend.position = "top", strip.text = element_text(face = "italic"))

ggsave("figures/boxplot_immune_markers.png", p, width = 12, height = 7, dpi = 300)
cat("\nFigure 3 saved.\n")

# DCIS subgroup immune analysis uses per-sample expression from gene_expression.csv
# (see run_model.py — immune_fractions_fixed.csv)
cat("\nImmune analysis complete.\n")
writeLines(capture.output(sessionInfo()), "sessionInfo_05_immune.txt")
