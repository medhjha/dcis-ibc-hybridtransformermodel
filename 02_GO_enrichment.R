suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(ggplot2)
  library(dplyr)
})

set.seed(42)
dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

top_degs <- read.csv("results/top_DEGs_IBC_vs_DCIS.csv", stringsAsFactors = FALSE)
cat("Loaded", nrow(top_degs), "DEGs for enrichment analysis\n")

up_genes   <- top_degs$gene[top_degs$logFC > 0]
down_genes <- top_degs$gene[top_degs$logFC < 0]
cat("Upregulated:", length(up_genes), "| Downregulated:", length(down_genes), "\n")

entrez_all <- bitr(top_degs$gene, fromType = "SYMBOL", toType = "ENTREZID",
                   OrgDb = org.Hs.eg.db)
entrez_up  <- bitr(up_genes,   fromType = "SYMBOL", toType = "ENTREZID",
                   OrgDb = org.Hs.eg.db)
entrez_down <- bitr(down_genes, fromType = "SYMBOL", toType = "ENTREZID",
                    OrgDb = org.Hs.eg.db)

go_up <- enrichGO(gene         = entrez_up$ENTREZID,
                  OrgDb         = org.Hs.eg.db,
                  ont           = "BP",
                  pAdjustMethod = "BH",
                  pvalueCutoff  = 0.05,
                  qvalueCutoff  = 0.05,
                  readable      = TRUE)

go_down <- enrichGO(gene         = entrez_down$ENTREZID,
                    OrgDb         = org.Hs.eg.db,
                    ont           = "BP",
                    pAdjustMethod = "BH",
                    pvalueCutoff  = 0.05,
                    qvalueCutoff  = 0.05,
                    readable      = TRUE)

write.csv(as.data.frame(go_up),   "results/GO_BP_up_in_IBC.csv",   row.names = FALSE)
write.csv(as.data.frame(go_down), "results/GO_BP_down_in_IBC.csv", row.names = FALSE)
cat("GO enrichment: up =", nrow(as.data.frame(go_up)),
    "| down =", nrow(as.data.frame(go_down)), "\n")

# KEGG enrichment
# KEGG pathway data used with permission from Kanehisa Laboratories
# (Kanehisa et al., Nucleic Acids Res. 2023; 51:D587-D592)
kegg_up <- enrichKEGG(gene         = entrez_up$ENTREZID,
                      organism     = "hsa",
                      pAdjustMethod = "BH",
                      pvalueCutoff  = 0.05)

kegg_down <- enrichKEGG(gene         = entrez_down$ENTREZID,
                        organism     = "hsa",
                        pAdjustMethod = "BH",
                        pvalueCutoff  = 0.05)

write.csv(as.data.frame(kegg_up),   "results/KEGG_up_in_IBC.csv",   row.names = FALSE)
write.csv(as.data.frame(kegg_down), "results/KEGG_down_in_IBC.csv", row.names = FALSE)
cat("KEGG enrichment: up =", nrow(as.data.frame(kegg_up)),
    "| down =", nrow(as.data.frame(kegg_down)), "\n")

# Figure 2A: upregulated GO terms
p1 <- dotplot(go_up, showCategory = 20, title = "GO BP: upregulated in IBC") +
  theme_classic(base_size = 11)
ggsave("figures/GO_BP_up_in_IBC.png", p1, width = 10, height = 8, dpi = 300)

# Figure 2B: downregulated GO terms
p2 <- dotplot(go_down, showCategory = 20, title = "GO BP: downregulated in IBC") +
  theme_classic(base_size = 11)
ggsave("figures/GO_BP_down_in_IBC.png", p2, width = 10, height = 8, dpi = 300)

cat("\nGO/KEGG enrichment complete.\n")
writeLines(capture.output(sessionInfo()), "sessionInfo_02_GO.txt")
