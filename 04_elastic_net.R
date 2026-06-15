suppressPackageStartupMessages({
  library(glmnet)
  library(pROC)
  library(ggplot2)
  library(dplyr)
})

set.seed(42)
dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

top_degs <- read.csv("results/top_DEGs_IBC_vs_DCIS.csv", stringsAsFactors = FALSE)
expr     <- read.csv("data/gene_expression.csv", row.names = 1, check.names = FALSE)
pheno    <- read.csv("data/pheno_GSE59246_clean.csv", stringsAsFactors = FALSE)

expr  <- expr[, pheno$sample_id]
y_bin <- ifelse(grepl("IBC|Invasive", pheno$disease_state, ignore.case = TRUE), 1, 0)

common_genes <- intersect(top_degs$gene, rownames(expr))
X <- t(expr[common_genes, ])
X <- scale(X)
cat("Feature matrix:", nrow(X), "samples x", ncol(X), "features\n")
cat("Class counts: IBC =", sum(y_bin == 1), "| DCIS =", sum(y_bin == 0), "\n")

cat("Running elastic net with 10-fold CV (alpha = 0.5)...\n")
cv_fit <- cv.glmnet(X, y_bin, family = "binomial", alpha = 0.5,
                    nfolds = 10, standardize = FALSE, type.measure = "auc")

cat("Lambda min:", cv_fit$lambda.min, "\n")
cat("Lambda 1se:", cv_fit$lambda.1se, "\n")

best_fit <- glmnet(X, y_bin, family = "binomial", alpha = 0.5,
                   lambda = cv_fit$lambda.min, standardize = FALSE)

coefs     <- coef(best_fit)
nonzero   <- coefs[coefs[, 1] != 0, , drop = FALSE]
selected_genes <- rownames(nonzero)[rownames(nonzero) != "(Intercept)"]
cat("Genes selected by elastic net:", length(selected_genes), "\n")

coef_df <- data.frame(
  gene        = selected_genes,
  coefficient = as.numeric(nonzero[selected_genes, 1]),
  stringsAsFactors = FALSE
)
coef_df <- coef_df[order(abs(coef_df$coefficient), decreasing = TRUE), ]

# Predict on training data
# Note: AUC reported here (0.9972) is an in-sample / feature-selection estimate
# based on fitting and evaluating on the same 102 samples used for DEG discovery.
# This inflated estimate reflects the discriminative power of the pre-selected
# features, not generalisation performance. The downstream transformer uses
# 5-fold cross-validation on the same cohort for a less optimistic performance
# estimate (AUC=0.9318 +/- 0.0132).
pred_prob <- predict(best_fit, X, type = "response")[, 1]
roc_obj   <- roc(y_bin, pred_prob, quiet = TRUE)
cat("In-sample AUC (feature selection estimate):", round(auc(roc_obj), 4), "\n")
cat("Sensitivity:", round(coords(roc_obj, "best")["sensitivity"], 4), "\n")
cat("Specificity:", round(coords(roc_obj, "best")["specificity"], 4), "\n")

write.csv(coef_df, "results/elastic_net_classifier_genes.csv", row.names = FALSE)
cat("Saved", nrow(coef_df), "selected genes to results/elastic_net_classifier_genes.csv\n")

# Extract expression for the 75-gene set and save for Python pipeline
gene75_expr <- expr[selected_genes[selected_genes %in% rownames(expr)], ]
write.csv(t(gene75_expr), "data/gene_expression_75.csv")
cat("Saved 75-gene expression matrix to data/gene_expression_75.csv\n")

# Table 3: top 10 by coefficient magnitude (as in manuscript)
cat("\nTop 10 elastic net genes by coefficient magnitude:\n")
print(head(coef_df, 10))

# CV lambda plot
png("figures/elastic_net_CV.png", width = 1800, height = 1400, res = 300)
plot(cv_fit, main = "Elastic net 10-fold CV (alpha=0.5)")
dev.off()

cat("\nElastic net analysis complete.\n")
writeLines(capture.output(sessionInfo()), "sessionInfo_04_elastic_net.txt")
