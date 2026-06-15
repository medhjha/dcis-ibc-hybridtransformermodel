DCIS/IBC Hybrid Multi-Modal Transformer
Code for: Multi-Omics Transformer Integration Reveals an Invasive-like Molecular Phenotype in Ductal Carcinoma In Situ
Medha Jha, Yasha Hasija — Department of Biotechnology, Delhi Technological University

How to run
Step 1 — R pipeline (run in order)
rsource("01_DEG_analysis.R")     # differential expression, 41-gene signature
source("02_GO_enrichment.R")    # GO/KEGG pathway enrichment
source("03_methylation.R")      # DNA methylation, GSE281307 cluster analysis
source("04_elastic_net.R")      # 75-gene panel selection
source("05_immune_analysis.R")  # immune checkpoint markers, GSE87517
Step 2 — Python pipeline
bashexport KMP_DUPLICATE_LIB_OK=TRUE
python run_model.py

Data
All datasets are publicly available from NCBI GEO:
DatasetDescriptionnGSE59246IBC/DCIS expression (discovery)102GSE26304IDC/DCIS expression (validation 1)67GSE33692IDC/DCIS expression (validation 2)20GSE87517Sorted T-cell RNA-seq41GSE281307DCIS clinical outcomes, methylation185

Requirements
R 4.4: GEOquery, limma, minfi, glmnet, clusterProfiler, ggplot2, dplyr, pROC, rstatix
Python 3.12: torch, scikit-learn, pandas, numpy, shap, umap-learn, matplotlib

Key results

AUC-ROC: 0.9318 +/- 0.0132 (5-fold CV) | held-out AUC: 0.8964
Sensitivity: 0.8214 | Specificity: 0.7609
Invasive-like DCIS: 15/46 (32.6%), threshold >= 0.3520
Clinical validation: OR=2.40 (p=0.008, methylation cluster), OR=3.71 (p=0.028, HER2)
Top SHAP features: ITGA2, SNCA, EN1, COL4A3, DEFB1


Notes on model design

No disease-stage label is used as input at any stage (prevents label leakage)
Immune fractions use per-sample expression values, not group means
SHAP values are on the log-odds scale; they identify model-predictive features, not causal drivers
The elastic net AUC (04_elastic_net.R) is an in-sample feature-selection estimate, not a generalisation estimate
