import torch
import torch.nn as nn
import torch.nn.functional as F
from TransformerEncoder import TransformerEncoder


class HybridTransformer(nn.Module):
    """Hybrid multi-modal Transformer for DCIS vs IBC classification.

    Four independent per-modality transformer encoders process gene expression,
    methylation proxy, pathway scores, and immune markers. Mean-pooled outputs
    are concatenated into a 512-d fused embedding passed to the classifier.
    No disease-stage label is used as input at any stage.
    """

    def __init__(self, gene_dim, meth_dim, pathway_dim, immune_dim,
                 hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.gene_enc = TransformerEncoder(gene_dim,    hidden_dim, 4, 2, dropout)
        self.meth_enc = TransformerEncoder(meth_dim,    hidden_dim, 4, 2, dropout)
        self.path_enc = TransformerEncoder(pathway_dim, hidden_dim, 4, 2, dropout)
        self.imm_enc  = TransformerEncoder(immune_dim,  hidden_dim, 4, 2, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, gene, meth, path, imm):
        g = self.gene_enc(gene).mean(dim=0)
        m = self.meth_enc(meth).mean(dim=0)
        p = self.path_enc(path).mean(dim=0)
        i = self.imm_enc(imm).mean(dim=0)
        h = torch.cat([g, m, p, i], dim=-1)
        return self.classifier(h), h
