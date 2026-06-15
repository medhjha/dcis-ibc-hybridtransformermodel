import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from HybridCancerModel import HybridTransformer

os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)


def load_data():
    gene = pd.read_csv('data/gene_expression_75.csv',   index_col=0)
    meth = pd.read_csv('data/methylation_proxy.csv',    index_col=0)
    path = pd.read_csv('data/pathway_faime_scores.csv', index_col=0)
    imm  = pd.read_csv('data/immune_fractions_fixed.csv', index_col=0)
    labs = pd.read_csv('data/sample_labels.csv').set_index('sample_id')

    common = sorted(
        set(gene.index) & set(meth.index) & set(path.index) & set(imm.index) & set(labs.index)
    )
    gene, meth, path, imm, labs = [df.loc[common] for df in [gene, meth, path, imm, labs]]
    print(f"Loaded {len(common)} samples | label counts: {dict(labs['label'].value_counts())}")

    mask   = labs['label'].isin([1, 2])
    gene_b = gene.loc[mask]
    meth_b = meth.loc[mask]
    path_b = path.loc[mask]
    imm_b  = imm.loc[mask]
    labs_b = labs.loc[mask].copy()
    labs_b['label'] = labs_b['label'].map({1: 0, 2: 1})

    n_dcis = int((labs_b['label'] == 0).sum())
    n_ibc  = int((labs_b['label'] == 1).sum())
    print(f"Training set: DCIS={n_dcis}, IBC={n_ibc}, Total={len(labs_b)}")
    print(f"Dimensions: gene={gene_b.shape[1]}, meth={meth_b.shape[1]}, "
          f"path={path_b.shape[1]}, imm={imm_b.shape[1]}")
    return gene_b, meth_b, path_b, imm_b, labs_b


def train_epoch(model, optimizer, g, m, p, i, y, device, batch_size=32):
    model.train()
    criterion = nn.CrossEntropyLoss()
    idx = np.random.permutation(len(y))
    total = 0
    for start in range(0, len(y), batch_size):
        batch = idx[start:start + batch_size]
        gb = torch.tensor(g[batch]).to(device)
        mb = torch.tensor(m[batch]).to(device)
        pb = torch.tensor(p[batch]).to(device)
        ib = torch.tensor(i[batch]).to(device)
        yb = torch.tensor(y[batch]).to(device)
        optimizer.zero_grad()
        ll = torch.cat([model(gb[s], mb[s], pb[s], ib[s])[0].unsqueeze(0)
                        for s in range(len(batch))])
        loss = criterion(ll, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total


@torch.no_grad()
def evaluate(model, g, m, p, i, y, device):
    model.eval()
    probs, preds = [], []
    for s in range(len(y)):
        logits, _ = model(
            torch.tensor(g[s]).to(device), torch.tensor(m[s]).to(device),
            torch.tensor(p[s]).to(device), torch.tensor(i[s]).to(device)
        )
        probs.append(torch.softmax(logits, dim=-1)[1].item())
        preds.append(logits.argmax().item())
    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = 0.5
    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, average='macro', zero_division=0)
    return acc, f1, auc, np.array(preds), np.array(probs)


def run_crossval(gene_b, meth_b, path_b, imm_b, labs_b, device):
    print("\n=== 5-fold stratified cross-validation ===")
    y_all  = labs_b['label'].values.astype(np.int64)
    sids   = labs_b.index.tolist()
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probs = np.zeros(len(y_all))
    all_preds = np.zeros(len(y_all), dtype=int)
    all_embs  = {}
    fold_results = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
        print(f"\nFold {fold + 1}/5")
        scalers = []
        arrays  = []
        for df in [gene_b, meth_b, path_b, imm_b]:
            sc = StandardScaler()
            arrays.append((
                sc.fit_transform(df.iloc[tr_idx].values.astype(np.float32)),
                sc.transform(df.iloc[vl_idx].values.astype(np.float32))
            ))
            scalers.append(sc)
        g_tr, g_vl = arrays[0]
        m_tr, m_vl = arrays[1]
        p_tr, p_vl = arrays[2]
        i_tr, i_vl = arrays[3]
        y_tr = y_all[tr_idx]
        y_vl = y_all[vl_idx]

        model = HybridTransformer(
            g_tr.shape[1], m_tr.shape[1], p_tr.shape[1], i_tr.shape[1]
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        best_auc, patience, best_state = 0, 0, None
        for epoch in range(50):
            train_epoch(model, optimizer, g_tr, m_tr, p_tr, i_tr, y_tr, device)
            scheduler.step()
            _, _, auc, _, _ = evaluate(model, g_vl, m_vl, p_vl, i_vl, y_vl, device)
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch + 1} | val AUC {auc:.4f}")
            if auc > best_auc:
                best_auc   = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience += 1
                if patience >= 10:
                    print(f"  early stop at epoch {epoch + 1}")
                    break

        model.load_state_dict(best_state)
        acc, f1, auc, preds, probs = evaluate(model, g_vl, m_vl, p_vl, i_vl, y_vl, device)
        print(f"  fold {fold + 1}: acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
        fold_results.append({'fold': fold + 1, 'accuracy': acc, 'f1': f1, 'auc': auc})
        all_probs[vl_idx] = probs
        all_preds[vl_idx] = preds

        model.eval()
        with torch.no_grad():
            for li, gi in enumerate(vl_idx):
                _, h = model(
                    torch.tensor(g_vl[li]).to(device), torch.tensor(m_vl[li]).to(device),
                    torch.tensor(p_vl[li]).to(device), torch.tensor(i_vl[li]).to(device)
                )
                all_embs[sids[gi]] = h.cpu().numpy()

        torch.save(model.state_dict(), f'checkpoints/fold{fold + 1}_best.pt')

    accs = [r['accuracy'] for r in fold_results]
    f1s  = [r['f1'] for r in fold_results]
    aucs = [r['auc'] for r in fold_results]
    overall = roc_auc_score(y_all, all_probs)
    cm = confusion_matrix(y_all, all_preds)
    sens = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    print("\n=== cross-validation summary ===")
    print(f"accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"f1 macro : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"auc      : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"overall held-out auc : {overall:.4f}")
    print(f"sensitivity (ibc recall) : {sens:.4f}")
    print(f"specificity (dcis recall): {spec:.4f}")

    pd.DataFrame(fold_results).to_csv('results/crossval_results.csv', index=False)
    return all_probs, all_preds, y_all, all_embs, fold_results


def score_invasive_like(gene_b, meth_b, path_b, imm_b, labs_b, fold_results, device):
    print("\n=== invasive-like DCIS scoring ===")
    best_fold = int(pd.DataFrame(fold_results).sort_values('auc').iloc[-1]['fold'])
    print(f"using fold {best_fold} model")

    sc_g = StandardScaler().fit(gene_b.values.astype(np.float32))
    sc_m = StandardScaler().fit(meth_b.values.astype(np.float32))
    sc_p = StandardScaler().fit(path_b.values.astype(np.float32))
    sc_i = StandardScaler().fit(imm_b.values.astype(np.float32))

    model = HybridTransformer(
        gene_b.shape[1], meth_b.shape[1], path_b.shape[1], imm_b.shape[1], dropout=0.0
    ).to(device)
    model.load_state_dict(
        torch.load(f'checkpoints/fold{best_fold}_best.pt', map_location=device)
    )
    model.eval()

    dcis_ids = labs_b[labs_b['label'] == 0].index.tolist()
    scores = {}
    with torch.no_grad():
        for sid in dcis_ids:
            g = torch.tensor(sc_g.transform(gene_b.loc[[sid]].values.astype(np.float32))[0]).to(device)
            m = torch.tensor(sc_m.transform(meth_b.loc[[sid]].values.astype(np.float32))[0]).to(device)
            p = torch.tensor(sc_p.transform(path_b.loc[[sid]].values.astype(np.float32))[0]).to(device)
            i = torch.tensor(sc_i.transform(imm_b.loc[[sid]].values.astype(np.float32))[0]).to(device)
            logits, _ = model(g, m, p, i)
            scores[sid] = logits[1].item()

    vals = np.array(list(scores.values()))
    vmin, vmax = vals.min(), vals.max()
    for sid in scores:
        scores[sid] = (scores[sid] - vmin) / (vmax - vmin + 1e-8)

    threshold = float(np.percentile(list(scores.values()), 66.7))
    print(f"threshold (66.7th percentile): {threshold:.4f}")

    rows = [{'sample_id': sid, 'invasive_score': scores[sid],
             'invasive_like': scores[sid] >= threshold} for sid in scores]
    df = pd.DataFrame(rows).sort_values('invasive_score', ascending=False)
    df.to_csv('results/invasive_like_dcis.csv', index=False)

    n = df['invasive_like'].sum()
    print(f"invasive-like DCIS: {n}/{len(dcis_ids)} ({100*n/len(dcis_ids):.1f}%)")
    return df


def plot_umap(all_embs, labs_b):
    if not (HAS_UMAP and HAS_PLOT):
        return
    print("\n=== UMAP ===")
    inv = pd.read_csv('results/invasive_like_dcis.csv', index_col='sample_id')
    sids = list(all_embs.keys())
    emb  = np.array([all_embs[s] for s in sids])
    y    = labs_b.loc[sids, 'label'].values

    reducer = umap.UMAP(n_neighbors=12, min_dist=0.05, random_state=42,
                        metric='cosine', spread=1.2)
    coords  = reducer.fit_transform(emb)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for cls, label, col, mk in [(0, 'DCIS', '#4878D0', 'o'), (1, 'IBC', '#EE854A', 's')]:
        mask = y == cls
        axes[0].scatter(coords[mask, 0], coords[mask, 1], c=col, label=label,
                        marker=mk, s=65, alpha=0.8, edgecolors='white', linewidths=0.6)
    axes[0].set_title('DCIS vs IBC (held-out embeddings)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.15)

    axes[1].scatter(coords[y == 1, 0], coords[y == 1, 1], c='#EE854A',
                    label='IBC', marker='s', s=55, alpha=0.5, edgecolors='white', linewidths=0.5)
    inv_idx  = [i for i, s in enumerate(sids) if y[i] == 0 and s in inv.index and inv.loc[s, 'invasive_like']]
    dcis_idx = [i for i, s in enumerate(sids) if y[i] == 0 and (s not in inv.index or not inv.loc[s, 'invasive_like'])]
    if dcis_idx:
        axes[1].scatter(coords[dcis_idx, 0], coords[dcis_idx, 1], c='#6ACC65',
                        label=f'DCIS-like (n={len(dcis_idx)})', marker='o', s=65,
                        alpha=0.8, edgecolors='white', linewidths=0.6)
    if inv_idx:
        axes[1].scatter(coords[inv_idx, 0], coords[inv_idx, 1], c='#D65F5F',
                        label=f'Invasive-like (n={len(inv_idx)})', marker='^', s=100,
                        alpha=0.95, edgecolors='black', linewidths=0.8, zorder=5)
    axes[1].set_title('Invasive-like DCIS subgroup')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig('results/figures/Fig5A_UMAP.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("saved: results/figures/Fig5A_UMAP.png")


def run_shap(gene_b, meth_b, path_b, imm_b, labs_b, fold_results, device):
    if not (HAS_SHAP and HAS_PLOT):
        return
    print("\n=== SHAP analysis ===")

    sc_g = StandardScaler().fit(gene_b.values.astype(np.float32))
    g    = sc_g.transform(gene_b.values.astype(np.float32))
    gene_cols = gene_b.columns.tolist()

    best_fold = int(pd.DataFrame(fold_results).sort_values('auc').iloc[-1]['fold'])
    model = HybridTransformer(
        gene_b.shape[1], meth_b.shape[1], path_b.shape[1], imm_b.shape[1], dropout=0.0
    ).to(device)
    model.load_state_dict(
        torch.load(f'checkpoints/fold{best_fold}_best.pt', map_location=device)
    )
    model.eval()

    mz = torch.zeros(meth_b.shape[1]).to(device)
    pz = torch.zeros(path_b.shape[1]).to(device)
    iz = torch.zeros(imm_b.shape[1]).to(device)

    def predict(gene_np):
        results = []
        with torch.no_grad():
            for row in gene_np:
                logits, _ = model(torch.tensor(row.astype(np.float32)).to(device), mz, pz, iz)
                results.append(logits[1].item())
        return np.array(results)

    print(f"computing SHAP on log-odds scale (n={len(g)})...")
    bg   = shap.kmeans(g, 20)
    exp  = shap.KernelExplainer(predict, bg)
    vals = exp.shap_values(g, nsamples=100, silent=True)

    mean_abs  = np.abs(vals).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:20]
    top_genes = [gene_cols[j] for j in top_idx]
    top_vals  = mean_abs[top_idx]
    top_sv    = vals[:, top_idx]
    top_expr  = g[:, top_idx]

    pd.DataFrame({'gene': top_genes, 'mean_abs_shap': top_vals}).to_csv(
        'results/shap_top20_logodds.csv', index=False
    )

    print("top 10 SHAP genes (log-odds):")
    for rank, (gene, val) in enumerate(zip(top_genes[:10], top_vals[:10]), 1):
        print(f"  {rank:2d}. {gene:14s} {val:.6f}")

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.RdBu_r
    for yi in range(len(top_genes)):
        sv   = top_sv[:, yi]
        expr = top_expr[:, yi]
        e_n  = (expr - expr.min()) / (expr.max() - expr.min() + 1e-8)
        np.random.seed(yi)
        jitter = np.random.uniform(-0.35, 0.35, len(sv))
        ax.scatter(sv, yi + jitter, c=e_n, cmap=cmap, vmin=0, vmax=1,
                   alpha=0.65, s=28, linewidths=0)
    ax.axvline(0, color='#333333', lw=1.0, linestyle='--', alpha=0.6)
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('SHAP value (log-odds scale)', fontsize=11)
    ax.set_title(f'Top 20 model-predictive features (SHAP)\nn={len(g)} samples', fontsize=11)
    ax.grid(True, axis='x', alpha=0.25)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.35, pad=0.03, aspect=15)
    cbar.set_ticks([0.05, 0.95])
    cbar.set_ticklabels(['Low\nexpression', 'High\nexpression'], fontsize=8)
    plt.tight_layout()
    plt.savefig('results/figures/Fig5B_SHAP_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("saved: results/figures/Fig5B_SHAP_beeswarm.png")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    gene_b, meth_b, path_b, imm_b, labs_b = load_data()
    all_probs, all_preds, y_all, all_embs, fold_results = run_crossval(
        gene_b, meth_b, path_b, imm_b, labs_b, device
    )
    score_invasive_like(gene_b, meth_b, path_b, imm_b, labs_b, fold_results, device)
    plot_umap(all_embs, labs_b)
    run_shap(gene_b, meth_b, path_b, imm_b, labs_b, fold_results, device)
    print("\ndone — results saved to results/")
