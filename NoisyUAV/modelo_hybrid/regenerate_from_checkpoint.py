"""
regenerate_from_checkpoint.py
==============================
Script independiente para regenerar TODAS las figuras y el informe
a partir del mejor checkpoint guardado, sin necesidad de re-entrenar.

Uso:
    cmd /c "python regenerate_from_checkpoint.py"
    cmd /c "python regenerate_from_checkpoint.py --threshold 0.45"

Requisitos:
    - resultados_hybrid/checkpoints/best_model.pt  (pesos del mejor modelo)
    - resultados_hybrid/features_cache.npz          (cache de features fisicas)
    - Dataset en C:\\TFM_data\\NoisyUAV\\drone_RF_data (para obtener splits)

Genera:
    - resultados_hybrid/figures/          (todas las figuras PNG)
    - resultados_hybrid/informe_hybrid_cvcnn.md
    - resultados_hybrid/metricas_completas.json  (actualizado)
"""

import sys, os, json, time, argparse
sys.path.insert(0, r"c:\repos\DroneDetectionRF")
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

from NoisyUAV.funciones.dataset import obtener_splits_dataset
from NoisyUAV.funciones.physical_features import load_features_cache, FEATURES_DIM
from NoisyUAV.modelos.hybrid_cvcnn import HybridCVCNN, HybridDataset, evaluate_hybrid

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION POR DEFECTO
# ─────────────────────────────────────────────────────────────────────────────

CKPT_PATH  = r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_hybrid\checkpoints\best_model.pt"
CACHE_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_hybrid\features_cache.npz"
OUT_DIR    = r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_hybrid"
DATA_DIR   = r"C:\TFM_data\NoisyUAV\drone_RF_data"


# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint_and_model(ckpt_path, device):
    """Carga el checkpoint y reconstruye el modelo."""
    print(f"  Cargando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt.get('cfg', {})
    model = HybridCVCNN(
        phys_dim    = int(cfg.get('phys_dim',    FEATURES_DIM)),
        cnn_embed   = int(cfg.get('cnn_embed',   256)),
        pool_size   = int(cfg.get('pool_size',   32)),
        hidden_dim  = int(cfg.get('hidden_dim',  256)),
        dropout_cnn = float(cfg.get('dropout_cnn',  0.3)),
        dropout_fuse= float(cfg.get('dropout_fuse', 0.4)),
    ).to(device)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    phys_mean = torch.tensor(ckpt['phys_mean'], dtype=torch.float32)
    phys_std  = torch.tensor(ckpt['phys_std'],  dtype=torch.float32)

    print(f"  Epoch: {ckpt['epoch']} | Val F1: {ckpt['val_f1']:.4f}")
    print(f"  Parametros: {model.count_parameters():,}")
    model.summary()

    return model, ckpt, phys_mean, phys_std, cfg


@torch.no_grad()
def evaluate_by_snr(model, df_test, cache, phys_mean, phys_std, device,
                    crop_len=131072, batch_size=32, threshold=0.5):
    """Evaluacion detallada por nivel de SNR."""
    criterion = nn.BCEWithLogitsLoss()
    results = {}
    for snr in sorted(df_test['snr'].unique()):
        df_s = df_test[df_test['snr'] == snr]
        ds = HybridDataset(df_s, cache, crop_len=crop_len, augment=False,
                           phys_mean=phys_mean, phys_std=phys_std)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        m  = evaluate_hybrid(model, dl, criterion, device, threshold=threshold)
        results[int(snr)] = {k: m[k] for k in ['acc', 'precision', 'recall', 'f1', 'specificity']}
        results[int(snr)]['n'] = len(df_s)
        print(
            f"  SNR={snr:+4d} dB | n={len(df_s):3d} | "
            f"Acc={m['acc']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} Spec={m['specificity']:.4f}"
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(test_res, snr_results, out_dir, ckpt):
    """Genera todas las figuras de evaluacion."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)

    BLUE  = "#2E86AB"; RED   = "#E84855"; GREEN = "#3BB273"
    AMBER = "#F4A261"; GRAY  = "#6C757D"; PURPLE= "#6A0572"
    FIGSAVE = dict(dpi=140, bbox_inches="tight", facecolor="#0F1923")

    def _dark(fig, axes):
        ax_list = axes if hasattr(axes, '__iter__') else [axes]
        for ax in ax_list:
            fig.patch.set_facecolor("#0F1923")
            ax.set_facecolor("#0F1923")
            ax.tick_params(colors="#A0ADB8")
            ax.xaxis.label.set_color("#E8EDF2")
            ax.yaxis.label.set_color("#E8EDF2")
            ax.title.set_color("#E8EDF2")
            for sp in ax.spines.values():
                sp.set_edgecolor("#1E2E3E")
            ax.grid(True, color="#1E2E3E", linestyle="--", alpha=0.5)

    saved = []
    y_true = test_res['labels']
    y_pred = test_res['preds']
    y_prob = test_res['probs']
    snrs   = sorted(snr_results.keys())

    # ── Fig 1: Confusion Matrix ──────────────────────────────────────────────
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Noise','Drone'], fontsize=12)
    ax.set_yticks([0,1]); ax.set_yticklabels(['Noise','Drone'], fontsize=12)
    ax.set_xlabel('Predicted', fontsize=11); ax.set_ylabel('True Label', fontsize=11)
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i,j] > cm.max()*0.7 else '#E8EDF2'
            ax.text(j, i, f'{cm[i,j]}\n({100*cm[i,j]/max(cm[i].sum(),1):.1f}%)',
                    ha='center', va='center', fontsize=13, color=color, fontweight='bold')
    acc = test_res['acc']; f1 = test_res['f1']
    ax.set_title(f'Confusion Matrix\nAcc={acc:.4f} | F1={f1:.4f} | Spec={test_res["specificity"]:.4f}',
                 fontsize=10)
    _dark(fig, ax); fig.tight_layout()
    p = fig_dir / "fig_01_confusion_matrix.png"
    fig.savefig(p, **FIGSAVE); plt.close(fig); saved.append(p.name)

    # ── Fig 2: ROC Curve ────────────────────────────────────────────────────
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_roc = roc_auc_score(y_true, y_prob)
        prec_r, rec_r, _ = precision_recall_curve(y_true, y_prob)
        from sklearn.metrics import average_precision_score
        auc_pr = average_precision_score(y_true, y_prob)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        # ROC
        axes[0].plot(fpr, tpr, color=BLUE, lw=2.5, label=f'HybridCVCNN (AUC={auc_roc:.4f})')
        axes[0].plot([0,1],[0,1], '--', color=GRAY, lw=1, label='Random')
        axes[0].fill_between(fpr, tpr, alpha=0.1, color=BLUE)
        axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve'); axes[0].legend(fontsize=9)
        # Precision-Recall
        axes[1].plot(rec_r, prec_r, color=GREEN, lw=2.5, label=f'AP={auc_pr:.4f}')
        axes[1].axhline(y_true.mean(), color=GRAY, linestyle='--', lw=1, label=f'Baseline={y_true.mean():.3f}')
        axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve'); axes[1].legend(fontsize=9)
        for ax in axes:
            _dark(fig, ax)
        fig.tight_layout()
        p = fig_dir / "fig_02_roc_pr_curves.png"
        fig.savefig(p, **FIGSAVE); plt.close(fig); saved.append(p.name)
    except Exception as e:
        print(f"  [WARN] ROC figure failed: {e}")

    # ── Fig 3: Score Distribution ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y_prob[y_true==1], bins=60, alpha=0.65, color=BLUE,  density=True, label='Drone (1)')
    ax.hist(y_prob[y_true==0], bins=60, alpha=0.65, color=RED,   density=True, label='Noise (0)')
    ax.axvline(0.5, color=AMBER, linestyle='--', lw=2, label='Threshold=0.5')
    ax.set_xlabel('P(Drone) Score', fontsize=11); ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Score Distribution — Overlapping regions = uncertainty', fontsize=11)
    ax.legend(fontsize=10); _dark(fig, ax); fig.tight_layout()
    p = fig_dir / "fig_03_score_distribution.png"
    fig.savefig(p, **FIGSAVE); plt.close(fig); saved.append(p.name)

    # ── Fig 4: Per-SNR Accuracy & F1 ────────────────────────────────────────
    accs  = [snr_results[s]['acc']        for s in snrs]
    f1s   = [snr_results[s]['f1']         for s in snrs]
    precs = [snr_results[s]['precision']  for s in snrs]
    recs  = [snr_results[s]['recall']     for s in snrs]
    specs = [snr_results[s]['specificity']for s in snrs]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(snrs, accs, 'o-', color=BLUE,  lw=2, ms=5, label='Accuracy')
    axes[0].plot(snrs, f1s,  's-', color=GREEN, lw=2, ms=5, label='F1-Score')
    axes[0].axhline(0.75,  color=AMBER, linestyle=':', lw=1.5, label='Baseline CV-CNN 75%')
    axes[0].axhline(0.6566,color=RED,   linestyle=':', lw=1.0, label='MaRNet-Fusion 65.66%')
    axes[0].set_ylabel('Score'); axes[0].legend(fontsize=9); axes[0].set_ylim(0.2, 1.05)

    axes[1].plot(snrs, precs, '^-', color=AMBER, lw=2, ms=5, label='Precision')
    axes[1].plot(snrs, recs,  'v-', color=RED,   lw=2, ms=5, label='Recall')
    axes[1].set_ylabel('Score'); axes[1].legend(fontsize=9); axes[1].set_ylim(0.0, 1.05)

    axes[2].plot(snrs, specs, 'D-', color=PURPLE, lw=2, ms=5, label='Specificity')
    axes[2].axhline(0.5, color=GRAY, linestyle=':', lw=1, label='Random')
    axes[2].set_xlabel('SNR (dB)'); axes[2].set_ylabel('Specificity')
    axes[2].legend(fontsize=9); axes[2].set_ylim(0.0, 1.05)

    for ax in axes:
        ax.axvline(-6,  color='orange', linestyle=':', lw=1.2, alpha=0.7)
        ax.axvline(10,  color='lime',   linestyle=':', lw=1.2, alpha=0.7)
        ax.fill_betweenx([0,1.1], -22, -6,  alpha=0.06, color='red')
        ax.fill_betweenx([0,1.1],  10,  32, alpha=0.06, color='green')
        _dark(fig, ax)

    fig.suptitle('HybridCVCNN — Metricas por Nivel de SNR', color='#E8EDF2',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p = fig_dir / "fig_04_per_snr_metrics.png"
    fig.savefig(p, **FIGSAVE); plt.close(fig); saved.append(p.name)

    # ── Fig 5: Group accuracy comparison bar ─────────────────────────────────
    groups = {
        'A\n(SNR>=10)':     [s for s in snrs if s >= 10],
        'B\n(-6<=SNR<10)': [s for s in snrs if -6 <= s < 10],
        'C\n(SNR<-6)':     [s for s in snrs if s < -6],
    }
    def gmean(keys, m):
        v = [snr_results[k][m] for k in keys if k in snr_results]
        return np.mean(v) if v else 0.0

    hybrid_accs  = [gmean(v,'acc') for v in groups.values()]
    baseline_accs = [0.75, 0.70, 0.65]   # estimated from memory.md
    marnet_accs   = [0.7159, 0.6718, 0.5084]

    x = np.arange(len(groups))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w,   hybrid_accs,   w, color=BLUE,  alpha=0.85, label='HybridCVCNN (nuevo)')
    b2 = ax.bar(x,        baseline_accs, w, color=AMBER, alpha=0.85, label='CV-CNN baseline')
    b3 = ax.bar(x + w,   marnet_accs,   w, color=RED,   alpha=0.85, label='MaRNet-Fusion')
    ax.set_xticks(x); ax.set_xticklabels(list(groups.keys()), fontsize=11)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Mean Accuracy', fontsize=11)
    ax.set_title('Comparacion de Modelos por Grupo SNR', fontsize=12)
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                    color='#E8EDF2', fontsize=9, fontweight='bold')
    ax.legend(fontsize=10); _dark(fig, ax); fig.tight_layout()
    p = fig_dir / "fig_05_group_comparison.png"
    fig.savefig(p, **FIGSAVE); plt.close(fig); saved.append(p.name)

    # ── Fig 6: Threshold sweep ───────────────────────────────────────────────
    thresholds = np.linspace(0.1, 0.9, 81)
    f1_sweep, acc_sweep, spec_sweep = [], [], []
    for t in thresholds:
        pred_t = (y_prob > t).astype(int)
        from sklearn.metrics import f1_score, accuracy_score
        f1_sweep.append(f1_score(y_true, pred_t, zero_division=0))
        acc_sweep.append(accuracy_score(y_true, pred_t))
        tn = ((y_true==0)&(pred_t==0)).sum()
        fp = ((y_true==0)&(pred_t==1)).sum()
        spec_sweep.append(tn/(tn+fp) if (tn+fp)>0 else 0.0)

    best_t = thresholds[np.argmax(f1_sweep)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(thresholds, f1_sweep,   color=GREEN, lw=2, label='F1-Score')
    ax.plot(thresholds, acc_sweep,  color=BLUE,  lw=2, label='Accuracy')
    ax.plot(thresholds, spec_sweep, color=RED,   lw=2, label='Specificity')
    ax.axvline(0.5,    color=GRAY,  linestyle='--', lw=1.5, label='Default thr=0.5')
    ax.axvline(best_t, color=AMBER, linestyle='--', lw=1.5,
               label=f'Optimal thr={best_t:.2f} (F1={max(f1_sweep):.4f})')
    ax.set_xlabel('Classification Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Threshold Sweep — Optimal Decision Boundary', fontsize=11)
    ax.legend(fontsize=9); _dark(fig, ax); fig.tight_layout()
    p = fig_dir / "fig_06_threshold_sweep.png"
    fig.savefig(p, **FIGSAVE); plt.close(fig); saved.append(p.name)

    print(f"  {len(saved)} figuras guardadas en {fig_dir}")
    return saved, str(fig_dir)


def generate_report(test_res, snr_results, out_dir, ckpt, cfg):
    """Genera el informe Markdown completo."""
    out_dir = Path(out_dir)
    snrs = sorted(snr_results.keys())
    group_a = [s for s in snrs if s >= 10]
    group_b = [s for s in snrs if -6 <= s < 10]
    group_c = [s for s in snrs if s < -6]
    def gmean(keys, m):
        v = [snr_results[k][m] for k in keys if k in snr_results]
        return np.mean(v) if v else 0.0

    try:
        auc = roc_auc_score(test_res['labels'], test_res['probs'])
    except Exception:
        auc = 0.0

    crop_len = int(cfg.get('crop_len', 131072))

    lines = [
        "# Informe de Evaluacion: HybridCVCNN",
        f"> Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> Checkpoint epoch: {ckpt['epoch']} | Val F1: {ckpt['val_f1']:.4f}  ",
        "> Arquitectura: **HybridCVCNN (CV-CNN + Physical Feature Fusion)**",
        "",
        "---",
        "",
        "## 1. Resumen Ejecutivo",
        "",
        "| Metrica | Valor |",
        "|---|---|",
        f"| **Accuracy (Test)** | **{test_res['acc']:.4f}** ({test_res['acc']*100:.2f}%) |",
        f"| **F1-Score (Test)** | **{test_res['f1']:.4f}** |",
        f"| **AUC-ROC** | **{auc:.4f}** |",
        f"| Precision | {test_res['precision']:.4f} ({test_res['precision']*100:.2f}%) |",
        f"| Recall | {test_res['recall']:.4f} ({test_res['recall']*100:.2f}%) |",
        f"| Especificidad | {test_res['specificity']:.4f} ({test_res['specificity']*100:.2f}%) |",
        f"| Baseline CV-CNN | ~75.00% |",
        f"| MaRNet-Fusion | 65.66% |",
        f"| **Mejora vs Baseline** | **{(test_res['acc']-0.75)*100:+.2f} p.p.** |",
        "",
        "---",
        "",
        "## 2. Resultados por Grupo SNR",
        "",
        "![Metricas por SNR](figures/fig_04_per_snr_metrics.png)",
        "![Comparacion de Modelos](figures/fig_05_group_comparison.png)",
        "",
        "| Grupo SNR | Rango | Acc | F1 | Precision | Recall |",
        "|---|---|---|---|---|---|",
        f"| A (facil)   | SNR>=10  | {gmean(group_a,'acc'):.4f} | {gmean(group_a,'f1'):.4f} | {gmean(group_a,'precision'):.4f} | {gmean(group_a,'recall'):.4f} |",
        f"| B (medio)   | -6..10   | {gmean(group_b,'acc'):.4f} | {gmean(group_b,'f1'):.4f} | {gmean(group_b,'precision'):.4f} | {gmean(group_b,'recall'):.4f} |",
        f"| C (dificil) | SNR<-6   | {gmean(group_c,'acc'):.4f} | {gmean(group_c,'f1'):.4f} | {gmean(group_c,'precision'):.4f} | {gmean(group_c,'recall'):.4f} |",
        "",
        "---",
        "",
        "## 3. Figuras de Evaluacion",
        "",
        "![Confusion Matrix](figures/fig_01_confusion_matrix.png)",
        "![ROC y PR Curves](figures/fig_02_roc_pr_curves.png)",
        "![Score Distribution](figures/fig_03_score_distribution.png)",
        "![Threshold Sweep](figures/fig_06_threshold_sweep.png)",
        "",
        "---",
        "",
        "## 4. Tabla Completa por SNR",
        "",
        "| SNR (dB) | Acc | Prec | Recall | F1 | Spec | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in snrs:
        r = snr_results[s]
        lines.append(
            f"| {s:+4d} | {r['acc']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} | {r['specificity']:.4f} | {r['n']} |"
        )

    lines += [
        "",
        "---",
        f"*Generado por `regenerate_from_checkpoint.py`*",
    ]

    path = out_dir / "informe_hybrid_cvcnn.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Informe guardado: {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Regenerate HybridCVCNN results from checkpoint')
    parser.add_argument('--ckpt',      default=CKPT_PATH,  help='Path to best_model.pt')
    parser.add_argument('--cache',     default=CACHE_PATH,  help='Path to features_cache.npz')
    parser.add_argument('--out',       default=OUT_DIR,     help='Output directory')
    parser.add_argument('--data_dir',  default=DATA_DIR,    help='Dataset root')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold')
    parser.add_argument('--batch_size',type=int,   default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  Regenerating HybridCVCNN Results from Checkpoint")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Threshold: {args.threshold}")
    print(f"{'='*60}\n")

    # 1. Cargar modelo desde checkpoint
    model, ckpt, phys_mean, phys_std, cfg = load_checkpoint_and_model(args.ckpt, device)

    # 2. Cargar cache de features
    print("\n  Cargando features cache...")
    cache = load_features_cache(args.cache)
    print(f"  Cache: {len(cache)} entradas")

    # 3. Obtener test split (misma semilla = mismo split)
    print("\n  Cargando dataset splits (random_state=42)...")
    _, _, df_test = obtener_splits_dataset(data_dir=args.data_dir)
    print(f"  Test set: {len(df_test):,} muestras")

    # 4. Evaluar en test set global
    crop_len = int(cfg.get('crop_len', 131072))
    ds_test = HybridDataset(df_test, cache, crop_len=crop_len, augment=False,
                             phys_mean=phys_mean, phys_std=phys_std)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("\n  Evaluando en test set global...")
    criterion = nn.BCEWithLogitsLoss()
    test_res = evaluate_hybrid(model, dl_test, criterion, device, threshold=args.threshold)

    print(f"\n  RESULTADOS GLOBALES:")
    print(f"    Accuracy    : {test_res['acc']:.4f} ({test_res['acc']*100:.2f}%)")
    print(f"    Precision   : {test_res['precision']:.4f}")
    print(f"    Recall      : {test_res['recall']:.4f}")
    print(f"    F1-Score    : {test_res['f1']:.4f}")
    print(f"    Specificity : {test_res['specificity']:.4f}")
    if SKLEARN_OK and len(test_res['probs']) > 0:
        try:
            print(f"    AUC-ROC     : {roc_auc_score(test_res['labels'], test_res['probs']):.4f}")
        except Exception:
            pass

    # 5. Evaluar por SNR
    print("\n  Evaluando por nivel de SNR...")
    snr_results = evaluate_by_snr(model, df_test, cache, phys_mean, phys_std, device,
                                   crop_len=crop_len, batch_size=args.batch_size,
                                   threshold=args.threshold)

    # 6. Figuras e informe
    print("\n  Generando figuras...")
    generate_all_figures(test_res, snr_results, args.out, ckpt)
    generate_report(test_res, snr_results, args.out, ckpt, cfg)

    # 7. Actualizar JSON de metricas
    out_dir = Path(args.out)
    metrics = {
        "test_acc":         test_res['acc'],
        "test_f1":          test_res['f1'],
        "test_precision":   test_res['precision'],
        "test_recall":      test_res['recall'],
        "test_specificity": test_res['specificity'],
        "threshold":        args.threshold,
        "test_probs":       test_res['probs'].tolist(),
        "test_labels":      test_res['labels'].tolist(),
        "snr_results":      {str(k): v for k, v in snr_results.items()},
        "checkpoint_epoch": ckpt['epoch'],
        "val_f1":           ckpt['val_f1'],
        "regenerated_at":   time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json_path = out_dir / "metricas_regenerated.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metricas JSON: {json_path}")

    print(f"\n{'='*60}")
    print(f"  DONE — Resultados regenerados desde epoca {ckpt['epoch']}")
    print(f"  Acc={test_res['acc']:.4f} | F1={test_res['f1']:.4f} | Spec={test_res['specificity']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
