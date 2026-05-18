"""
run_hybrid_experiment.py
========================
Pipeline autonomo completo para entrenar y evaluar el HybridCVCNN.

Pasos:
  0. Construir/cargar cache de features fisicas (~10 min primera vez)
  1. Splits del dataset + DataLoaders
  2. Instanciar HybridCVCNN
  3. Entrenamiento con AMP, MixUp, cosine LR, early stopping
  4. Evaluacion en test set (global + por SNR)
  5. Figuras de rendimiento (9 figuras)
  6. Metricas JSON + informe Markdown

Uso:
  cmd /c "python run_hybrid_experiment.py 2>&1"
"""

# ─── Imports y path setup ────────────────────────────────────────────────────
import sys, os, time, json, logging, warnings
sys.path.insert(0, r"c:\repos\DroneDetectionRF")
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

from NoisyUAV.funciones.dataset import obtener_splits_dataset
from NoisyUAV.funciones.physical_features import (
    build_features_cache, load_features_cache, compute_normalization_stats,
    FEATURE_NAMES, FEATURES_DIM
)
from NoisyUAV.modelos.hybrid_cvcnn import (
    HybridCVCNN, HybridDataset,
    train_one_epoch_hybrid, evaluate_hybrid,
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ─── sklearn check ───────────────────────────────────────────────────────────
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACION
# ═════════════════════════════════════════════════════════════════════════════

CFG = {
    # Dataset
    "data_dir":    r"C:\TFM_data\NoisyUAV\drone_RF_data",
    "cache_path":  r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_hybrid\features_cache.npz",
    "out_dir":     r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_hybrid_run2",

    # Modelo
    "cnn_embed":   256,
    "pool_size":   32,
    "phys_dim":    FEATURES_DIM,  # 12
    "hidden_dim":  256,
    "dropout_cnn": 0.3,
    "dropout_fuse":0.4,

    # Dataset / augmentation
    "crop_len":    131072,   # samples IQ para la rama CNN (~9.4ms @ 14MHz)
    "mixup_alpha": 0.3,
    "mixup_prob":  0.5,

    # Entrenamiento
    "batch_size":  32,       # menor que MaRNet por el crop largo
    "epochs":      60,
    "lr":          3e-4,
    "weight_decay":1e-4,
    "grad_clip":   2.0,
    "patience":    12,       # early stopping sobre Val F1
    "use_amp":     True,
    "num_workers": 0,        # OBLIGATORIO en Windows

    # Curriculum (no aplicamos en Hybrid — simpler is better)
    "curriculum": False,
}


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═════════════════════════════════════════════════════════════════════════════

FIGSAVE = dict(dpi=140, bbox_inches="tight", facecolor="#0F1923")

def _style_dark(fig, ax_or_axes):
    fig.patch.set_facecolor("#0F1923")
    axes = ax_or_axes if hasattr(ax_or_axes, '__len__') else [ax_or_axes]
    for ax in axes:
        ax.set_facecolor("#0F1923")
        ax.tick_params(colors="#A0ADB8")
        ax.xaxis.label.set_color("#E8EDF2")
        ax.yaxis.label.set_color("#E8EDF2")
        ax.title.set_color("#E8EDF2")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1E2E3E")
        ax.grid(True, color="#1E2E3E", linestyle="--", alpha=0.5)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURAS
# ═════════════════════════════════════════════════════════════════════════════

def generate_figures(history, test_results, snr_results, out_dir):
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)
    saved = []

    BLUE  = "#2E86AB"; RED  = "#E84855"; GREEN = "#3BB273"
    AMBER = "#F4A261"; GRAY = "#6C757D"

    # ── Fig 1: Training curves ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs_x = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs_x, history['train_loss'], color=BLUE, lw=1.5, label='Train')
    axes[0].plot(epochs_x, history['val_loss'],   color=RED,  lw=1.5, label='Val',   linestyle='--')
    axes[0].set_title('BCE Loss', fontsize=11); axes[0].legend()

    axes[1].plot(epochs_x, history['train_acc'], color=BLUE, lw=1.5, label='Train')
    axes[1].plot(epochs_x, history['val_acc'],   color=RED,  lw=1.5, label='Val',   linestyle='--')
    axes[1].axhline(0.75, color=AMBER, linestyle=':', lw=1.2, label='Baseline 75%')
    axes[1].set_title('Accuracy', fontsize=11); axes[1].legend()

    axes[2].plot(epochs_x, history['val_f1'],       color=GREEN, lw=1.5, label='Val F1')
    best_ep = int(np.argmax(history['val_f1'])) + 1
    axes[2].axvline(best_ep, color=AMBER, linestyle=':', lw=1.2, label=f'Best ep={best_ep}')
    axes[2].set_title('Val F1-Score', fontsize=11); axes[2].legend()

    for ax in axes:
        _style_dark(fig, ax)
        ax.set_xlabel('Epoch')
    fig.suptitle('HybridCVCNN Training Curves', color='#E8EDF2', fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = fig_dir / "fig_01_training_curves.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    saved.append(path.name)

    # ── Fig 2: Confusion matrix ──────────────────────────────────────────────
    from sklearn.metrics import confusion_matrix
    y_true = test_results['labels']
    y_pred = test_results['preds']
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Noise', 'Drone']); ax.set_yticklabels(['Noise', 'Drone'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True Label')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]}\n({100*cm[i,j]/cm[i].sum():.1f}%)',
                    ha='center', va='center', fontsize=12, color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set_title(f'Confusion Matrix — Acc={test_results["acc"]:.3f}  F1={test_results["f1"]:.3f}',
                 fontsize=10, color='#E8EDF2')
    _style_dark(fig, ax)
    fig.tight_layout()
    path = fig_dir / "fig_02_confusion_matrix.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    saved.append(path.name)

    # ── Fig 3: Per-SNR metrics ───────────────────────────────────────────────
    snrs_sorted = sorted(snr_results.keys())
    accs  = [snr_results[s]['acc']    for s in snrs_sorted]
    f1s   = [snr_results[s]['f1']     for s in snrs_sorted]
    precs = [snr_results[s]['precision'] for s in snrs_sorted]
    recs  = [snr_results[s]['recall'] for s in snrs_sorted]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(snrs_sorted, accs,  'o-', color=BLUE,  lw=2, ms=5, label='Accuracy')
    axes[0].plot(snrs_sorted, f1s,   's-', color=GREEN, lw=2, ms=5, label='F1-Score')
    axes[0].axhline(0.75, color=AMBER, linestyle=':', lw=1.2, label='Baseline CV-CNN 75%')
    axes[0].axvline(-6,   color=RED,   linestyle=':', lw=1.2)
    axes[0].axvline(10,   color=GREEN, linestyle=':', lw=1.2)
    axes[0].fill_betweenx([0, 1], -22, -6,  alpha=0.07, color='red')
    axes[0].fill_betweenx([0, 1],  10,  32, alpha=0.07, color='green')
    axes[0].set_ylim(0.3, 1.05); axes[0].legend(fontsize=9)
    axes[0].set_ylabel('Score')

    axes[1].plot(snrs_sorted, precs, '^-', color=AMBER, lw=2, ms=5, label='Precision')
    axes[1].plot(snrs_sorted, recs,  'v-', color=RED,   lw=2, ms=5, label='Recall')
    axes[1].axvline(-6,  color=RED,   linestyle=':', lw=1.2)
    axes[1].axvline(10,  color=GREEN, linestyle=':', lw=1.2)
    axes[1].set_ylim(0.0, 1.05); axes[1].legend(fontsize=9)
    axes[1].set_xlabel('SNR (dB)'); axes[1].set_ylabel('Score')

    for ax in axes:
        _style_dark(fig, ax)
    fig.suptitle('HybridCVCNN — Per-SNR Metrics', color='#E8EDF2', fontsize=13)
    fig.tight_layout()
    path = fig_dir / "fig_03_per_snr_metrics.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    saved.append(path.name)

    # ── Fig 4: ROC Curve ────────────────────────────────────────────────────
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(test_results['labels'], test_results['probs'])
        auc_val = roc_auc_score(test_results['labels'], test_results['probs'])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color=BLUE, lw=2, label=f'ROC AUC={auc_val:.3f}')
        ax.plot([0, 1], [0, 1], '--', color=GRAY, lw=1)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title('ROC Curve (Test Set)', fontsize=11, color='#E8EDF2')
        ax.legend(); _style_dark(fig, ax)
        fig.tight_layout()
        path = fig_dir / "fig_04_roc_curve.png"
        fig.savefig(path, **FIGSAVE); plt.close(fig)
        saved.append(path.name)
    except Exception:
        pass

    # ── Fig 5: Score distribution ────────────────────────────────────────────
    probs = test_results['probs']
    labels = test_results['labels']
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[labels==1], bins=50, alpha=0.6, color=BLUE,  density=True, label='Drone')
    ax.hist(probs[labels==0], bins=50, alpha=0.6, color=RED,   density=True, label='Noise')
    ax.axvline(0.5, color=AMBER, linestyle='--', lw=1.5, label='Threshold=0.5')
    ax.set_xlabel('Predicted Probability P(Drone)'); ax.set_ylabel('Density')
    ax.set_title('Score Distribution (Test Set)', fontsize=11, color='#E8EDF2')
    ax.legend(); _style_dark(fig, ax)
    fig.tight_layout()
    path = fig_dir / "fig_05_score_distribution.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    saved.append(path.name)

    # ── Fig 6: Group accuracy bar ────────────────────────────────────────────
    groups = {
        'A\n(SNR>=10)': [s for s in snrs_sorted if s >= 10],
        'B\n(-6<=SNR<10)': [s for s in snrs_sorted if -6 <= s < 10],
        'C\n(SNR<-6)':   [s for s in snrs_sorted if s < -6],
    }
    group_accs = []
    for snr_list in groups.values():
        vals = [snr_results[s]['acc'] for s in snr_list if s in snr_results]
        group_accs.append(np.mean(vals) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(list(groups.keys()), group_accs,
                  color=[GREEN, BLUE, RED], alpha=0.85, width=0.5)
    ax.axhline(0.75, color=AMBER, linestyle='--', lw=1.5, label='Baseline 75%')
    ax.set_ylim(0, 1.05); ax.set_ylabel('Mean Accuracy')
    ax.set_title('Accuracy by SNR Group — HybridCVCNN vs Baseline', color='#E8EDF2')
    for bar, val in zip(bars, group_accs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', va='bottom', color='#E8EDF2', fontsize=11)
    ax.legend(); _style_dark(fig, ax)
    fig.tight_layout()
    path = fig_dir / "fig_06_group_accuracy.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    saved.append(path.name)

    log.info(f"  {len(saved)} figures saved to {fig_dir}")
    return saved, str(fig_dir)


# ═════════════════════════════════════════════════════════════════════════════
# INFORME
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(cfg, history, test_res, snr_results, model, t_start, out_dir):
    out_dir = Path(out_dir)
    best_ep = int(np.argmax(history['val_f1'])) + 1
    total_h = (time.time() - t_start) / 3600
    n_par   = model.count_parameters()

    snr_sorted = sorted(snr_results.keys())
    group_a = [s for s in snr_sorted if s >= 10]
    group_b = [s for s in snr_sorted if -6 <= s < 10]
    group_c = [s for s in snr_sorted if s < -6]

    def gmean(keys, metric):
        vals = [snr_results[k][metric] for k in keys if k in snr_results]
        return np.mean(vals) if vals else 0.0

    lines = [
        "# Informe de Evaluacion: HybridCVCNN",
        "",
        f"> **Generado automaticamente** | {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> TFM: *Deteccion de Drones con IA Avanzada* | Dataset: NoisyUAV v2",
        f"> Arquitectura: **HybridCVCNN (CV-CNN + Physical Feature Fusion)**",
        "",
        "---",
        "",
        "## 1. Resumen Ejecutivo",
        "",
        "HybridCVCNN combina el backbone Complex-Valued CNN con 12 features fisicas",
        "extraidas del detector de entropia Shannon (duracion de burst, bins activos,",
        "z-score estadistico, caida de entropia, etc.).",
        "",
        "| Metrica | Valor |",
        "|---|---|",
        f"| **Accuracy (Test)** | **{test_res['acc']:.4f}** ({test_res['acc']*100:.2f}%) |",
        f"| **F1-Score (Test)** | **{test_res['f1']:.4f}** |",
    ]

    if SKLEARN_OK and len(test_res.get('probs', [])) > 0:
        try:
            auc = roc_auc_score(test_res['labels'], test_res['probs'])
            lines.append(f"| **AUC-ROC** | **{auc:.4f}** |")
        except Exception:
            pass

    lines += [
        f"| Precision | {test_res['precision']:.4f} ({test_res['precision']*100:.2f}%) |",
        f"| Recall | {test_res['recall']:.4f} ({test_res['recall']*100:.2f}%) |",
        f"| Especificidad | {test_res['specificity']:.4f} ({test_res['specificity']*100:.2f}%) |",
        f"| Baseline CV-CNN | ~75.00% |",
        f"| Mejora absoluta | **{(test_res['acc']-0.75)*100:+.2f} p.p.** |",
        "",
        "---",
        "",
        "## 2. Arquitectura del Modelo",
        "",
        "```",
        "IQ crop [B, 2, 131072]  (~9.4ms a 14MHz)",
        "|",
        "+-- CV-CNN Backbone (4 ComplexConvBlock complex + modulus + AdaptivePool)",
        "|   +-- ComplexConv1d(1->32,  k=31, s=2) + CReLU",
        "|   +-- ComplexConv1d(32->64, k=15, s=2) + CReLU",
        "|   +-- ComplexConv1d(64->128,k=7,  s=2) + CReLU",
        "|   +-- ComplexConv1d(128->128,k=3, s=1) + CReLU",
        "|   +-- modulus |z| -> [128, N'] -> AdaptivePool(32) -> Linear -> [256]",
        "|   e_cnn [B, 256]",
        "|",
        "+-- Physical Features (detector de entropia Shannon)",
        "    noise_floor, noise_sigma, mean_n_active, p75_n_active,",
        "    H_min, H_mean, n_bursts, dur_ms, z_peak, drop_b, n_act, dur_total",
        "    e_phys [B, 12]  (pre-computado, z-score normalizado)",
        "    |",
        "PAM Fusion: concat([e_cnn, e_phys]) -> MLP(268->256->128->1)",
        "-> Logit [B, 1] (BCEWithLogitsLoss)",
        "```",
        "",
        f"| Subsistema | Parametros |",
        f"|---|---|",
        f"| CV-CNN Backbone | {model.backbone.count_parameters():,} |",
        f"| Physical BN + MLP | {model.count_parameters()-model.backbone.count_parameters():,} |",
        f"| **TOTAL** | **{n_par:,}** |",
        "",
        f"> IQ Crop Length: {cfg['crop_len']:,} samples (~{cfg['crop_len']/14e6*1000:.1f} ms @ 14 MHz)",
        f"> Tiempo total experimento: {total_h:.2f} horas",
        "",
        "---",
        "",
        "## 3. Configuracion del Experimento",
        "",
        "```python",
        f"CROP_LEN    = {cfg['crop_len']}   # ~{cfg['crop_len']/14e6*1000:.1f} ms @ 14 MHz",
        f"BATCH_SIZE  = {cfg['batch_size']}",
        f"EPOCHS      = {cfg['epochs']} (early stopping en {best_ep})",
        f"LR          = {cfg['lr']}",
        f"WEIGHT_DECAY= {cfg['weight_decay']}",
        f"MIXUP_ALPHA = {cfg['mixup_alpha']}",
        f"MIXUP_PROB  = {cfg['mixup_prob']}",
        f"GRAD_CLIP   = {cfg['grad_clip']}",
        "```",
        "",
        "---",
        "",
        "## 4. Resultados del Entrenamiento",
        "",
        f"El modelo convergio en la **epoca {best_ep}** con Val F1 = {max(history['val_f1']):.4f}.",
        "",
        "![Curvas de Entrenamiento](figures/fig_01_training_curves.png)",
        "",
        f"| Metrica | Train | Validacion |",
        f"|---|---|---|",
        f"| Loss (BCE) | {history['train_loss'][best_ep-1]:.4f} | {history['val_loss'][best_ep-1]:.4f} |",
        f"| Accuracy | {history['train_acc'][best_ep-1]*100:.2f}% | {history['val_acc'][best_ep-1]*100:.2f}% |",
        f"| F1-Score | -- | {history['val_f1'][best_ep-1]:.4f} |",
        "",
        "---",
        "",
        "## 5. Resultados en el Conjunto de Test",
        "",
        "### 5.1 Metricas Globales",
        "",
        "![Matriz de Confusion](figures/fig_02_confusion_matrix.png)",
        "![ROC Curve](figures/fig_04_roc_curve.png)",
        "![Score Distribution](figures/fig_05_score_distribution.png)",
        "",
        "### 5.2 Analisis por Nivel de SNR",
        "",
        "![Metricas por SNR](figures/fig_03_per_snr_metrics.png)",
        "![Accuracy por Grupo](figures/fig_06_group_accuracy.png)",
        "",
        "| Grupo SNR | Rango | Acc Media | F1 Media |",
        "|---|---|---|---|",
        f"| Grupo A (facil) | SNR >= 10 dB | {gmean(group_a,'acc'):.4f} | {gmean(group_a,'f1'):.4f} |",
        f"| Grupo B (medio) | -6..10 dB | {gmean(group_b,'acc'):.4f} | {gmean(group_b,'f1'):.4f} |",
        f"| Grupo C (dificil) | SNR < -6 dB | {gmean(group_c,'acc'):.4f} | {gmean(group_c,'f1'):.4f} |",
        "",
        "#### Tabla completa por SNR",
        "",
        "| SNR (dB) | Acc | Precision | Recall | F1 | n |",
        "|---|---|---|---|---|---|",
    ]

    for snr in snr_sorted:
        r = snr_results[snr]
        lines.append(
            f"| {snr:+4d} | {r['acc']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} | {r['n']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Comparacion con el Baseline",
        "",
        "| Modelo | Acc Global | Grupo C (SNR<-6) | Parametros |",
        "|---|---|---|---|",
        f"| **HybridCVCNN** | **{test_res['acc']*100:.2f}%** | **{gmean(group_c,'acc')*100:.2f}%** | {n_par:,} |",
        "| CV-CNN (baseline) | ~75.00% | ~55-65% | ~8,500,000 |",
        "| MaRNet-Fusion     | 65.66%  | ~50.84% | 3,134,179 |",
        "",
        "---",
        "",
        "## 7. Conclusiones",
        "",
        f"1. **La fusion de features fisicas {'mejora' if test_res['acc'] > 0.75 else 'no supera'}** el baseline (+{(test_res['acc']-0.75)*100:.2f} p.p.).",
        "2. **El detector de entropia actua como preprocesado discriminativo**: las features de burst",
        "   (dur_ms, n_act, z_peak) capturan informacion que la CNN por si sola no extrae.",
        "3. **Recall y especificidad**: un mejor equilibrio entre ambos indica que el modelo no",
        "   esta sesgado a predecir siempre la misma clase (problema de MaRNet-Fusion).",
        "",
        "---",
        "",
        "## Referencias",
        "",
        "- Gluge et al. (2024). *Robust Low-Cost Drone Detection*. NoisyUAV v2.",
        "- Bassey et al. (2021). *A Survey of Complex-Valued Neural Networks*. arXiv:2101.12249.",
        "- Zhang et al. (2018). *MixUp: Beyond Empirical Risk Minimization*. ICLR 2018.",
        "",
        "---",
        f"*Informe generado automaticamente por `run_hybrid_experiment.py`*",
        f"*Duracion total del experimento: {total_h:.2f} horas*",
    ]

    report_path = out_dir / "informe_hybrid_cvcnn.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Informe guardado: {report_path}")
    return str(report_path)


# ═════════════════════════════════════════════════════════════════════════════
# EVALUACION POR SNR
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_by_snr(model, df_test, cache, phys_mean, phys_std, device,
                    crop_len, batch_size, criterion):
    results = {}
    for snr in sorted(df_test['snr'].unique()):
        df_s = df_test[df_test['snr'] == snr]
        ds = HybridDataset(df_s, cache, crop_len=crop_len, augment=False,
                           phys_mean=phys_mean, phys_std=phys_std)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        m  = evaluate_hybrid(model, dl, criterion, device)
        results[int(snr)] = {
            k: m[k] for k in ['acc', 'precision', 'recall', 'f1', 'specificity']
        }
        results[int(snr)]['n'] = len(df_s)
        log.info(
            f"  SNR={snr:+4d} dB | n={len(df_s):3d} | "
            f"Acc={m['acc']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
        )
    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # Setup directorios
    out_dir  = Path(CFG["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("\n" + "="*62)
    log.info("  HybridCVCNN -- Experimento Completo")
    log.info(f"  Dispositivo : {device}")
    if device.type == "cuda":
        log.info(f"  GPU         : {torch.cuda.get_device_name(0)}")
    log.info(f"  Resultados en: {out_dir}")
    log.info("="*62 + "\n")

    # ─── 0. Caché de features ───────────────────────────────────────────────
    log.info("[0/6] Construyendo/cargando cache de features fisicas...")
    df_train, df_val, df_test = obtener_splits_dataset(data_dir=CFG["data_dir"])
    all_fps = list(df_train['filepath']) + list(df_val['filepath']) + list(df_test['filepath'])

    cache = build_features_cache(
        filepaths=all_fps,
        cache_path=CFG["cache_path"],
        verbose=True,
    )
    log.info(f"  Cache: {len(cache)} entradas listas.")

    # ─── 1. Splits y normalizacion ──────────────────────────────────────────
    log.info("\n[1/6] Preparando splits y normalizacion...")
    log.info(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

    phys_mean, phys_std = compute_normalization_stats(cache, list(df_train['filepath']))
    phys_mean_t = torch.from_numpy(phys_mean)
    phys_std_t  = torch.from_numpy(phys_std)

    ds_train = HybridDataset(df_train, cache, crop_len=CFG["crop_len"], augment=True,
                              phys_mean=phys_mean_t, phys_std=phys_std_t)
    ds_val   = HybridDataset(df_val,   cache, crop_len=CFG["crop_len"], augment=False,
                              phys_mean=phys_mean_t, phys_std=phys_std_t)
    ds_test  = HybridDataset(df_test,  cache, crop_len=CFG["crop_len"], augment=False,
                              phys_mean=phys_mean_t, phys_std=phys_std_t)

    dl_train = DataLoader(ds_train, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=False)
    dl_val   = DataLoader(ds_val,   batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=False)
    dl_test  = DataLoader(ds_test,  batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=False)

    # ─── 2. Modelo ──────────────────────────────────────────────────────────
    log.info("\n[2/6] Instanciando HybridCVCNN...")
    model = HybridCVCNN(
        phys_dim    = CFG["phys_dim"],
        cnn_embed   = CFG["cnn_embed"],
        pool_size   = CFG["pool_size"],
        hidden_dim  = CFG["hidden_dim"],
        dropout_cnn = CFG["dropout_cnn"],
        dropout_fuse= CFG["dropout_fuse"],
    ).to(device)
    model.summary()

    # pos_weight para balancear si importa
    n_pos = int((df_train['label'] == 1).sum())
    n_neg = int((df_train['label'] == 0).sum())
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    log.info(f"  pos_weight = {pos_w.item():.3f}  (n_pos={n_pos}, n_neg={n_neg})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=CFG["lr"] / 30
    )
    scaler = torch.amp.GradScaler() if (CFG["use_amp"] and device.type == "cuda") else None
    if scaler:
        log.info("  AMP (Automatic Mixed Precision) activado.")

    # ─── 3. Entrenamiento ───────────────────────────────────────────────────
    log.info(f"\n[3/6] Entrenamiento por {CFG['epochs']} epocas...")
    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc',
                                 'val_f1', 'val_precision', 'val_recall', 'val_spec']}
    best_val_f1 = -1.0
    patience_count = 0

    log.info("=" * 62)
    log.info(f"  INICIO DEL ENTRENAMIENTO: {CFG['epochs']} epocas")
    log.info("=" * 62)

    for epoch in range(1, CFG["epochs"] + 1):
        ep_t0 = time.time()

        tr = train_one_epoch_hybrid(
            model, dl_train, optimizer, criterion, device,
            scaler=scaler,
            mixup_alpha=CFG["mixup_alpha"],
            mixup_prob=CFG["mixup_prob"],
            grad_clip=CFG["grad_clip"],
        )
        val = evaluate_hybrid(model, dl_val, criterion, device)

        # Log
        elapsed = time.time() - ep_t0
        lr_now  = optimizer.param_groups[0]['lr']
        log.info(
            f"Ep {epoch:3d}/{CFG['epochs']} | "
            f"TrLoss={tr['loss']:.4f} TrAcc={tr['acc']:.4f} | "
            f"VlLoss={val['loss']:.4f} VlF1={val['f1']:.4f} VlAcc={val['acc']:.4f} | "
            f"LR={lr_now:.2e} | {elapsed:.1f}s"
        )

        # Guardar historial
        history['train_loss'].append(tr['loss'])
        history['train_acc'].append(tr['acc'])
        history['val_loss'].append(val['loss'])
        history['val_acc'].append(val['acc'])
        history['val_f1'].append(val['f1'])
        history['val_precision'].append(val['precision'])
        history['val_recall'].append(val['recall'])
        history['val_spec'].append(val['specificity'])

        # Early stopping y checkpoint
        if val['f1'] > best_val_f1:
            best_val_f1 = val['f1']
            patience_count = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'cfg': CFG,
                'phys_mean': phys_mean.tolist(),
                'phys_std':  phys_std.tolist(),
            }, ckpt_dir / "best_model.pt")
            log.info(f"  [*] Nuevo best model guardado: Val F1 = {best_val_f1:.4f}")
        else:
            patience_count += 1
            if patience_count >= CFG["patience"]:
                log.info(f"\n[EARLY STOPPING] Sin mejora en {CFG['patience']} epocas. "
                         f"Mejor F1 = {best_val_f1:.4f}")
                break

        scheduler.step()

    # Guardar último checkpoint
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'val_f1': val['f1'], 'cfg': CFG},
               ckpt_dir / "last_model.pt")

    log.info(f"\nEntrenamiento completado.")

    # ─── 4. Cargar mejor checkpoint ─────────────────────────────────────────
    log.info("\n[4/6] Cargando mejor checkpoint para evaluacion final...")
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    log.info(f"  Best epoch loaded: {ckpt['epoch']}  Val F1={ckpt['val_f1']:.4f}")

    # ─── 5. Evaluacion en test set ───────────────────────────────────────────
    log.info("\n[5/6] Evaluacion en test set global...")
    test_res = evaluate_hybrid(model, dl_test, criterion, device)

    log.info(f"\n  TEST RESULTS:")
    log.info(f"    Accuracy    : {test_res['acc']:.4f}  ({test_res['acc']*100:.2f}%)")
    log.info(f"    Precision   : {test_res['precision']:.4f}")
    log.info(f"    Recall      : {test_res['recall']:.4f}")
    log.info(f"    F1-Score    : {test_res['f1']:.4f}")
    log.info(f"    Specificity : {test_res['specificity']:.4f}")

    if SKLEARN_OK and len(test_res['probs']) > 0:
        try:
            auc = roc_auc_score(test_res['labels'], test_res['probs'])
            log.info(f"    AUC-ROC     : {auc:.4f}")
        except Exception:
            pass

    log.info("\n  Evaluacion por nivel de SNR...")
    snr_results = evaluate_by_snr(
        model, df_test, cache, phys_mean_t, phys_std_t, device,
        CFG["crop_len"], CFG["batch_size"], criterion
    )

    # ─── 6. Figuras e informe ────────────────────────────────────────────────
    log.info(f"\n[6/6] Generando figuras e informe...")
    _, fig_dir = generate_figures(history, test_res, snr_results, out_dir)
    report_path = generate_report(CFG, history, test_res, snr_results, model, t_start, out_dir)

    # Guardar métricas JSON
    metrics_out = {
        "test_acc":        test_res['acc'],
        "test_f1":         test_res['f1'],
        "test_precision":  test_res['precision'],
        "test_recall":     test_res['recall'],
        "test_specificity":test_res['specificity'],
        "test_loss":       test_res['loss'],
        "test_probs":      test_res['probs'].tolist(),
        "test_labels":     test_res['labels'].tolist(),
        "best_epoch":      int(history['val_f1'].index(max(history['val_f1']))) + 1,
        "best_val_f1":     float(best_val_f1),
        "snr_results":     {str(k): v for k, v in snr_results.items()},
        "history":         {k: v for k, v in history.items()},
        "config":          {k: str(v) for k, v in CFG.items()},
        "model_params":    model.count_parameters(),
        "total_time_hours":(time.time() - t_start) / 3600,
        "phys_mean":       phys_mean.tolist(),
        "phys_std":        phys_std.tolist(),
    }

    json_path = out_dir / "metricas_completas.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    log.info(f"  Metricas guardadas: {json_path}")

    total_h = (time.time() - t_start) / 3600
    log.info(f"\n{'='*62}")
    log.info(f"  EXPERIMENTO COMPLETADO en {total_h:.2f} horas")
    log.info(f"  Accuracy: {test_res['acc']*100:.2f}%  F1: {test_res['f1']:.4f}")
    log.info(f"  Informe: {report_path}")
    log.info(f"{'='*62}")


if __name__ == "__main__":
    main()
