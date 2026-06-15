import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             f1_score, precision_score, precision_recall_curve,
                             average_precision_score, auc)

# --- Estilo académico unificado ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})

TARGET_NAMES_FIXED = {
    0: "DJI (T0)", 1: "FutabaT14 (T1)", 2: "FutabaT7 (T2)",
    3: "Graupner (T3)", 5: "Taranis (T5)", 6: "Turnigy (T6)",
    4: "Ruido (T4)",
}
TARGET_COLORS_FIXED = {
    0: "#1A237E",
    1: "#D62828", 2: "#118AB2",
    3: "#2D6A4F", 5: "#E07C00", 6: "#7B2D8B",
}
FIGSAVE_FIXED = dict(dpi=300, bbox_inches="tight", format="pdf")

# --- Rutas ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from NoisyUAV.modelo_v2_1_dual.model import DualStreamCVCNN
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

GOLDEN_CSV = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
DATA_DIR   = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CKPT_PATH  = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual\checkpoints\best_model.pth"
OUT_DIR    = r"C:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\DualStream-SlidingWindow"

FS      = 14e6
WIN_LEN = 131072
N_STEPS = 16


def calculate_phys_features(iq_tensor, adaptive_window_ms):
    """Extrae los 3 parámetros físicos que espera el modelo v2.1 (global CFAR)."""
    try:
        _, _, H_smooth, _, nf_v, ns, _, bursts = detectar_bursts(
            iq_tensor, fs=FS, nperseg=2048,
            adaptive_window_ms=adaptive_window_ms, min_burst_ms=0.3, z_thresh=2.5
        )
        global_nf = float(np.median(nf_v))
        global_H  = float(np.mean(H_smooth))
        if bursts:
            z_peak = max([abs(b['z_peak']) for b in bursts])
        else:
            z_peak = float((np.min(H_smooth) - global_nf) / (ns + 1e-10))
        z_peak = float(np.clip(abs(z_peak), 0, 30))
        return [global_nf, global_H, z_peak]
    except Exception:
        return [0.0, 0.0, 0.0]


def main():
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Cargando DualStream-SlidingWindow (v2.1) desde {CKPT_PATH}...")
    model = DualStreamCVCNN().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    df_golden = pd.read_csv(GOLDEN_CSV)
    print(f"Golden Set: {len(df_golden)} ficheros")

    csv_path = os.path.join(OUT_DIR, "golden_results_dualstream_sliding.csv")
    if os.path.exists(csv_path):
        print(f"Cargando resultados previos desde {csv_path}...")
        df_res = pd.read_csv(csv_path)
    else:
        results = []
        for _, row in tqdm(df_golden.iterrows(), total=len(df_golden),
                           desc="Inferencia DualStream-SlidingWindow"):
            fpath = os.path.join(DATA_DIR, row['filename'])
            if not os.path.exists(fpath):
                continue
            try:
                d = torch.load(fpath, map_location='cpu', weights_only=False)
                iq_full = d['x_iq'].float()
                max_idx = iq_full.shape[1]

                # CFAR global sobre los 75 ms completos
                phys_vals = calculate_phys_features(iq_full, adaptive_window_ms=15.0)
                phys_tensor = torch.tensor([phys_vals], dtype=torch.float32).to(device)

                step = (max_idx - WIN_LEN) // (N_STEPS - 1)
                max_prob = 0.0
                for i in range(N_STEPS):
                    start = i * step
                    win = iq_full[:, start:start + WIN_LEN]
                    power = win.pow(2).mean().clamp(min=1e-12).sqrt()
                    win_norm = (win / power).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, _ = model(win_norm, phys_tensor)
                        prob = torch.sigmoid(logits).item()
                    if prob > max_prob:
                        max_prob = prob

                label_bin = 0 if int(row['target']) == 4 else 1
                results.append({
                    'filename': row['filename'],
                    'label': label_bin,
                    'snr': row['snr'],
                    'target_multiclass': row['target'],
                    'prob_max': max_prob
                })
            except Exception:
                continue

        df_res = pd.DataFrame(results)
        df_res.to_csv(csv_path, index=False)

    y_true  = df_res['label']
    y_probs = df_res['prob_max']

    # --- UMBRAL ÓPTIMO (Max F1) ---
    precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * recall_curve * precision_curve / (recall_curve + precision_curve + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    opt_th = thresholds_curve[optimal_idx] if optimal_idx < len(thresholds_curve) else 0.5
    
    print(f"\nUmbral óptimo calculado (Max F1): {opt_th:.4f}")

    df_res['predicted'] = (df_res['prob_max'] >= opt_th).astype(int)
    df_res['correct'] = (df_res['label'] == df_res['predicted']).astype(int)
    y_pred  = df_res['predicted']

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} | Recall: {rec:.4f} | Precision: {pre:.4f} | F1: {f1:.4f}")

    title_suffix = f" (Umbral={opt_th:.2f})"

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", linewidths=0.4, linecolor="#1a1a2e",
                xticklabels=['Ruido', 'Dron'], yticklabels=['Ruido', 'Dron'], ax=ax)
    ax.set_title(f"Matriz de Confusión sobre Test Ciego{title_suffix}", fontweight="bold", pad=12)
    ax.set_ylabel("Clase Real")
    ax.set_xlabel("Clase Predicha")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 2. Curva Precision-Recall
    ap     = average_precision_score(y_true, y_probs)
    auc_pr = auc(recall_curve, precision_curve)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall_curve, precision_curve, linewidth=2.5, color="#4ECDC4",
            label=f"DualStream-SlidingWindow (AP={ap:.4f} | AUC-PR={auc_pr:.4f})")
    ax.axhline(baseline, color="#888", linestyle="--", linewidth=1.5,
               label=f"Línea Base Aleatoria ({baseline:.2f})")
    ax.scatter([recall_curve[optimal_idx]], [precision_curve[optimal_idx]], color="#FF6B6B", s=120, zorder=5,
               label=f"Umbral Óptimo = {opt_th:.2f}")
    ax.set_xlabel("Exhaustividad (Recall)")
    ax.set_ylabel("Precisión (Precision)")
    ax.set_title(f"Curva Precision-Recall{title_suffix}", fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pr_curve.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 3. Recall vs SNR por emisor
    drones = df_res[df_res['label'] == 1].copy()
    snrs   = sorted(drones['snr'].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    for target in sorted(drones['target_multiclass'].unique()):
        sub = drones[drones['target_multiclass'] == target]
        rec_snr = [sub[sub['snr'] == s]['correct'].mean() for s in snrs]
        ax.plot(snrs, rec_snr, marker="o", markersize=5, linewidth=2,
                color=TARGET_COLORS_FIXED.get(target), label=TARGET_NAMES_FIXED.get(target))
    global_recall = [drones[drones['snr'] == s]['correct'].mean() for s in snrs]
    ax.plot(snrs, global_recall, marker="D", markersize=6, linewidth=2.5,
            color="#1A1A1A", linestyle="--", label="Media Global (Drones)", zorder=5)
    ax.axvspan(-20, -7, alpha=0.07, color="red",    label="Grupo C")
    ax.axvspan(-6,   9, alpha=0.05, color="yellow", label="Grupo B")
    ax.axvspan(10,  30, alpha=0.07, color="green",  label="Grupo A")
    ax.set_xlabel("Relación Señal a Ruido (SNR) [dB]")
    ax.set_ylabel("Tasa de Detección (Recall)")
    ax.set_title(f"Tasa de Detección por Emisor RF en función de la SNR{title_suffix}", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(snrs) - 1, max(snrs) + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "recall_snr_lines.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 4. Accuracy / Recall / Especificidad vs SNR
    common_snrs = sorted(df_res['snr'].unique())
    acc_global = [df_res[df_res['snr'] == s]['correct'].mean() for s in common_snrs]
    acc_drones = [df_res[(df_res['snr'] == s) & (df_res['label'] == 1)]['correct'].mean() for s in common_snrs]
    acc_noise  = [df_res[(df_res['snr'] == s) & (df_res['label'] == 0)]['correct'].mean() for s in common_snrs]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(common_snrs, acc_global, marker="D", markersize=6, linewidth=2.5, color="#1A1A1A", label="Exactitud Global")
    ax.plot(common_snrs, acc_drones, marker="o", markersize=5, linewidth=2,   color="#0077B6", label="Sensibilidad (Drones)")
    ax.plot(common_snrs, acc_noise,  marker="s", markersize=5, linewidth=2,   color="#D62828", linestyle="--", label="Especificidad (Ruido)")
    ax.set_xlabel("Relación Señal a Ruido (SNR) [dB]")
    ax.set_ylabel("Tasa de Acierto")
    ax.set_title(f"Exactitud del Clasificador por Nivel de SNR{title_suffix}", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(common_snrs) - 1, max(common_snrs) + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "accuracy_snr_bars.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 5. Heatmap Clase × SNR
    hm_data = df_res.pivot_table(index='target_multiclass', columns='snr', values='correct', aggfunc='mean')
    hm_data.index = [TARGET_NAMES_FIXED.get(i) for i in hm_data.index]
    fig, ax = plt.subplots(figsize=(max(14, len(hm_data.columns) * 0.55), max(6, len(hm_data) * 0.9)))
    sns.heatmap(hm_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e",
                cbar_kws={'label': 'Tasa de Acierto (Recall)'}, ax=ax)
    ax.set_title(f"Mapa de Calor: Tasa de Acierto por Clase y SNR{title_suffix}", fontweight="bold")
    ax.set_xlabel("Relación Señal a Ruido (SNR) [dB]")
    ax.set_ylabel("Emisor RF")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "heatmap_snr_class.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # Resumen en texto
    report = (
        f"=================================================\n"
        f"INFORME GOLDEN SET — DualStream-SlidingWindow\n"
        f"=================================================\n"
        f"Umbral Óptimo   : {opt_th:.4f}\n"
        f"Accuracy Global : {acc:.4f}\n"
        f"Recall (Drones) : {rec:.4f}\n"
        f"Precision       : {pre:.4f}\n"
        f"F1-Score        : {f1:.4f}\n"
        f"Average Prec    : {ap:.4f}\n"
        f"AUC-PR          : {auc_pr:.4f}\n"
        f"=================================================\n"
    )
    with open(os.path.join(OUT_DIR, "summary_golden.txt"), 'w') as fh:
        fh.write(report)
    print(report)
    print(f"Figuras generadas en: {OUT_DIR}")


if __name__ == "__main__":
    main()
