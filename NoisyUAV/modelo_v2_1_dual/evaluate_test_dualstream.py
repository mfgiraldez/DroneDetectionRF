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
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, auc

# Configure matplotlib for academic style
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

# Asegurar importaciones
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

from NoisyUAV.modelo_v2_1_dual.model import DualStreamCVCNN
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

# Configuración de rutas
TEST_CSV   = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v1\alumn_dataset_pseudo_v3.csv"
DATA_DIR   = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CKPT_PATH  = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual\checkpoints\best_model.pth"
OUT_DIR    = r"C:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\DualStream-Attention"

FS = 14e6
WIN_LEN = 131072
N_STEPS = 16
THRESHOLD_V2 = 0.75
MIN_CONSECUTIVE = 2

TARGET_NAMES_FIXED = {
    0: "DJI (T0)", 1: "FutabaT14 (T1)", 2: "FutabaT7 (T2)",
    3: "Graupner (T3)", 5: "Taranis (T5)", 6: "Turnigy (T6)",
    4: "Ruido (T4)",
}
TARGET_COLORS_FIXED = {
    0: "#0077B6", 1: "#D62828", 2: "#118AB2",
    3: "#2D6A4F", 5: "#E07C00", 6: "#7B2D8B",
}
FIGSAVE_FIXED = dict(dpi=300, bbox_inches="tight", format="pdf")


def calculate_phys_features(iq_full, adaptive_window_ms):
    try:
        _, _, H_smooth, _, nf_v, ns, _, bursts = detectar_bursts(
            iq_full, fs=FS, nperseg=2048, adaptive_window_ms=adaptive_window_ms, min_burst_ms=0.3, z_thresh=2.5
        )
        global_nf = float(np.median(nf_v))
        global_H  = float(np.mean(H_smooth))
        if bursts:
            z_peak = max([abs(b['z_peak']) for b in bursts])
        else:
            z_peak = (np.min(H_smooth) - global_nf) / (ns + 1e-10)
        return [global_nf, global_H, float(np.clip(abs(z_peak), 0, 30))]
    except: return [0.0, 0.0, 0.0]

@torch.no_grad()
def evaluate_test_set():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Loading DualStream model from {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    
    model = DualStreamCVCNN().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Cargar CSV original y filtrar por test
    df_all = pd.read_csv(TEST_CSV)
    df_test = df_all[df_all['split'] == 'test'].copy()
    
    # Quedarnos con una sola fila por fichero para la evaluación completa (production-like)
    df_unique = df_test.drop_duplicates(subset=['file_path']).copy()
    print(f"Original Test Set: {len(df_unique)} unique files")

    csv_path = os.path.join(OUT_DIR, "test_results_dualstream.csv")
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}...")
        df_res = pd.read_csv(csv_path)
    else:
        results = []
        for idx, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="Evaluating DualStream Test Set"):
            fpath = os.path.join(DATA_DIR, row['file_path'])
            if not os.path.exists(fpath): continue

            # Cargar IQ completo (75ms)
            d = torch.load(fpath, map_location='cpu', weights_only=False)
            iq_full = d['x_iq'].float()
            
            # 1. Física Global
            phys_vals = calculate_phys_features(iq_full, adaptive_window_ms=15.0)
            phys_tensor = torch.tensor([phys_vals], dtype=torch.float32).to(device)

            # 2. Sliding Window
            max_idx = iq_full.shape[1]
            step = (max_idx - WIN_LEN) // (N_STEPS - 1)
            probs = []

            for i in range(N_STEPS):
                start = i * step
                win = iq_full[:, start:start+WIN_LEN]
                
                power = win.pow(2).mean().clamp(min=1e-12).sqrt()
                win_norm = (win / power).unsqueeze(0).to(device)
                logits, _ = model(win_norm, phys_tensor)
                probs.append(torch.sigmoid(logits).item())

            # 3. Lógica V2 (Filtrado Temporal)
            consecutive_count = 0
            confirmed = 0
            for p in probs:
                if p >= THRESHOLD_V2:
                    consecutive_count += 1
                    if consecutive_count >= MIN_CONSECUTIVE:
                        confirmed = 1
                        break
                else:
                    consecutive_count = 0

            label_bin = 0 if int(row['target_multiclass']) == 4 else 1
            max_prob = max(probs)
            
            results.append({
                'filename': row['file_path'],
                'label': label_bin,
                'snr': row['snr'],
                'target_multiclass': row['target_multiclass'],
                'prob_max': max_prob,
                'predicted': confirmed,
                'window_probs': str(probs)
            })
        df_res = pd.DataFrame(results)
        df_res.to_csv(csv_path, index=False)

    df_res['correct'] = (df_res['label'] == df_res['predicted']).astype(int)
    
    y_true = df_res['label']
    y_pred = df_res['predicted']
    y_probs = df_res['prob_max']
    
    # Metricas
    print("\n" + "="*30)
    print(f"DUALSTREAM-ATTENTION ORIGINAL TEST SET")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print("="*30)

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", linewidths=0.4, linecolor="#1a1a2e",
                xticklabels=['Ruido', 'Dron'], yticklabels=['Ruido', 'Dron'], ax=ax)
    ax.set_title("Matriz de Confusión sobre el Conjunto de Test Original", fontweight="bold", pad=12)
    ax.set_ylabel("Clase Real")
    ax.set_xlabel("Clase Predicha")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 2. Curva Precision-Recall
    precision, recall_pr, thresholds = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    auc_pr = auc(recall_pr, precision)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall_pr, precision, linewidth=2.5, color="#4ECDC4", 
            label=f"DualStream-Attention (AP={ap:.4f} | AUC-PR={auc_pr:.4f})")
    ax.axhline(baseline, color="#888", linestyle="--", linewidth=1.5, label=f"Línea Base Aleatoria ({baseline:.2f})")
    
    ax.set_xlabel("Exhaustividad (Recall)")
    ax.set_ylabel("Precisión (Precision)")
    ax.set_title("Curva Precision-Recall sobre el Conjunto de Test Original", fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pr_curve.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 3. Recall vs SNR
    drones = df_res[df_res['label'] == 1].copy()
    snrs = sorted(drones['snr'].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    for target in sorted(drones['target_multiclass'].unique()):
        sub = drones[drones['target_multiclass'] == target]
        rec = [sub[sub['snr'] == s]['correct'].mean() for s in snrs]
        ax.plot(snrs, rec, marker="o", markersize=5, linewidth=2,
                color=TARGET_COLORS_FIXED.get(target), label=TARGET_NAMES_FIXED.get(target))
    
    global_recall = [drones[drones['snr'] == s]['correct'].mean() for s in snrs]
    ax.plot(snrs, global_recall, marker="D", markersize=6, linewidth=2.5,
            color="#1A1A1A", linestyle="--", label="Media Global (Drones)", zorder=5)
    
    ax.axvspan(-20, -7,  alpha=0.07, color="red",    label="Grupo C")
    ax.axvspan(-6,   9,  alpha=0.05, color="yellow", label="Grupo B")
    ax.axvspan(10,  30,  alpha=0.07, color="green",  label="Grupo A")
    ax.set_xlabel("Relación Señal a Ruido (SNR) [dB]")
    ax.set_ylabel("Tasa de Detección (Recall)")
    ax.set_title("Tasa de Detección por Emisor RF en función de la SNR", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(snrs)-1, max(snrs)+1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "recall_snr_lines.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 4. Accuracy per SNR
    fig, ax = plt.subplots(figsize=(11, 6))
    acc_global = [df_res[df_res['snr'] == s]['correct'].mean() for s in snrs]
    acc_drones = [df_res[(df_res['snr'] == s) & (df_res['label'] == 1)]['correct'].mean() for s in snrs]
    acc_noise  = [df_res[(df_res['snr'] == s) & (df_res['label'] == 0)]['correct'].mean() for s in snrs]
    
    ax.plot(snrs, acc_global, marker="D", markersize=6, linewidth=2.5, color="#1A1A1A", label="Exactitud Global")
    ax.plot(snrs, acc_drones, marker="o", markersize=5, linewidth=2,   color="#0077B6", label="Sensibilidad (Drones)")
    ax.plot(snrs, acc_noise,  marker="s", markersize=5, linewidth=2,   color="#D62828", linestyle="--", label="Especificidad (Ruido)")
    ax.set_xlabel("Relación Señal a Ruido (SNR) [dB]")
    ax.set_ylabel("Tasa de Acierto")
    ax.set_title("Exactitud del Clasificador por Nivel de SNR", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(snrs)-1, max(snrs)+1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "accuracy_snr_bars.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 5. Heatmap
    hm_data = df_res.pivot_table(index='target_multiclass', columns='snr', values='correct', aggfunc='mean')
    hm_data.index = [TARGET_NAMES_FIXED.get(i) for i in hm_data.index]
    fig, ax = plt.subplots(figsize=(max(14, len(hm_data.columns)*0.55), max(6, len(hm_data)*0.9)))
    sns.heatmap(hm_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e", cbar_kws={'label': 'Tasa de Acierto (Recall)'})
    ax.set_title("Mapa de Calor: Tasa de Acierto por Clase y SNR", fontweight="bold")
    ax.set_xlabel("Relación Señal a Ruido (SNR) [dB]")
    ax.set_ylabel("Emisor RF")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "heatmap_snr_class.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    print(f"Figuras generadas exitosamente en formato PDF en: {OUT_DIR}")

if __name__ == "__main__":
    evaluate_test_set()
