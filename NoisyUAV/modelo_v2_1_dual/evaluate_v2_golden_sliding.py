import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, roc_auc_score

# Asegurar que podemos importar NoisyUAV y los módulos locales
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

from model import DualStreamCVCNN
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

# Configuración
GOLDEN_CSV = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
DATA_DIR   = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CKPT_PATH  = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual\checkpoints\best_model.pth"
OUT_DIR    = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual\figures_golden"
FS = 14e6
WIN_LEN = 131072
N_STEPS = 16  # Número de ventanas deslizantes

def calculate_phys_features(iq_tensor):
    """ Calcula las 3 features que espera el modelo V2.1 """
    try:
        _, _, H_smooth, _, nf_v, _, _, bursts = detectar_bursts(
            iq_tensor, fs=FS, nperseg=2048,
            adaptive_window_ms=15.0, min_burst_ms=0.3, z_thresh=2.5
        )
        global_nf = float(np.median(nf_v))
        global_H  = float(np.mean(H_smooth))
        z_peak    = 0.0
        if bursts:
            b0 = max(bursts, key=lambda b: abs(b['z_peak']))
            z_peak = float(np.clip(abs(b0['z_peak']), 0, 30))
        return [global_nf, global_H, z_peak]
    except Exception:
        return [0.0, 0.0, 0.0]

@torch.no_grad()
def evaluate_golden():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {CKPT_PATH}...")
    model = DualStreamCVCNN().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    df_golden = pd.read_csv(GOLDEN_CSV)
    print(f"Golden Set: {len(df_golden)} files")

    csv_path = os.path.join(OUT_DIR, "golden_results_v2_1.csv")
    if os.path.exists(csv_path):
        print(f"Loading existing results from {csv_path}...")
        df_res = pd.read_csv(csv_path)
    else:
        results = []
        for idx, row in tqdm(df_golden.iterrows(), total=len(df_golden), desc="Evaluating Golden Set"):
            fpath = os.path.join(DATA_DIR, row['filename'])
            if not os.path.exists(fpath): continue

            # Cargar IQ completo (75ms)
            d = torch.load(fpath, map_location='cpu', weights_only=False)
            iq_full = d['x_iq'].float()
            max_idx = iq_full.shape[1]

            # Sliding Window
            step = (max_idx - WIN_LEN) // (N_STEPS - 1)
            max_prob = 0.0

            for i in range(N_STEPS):
                start = i * step
                end = start + WIN_LEN
                win = iq_full[:, start:end]

                # Features Físicas
                phys_vals = calculate_phys_features(win)
                phys_tensor = torch.tensor([phys_vals], dtype=torch.float32).to(device)

                # Normalización RMS
                power = win.pow(2).mean().clamp(min=1e-12).sqrt()
                win_norm = (win / power).unsqueeze(0).to(device)

                # Inferencia
                logits, _ = model(win_norm, phys_tensor)
                prob = torch.sigmoid(logits).item()
                if prob > max_prob:
                    max_prob = prob
            
            # Obtener etiqueta binaria: Target 4 es Ruido (0), el resto son Drones (1)
            label_bin = 0 if int(row['target']) == 4 else 1
            
            results.append({
                'filename': row['filename'],
                'label': label_bin,
                'snr': row['snr'],
                'target_multiclass': row['target'],
                'prob_max': max_prob,
                'predicted': 1 if max_prob > 0.5 else 0
            })
        df_res = pd.DataFrame(results)
    df_res['correct'] = (df_res['label'] == df_res['predicted']).astype(int)
    
    # Métricas Globales
    y_true = df_res['label']
    y_pred = df_res['predicted']
    
    print("\n" + "="*30)
    print(f"GOLDEN SET RESULTS (V2.1)")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print("="*30)

    # ------------------------------------------------------------------------
    # ESTILOS Y COLORES (COPIA LITERAL DE alumn_v2_dual_eval.py)
    # ------------------------------------------------------------------------
    import matplotlib.ticker as mticker
    TARGET_NAMES_FIXED = {
        0: "DJI (T0)", 1: "FutabaT14 (T1)", 2: "FutabaT7 (T2)",
        3: "Graupner (T3)", 5: "Taranis (T5)", 6: "Turnigy (T6)",
        4: "Ruido (T4)",
    }
    TARGET_COLORS_FIXED = {
        0: "#0077B6", 1: "#D62828", 2: "#118AB2",
        3: "#2D6A4F", 5: "#E07C00", 6: "#7B2D8B",
    }
    FIGSAVE_FIXED = dict(dpi=150, bbox_inches="tight")

    # 1. Matriz de Confusión (Estilo Original)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", linewidths=0.4, linecolor="#1a1a2e",
                xticklabels=['Ruido', 'Dron'], yticklabels=['Ruido', 'Dron'], ax=ax)
    ax.set_title("Dual-Stream V2 -- Matriz de Confusión (Golden Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Real", fontsize=11); ax.set_xlabel("Predicho", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix_golden.png"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 2. Curva Precision-Recall (Estilo Original)
    from sklearn.metrics import precision_recall_curve, average_precision_score, auc
    y_probs = df_res['prob_max']
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    auc_pr = auc(recall, precision)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, linewidth=2.5, color="#4ECDC4", 
            label=f"Dual-Stream V2 (AP={ap:.4f} | AUC-PR={auc_pr:.4f})")
    ax.axhline(baseline, color="#888", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline:.2f})")
    # Punto de operación 0.5
    th_idx = min(np.searchsorted(thresholds, 0.5), len(precision) - 2)
    ax.scatter([recall[th_idx]], [precision[th_idx]], color="#FF6B6B", s=120, zorder=5,
               label=f"Umbral=0.5 (P={precision[th_idx]:.3f}, R={recall[th_idx]:.3f})")
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Dual-Stream V2 -- Curva Precision-Recall (Golden Set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pr_curve_golden.png"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 3. Recall vs SNR Lines (COPIA LITERAL LOGICA)
    drones = df_res[df_res['label'] == 1].copy()
    snrs = sorted(drones['snr'].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    for target in sorted(drones['target_multiclass'].unique()):
        sub = drones[drones['target_multiclass'] == target]
        rec = [sub[sub['snr'] == s]['correct'].mean() for s in snrs]
        ax.plot(snrs, rec, marker="o", markersize=5, linewidth=2,
                color=TARGET_COLORS_FIXED.get(target), label=TARGET_NAMES_FIXED.get(target))
    # Linea Global Negra con Diamantes
    global_recall = [drones[drones['snr'] == s]['correct'].mean() for s in snrs]
    ax.plot(snrs, global_recall, marker="D", markersize=6, linewidth=2.5,
            color="#1A1A1A", linestyle="--", label="Media Global (Drones)", zorder=5)
    
    # Sombreado Grupos A, B, C
    ax.axvspan(-20, -7,  alpha=0.07, color="red",    label="Grupo C")
    ax.axvspan(-6,   9,  alpha=0.05, color="yellow", label="Grupo B")
    ax.axvspan(10,  30,  alpha=0.07, color="green",  label="Grupo A")
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Dual-Stream V2 -- Recall vs SNR por Emisor RF (Golden Set)", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(min(snrs)-1, max(snrs)+1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "recall_snr_lines_golden.png"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 4. Accuracy per SNR (3 LINEAS: Global, Drones, Ruido)
    fig, ax = plt.subplots(figsize=(11, 6))
    acc_global = [df_res[df_res['snr'] == s]['correct'].mean() for s in sorted(df_res['snr'].unique())]
    acc_drones = [df_res[(df_res['snr'] == s) & (df_res['label'] == 1)]['correct'].mean() for s in sorted(df_res['snr'].unique())]
    acc_noise  = [df_res[(df_res['snr'] == s) & (df_res['label'] == 0)]['correct'].mean() for s in sorted(df_res['snr'].unique())]
    common_snrs = sorted(df_res['snr'].unique())
    
    ax.plot(common_snrs, acc_global, marker="D", markersize=6, linewidth=2.5, color="#1A1A1A", label="Accuracy Global")
    ax.plot(common_snrs, acc_drones, marker="o", markersize=5, linewidth=2,   color="#0077B6", label="Recall Drones")
    ax.plot(common_snrs, acc_noise,  marker="s", markersize=5, linewidth=2,   color="#D62828", linestyle="--", label="Especificidad Ruido")
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Tasa de Acierto", fontsize=12)
    ax.set_title("Dual-Stream V2 -- Accuracy por Nivel de SNR (Golden Set)", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05); ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "accuracy_snr_bars_golden.png"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 5. Heatmap (Ajuste Figsize y Estilo)
    hm_data = df_res.pivot_table(index='target_multiclass', columns='snr', values='correct', aggfunc='mean')
    hm_data.index = [TARGET_NAMES_FIXED.get(i) for i in hm_data.index]
    fig, ax = plt.subplots(figsize=(max(14, len(hm_data.columns)*0.55), max(6, len(hm_data)*0.9)))
    sns.heatmap(hm_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e", ax=ax)
    ax.set_title("Dual-Stream V2 -- Recall/TNR por Clase y SNR (Golden Set)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "heatmap_snr_class_golden.png"), **FIGSAVE_FIXED)
    plt.close(fig)

    # Guardar Reporte Final
    report = f"""
    =================================================
    INFORME FINAL GOLDEN SET - MODELO V2.1 (LEGACY STYLE)
    =================================================
    Accuracy Global: {accuracy_score(y_true, y_pred):.4f}
    Recall (Drones): {recall_score(y_true, y_pred):.4f}
    Precision:       {precision_score(y_true, y_pred):.4f}
    F1-Score:        {f1_score(y_true, y_pred):.4f}
    Average Prec:    {ap:.4f}
    =================================================
    """
    with open(os.path.join(OUT_DIR, "summary_golden.txt"), 'w') as f:
        f.write(report)
    print(report)
    df_res.to_csv(os.path.join(OUT_DIR, "golden_results_v2_1.csv"), index=False)
    print(f"Figuras LITERALMENTE corregidas en: {OUT_DIR}")

if __name__ == "__main__":
    evaluate_golden()
