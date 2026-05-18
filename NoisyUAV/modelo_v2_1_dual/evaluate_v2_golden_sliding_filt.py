import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, average_precision_score
from pathlib import Path

# Rutas y Config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

from model import DualStreamCVCNN
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

# Configuración
OUT_DIR = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual"
DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CSV_GOLDEN = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual\dataset_v2_1_clean_pointers.csv"
CKPT_PATH = os.path.join(OUT_DIR, "checkpoints", "best_model.pth")
RESULTS_DIR = os.path.join(OUT_DIR, "figuras_golden_filtrado_umbral75")

FS = 14e6
WIN_LEN = 131072
N_STEPS = 16
THRESHOLD_V2 = 0.75
MIN_CONSECUTIVE = 2

def calculate_phys_features(iq_full):
    try:
        _, _, H_smooth, _, nf_v, _, _, bursts = detectar_bursts(
            iq_full, fs=FS, nperseg=2048, z_thresh=2.5
        )
        global_nf = float(np.median(nf_v))
        global_H  = float(np.mean(H_smooth))
        z_peak    = max([abs(b['z_peak']) for b in bursts]) if bursts else 0.0
        return [global_nf, global_H, min(z_peak, 30.0)]
    except: return [0.0, 0.0, 0.0]

def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar Modelo
    model = DualStreamCVCNN().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Cargar Golden Set REAL y SAGRADO (3.744 ficheros)
    CSV_GOLDEN_REAL = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
    df_unique = pd.read_csv(CSV_GOLDEN_REAL)
    print(f"Evaluando Lógica V2 sobre el GOLDEN SET REAL ({len(df_unique)} ficheros)...")

    csv_path = os.path.join(RESULTS_DIR, "golden_results_v2_filt.csv")
    results = []

    with torch.no_grad():
        for _, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="Inferencia V2"):
            fpath = os.path.join(DATA_DIR, row['filename'])
            if not os.path.exists(fpath): continue

            d = torch.load(fpath, map_location='cpu', weights_only=False)
            iq_full = d['x_iq'].float()
            
            # 1. Física Global
            phys_vals = calculate_phys_features(iq_full)
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
            
            label_bin = 0 if int(row['target']) == 4 else 1
            results.append({
                'filename': row['filename'],
                'target': row['target'],
                'snr': row['snr'],
                'label_bin': label_bin,
                'pred_bin': confirmed,
                'max_prob': max(probs),
                'window_probs': str(probs)
            })

    df_res = pd.DataFrame(results)
    df_res.to_csv(csv_path, index=False)

    # 4. Cálculo de Métricas y Gráficas (Mismo estilo que V1 para comparar)
    print("\n--- MÉTRICAS LÓGICA V2 (FILTRADA) ---")
    recall = recall_score(df_res['label_bin'], df_res['pred_bin'])
    precision = precision_score(df_res['label_bin'], df_res['pred_bin'])
    acc = accuracy_score(df_res['label_bin'], df_res['pred_bin'])
    f1 = f1_score(df_res['label_bin'], df_res['pred_bin'])
    ap = average_precision_score(df_res['label_bin'], df_res['max_prob'])
    
    print(f"Recall: {recall:.4f} | Precision: {precision:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AP: {ap:.4f}")

    # --- FIGURAS ---
    # 1. Heatmap Recall por SNR y Clase
    df_drones = df_res[df_res['label_bin'] == 1].copy()
    pivot_recall = df_drones.groupby(['target', 'snr'])['pred_bin'].mean().unstack()
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_recall, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Recall'})
    plt.title(f"Recall V2 (Filtrado) por SNR y Target\nThreshold={THRESHOLD_V2}, Cons={MIN_CONSECUTIVE}")
    plt.savefig(os.path.join(RESULTS_DIR, "heatmap_recall_v2_filt.png"), dpi=150)

    # 2. Accuracy en Ruido (FPR invertido)
    df_noise = df_res[df_res['label_bin'] == 0].copy()
    noise_acc = df_noise.groupby('snr')['pred_bin'].apply(lambda x: 1 - x.mean())
    plt.figure(figsize=(10, 5))
    noise_acc.plot(kind='bar', color='salmon')
    plt.title("Especificidad en Ruido (1 - FPR) - Lógica V2 Filtrada")
    plt.ylabel("Prob. de clasificar correctamente como Ruido")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "noise_accuracy_v2_filt.png"), dpi=150)

    print(f"Evaluación completada. Resultados en {RESULTS_DIR}")

if __name__ == "__main__":
    main()
