import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc

# Asegurar importe de los módulos del proyecto raíz
# Estamos en NoisyUAV/curriculum_alumn_v1, el root es .. o ../.. dependiendo de la estructura
# Dado que el repo es c:\repos\DroneDetectionRF, y estamos en c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from NoisyUAV.modelos.burst_cvcnn import BurstCVCNN as AlumnCVCNN, BurstCVCNNDataset as AlumnDataset, pad_seq_collate
from NoisyUAV.funciones.plot_styles import apply_ieee_style, get_color_palette, clean_spines

# --- CONFIGURACIÓN ---
DATA_DIR   = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CSV_PATH   = r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1\alumn_dataset_pseudo.csv"
MODEL_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1\checkpoints\alumn_model_best.pt"
OUTPUT_DIR = r"c:\repos\DroneDetectionRF\NoisyUAV\curriculum_alumn_v1\evaluation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    apply_ieee_style()
    colors = get_color_palette()
    
    print(f"--- Cargando Dataset y Modelo ---")
    if not os.path.exists(CSV_PATH):
        print(f"Error: No se encuentra {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    # Filtrar solo el conjunto de test
    df_test = df[df['split'] == 'test'].copy()
    
    # Cargar checkpoint y pesos
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encuentra {MODEL_PATH}")
        return
        
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    phys_mean = torch.from_numpy(ckpt['phys_mean'])
    phys_std  = torch.from_numpy(ckpt['phys_std'])
    
    model = AlumnCVCNN(dropout_cnn=0.3, dropout_fuse=0.4).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    # Dataset & Loader
    ds_test = AlumnDataset(df_test, DATA_DIR, augment=False, phys_mean=phys_mean, phys_std=phys_std)
    ld_test = torch.utils.data.DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=0, collate_fn=pad_seq_collate)
    
    print(f"--- Ejecutando Inferencia en {len(ds_test)} ráfagas de test ---")
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(ld_test):
            iq = batch['iq'].to(DEVICE)
            f  = batch['feats'].to(DEVICE)
            logits = model(iq, f)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
    
    df_test['prob'] = all_probs
    df_test['pred'] = (df_test['prob'] > 0.5).astype(int)
    
    # --- 1. Heatmap: Recall por Modelo de Dron y SNR ---
    # Filtrar solo drones (label=1) para calcular Recall
    df_drones = df_test[df_test['label'] == 1].copy()
    
    # Mapear nombres de target para que queden bonitos
    target_names = {
        0: "Target 0", 1: "Target 1", 2: "Target 2", 
        3: "Target 3", 5: "Target 5", 6: "Target 6"
    }
    df_drones['target_name'] = df_drones['target_multiclass'].map(target_names)
    
    # Crear pivote para el heatmap (asegurando floats en SNR para estética)
    df_drones['snr'] = df_drones['snr'].astype(float)
    heatmap_data = df_drones.pivot_table(index='target_name', columns='snr', values='pred', aggfunc='mean')
    
    plt.figure(figsize=(12, 4))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("Alumno V1 por Modelo de Dron y SNR", fontsize=14)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Emisor RF")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_01_heatmap_recall.png"), dpi=300)
    print(f"  -> Guardado: fig_01_heatmap_recall.png")
    
    # --- 2. Line Plot: Degradación de Recall por SNR ---
    plt.figure(figsize=(14, 8))
    # Usar una paleta de colores distinta para cada target
    target_colors = sns.color_palette("bright", n_colors=len(df_drones['target_multiclass'].unique()))
    
    for i, target in enumerate(sorted(df_drones['target_multiclass'].unique())):
        subset = df_drones[df_drones['target_multiclass'] == target]
        # Calcular media y error estándar (para el sombreado)
        grouped = subset.groupby('snr')['pred'].agg(['mean', 'std', 'count']).reset_index()
        # Evitar división por cero en std/sqrt(n)
        se = grouped['std'] / np.sqrt(grouped['count'].replace(0, 1))
        
        plt.plot(grouped['snr'], grouped['mean'], marker='o', markersize=6, linewidth=2.5, 
                 label=f"Target {target}", color=target_colors[i])
        plt.fill_between(grouped['snr'], 
                         np.clip(grouped['mean'] - se, 0, 1), 
                         np.clip(grouped['mean'] + se, 0, 1), 
                         alpha=0.15, color=target_colors[i])
        
    plt.title("Degradación de Recall por SNR (Alumno V1)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Relación Señal/Ruido (SNR) [dB]", fontsize=13, fontweight='bold')
    plt.ylabel("Tasa de Acierto (Recall)", fontsize=13, fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Modelo de Dron", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_02_recall_snr_lines.png"), dpi=300)
    print(f"  -> Guardado: fig_02_recall_snr_lines.png")
    
    # --- 3. Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(df_test['label'], df_test['prob'])
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(9, 8))
    plt.plot(recall, precision, color='purple', lw=3, label=f'PR curve (AUC-PR = {pr_auc:.4f})')
    plt.fill_between(recall, precision, alpha=0.1, color='purple')
    
    plt.title("Curva Precision-Recall (Alumno V1 Test Set)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Recall", fontsize=13, fontweight='bold')
    plt.ylabel("Precision", fontsize=13, fontweight='bold')
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_03_pr_curve.png"), dpi=300)
    print(f"  -> Guardado: fig_03_pr_curve.png")
    
    # --- 4. Resumen por SNR (Opcional, muy útil) ---
    plt.figure(figsize=(10, 6))
    snr_acc = df_test.groupby('snr')['pred'].apply(lambda x: (x == df_test.loc[x.index, 'label']).mean())
    plt.bar(snr_acc.index, snr_acc.values, color=colors['blue'], alpha=0.7, width=1.5)
    plt.title("Accuracy Global por SNR (Alumno V1)", fontsize=14, fontweight='bold')
    plt.xlabel("SNR [dB]")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_04_accuracy_per_snr.png"), dpi=300)
    print(f"  -> Guardado: fig_04_accuracy_per_snr.png")
    
    print(f"\n✅ Proceso completado con éxito. Figuras listas en:\n   {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
