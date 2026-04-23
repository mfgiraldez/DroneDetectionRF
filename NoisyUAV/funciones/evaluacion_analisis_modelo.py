"""
evaluacion_analisis_modelo.py

Módulo unificado para extraer las métricas de clasificación de la CV-CNN
(Confusión, ROC, PR) y el análisis científico de degradación al Ruido (SNR).
Equipado con un conmutador paramétrico para alternar de manera fluida entre 
el test de la red Base y el test Normalizado (Min-Max en tiempo real).
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split

# Asegurar importe de los módulos del proyecto raíz
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from funciones.dataset_stage2 import get_dataloaders
from modelos.cvcnn import ComplexConv1DNet

# ==========================================
# RUTINAS MATEMÁTICAS INFERENCIA
# ==========================================
def apply_agc_batch(inputs: torch.Tensor) -> torch.Tensor:
    """Aplica Control Automático de Ganancia (Min-Max) a todo el Batch de forma independiente."""
    for b in range(inputs.size(0)):
        max_val = torch.max(torch.abs(inputs[b]))
        if max_val > 0:
            inputs[b] = inputs[b] / max_val
    return inputs

# ==========================================
# MOTOR GRÁFICO 
# ==========================================
def plot_confusion_matrix(y_true, y_pred, save_path, normalized_flag):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Noise (0)", "Drone (1)"], 
                yticklabels=["Noise (0)", "Drone (1)"])
    tag = "(Normalized)" if normalized_flag else "(Raw Scale)"
    plt.title(f"Test Set Confusion Matrix {tag}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falsos Positivos (FPR)')
    plt.ylabel('Verdaderos Positivos (TPR) - Recall')
    plt.title('Curva ROC de Clasificación')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pr_curve(y_true, y_probs, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AUC-PR = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_snr_line(drones_df, save_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=drones_df, x="snr", y="correct", hue="Modelo de Dron", marker="o", lw=2)
    plt.title("Degradación de Recall por SNR", fontsize=14)
    plt.xlabel("Relación Señal/Ruido (SNR) [dB]", fontsize=12)
    plt.ylabel("Tasa de Acierto (Recall)", fontsize=12)
    plt.ylim([-0.05, 1.05])
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_snr_heatmap(drones_df, save_path):
    heatmap_data = drones_df.pivot_table(
        index='Modelo de Dron', 
        columns='snr', 
        values='correct', 
        aggfunc='mean'
    )
    plt.figure(figsize=(12, 4))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    plt.title("CV-CNN por Modelo de Dron y SNR", fontsize=14)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Emisor RF")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# ORQUESTADOR PRINCIPAL JUPYTER/CLI
# ==========================================
def evaluar_y_analizar(normalized=False):
    """
    Función principal llamada desde Python plano o Jupyter Notebooks.
    
    Args:
        normalized (bool): Si es True usa stage2_norm y aplica el filtro x/max(abs(x)).
    """
    if normalized:
        OUT_DIR = r"C:\TFM_data\NoisyUAV\stage2_norm"
        print(">> MODO NORMALIZADO ACTIVADO <<")
    else:
        OUT_DIR = r"C:\TFM_data\NoisyUAV\stage2"
        print(">> MODO RAW (BASE) ACTIVADO <<")
        
    CSV_METADATA = os.path.join(OUT_DIR, "metadata_stage2.csv") # El stage base se usó de molde, pero podemos referenciar stage2 local siempre
    if not os.path.exists(CSV_METADATA):
        # Fallback al CSV original por si la normalizada no copió el CSV físico
        CSV_METADATA = r"C:\TFM_data\NoisyUAV\stage2\metadata_stage2.csv"
        
    CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")
    PLOTS_DIR = os.path.join(OUT_DIR, "plots")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "cvcnn_best_model.pth")
    
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"❌ ERROR: Faltan pesos en {BEST_MODEL_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Emparejamiento de dataframe de fondo para Analisis del SNR
    df = pd.read_csv(CSV_METADATA)
    df_train, df_temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp["label"], random_state=42)
    df_test = df_test.reset_index(drop=True)

    # 2. Cargar Dataloader
    _, _, test_loader = get_dataloaders(CSV_METADATA, batch_size=64, num_workers=4)
    model = ComplexConv1DNet(num_classes=2, pool_output_size=64, dropout=0.5)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    print("\nCalculando matriz continua (Inferencia Test Ciego)...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing Model"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 💉 INYECTOR DE NORMALIZACIÓN CONDICIONAL
            if normalized:
                inputs = apply_agc_batch(inputs)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # 3. Vincular vector a Df Analitico
    df_test["predicted"] = all_preds
    df_test["correct"] = (df_test["label"] == df_test["predicted"]).astype(int)
    
    drones_df = df_test[df_test["label"] == 1].copy()
    drones_df["target"] = drones_df["target"].astype(str)
    target_names = {"1": "Target 1", "2": "Target 2", "3": "Target 3", "5": "Target 5", "6": "Target 6"}
    drones_df["Modelo de Dron"] = drones_df["target"].map(lambda x: target_names.get(x, f"Target {x}"))
    
    print("\nDibujando Figuras Matemáticas...")
    
    # 4. Renderización Física
    plot_path = lambda filename: os.path.join(PLOTS_DIR, filename)
    
    plot_confusion_matrix(all_targets, all_preds, plot_path("test_confusion_matrix_final.png"), normalized)
    plot_roc_curve(all_targets, all_probs, plot_path("test_roc_curve.png"))
    plot_pr_curve(all_targets, all_probs, plot_path("test_precision_recall_curve.png"))
    
    plot_snr_line(drones_df, plot_path("test_snr_sensibilidad_lineas.png"))
    plot_snr_heatmap(drones_df, plot_path("test_snr_sensibilidad_heatmap.png"))

    print(f"✅ ¡Operación exitosa! Todos los gráficos de evaluación disponibles en:\n-> {PLOTS_DIR}")

# ==========================================
# PUNTO DE ENTRADA CLI
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluación integral de la arquitectura Híbrida RF.')
    parser.add_argument('--norm', action='store_true', help='Usar modelo Normalizado (scale invariance)')
    args = parser.parse_args()
    
    torch.multiprocessing.freeze_support()
    evaluar_y_analizar(normalized=args.norm)
