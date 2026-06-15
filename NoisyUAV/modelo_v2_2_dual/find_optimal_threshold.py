import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.metrics import recall_score, precision_score, f1_score

# Configuración
RESULTS_DIR = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_2_dual\figuras_golden_filtrado_umbral75"
CSV_PATH = os.path.join(RESULTS_DIR, "golden_results_v2_filt.csv")
MIN_CONSECUTIVE = 2

def apply_temporal_filter(probs_list, threshold, min_consecutive):
    consecutive_count = 0
    for p in probs_list:
        if p >= threshold:
            consecutive_count += 1
            if consecutive_count >= min_consecutive:
                return 1
        else:
            consecutive_count = 0
    return 0

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Buscando {CSV_PATH} pero no existe aún. Espera a que termine la evaluación.")
        return
        
    print(f"Cargando {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Convertir la columna string a lista real de python
    df['window_probs_list'] = df['window_probs'].apply(ast.literal_eval)
    
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = []
    precisions = []
    recalls = []
    
    y_true = df['label_bin']
    
    print("Iniciando barrido paramétrico...")
    for th in thresholds:
        y_pred = df['window_probs_list'].apply(lambda x: apply_temporal_filter(x, th, MIN_CONSECUTIVE))
        
        f1_scores.append(f1_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred))
        
    # Encontrar el óptimo
    max_f1 = max(f1_scores)
    best_th = thresholds[f1_scores.index(max_f1)]
    best_prec = precisions[f1_scores.index(max_f1)]
    best_rec = recalls[f1_scores.index(max_f1)]
    
    print("\n" + "="*40)
    print(f"RESULTADOS DEL BARRIDO PARAMÉTRICO (N={MIN_CONSECUTIVE})")
    print(f"F1-Score Máximo: {max_f1:.4f}")
    print(f"Umbral Óptimo:   {best_th:.2f}")
    print(f"Precisión en óptimo: {best_prec:.4f}")
    print(f"Recall en óptimo:    {best_rec:.4f}")
    print("="*40)
    
    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label="F1-Score", linewidth=3, color="#2D6A4F")
    plt.plot(thresholds, precisions, label="Precisión", linestyle="--", color="#118AB2")
    plt.plot(thresholds, recalls, label="Recall", linestyle=":", color="#D62828")
    
    # Marcar el óptimo
    plt.axvline(x=best_th, color="gray", linestyle="-.", alpha=0.7)
    plt.plot(best_th, max_f1, marker="*", markersize=15, color="gold", markeredgecolor="black", label=f"Óptimo ({best_th:.2f})")
    
    plt.title(f"Optimización Paramétrica del Umbral V2 (Filtro Temporal N={MIN_CONSECUTIVE})", fontweight="bold")
    plt.xlabel("Umbral de Decisión (Threshold)")
    plt.ylabel("Puntuación Métrica")
    plt.legend(loc="lower center")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    out_plot = os.path.join(RESULTS_DIR, "parametric_sweep_threshold.png")
    plt.savefig(out_plot, dpi=150)
    print(f"\nGráfica guardada en: {out_plot}")

if __name__ == "__main__":
    main()
