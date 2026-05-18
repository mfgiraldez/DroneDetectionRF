import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, precision_recall_curve, auc, average_precision_score, roc_curve, roc_auc_score

# Config
OUT_DIR = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual\figures_golden_filtrado_umbral75"
CSV_PATH = os.path.join(OUT_DIR, "golden_results_v2_filt.csv")

TARGET_NAMES_FIXED = {
    0: "DJI (T0)", 1: "FutabaT14 (T1)", 2: "FutabaT7 (T2)",
    3: "Graupner (T3)", 5: "Taranis (T5)", 6: "Turnigy (T6)",
    4: "Ruido (T4)",
}
TARGET_COLORS_FIXED = {
    0: "#3F37C9", 1: "#D62828", 2: "#118AB2",
    3: "#2D6A4F", 5: "#E07C00", 6: "#7B2D8B",
}
FIGSAVE_FIXED = dict(dpi=150, bbox_inches="tight")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: No existe {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    # Mapeo de columnas para compatibilidad con código legacy
    df['label'] = df['label_bin']
    df['predicted'] = df['pred_bin']
    df['prob_max'] = df['max_prob']
    df['target_multiclass'] = df['target']
    df['correct'] = (df['label'] == df['predicted']).astype(int)

    y_true = df['label']
    y_pred = df['predicted']
    y_probs = df['prob_max']

    print("\n" + "="*40)
    print(f"REPORTE FINAL V2 (LÓGICA FILTRADA)")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print("="*40)

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", linewidths=0.4, linecolor="#1a1a2e",
                xticklabels=['Ruido', 'Dron'], yticklabels=['Ruido', 'Dron'], ax=ax)
    ax.set_title("Arquitectura Dual-Stream V2.1 - Matriz de Confusión Global (Umbral: 0.75)", fontsize=13, pad=12)
    ax.set_ylabel("Real", fontsize=11); ax.set_xlabel("Predicho", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix_golden.png"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 2. Curva Precision-Recall
    precision_pts, recall_pts, thresholds = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    auc_pr = auc(recall_pts, precision_pts)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Líneas Iso-F1
    for f_score in [0.5, 0.6, 0.7, 0.8, 0.9]:
        x = np.linspace(0.01, 1, 500)
        y = f_score * x / (2 * x - f_score)
        valid = (y >= 0) & (y <= 1)
        ax.plot(x[valid], y[valid], color="gray", alpha=0.4, linestyle=":", linewidth=1)
        if len(x[valid]) > 0:
            ax.text(x[valid][-10], y[valid][-10], f"F1={f_score}", color="gray", alpha=0.8, fontsize=8)

    ax.plot(recall_pts, precision_pts, linewidth=2.5, color="#4ECDC4", 
            label=f"Dual-Stream V2.1 (AP={ap:.4f} | AUC-PR={auc_pr:.4f})")
            
    # Marcar el punto óptimo operativo
    curr_precision = precision_score(y_true, y_pred, zero_division=0)
    curr_recall = recall_score(y_true, y_pred)
    ax.plot(curr_recall, curr_precision, marker='o', markersize=12, color='#FF595E', 
            linestyle='None', label=f"Punto Operativo (Umbral=0.75)\nP={curr_precision:.3f} R={curr_recall:.3f}")

    ax.axhline(baseline, color="#888", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precisión", fontsize=12)
    ax.set_title("Arquitectura Dual-Stream V2.1 - Curva de Precisión-Recall (PR)", fontsize=13)
    ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pr_curve_golden.png"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 2.5 Curva ROC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, linewidth=2.5, color="#7B2D8B", label=f"Dual-Stream V2.1 (AUC-ROC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1.5, label="Aleatorio (0.50)")
    
    # Marcar el punto óptimo operativo en la ROC
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    curr_fpr_calc = fp / (fp + tn)
    curr_tpr_calc = tp / (tp + fn)
    
    ax.plot(curr_fpr_calc, curr_tpr_calc, marker='o', markersize=12, color='#FF595E', 
            linestyle='None', label=f"Punto Operativo (Umbral=0.75)\nFPR={curr_fpr_calc:.3f} TPR={curr_tpr_calc:.3f}")

    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("TPR (Recall)", fontsize=12)
    ax.set_title("Arquitectura Dual-Stream V2.1 - Curva ROC", fontsize=13)
    ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "roc_curve_golden.png"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 3. Recall vs SNR Lines
    drones = df[df['label'] == 1].copy()
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
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Arquitectura Dual-Stream V2.1 - Recall frente a Relación Señal-Ruido", fontsize=13)
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(min(snrs)-1, max(snrs)+1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "recall_snr_lines_golden.png"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 4. Accuracy per SNR
    fig, ax = plt.subplots(figsize=(11, 6))
    common_snrs = sorted(df['snr'].unique())
    acc_global = [df[df['snr'] == s]['correct'].mean() for s in common_snrs]
    acc_drones = [df[(df['snr'] == s) & (df['label'] == 1)]['correct'].mean() for s in common_snrs]
    acc_noise  = [df[(df['snr'] == s) & (df['label'] == 0)]['correct'].mean() for s in common_snrs]
    
    ax.plot(common_snrs, acc_global, marker="D", markersize=6, linewidth=2.5, color="#1A1A1A", label="Accuracy Global")
    ax.plot(common_snrs, acc_drones, marker="o", markersize=5, linewidth=2,   color="#0077B6", label="Recall Drones")
    ax.plot(common_snrs, acc_noise,  marker="s", markersize=5, linewidth=2,   color="#D62828", linestyle="--", label="Especificidad Ruido")
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Tasa de Acierto", fontsize=12)
    ax.set_title("Arquitectura Dual-Stream V2.1 - Tasa de Acierto Global frente a Relación Señal-Ruido", fontsize=13)
    ax.set_ylim(-0.05, 1.05); ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "accuracy_snr_bars_golden.png"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 5. Heatmap
    hm_data = df.pivot_table(index='target_multiclass', columns='snr', values='correct', aggfunc='mean')
    hm_data.index = [TARGET_NAMES_FIXED.get(i) for i in hm_data.index]
    fig, ax = plt.subplots(figsize=(max(14, len(hm_data.columns)*0.55), max(6, len(hm_data)*0.9)))
    sns.heatmap(hm_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e", ax=ax, 
                cbar_kws={'label': 'Recall / Especificidad'})
    ax.set_ylabel("Emisor RF", fontsize=12)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_title("Arquitectura Dual-Stream V2.1 - Mapa de Calor de Rendimiento (Recall/Especificidad)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "heatmap_snr_class_golden.png"), **FIGSAVE_FIXED)
    plt.close(fig)
    
    print(f"Finalizado. Todas las figuras regeneradas con estilo Legacy en: {OUT_DIR}")

if __name__ == "__main__":
    main()
