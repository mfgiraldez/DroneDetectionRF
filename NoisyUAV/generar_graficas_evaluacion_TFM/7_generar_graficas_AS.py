import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score,
                             f1_score, precision_score, precision_recall_curve,
                             average_precision_score, auc)
from pathlib import Path

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
    0: "#1A237E", 1: "#D62828", 2: "#118AB2",
    3: "#2D6A4F", 5: "#E07C00", 6: "#7B2D8B",
}
FIGSAVE_FIXED = dict(dpi=300, bbox_inches="tight", format="pdf")

CONFIGS = [
    {
        "name": "DualStream-SlidingWindow",
        "csv_path": r"C:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\DualStream-SlidingWindow\golden_results_dualstream_sliding.csv",
        "out_dir": r"C:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\DualStream-SlidingWindow_AS",
        "fig_suffix": "_AS"
    },
    {
        "name": "DualStream-DynamicSlidingWindow (global CFAR)",
        "csv_path": r"C:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\DualStream-DynamicSlidingWindow\golden_results_dualstream_dynamic_global.csv",
        "out_dir": r"C:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\DualStream-DynamicSlidingWindow_AS",
        "fig_suffix": "_global_AS"
    },
    {
        "name": "DualStream-DynamicSlidingWindow (local CFAR)",
        "csv_path": r"C:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\DualStream-DynamicSlidingWindow\golden_results_dualstream_dynamic_local.csv",
        "out_dir": r"C:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\DualStream-DynamicSlidingWindow_AS",
        "fig_suffix": "_local_AS"
    }
]

OPT_TH = 0.75
TITLE_SUFFIX = f"\n- Modo Alta Sensibilidad (Umbral={OPT_TH:.2f})"

def generate_plots(config):
    csv_path = config["csv_path"]
    if not os.path.exists(csv_path):
        print(f"No se encuentra el CSV: {csv_path}")
        return

    out_dir = config["out_dir"]
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    fsuffix = config["fig_suffix"]

    df_res = pd.read_csv(csv_path)
    
    df_res['predicted'] = (df_res['prob_max'] >= OPT_TH).astype(int)
    df_res['correct'] = (df_res['label'] == df_res['predicted']).astype(int)

    y_true = df_res['label']
    y_pred = df_res['predicted']
    y_probs = df_res['prob_max']

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    print(f"\n[{config['name']}] Alta Sensibilidad:")
    print(f"Accuracy: {acc:.4f} | Recall: {rec:.4f} | Precision: {pre:.4f} | F1: {f1:.4f}")

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", linewidths=0.4, linecolor="#1a1a2e",
                xticklabels=['Ruido', 'Dron'], yticklabels=['Ruido', 'Dron'], ax=ax)
    ax.set_title(f"Matriz de Confusión sobre Test Ciego{TITLE_SUFFIX}", fontweight="bold", pad=12)
    ax.set_ylabel("Clase Real")
    ax.set_xlabel("Clase Predicha")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"confusion_matrix{fsuffix}.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 2. Curva Precision-Recall
    precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    auc_pr = auc(recall_curve, precision_curve)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall_curve, precision_curve, linewidth=2.5, color="#4ECDC4",
            label=f"{config['name']} (AP={ap:.4f})")
    ax.axhline(baseline, color="#888", linestyle="--", linewidth=1.5,
               label=f"Línea Base Aleatoria ({baseline:.2f})")
    
    # Plot the AS threshold point
    idx = min(np.searchsorted(thresholds_curve, OPT_TH), len(precision_curve) - 2)
    ax.scatter([recall_curve[idx]], [precision_curve[idx]], color="#FF6B6B", s=120, zorder=5,
               label=f"Umbral AS = {OPT_TH:.2f}")
               
    ax.set_xlabel("Exhaustividad (Recall)")
    ax.set_ylabel("Precisión (Precision)")
    ax.set_title(f"Curva Precision-Recall{TITLE_SUFFIX}", fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"pr_curve{fsuffix}.pdf"), **FIGSAVE_FIXED)
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
    ax.set_title(f"Tasa de Detección por Emisor RF en función de la SNR{TITLE_SUFFIX}", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(snrs) - 1, max(snrs) + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"recall_snr_lines{fsuffix}.pdf"), **FIGSAVE_FIXED)
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
    ax.set_title(f"Exactitud del Clasificador por Nivel de SNR{TITLE_SUFFIX}", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(common_snrs) - 1, max(common_snrs) + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"accuracy_snr_bars{fsuffix}.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 5. Heatmap Clase × SNR
    hm_data = df_res.pivot_table(index='target_multiclass', columns='snr', values='correct', aggfunc='mean')
    hm_data.index = [TARGET_NAMES_FIXED.get(i) for i in hm_data.index]
    fig, ax = plt.subplots(figsize=(max(14, len(hm_data.columns) * 0.55), max(6, len(hm_data) * 0.9)))
    sns.heatmap(hm_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e",
                cbar_kws={'label': 'Tasa de Acierto (Recall)'}, ax=ax)
    ax.set_title(f"Mapa de Calor: Tasa de Acierto por Clase y SNR{TITLE_SUFFIX}", fontweight="bold")
    ax.set_xlabel("Relación Señal a Ruido (SNR) [dB]")
    ax.set_ylabel("Emisor RF")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"heatmap_snr_class{fsuffix}.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    report = (
        f"=================================================\n"
        f"INFORME GOLDEN SET — {config['name']} (Alta Sensibilidad)\n"
        f"=================================================\n"
        f"Umbral AS       : {OPT_TH:.4f}\n"
        f"Accuracy Global : {acc:.4f}\n"
        f"Recall (Drones) : {rec:.4f}\n"
        f"Precision       : {pre:.4f}\n"
        f"F1-Score        : {f1:.4f}\n"
        f"=================================================\n"
    )
    with open(os.path.join(out_dir, f"summary_golden{fsuffix}.txt"), 'w') as fh:
        fh.write(report)

if __name__ == "__main__":
    for c in CONFIGS:
        generate_plots(c)
