import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, precision_recall_curve, auc, average_precision_score, roc_curve, roc_auc_score

# Config
OUT_DIR  = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual_hard_test\figures_ht"
CSV_PATH = os.path.join(OUT_DIR, "ht_golden_results.csv")

TARGET_HELD_OUT = 5   # Taranis — nunca visto en entrenamiento

TARGET_NAMES_FIXED = {
    0: "DJI (T0)", 1: "FutabaT14 (T1)", 2: "FutabaT7 (T2)",
    3: "Graupner (T3)", 5: "Taranis (T5) ★", 6: "Turnigy (T6)",
    4: "Ruido (T4)",
}
TARGET_COLORS_FIXED = {
    0: "#3F37C9", 1: "#D62828", 2: "#118AB2",
    3: "#2D6A4F", 5: "#FF6B35", 6: "#7B2D8B",
}
FIGSAVE_FIXED = dict(dpi=150, bbox_inches="tight")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: No existe {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    df['label']            = df['label_bin']
    df['predicted']        = df['pred_bin']
    df['prob_max']         = df['max_prob']
    df['target_multiclass']= df['target']
    df['correct']          = (df['label'] == df['predicted']).astype(int)

    y_true  = df['label']
    y_pred  = df['predicted']
    y_probs = df['prob_max']

    print("\n" + "="*50)
    print("REPORTE HARD TEST V2.1 (Sin Target=5 en train)")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")

    drones = df[df['label']==1]
    r_t5   = drones[drones['target']==TARGET_HELD_OUT]['correct'].mean()
    r_rest = drones[drones['target']!=TARGET_HELD_OUT]['correct'].mean()
    print(f"\nRecall T5 Taranis (HELD-OUT, nunca visto): {r_t5:.4f}")
    print(f"Recall otros drones (vistos en train):     {r_rest:.4f}")
    print(f"Delta: {r_t5-r_rest:+.4f}")
    print("="*50)

    # 1. Matriz de Confusion
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", linewidths=0.4, linecolor="#1a1a2e",
                xticklabels=['Ruido', 'Dron'], yticklabels=['Ruido', 'Dron'], ax=ax)
    ax.set_title("Hard Test V2.1 - Matriz de Confusion Global\n(Umbral: 0.75 | Target=5 Taranis no visto en train)", fontsize=12, pad=12)
    ax.set_ylabel("Real", fontsize=11); ax.set_xlabel("Predicho", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ht_confusion_matrix.png"), **FIGSAVE_FIXED)
    plt.close(fig)
    print("OK: confusion_matrix")

    # 2. Curva Precision-Recall
    precision_pts, recall_pts, thresholds = precision_recall_curve(y_true, y_probs)
    ap      = average_precision_score(y_true, y_probs)
    auc_pr  = auc(recall_pts, precision_pts)
    baseline= y_true.mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    for f_score in [0.5, 0.6, 0.7, 0.8, 0.9]:
        x = np.linspace(0.01, 1, 500)
        y = f_score * x / (2 * x - f_score)
        valid = (y >= 0) & (y <= 1)
        ax.plot(x[valid], y[valid], color="gray", alpha=0.4, linestyle=":", linewidth=1)
        if len(x[valid]) > 0:
            ax.text(x[valid][-10], y[valid][-10], f"F1={f_score}", color="gray", alpha=0.8, fontsize=8)
    ax.plot(recall_pts, precision_pts, linewidth=2.5, color="#4ECDC4",
            label=f"Hard Test V2.1 (AP={ap:.4f} | AUC-PR={auc_pr:.4f})")
    curr_precision = precision_score(y_true, y_pred, zero_division=0)
    curr_recall    = recall_score(y_true, y_pred)
    ax.plot(curr_recall, curr_precision, marker='o', markersize=12, color='#FF595E',
            linestyle='None', label=f"Punto Operativo (Umbral=0.75)\nP={curr_precision:.3f} R={curr_recall:.3f}")
    ax.axhline(baseline, color="#888", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Hard Test V2.1 - Curva de Precision-Recall\n(Target=5 Taranis excluido del entrenamiento)", fontsize=13)
    ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ht_pr_curve.png"), **FIGSAVE_FIXED)
    plt.close(fig)
    print("OK: pr_curve")

    # 3. Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, linewidth=2.5, color="#7B2D8B", label=f"Hard Test V2.1 (AUC-ROC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1.5, label="Aleatorio (0.50)")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    curr_fpr_calc = fp / (fp + tn)
    curr_tpr_calc = tp / (tp + fn)
    ax.plot(curr_fpr_calc, curr_tpr_calc, marker='o', markersize=12, color='#FF595E',
            linestyle='None', label=f"Punto Operativo (Umbral=0.75)\nFPR={curr_fpr_calc:.3f} TPR={curr_tpr_calc:.3f}")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("TPR (Recall)", fontsize=12)
    ax.set_title("Hard Test V2.1 - Curva ROC\n(Target=5 Taranis excluido del entrenamiento)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ht_roc_curve.png"), **FIGSAVE_FIXED)
    plt.close(fig)
    print("OK: roc_curve")

    # 4. Recall vs SNR por target (lineas) — igual que v2_1 pero con T5 destacado
    drones = df[df['label'] == 1].copy()
    snrs   = sorted(drones['snr'].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    for target in sorted(drones['target_multiclass'].unique()):
        sub = drones[drones['target_multiclass'] == target]
        rec = [sub[sub['snr'] == s]['correct'].mean() for s in snrs]
        lw  = 3.0 if target == TARGET_HELD_OUT else 1.8
        mk  = 's' if target == TARGET_HELD_OUT else 'o'
        ms  = 8   if target == TARGET_HELD_OUT else 5
        ax.plot(snrs, rec, marker=mk, markersize=ms, linewidth=lw,
                color=TARGET_COLORS_FIXED.get(target), label=TARGET_NAMES_FIXED.get(target))
    global_recall = [drones[drones['snr'] == s]['correct'].mean() for s in snrs]
    ax.plot(snrs, global_recall, marker="D", markersize=6, linewidth=2.5,
            color="#1A1A1A", linestyle="--", label="Media Global (Drones)", zorder=5)
    ax.axvspan(-20, -7,  alpha=0.07, color="red",    label="Grupo C")
    ax.axvspan(-6,   9,  alpha=0.05, color="yellow", label="Grupo B")
    ax.axvspan(10,  30,  alpha=0.07, color="green",  label="Grupo A")
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Hard Test V2.1 - Recall vs SNR\n(cuadrados naranjas = Taranis T5, HELD-OUT, nunca visto en entrenamiento)", fontsize=12)
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(min(snrs)-1, max(snrs)+1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ht_recall_snr_lines.png"), **FIGSAVE_FIXED)
    plt.close(fig)
    print("OK: recall_snr_lines")

    # 5. Accuracy per SNR (global + drones + ruido)
    fig, ax = plt.subplots(figsize=(11, 6))
    common_snrs = sorted(df['snr'].unique())
    acc_global  = [df[df['snr'] == s]['correct'].mean() for s in common_snrs]
    acc_drones  = [df[(df['snr'] == s) & (df['label'] == 1)]['correct'].mean() for s in common_snrs]
    acc_noise   = [df[(df['snr'] == s) & (df['label'] == 0)]['correct'].mean() for s in common_snrs]
    ax.plot(common_snrs, acc_global, marker="D", markersize=6, linewidth=2.5, color="#1A1A1A", label="Accuracy Global")
    ax.plot(common_snrs, acc_drones, marker="o", markersize=5, linewidth=2,   color="#0077B6", label="Recall Drones")
    ax.plot(common_snrs, acc_noise,  marker="s", markersize=5, linewidth=2,   color="#D62828", linestyle="--", label="Especificidad Ruido")
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Tasa de Acierto", fontsize=12)
    ax.set_title("Hard Test V2.1 - Tasa de Acierto Global vs SNR\n(Target=5 Taranis excluido del entrenamiento)", fontsize=13)
    ax.set_ylim(-0.05, 1.05); ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray"); ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ht_accuracy_snr.png"), **FIGSAVE_FIXED)
    plt.close(fig)
    print("OK: accuracy_snr")

    # 6. HEATMAP completo (todos los targets, incluido T5)
    hm_data = df.pivot_table(index='target_multiclass', columns='snr', values='correct', aggfunc='mean')
    hm_data.index = [TARGET_NAMES_FIXED.get(i, f"T{i}") for i in hm_data.index]
    fig, ax = plt.subplots(figsize=(max(14, len(hm_data.columns)*0.55), max(6, len(hm_data)*0.9)))
    sns.heatmap(hm_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e", ax=ax,
                cbar_kws={'label': 'Recall / Especificidad'})
    ax.set_ylabel("Emisor RF", fontsize=12)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_title("Hard Test V2.1 - Mapa de Calor de Rendimiento\n"
                 "(★ Taranis T5 = HELD-OUT, nunca visto durante el entrenamiento)", fontsize=13)
    # Resaltar la fila de T5 con un borde
    t5_row_idx = list(hm_data.index).index("Taranis (T5) ★") if "Taranis (T5) ★" in hm_data.index else None
    if t5_row_idx is not None:
        ax.add_patch(plt.Rectangle(
            (0, t5_row_idx), len(hm_data.columns), 1,
            fill=False, edgecolor='#FF6B35', lw=3, clip_on=False
        ))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ht_heatmap_snr_target.png"), **FIGSAVE_FIXED)
    plt.close(fig)
    print("OK: heatmap_snr_target")

    # 7. Recall por target (barras) — la figura clave del experimento
    targets_drone = sorted(drones['target_multiclass'].unique())
    recall_per_t  = [drones[drones['target_multiclass']==t]['correct'].mean() for t in targets_drone]
    colors_bar    = [TARGET_COLORS_FIXED.get(t, '#999') for t in targets_drone]
    names_bar     = [TARGET_NAMES_FIXED.get(t, f'T{t}') for t in targets_drone]
    edge_colors   = ['#FF0000' if t == TARGET_HELD_OUT else '#FFFFFF' for t in targets_drone]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names_bar, recall_per_t, color=colors_bar, edgecolor=edge_colors, linewidth=2.5)
    ax.axhline(r_rest, color='#0077B6', linestyle='--', lw=1.5, label=f'Recall medio drones vistos ({r_rest:.3f})')
    ax.axhline(r_t5,   color='#FF6B35', linestyle='--', lw=1.5, label=f'Recall Taranis held-out ({r_t5:.3f})')
    ax.set_ylim(0, 1.1); ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Hard Test V2.1 - Recall por Emisor RF\n"
                 "Taranis (★) fue EXCLUIDO del entrenamiento — test de generalizacion zero-shot", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, recall_per_t):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.3f}",
                ha='center', fontsize=9, fontweight='bold')
    plt.xticks(rotation=25, ha='right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ht_recall_per_target.png"), **FIGSAVE_FIXED)
    plt.close(fig)
    print("OK: recall_per_target")

    print(f"\nFinalizado. Todas las figuras en: {OUT_DIR}")

if __name__ == "__main__":
    main()
