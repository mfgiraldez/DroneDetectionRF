"""
evaluate_ht_golden.py — Evaluación Hard Test sobre el Golden Set
=================================================================
Evalúa el modelo entrenado SIN Target=5 (Taranis) sobre el Golden Set completo.

MÉTRICA CLAVE: Comparar Recall(Target=5) vs Recall(otros targets).
Si son similares → el modelo aprendió la firma universal FHSS (generalización real).
Si Target=5 cae significativamente → hay memorización parcial.
"""
import os, sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             accuracy_score, average_precision_score,
                             roc_auc_score, roc_curve, precision_recall_curve, auc,
                             confusion_matrix)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))

from model import DualStreamCVCNN
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

TARGET_HELD_OUT = 5   # Taranis — el dron que NO vio el modelo

OUT_DIR     = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual_hard_test"
DATA_DIR    = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CKPT_PATH   = os.path.join(OUT_DIR, "checkpoints", "best_model.pth")
RESULTS_DIR = os.path.join(OUT_DIR, "figures_ht")
CSV_GOLDEN  = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"

FS              = 14e6
WIN_LEN         = 131072
N_STEPS         = 16
THRESHOLD_V2    = 0.75
MIN_CONSECUTIVE = 2

TARGET_NAMES = {0:"DJI (T0)", 1:"FutabaT14 (T1)", 2:"FutabaT7 (T2)",
                3:"Graupner (T3)", 5:"Taranis (T5) ★HELD-OUT★", 6:"Turnigy (T6)", 4:"Ruido (T4)"}
TARGET_COLORS = {0:"#3F37C9", 1:"#D62828", 2:"#118AB2",
                 3:"#2D6A4F", 5:"#FF6B35", 6:"#7B2D8B"}

FIGSAVE = dict(dpi=150, bbox_inches="tight")

def calc_phys(iq_full):
    try:
        _, _, H_smooth, _, nf_v, _, _, bursts = detectar_bursts(
            iq_full, fs=FS, nperseg=2048, z_thresh=2.5)
        nf   = float(np.median(nf_v))
        Hmean= float(np.mean(H_smooth))
        zp   = max([abs(b['z_peak']) for b in bursts]) if bursts else 0.0
        return [nf, Hmean, min(zp, 30.0)]
    except:
        return [0.0, 0.0, 0.0]

def main():
    Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HT-EVAL] Device: {device}")
    print(f"[HT-EVAL] Target HELD-OUT (nunca visto): Target={TARGET_HELD_OUT} (Taranis)")

    model = DualStreamCVCNN().to(device)
    ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[HT-EVAL] Modelo cargado: epoch={ckpt.get('epoch','?')} | Val F1={ckpt.get('val_f1','?'):.4f}")

    df_golden = pd.read_csv(CSV_GOLDEN)
    print(f"[HT-EVAL] Golden Set: {len(df_golden)} ficheros")

    results = []
    with torch.no_grad():
        for _, row in tqdm(df_golden.iterrows(), total=len(df_golden), desc="Inferencia HT"):
            fpath = os.path.join(DATA_DIR, row['filename'])
            if not os.path.exists(fpath): continue

            d = torch.load(fpath, map_location='cpu', weights_only=False)
            iq_full = d['x_iq'].float()

            phys_vals  = calc_phys(iq_full)
            phys_tensor= torch.tensor([phys_vals], dtype=torch.float32).to(device)

            max_idx = iq_full.shape[1]
            step    = (max_idx - WIN_LEN) // (N_STEPS - 1)
            probs   = []
            for i in range(N_STEPS):
                start = i * step
                win   = iq_full[:, start:start+WIN_LEN]
                power = win.pow(2).mean().clamp(min=1e-12).sqrt()
                win_n = (win / power).unsqueeze(0).to(device)
                logits, _ = model(win_n, phys_tensor)
                probs.append(torch.sigmoid(logits).item())

            consec = 0; confirmed = 0
            for p in probs:
                if p >= THRESHOLD_V2:
                    consec += 1
                    if consec >= MIN_CONSECUTIVE:
                        confirmed = 1; break
                else:
                    consec = 0

            label_bin = 0 if int(row['target']) == 4 else 1
            is_held_out = (int(row['target']) == TARGET_HELD_OUT)
            results.append({'filename': row['filename'], 'target': row['target'],
                            'snr': row['snr'], 'label_bin': label_bin,
                            'pred_bin': confirmed, 'max_prob': max(probs),
                            'is_held_out': is_held_out})

    df = pd.DataFrame(results)
    df['correct'] = (df['label_bin'] == df['pred_bin']).astype(int)
    csv_out = os.path.join(RESULTS_DIR, "ht_golden_results.csv")
    df.to_csv(csv_out, index=False)

    # ── MÉTRICAS GLOBALES ────────────────────────────────────────────────────
    y_true = df['label_bin']; y_pred = df['pred_bin']; y_prob = df['max_prob']
    print("\n" + "="*60)
    print("MÉTRICAS GLOBALES — Hard Test (Golden Set)")
    print(f"  Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  AP:        {average_precision_score(y_true, y_prob):.4f}")

    # ── MÉTRICA CLAVE: T5 (Held-Out) vs Resto ───────────────────────────────
    df_drones = df[df['label_bin'] == 1].copy()
    print("\n" + "="*60)
    print("ANÁLISIS DE GENERALIZACIÓN ZERO-SHOT (Target=5 Taranis)")
    t5  = df_drones[df_drones['target'] == TARGET_HELD_OUT]
    rest= df_drones[df_drones['target'] != TARGET_HELD_OUT]
    r_t5   = t5['correct'].mean()   if len(t5)   > 0 else float('nan')
    r_rest = rest['correct'].mean() if len(rest)  > 0 else float('nan')
    print(f"  Recall Target={TARGET_HELD_OUT} (Taranis, NUNCA VISTO): {r_t5:.4f}  (n={len(t5)})")
    print(f"  Recall Otros Drones  (sí vistos en train):        {r_rest:.4f}  (n={len(rest)})")
    delta = r_t5 - r_rest
    print(f"  Δ Recall (T5 - Resto): {delta:+.4f}")
    if abs(delta) < 0.05:
        print("  >> CONCLUSIÓN: Generalización EXCELENTE — el modelo aprendió la firma FHSS universal.")
    elif delta > -0.10:
        print("  >> CONCLUSIÓN: Generalización BUENA — ligera degradación para el dron no visto.")
    else:
        print("  >> CONCLUSIÓN: Memorización PARCIAL — el modelo depende de firmas específicas.")
    print("="*60)

    # ── FIGURA 1: Recall por Target (comparativa principal) ──────────────────
    targets_drone = sorted(df_drones['target'].unique())
    recall_per_t  = [df_drones[df_drones['target']==t]['correct'].mean() for t in targets_drone]
    colors = [TARGET_COLORS.get(t, '#999') for t in targets_drone]
    names  = [TARGET_NAMES.get(t, f'T{t}') for t in targets_drone]
    edge_colors = ['#FF0000' if t == TARGET_HELD_OUT else 'white' for t in targets_drone]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, recall_per_t, color=colors, edgecolor=edge_colors, linewidth=2.5)
    ax.axhline(r_rest, color='#0077B6', linestyle='--', lw=1.5, label=f'Recall medio vistos ({r_rest:.3f})')
    ax.axhline(r_t5,   color='#FF6B35', linestyle='--', lw=1.5, label=f'Recall T5 held-out ({r_t5:.3f})')
    ax.set_ylim(0, 1.1); ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Hard Test — Recall por Emisor RF\n"
                 f"★ Target={TARGET_HELD_OUT} (Taranis) fue EXCLUIDO del entrenamiento",
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, recall_per_t):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.3f}",
                ha='center', fontsize=9, fontweight='bold')
    plt.xticks(rotation=25, ha='right')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ht_recall_per_target.png"), **FIGSAVE)
    plt.close(fig)

    # ── FIGURA 2: Heatmap Recall por SNR y Target ────────────────────────────
    pivot = df_drones.pivot_table(index='target', columns='snr', values='correct', aggfunc='mean')
    pivot.index = [TARGET_NAMES.get(i, f'T{i}') for i in pivot.index]
    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns)*0.55), max(6, len(pivot)*0.9)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e", ax=ax, cbar_kws={'label':'Recall'})
    ax.set_title(f"Hard Test — Recall por SNR y Emisor\n"
                 f"★ Taranis (T5) = held-out, no visto en entrenamiento", fontsize=12)
    ax.set_ylabel("Emisor RF"); ax.set_xlabel("SNR (dB)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ht_heatmap_snr_target.png"), **FIGSAVE)
    plt.close(fig)

    # ── FIGURA 3: Recall vs SNR (líneas por target) ──────────────────────────
    snrs = sorted(df_drones['snr'].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    for t in targets_drone:
        sub = df_drones[df_drones['target'] == t]
        rec = [sub[sub['snr']==s]['correct'].mean() for s in snrs]
        lw  = 3.0 if t == TARGET_HELD_OUT else 1.8
        ls  = '-'
        mk  = 's' if t == TARGET_HELD_OUT else 'o'
        ax.plot(snrs, rec, marker=mk, markersize=6 if t==TARGET_HELD_OUT else 4,
                linewidth=lw, linestyle=ls,
                color=TARGET_COLORS.get(t,'#999'), label=TARGET_NAMES.get(t,f'T{t}'))
    ax.axvspan(-20, -7,  alpha=0.07, color="red",    label="Grupo C")
    ax.axvspan(-6,   9,  alpha=0.05, color="yellow", label="Grupo B")
    ax.axvspan(10,  30,  alpha=0.07, color="green",  label="Grupo A")
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Hard Test — Recall vs SNR\n"
                 f"Línea más gruesa = Target={TARGET_HELD_OUT} (Taranis, held-out)", fontsize=12)
    ax.set_ylim(-0.05, 1.05); ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2); ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ht_recall_snr_lines.png"), **FIGSAVE)
    plt.close(fig)

    # ── FIGURA 4: Curva PR ───────────────────────────────────────────────────
    prec_pts, rec_pts, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 7))
    for f_score in [0.5, 0.6, 0.7, 0.8, 0.9]:
        x = np.linspace(0.01, 1, 500)
        y = f_score * x / (2 * x - f_score)
        valid = (y >= 0) & (y <= 1)
        ax.plot(x[valid], y[valid], color="gray", alpha=0.4, linestyle=":", linewidth=1)
        if len(x[valid]) > 0:
            ax.text(x[valid][-10], y[valid][-10], f"F1={f_score}", color="gray", fontsize=8)
    ax.plot(rec_pts, prec_pts, lw=2.5, color="#4ECDC4",
            label=f"Dual-Stream HT (AP={ap:.4f})")
    curr_p = precision_score(y_true, y_pred, zero_division=0)
    curr_r = recall_score(y_true, y_pred)
    ax.plot(curr_r, curr_p, 'o', markersize=12, color='#FF595E',
            label=f"Op. Point (τ=0.75)\nP={curr_p:.3f} R={curr_r:.3f}")
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precisión", fontsize=12)
    ax.set_title("Hard Test — Curva Precision-Recall", fontsize=13)
    ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ht_pr_curve.png"), **FIGSAVE)
    plt.close(fig)

    # ── FIGURA 5: Confusion Matrix ───────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=['Ruido', 'Dron'], yticklabels=['Ruido', 'Dron'], ax=ax)
    ax.set_title("Hard Test — Matriz de Confusión (Umbral: 0.75)", fontsize=12)
    ax.set_ylabel("Real"); ax.set_xlabel("Predicho")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ht_confusion_matrix.png"), **FIGSAVE)
    plt.close(fig)

    print(f"\n[HT-EVAL] Figuras guardadas en: {RESULTS_DIR}")
    print(f"[HT-EVAL] Resultados CSV: {csv_out}")

if __name__ == "__main__":
    main()
