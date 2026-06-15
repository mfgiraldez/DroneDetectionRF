import os, sys, torch, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, precision_recall_curve, average_precision_score, auc

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelo_v4.model_v4 import DualStreamCVCNN_V4

GOLDEN_CSV = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
DATA_DIR   = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CKPT_PATH  = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v4\checkpoints\best_model.pth"
OUT_DIR    = r"c:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\DualStream-Deep"
WINDOW_LEN = 131072

def sliding_window_inference(model, iq_full, device):
    L = iq_full.shape[1]
    step = WINDOW_LEN // 2
    windows = []
    for start in range(0, L - WINDOW_LEN + 1, step):
        crop = iq_full[:, start : start + WINDOW_LEN].clone()
        rms = torch.sqrt(torch.mean(crop**2) + 1e-12)
        windows.append(crop / rms)
    if len(windows) == 0:
        return 0.0
    batch = torch.stack(windows).to(device)
    global_nf = torch.median(torch.sqrt(torch.mean(iq_full**2, dim=0)))
    phys = torch.zeros(batch.size(0), 4).to(device)
    phys[:, 0] = global_nf / 10.0
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits, _ = model(batch, phys)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return float(np.max(probs))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"Cargando modelo V4 (DualStream-Deep) desde {CKPT_PATH}...")
    model = DualStreamCVCNN_V4().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    df_golden = pd.read_csv(GOLDEN_CSV)
    csv_path = os.path.join(OUT_DIR, "golden_results_v4.csv")
    
    if os.path.exists(csv_path):
        print("Cargando resultados previos de CSV...")
        df_res = pd.read_csv(csv_path)
    else:
        results = []
        for idx, row in tqdm(df_golden.iterrows(), total=len(df_golden), desc="Evaluando Golden Set"):
            fpath = os.path.join(DATA_DIR, row['filename'])
            if not os.path.exists(fpath): continue
            try:
                d = torch.load(fpath, map_location='cpu', weights_only=False)
                iq_full = d['x_iq'].float()
                prob_max = sliding_window_inference(model, iq_full, device)
                
                label_bin = 0 if int(row['target']) == 4 else 1
                results.append({
                    'filename': row['filename'],
                    'label': label_bin,
                    'snr': row['snr'],
                    'target_multiclass': row['target'],
                    'prob_max': prob_max,
                    'predicted': 1 if prob_max >= 0.5 else 0
                })
            except Exception as e:
                continue
        df_res = pd.DataFrame(results)
        df_res.to_csv(csv_path, index=False)

    df_res['correct'] = (df_res['label'] == df_res['predicted']).astype(int)
    y_true = df_res['label']
    y_pred = df_res['predicted']
    
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

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", linewidths=0.4, linecolor="#1a1a2e",
                xticklabels=['Ruido', 'Dron'], yticklabels=['Ruido', 'Dron'], ax=ax)
    ax.set_title("DualStream-Deep (V4) -- Matriz de Confusión", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Real", fontsize=11); ax.set_xlabel("Predicho", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 2. PR Curve
    y_probs = df_res['prob_max']
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    auc_pr = auc(recall, precision)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, linewidth=2.5, color="#4ECDC4", 
            label=f"DualStream-Deep (V4) (AP={ap:.4f} | AUC-PR={auc_pr:.4f})")
    ax.axhline(baseline, color="#888", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline:.2f})")
    th_idx = min(np.searchsorted(thresholds, 0.5), len(precision) - 2)
    ax.scatter([recall[th_idx]], [precision[th_idx]], color="#FF6B6B", s=120, zorder=5,
               label=f"Umbral=0.5 (P={precision[th_idx]:.3f}, R={recall[th_idx]:.3f})")
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("DualStream-Deep (V4) -- Curva Precision-Recall", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pr_curve.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 3. Recall SNR Lines
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
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("DualStream-Deep (V4) -- Recall vs SNR por Emisor", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(min(snrs)-1, max(snrs)+1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "recall_snr_lines.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 4. Accuracy SNR Bars
    fig, ax = plt.subplots(figsize=(11, 6))
    acc_global = [df_res[df_res['snr'] == s]['correct'].mean() for s in sorted(df_res['snr'].unique())]
    acc_drones = [df_res[(df_res['snr'] == s) & (df_res['label'] == 1)]['correct'].mean() for s in sorted(df_res['snr'].unique())]
    acc_noise  = [df_res[(df_res['snr'] == s) & (df_res['label'] == 0)]['correct'].mean() for s in sorted(df_res['snr'].unique())]
    common_snrs = sorted(df_res['snr'].unique())
    ax.plot(common_snrs, acc_global, marker="D", markersize=6, linewidth=2.5, color="#1A1A1A", label="Accuracy Global")
    ax.plot(common_snrs, acc_drones, marker="o", markersize=5, linewidth=2,   color="#0077B6", label="Recall Drones")
    ax.plot(common_snrs, acc_noise,  marker="s", markersize=5, linewidth=2,   color="#D62828", linestyle="--", label="Especificidad Ruido")
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Tasa de Acierto", fontsize=12)
    ax.set_title("DualStream-Deep (V4) -- Accuracy por SNR", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05); ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "accuracy_snr_bars.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    # 5. Heatmap Class x SNR
    hm_data = df_res.pivot_table(index='target_multiclass', columns='snr', values='correct', aggfunc='mean')
    hm_data.index = [TARGET_NAMES_FIXED.get(i) for i in hm_data.index]
    fig, ax = plt.subplots(figsize=(max(14, len(hm_data.columns)*0.55), max(6, len(hm_data)*0.9)))
    sns.heatmap(hm_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e", ax=ax)
    ax.set_title("DualStream-Deep (V4) -- Recall/TNR por Clase y SNR", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "heatmap_snr_class.pdf"), **FIGSAVE_FIXED)
    plt.close(fig)

    print("¡FINALIZADO! Figuras PDF generadas exitosamente.")

if __name__ == "__main__":
    main()
