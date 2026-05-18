"""
evaluate_v4_cfar.py — Evaluación completa del modelo V4 Guiado por CFAR sobre el Golden Set
========================================================================================
Genera las figuras avanzadas adaptadas de alumn_v2_dual_eval.py.
"""

import sys, os, argparse, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (precision_recall_curve, auc, average_precision_score,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
os.environ["PYTHONIOENCODING"] = "utf-8"

from NoisyUAV.modelo_v4.model_v4 import DualStreamCVCNN_V4
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

# Config
GOLDEN_CSV = r"C:\TFM_data\NoisyUAV\ground_truth_test_set.csv"
DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
OUT_DIR = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v4\figures_golden_cfar"
WINDOW_LEN = 131072
FS = 14e6
NPERSEG = 2048

TARGET_NAMES = {
    0: "DJI (T0)", 1: "FutabaT14 (T1)", 2: "FutabaT7 (T2)",
    3: "Graupner (T3)", 5: "Taranis (T5)", 6: "Turnigy (T6)",
    4: "Ruido (T4)",
}
TARGET_COLORS = {
    0: "#0077B6", 1: "#D62828", 2: "#118AB2",
    3: "#2D6A4F", 5: "#E07C00", 6: "#7B2D8B",
}
FIGSAVE = dict(dpi=150, bbox_inches="tight")


def sliding_window_inference_cfar(model, iq_full, device):
    """ Escanea el fichero de 75ms asignando features del CFAR a las ventanas """
    L = iq_full.shape[1]
    
    t_ms, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts = detectar_bursts(
        iq_full.cpu(), fs=FS, nperseg=NPERSEG, z_thresh=1.5,
        min_burst_ms=0.4, merge_gap_ms=0.5, min_z_abs=2.0,
        bg_mult=4, max_bins_frac=1.0, smooth_ms=0.2)
        
    global_nf = float(np.median(nf_v)) if len(nf_v) > 0 else 0.0
    global_H_mean = float(np.mean(H_smooth)) if len(H_smooth) > 0 else 0.0
    
    step = WINDOW_LEN // 2
    windows = []
    phys_features = []
    
    for start in range(0, L - WINDOW_LEN + 1, step):
        end = start + WINDOW_LEN
        start_ms = (start / FS) * 1000.0
        end_ms = (end / FS) * 1000.0
        
        local_z_peak = 0.0
        local_n_bins = 0
        
        for b in bursts:
            if max(start_ms, b['t0']) < min(end_ms, b['t1']):
                if abs(b['z_peak']) > local_z_peak:
                    local_z_peak = abs(b['z_peak'])
                    local_n_bins = b.get('n_act', 0) # ARREGLADO EL KEY ERROR AQUI
        
        crop = iq_full[:, start : end].clone()
        rms = torch.sqrt(torch.mean(crop**2) + 1e-12)
        windows.append(crop / rms)
        
        phys_features.append([
            global_nf / 10.0,
            global_H_mean / 10.0,
            local_z_peak / 50.0,
            local_n_bins / 2048.0
        ])
    
    batch = torch.stack(windows).to(device)
    phys = torch.tensor(phys_features, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits, _ = model(batch, phys)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    return np.max(probs)


# ---------------------------------------------------------------------------- #
# Figuras Avanzadas                                                            #
# ---------------------------------------------------------------------------- #

def plot_heatmap_drones(df, out_dir):
    drones = df[df["label"] == 1].copy()
    drones["Emisor RF"] = drones["target_multiclass"].map(lambda x: TARGET_NAMES.get(x, f"Target {x}"))
    hm = drones.pivot_table(index="Emisor RF", columns="snr", values="correct", aggfunc="mean").sort_index()
    hm = hm[sorted(hm.columns)]
    fig, ax = plt.subplots(figsize=(max(14, len(hm.columns)*0.55), max(3, len(hm)*0.85)))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e", ax=ax,
                cbar_kws={"label": "Tasa de Acierto (Recall)"})
    ax.set_title("V4 CFAR-Guided -- Deteccion por Emisor RF y SNR (Golden Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("SNR (dB)", fontsize=11); ax.set_ylabel("Emisor RF", fontsize=11)
    ax.tick_params(axis="x", rotation=45); ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "heatmap_target_snr.png", **FIGSAVE); plt.close(fig)

def plot_heatmap_noise(df, out_dir):
    noise = df[df["label"] == 0].copy()
    # En test 75ms target_multiclass para ruido es siempre 4, lo tratamos global
    noise["Emisor RF"] = noise["target_multiclass"].map(lambda x: TARGET_NAMES.get(x, f"Ruido T{x}"))
    hm = noise.pivot_table(index="Emisor RF", columns="snr", values="correct", aggfunc="mean").sort_index()
    hm = hm[sorted(hm.columns)]
    fig, ax = plt.subplots(figsize=(max(14, len(hm.columns)*0.55), max(2, len(hm)*0.9)))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e", ax=ax, cbar_kws={"label": "Especificidad (1 - FPR)"})
    ax.set_title("V4 CFAR-Guided -- Especificidad por SNR (Golden Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("SNR (dB)", fontsize=11); ax.set_ylabel("Fondo", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "heatmap_noise_snr.png", **FIGSAVE); plt.close(fig)

def plot_recall_snr_lines(df, out_dir):
    drones = df[df["label"] == 1].copy()
    snrs   = sorted(drones["snr"].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    for target in sorted(drones["target_multiclass"].unique()):
        sub = drones[drones["target_multiclass"] == target]
        recall = [sub[sub["snr"] == s]["correct"].mean() for s in snrs]
        ax.plot(snrs, recall, marker="o", markersize=5, linewidth=2,
                color=TARGET_COLORS.get(target), label=TARGET_NAMES.get(target))
    
    global_recall = [drones[drones["snr"] == s]["correct"].mean() for s in snrs]
    ax.plot(snrs, global_recall, marker="D", markersize=6, linewidth=2.5,
            color="#1A1A1A", linestyle="--", label="Media Global (Drones)", zorder=5)
    
    ax.axhline(0.9, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(0,   color="#555", linestyle="--", linewidth=1, alpha=0.5)
    
    snr_min, snr_max = min(snrs), max(snrs)
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("V4 CFAR-Guided -- Recall vs SNR por Emisor RF", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(snr_min - 1, snr_max + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "recall_snr_lines.png", **FIGSAVE); plt.close(fig)

def plot_accuracy_per_snr(df, out_dir):
    snrs = sorted(df["snr"].unique())
    acc_global = [df[df["snr"] == s]["correct"].mean() for s in snrs]
    acc_drones = [df[(df["snr"] == s) & (df["label"] == 1)]["correct"].mean() for s in snrs]
    acc_noise  = [df[(df["snr"] == s) & (df["label"] == 0)]["correct"].mean() for s in snrs]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(snrs, acc_global, marker="D", markersize=6, linewidth=2.5, color="#1A1A1A", label="Accuracy Global")
    ax.plot(snrs, acc_drones, marker="o", markersize=5, linewidth=2, color="#0077B6", label="Recall Drones")
    ax.plot(snrs, acc_noise,  marker="s", markersize=5, linewidth=2, color="#D62828", linestyle="--", label="Especificidad Ruido")
    
    ax.axhline(0.9, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(0, color="#555", linestyle="--", linewidth=1, alpha=0.5)
    
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Tasa de Acierto", fontsize=12)
    ax.set_title("V4 CFAR-Guided -- Accuracy por Nivel de SNR", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(min(snrs) - 1, max(snrs) + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "accuracy_per_snr.png", **FIGSAVE); plt.close(fig)

def plot_pr_curve(df, out_dir):
    y_true = df["label"].values
    y_prob = df["prob_drone"].values
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    auc_pr = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, linewidth=2.5, color="#4ECDC4", label=f"V4 (AP={ap:.4f} | AUC-PR={auc_pr:.4f})")
    ax.axhline(y_true.mean(), color="#888", linestyle="--", linewidth=1.5, label=f"Baseline ({y_true.mean():.2f})")
    
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("V4 CFAR-Guided -- Curva Precision-Recall", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "pr_curve.png", **FIGSAVE); plt.close(fig)
    return ap, auc_pr

def plot_confusion_matrix(df, out_dir):
    cm = confusion_matrix(df["label"].values, df["pred"].values)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ruido (0)", "Dron (1)"],
                yticklabels=["Ruido (0)", "Dron (1)"], ax=ax)
    ax.set_title("V4 CFAR-Guided -- Matriz de Confusion", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "confusion_matrix.png", **FIGSAVE); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    
    model = DualStreamCVCNN_V4().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    df = pd.read_csv(GOLDEN_CSV)
    results = []
    
    print(f"Evaluando Golden Set ({len(df)} archivos) guiado por CFAR...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fpath = os.path.join(DATA_DIR, row['filename'])
        try:
            d = torch.load(fpath, map_location='cpu', weights_only=False)
            iq = d['x_iq'].float()
            
            prob_max = sliding_window_inference_cfar(model, iq, device)
            
            results.append({
                'filename': row['filename'],
                'label': 0 if row['target'] == 4 else 1,
                'target_multiclass': row['target'],
                'snr': row['snr'],
                'prob_drone': prob_max,
                'pred': 1 if prob_max >= 0.5 else 0,
            })
        except Exception as e: 
            print(f"Error procesando {row['filename']}: {e}")
            continue
        
    res_df = pd.DataFrame(results)
    res_df['correct'] = (res_df['pred'] == res_df['label']).astype(int)
    
    print(f"\nGenerando figuras avanzadas en: {OUT_DIR}")
    plot_heatmap_drones(res_df, OUT_DIR)
    plot_heatmap_noise(res_df, OUT_DIR)
    plot_recall_snr_lines(res_df, OUT_DIR)
    plot_accuracy_per_snr(res_df, OUT_DIR)
    plot_confusion_matrix(res_df, OUT_DIR)
    ap, auc_pr = plot_pr_curve(res_df, OUT_DIR)
    
    y_true = res_df['label'].values
    metrics = {
        "f1": float(f1_score(y_true, res_df['pred'])),
        "acc": float(accuracy_score(y_true, res_df['pred'])),
        "ap": float(ap),
        "auc_pr": float(auc_pr)
    }
    print(f"\nRESULTADO FINAL: F1={metrics['f1']:.4f} | Acc={metrics['acc']:.4f} | AUC-PR={metrics['auc_pr']:.4f}")
    with open(os.path.join(OUT_DIR, "metrics_cfar.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()