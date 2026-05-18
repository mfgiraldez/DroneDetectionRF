"""
alumn_v3b_eval.py — Evaluacion completa del modelo Dual-Stream V3b
===================================================================
"""
import sys, os, json, argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, r"c:\repos\DroneDetectionRF")
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, auc, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

from model import DualStreamCVCNN
from dataset_dual import DualDataset

# ── Configuración por defecto ──────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
CKPT_DEFAULT = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")
CSV_DEFAULT  = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v3_dual_pro\modelo_alumn_v3a\dataset_v6_pointers.csv"
DATA_DIR     = r"C:\TFM_data\NoisyUAV\drone_RF_data"
OUT_DEFAULT  = os.path.join(BASE_DIR, "plots_test")

TARGET_NAMES = {
    0: "DJI (T0)", 1: "FutabaT14 (T1)", 2: "FutabaT7 (T2)",
    3: "Graupner (T3)", 5: "Taranis (T5)", 6: "Turnigy (T6)",
    4: "Ruido (T4)",
}
TARGET_COLORS = {
    0: "#0077B6", 1: "#D62828", 2: "#118AB2",
    3: "#2D6A4F",  5: "#E07C00", 6: "#7B2D8B",
}
FIGSAVE = dict(dpi=150, bbox_inches="tight")


def load_model(ckpt_path: str, device: torch.device):
    print(f"  Cargando checkpoint: {Path(ckpt_path).name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DualStreamCVCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    epoch   = ckpt.get("epoch", "?")
    val_f1  = ckpt.get("val_f1", float("nan"))
    val_acc = ckpt.get("val_acc", float("nan"))
    print(f"  Epoch: {epoch} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
    return model

@torch.no_grad()
def run_inference(model, csv_path, data_dir, split, batch_size, device):
    ds = DualDataset(csv_path, data_dir, split=split)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    df_split = pd.read_csv(csv_path)
    df_split = df_split[df_split["split"] == split].reset_index(drop=True)

    all_probs, all_preds, all_labels = [], [], []

    for iq_b, phys_b, lbl_b, _ in tqdm(dl, desc=f"Inferencia ({split})"):
        iq_b, phys_b = iq_b.to(device), phys_b.to(device)
        with torch.amp.autocast('cuda'):
            logits, _ = model(iq_b, phys_b)
        prob = torch.sigmoid(logits).squeeze(1).cpu()
        all_probs.extend(prob.tolist())
        all_preds.extend((prob >= 0.5).long().tolist())
        all_labels.extend(lbl_b.tolist())

    result = pd.DataFrame({
        "label":             all_labels,
        "prob_drone":        all_probs,
        "predicted":         all_preds,
        "snr":               df_split["snr"].values[:len(all_labels)],
        "target_multiclass": df_split["target_multiclass"].values[:len(all_labels)],
    })
    result["correct"] = (result["label"] == result["predicted"]).astype(int)
    return result

def plot_heatmap_drones(df, out_dir):
    drones = df[df["label"] == 1].copy()
    drones["Emisor RF"] = drones["target_multiclass"].map(lambda x: TARGET_NAMES.get(x, f"Target {x}"))
    hm = drones.pivot_table(index="Emisor RF", columns="snr", values="correct", aggfunc="mean").sort_index()
    hm = hm[sorted(hm.columns)]
    fig, ax = plt.subplots(figsize=(max(14, len(hm.columns)*0.55), max(3, len(hm)*0.85)))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Dual-Stream V3b — Detección por Emisor RF y SNR", fontweight="bold", pad=12)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "heatmap_target_snr.png", **FIGSAVE); plt.close(fig)

def plot_heatmap_noise(df, out_dir):
    noise = df[df["label"] == 0].copy()
    noise["Emisor RF"] = noise["target_multiclass"].map(lambda x: TARGET_NAMES.get(x, f"Ruido T{x}"))
    hm = noise.pivot_table(index="Emisor RF", columns="snr", values="correct", aggfunc="mean").sort_index()
    hm = hm[sorted(hm.columns)]
    fig, ax = plt.subplots(figsize=(max(14, len(hm.columns)*0.55), max(2, len(hm)*0.9)))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Dual-Stream V3b — Especificidad por Emisor Ruido y SNR", fontweight="bold", pad=12)
    fig.tight_layout(); fig.savefig(Path(out_dir) / "heatmap_noise_snr.png", **FIGSAVE); plt.close(fig)

def plot_accuracy_per_snr(df, out_dir):
    snrs = sorted(df["snr"].unique())
    acc_global = [df[df["snr"] == s]["correct"].mean() for s in snrs]
    acc_drones = [df[(df["snr"] == s) & (df["label"] == 1)]["correct"].mean() for s in snrs]
    acc_noise  = [df[(df["snr"] == s) & (df["label"] == 0)]["correct"].mean() for s in snrs]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(snrs, acc_global, marker="D", color="#1A1A1A", label="Accuracy Global")
    ax.plot(snrs, acc_drones, marker="o", color="#0077B6", label="Recall Drones")
    ax.plot(snrs, acc_noise,  marker="s", color="#D62828", linestyle="--", label="Especificidad Ruido")
    ax.axhline(0.9, color="#888", linestyle=":"); ax.axvline(0, color="#555", linestyle="--")
    ax.set_title("Dual-Stream V3b — Accuracy por SNR", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.2); fig.tight_layout()
    fig.savefig(Path(out_dir) / "accuracy_per_snr.png", **FIGSAVE); plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=CKPT_DEFAULT)
    parser.add_argument("--csv", type=str, default=CSV_DEFAULT)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--out", type=str, default=OUT_DEFAULT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model = load_model(args.ckpt, device)
    df = run_inference(model, args.csv, args.data_dir, args.split, args.batch_size, device)
    
    plot_heatmap_drones(df, args.out)
    plot_heatmap_noise(df, args.out)
    plot_accuracy_per_snr(df, args.out)

    metrics = {
        "accuracy": float(accuracy_score(df["label"], df["predicted"])),
        "f1": float(f1_score(df["label"], df["predicted"])),
        "n_test": len(df)
    }
    with open(Path(args.out) / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("DONE V3b Eval.")

if __name__ == "__main__":
    main()
