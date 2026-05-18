"""
alumn_v2_dual_eval.py — Evaluacion completa del modelo Dual-Stream V2 sobre el test set
========================================================================================
Genera las siguientes figuras:

    1. heatmap_target_snr.png     -- Recall por Emisor RF (drone) x SNR
    2. heatmap_noise_snr.png      -- Especificidad por Emisor de Ruido x SNR
    3. recall_snr_lines.png       -- Recall vs SNR, una linea por target de drone
    4. accuracy_per_snr.png       -- Accuracy global vs SNR
    5. pr_curve.png               -- Curva Precision-Recall con AUC

Evaluacion a nivel de INSTANCIA (1 ventana CFAR = 1 muestra).
La etiqueta viene del ground truth del fichero (determinista).

Uso:
    conda activate IAIAVv3
    cd C:\\repos\\DroneDetectionRF\\NoisyUAV\\modelo_alumn_v2_dual
    python alumn_v2_dual_eval.py
"""

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
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
from sklearn.metrics import (precision_recall_curve, auc, average_precision_score,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)

from NoisyUAV.modelo_alumn_v2_dual.model import DualStreamCVCNN
from NoisyUAV.modelo_alumn_v2_dual.dataset_dual import DualDataset

# ---------------------------------------------------------------------------- #
# Configuracion por defecto                                                     #
# ---------------------------------------------------------------------------- #
CKPT_DEFAULT = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v2_dual\checkpoints\best_model.pth"
CSV_DEFAULT  = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v2_dual\dataset_v5_pointers.csv"
DATA_DIR     = r"C:\TFM_data\NoisyUAV\drone_RF_data"
OUT_DEFAULT  = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v2_dual\figures_test"

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


# ---------------------------------------------------------------------------- #
# Carga del modelo                                                               #
# ---------------------------------------------------------------------------- #

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


# ---------------------------------------------------------------------------- #
# Inferencia completa con metadatos                                              #
# ---------------------------------------------------------------------------- #

@torch.no_grad()
def run_inference(model, csv_path, data_dir, split, batch_size, device) -> pd.DataFrame:
    ds = DualDataset(csv_path, data_dir, split=split)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Leer metadatos del CSV para SNR y target_multiclass
    df_meta = pd.read_csv(csv_path)
    df_split = df_meta[df_meta["split"] == split].reset_index(drop=True)

    all_probs, all_preds, all_labels = [], [], []

    for iq_batch, phys_batch, labels_batch in tqdm(dl, desc=f"Inferencia ({split})"):
        iq_batch = iq_batch.to(device)
        phys_batch = phys_batch.to(device)

        with torch.amp.autocast('cuda'):
            logits, _ = model(iq_batch, phys_batch)

        prob = torch.sigmoid(logits).squeeze(1).cpu()
        pred = (prob >= 0.5).long()

        all_probs.extend(prob.tolist())
        all_preds.extend(pred.tolist())
        all_labels.extend(labels_batch.tolist())

    result = pd.DataFrame({
        "label":             all_labels,
        "prob_drone":        all_probs,
        "predicted":         all_preds,
        "snr":               df_split["snr"].values[:len(all_labels)],
        "target_multiclass": df_split["target_multiclass"].values[:len(all_labels)],
    })
    result["correct"] = (result["label"] == result["predicted"]).astype(int)
    return result


# ---------------------------------------------------------------------------- #
# Figuras                                                                        #
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
    ax.set_title("Dual-Stream V2 -- Deteccion por Emisor RF y SNR (Test Set)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("SNR (dB)", fontsize=11); ax.set_ylabel("Emisor RF", fontsize=11)
    ax.tick_params(axis="x", rotation=45); ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    path = Path(out_dir) / "heatmap_target_snr.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


def plot_heatmap_noise(df, out_dir):
    noise = df[df["label"] == 0].copy()
    noise["Emisor RF"] = noise["target_multiclass"].map(lambda x: TARGET_NAMES.get(x, f"Ruido T{x}"))
    hm = noise.pivot_table(index="Emisor RF", columns="snr", values="correct", aggfunc="mean").sort_index()
    hm = hm[sorted(hm.columns)]
    fig, ax = plt.subplots(figsize=(max(14, len(hm.columns)*0.55), max(2, len(hm)*0.9)))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e", ax=ax,
                cbar_kws={"label": "Especificidad (1 - FPR)"})
    ax.set_title("Dual-Stream V2 -- Especificidad por Emisor de Ruido y SNR (Test Set)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("SNR (dB)", fontsize=11); ax.set_ylabel("Emisor RF", fontsize=11)
    ax.tick_params(axis="x", rotation=45); ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    path = Path(out_dir) / "heatmap_noise_snr.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


def plot_recall_snr_lines(df, out_dir):
    drones = df[df["label"] == 1].copy()
    snrs   = sorted(drones["snr"].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    for target in sorted(drones["target_multiclass"].unique()):
        sub    = drones[drones["target_multiclass"] == target]
        recall = [sub[sub["snr"] == s]["correct"].mean() for s in snrs]
        ax.plot(snrs, recall, marker="o", markersize=5, linewidth=2,
                color=TARGET_COLORS.get(target),
                label=TARGET_NAMES.get(target, f"Target {target}"))
    global_recall = [drones[drones["snr"] == s]["correct"].mean() for s in snrs]
    ax.plot(snrs, global_recall, marker="D", markersize=6, linewidth=2.5,
            color="#1A1A1A", linestyle="--", label="Media Global (Drones)", zorder=5)
    ax.axhline(0.9, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(0,   color="#555", linestyle="--", linewidth=1, alpha=0.5)
    snr_min, snr_max = min(snrs), max(snrs)
    ax.axvspan(max(-6, snr_min), min(-1, snr_max), alpha=0.07, color="red",    label="Grupo C")
    ax.axvspan(max(-6, snr_min), min( 9, snr_max), alpha=0.05, color="yellow", label="Grupo B")
    ax.axvspan(max(10, snr_min), snr_max,           alpha=0.07, color="green",  label="Grupo A")
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Dual-Stream V2 -- Recall vs SNR por Emisor RF (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(snr_min - 1, snr_max + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    path = Path(out_dir) / "recall_snr_lines.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


def plot_accuracy_per_snr(df, out_dir):
    snrs = sorted(df["snr"].unique())
    acc_global = [df[df["snr"] == s]["correct"].mean()                         for s in snrs]
    acc_drones = [df[(df["snr"] == s) & (df["label"] == 1)]["correct"].mean()  for s in snrs]
    acc_noise  = [df[(df["snr"] == s) & (df["label"] == 0)]["correct"].mean()  for s in snrs]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(snrs, acc_global, marker="D", markersize=6, linewidth=2.5, color="#1A1A1A", label="Accuracy Global")
    ax.plot(snrs, acc_drones, marker="o", markersize=5, linewidth=2,   color="#0077B6", label="Recall Drones")
    ax.plot(snrs, acc_noise,  marker="s", markersize=5, linewidth=2,   color="#D62828", linestyle="--", label="Especificidad Ruido")
    ax.axhline(0.9, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(0, color="#555", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(-6, color="#555", linestyle=":", linewidth=1, alpha=0.4)
    ax.axvline(10, color="#555", linestyle=":", linewidth=1, alpha=0.4)
    y_ann = 0.03
    ax.text(-14, y_ann, "Grupo C", fontsize=9, color="#aaa", ha="center")
    ax.text(2,   y_ann, "Grupo B", fontsize=9, color="#aaa", ha="center")
    ax.text(16,  y_ann, "Grupo A", fontsize=9, color="#aaa", ha="center")
    ax.set_xlabel("SNR (dB)", fontsize=12); ax.set_ylabel("Tasa de Acierto", fontsize=12)
    ax.set_title("Dual-Stream V2 -- Accuracy por Nivel de SNR (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(min(snrs) - 1, max(snrs) + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    path = Path(out_dir) / "accuracy_per_snr.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


def plot_pr_curve(df, out_dir):
    y_true = df["label"].values
    y_prob = df["prob_drone"].values
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap     = average_precision_score(y_true, y_prob)
    auc_pr = auc(recall, precision)
    th_idx       = min(np.searchsorted(thresholds, 0.5), len(precision) - 2)
    op_recall    = recall[th_idx]
    op_precision = precision[th_idx]
    baseline     = y_true.mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, linewidth=2.5, color="#4ECDC4",
            label=f"Dual-Stream V2 (AP={ap:.4f} | AUC-PR={auc_pr:.4f})")
    ax.axhline(baseline, color="#888", linestyle="--", linewidth=1.5,
               label=f"Baseline ({baseline:.2f})")
    ax.scatter([op_recall], [op_precision], color="#FF6B6B", s=120, zorder=5,
               label=f"Umbral=0.5  P={op_precision:.3f} R={op_recall:.3f}")
    ax.annotate(f"  P={op_precision:.2f}\n  R={op_recall:.2f}",
                xy=(op_recall, op_precision), fontsize=9, color="#FF6B6B",
                xytext=(op_recall + 0.02, op_precision - 0.07))
    for f1_target in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r_r = np.linspace(0.01, 1.0, 200)
        p_r = f1_target * r_r / (2 * r_r - f1_target + 1e-9)
        v   = (p_r >= 0) & (p_r <= 1)
        ax.plot(r_r[v], p_r[v], color="#555", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.text(r_r[v][-1] - 0.01, p_r[v][-1], f"F1={f1_target:.1f}", fontsize=7, color="#888", ha="right")
    ax.set_xlabel("Recall", fontsize=12); ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Dual-Stream V2 -- Curva Precision-Recall (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    path = Path(out_dir) / "pr_curve.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")
    return ap, auc_pr


def plot_confusion_matrix(df, out_dir):
    y_true = df["label"].values
    y_pred = df["predicted"].values
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ruido (0)", "Dron (1)"],
                yticklabels=["Ruido (0)", "Dron (1)"], ax=ax)
    ax.set_title("Dual-Stream V2 -- Matriz de Confusion (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    fig.tight_layout()
    path = Path(out_dir) / "confusion_matrix.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


def plot_accuracy_snr_bars(df, out_dir):
    snrs = sorted(df["snr"].unique())
    acc_global = [df[df["snr"] == s]["correct"].mean() * 100 for s in snrs]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(snrs, acc_global, color="#4A90E2", edgecolor="black", alpha=0.8)
    for bar, s in zip(bars, snrs):
        if s < -6:
            bar.set_color("#E74C3C")
        elif s < 10:
            bar.set_color("#F1C40F")
        else:
            bar.set_color("#2ECC71")
        bar.set_edgecolor("black")
    for bar, acc in zip(bars, acc_global):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.axhline(90, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Dual-Stream V2 -- Accuracy por SNR (Barras)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_xticks(snrs)
    ax.grid(axis="y", alpha=0.3, color="gray")
    fig.tight_layout()
    path = Path(out_dir) / "accuracy_snr_bars.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


def plot_training_history(out_dir):
    history_path = os.path.join(os.path.dirname(out_dir), "metrics_history.json")
    if not os.path.exists(history_path):
        print(f"  [!] No se encontro: {history_path}")
        return
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3, color="blue")
    ax1.plot(epochs, val_loss, label="Val Loss", marker="s", markersize=3, color="red")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Dual-Stream V2 -- Curvas de Perdida (Loss)", fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1, label="Val F1", marker="o", markersize=3, color="green")
    ax2.plot(epochs, val_acc, label="Val Accuracy", marker="s", markersize=3, color="purple")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.set_title("Dual-Stream V2 -- Evolucion de Metricas", fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(out_dir) / "training_curves.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------------------- #
# Resumen + JSON                                                                 #
# ---------------------------------------------------------------------------- #

def print_summary_and_save(df, out_dir):
    print("\n" + "=" * 65)
    print("  RESUMEN POR EMISOR RF (DRONES, label=1)")
    print("=" * 65)
    drones = df[df["label"] == 1]
    for target in sorted(drones["target_multiclass"].unique()):
        sub   = drones[drones["target_multiclass"] == target]
        g_acc = sub["correct"].mean()
        a_sub = sub[sub["snr"] >= 10]
        a_acc = a_sub["correct"].mean() if len(a_sub) > 0 else float("nan")
        b_sub = sub[(sub["snr"] >= -6) & (sub["snr"] < 10)]
        b_acc = b_sub["correct"].mean() if len(b_sub) > 0 else float("nan")
        c_sub = sub[sub["snr"] < -6]
        c_acc = c_sub["correct"].mean() if len(c_sub) > 0 else float("nan")
        name = TARGET_NAMES.get(target, f"Target {target}")
        print(f"  {name}: Global={g_acc:.3f} | A(>=10)={a_acc:.3f} | "
              f"B(-6..10)={b_acc:.3f} | C(<-6)={c_acc:.3f} | n={len(sub)}")

    print("\n" + "=" * 65)
    print("  ESPECIFICIDAD POR TIPO DE RUIDO (label=0)")
    print("=" * 65)
    noise = df[df["label"] == 0]
    for target in sorted(noise["target_multiclass"].unique()):
        sub  = noise[noise["target_multiclass"] == target]
        name = TARGET_NAMES.get(target, f"Ruido T{target}")
        spec = sub["correct"].mean()
        a_sub = sub[sub["snr"] >= 10]
        a_spec = a_sub["correct"].mean() if len(a_sub) > 0 else float("nan")
        c_sub  = sub[sub["snr"] < -6]
        c_spec = c_sub["correct"].mean() if len(c_sub) > 0 else float("nan")
        print(f"  {name}: Spec={spec:.3f} (FP={1-spec:.3f}) | "
              f"A(>=10)={a_spec:.3f} | C(<-6)={c_spec:.3f} | n={len(sub)}")

    y_true = df["label"].values; y_pred = df["predicted"].values; y_prob = df["prob_drone"].values
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_val = float("nan")

    global_metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "auc_roc":   float(auc_val),
        "n_test":    int(len(df)),
        "n_drones":  int((df["label"] == 1).sum()),
        "n_noise":   int((df["label"] == 0).sum()),
    }
    print("\n" + "=" * 65 + "\n  METRICAS GLOBALES\n" + "=" * 65)
    for k, v in global_metrics.items():
        print(f"  {k:<14}: {v:.4f}" if isinstance(v, float) else f"  {k:<14}: {v}")
    print("=" * 65)
    path = Path(out_dir) / "test_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(global_metrics, f, indent=2)
    print(f"\n  Metricas guardadas en: {path}")


# ---------------------------------------------------------------------------- #
# Entry point                                                                   #
# ---------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluacion completa del modelo Dual-Stream V2 sobre el test set.")
    parser.add_argument("--ckpt",       type=str, default=CKPT_DEFAULT)
    parser.add_argument("--csv",        type=str, default=CSV_DEFAULT)
    parser.add_argument("--data_dir",   type=str, default=DATA_DIR)
    parser.add_argument("--out",        type=str, default=OUT_DEFAULT)
    parser.add_argument("--split",      type=str, default="test", choices=["test", "val"])
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Dual-Stream V2 -- Evaluacion sobre Test Set")
    print(f"  Device : {device}")
    print(f"  Split  : {args.split}")
    print(f"  Output : {args.out}")
    print("=" * 65)

    model = load_model(args.ckpt, device)

    print(f"\n  Ejecutando inferencia ({args.split})...")
    df = run_inference(model, args.csv, args.data_dir, args.split,
                       args.batch_size, device)
    print(f"  Muestras evaluadas: {len(df)}")
    print(f"  Accuracy global: {df['correct'].mean():.4f}")

    print_summary_and_save(df, args.out)

    print(f"\n  Generando figuras en: {args.out}")
    plot_heatmap_drones(df, args.out)
    plot_heatmap_noise(df, args.out)
    plot_recall_snr_lines(df, args.out)
    plot_accuracy_per_snr(df, args.out)
    ap, auc_pr = plot_pr_curve(df, args.out)
    plot_confusion_matrix(df, args.out)
    plot_accuracy_snr_bars(df, args.out)
    plot_training_history(args.out)

    print("\n" + "=" * 65)
    print(f"  AP (Average Precision): {ap:.4f}")
    print(f"  AUC-PR:                 {auc_pr:.4f}")
    print(f"  DONE -- Figuras en: {args.out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
