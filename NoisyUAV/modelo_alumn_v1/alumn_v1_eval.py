"""
alumn_v1_eval.py - Evaluacion completa del modelo Alumno V1 (BurstCVCNN) sobre el test set
===========================================================================================
Genera las siguientes figuras (estilo consistente con xin_eval.py):

    1. heatmap_target_snr.png     -- Recall por Emisor RF (drone) x SNR
    2. heatmap_noise_snr.png      -- Especificidad por Emisor de Ruido x SNR
    3. recall_snr_lines.png       -- Recall vs SNR, una linea por target de drone
    4. accuracy_per_snr.png       -- Accuracy global vs SNR (drones + ruido)
    5. pr_curve.png               -- Curva Precision-Recall con AUC

Uso:
    conda activate IAIAVv3
    python -m NoisyUAV.modelo_alumn_v1.alumn_v1_eval ^
        --ckpt  C:\\repos\\DroneDetectionRF\\NoisyUAV\\modelo_alumn_v1\\checkpoints\\alumn_model_best.pt ^
        --csv   C:\\repos\\DroneDetectionRF\\NoisyUAV\\modelo_alumn_v1\\alumn_dataset_pseudo_v3.csv ^
        --out   C:\\repos\\DroneDetectionRF\\NoisyUAV\\modelo_alumn_v1\\plots
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, confusion_matrix

from NoisyUAV.modelos.burst_cvcnn import BurstCVCNN, BurstCVCNNDataset, pad_seq_collate

# ---------------------------------------------------------------------------- #
# Configuracion por defecto                                                    #
# ---------------------------------------------------------------------------- #

CKPT_DEFAULT = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v1\checkpoints\alumn_model_best.pt"
CSV_DEFAULT  = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v1\alumn_dataset_pseudo_v3.csv"
DATA_DIR     = r"C:\TFM_data\NoisyUAV\drone_RF_data"
OUT_DEFAULT  = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_alumn_v1\figures_test"

TARGET_NAMES = {
    0: "Target 0", 1: "Target 1", 2: "Target 2",
    3: "Target 3", 5: "Target 5", 6: "Target 6",
    4: "Ruido (T4)",
}
DRONE_TARGETS = [0, 1, 2, 3, 5, 6]
NOISE_TARGETS = [4]

TARGET_COLORS = {
    0: "#0077B6", 1: "#D62828", 2: "#118AB2",
    3: "#2D6A4F", 5: "#E07C00", 6: "#7B2D8B",
}

FIGSAVE = dict(dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------- #
# Carga del modelo                                                             #
# ---------------------------------------------------------------------------- #

def load_model(ckpt_path: str, device: torch.device):
    print(f"  Cargando checkpoint: {Path(ckpt_path).name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = BurstCVCNN(dropout_cnn=0.3, dropout_fuse=0.4).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch   = ckpt.get("epoch", "?")
    val_f1  = ckpt.get("val_f1", float("nan"))
    print(f"  Epoch guardado: {epoch} | Val F1: {val_f1:.4f}")

    phys_mean = torch.from_numpy(ckpt["phys_mean"])
    phys_std  = torch.from_numpy(ckpt["phys_std"])
    return model, phys_mean, phys_std


# ---------------------------------------------------------------------------- #
# Inferencia completa con metadatos                                            #
# ---------------------------------------------------------------------------- #

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    csv_path: str,
    data_dir: str,
    split: str,
    batch_size: int,
    device: torch.device,
    phys_mean: torch.Tensor,
    phys_std: torch.Tensor,
) -> pd.DataFrame:
    """
    Corre inferencia sobre el split indicado y retorna un DataFrame con:
        prob_drone, predicted, correct, label, target_multiclass, snr
    """
    df_full = pd.read_csv(csv_path)
    df_full["label"] = df_full["pseudo_label"].astype(float)
    df_split = df_full[df_full["split"] == split].copy()

    ds = BurstCVCNNDataset(df_split, data_dir, augment=False,
                           phys_mean=phys_mean, phys_std=phys_std)

    # Collate personalizado que conserva metadatos
    def collate_with_meta(batch):
        iqs    = torch.stack([b["iq"] for b in batch])
        feats  = torch.stack([b["feats"] for b in batch])
        labels = torch.stack([b["label"] for b in batch])
        snrs   = [b["snr"] for b in batch]
        tmc    = [b["target_mc"] for b in batch]
        return iqs, feats, labels, snrs, tmc

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=0, collate_fn=collate_with_meta)

    all_probs, all_preds, all_labels = [], [], []
    all_snrs, all_tmc = [], []

    for iqs, feats, labels, snrs, tmc in dl:
        iqs   = iqs.to(device)
        feats = feats.to(device)
        logit = model(iqs, feats)
        prob  = logit.sigmoid().squeeze(1)
        pred  = (prob >= 0.5).long()
        all_probs.extend(prob.cpu().tolist())
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(labels.squeeze(1).tolist())
        all_snrs.extend(snrs)
        all_tmc.extend(tmc)

    result = pd.DataFrame({
        "label":             all_labels,
        "prob_drone":        all_probs,
        "predicted":         all_preds,
        "snr":               all_snrs,
        "target_multiclass": all_tmc,
    })
    result["correct"] = (result["label"] == result["predicted"]).astype(int)
    return result


# ---------------------------------------------------------------------------- #
# 1. Heatmap: Recall por Drone x SNR                                          #
# ---------------------------------------------------------------------------- #

def plot_heatmap_drones(df: pd.DataFrame, out_dir: str) -> None:
    drones = df[df["label"] == 1].copy()
    drones["Emisor RF"] = drones["target_multiclass"].map(
        lambda x: TARGET_NAMES.get(x, f"Target {x}")
    )
    hm = drones.pivot_table(
        index="Emisor RF", columns="snr", values="correct", aggfunc="mean"
    ).sort_index()
    hm = hm[sorted(hm.columns)]

    fig_w = max(14, len(hm.columns) * 0.55)
    fig_h = max(3,  len(hm) * 0.85)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e", ax=ax,
                cbar_kws={"label": "Tasa de Acierto (Recall)"})
    ax.set_title("Alumno V1 (BurstCVCNN) -- Deteccion por Emisor RF y SNR (Test Set)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("Emisor RF", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    path = Path(out_dir) / "heatmap_target_snr.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------------------- #
# 2. Heatmap: Especificidad por Ruido x SNR                                   #
# ---------------------------------------------------------------------------- #

def plot_heatmap_noise(df: pd.DataFrame, out_dir: str) -> None:
    noise = df[df["label"] == 0].copy()
    noise["Emisor RF"] = noise["target_multiclass"].map(
        lambda x: TARGET_NAMES.get(x, f"Ruido T{x}")
    )
    hm = noise.pivot_table(
        index="Emisor RF", columns="snr", values="correct", aggfunc="mean"
    ).sort_index()
    hm = hm[sorted(hm.columns)]

    fig_w = max(14, len(hm.columns) * 0.55)
    fig_h = max(2,  len(hm) * 0.9)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.4, linecolor="#1a1a2e", ax=ax,
                cbar_kws={"label": "Especificidad (1 - FPR)"})
    ax.set_title("Alumno V1 (BurstCVCNN) -- Especificidad por Emisor de Ruido y SNR (Test Set)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("Emisor RF", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    path = Path(out_dir) / "heatmap_noise_snr.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------------------- #
# 3. Recall vs SNR (lineas por target)                                        #
# ---------------------------------------------------------------------------- #

def plot_recall_snr_lines(df: pd.DataFrame, out_dir: str) -> None:
    drones = df[df["label"] == 1].copy()
    snrs   = sorted(drones["snr"].unique())

    fig, ax = plt.subplots(figsize=(11, 6))

    for target in sorted(drones["target_multiclass"].unique()):
        sub    = drones[drones["target_multiclass"] == target]
        recall = [sub[sub["snr"] == s]["correct"].mean() for s in snrs]
        color  = TARGET_COLORS.get(target, None)
        ax.plot(snrs, recall, marker="o", markersize=5, linewidth=2, color=color,
                label=TARGET_NAMES.get(target, f"Target {target}"))

    global_recall = [drones[drones["snr"] == s]["correct"].mean() for s in snrs]
    ax.plot(snrs, global_recall, marker="D", markersize=6, linewidth=2.5,
            color="#1A1A1A", linestyle="--", label="Media Global (Drones)", zorder=5)

    ax.axhline(0.9, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(0,   color="#555", linestyle="--", linewidth=1, alpha=0.5)

    snr_min, snr_max = min(snrs), max(snrs)
    ax.axvspan(max(-6, snr_min), min(-1, snr_max), alpha=0.07, color="red",    label="Grupo C (<-6 dB)")
    ax.axvspan(max(-6, snr_min), min( 9, snr_max), alpha=0.05, color="yellow", label="Grupo B (-6..9 dB)")
    ax.axvspan(max(10, snr_min), snr_max,          alpha=0.07, color="green",  label="Grupo A (>=10 dB)")

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Recall (Tasa de Deteccion)", fontsize=12)
    ax.set_title("Alumno V1 (BurstCVCNN) -- Recall vs SNR por Emisor RF (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(snr_min - 1, snr_max + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    path = Path(out_dir) / "recall_snr_lines.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------------------- #
# 4. Accuracy por SNR                                                          #
# ---------------------------------------------------------------------------- #

def plot_accuracy_per_snr(df: pd.DataFrame, out_dir: str) -> None:
    snrs = sorted(df["snr"].unique())
    acc_global = [df[df["snr"] == s]["correct"].mean()                          for s in snrs]
    acc_drones = [df[(df["snr"] == s) & (df["label"] == 1)]["correct"].mean()   for s in snrs]
    acc_noise  = [df[(df["snr"] == s) & (df["label"] == 0)]["correct"].mean()   for s in snrs]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(snrs, acc_global, marker="D", markersize=6, linewidth=2.5,
            color="#1A1A1A", linestyle="-",  label="Accuracy Global")
    ax.plot(snrs, acc_drones, marker="o", markersize=5, linewidth=2,
            color="#0077B6", linestyle="-",  label="Accuracy Drones (Recall)")
    ax.plot(snrs, acc_noise,  marker="s", markersize=5, linewidth=2,
            color="#D62828", linestyle="--", label="Especificidad Ruido")

    ax.axhline(0.9, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(0,   color="#555", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(-6,  color="#555", linestyle=":",  linewidth=1, alpha=0.4)
    ax.axvline(10,  color="#555", linestyle=":",  linewidth=1, alpha=0.4)

    y_ann = 0.03
    ax.text(-14, y_ann, "Grupo C", fontsize=9, color="#aaa", ha="center")
    ax.text(2,   y_ann, "Grupo B", fontsize=9, color="#aaa", ha="center")
    ax.text(16,  y_ann, "Grupo A", fontsize=9, color="#aaa", ha="center")

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Tasa de Acierto", fontsize=12)
    ax.set_title("Alumno V1 (BurstCVCNN) -- Accuracy por Nivel de SNR (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(snrs) - 1, max(snrs) + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    path = Path(out_dir) / "accuracy_per_snr.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------------------- #
# 5. Curva Precision-Recall                                                    #
# ---------------------------------------------------------------------------- #

def plot_pr_curve(df: pd.DataFrame, out_dir: str):
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
            label=f"Alumno V1 (AP = {ap:.4f} | AUC-PR = {auc_pr:.4f})")
    ax.axhline(baseline, color="#888", linestyle="--", linewidth=1.5,
               label=f"Baseline aleatorio ({baseline:.2f})")
    ax.scatter([op_recall], [op_precision], color="#FF6B6B", s=120, zorder=5,
               label=f"Umbral=0.5  P={op_precision:.3f} R={op_recall:.3f}")
    ax.annotate(f"  P={op_precision:.2f}\n  R={op_recall:.2f}",
                xy=(op_recall, op_precision), fontsize=9, color="#FF6B6B",
                xytext=(op_recall + 0.02, op_precision - 0.07))

    for f1_target in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r_range = np.linspace(0.01, 1.0, 200)
        p_range = f1_target * r_range / (2 * r_range - f1_target + 1e-9)
        valid   = (p_range >= 0) & (p_range <= 1)
        ax.plot(r_range[valid], p_range[valid], color="#555", linewidth=0.6,
                linestyle=":", alpha=0.6)
        ax.text(r_range[valid][-1] - 0.01, p_range[valid][-1],
                f"F1={f1_target:.1f}", fontsize=7, color="#888", ha="right")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Alumno V1 (BurstCVCNN) -- Curva Precision-Recall (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    path = Path(out_dir) / "pr_curve.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")
    return ap, auc_pr


# ---------------------------------------------------------------------------- #
# 6. Matriz de Confusion                                                       #
# ---------------------------------------------------------------------------- #

def plot_confusion_matrix(df: pd.DataFrame, out_dir: str) -> None:
    y_true = df["label"].values
    y_pred = df["predicted"].values
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ruido (0)", "Dron (1)"],
                yticklabels=["Ruido (0)", "Dron (1)"], ax=ax)
    ax.set_title("Matriz de Confusion (Test Set)", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    fig.tight_layout()
    path = Path(out_dir) / "confusion_matrix.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------------------- #
# 7. Accuracy por SNR (Grafico de Barras)                                      #
# ---------------------------------------------------------------------------- #

def plot_accuracy_snr_bars(df: pd.DataFrame, out_dir: str) -> None:
    snrs = sorted(df["snr"].unique())
    acc_global = [df[df["snr"] == s]["correct"].mean() * 100 for s in snrs]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(snrs, acc_global, color="#4A90E2", edgecolor="black", alpha=0.8)
    
    # Colorear barras segun grupo
    for bar, s in zip(bars, snrs):
        if s < -6:
            bar.set_color("#E74C3C") # Rojo para Grupo C
        elif s < 10:
            bar.set_color("#F1C40F") # Amarillo para Grupo B
        else:
            bar.set_color("#2ECC71") # Verde para Grupo A
        bar.set_edgecolor("black")
    
    for bar, acc in zip(bars, acc_global):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.axhline(90, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Alumno V1 (BurstCVCNN) -- Accuracy por SNR (Barras)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_xticks(snrs)
    ax.grid(axis="y", alpha=0.3, color="gray")
    fig.tight_layout()
    path = Path(out_dir) / "accuracy_snr_bars.png"
    fig.savefig(path, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------------------- #
# 8. Curvas de Entrenamiento (Loss & F1) desde el Checkpoint                   #
# ---------------------------------------------------------------------------- #

def plot_training_history(ckpt_path: str, out_dir: str) -> None:
    # Buscar history.json en la carpeta padre de out_dir o en la raiz del modelo
    base_dir = os.path.dirname(os.path.dirname(ckpt_path))
    history_path = os.path.join(base_dir, "history.json")
    
    if not os.path.exists(history_path):
        print(f"  [!] No se encontro el archivo de historial: {history_path}")
        return
        
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
        
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Curvas de Perdida (Loss) -- Alumno V1")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = Path(out_dir) / "train_loss_curve.png"
    fig.savefig(path1, **FIGSAVE); plt.close(fig)
    print(f"  Guardado: {path1}")

    if "val_f1" in history: # Nota: history.json de v1 puede no tener train_f1, pero tiene val_f1
        fig, ax = plt.subplots(figsize=(8, 5))
        if "train_f1" in history:
            ax.plot(epochs, history["train_f1"], label="Train F1", marker="o", markersize=3)
        ax.plot(epochs, history["val_f1"], label="Val F1", marker="s", markersize=3)
        ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Score")
        ax.set_title("Evolucion del F1-Score -- Alumno V1")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path2 = Path(out_dir) / "train_f1_curve.png"
        fig.savefig(path2, **FIGSAVE); plt.close(fig)
        print(f"  Guardado: {path2}")


# ---------------------------------------------------------------------------- #
# Resumen textual y guardado JSON                                              #
# ---------------------------------------------------------------------------- #

def print_summary_and_save(df: pd.DataFrame, out_dir: str) -> None:
    print("\n" + "=" * 65)
    print("  RESUMEN POR EMISOR RF (DRONES, label=1)")
    print("=" * 65)

    drones = df[df["label"] == 1]
    for target in sorted(drones["target_multiclass"].unique()):
        sub   = drones[drones["target_multiclass"] == target]
        g_acc = sub["correct"].mean()
        a_acc = sub[sub["snr"] >= 10]["correct"].mean()
        b_sub = sub[(sub["snr"] >= -6) & (sub["snr"] < 10)]
        b_acc = b_sub["correct"].mean() if len(b_sub) > 0 else float("nan")
        c_sub = sub[sub["snr"] < -6]
        c_acc = c_sub["correct"].mean() if len(c_sub) > 0 else float("nan")
        print(f"  Target {target}: Global={g_acc:.3f} | "
              f"A(>=10)={a_acc:.3f} | B(-6..10)={b_acc:.3f} | C(<-6)={c_acc:.3f} "
              f"| n={len(sub)}")

    print("\n" + "=" * 65)
    print("  ESPECIFICIDAD POR TIPO DE RUIDO (label=0)")
    print("=" * 65)
    noise = df[df["label"] == 0]
    for target in sorted(noise["target_multiclass"].unique()):
        sub  = noise[noise["target_multiclass"] == target]
        name = TARGET_NAMES.get(target, f"Ruido T{target}")
        spec = sub["correct"].mean()
        a_spec = sub[sub["snr"] >= 10]["correct"].mean()
        c_sub  = sub[sub["snr"] < -6]
        c_spec = c_sub["correct"].mean() if len(c_sub) > 0 else float("nan")
        print(f"  {name}: Spec Global={spec:.3f} (FP={1-spec:.3f}) | "
              f"A(>=10)={a_spec:.3f} | C(<-6)={c_spec:.3f} | n={len(sub)}")

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    y_true = df["label"].values
    y_pred = df["predicted"].values

    global_metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "n_test":    int(len(df)),
        "n_drones":  int((df["label"] == 1).sum()),
        "n_noise":   int((df["label"] == 0).sum()),
    }

    print("\n" + "=" * 65)
    print("  METRICAS GLOBALES (Test Set)")
    print("=" * 65)
    for k, v in global_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<14}: {v:.4f}")
        else:
            print(f"  {k:<14}: {v}")
    print("=" * 65)

    path = Path(out_dir) / "test_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(global_metrics, f, indent=2)
    print(f"\n  Metricas guardadas en: {path}")


# ---------------------------------------------------------------------------- #
# Entry point                                                                  #
# ---------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluacion completa del modelo Alumno V1 (BurstCVCNN) sobre el test set."
    )
    parser.add_argument("--ckpt",       type=str, default=CKPT_DEFAULT, help="Ruta al checkpoint BEST.")
    parser.add_argument("--csv",        type=str, default=CSV_DEFAULT,  help="Ruta al CSV pseudo-labels.")
    parser.add_argument("--data_dir",   type=str, default=DATA_DIR,     help="Directorio de datos IQ .pt")
    parser.add_argument("--out",        type=str, default=OUT_DEFAULT,  help="Directorio de salida de figuras.")
    parser.add_argument("--split",      type=str, default="test",       choices=["test", "val"], help="Split a evaluar.")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Alumno V1 (BurstCVCNN) -- Evaluacion sobre Test Set")
    print(f"  Device : {device}")
    print(f"  Split  : {args.split}")
    print(f"  Output : {args.out}")
    print("=" * 65)

    # 1. Modelo
    model, phys_mean, phys_std = load_model(args.ckpt, device)

    # 2. Inferencia
    print(f"\n  Ejecutando inferencia ({args.split})...")
    df = run_inference(model, args.csv, args.data_dir, args.split,
                       args.batch_size, device, phys_mean, phys_std)
    acc_global = df["correct"].mean()
    print(f"  Accuracy global ({args.split}): {acc_global:.4f} ({acc_global*100:.2f}%)")

    # 3. Resumen textual + JSON
    print_summary_and_save(df, args.out)

    # 4. Figuras
    print(f"\n  Generando figuras en: {args.out}")
    plot_heatmap_drones(df, args.out)
    plot_heatmap_noise(df, args.out)
    plot_recall_snr_lines(df, args.out)
    plot_accuracy_per_snr(df, args.out)
    ap, auc_pr = plot_pr_curve(df, args.out)
    plot_confusion_matrix(df, args.out)
    plot_accuracy_snr_bars(df, args.out)
    plot_training_history(args.ckpt, args.out)

    print("\n" + "=" * 65)
    print(f"  AP (Average Precision): {ap:.4f}")
    print(f"  AUC-PR:                 {auc_pr:.4f}")
    print(f"  DONE -- Figuras en: {args.out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
