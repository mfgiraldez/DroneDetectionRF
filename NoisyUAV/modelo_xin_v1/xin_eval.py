"""
xin_eval.py - Evaluacion completa del modelo Xin CV-CNN 2D sobre el test set
=============================================================================
Genera las siguientes figuras (estilo consistente con HybridCVCNN):

    1. heatmap_target_snr.png     -- Recall por Emisor RF (drone) x SNR
    2. heatmap_noise_snr.png      -- Especificidad por Emisor de Ruido x SNR
    3. recall_snr_lines.png       -- Recall vs SNR, una linea por target de drone
    4. accuracy_per_snr.png       -- Accuracy global vs SNR (drones + ruido)
    5. pr_curve.png               -- Curva Precision-Recall con AUC

Uso:
    conda activate IAIAVv3
    python -m NoisyUAV.modelo_xin_v1.xin_eval ^
        --ckpt   C:\\repos\\DroneDetectionRF\\NoisyUAV\\modelo_xin_v1\\resultados\\checkpoints\\xin_model_best.pt ^
        --csv    C:\\repos\\DroneDetectionRF\\NoisyUAV\\modelo_xin_v1\\resultados\\xin_splits.csv ^
        --cache_dir C:\\TFM_data\\NoisyUAV\\xin_cache_256x256 ^
        --out    C:\\repos\\DroneDetectionRF\\NoisyUAV\\modelo_xin_v1\\resultados\\figures_test
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
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

from NoisyUAV.modelo_xin_v1.xin_cvcnn import XinCVCNN
from NoisyUAV.modelo_xin_v1.xin_dataset import XinSpectrogramDataset

# ---------------------------------------------------------------------------- #
# Configuracion por defecto                                                    #
# ---------------------------------------------------------------------------- #

CKPT_DEFAULT     = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_xin_v1\resultados\checkpoints\xin_model_best.pt"
CSV_DEFAULT      = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_xin_v1\resultados\xin_splits.csv"
CACHE_DEFAULT    = r"C:\TFM_data\NoisyUAV\xin_cache_256x256"
OUT_DEFAULT      = r"C:\repos\DroneDetectionRF\NoisyUAV\modelo_xin_v1\resultados\figures_test"

TARGET_NAMES = {
    0: "Target 0",
    1: "Target 1",
    2: "Target 2",
    3: "Target 3",
    5: "Target 5",
    6: "Target 6",
    4: "Ruido (T4)",
}
DRONE_TARGETS = [0, 1, 2, 3, 5, 6]
NOISE_TARGETS = [4]

# Paleta de colores saturados y oscuros — visibles sobre fondo blanco
TARGET_COLORS = {
    0: "#0077B6",   # azul oceano
    1: "#D62828",   # rojo intenso
    2: "#118AB2",   # azul acero
    3: "#2D6A4F",   # verde bosque
    5: "#E07C00",   # naranja quemado
    6: "#7B2D8B",   # purpura oscuro
}

FIGSAVE = dict(dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------- #
# Carga del modelo                                                             #
# ---------------------------------------------------------------------------- #

def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    print(f"  Cargando checkpoint: {Path(ckpt_path).name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = XinCVCNN(num_classes=1, kernel_size=5).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch       = ckpt.get("epoch", "?")
    best_val_f1 = ckpt.get("best_val_f1", float("nan"))
    print(f"  Epoch guardado: {epoch} | Mejor Val F1: {best_val_f1:.4f}")
    return model


# ---------------------------------------------------------------------------- #
# Inferencia completa con metadatos                                            #
# ---------------------------------------------------------------------------- #

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    csv_path: str,
    cache_dir: str,
    split: str,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """
    Corre inferencia sobre el split indicado y retorna un DataFrame con:
        filepath, filename, target, snr, is_drone, prob_drone, predicted, correct
    """
    # Dataset en modo cache para maxima velocidad
    ds = XinSpectrogramDataset(
        csv_path  = csv_path,
        split     = split,
        cache_dir = cache_dir,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    all_probs = []
    all_preds = []

    for x, _ in dl:
        x = x.to(device)
        logit = model(x)                    # [B, 1]
        prob  = logit.sigmoid().squeeze(1)  # [B]
        pred  = (prob >= 0.5).long()
        all_probs.extend(prob.cpu().tolist())
        all_preds.extend(pred.cpu().tolist())

    df = ds.data.copy()
    df["prob_drone"] = all_probs
    df["predicted"]  = all_preds
    df["correct"]    = (df["is_drone"] == df["predicted"]).astype(int)
    # Alias para compatibilidad con funciones de plot
    df["label"]               = df["is_drone"]
    df["target_multiclass"]   = df["target"]
    return df


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
    )
    hm = hm.sort_index()
    hm = hm[sorted(hm.columns)]

    n_cols = len(hm.columns)
    n_rows = len(hm)
    fig_w  = max(14, n_cols * 0.55)
    fig_h  = max(3,  n_rows * 0.85)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        hm,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
        linewidths=0.4,
        linecolor="#1a1a2e",
        ax=ax,
        cbar_kws={"label": "Tasa de Acierto (Recall)"},
    )
    ax.set_title(
        "Xin CV-CNN 2D -- Deteccion por Emisor RF y SNR (Test Set)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("Emisor RF", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    path = Path(out_dir) / "heatmap_target_snr.png"
    fig.savefig(path, **FIGSAVE)
    plt.close(fig)
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
    )
    hm = hm.sort_index()
    hm = hm[sorted(hm.columns)]

    n_cols = len(hm.columns)
    n_rows = len(hm)
    fig_w  = max(14, n_cols * 0.55)
    fig_h  = max(2,  n_rows * 0.9)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        hm,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0, vmax=1.0,
        linewidths=0.4,
        linecolor="#1a1a2e",
        ax=ax,
        cbar_kws={"label": "Especificidad (1 - FPR)"},
    )
    ax.set_title(
        "Xin CV-CNN 2D -- Especificidad por Emisor de Ruido y SNR (Test Set)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("Emisor RF", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    path = Path(out_dir) / "heatmap_noise_snr.png"
    fig.savefig(path, **FIGSAVE)
    plt.close(fig)
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
        ax.plot(
            snrs, recall,
            marker="o", markersize=5,
            linewidth=2, color=color,
            label=TARGET_NAMES.get(target, f"Target {target}"),
        )

    # Linea global de drones — negro con borde para maxima visibilidad
    global_recall = [drones[drones["snr"] == s]["correct"].mean() for s in snrs]
    ax.plot(
        snrs, global_recall,
        marker="D", markersize=6, linewidth=2.5,
        color="#1A1A1A", linestyle="--",
        label="Media Global (Drones)",
        zorder=5,
    )

    ax.axhline(0.9, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(0.5, color="#888", linestyle=":", linewidth=1, alpha=0.6)
    ax.axvline(0,   color="#555", linestyle="--", linewidth=1, alpha=0.5)

    # Sombreado de grupos SNR
    snr_min, snr_max = min(snrs), max(snrs)
    ax.axvspan(max(-6, snr_min),  min(-1, snr_max), alpha=0.07, color="red",    label="Grupo C (<-6 dB)")
    ax.axvspan(max(-6, snr_min),  min( 9, snr_max), alpha=0.05, color="yellow", label="Grupo B (-6..9 dB)")
    ax.axvspan(max(10, snr_min),  snr_max,           alpha=0.07, color="green",  label="Grupo A (>=10 dB)")

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Recall (Tasa de Deteccion)", fontsize=12)
    ax.set_title(
        "Xin CV-CNN 2D -- Recall vs SNR por Emisor RF (Test Set)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(snr_min - 1, snr_max + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()

    path = Path(out_dir) / "recall_snr_lines.png"
    fig.savefig(path, **FIGSAVE)
    plt.close(fig)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------------------- #
# 4. Accuracy por SNR                                                          #
# ---------------------------------------------------------------------------- #

def plot_accuracy_per_snr(df: pd.DataFrame, out_dir: str) -> None:
    snrs = sorted(df["snr"].unique())

    acc_global = [df[df["snr"] == s]["correct"].mean()              for s in snrs]
    acc_drones = [df[(df["snr"] == s) & (df["label"] == 1)]["correct"].mean() for s in snrs]
    acc_noise  = [df[(df["snr"] == s) & (df["label"] == 0)]["correct"].mean() for s in snrs]

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

    # Etiquetas de grupos SNR
    y_ann = 0.03
    ax.text(-14, y_ann, "Grupo C", fontsize=9, color="#aaa", ha="center")
    ax.text(2,   y_ann, "Grupo B", fontsize=9, color="#aaa", ha="center")
    ax.text(16,  y_ann, "Grupo A", fontsize=9, color="#aaa", ha="center")

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Tasa de Acierto", fontsize=12)
    ax.set_title(
        "Xin CV-CNN 2D -- Accuracy por Nivel de SNR (Test Set)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(snrs) - 1, max(snrs) + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()

    path = Path(out_dir) / "accuracy_per_snr.png"
    fig.savefig(path, **FIGSAVE)
    plt.close(fig)
    print(f"  Guardado: {path}")


# ---------------------------------------------------------------------------- #
# 5. Curva Precision-Recall                                                    #
# ---------------------------------------------------------------------------- #

def plot_pr_curve(df: pd.DataFrame, out_dir: str) -> None:
    y_true = df["label"].values
    y_prob = df["prob_drone"].values

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    auc_pr = auc(recall, precision)

    # Punto de operacion actual (umbral 0.5)
    th_idx = np.searchsorted(thresholds, 0.5)
    th_idx = min(th_idx, len(precision) - 2)  # evitar out of bounds
    op_recall    = recall[th_idx]
    op_precision = precision[th_idx]

    # Baseline (clasificador aleatorio)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(recall, precision, linewidth=2.5, color="#4ECDC4",
            label=f"Xin CV-CNN 2D (AP = {ap:.4f} | AUC-PR = {auc_pr:.4f})")
    ax.axhline(baseline, color="#888", linestyle="--", linewidth=1.5,
               label=f"Baseline aleatorio ({baseline:.2f})")

    # Punto de operacion (umbral=0.5)
    ax.scatter([op_recall], [op_precision], color="#FF6B6B", s=120, zorder=5,
               label=f"Umbral=0.5  P={op_precision:.3f} R={op_recall:.3f}")
    ax.annotate(
        f"  P={op_precision:.2f}\n  R={op_recall:.2f}",
        xy=(op_recall, op_precision),
        fontsize=9, color="#FF6B6B",
        xytext=(op_recall + 0.02, op_precision - 0.07),
    )

    # Isocurvas de F1
    f1_vals = [0.5, 0.6, 0.7, 0.8, 0.9]
    for f1_target in f1_vals:
        r_range = np.linspace(0.01, 1.0, 200)
        p_range = f1_target * r_range / (2 * r_range - f1_target + 1e-9)
        valid   = (p_range >= 0) & (p_range <= 1)
        ax.plot(r_range[valid], p_range[valid], color="#555", linewidth=0.6,
                linestyle=":", alpha=0.6)
        ax.text(r_range[valid][-1] - 0.01, p_range[valid][-1],
                f"F1={f1_target:.1f}", fontsize=7, color="#888", ha="right")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Xin CV-CNN 2D -- Curva Precision-Recall (Test Set)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2, color="gray")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()

    path = Path(out_dir) / "pr_curve.png"
    fig.savefig(path, **FIGSAVE)
    plt.close(fig)
    print(f"  Guardado: {path}")

    return ap, auc_pr


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
        print(
            f"  Target {target}: Global={g_acc:.3f} | "
            f"A(>=10)={a_acc:.3f} | B(-6..10)={b_acc:.3f} | C(<-6)={c_acc:.3f} "
            f"| n={len(sub)}"
        )

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
        print(
            f"  {name}: Spec Global={spec:.3f} (FP={1-spec:.3f}) | "
            f"A(>=10)={a_spec:.3f} | C(<-6)={c_spec:.3f} | n={len(sub)}"
        )

    # Metricas globales
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

    # Guardar JSON
    path = Path(out_dir) / "test_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(global_metrics, f, indent=2)
    print(f"\n  Metricas guardadas en: {path}")


# ---------------------------------------------------------------------------- #
# Entry point                                                                  #
# ---------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluacion completa del modelo Xin CV-CNN 2D sobre el test set."
    )
    parser.add_argument("--ckpt",       type=str, default=CKPT_DEFAULT,  help="Ruta al checkpoint BEST.")
    parser.add_argument("--csv",        type=str, default=CSV_DEFAULT,   help="Ruta al CSV de splits.")
    parser.add_argument("--cache_dir",  type=str, default=CACHE_DEFAULT, help="Directorio de cache STFT+Sobel.")
    parser.add_argument("--out",        type=str, default=OUT_DEFAULT,   help="Directorio de salida de figuras.")
    parser.add_argument("--split",      type=str, default="test",        choices=["test", "val"], help="Split a evaluar.")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Xin CV-CNN 2D -- Evaluacion sobre Test Set")
    print(f"  Device : {device}")
    print(f"  Split  : {args.split}")
    print(f"  Output : {args.out}")
    print("=" * 65)

    # 1. Modelo
    model = load_model(args.ckpt, device)

    # 2. Inferencia
    print(f"\n  Ejecutando inferencia ({args.split})...")
    df = run_inference(model, args.csv, args.cache_dir, args.split, args.batch_size, device)
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

    print("\n" + "=" * 65)
    print(f"  AP (Average Precision): {ap:.4f}")
    print(f"  AUC-PR:                 {auc_pr:.4f}")
    print(f"  DONE -- Figuras en: {args.out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
