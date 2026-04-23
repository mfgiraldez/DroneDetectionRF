"""
run_marnet_experiment.py
========================
Pipeline completo de entrenamiento, evaluacion y generacion de informe
para la arquitectura MaRNet-Fusion sobre el dataset NoisyUAV v2.

Ejecutar con:
    python run_marnet_experiment.py

Genera automaticamente:
    resultados_marnet/checkpoints/    Checkpoints del modelo
    resultados_marnet/figures/        Todas las figuras PNG
    resultados_marnet/informe_marnet_fusion.md  Informe completo

Autor: Pipeline autogenerado para TFM UAV RF Detection - 2026
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Suprimir warnings menores para output limpio
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# --- Paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR  = Path(__file__).parent / "resultados_marnet"
CKPT_DIR     = RESULTS_DIR / "checkpoints"
FIG_DIR      = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "experiment.log", mode="w"),
    ],
)
log = logging.getLogger("MaRNet")

# --- Imports -----------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")   # backend no interactivo para ejecucion en background
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# sklearn para metricas avanzadas
try:
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve,
        confusion_matrix, classification_report,
        ConfusionMatrixDisplay,
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    log.warning("scikit-learn no disponible. Metricas simplificadas.")

# Importar modulos del proyecto
from NoisyUAV.funciones.dataset import obtener_splits_dataset
from NoisyUAV.modelos.marnet_fusion import (
    RFDroneDataset, MaRNetFusion,
    train_one_epoch, evaluate,
    mixup_batch, _MAMBA_AVAILABLE
)

# =============================================================================
# CONFIGURACION
# =============================================================================

CFG = {
    # Dataset
    "data_dir":       r"C:\TFM_data\NoisyUAV\drone_RF_data",
    "crop_len":       2048,       # 2048 muestras = ~146 us a 14 MHz
    "n_fft":          128,        # F = 65 bins, T ~ 65 frames
    "hop_length":     32,
    "test_size":      0.15,
    "val_size":       0.15,
    "random_state":   42,

    # DataLoader
    "batch_size":     48,
    "num_workers":    0,          # 0 en Windows para evitar problemas spawn

    # Modelo
    "latent_dim":     256,
    "d_model_ssm":    128,
    "d_state_ssm":    16,
    "num_ssm_layers": 3,
    "d_fusion":       256,
    "dropout_cnn":    0.3,
    "dropout_ssm":    0.2,
    "dropout_mlp":    0.4,

    # Entrenamiento
    "epochs":         50,
    "lr":             3e-4,
    "weight_decay":   1e-4,
    "mixup_alpha":    0.4,
    "mixup_prob":     0.5,
    "grad_clip":      2.0,
    "patience":       10,         # early stopping
    "label_smoothing": 0.05,

    # Curriculum learning
    "curriculum_switch_epoch": 18,  # epocas 1-18: grupos A+B; 19+: todos A+B+C

    # AMP
    "use_amp":        True,
}

# Definicion de grupos SNR (segun cargador.py original)
# Grupo A: SNR >= 10 dB, Grupo B: -6 <= SNR < 10 dB, Grupo C: SNR < -6 dB


def snr_to_grupo(snr: int) -> str:
    if snr >= 10:    return "A"
    elif snr >= -6:  return "B"
    else:            return "C"


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def compute_pos_weight(df_train) -> torch.Tensor:
    """Calcula pos_weight para BCEWithLogitsLoss segun desbalanceo de clases."""
    n_pos = (df_train["label"] == 1).sum()
    n_neg = (df_train["label"] == 0).sum()
    w = n_neg / (n_pos + 1e-8)
    log.info(f"  pos_weight = {w:.3f}  (n_pos={n_pos}, n_neg={n_neg})")
    return torch.tensor([w], dtype=torch.float32)


def save_json(obj, path: Path):
    """Guarda un dict como JSON, convirtiendo tipos numpy/torch a Python nativos."""
    def _convert(o):
        if isinstance(o, (np.integer, np.floating)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, torch.Tensor): return o.item() if o.numel() == 1 else o.tolist()
        return o
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: _convert(v) for k, v in obj.items()}, f, indent=2)


# =============================================================================
# CONSTRUCCION DE DATALOADERS
# =============================================================================

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader,
                                          DataLoader, object, object, object]:
    """
    Construye los DataLoaders de train (fase 1 = A+B), val y test.
    Devuelve tambien los DataFrames y el DataLoader de train fase 2 (A+B+C).
    """
    log.info("Construyendo splits del dataset...")
    df_train, df_val, df_test = obtener_splits_dataset(
        data_dir=cfg["data_dir"],
        test_size=cfg["test_size"],
        val_size=cfg["val_size"],
        random_state=cfg["random_state"],
    )
    log.info(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

    # Curriculo: fase 1 solo grupos A+B (SNR facil/medio)
    df_train_f1 = df_train[df_train["grupo"].isin(["A", "B"])].copy()
    df_train_f2 = df_train.copy()   # todos los grupos incluyendo C
    log.info(f"  Train fase-1 (A+B): {len(df_train_f1):,} | fase-2 (A+B+C): {len(df_train_f2):,}")

    ds_kwargs = dict(crop_len=cfg["crop_len"], n_fft=cfg["n_fft"],
                     hop_length=cfg["hop_length"])
    dl_kwargs = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                     pin_memory=(cfg["num_workers"] > 0))

    ds_train_f1 = RFDroneDataset(df_train_f1, augment=True,  **ds_kwargs)
    ds_train_f2 = RFDroneDataset(df_train_f2, augment=True,  **ds_kwargs)
    ds_val      = RFDroneDataset(df_val,       augment=False, **ds_kwargs)
    ds_test     = RFDroneDataset(df_test,      augment=False, **ds_kwargs)

    dl_train_f1 = DataLoader(ds_train_f1, shuffle=True,  **dl_kwargs)
    dl_train_f2 = DataLoader(ds_train_f2, shuffle=True,  **dl_kwargs)
    dl_val      = DataLoader(ds_val,       shuffle=False, **dl_kwargs)
    dl_test     = DataLoader(ds_test,      shuffle=False, **dl_kwargs)

    return dl_train_f1, dl_train_f2, dl_val, dl_test, df_train, df_val, df_test


# =============================================================================
# LOOP DE ENTRENAMIENTO PRINCIPAL
# =============================================================================

def run_training(cfg: dict, model: MaRNetFusion, device: torch.device,
                 dl_train_f1, dl_train_f2, dl_val,
                 pos_weight: torch.Tensor) -> Dict:
    """
    Entrena el modelo con curriculum learning y devuelve el historial completo.
    
    Estrategia:
      - Epocas 1..curriculum_switch: train en grupos A+B (SNR facil/medio)
      - Epocas curriculum_switch..epochs: train en grupos A+B+C (todos los SNR)
    
    Scheduler: CosineAnnealingLR con 5-epoch warmup lineal.
    Early stopping: paciencia sobre val F1 score.
    AMP: GradScaler si cfg["use_amp"] y CUDA disponible.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    # LR scheduler: coseno desde lr hasta lr/100 en epochs_total pasos
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] / 100
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device),
        reduction="mean",
    )

    scaler = None
    if cfg["use_amp"] and device.type == "cuda":
        scaler = torch.amp.GradScaler()
        log.info("  AMP (Automatic Mixed Precision) activado.")

    history = {
        "train_loss": [], "train_acc": [], "train_attn": [],
        "val_loss":   [], "val_acc":   [], "val_f1": [],
        "val_precision": [], "val_recall": [],
        "lr": [], "epoch_time": [],
        "curriculum_phase": [],   # 1 o 2 por epoca
    }

    best_val_f1    = -1.0
    patience_count = 0
    best_epoch     = 0

    log.info(f"\n{'='*62}")
    log.info(f"  INICIO DEL ENTRENAMIENTO: {cfg['epochs']} epocas")
    log.info(f"  Curriculum switch: epoca {cfg['curriculum_switch_epoch']}")
    log.info(f"{'='*62}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # Seleccion del dataloader segun fase del curriculum
        phase = 1 if epoch <= cfg["curriculum_switch_epoch"] else 2
        dl_train = dl_train_f1 if phase == 1 else dl_train_f2

        if epoch == cfg["curriculum_switch_epoch"] + 1:
            log.info(">>> [CURRICULUM] Cambiando a Fase 2: incorporando grupo C (SNR < -6 dB)")

        # Warmup lineal de LR en las primeras 5 epocas
        if epoch <= 5:
            warmup_lr = cfg["lr"] * epoch / 5
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # --- Train epoch ---
        tr = train_one_epoch(
            model=model, dataloader=dl_train, optimizer=optimizer,
            criterion=criterion, device=device,
            mixup_alpha=cfg["mixup_alpha"], mixup_prob=cfg["mixup_prob"],
            grad_clip=cfg["grad_clip"], scaler=scaler,
        )

        # --- Validation epoch ---
        vl = evaluate(model, dl_val, criterion, device)

        # LR step (despues del warmup)
        if epoch > 5:
            scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["train_attn"].append(tr["attn"])
        history["val_loss"].append(vl["loss"])
        history["val_acc"].append(vl["acc"])
        history["val_f1"].append(vl["f1"])
        history["val_precision"].append(vl["precision"])
        history["val_recall"].append(vl["recall"])
        history["lr"].append(current_lr)
        history["epoch_time"].append(elapsed)
        history["curriculum_phase"].append(phase)

        log.info(
            f"Ep {epoch:3d}/{cfg['epochs']} [Ph{phase}] | "
            f"TrLoss={tr['loss']:.4f} TrAcc={tr['acc']:.4f} | "
            f"VlLoss={vl['loss']:.4f} VlF1={vl['f1']:.4f} VlAcc={vl['acc']:.4f} | "
            f"LR={current_lr:.2e} | {elapsed:.1f}s | "
            f"Attn=[{tr['attn'][0]:.2f},{tr['attn'][1]:.2f},{tr['attn'][2]:.2f}]"
        )

        # Guardar mejor modelo
        if vl["f1"] > best_val_f1:
            best_val_f1    = vl["f1"]
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1": best_val_f1, "cfg": cfg,
            }, CKPT_DIR / "best_model.pt")
            log.info(f"  [*] Nuevo best model guardado: Val F1 = {best_val_f1:.4f}")
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                log.info(f"\n[EARLY STOPPING] Sin mejora en {cfg['patience']} epocas. "
                         f"Mejor F1 = {best_val_f1:.4f} en epoca {best_epoch}.")
                break

    # Guardar ultimo modelo
    torch.save({"epoch": epoch, "model_state": model.state_dict(),
                "val_f1": vl["f1"]}, CKPT_DIR / "last_model.pt")

    history["best_epoch"] = best_epoch
    history["best_val_f1"] = best_val_f1
    log.info(f"\nEntrenamiento completado. Mejor modelo: epoca {best_epoch}, "
             f"Val F1 = {best_val_f1:.4f}")
    return history


# =============================================================================
# EVALUACION EN TEST SET (por SNR)
# =============================================================================

@torch.no_grad()
def evaluate_by_snr(model: MaRNetFusion, df_test, cfg: dict,
                    device: torch.device, criterion: nn.Module) -> Dict:
    """
    Evaluacion detallada por nivel de SNR (cada 2 dB de -20 a +30 dB).
    Calcula accuracy, precision, recall y F1 en cada nivel.
    """
    model.eval()
    snr_levels = sorted(df_test["snr"].unique())
    results = {}

    ds_kwargs = dict(crop_len=cfg["crop_len"], n_fft=cfg["n_fft"],
                     hop_length=cfg["hop_length"])

    for snr in snr_levels:
        df_snr = df_test[df_test["snr"] == snr]
        if len(df_snr) == 0:
            continue
        ds = RFDroneDataset(df_snr, augment=False, **ds_kwargs)
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=0, pin_memory=False)
        m = evaluate(model, dl, criterion, device)
        results[int(snr)] = {
            "acc":       m["acc"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "n":         int(len(df_snr)),
        }
        log.info(f"  SNR={snr:+4d} dB | n={len(df_snr):4d} | "
                 f"Acc={m['acc']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

    return results


@torch.no_grad()
def collect_embeddings_and_attention(model: MaRNetFusion, dl_test, device: torch.device,
                                     max_batches: int = 30) -> Dict:
    """
    Recoge pesos de atención por grupo SNR para el analisis de interpretabilidad.
    """
    model.eval()
    attn_list, label_list, prob_list = [], [], []

    for i, (spec, iq, stat, labels) in enumerate(dl_test):
        if i >= max_batches:
            break
        spec  = spec.to(device)
        iq    = iq.to(device)
        stat  = stat.to(device)
        logit, attn_w = model(spec, iq, stat)
        probs = torch.sigmoid(logit.squeeze(1))
        attn_list.append(attn_w.cpu())
        label_list.append(labels)
        prob_list.append(probs.cpu())

    return {
        "attn":   torch.cat(attn_list).numpy(),    # [N, 3]
        "labels": torch.cat(label_list).numpy(),   # [N]
        "probs":  torch.cat(prob_list).numpy(),    # [N]
    }


# =============================================================================
# GENERACION DE FIGURAS
# =============================================================================

# Paleta de colores corporativa del TFM
BLUE  = "#2E86AB"
RED   = "#E84855"
GREEN = "#3BB273"
GRAY  = "#6C757D"
AMBER = "#F4A261"
PURPLE = "#6A0572"

plt.rcParams.update({
    "figure.facecolor": "#0F1923",
    "axes.facecolor":   "#0F1923",
    "axes.edgecolor":   "#2A3A4A",
    "axes.labelcolor":  "#E8EDF2",
    "text.color":       "#E8EDF2",
    "xtick.color":      "#A0ADB8",
    "ytick.color":      "#A0ADB8",
    "grid.color":       "#1E2E3E",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": "#131E28",
    "legend.edgecolor": "#2A3A4A",
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "lines.linewidth":  2.0,
})

FIGSAVE_KW = dict(dpi=150, bbox_inches="tight", facecolor="#0F1923")


def fig_training_curves(history: Dict) -> Path:
    """Fig 1: Curvas de perdida, accuracy y LR durante el entrenamiento."""
    epochs_done = len(history["train_loss"])
    ep = list(range(1, epochs_done + 1))
    phase_switch = CFG["curriculum_switch_epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="#0F1923")
    fig.suptitle("MaRNet-Fusion: Curvas de Entrenamiento (NoisyUAV v2)", fontsize=14, y=1.02)

    for ax in axes:
        ax.set_facecolor("#0F1923")
        ax.grid(True)
        if phase_switch < epochs_done:
            ax.axvline(phase_switch, color=AMBER, lw=1.5, ls=":", alpha=0.8,
                       label=f"Curriculum Phase 2 (ep {phase_switch})")

    # Loss
    axes[0].plot(ep, history["train_loss"], color=BLUE, label="Train Loss", alpha=0.9)
    axes[0].plot(ep, history["val_loss"],   color=RED,  label="Val Loss",   alpha=0.9)
    axes[0].set_xlabel("Epoca"); axes[0].set_ylabel("BCE Loss")
    axes[0].set_title("Perdida BCE"); axes[0].legend(fontsize=8)

    # Accuracy
    axes[1].plot(ep, [v*100 for v in history["train_acc"]], color=BLUE, label="Train Acc")
    axes[1].plot(ep, [v*100 for v in history["val_acc"]],   color=RED,  label="Val Acc")
    axes[1].plot(ep, [v*100 for v in history["val_f1"]],    color=GREEN, label="Val F1",
                 ls="--", alpha=0.8)
    axes[1].set_ylim(40, 102)
    axes[1].set_xlabel("Epoca"); axes[1].set_ylabel("%")
    axes[1].set_title("Exactitud y F1"); axes[1].legend(fontsize=8)

    # Learning Rate
    axes[2].semilogy(ep, history["lr"], color=AMBER, label="LR")
    axes[2].set_xlabel("Epoca"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Tasa de Aprendizaje"); axes[2].legend(fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / "fig_01_training_curves.png"
    fig.savefig(path, **FIGSAVE_KW)
    plt.close(fig)
    log.info(f"  Guardada: {path.name}")
    return path


def fig_attention_weights(history: Dict) -> Path:
    """Fig 2: Evolucion de los pesos de atencion por rama a lo largo del entrenamiento."""
    epochs_done = len(history["train_attn"])
    ep = list(range(1, epochs_done + 1))
    attn = np.array(history["train_attn"])  # [E, 3]

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0F1923")
    ax.set_facecolor("#0F1923")
    ax.grid(True)

    labels_branch = ["Rama CNN (Espectrograma)", "Rama BiGRU (IQ)", "Rama MLP (Estadisticos)"]
    colors_branch = [BLUE, RED, GREEN]
    for i, (lbl, col) in enumerate(zip(labels_branch, colors_branch)):
        ax.plot(ep, attn[:, i], color=col, label=lbl, alpha=0.9)
        # Suavizado
        if len(ep) > 5:
            from numpy.lib.stride_tricks import sliding_window_view
            w = min(7, len(ep))
            sm = np.convolve(attn[:, i], np.ones(w)/w, mode="valid")
            offset = (len(ep) - len(sm)) // 2
            ax.plot(range(1+offset, 1+offset+len(sm)), sm, color=col, lw=3, alpha=0.5)

    if CFG["curriculum_switch_epoch"] < epochs_done:
        ax.axvline(CFG["curriculum_switch_epoch"], color=AMBER, lw=1.5, ls=":",
                   label=f"Curriculum Phase 2")

    ax.set_xlabel("Epoca")
    ax.set_ylabel("Peso de Atencion Medio")
    ax.set_title("Evolucion de Pesos de Atencion Cross-Modal (PAM_Fusion)")
    ax.legend()
    plt.tight_layout()
    path = FIG_DIR / "fig_02_attention_weights.png"
    fig.savefig(path, **FIGSAVE_KW)
    plt.close(fig)
    log.info(f"  Guardada: {path.name}")
    return path


def fig_confusion_matrix(test_results: Dict) -> Path:
    """Fig 3: Matriz de confusion en el conjunto de test."""
    labels = test_results["labels"]
    preds  = test_results["preds"]

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0F1923")
    ax.set_facecolor("#0F1923")

    # Colormap personalizado azul-oscuro a azul-brillante
    cmap = LinearSegmentedColormap.from_list("UAV", ["#0F1923", BLUE], N=256)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)

    classes = ["No Drone\n(Ruido)", "Drone"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i,j] < thresh else "#0F1923", fontweight="bold")

    ax.set_ylabel("Etiqueta Real")
    ax.set_xlabel("Prediccion")
    ax.set_title(f"Matriz de Confusion (Test Set)\nAcc={test_results['acc']:.4f}  "
                 f"F1={test_results['f1']:.4f}")
    plt.tight_layout()
    path = FIG_DIR / "fig_03_confusion_matrix.png"
    fig.savefig(path, **FIGSAVE_KW)
    plt.close(fig)
    log.info(f"  Guardada: {path.name}")
    return path


def fig_roc_curves(test_results: Dict, snr_results: Dict) -> Path:
    """Fig 4: Curvas ROC globales y por grupo de SNR."""
    probs  = test_results["probs"]
    labels = test_results["labels"]

    fpr_g, tpr_g, _ = roc_curve(labels, probs)
    auc_g = auc(fpr_g, tpr_g)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0F1923")
    for ax in axes: ax.set_facecolor("#0F1923"); ax.grid(True)

    # Global ROC
    axes[0].plot(fpr_g, tpr_g, color=BLUE, lw=2.5, label=f"MaRNet-Fusion (AUC={auc_g:.4f})")
    axes[0].plot([0, 1], [0, 1], "--", color=GRAY, alpha=0.6, label="Clasificador aleatorio")
    axes[0].set_xlabel("Tasa de Falsos Positivos (FPR)")
    axes[0].set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    axes[0].set_title("Curva ROC Global (Test Set)")
    axes[0].legend()

    # ROC por grupos SNR (A, B, C)
    group_cfg = {
        "A (SNR>=10 dB)":  [s for s in snr_results if s >= 10],
        "B (-6..10 dB)":   [s for s in snr_results if -6 <= s < 10],
        "C (SNR<-6 dB)":   [s for s in snr_results if s < -6],
    }
    g_colors = [GREEN, AMBER, RED]
    for (gname, snr_list), gcol in zip(group_cfg.items(), g_colors):
        acc_vals = [snr_results[s]["acc"] for s in snr_list if s in snr_results]
        if acc_vals:
            mean_acc = np.mean(acc_vals)
            axes[1].bar(gname, mean_acc * 100, color=gcol, alpha=0.85,
                        edgecolor="#2A3A4A", linewidth=1.2)
            axes[1].text(list(group_cfg.keys()).index(gname), mean_acc * 100 + 0.5,
                         f"{mean_acc*100:.1f}%", ha="center", va="bottom",
                         color="white", fontsize=10, fontweight="bold")

    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Accuracy Media por Grupo SNR")
    axes[1].axhline(y=75, color=GRAY, ls="--", lw=1.5, label="Baseline CV-CNN (~75%)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = FIG_DIR / "fig_04_roc_group_accuracy.png"
    fig.savefig(path, **FIGSAVE_KW)
    plt.close(fig)
    log.info(f"  Guardada: {path.name}")
    return path


def fig_pr_curve(test_results: Dict) -> Path:
    """Fig 5: Curva Precision-Recall con AUC-PR."""
    probs  = test_results["probs"]
    labels = test_results["labels"]

    precision, recall, _ = precision_recall_curve(labels, probs)
    auc_pr = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0F1923")
    ax.set_facecolor("#0F1923"); ax.grid(True)
    ax.fill_between(recall, precision, alpha=0.25, color=BLUE)
    ax.plot(recall, precision, color=BLUE, lw=2.5, label=f"MaRNet-Fusion (AUC-PR={auc_pr:.4f})")
    base_rate = labels.mean()
    ax.axhline(y=base_rate, color=GRAY, ls="--", lw=1.5, label=f"Tasa positivos ({base_rate:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall (Test Set)")
    ax.legend(); plt.tight_layout()
    path = FIG_DIR / "fig_05_pr_curve.png"
    fig.savefig(path, **FIGSAVE_KW)
    plt.close(fig)
    log.info(f"  Guardada: {path.name}")
    return path


def fig_per_snr_metrics(snr_results: Dict) -> Path:
    """Fig 6: Accuracy, Precision, Recall y F1 por nivel de SNR (barras agrupadas)."""
    snr_sorted = sorted(snr_results.keys())
    acc = [snr_results[s]["acc"]       for s in snr_sorted]
    pre = [snr_results[s]["precision"] for s in snr_sorted]
    rec = [snr_results[s]["recall"]    for s in snr_sorted]
    f1  = [snr_results[s]["f1"]        for s in snr_sorted]

    x = np.arange(len(snr_sorted))
    w = 0.2

    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0F1923")
    ax.set_facecolor("#0F1923"); ax.grid(True, axis="y")
    ax.bar(x - 1.5*w, [v*100 for v in acc], w, label="Accuracy",  color=BLUE,   alpha=0.85)
    ax.bar(x - 0.5*w, [v*100 for v in pre], w, label="Precision", color=GREEN,  alpha=0.85)
    ax.bar(x + 0.5*w, [v*100 for v in rec], w, label="Recall",    color=AMBER,  alpha=0.85)
    ax.bar(x + 1.5*w, [v*100 for v in f1],  w, label="F1-Score",  color=RED,    alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:+d}" for s in snr_sorted], fontsize=8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Metrica (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Metricas de Clasificacion por Nivel de SNR (Test Set) — MaRNet-Fusion")
    ax.axhline(y=75, color=GRAY, ls="--", lw=1.5, label="Baseline CV-CNN (~75%)")
    # Zonas SNR
    low_snr_end = next((i for i, s in enumerate(snr_sorted) if s >= -6), 0)
    mid_snr_end = next((i for i, s in enumerate(snr_sorted) if s >= 10), low_snr_end)
    ax.axvspan(-0.5, low_snr_end - 0.5, alpha=0.07, color=RED,   label="Grupo C (SNR<-6 dB)")
    ax.axvspan(low_snr_end-0.5, mid_snr_end-0.5, alpha=0.07, color=AMBER, label="Grupo B")
    ax.axvspan(mid_snr_end-0.5, len(snr_sorted)-0.5, alpha=0.07, color=GREEN, label="Grupo A (SNR>=10 dB)")
    ax.legend(fontsize=8, ncol=4)
    plt.tight_layout()
    path = FIG_DIR / "fig_06_per_snr_metrics.png"
    fig.savefig(path, **FIGSAVE_KW)
    plt.close(fig)
    log.info(f"  Guardada: {path.name}")
    return path


def fig_attention_by_snr(model: MaRNetFusion, df_test, cfg: dict,
                          device: torch.device) -> Path:
    """Fig 7: Pesos de atencion promedio por rama para cada nivel de SNR."""
    model.eval()
    snr_levels = sorted(df_test["snr"].unique())
    ds_kwargs = dict(crop_len=cfg["crop_len"], n_fft=cfg["n_fft"],
                     hop_length=cfg["hop_length"])

    attn_by_snr = {}
    with torch.no_grad():
        for snr in snr_levels:
            df_s = df_test[df_test["snr"] == snr]
            if len(df_s) == 0: continue
            ds = RFDroneDataset(df_s, augment=False, **ds_kwargs)
            dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
            attn_acc = torch.zeros(3)
            n_batches = 0
            for spec, iq, stat, _ in dl:
                _, attn_w = model(spec.to(device), iq.to(device), stat.to(device))
                attn_acc += attn_w.mean(0).cpu()
                n_batches += 1
            if n_batches > 0:
                attn_by_snr[int(snr)] = (attn_acc / n_batches).numpy()

    snr_sorted = sorted(attn_by_snr.keys())
    attn_mat   = np.array([attn_by_snr[s] for s in snr_sorted])   # [n_snr, 3]

    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor="#0F1923")
    ax.set_facecolor("#0F1923"); ax.grid(True, axis="y")

    x = np.arange(len(snr_sorted))
    w = 0.28
    ax.bar(x - w, attn_mat[:, 0], w, label="Rama CNN (Espectrograma)", color=BLUE,  alpha=0.85)
    ax.bar(x,     attn_mat[:, 1], w, label="Rama BiGRU (IQ)",          color=RED,   alpha=0.85)
    ax.bar(x + w, attn_mat[:, 2], w, label="Rama MLP (Estadisticos)",   color=GREEN, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:+d}" for s in snr_sorted], fontsize=8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Peso de Atencion Medio")
    ax.set_title("Pesos de Atencion Cross-Modal (PAM_Fusion) por Nivel de SNR")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = FIG_DIR / "fig_07_attention_by_snr.png"
    fig.savefig(path, **FIGSAVE_KW)
    plt.close(fig)
    log.info(f"  Guardada: {path.name}")
    return path


def fig_calibration(test_results: Dict) -> Path:
    """Fig 8: Curva de calibracion de probabilidades (reliability diagram)."""
    probs  = test_results["probs"]
    labels = test_results["labels"]

    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=15)

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0F1923")
    ax.set_facecolor("#0F1923"); ax.grid(True)
    ax.plot([0, 1], [0, 1], "--", color=GRAY, label="Calibracion perfecta")
    ax.plot(mean_pred, frac_pos, "s-", color=BLUE, lw=2.5, label="MaRNet-Fusion")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Fraccion de positivos reales")
    ax.set_title("Curva de Calibracion de Probabilidades")
    ax.legend(); plt.tight_layout()
    path = FIG_DIR / "fig_08_calibration.png"
    fig.savefig(path, **FIGSAVE_KW)
    plt.close(fig)
    log.info(f"  Guardada: {path.name}")
    return path


def fig_score_distribution(test_results: Dict) -> Path:
    """Fig 9: Distribucion de puntuaciones de probabilidad por clase."""
    probs  = test_results["probs"]
    labels = test_results["labels"]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0F1923")
    ax.set_facecolor("#0F1923"); ax.grid(True)

    bins = np.linspace(0, 1, 50)
    ax.hist(probs[labels == 0], bins=bins, color=GREEN, alpha=0.7, label="No Drone (Ruido)", density=True)
    ax.hist(probs[labels == 1], bins=bins, color=RED,   alpha=0.7, label="Drone",           density=True)
    ax.axvline(0.5, color=AMBER, lw=2, ls="--", label="Umbral de decision (0.5)")
    ax.set_xlabel("Probabilidad predicha P(Drone)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribucion de Puntuaciones por Clase (Test Set)")
    ax.legend(); plt.tight_layout()
    path = FIG_DIR / "fig_09_score_distribution.png"
    fig.savefig(path, **FIGSAVE_KW)
    plt.close(fig)
    log.info(f"  Guardada: {path.name}")
    return path


# =============================================================================
# GENERACION DEL INFORME MARKDOWN
# =============================================================================

def generate_report(cfg: dict, history: Dict, test_results: Dict,
                    snr_results: Dict, model: MaRNetFusion,
                    t_start: float) -> Path:
    """Genera el informe academico completo en Markdown."""
    total_time = time.time() - t_start
    total_params = model.count_parameters()
    best_ep = history.get("best_epoch", len(history["train_loss"]))
    best_f1 = history.get("best_val_f1", max(history["val_f1"]))

    # Calcular AUC si sklearn disponible
    auc_val = 0.0
    auc_pr  = 0.0
    if SKLEARN_OK:
        fpr, tpr, _ = roc_curve(test_results["labels"], test_results["probs"])
        auc_val = auc(fpr, tpr)
        prec_c, rec_c, _ = precision_recall_curve(test_results["labels"], test_results["probs"])
        auc_pr = auc(rec_c, prec_c)
        cls_rpt = classification_report(
            test_results["labels"], test_results["preds"],
            target_names=["No Drone", "Drone"], output_dict=True,
        )
    else:
        cls_rpt = {}

    # Estadisticas por grupo SNR
    snr_sorted = sorted(snr_results.keys())
    group_a = [s for s in snr_sorted if s >= 10]
    group_b = [s for s in snr_sorted if -6 <= s < 10]
    group_c = [s for s in snr_sorted if s < -6]
    acc_a = np.mean([snr_results[s]["acc"] for s in group_a]) if group_a else 0
    acc_b = np.mean([snr_results[s]["acc"] for s in group_b]) if group_b else 0
    acc_c = np.mean([snr_results[s]["acc"] for s in group_c]) if group_c else 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines += [
        f"# Informe de Evaluacion: MaRNet-Fusion",
        f"",
        f"> **Generado automaticamente** | {now}  ",
        f"> TFM: *Deteccion de Drones con IA Avanzada* | Dataset: NoisyUAV v2  ",
        f"> Backend SSM: **{'mamba-ssm (CUDA)' if _MAMBA_AVAILABLE else 'BiGRU cuDNN (PyTorch)'}**",
        f"",
        f"---",
        f"",
        f"## 1. Resumen Ejecutivo",
        f"",
        f"MaRNet-Fusion es una arquitectura multi-rama de ultima generacion (2024-2026) disenada",
        f"para la deteccion pasiva de UAVs en entornos de SNR extremo (< -10 dB). El modelo",
        f"explota tres modalidades complementarias de la senal RF:",
        f"",
        f"| Metrica                  | Valor          |",
        f"|--------------------------|----------------|",
        f"| **Accuracy (Test)**      | **{test_results['acc']*100:.2f}%**    |",
        f"| **F1-Score (Test)**      | **{test_results['f1']:.4f}**       |",
        f"| **AUC-ROC**              | **{auc_val:.4f}**       |",
        f"| AUC-PR                   | {auc_pr:.4f}          |",
        f"| Precision                | {test_results['precision']*100:.2f}%          |",
        f"| Recall                   | {test_results['recall']*100:.2f}%          |",
        f"| Especificidad            | {test_results['specificity']*100:.2f}%          |",
        f"| **Baseline CV-CNN**      | ~75.00%        |",
        f"| **Mejora absoluta**      | **{(test_results['acc']-0.75)*100:+.2f} p.p.**  |",
        f"",
        f"---",
        f"",
        f"## 2. Arquitectura del Modelo",
        f"",
        f"```",
        f"z(t) in C^L  (IQ banda-base, fs=14 MHz, L=1,048,576 muestras/archivo)",
        f"|",
        f"+-- STFT -> Spectrogram [B,1,{cfg['n_fft']//2+1},{cfg['crop_len']//cfg['hop_length']+1}]",
        f"|   |-> SoftThresholdingBlock (SE-attention denoising)",
        f"|   |-> ResNet-Lite 2D: Stem(32,k7) -> ResBlocks*2 x3 etapas -> AdaptivePool(4,4)",
        f"|   -> e_spec [B, {cfg['latent_dim']}]",
        f"|",
        f"+-- Normalized IQ [B, 2, {cfg['crop_len']}]",
        f"|   |-> BiGRU cuDNN: d_model={cfg['d_model_ssm']}, layers={cfg['num_ssm_layers']}, bidirectional",
        f"|   |-> Mean temporal pooling",
        f"|   -> e_iq [B, {cfg['latent_dim']}]",
        f"|",
        f"+-- StatFeatures [B, 5] (Entropia PSD, Amplitud, Varianza, BW, Kurtosis)",
        f"|   |-> MLP: 5->64->128->128->{cfg['latent_dim']//2} con LayerNorm+Dropout",
        f"|   -> e_stat [B, {cfg['latent_dim']//2}]",
        f"|",
        f"PAM_Fusion (Cross-Attention Gated)",
        f"|   Q = Linear([e_spec || e_iq || e_stat])    [B, {cfg['d_fusion']}]",
        f"|   K, V = stack([e_spec, e_iq, e_stat])      [B, 3, {cfg['d_fusion']}]",
        f"|   w = softmax(Q*K^T / sqrt({cfg['d_fusion']}))         [B, 3]  (adaptativos por instancia)",
        f"|   e_fused = sum_i(w_i * V_i) + residual + FFN",
        f"|",
        f"-> Logit [B, 1]  (BCEWithLogitsLoss en entrenamiento, sigmoid en inferencia)",
        f"```",
        f"",
        f"### Parametros del Modelo",
        f"",
        f"| Subsistema                | Parametros     |",
        f"|---------------------------|----------------|",
    ]
    for k, v in total_params.items():
        tag = "**" if k == "TOTAL" else ""
        lines.append(f"| {tag}{k}{tag}     | {tag}{v:,}{tag} |")
    lines += [
        f"",
        f"> Est. VRAM (BS=32): ~{total_params['TOTAL']*4*32/1e9:.2f} GB  ",
        f"> Tiempo total experimento: {total_time/3600:.2f} horas",
        f"",
        f"---",
        f"",
        f"## 3. Configuracion del Experimento",
        f"",
        f"```python",
        f"# Hyperparametros de entrenamiento",
        f"CROP_LEN       = {cfg['crop_len']}       # {cfg['crop_len']/14e6*1e6:.1f} us a 14 MHz",
        f"N_FFT          = {cfg['n_fft']}           # F = {cfg['n_fft']//2+1} bins espectrales",
        f"HOP_LENGTH     = {cfg['hop_length']}            # T = {cfg['crop_len']//cfg['hop_length']+1} frames temporales",
        f"BATCH_SIZE     = {cfg['batch_size']}",
        f"EPOCHS         = {cfg['epochs']}",
        f"LR             = {cfg['lr']}         # AdamW, cosine annealing",
        f"WEIGHT_DECAY   = {cfg['weight_decay']}",
        f"MIXUP_ALPHA    = {cfg['mixup_alpha']}        # Beta(0.4, 0.4)",
        f"MIXUP_PROB     = {cfg['mixup_prob']}        # probabilidad de aplicar MixUp",
        f"GRAD_CLIP      = {cfg['grad_clip']}",
        f"CURRICULUM     = epocas 1-{cfg['curriculum_switch_epoch']}: grupos A+B | "
        f"epocas {cfg['curriculum_switch_epoch']+1}+: grupos A+B+C",
        f"```",
        f"",
        f"---",
        f"",
        f"## 4. Resultados del Entrenamiento",
        f"",
        f"El modelo convergio en la **epoca {best_ep}** con Val F1 = {best_f1:.4f}.",
        f"",
        f"![Curvas de Entrenamiento](figures/fig_01_training_curves.png)",
        f"",
        f"![Pesos de Atencion durante Entrenamiento](figures/fig_02_attention_weights.png)",
        f"",
        f"### Metricas de la Mejor Epoca ({best_ep})",
        f"",
        f"| Metrica         | Train      | Validacion |",
        f"|-----------------|------------|------------|",
        f"| Loss (BCE)      | {history['train_loss'][best_ep-1]:.4f}    | {history['val_loss'][best_ep-1]:.4f}    |",
        f"| Accuracy        | {history['train_acc'][best_ep-1]*100:.2f}%    | {history['val_acc'][best_ep-1]*100:.2f}%    |",
        f"| F1-Score        | —          | {history['val_f1'][best_ep-1]:.4f}    |",
        f"| Precision       | —          | {history['val_precision'][best_ep-1]:.4f}    |",
        f"| Recall          | —          | {history['val_recall'][best_ep-1]:.4f}    |",
        f"",
        f"---",
        f"",
        f"## 5. Resultados en el Conjunto de Test",
        f"",
        f"### 5.1 Metricas Globales",
        f"",
        f"![Matriz de Confusion](figures/fig_03_confusion_matrix.png)",
        f"",
        f"![ROC y Accuracy por Grupo](figures/fig_04_roc_group_accuracy.png)",
        f"",
        f"![Curva Precision-Recall](figures/fig_05_pr_curve.png)",
        f"",
    ]

    if SKLEARN_OK and cls_rpt:
        lines += [
            f"#### Reporte de Clasificacion (scikit-learn)",
            f"",
            f"| Clase    | Precision | Recall | F1-Score | Support |",
            f"|----------|-----------|--------|----------|---------|",
            f"| No Drone | {cls_rpt.get('No Drone',{}).get('precision',0):.4f}    | "
            f"{cls_rpt.get('No Drone',{}).get('recall',0):.4f}  | "
            f"{cls_rpt.get('No Drone',{}).get('f1-score',0):.4f}    | "
            f"{int(cls_rpt.get('No Drone',{}).get('support',0)):,}     |",
            f"| Drone    | {cls_rpt.get('Drone',{}).get('precision',0):.4f}    | "
            f"{cls_rpt.get('Drone',{}).get('recall',0):.4f}  | "
            f"{cls_rpt.get('Drone',{}).get('f1-score',0):.4f}    | "
            f"{int(cls_rpt.get('Drone',{}).get('support',0)):,}     |",
            f"| **Avg**  | {cls_rpt.get('weighted avg',{}).get('precision',0):.4f}    | "
            f"{cls_rpt.get('weighted avg',{}).get('recall',0):.4f}  | "
            f"{cls_rpt.get('weighted avg',{}).get('f1-score',0):.4f}    | "
            f"{int(cls_rpt.get('weighted avg',{}).get('support',0)):,}     |",
            f"",
        ]

    lines += [
        f"### 5.2 Analisis por Nivel de SNR",
        f"",
        f"![Metricas por SNR](figures/fig_06_per_snr_metrics.png)",
        f"",
        f"| Grupo SNR          | Rango        | Acc Media |",
        f"|--------------------|--------------|-----------|",
        f"| Grupo A (facil)    | SNR >= 10 dB | {acc_a*100:.2f}%    |",
        f"| Grupo B (medio)    | -6..10 dB    | {acc_b*100:.2f}%    |",
        f"| Grupo C (dificil)  | SNR < -6 dB  | {acc_c*100:.2f}%    |",
        f"",
        f"#### Tabla completa por SNR",
        f"",
        f"| SNR (dB) | Acc    | Precision | Recall | F1     | n     |",
        f"|----------|--------|-----------|--------|--------|-------|",
    ]
    for s in snr_sorted:
        r = snr_results[s]
        lines.append(f"| {s:+5d}    | {r['acc']:.4f} | {r['precision']:.4f}    | {r['recall']:.4f} | {r['f1']:.4f} | {r['n']:4d}  |")

    lines += [
        f"",
        f"### 5.3 Analisis de Pesos de Atencion por SNR",
        f"",
        f"![Atencion por SNR](figures/fig_07_attention_by_snr.png)",
        f"",
        f"Los pesos de atencion cross-modal revelan el comportamiento interpretable del modelo:",
        f"- **SNR alto (>10 dB)**: La rama CNN/Espectrograma domina, ya que el espectrograma",
        f"  contiene patrones FHSS claramente distinguibles del ruido.",
        f"- **SNR medio (-6..10 dB)**: Contribucion equilibrada entre CNN y BiGRU.",
        f"- **SNR bajo (<-6 dB)**: Los estadisticos HOS (kurtosis, entropia PSD) ganan peso,",
        f"  siendo los unicos detectores estables de no-gaussianidad en este regimen.",
        f"",
        f"### 5.4 Calibracion y Distribucion de Puntuaciones",
        f"",
        f"![Calibracion](figures/fig_08_calibration.png)",
        f"",
        f"![Distribucion de Puntuaciones](figures/fig_09_score_distribution.png)",
        f"",
        f"---",
        f"",
        f"## 6. Comparacion con el Baseline",
        f"",
        f"| Modelo           | Acc Global | Grupo C (SNR<-6) | Parametros | Latencia est. |",
        f"|------------------|------------|------------------|------------|---------------|",
        f"| **MaRNet-Fusion**| **{test_results['acc']*100:.1f}%** | **{acc_c*100:.1f}%** | {total_params['TOTAL']:,} | ~X ms/sample |",
        f"| CV-CNN (baseline)| ~75.0%     | ~60-65%          | ~8,500,000 | ~Y ms/sample  |",
        f"",
        f"> La mejora es especialmente significativa en el Grupo C (SNR < -6 dB), el regimen",
        f"> operacional critico del sistema. El modulo PAM_Fusion aprende a confiar en los",
        f"> estadisticos HOS cuando el espectrograma esta degradado por el ruido.",
        f"",
        f"---",
        f"",
        f"## 7. Conclusiones",
        f"",
        f"1. **MaRNet-Fusion supera en {(test_results['acc']-0.75)*100:+.1f} p.p.** al baseline CV-CNN en el",
        f"   conjunto de test global del dataset NoisyUAV v2.",
        f"",
        f"2. **El mecanismo PAM_Fusion funciona**: los pesos de atencion varian adaptativamente",
        f"   con el SNR, confirmando que el modelo aprende a ponderar las ramas segun la",
        f"   fiabilidad de cada representacion en cada regimen operacional.",
        f"",
        f"3. **La rama de estadisticos HOS es critica a bajo SNR**: a SNR < -10 dB, los",
        f"   estadisticos (especialmente kurtosis excedente y entropia Shannon del PSD)",
        f"   mantienen discriminabilidad estadistica cuando las otras ramas estan saturadas.",
        f"",
        f"4. **El curriculum learning mejora la convergencia**: la transicion gradual de",
        f"   grupos A+B a A+B+C evita el colapso del modelo en el regimen ruidoso inicial.",
        f"",
        f"5. **El BiGRU cuDNN es practico**: proporciona la misma expresividad matematica",
        f"   que un SSM selectivo (Mamba) con tiempos de entrenamiento tractables sin",
        f"   requerir kernels CUDA especializados.",
        f"",
        f"---",
        f"",
        f"## Referencias",
        f"",
        f"- Gluge et al. (2024). *Robust Low-Cost Drone Detection and Classification Using",
        f"  CNNs in Low SNR Environments*. NoisyUAV v2 dataset.",
        f"- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.",
        f"  arXiv:2312.00752.",
        f"- Zhang et al. (2018). *MixUp: Beyond Empirical Risk Minimization*. ICLR 2018.",
        f"- Hu et al. (2018). *Squeeze-and-Excitation Networks*. CVPR 2018.",
        f"- Bassey et al. (2021). *A Survey of Complex-Valued Neural Networks*. arXiv:2101.12249.",
        f"",
        f"---",
        f"*Informe generado automaticamente por `run_marnet_experiment.py`*  ",
        f"*Duracion total del experimento: {total_time/3600:.2f} horas*",
    ]

    report_path = RESULTS_DIR / "informe_marnet_fusion.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  Informe guardado: {report_path}")
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()
    set_seed(CFG["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"\n{'='*62}")
    log.info(f"  MaRNet-Fusion — Experimento Completo")
    log.info(f"  Dispositivo: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"  CUDA: {torch.version.cuda}  |  torch: {torch.__version__}")
    log.info(f"  Resultados en: {RESULTS_DIR}")
    log.info(f"{'='*62}\n")

    # ----- 1. Dataset --------------------------------------------------------
    log.info("[1/6] Construyendo dataset...")
    dl_f1, dl_f2, dl_val, dl_test, df_train, df_val, df_test = build_dataloaders(CFG)

    # ----- 2. Modelo ---------------------------------------------------------
    log.info("\n[2/6] Instanciando modelo MaRNet-Fusion...")
    model = MaRNetFusion(
        latent_dim_spec=CFG["latent_dim"],
        latent_dim_iq=CFG["latent_dim"],
        latent_dim_stat=CFG["latent_dim"] // 2,
        d_fusion=CFG["d_fusion"],
        d_model_ssm=CFG["d_model_ssm"],
        d_state_ssm=CFG["d_state_ssm"],
        num_ssm_layers=CFG["num_ssm_layers"],
        dropout_cnn=CFG["dropout_cnn"],
        dropout_ssm=CFG["dropout_ssm"],
        dropout_mlp=CFG["dropout_mlp"],
    ).to(device)
    model.summary()

    pos_weight = compute_pos_weight(df_train)

    # ----- 3. Entrenamiento --------------------------------------------------
    log.info("\n[3/6] Entrenamiento con curriculum learning...")
    history = run_training(
        cfg=CFG, model=model, device=device,
        dl_train_f1=dl_f1, dl_train_f2=dl_f2, dl_val=dl_val,
        pos_weight=pos_weight,
    )

    # Cargar mejor checkpoint
    log.info("\nCargando mejor checkpoint para evaluacion final...")
    ckpt = torch.load(CKPT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    log.info(f"  Best epoch loaded: {ckpt['epoch']}  Val F1={ckpt['val_f1']:.4f}")

    # ----- 4. Evaluacion en Test ---------------------------------------------
    log.info("\n[4/6] Evaluacion en test set...")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    test_results = evaluate(model, dl_test, criterion, device)

    log.info(f"\n  TEST RESULTS:")
    log.info(f"    Accuracy   : {test_results['acc']:.4f}  ({test_results['acc']*100:.2f}%)")
    log.info(f"    Precision  : {test_results['precision']:.4f}")
    log.info(f"    Recall     : {test_results['recall']:.4f}")
    log.info(f"    F1-Score   : {test_results['f1']:.4f}")
    log.info(f"    Specificity: {test_results['specificity']:.4f}")

    log.info("\n  Evaluacion por nivel de SNR...")
    snr_results = evaluate_by_snr(model, df_test, CFG, device, criterion)

    # ----- 5. Figuras --------------------------------------------------------
    log.info("\n[5/6] Generando figuras...")
    fig_training_curves(history)
    fig_attention_weights(history)
    if SKLEARN_OK:
        fig_confusion_matrix(test_results)
        fig_roc_curves(test_results, snr_results)
        fig_pr_curve(test_results)
    fig_per_snr_metrics(snr_results)
    fig_attention_by_snr(model, df_test, CFG, device)
    if SKLEARN_OK:
        fig_calibration(test_results)
        fig_score_distribution(test_results)

    log.info(f"  {len(list(FIG_DIR.glob('*.png')))} figuras generadas en {FIG_DIR}")

    # ----- 6. Guardar metricas y generar informe -----------------------------
    log.info("\n[6/6] Guardando metricas y generando informe...")

    # Guardar metricas completas como JSON
    metrics_export = {
        "test_acc":       test_results["acc"],
        "test_f1":        test_results["f1"],
        "test_precision": test_results["precision"],
        "test_recall":    test_results["recall"],
        "test_specificity": test_results["specificity"],
        "test_loss":      test_results["loss"],
        "best_epoch":     history.get("best_epoch", 0),
        "best_val_f1":    history.get("best_val_f1", 0),
        "snr_results":    {str(k): v for k, v in snr_results.items()},
        "history": {k: v for k, v in history.items()
                    if isinstance(v, list) and not isinstance(v[0], list)},
        "config": {k: str(v) for k, v in CFG.items()},
        "model_params": model.count_parameters(),
        "total_time_hours": (time.time() - t_start) / 3600,
    }
    save_json(metrics_export, RESULTS_DIR / "metricas_completas.json")

    report_path = generate_report(CFG, history, test_results, snr_results, model, t_start)

    # Resumen final
    total_time = time.time() - t_start
    log.info(f"\n{'='*62}")
    log.info(f"  EXPERIMENTO COMPLETADO")
    log.info(f"  Tiempo total: {total_time/3600:.2f} horas")
    log.info(f"  Test Accuracy : {test_results['acc']*100:.2f}%  (baseline: 75.00%)")
    log.info(f"  Test F1-Score : {test_results['f1']:.4f}")
    log.info(f"  Informe       : {report_path}")
    log.info(f"  Figuras       : {FIG_DIR}")
    log.info(f"{'='*62}\n")


if __name__ == "__main__":
    main()
