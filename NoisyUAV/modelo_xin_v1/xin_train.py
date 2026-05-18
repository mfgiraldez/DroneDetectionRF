"""
xin_train.py — Bucle de Entrenamiento con Trazabilidad Científica Completa
===========================================================================
Características:
    - Reanudación automática desde el último epoch (checkpoint `last`).
    - Guardado del mejor modelo por Val F1 (checkpoint `best`).
    - Métricas por epoch: Loss, Accuracy, Precision, Recall, F1 (train + val).
    - Figuras de curvas de entrenamiento generadas y guardadas en cada epoch.
    - Exportación de métricas completas a JSON.
    - Logging estructurado con timestamps.
"""

import os
import json
import logging
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import matplotlib
matplotlib.use("Agg")   # Sin display (entorno headless / background)
import matplotlib.pyplot as plt
import numpy as np

from .xin_dataset import XinSpectrogramDataset
from .xin_cvcnn import XinCVCNN


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación de un epoch
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Ejecuta un pase completo de evaluación (sin backprop) sobre `loader`.

    Retorna un diccionario con: loss, accuracy, precision, recall, f1.
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = (logits.sigmoid() >= 0.5).long().cpu().numpy().flatten()
            labels = y.long().cpu().numpy().flatten()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    n = len(loader.dataset)
    return {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Ejecuta un epoch de entrenamiento completo.

    Retorna las mismas métricas que evaluate_epoch para consistencia.
    """
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        # Gradient clipping: previene explosión de gradientes en redes complejas
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = (logits.sigmoid() >= 0.5).long().detach().cpu().numpy().flatten()
        labels = y.long().cpu().numpy().flatten()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    n = len(loader.dataset)
    return {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Generación de figuras de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def save_training_figures(history: Dict[str, List], figures_dir: str) -> None:
    """
    Genera y guarda las figuras de curvas de entrenamiento.

    Produce dos ficheros PNG:
        - fig_01_loss_curves.png   : Train/Val Loss por epoch.
        - fig_02_metrics_curves.png: Accuracy, F1, Precision, Recall por epoch.

    Parámetros
    ----------
    history    : Diccionario con listas de métricas por epoch.
    figures_dir: Directorio donde se guardan las figuras.
    """
    os.makedirs(figures_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Figura 1: Loss ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Curvas de pérdida — Xin CV-CNN 2D (NoisyUAV)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig_01_loss_curves.png"), dpi=150)
    plt.close(fig)

    # ── Figura 2: Métricas de clasificación ─────────────────────────────────
    metrics = ["accuracy", "f1", "precision", "recall"]
    labels  = ["Accuracy", "F1-Score", "Precision", "Recall"]
    colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        tr_key = f"train_{metric}"
        vl_key = f"val_{metric}"
        ax.plot(epochs, history[tr_key], label="Train", color=color,      marker="o", markersize=3)
        ax.plot(epochs, history[vl_key], label="Val",   color=color,      marker="s", markersize=3,
                linestyle="--", alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Métricas de clasificación — Xin CV-CNN 2D (NoisyUAV)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig_02_metrics_curves.png"), dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_f1: float,
    history: Dict,
) -> None:
    torch.save(
        {
            "epoch":        epoch,
            "best_val_f1":  best_val_f1,
            "model_state":  model.state_dict(),
            "optim_state":  optimizer.state_dict(),
            "history":      history,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, float, Dict]:
    """Carga un checkpoint y retorna (epoch_inicio, best_val_f1, history)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt["epoch"], ckpt["best_val_f1"], ckpt.get("history", {})


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def run_training(cfg: Dict) -> None:
    """
    Ejecuta el entrenamiento completo del modelo XinCVCNN con reanudación
    automática desde el último checkpoint.

    Parámetros del diccionario `cfg`
    --------------------------------
    csv_path     : str  — Ruta al CSV de splits generado por xin_build_dataset.
    data_dir     : str  — Directorio con los ficheros .pt del dataset.
    output_dir   : str  — Directorio de salida (checkpoints, figuras, métricas).
    epochs       : int  — Número total de epochs de entrenamiento.
    batch_size   : int  — Tamaño de batch.
    lr           : float— Tasa de aprendizaje inicial.
    weight_decay : float— Penalización L2 del optimizador.
    num_workers  : int  — Workers del DataLoader. **Fijar a 0 en Windows.**
    spec_h       : int  — Alto del espectrograma (eje frecuencial).
    spec_w       : int  — Ancho del espectrograma (eje temporal).
    nfft         : int  — Puntos de la STFT.
    """
    # ── Directorios de salida ───────────────────────────────────────────────
    ckpt_dir    = os.path.join(cfg["output_dir"], "checkpoints")
    figures_dir = os.path.join(cfg["output_dir"], "figures")
    os.makedirs(ckpt_dir,    exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    path_last = os.path.join(ckpt_dir, "xin_model_last.pt")
    path_best = os.path.join(ckpt_dir, "xin_model_best.pt")

    # ── Logging ─────────────────────────────────────────────────────────────
    log_path = os.path.join(cfg["output_dir"], "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("=" * 70)
    logging.info("INICIO DE ENTRENAMIENTO — Xin CV-CNN 2D (NoisyUAV)")
    logging.info("=" * 70)
    logging.info(f"Configuración: {json.dumps(cfg, indent=2)}")

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Dispositivo: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Datasets y DataLoaders ───────────────────────────────────────────────
    ds_kwargs = dict(
        csv_path   = cfg["csv_path"],
        nfft       = cfg.get("nfft", 1024),
        spec_h     = cfg.get("spec_h", 256),
        spec_w     = cfg.get("spec_w", 256),
        cache_dir  = cfg.get("cache_dir", None),   # None = on-the-fly
    )
    train_ds = XinSpectrogramDataset(**ds_kwargs, split="train")
    val_ds   = XinSpectrogramDataset(**ds_kwargs, split="val")

    dl_kwargs = dict(
        batch_size  = cfg["batch_size"],
        num_workers = cfg.get("num_workers", 0),
        pin_memory  = (device.type == "cuda"),
        drop_last   = False,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)

    logging.info(f"Train: {len(train_ds)} muestras | Val: {len(val_ds)} muestras")

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = XinCVCNN(num_classes=1, kernel_size=5).to(device)
    n_params = model.count_parameters()
    logging.info(f"Parámetros entrenables: {n_params:,}")

    # ── Optimizador (hiperparámetros de Xin et al.) ─────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg.get("lr", 1e-3),
        betas        = (0.9, 0.999),
        weight_decay = cfg.get("weight_decay", 1e-5),
    )

    # ── Función de pérdida ───────────────────────────────────────────────────
    # BCEWithLogitsLoss: variante numéricamente estable (fused sigmoid + BCE)
    criterion = nn.BCEWithLogitsLoss()

    # ── Reanudación automática ───────────────────────────────────────────────
    start_epoch = 0
    best_val_f1 = 0.0
    history: Dict[str, List] = {
        k: [] for k in [
            "train_loss", "train_accuracy", "train_precision",
            "train_recall", "train_f1",
            "val_loss",   "val_accuracy",   "val_precision",
            "val_recall", "val_f1",
        ]
    }

    # Prioridad de reanudación: LAST > BEST (igual que teacher_experiment)
    if os.path.exists(path_last):
        logging.info(f"Reanudando desde checkpoint LAST: {path_last}")
        start_epoch, best_val_f1, history = load_checkpoint(
            path_last, model, optimizer, device
        )
        logging.info(f"  → Epoch {start_epoch} | Mejor Val F1 histórico: {best_val_f1:.4f}")
    elif os.path.exists(path_best):
        logging.info(f"Reanudando desde checkpoint BEST: {path_best}")
        start_epoch, best_val_f1, history = load_checkpoint(
            path_best, model, optimizer, device
        )
        logging.info(f"  → Epoch {start_epoch} | Mejor Val F1 histórico: {best_val_f1:.4f}")
    else:
        logging.info("No se encontró checkpoint previo. Entrenamiento desde cero.")

    total_epochs = cfg["epochs"]

    # ── Bucle de entrenamiento ───────────────────────────────────────────────
    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()

        tr_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_metrics = evaluate_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        # Registro en history
        for k, v in tr_metrics.items():
            history[f"train_{k}"].append(v)
        for k, v in vl_metrics.items():
            history[f"val_{k}"].append(v)

        # Log de epoch
        logging.info(
            f"Epoch {epoch:03d}/{total_epochs} "
            f"[{elapsed:.0f}s] | "
            f"TrLoss={tr_metrics['loss']:.4f} TrAcc={tr_metrics['accuracy']:.4f} TrF1={tr_metrics['f1']:.4f} | "
            f"VlLoss={vl_metrics['loss']:.4f} VlAcc={vl_metrics['accuracy']:.4f} VlF1={vl_metrics['f1']:.4f}"
        )

        # ── Checkpoint LAST (siempre) ────────────────────────────────────────
        save_checkpoint(path_last, model, optimizer, epoch, best_val_f1, history)

        # ── Checkpoint BEST (solo si mejora Val F1) ──────────────────────────
        if vl_metrics["f1"] > best_val_f1:
            best_val_f1 = vl_metrics["f1"]
            save_checkpoint(path_best, model, optimizer, epoch, best_val_f1, history)
            logging.info(
                f"  ✓ Nuevo mejor modelo guardado — Val F1: {best_val_f1:.4f}"
            )

        # ── Figuras (actualizadas en cada epoch) ─────────────────────────────
        save_training_figures(history, figures_dir)

        # ── Métricas JSON ────────────────────────────────────────────────────
        metrics_path = os.path.join(cfg["output_dir"], "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"history": history, "best_val_f1": best_val_f1}, f, indent=2)

    logging.info("=" * 70)
    logging.info(f"Entrenamiento finalizado. Mejor Val F1: {best_val_f1:.4f}")
    logging.info(f"Checkpoint BEST: {path_best}")
    logging.info("=" * 70)
