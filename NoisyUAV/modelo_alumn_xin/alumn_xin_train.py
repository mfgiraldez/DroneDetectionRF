"""
alumn_xin_train.py — Bucle de Entrenamiento del Modelo Hibrido AlumnXin
=========================================================================
Caracteristicas:
  - Reanudacion automatica desde ultimo epoch (checkpoint LAST > BEST).
  - Guardado del mejor modelo por Val F1.
  - WeightedRandomSampler para balancear clases desbalanceadas del pseudo-dataset.
  - Figuras de curvas de entrenamiento actualizadas por epoch.
  - Exportacion de metricas a JSON.
  - Checkpoint guarda phys_mean/phys_std para que eval lo pueda cargar sin el CSV.
"""

import os
import json
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from NoisyUAV.modelo_alumn_xin.alumn_xin_dataset import AlumnXinDataset, collate_alumn_xin, PHYS_COLS
from NoisyUAV.modelo_alumn_xin.alumn_xin_cvcnn import AlumnXinCVCNN


# ---------------------------------------------------------------------------- #
# Evaluacion de un epoch                                                        #
# ---------------------------------------------------------------------------- #

def evaluate_epoch(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            spec  = batch["spec"].to(device, non_blocking=True)
            feats = batch["feats"].to(device, non_blocking=True)
            y     = batch["label"].to(device, non_blocking=True)

            logits = model(spec, feats)
            loss   = criterion(logits, y)
            total_loss += loss.item() * spec.size(0)

            probs  = logits.sigmoid().cpu().numpy().flatten()
            preds  = (probs >= 0.5).astype(int)
            labels = y.long().cpu().numpy().flatten()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    n = len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")

    return {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "auc":       auc,
    }


def train_epoch(model, loader, criterion, optimizer, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        spec  = batch["spec"].to(device, non_blocking=True)
        feats = batch["feats"].to(device, non_blocking=True)
        y     = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(spec, feats)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * spec.size(0)
        preds  = (logits.sigmoid() >= 0.5).long().detach().cpu().numpy().flatten()
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
        "auc":       float("nan"),
    }


# ---------------------------------------------------------------------------- #
# Figuras de entrenamiento                                                      #
# ---------------------------------------------------------------------------- #

def save_training_figures(history: Dict[str, List], figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   marker="s", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("Curvas de perdida — AlumnXin Hybrid (NoisyUAV)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig_01_loss_curves.png"), dpi=150)
    plt.close(fig)

    metrics = ["accuracy", "f1", "precision", "recall"]
    labels  = ["Accuracy", "F1-Score", "Precision", "Recall"]
    colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    for ax, metric, lbl, color in zip(axes, metrics, labels, colors):
        ax.plot(epochs, history[f"train_{metric}"], label="Train", color=color, marker="o", markersize=3)
        ax.plot(epochs, history[f"val_{metric}"],   label="Val",   color=color, marker="s", markersize=3,
                linestyle="--", alpha=0.8)
        ax.set_title(lbl); ax.set_xlabel("Epoch"); ax.set_ylabel(lbl)
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Metricas — AlumnXin Hybrid (NoisyUAV)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig_02_metrics_curves.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# Checkpoint I/O                                                                #
# ---------------------------------------------------------------------------- #

def save_checkpoint(path, model, optimizer, epoch, best_val_f1, history,
                    phys_mean, phys_std):
    torch.save({
        "epoch":       epoch,
        "val_f1":      best_val_f1,
        "best_val_f1": best_val_f1,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "history":     history,
        "phys_mean":   phys_mean.numpy(),
        "phys_std":    phys_std.numpy(),
    }, path)


def load_checkpoint(path, model, optimizer, device) -> Tuple[int, float, Dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt["epoch"], ckpt.get("best_val_f1", 0.0), ckpt.get("history", {})


# ---------------------------------------------------------------------------- #
# Funcion principal de entrenamiento                                            #
# ---------------------------------------------------------------------------- #

def run_training(cfg: Dict) -> None:
    ckpt_dir    = os.path.join(cfg["output_dir"], "checkpoints")
    figures_dir = os.path.join(cfg["output_dir"], "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    path_last = os.path.join(ckpt_dir, "alumn_xin_model_last.pt")
    path_best = os.path.join(ckpt_dir, "alumn_xin_model_best.pt")

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
    logging.info("INICIO ENTRENAMIENTO — AlumnXin Hybrid CV-CNN 2D (NoisyUAV)")
    logging.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Datos ────────────────────────────────────────────────────────────────
    df_full = pd.read_csv(cfg["csv_path"])
    df_full["label"] = df_full["pseudo_label"].astype(float)

    df_train = df_full[df_full["split"] == "train"].copy()
    df_val   = df_full[df_full["split"] == "val"].copy()

    train_ds = AlumnXinDataset(df_train, cfg["data_dir"],
                               cache_dir=cfg.get("cache_dir"),
                               phys_mean=None, phys_std=None)
    val_ds   = AlumnXinDataset(df_val, cfg["data_dir"],
                               cache_dir=cfg.get("cache_dir"),
                               phys_mean=train_ds.phys_mean,
                               phys_std=train_ds.phys_std)

    # WeightedRandomSampler para balancear clases
    labels_train = df_train["pseudo_label"].values.astype(int)
    class_counts = np.bincount(labels_train)
    weights_per_class = 1.0 / class_counts
    sample_weights = weights_per_class[labels_train]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(train_ds), replacement=True,
    )

    dl_kw = dict(batch_size=cfg["batch_size"], num_workers=0,
                 pin_memory=(device.type == "cuda"), collate_fn=collate_alumn_xin)
    train_loader = DataLoader(train_ds, sampler=sampler,  **dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False,    **dl_kw)

    logging.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = AlumnXinCVCNN(phys_dim=8, kernel_size=5,
                          fusion_hidden=cfg.get("fusion_hidden", 512),
                          dropout_fusion=cfg.get("dropout_fusion", 0.4)).to(device)
    logging.info(f"Parametros: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        betas=(0.9, 0.999),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # Peso positivo para compensar desbalance residual
    pos_weight = torch.tensor(
        [class_counts[0] / max(class_counts[1], 1)], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Reanudacion ───────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_f1 = 0.0
    history_keys = ["train_loss","train_accuracy","train_precision","train_recall","train_f1",
                    "val_loss",  "val_accuracy",  "val_precision",  "val_recall",  "val_f1","val_auc"]
    history = {k: [] for k in history_keys}

    if os.path.exists(path_last):
        logging.info(f"Reanudando desde LAST: {path_last}")
        start_epoch, best_val_f1, history = load_checkpoint(path_last, model, optimizer, device)
        history = {k: history.get(k, []) for k in history_keys}
        logging.info(f"  Epoch {start_epoch} | Mejor Val F1: {best_val_f1:.4f}")
    elif os.path.exists(path_best):
        logging.info(f"Reanudando desde BEST: {path_best}")
        start_epoch, best_val_f1, history = load_checkpoint(path_best, model, optimizer, device)
        history = {k: history.get(k, []) for k in history_keys}
    else:
        logging.info("Entrenamiento desde cero.")

    total_epochs = cfg["epochs"]
    phys_mean = train_ds.phys_mean.cpu()
    phys_std  = train_ds.phys_std.cpu()

    # ── Bucle ─────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch + 1, total_epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, criterion, optimizer, device)
        vl = evaluate_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        for k, v in tr.items():
            if k != "auc":
                history[f"train_{k}"].append(v)
        for k, v in vl.items():
            if k != "auc":
                history[f"val_{k}"].append(v)
        history["val_auc"].append(vl["auc"])

        logging.info(
            f"Epoch {epoch:03d}/{total_epochs} [{elapsed:.0f}s] | "
            f"TrLoss={tr['loss']:.4f} TrF1={tr['f1']:.4f} | "
            f"VlLoss={vl['loss']:.4f} VlF1={vl['f1']:.4f} VlAUC={vl['auc']:.4f}"
        )

        save_checkpoint(path_last, model, optimizer, epoch, best_val_f1, history,
                        phys_mean, phys_std)

        if vl["f1"] > best_val_f1:
            best_val_f1 = vl["f1"]
            save_checkpoint(path_best, model, optimizer, epoch, best_val_f1, history,
                            phys_mean, phys_std)
            logging.info(f"  Nuevo mejor modelo — Val F1: {best_val_f1:.4f}")

        save_training_figures(history, figures_dir)

        metrics_path = os.path.join(cfg["output_dir"], "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"history": history, "best_val_f1": best_val_f1}, f, indent=2)

    logging.info("=" * 70)
    logging.info(f"Entrenamiento finalizado. Mejor Val F1: {best_val_f1:.4f}")
    logging.info("=" * 70)
