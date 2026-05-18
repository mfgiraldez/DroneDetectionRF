"""
Detección binaria de drones mediante Raw IQ + ResNet18-1D.

Metodología basada en Zheng et al. (2025): "Deep Learning-Based Individual
Drone Identification Using Raw IQ Data". La red ingiere la señal IQ cruda sin
ninguna transformación tiempo-frecuencia previa.

Tarea   : Clasificación binaria → {0: No-Dron (ruido/interferencia), 1: Dron}
Dataset : NoisyUAV v2 (Glüge et al. 2024), archivos .pt [2, 1048576]
Modelo  : ResNet18-1D con in_channels=2 (canales I y Q)

Ejecución: python train_detector.py
"""

import os
import glob
import re
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR     = r"C:\TFM_data\NoisyUAV\drone_RF_data"
MODEL_PATH   = r"C:\TFM_data\resnet18_iq_best.pt"
METRICS_PATH = r"C:\TFM_data\resnet18_iq_metrics.json"
SPLIT_PATH   = r"C:\TFM_data\resnet18_iq_splits.json"
PLOTS_DIR    = r"C:\TFM_data\plots"

NOISE_CLASS  = 4        # Target 4 = Ruido en NoisyUAV
WINDOW_SIZE  = 131072   # ~9.4 ms a 14 MHz — captura al menos 1 burst FHSS

BATCH_SIZE   = 16       # Reducido por el mayor tamaño de ventana
EPOCHS       = 60       # Aumentado: el modelo no convergió en 30 épocas
LR           = 1e-3
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
SEED         = 42
NUM_WORKERS  = 4        # Ponlo a 0 si hay problemas en Windows
RESUME_FROM  = MODEL_PATH  # Reanudar desde checkpoint existente (None para empezar de cero)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Regex para parsear nombres de archivo
_PATTERN = re.compile(r"IQdata_sample(\d+)_target(\d+)_snr(-?\d+)\.pt")


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTRUCCIÓN DEL CATÁLOGO DE ARCHIVOS Y SPLITS
# ─────────────────────────────────────────────────────────────────────────────

def build_file_catalog(data_dir: str) -> list:
    """
    Escanea data_dir y devuelve una lista de dicts con las claves:
    {path, target, snr, label} donde label=0 (ruido) o 1 (dron).
    """
    files = glob.glob(os.path.join(data_dir, "IQdata_*.pt"))
    catalog = []
    for f in files:
        m = _PATTERN.match(Path(f).name)
        if m:
            target = int(m.group(2))
            snr    = int(m.group(3))
            catalog.append({
                "path":   f,
                "target": target,
                "snr":    snr,
                "label":  0 if target == NOISE_CLASS else 1,
            })
    return catalog


def split_catalog(catalog: list, val_ratio: float, test_ratio: float,
                  seed: int) -> tuple:
    """
    Divide el catálogo en train/val/test estratificando por (label, snr)
    para garantizar representación equitativa de todas las condiciones.
    Guarda los splits en SPLIT_PATH para trazabilidad.
    """
    strat_keys = [f"{e['label']}_{e['snr']}" for e in catalog]

    # Separar Test primero
    idx_all  = list(range(len(catalog)))
    idx_temp, idx_test = train_test_split(
        idx_all, test_size=test_ratio, stratify=strat_keys,
        random_state=seed
    )
    strat_temp = [strat_keys[i] for i in idx_temp]
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=val_ratio / (1 - test_ratio),
        stratify=strat_temp, random_state=seed
    )

    splits = {"train": idx_train, "val": idx_val, "test": idx_test}
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    with open(SPLIT_PATH, "w") as fh:
        json.dump(splits, fh)

    return idx_train, idx_val, idx_test


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class IQWindowDataset(Dataset):
    """
    Dataset de detección binaria sobre archivos .pt de NoisyUAV.

    Por cada muestra:
      1. Carga el tensor [2, 1048576] desde disco.
      2. Extrae una sub-ventana aleatoria de WINDOW_SIZE muestras.
      3. Normaliza por desviación estándar (robusto a variaciones de ganancia).
      4. Aplica Phase-Shift Augmentation si se pide (Zheng propone esta técnica).
    """
    def __init__(self, catalog: list, indices: list,
                 window_size: int = WINDOW_SIZE, augment: bool = False):
        self.entries     = [catalog[i] for i in indices]
        self.window_size = window_size
        self.augment     = augment

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        data  = torch.load(entry["path"], map_location="cpu", weights_only=False)
        x     = data["x_iq"].float()   # [2, 1048576]

        # Extracción aleatoria de ventana
        max_start = x.shape[1] - self.window_size
        start = random.randint(0, max_start) if self.augment else max_start // 2
        x = x[:, start : start + self.window_size]   # [2, window_size]

        # Normalización por desviación estándar
        std = x.std() + 1e-8
        x   = x / std

        # Phase-Shift Data Augmentation (Wang et al. 2025)
        if self.augment:
            phi     = random.uniform(0.0, 2.0 * np.pi)
            cos_phi = float(np.cos(phi))
            sin_phi = float(np.sin(phi))
            I_rot   = cos_phi * x[0] - sin_phi * x[1]
            Q_rot   = sin_phi * x[0] + cos_phi * x[1]
            x       = torch.stack([I_rot, Q_rot])

        return x, entry["label"], entry["snr"]


# ─────────────────────────────────────────────────────────────────────────────
#  MODELO: ResNet18-1D  (Zheng et al. 2025, adaptación 1D)
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock1D(nn.Module):
    """Bloque residual 1D equivalente al BasicBlock de ResNet18."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class ResNet18_IQ(nn.Module):
    """
    ResNet18 adaptado a señales IQ 1D.

    Input  : [B, 2, window_size]  (canales I y Q)
    Output : [B, 2]               (logits No-Dron / Dron)

    Sigue exactamente la arquitectura de ResNet18:
      Stem → MaxPool → [2,2,2,2] bloques → AvgPool → FC
    con canales [64, 64, 128, 256, 512] y Conv1d en lugar de Conv2d.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64,  64,  2, stride=1)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch: int, out_ch: int,
                    n_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock1D(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)
        return self.fc(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
#  BUCLES DE ENTRENAMIENTO Y EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scheduler, scaler, device):
    model.train()
    total_loss = correct = total = 0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs          = x.size(0)
        total_loss += loss.item() * bs
        preds       = logits.detach().argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += bs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_preds, all_labels, all_snrs = [], [], [], []

    for x, y, snr in loader:
        x_d = x.to(device, non_blocking=True)
        y_d = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x_d)
            loss   = criterion(logits, y_d)

        probs = logits.softmax(dim=1)[:, 1].cpu().float().numpy()
        total_loss += loss.item() * x.size(0)
        all_probs.extend(probs.tolist())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_labels.extend(y.numpy().tolist())
        all_snrs.extend(snr.numpy().tolist())

    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)
    snrs   = np.array(all_snrs)
    n      = len(labels)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    rep = classification_report(labels, preds,
                                target_names=["No-Dron", "Dron"],
                                output_dict=True, zero_division=0)

    # Métricas por nivel de SNR
    snr_metrics = {}
    for sv in sorted(np.unique(snrs)):
        mask    = snrs == sv
        sub_l   = labels[mask]
        sub_p   = preds[mask]
        acc     = (sub_p == sub_l).mean()
        s_m     = sub_l == 1
        n_m     = sub_l == 0
        pd      = sub_p[s_m].mean() if s_m.sum() > 0 else float("nan")
        pfa     = sub_p[n_m].mean() if n_m.sum() > 0 else float("nan")
        snr_metrics[int(sv)] = {
            "accuracy": float(acc), "Pd": float(pd), "Pfa": float(pfa)
        }

    return {
        "loss":        total_loss / n,
        "accuracy":    float(rep["accuracy"]),
        "precision":   float(rep["Dron"]["precision"]),
        "recall":      float(rep["Dron"]["recall"]),
        "f1":          float(rep["Dron"]["f1-score"]),
        "auc":         float(auc),
        "snr_metrics": snr_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────

def plot_training(history: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train")
    axes[0].plot(epochs, [h["val_loss"]   for h in history], label="Val")
    axes[0].set_title("Loss por Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["val_auc"] for h in history],
                 label="AUC-ROC", color="green")
    axes[1].plot(epochs, [h["val_f1"]  for h in history],
                 label="F1",      color="orange")
    axes[1].set_title("AUC-ROC y F1 (Validación)")
    axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[1].set_ylim([0, 1]); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [h["val_recall"] for h in history],
                 label="Recall / Pd", color="blue")
    axes[2].set_title("Probabilidad de Detección (Pd)")
    axes[2].set_xlabel("Epoch"); axes[2].legend()
    axes[2].set_ylim([0, 1]); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


def plot_snr(snr_metrics: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    snrs = sorted(snr_metrics.keys())
    pd   = [snr_metrics[s]["Pd"]       for s in snrs]
    pfa  = [snr_metrics[s]["Pfa"]      for s in snrs]
    acc  = [snr_metrics[s]["accuracy"] for s in snrs]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(snrs, pd,  "b-o",  label="Pd  (Prob. Detección)")
    ax.plot(snrs, pfa, "r-s",  label="Pfa (Prob. Falsa Alarma)")
    ax.plot(snrs, acc, "g--^", label="Accuracy")
    ax.axvline(x=-12, color="purple", linestyle="--", alpha=0.7,
               label="Límite SOTA Glüge (-12 dB)")
    ax.axvline(x=0,   color="gray",   linestyle=":",  alpha=0.5)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Probabilidad")
    ax.set_title("Rendimiento del Detector por Nivel de SNR (ResNet18-IQ)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_performance.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ResNet18-IQ: Detector Binario Dron / No-Dron")
    print("=" * 60)
    print(f"  Dispositivo : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Ventana     : {WINDOW_SIZE:,} muestras ({WINDOW_SIZE/14e6*1000:.1f} ms)")
    print()

    # Catálogo y splits
    print("Escaneando dataset...")
    catalog = build_file_catalog(DATA_DIR)
    if not catalog:
        raise FileNotFoundError(f"No se encontraron archivos .pt en: {DATA_DIR}")
    print(f"  Encontrados: {len(catalog):,} archivos")

    n_drone = sum(1 for e in catalog if e["label"] == 1)
    n_noise = sum(1 for e in catalog if e["label"] == 0)
    print(f"  Dron: {n_drone:,}  |  Ruido: {n_noise:,}")
    print()

    idx_train, idx_val, idx_test = split_catalog(
        catalog, VAL_RATIO, TEST_RATIO, SEED)

    train_labels = [catalog[i]["label"] for i in idx_train]
    val_labels   = [catalog[i]["label"] for i in idx_val]
    test_labels  = [catalog[i]["label"] for i in idx_test]
    print(f"  Train: {len(idx_train):,}  "
          f"({sum(train_labels):,} dron / {len(idx_train)-sum(train_labels):,} ruido)")
    print(f"  Val  : {len(idx_val):,}  "
          f"({sum(val_labels):,} dron / {len(idx_val)-sum(val_labels):,} ruido)")
    print(f"  Test : {len(idx_test):,}  "
          f"({sum(test_labels):,} dron / {len(idx_test)-sum(test_labels):,} ruido)")
    print()

    # DataLoaders
    train_ds = IQWindowDataset(catalog, idx_train, WINDOW_SIZE, augment=True)
    val_ds   = IQWindowDataset(catalog, idx_val,   WINDOW_SIZE, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    # Modelo (con soporte para reanudar desde checkpoint)
    model = ResNet18_IQ(num_classes=2).to(DEVICE)
    start_epoch = 1
    best_auc    = 0.0

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_auc    = ckpt["val_metrics"]["auc"]
        start_epoch = ckpt["epoch"] + 1
        print(f"  Reanudando desde epoch {ckpt['epoch']}  (AUC anterior: {best_auc:.4f})")
    else:
        print(f"  Entrenando desde cero")

    print(f"  Parámetros del modelo: {model.count_params():,}")
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS, pct_start=0.05,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # Entrenamiento
    history  = []

    print(f"{'─'*72}")
    print(f"{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} "
          f"{'AUC':>7} {'F1':>7} {'Pd':>7} {'Pfa':>7}")
    print(f"{'─'*72}")

    for epoch in range(start_epoch, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, scaler, DEVICE)
        val_m = evaluate(model, val_loader, criterion, DEVICE)

        pfa_0 = val_m["snr_metrics"].get(0, {}).get("Pfa", float("nan"))

        saved = ""
        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics":      val_m,
                "config": {
                    "window_size": WINDOW_SIZE,
                    "architecture": "ResNet18_IQ",
                    "noise_class": NOISE_CLASS,
                },
            }, MODEL_PATH)
            saved = "  *"

        print(f"  {epoch:3d}/{EPOCHS}  "
              f"{tr_loss:7.4f}  {tr_acc:6.4f}  "
              f"{val_m['loss']:7.4f}  {val_m['auc']:6.4f}  "
              f"{val_m['f1']:6.4f}  {val_m['recall']:6.4f}  "
              f"{pfa_0:6.4f}"
              f"{saved}")

        history.append({
            "epoch":      epoch,
            "train_loss": tr_loss,
            "train_acc":  tr_acc,
            "val_loss":   val_m["loss"],
            "val_auc":    val_m["auc"],
            "val_f1":     val_m["f1"],
            "val_recall": val_m["recall"],
        })

    print(f"{'─'*72}")
    print(f"\n  Mejor AUC-ROC: {best_auc:.4f}  ({MODEL_PATH})")

    # Guardar métricas y gráficas
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as fh:
        json.dump(history, fh, indent=2)

    ckpt    = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    snr_m   = ckpt["val_metrics"]["snr_metrics"]
    plot_training(history, PLOTS_DIR)
    plot_snr(snr_m, PLOTS_DIR)

    # Tabla SNR por consola
    print(f"\n{'─'*55}")
    print("  RENDIMIENTO POR SNR (mejor checkpoint)")
    print(f"  {'SNR':>6}  {'Pd':>7}  {'Pfa':>7}  {'Acc':>7}")
    print(f"  {'─'*35}")
    for sv in sorted(snr_m.keys()):
        m   = snr_m[sv]
        pd  = m.get("Pd", float("nan"))
        bar = ("█" * int(pd * 20)) if not (pd != pd) else ""
        print(f"  {sv:+4d} dB  {pd:7.4f}  "
              f"{m.get('Pfa', float('nan')):7.4f}  "
              f"{m['accuracy']:7.4f}  {bar}")
    print(f"{'─'*55}")
    print("\nSiguiente paso: python run_detector.py")


if __name__ == "__main__":
    main()
