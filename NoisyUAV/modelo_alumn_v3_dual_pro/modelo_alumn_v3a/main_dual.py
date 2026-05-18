"""
Entrenamiento Dual-Stream V3a — n_bins_peak como 4ª feature física
===================================================================
Idéntico al training loop de V2b salvo:
  - Carga DualStreamCVCNN de model.py local (4 features en phys_mlp)
  - Carga DualDataset de dataset_dual.py local (CSV V6)
  - Rutas actualizadas a modelo_alumn_v3a/
  - 50 épocas (vs 40 en V2b) para dar margen a la nueva feature
"""
import os, sys, time, json
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importaciones locales (ficheros en la misma carpeta)
sys.path.insert(0, os.path.dirname(__file__))
from model import DualStreamCVCNN
from dataset_dual import DualDataset

# ── Rutas ──────────────────────────────────────────────────────────────────────
OUT_DIR      = os.path.dirname(__file__)   # carpeta modelo_alumn_v3a/
CSV_PATH     = os.path.join(OUT_DIR, "dataset_v6_pointers.csv")
DATA_DIR     = r"C:\TFM_data\NoisyUAV\drone_RF_data"
METRICS_JSON = os.path.join(OUT_DIR, "metrics_history.json")
CKPT_DIR     = os.path.join(OUT_DIR, "checkpoints")
PLOTS_DIR    = os.path.join(OUT_DIR, "plots")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: No se encuentra {CSV_PATH}.")
        print("       Ejecuta build_dataset_v6_pointers.py primero.")
        return

    # ── Datasets ──────────────────────────────────────────────────────────────
    ds_train = DualDataset(CSV_PATH, DATA_DIR, split='train')
    ds_val   = DualDataset(CSV_PATH, DATA_DIR, split='val')
    print(f"Train: {len(ds_train)} instancias | Val: {len(ds_val)} instancias")

    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True,
                          num_workers=0, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=32, shuffle=False,
                          num_workers=0, pin_memory=True)

    # ── Modelo ────────────────────────────────────────────────────────────────
    model     = DualStreamCVCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ── Reanudación desde last_model ──────────────────────────────────────────
    start_epoch = 0
    best_f1     = 0.0
    history     = []

    last_ckpt = os.path.join(CKPT_DIR, "last_model.pth")
    best_ckpt = os.path.join(CKPT_DIR, "best_model.pth")

    if os.path.exists(last_ckpt):
        print(f"Reanudando desde {last_ckpt}...")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch']
        if os.path.exists(best_ckpt):
            bc = torch.load(best_ckpt, map_location='cpu', weights_only=False)
            best_f1 = bc.get('val_f1', 0.0)
        if os.path.exists(METRICS_JSON):
            with open(METRICS_JSON, 'r') as f:
                history = json.load(f)
        print(f"  Reanudando desde epoch {start_epoch + 1} | Best F1: {best_f1:.4f}")

    epochs = 50

    print(f"\nEntrenamiento Dual-Stream V3a ({epochs} épocas)...")
    print("=" * 60)

    for ep in range(start_epoch, epochs):
        t0         = time.time()
        train_loss = 0.0
        model.train()

        for iq_b, phys_b, lbl_b in tqdm(dl_train, desc=f'Ep {ep+1:02d} Train', leave=False):
            iq_b  = iq_b.to(device)
            phys_b = phys_b.to(device)
            lbl_b  = lbl_b.to(device).unsqueeze(1)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, _ = model(iq_b, phys_b)
                loss = criterion(logits, lbl_b)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(dl_train)

        # ── Validación ────────────────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for iq_b, phys_b, lbl_b in tqdm(dl_val, desc=f'Ep {ep+1:02d} Val', leave=False):
                iq_b   = iq_b.to(device)
                phys_b = phys_b.to(device)
                lbl_b  = lbl_b.to(device).unsqueeze(1)

                with torch.amp.autocast('cuda'):
                    logits, _ = model(iq_b, phys_b)
                    loss = criterion(logits, lbl_b)
                    val_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_preds.extend((probs > 0.5).cpu().numpy().flatten())
                all_labels.extend(lbl_b.cpu().numpy().flatten())

        val_loss   /= len(dl_val)
        f1          = f1_score(all_labels, all_preds)
        acc         = accuracy_score(all_labels, all_preds)
        elapsed     = time.time() - t0
        current_lr  = optimizer.param_groups[0]['lr']

        scheduler.step(f1)

        print(f"Ep {ep+1:02d} | "
              f"L_Tr={train_loss:.4f} | L_Val={val_loss:.4f} | "
              f"F1={f1:.4f} | Acc={acc:.4f} | "
              f"LR={current_lr:.1e} | {elapsed:.0f}s")

        # ── Historial ─────────────────────────────────────────────────────────
        history.append({
            'epoch':      ep + 1,
            'train_loss': round(train_loss, 6),
            'val_loss':   round(val_loss, 6),
            'val_f1':     round(f1, 6),
            'val_acc':    round(acc, 6),
            'lr':         current_lr,
            'time_s':     round(elapsed, 1),
        })
        with open(METRICS_JSON, 'w') as fj:
            json.dump(history, fj, indent=2)

        # ── Gráfica de entrenamiento ──────────────────────────────────────────
        h_tr = [h['train_loss'] for h in history]
        h_vl = [h['val_loss']   for h in history]
        h_f1 = [h['val_f1']     for h in history]
        h_ac = [h['val_acc']    for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(h_tr, label='Train Loss', color='blue')
        ax1.plot(h_vl, label='Val Loss',   color='red')
        ax1.set_title('Dual-Stream V3a — Pérdida')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('BCE Loss')
        ax1.legend(); ax1.grid(True)

        ax2.plot(h_f1, label='Val F1',       color='green')
        ax2.plot(h_ac, label='Val Accuracy', color='purple')
        ax2.set_title('Dual-Stream V3a — Métricas')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "training_curves.png"), dpi=150)
        plt.close(fig)

        # ── Checkpoints ───────────────────────────────────────────────────────
        torch.save({
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state':    scaler.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch':           ep + 1,
            'val_f1':          f1,
        }, last_ckpt)

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state': model.state_dict(),
                'epoch':       ep + 1,
                'val_f1':      f1,
                'val_acc':     acc,
                'val_loss':    val_loss,
            }, best_ckpt)
            print(f"  → Nuevo best model guardado (F1={f1:.4f})")

    print("=" * 60)
    print(f"Entrenamiento terminado. Best Val F1: {best_f1:.4f}")
    print(f"Best model: {best_ckpt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
