"""
main_ht.py — Entrenamiento Dual-Stream V2.1 Hard Test (Sin Target=5)
=====================================================================
Idéntico al main_dual.py del V2.1. Solo cambian las rutas.
El modelo NO verá ninguna señal de Target=5 (Taranis) durante el entrenamiento.
"""
import os, sys, time, json
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import DualStreamCVCNN
from dataset_ht import HardTestDataset

OUT_DIR  = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual_hard_test"
CSV_PATH = os.path.join(OUT_DIR, "dataset_ht.csv")
DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
METRICS_JSON = os.path.join(OUT_DIR, "metrics_history.json")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HT] Device: {device}")
    print(f"[HT] EXPERIMENTO: Entrenamiento SIN Target=5 (Taranis)")
    print(f"[HT] Objetivo: validar generalización zero-shot al Taranis")

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: No existe {CSV_PATH}. Ejecuta prepare_ht_dataset.py primero.")
        return

    ds_train = HardTestDataset(CSV_PATH, DATA_DIR, split='train')
    ds_val   = HardTestDataset(CSV_PATH, DATA_DIR, split='val')
    print(f"[HT] Train: {len(ds_train)} instancias | Val: {len(ds_val)} instancias")

    dl_train = DataLoader(ds_train, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model     = DualStreamCVCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    start_epoch = 0
    best_f1     = 0.0
    history     = []

    last_ckpt = os.path.join(OUT_DIR, "checkpoints", "last_model.pth")
    best_ckpt = os.path.join(OUT_DIR, "checkpoints", "best_model.pth")

    if os.path.exists(last_ckpt):
        print(f"[HT] Reanudando desde {last_ckpt}...")
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch']
        if os.path.exists(best_ckpt):
            best_f1 = torch.load(best_ckpt, map_location='cpu', weights_only=False).get('val_f1', 0.0)
        if os.path.exists(METRICS_JSON):
            with open(METRICS_JSON) as f:
                history = json.load(f)
        print(f"[HT] Reanudando epoch {start_epoch + 1} | Best F1: {best_f1:.4f}")

    EPOCHS = 40
    print(f"[HT] Comenzando entrenamiento ({EPOCHS} epochs)...")

    for ep in range(start_epoch, EPOCHS):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for iq_b, phys_b, lbl_b in tqdm(dl_train, desc=f'Ep {ep+1:02d} Train', leave=False):
            iq_b  = iq_b.to(device)
            phys_b= phys_b.to(device)
            lbl_b = lbl_b.to(device).unsqueeze(1)
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

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for iq_b, phys_b, lbl_b in tqdm(dl_val, desc=f'Ep {ep+1:02d} Val', leave=False):
                iq_b   = iq_b.to(device); phys_b = phys_b.to(device)
                lbl_b  = lbl_b.to(device).unsqueeze(1)
                with torch.amp.autocast('cuda'):
                    logits, _ = model(iq_b, phys_b)
                    val_loss += criterion(logits, lbl_b).item()
                probs = torch.sigmoid(logits)
                all_preds.extend((probs > 0.5).cpu().numpy().flatten())
                all_labels.extend(lbl_b.cpu().numpy().flatten())

        val_loss /= len(dl_val)
        f1  = f1_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        lr  = optimizer.param_groups[0]['lr']
        scheduler.step(f1)
        elapsed = time.time() - t0

        print(f"Ep {ep+1:02d} | L_Tr:{train_loss:.4f} | L_Val:{val_loss:.4f} | "
              f"F1:{f1:.4f} | Acc:{acc:.4f} | LR:{lr:.1e} | {elapsed:.0f}s")

        history.append({'epoch': ep+1, 'train_loss': round(train_loss,6),
                        'val_loss': round(val_loss,6), 'val_f1': round(f1,6),
                        'val_acc': round(acc,6), 'lr': lr, 'time_s': round(elapsed,1)})
        with open(METRICS_JSON, 'w') as f:
            json.dump(history, f, indent=2)

        # Curvas de entrenamiento
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot([h['train_loss'] for h in history], label='Train Loss', color='blue')
        ax1.plot([h['val_loss']   for h in history], label='Val Loss',   color='red')
        ax1.set_title('[HT] Loss — Sin Target=5'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True)
        ax2.plot([h['val_f1']  for h in history], label='Val F1',  color='green')
        ax2.plot([h['val_acc'] for h in history], label='Val Acc', color='purple')
        ax2.set_title('[HT] Métricas'); ax2.set_xlabel('Epoch'); ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "training_curves_ht.png"), dpi=150)
        plt.close(fig)

        # Checkpoint LAST (siempre)
        torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                    'scaler_state': scaler.state_dict(), 'scheduler_state': scheduler.state_dict(),
                    'epoch': ep+1, 'val_f1': f1}, last_ckpt)

        # Checkpoint BEST
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'model_state': model.state_dict(), 'epoch': ep+1,
                        'val_f1': f1, 'val_acc': acc, 'val_loss': val_loss}, best_ckpt)
            print(f"  -> [HT] Nuevo best model (F1={f1:.4f})")

    print("=" * 60)
    print(f"[HT] Entrenamiento completado. Best Val F1: {best_f1:.4f}")
    print(f"[HT] NOTA: Target=5 (Taranis) NO fue visto en ningún momento.")
    print(f"[HT] Lanzar evaluate_ht_golden.py para el test de generalización.")
    print("=" * 60)

if __name__ == "__main__":
    main()
