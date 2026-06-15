"""
Entrenamiento Dual-Stream V2 — Nivel de Instancia (sin MIL)
============================================================
BCE directa por muestra. Sin Max-Logit Pooling. Sin bolsas.
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

from model import DualStreamCVCNN
from dataset_dual import DualDataset

OUT_DIR = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_2_dual"
CSV_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_1_dual\dataset_v2_1_clean_pointers.csv"
DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
METRICS_JSON = os.path.join(OUT_DIR, "metrics_history.json")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: No se encuentra {CSV_PATH}. Ejecuta build_dataset_v5_pointers.py primero.")
        return

    # Datasets (el filtrado por split se hace dentro de DualDataset)
    ds_train = DualDataset(CSV_PATH, DATA_DIR, split='train')
    ds_val   = DualDataset(CSV_PATH, DATA_DIR, split='val')
    
    print(f"Train: {len(ds_train)} instancias | Val: {len(ds_val)} instancias")
    
    # DataLoaders optimizados para velocidad
    dl_train = DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    model = DualStreamCVCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # Scheduler: reduce LR si F1 se estanca 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # ---------- Reanudación desde last_model ----------
    start_epoch = 0
    best_f1 = 0.0
    history = []
    
    last_ckpt_path = os.path.join(OUT_DIR, "checkpoints", "last_model.pth")
    best_ckpt_path = os.path.join(OUT_DIR, "checkpoints", "best_model.pth")
    
    if os.path.exists(last_ckpt_path):
        print(f"Reanudando desde {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch']
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
            best_f1 = best_ckpt.get('val_f1', 0.0)
        if os.path.exists(METRICS_JSON):
            with open(METRICS_JSON, 'r') as f:
                history = json.load(f)
        print(f"  Reanudando desde epoch {start_epoch + 1} | Best F1: {best_f1:.4f}")

    epochs = 40

    print("Comenzando entrenamiento Dual-Stream V2 (instancias)...")
    for ep in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        
        for iq_batch, phys_batch, labels_batch in tqdm(dl_train, desc=f'Ep {ep+1:02d} Train', leave=False):
            iq_batch = iq_batch.to(device)
            phys_batch = phys_batch.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, attn = model(iq_batch, phys_batch)  # [B, 1]
                loss = criterion(logits, labels_batch)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        train_loss /= len(dl_train)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for iq_batch, phys_batch, labels_batch in tqdm(dl_val, desc=f'Ep {ep+1:02d} Val', leave=False):
                iq_batch = iq_batch.to(device)
                phys_batch = phys_batch.to(device)
                labels_batch = labels_batch.to(device).unsqueeze(1)
                
                with torch.amp.autocast('cuda'):
                    logits, _ = model(iq_batch, phys_batch)
                    loss = criterion(logits, labels_batch)
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(logits)
                    all_preds.extend((probs > 0.5).cpu().numpy().flatten())
                    all_labels.extend(labels_batch.cpu().numpy().flatten())
        
        val_loss /= len(dl_val)
        f1 = f1_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step(f1)
        
        print(f"Ep {ep+1:02d} | L_Tr: {train_loss:.4f} | L_Val: {val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | LR: {current_lr:.1e} | {elapsed:.0f}s")
        
        # Historial completo
        epoch_metrics = {
            'epoch': ep + 1,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'val_f1': round(f1, 6),
            'val_acc': round(acc, 6),
            'lr': current_lr,
            'time_s': round(elapsed, 1),
        }
        history.append(epoch_metrics)
        
        with open(METRICS_JSON, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Grafica
        h_tr = [h['train_loss'] for h in history]
        h_vl = [h['val_loss'] for h in history]
        h_f1 = [h['val_f1'] for h in history]
        h_ac = [h['val_acc'] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(h_tr, label='Train Loss', color='blue')
        ax1.plot(h_vl, label='Val Loss', color='red')
        ax1.set_title('Evolucion de la Perdida (Loss)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('BCE Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(h_f1, label='Val F1-Score', color='green')
        ax2.plot(h_ac, label='Val Accuracy', color='purple')
        ax2.set_title('Evolucion de Metricas')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
        plt.close(fig)
        
        # Checkpoint LAST
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'epoch': ep + 1,
            'val_f1': f1,
        }, last_ckpt_path)
        
        # Checkpoint BEST
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state': model.state_dict(),
                'epoch': ep + 1,
                'val_f1': f1,
                'val_acc': acc,
                'val_loss': val_loss,
            }, best_ckpt_path)
            print(f"  -> Nuevo best model guardado (F1={f1:.4f})")
    
    print("=" * 60)
    print(f"Entrenamiento terminado. Best F1: {best_f1:.4f}")
    print(f"Historial: {METRICS_JSON}")
    print(f"Best model: {best_ckpt_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
