import sys
import os
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from copy import deepcopy
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # Disable GUI to avoid freezing in Windows during ML training
import matplotlib.pyplot as plt

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelos.burst_cvcnn import BurstCVCNN, BurstCVCNNDataset, pad_seq_collate
from NoisyUAV.funciones.plot_styles import apply_ieee_style, get_color_palette, clean_spines

os.environ['PYTHONIOENCODING'] = 'utf-8'

# ================================
# CONFIGURACION
# ================================
DATA_DIR    = Path(r"C:\TFM_data\NoisyUAV\drone_RF_data")
CSV_PATH    = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_burst_v2\bursts_dataset.csv")
OUT_DIR     = Path(r"c:\repos\DroneDetectionRF\NoisyUAV\resultados_burst_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT  = OUT_DIR / "checkpoints" / "best_model.pt"
CHECKPOINT.parent.mkdir(exist_ok=True)
FIG_DIR     = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

EPOCHS      = 60
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL   = 16  # Más pequeño porque la validación puede cargar señales de 75ms enteras (fallback)
LR          = 3e-4
WEIGHT_DECAY= 1e-3
CROP_LEN    = 65536
MIXUP_ALPHA = 0.2

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    apply_ieee_style()
    colors = get_color_palette()
    
    print(f"--- Entrenamiento Burst-Centric Classifier ---")
    print(f"Device: {DEVICE}")
    
    df = pd.read_csv(CSV_PATH)
    print(f"Bursts totales: {len(df)}")
    
    df_train = df[df['split'] == 'train']
    df_val   = df[df['split'] == 'val']
    df_test  = df[df['split'] == 'test']
    
    # IMPORTANTE: En Train, quitamos los is_dummy para no confundir al modelo
    # con falsas afirmaciones (ej. decir que ruido puro es dron porque falló el CFAR)
    df_train = df_train[df_train['is_dummy'] == False].copy()
    
    # En Val/Test MANTENEMOS los is_dummy para que cuente como fallo si CFAR no vio un dron.
    print(f"Train (sin dummies): {len(df_train)}")
    print(f"Val: {len(df_val)}   Test: {len(df_test)}")
    
    # Dataset Train (computes statistics)
    ds_train = BurstCVCNNDataset(df_train, DATA_DIR, augment=True)
    mean, std = ds_train.phys_mean, ds_train.phys_std
    
    ds_val = BurstCVCNNDataset(df_val, DATA_DIR, augment=False, phys_mean=mean, phys_std=std)
    ds_test= BurstCVCNNDataset(df_test, DATA_DIR, augment=False, phys_mean=mean, phys_std=std)
    
    # IMPORTANTE WINDOWS: num_workers=0
    ld_train = DataLoader(ds_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  num_workers=0, collate_fn=pad_seq_collate)
    ld_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE_VAL,   shuffle=False, num_workers=0, collate_fn=pad_seq_collate)
    
    # Pos weight for Class Imbalance (if any)
    n_drone = (df_train['label'] == 1).sum()
    n_noise = (df_train['label'] == 0).sum()
    pos_w = torch.tensor([n_noise / n_drone], dtype=torch.float32).to(DEVICE)
    print(f"Pos Weight calculado: {pos_w.item():.4f}")
    
    model = BurstCVCNN(dropout_cnn=0.3, dropout_fuse=0.4).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    scaler    = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    best_f1 = 0
    start_epoch = 0
    patience_cnt = 0
    history = {'train_loss':[], 'val_loss':[], 'val_f1':[], 'val_acc':[]}
    
    # === RESUME LOGIC ===
    if CHECKPOINT.exists():
        print(f"---> Encontrado Checkpoint: {CHECKPOINT.name}. Intentando restaurar...")
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        if 'scaler_state' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state'])
            
        best_f1 = ckpt.get('val_f1', 0)
        start_epoch = ckpt.get('epoch', 0) + 1
        
        hist_file = OUT_DIR / "history.json"
        if hist_file.exists():
            with open(hist_file, "r") as f:
                history = json.load(f)
                
        print(f"---> Reanudando en Época {start_epoch+1}. (Mejor F1 anterior: {best_f1:.4f})")
    
    fig_hist, ax_hist = plt.subplots(1, 2, figsize=(10, 4))
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t0 = time.time()
        
        train_loss = 0
        for batch in tqdm(ld_train, desc=f"Ep {epoch+1:02d} Train", leave=False):
            optimizer.zero_grad()
            iq = batch['iq'].to(DEVICE)
            f  = batch['feats'].to(DEVICE)
            lbl= batch['label'].to(DEVICE)
            
            # MixUp implementation
            if np.random.rand() < 0.5:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                idx = torch.randperm(iq.size(0))
                iq = lam * iq + (1-lam) * iq[idx]
                f  = lam * f  + (1-lam) * f[idx]
                lbl_b = lbl[idx]
                
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logit = model(iq, f)
                    loss = lam * criterion(logit, lbl) + (1-lam) * criterion(logit, lbl_b)
            else:
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logit = model(iq, f)
                    loss = criterion(logit, lbl)
                    
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        train_loss /= len(ld_train)
        
        # Validation
        model.eval()
        val_loss = 0
        all_lbl, all_prob, all_pred = [], [], []
        with torch.no_grad():
            for batch in tqdm(ld_val, desc=f"Ep {epoch+1:02d} Val", leave=False):
                iq = batch['iq'].to(DEVICE)
                f  = batch['feats'].to(DEVICE)
                lbl= batch['label'].to(DEVICE)
                
                logit = model(iq, f)
                loss = criterion(logit, lbl)
                val_loss += loss.item()
                
                prob = torch.sigmoid(logit)
                pred = (prob > 0.5).float()
                
                all_lbl.extend(lbl.cpu().numpy())
                all_prob.extend(prob.cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                
        val_loss /= len(ld_val)
        val_acc = (np.array(all_pred) == np.array(all_lbl)).mean()
        val_f1  = f1_score(all_lbl, all_pred, zero_division=0)
        
        scheduler.step(val_f1)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        
        print(f"Ep {epoch+1:02d} | Tr L: {train_loss:.4f} | Val L: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | T: {time.time()-t0:.1f}s")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'phys_mean': mean.numpy(),
                'phys_std': std.numpy(),
                'val_f1': best_f1,
                'epoch': epoch
            }, CHECKPOINT)
            print(f"  --> Best Model Saved (Val F1: {best_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= 12:
                print("Early Stopping!")
                break
                
        # Live Plotting
        ax_hist[0].clear()
        ax_hist[1].clear()
        ax_hist[0].plot(history['train_loss'], c=colors['blue'], label='Train Loss')
        ax_hist[0].plot(history['val_loss'], c=colors['orange'], label='Val Loss')
        ax_hist[0].set_title("Loss", fontweight='bold')
        ax_hist[0].legend()
        clean_spines(ax_hist[0])
        
        ax_hist[1].plot(history['val_acc'], c=colors['green'], label='Val Acc')
        ax_hist[1].plot(history['val_f1'], c=colors['purple'], label='Val F1')
        ax_hist[1].set_title("Metrics", fontweight='bold')
        ax_hist[1].legend()
        clean_spines(ax_hist[1])
        
        plt.tight_layout()
        fig_hist.savefig(FIG_DIR / "training_curves_ieee.png", dpi=300)
        
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f)
        
    print("Entrenamiento completado!")

if __name__ == '__main__':
    main()
