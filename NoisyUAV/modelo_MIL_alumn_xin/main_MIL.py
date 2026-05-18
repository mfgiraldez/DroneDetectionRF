import os
import json
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

from dataset_mil import MILAlumnXinDataset
from model import MILAlumnXinCVCNN

# Configuración de logging mejorada
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

def plot_metrics(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label='Train Loss')
    plt.plot(epochs, history["val_loss"], label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_f1"], label='Val F1')
    plt.plot(epochs, history["val_acc"], label='Val Acc')
    plt.plot(epochs, history["val_auc"], label='Val AUC')
    plt.title('Metrics per Epoch')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()

def get_args():
    parser = argparse.ArgumentParser()
    # Usaremos el V4 que generaremos ahora
    parser.add_argument("--csv_path", type=str, default=r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_MIL_alumn_xin\alumn_dataset_pseudo_v4.csv")
    parser.add_argument("--data_dir", type=str, default=r"C:\TFM_data\NoisyUAV\drone_RF_data")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=2) 
    parser.add_argument("--accum_iter", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default=r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_MIL_alumn_xin")
    return parser.parse_args()

def collate_mil(batch):
    max_n = max(item["spec"].shape[0] for item in batch)
    specs, feats, labels, probs = [], [], [], []
    snrs, fallbacks = [], []
    for item in batch:
        s, f = item["spec"], item["feats"]
        curr_n = s.shape[0]
        if curr_n < max_n:
            s = torch.cat([s, s[-1:].repeat(max_n - curr_n, 1, 1, 1)], dim=0)
            f = torch.cat([f, f[-1:].repeat(max_n - curr_n, 1)], dim=0)
        specs.append(s); feats.append(f); labels.append(item["label"])
        probs.append(item["prob_ia"]); snrs.append(item["snr"]); fallbacks.append(item["fallback"])
    return {
        "spec": torch.stack(specs), "feats": torch.stack(feats),
        "label": torch.stack(labels), "prob_ia": torch.stack(probs),
        "snr": snrs, "fallback": fallbacks
    }

def train_one_epoch(model, dataloader, optimizer, criterion_dron, device, epoch, accum_iter, scaler):
    model.train()
    total_loss_dron = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"  Train Ep {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        spec = batch["spec"].to(device)
        feats = batch["feats"].to(device)
        label = batch["label"].to(device)
        prob_ia = batch["prob_ia"].to(device)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            logit_dron = model(spec, feats)
            # Loss de clasificación ponderada por la confianza del Oráculo
            loss = (criterion_dron(logit_dron, label) * prob_ia).mean()
            loss = loss / accum_iter
            
        scaler.scale(loss).backward()
        
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if device.type == 'cuda': torch.cuda.empty_cache()
        
        total_loss_dron += loss.item() * accum_iter
        pbar.set_postfix({"loss": f"{loss.item() * accum_iter:.4f}"})
        
    n = len(dataloader)
    return total_loss_dron/n

def validate_one_epoch(model, dataloader, criterion_dron, device, epoch):
    model.eval()
    total_loss_dron = 0.0
    all_labels, all_probs = [], []
    
    pbar = tqdm(dataloader, desc=f"  Val   Ep {epoch}", leave=False)
    with torch.no_grad():
        for batch in pbar:
            spec, feats, label = batch["spec"].to(device), batch["feats"].to(device), batch["label"].to(device)
            logit_dron = model(spec, feats)
            total_loss_dron += criterion_dron(logit_dron, label).mean().item()
            all_probs.extend(torch.sigmoid(logit_dron).cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy().flatten())
            
    n = len(dataloader)
    avg_loss = total_loss_dron / n if n > 0 else 0.0
    y_true, y_prob = np.array(all_labels), np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true) > 0 else 0.0
    try: auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.0
    return avg_loss, acc, f1, auc

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("="*50)
    logging.info(f"INICIANDO PIPELINE MIL-ALUMN-XIN (VERSIÓN PURGADA)")
    logging.info(f"Dispositivo: {device}")
    logging.info(f"Batch Size: {args.batch_size} | Num Workers: {args.num_workers}")
    logging.info(f"Salida: {args.out_dir}")
    logging.info("="*50)

    # Si el V4 no existe, avisar, pero intentar cargar
    if not os.path.exists(args.csv_path):
        logging.error(f"¡El dataset {args.csv_path} no existe! Genera el dataset V4 primero.")
        return

    logging.info(f"Cargando dataset desde {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    df_train, df_val = train_test_split(df, test_size=0.15, random_state=42, stratify=df["pseudo_label"])
    logging.info(f"Dataset cargado. Train: {len(df_train)} muestras, Val: {len(df_val)} muestras")
    
    ds_train = MILAlumnXinDataset(df_train, args.data_dir, is_train=True)
    ds_val = MILAlumnXinDataset(df_val, args.data_dir, phys_mean=ds_train.phys_mean, phys_std=ds_train.phys_std, is_train=False)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_mil)
    
    logging.info("Inicializando modelo MILAlumnXinCVCNN (Arquitectura Pura)...")
    model = MILAlumnXinCVCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion_dron = nn.BCEWithLogitsLoss(reduction='none')
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None
    
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": [], "val_auc": []}
    start_epoch = 1
    best_f1 = 0.0
    
    hist_path = os.path.join(args.out_dir, "history.json")
    if os.path.exists(hist_path):
        try:
            with open(hist_path, "r") as f:
                history = json.load(f)
            start_epoch = len(history["train_loss"]) + 1
            best_f1 = max(history["val_f1"]) if history["val_f1"] else 0.0
        except Exception as e:
            logging.warning(f"No se pudo cargar history.json: {e}")
            
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints = glob.glob(os.path.join(ckpt_dir, "mil_alumn_xin_ep*.pth"))
    if checkpoints:
        latest_ckpt = max(checkpoints, key=os.path.getctime)
        logging.info(f"Reanudando desde {latest_ckpt} (Epoch {start_epoch-1})")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    
    logging.info(f"Comenzando bucle de entrenamiento (restantes: {args.epochs - start_epoch + 1} epochs)...")
    for epoch in range(start_epoch, args.epochs + 1):
        # Currículum Learning
        phase = 1 if epoch <= 20 else (2 if epoch <= 40 else 3)
        ds_train.set_phase(phase)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_mil)
        
        t0 = time.time()
        loss_t = train_one_epoch(model, dl_train, optimizer, criterion_dron, device, epoch, args.accum_iter, scaler)
        val_loss, val_acc, val_f1, val_auc = validate_one_epoch(model, dl_val, criterion_dron, device, epoch)
        dt = time.time() - t0
        
        logging.info(
            f"Ep {epoch:02d}/{args.epochs} [Ph{phase}] | {dt:.1f}s | "
            f"L_Tr:{loss_t:.3f} | L_Val:{val_loss:.3f} | F1:{val_f1:.3f} | Acc:{val_acc:.3f}"
        )
        
        history["train_loss"].append(loss_t); history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1); history["val_acc"].append(val_acc); history["val_auc"].append(val_auc)
                     
        # GUARDAR SIEMPRE EN CADA EPOCH
        ckpt_path = os.path.join(ckpt_dir, f"mil_alumn_xin_ep{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(model.state_dict(), best_ckpt)
            logging.info(f"*** Nuevo best model guardado con F1: {best_f1:.4f} ***")
            
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)
            
        # Graficar progreso
        plot_metrics(history, args.out_dir)

if __name__ == "__main__":
    # Necesario para num_workers > 0 en Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
