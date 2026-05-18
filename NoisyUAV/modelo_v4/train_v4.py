import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os, sys, json
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sys.path.insert(0, r"c:\repos\DroneDetectionRF")
from NoisyUAV.modelo_v4.model_v4 import DualStreamCVCNN_V4
from NoisyUAV.modelo_v4.dataset_v4 import DualDatasetV4

# Config
DATA_DIR = r"C:\TFM_data\NoisyUAV\drone_RF_data"
CSV_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v4\dataset_v4_train_val.csv"
OUT_DIR  = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v4\checkpoints"
LOG_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v4\metrics_history.json"
PLOT_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v4\training_curves.png"

EPOCHS = 50
BATCH_SIZE = 32
LR = 3e-4

def plot_progress(history):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_f1"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_prec = [h.get("val_prec", 0) for h in history]
    val_rec = [h.get("val_rec", 0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_loss, label="Train Loss", color="blue")
    ax1.plot(epochs, val_loss, label="Val Loss", color="red")
    ax1.set_title("Loss Curves")
    ax1.legend(); ax1.grid(True)
    
    ax2.plot(epochs, val_f1, label="F1", color="green")
    ax2.plot(epochs, val_acc, label="Acc", color="purple")
    ax2.plot(epochs, val_prec, label="Prec", color="orange", linestyle="--")
    ax2.plot(epochs, val_rec, label="Rec", color="cyan", linestyle="--")
    ax2.set_title("Metrics Evolution")
    ax2.set_ylim(0, 1)
    ax2.legend(); ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

def train():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds = DualDatasetV4(CSV_PATH, DATA_DIR, split='train', augmentation=True)
    val_ds   = DualDatasetV4(CSV_PATH, DATA_DIR, split='val',   augmentation=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = DualStreamCVCNN_V4().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_epoch = 1
    history = []
    best_f1 = 0.0
    
    # RESUME LOGIC
    last_path = os.path.join(OUT_DIR, "last_model.pth")
    if os.path.exists(last_path):
        print(f"  >>> Reanudando entrenamiento desde {last_path}")
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt: scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt: scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r") as f: history = json.load(f)
            best_f1 = max([h["val_f1"] for h in history]) if history else 0.0

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for iq, phys, labels in pbar:
            iq, phys, labels = iq.to(device), phys.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                logits, _ = model(iq, phys)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        model.eval()
        val_preds, val_gt, val_probs = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for iq, phys, labels in val_loader:
                iq, phys, labels = iq.to(device), phys.to(device), labels.to(device)
                logits, _ = model(iq, phys)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_probs.extend(probs.tolist())
                val_preds.extend((probs >= 0.5).astype(int))
                val_gt.extend(labels.cpu().numpy().flatten().astype(int))
        
        f1 = f1_score(val_gt, val_preds, zero_division=0)
        acc = accuracy_score(val_gt, val_preds)
        prec = precision_score(val_gt, val_preds, zero_division=0)
        rec = recall_score(val_gt, val_preds, zero_division=0)
        
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "val_f1": f1,
            "val_acc": acc,
            "val_prec": prec,
            "val_rec": rec
        }
        history.append(epoch_metrics)
        print(f"  [VAL] Loss: {epoch_metrics['val_loss']:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        
        scheduler.step(f1)
        plot_progress(history)
        
        # Guardar Checkpoint para Reanudación
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "val_f1": f1
        }, last_path)
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_f1": f1
            }, os.path.join(OUT_DIR, "best_model.pth"))
            print(f"  [NEW BEST] F1={f1:.4f}")
            
        with open(LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)

if __name__ == "__main__":
    train()

if __name__ == "__main__":
    train()
