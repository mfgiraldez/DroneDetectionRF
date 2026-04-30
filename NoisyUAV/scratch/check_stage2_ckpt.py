import torch
import os

path = r'C:\TFM_data\NoisyUAV\stage2_norm\checkpoints\checkpoint_epoch_29.pth'
if os.path.exists(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Best Val Acc: {ckpt.get('best_val_acc', 'N/A')}")
    history = ckpt.get('history', {})
    if history:
        print(f"Last Val Acc: {history.get('val_acc', [-1])[-1]:.2f}%")
        print(f"Total Epochs: {len(history.get('val_acc', []))}")
else:
    print("Checkpoint not found")
