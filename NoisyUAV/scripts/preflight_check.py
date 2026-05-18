import sys, os
sys.path.insert(0, r"c:\repos\DroneDetectionRF")
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from NoisyUAV.funciones.physical_features import extract_features, FEATURES_DIM
from NoisyUAV.modelos.hybrid_cvcnn import HybridCVCNN, HybridDataset
from NoisyUAV.funciones.dataset import obtener_splits_dataset

print("Imports OK")

model = HybridCVCNN()
iq  = torch.randn(2, 2, 131072)
f   = torch.randn(2, 12)
out = model(iq, f)
print(f"PRE-FLIGHT OK")
print(f"Output shape : {out.shape}")
print(f"Params       : {model.count_parameters():,}")
model.summary()

# Quick dataset check
df_tr, df_val, df_test = obtener_splits_dataset()
print(f"\nDataset splits OK")
print(f"  Train={len(df_tr):,} | Val={len(df_val):,} | Test={len(df_test):,}")
print(f"  Label counts (train): {df_tr['label'].value_counts().to_dict()}")
