import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

from NoisyUAV.modelos.cvcnn import ComplexConvBlock, modulus

PHYS_DIM   = 8    # 8 features per burst
CNN_EMBED  = 256  

class CVCNNBackbone(nn.Module):
    def __init__(self, pool_size=32):
        super().__init__()
        self.conv1 = ComplexConvBlock(1, 32, kernel_size=11, stride=2, padding=5)
        self.conv2 = ComplexConvBlock(32, 64, kernel_size=11, stride=2, padding=5)
        self.conv3 = ComplexConvBlock(64, 128, kernel_size=11, stride=2, padding=5)
        self.conv4 = ComplexConvBlock(128, 128, kernel_size=11, stride=1, padding=5)
        
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        self.fc = nn.Linear(128 * pool_size, CNN_EMBED)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x_real = modulus(x)
        x_pool = self.pool(x_real)
        x_flat = x_pool.view(x_pool.size(0), -1)
        embed = self.fc(x_flat)
        return embed

class BurstCVCNN(nn.Module):
    def __init__(self, phys_dim=PHYS_DIM, cnn_embed=CNN_EMBED, pool_size=32, hidden_dim=256, dropout_cnn=0.3, dropout_fuse=0.4):
        super().__init__()
        
        self.backbone = CVCNNBackbone(pool_size=pool_size)
        self.drop_cnn = nn.Dropout(dropout_cnn)
        
        self.phys_bn = nn.BatchNorm1d(phys_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(cnn_embed + phys_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_fuse),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_fuse),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_iq, feats):
        cnn_embed = self.backbone(x_iq)
        cnn_embed = self.drop_cnn(cnn_embed)
        
        feats_norm = self.phys_bn(feats)
        
        fused = torch.cat([cnn_embed, feats_norm], dim=1)
        logit = self.mlp(fused)
        return logit

class BurstCVCNNDataset(Dataset):
    def __init__(self, csv_df, data_dir, augment=False, phys_mean=None, phys_std=None):
        self.df = csv_df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.fs = 14e6
        self.crop_len = 131072  # ~9.4ms para fallbacks
        
        
        self.phys_cols = ['dur_ms', 'z_peak', 'drop_b', 'n_act_burst', 
                          'global_nf', 'global_ns', 'global_H_mean', 'global_p75_act']
                          
        self.phys_features = torch.tensor(self.df[self.phys_cols].values, dtype=torch.float32)
        if phys_mean is not None and phys_std is not None:
            self.phys_mean = phys_mean
            self.phys_std = phys_std
        else:
            self.phys_mean = self.phys_features.mean(dim=0)
            self.phys_std  = self.phys_features.std(dim=0).clamp(min=1e-8)
            
        self.phys_features = (self.phys_features - self.phys_mean) / self.phys_std
        self.phys_features = self.phys_features.clamp(-5.0, 5.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['file_path']
        
        # Load IQ
        full_path = self.data_dir / file_path
        d = torch.load(full_path, map_location='cpu', weights_only=False)
        iq = d['x_iq'].float()  # [2, 1048576]
        L = iq.shape[1]
        
        if row.get('is_dummy', False):
            # Fallback a un crop estandar de 9.4ms para no asfixiar la GPU con 1 millon de ceros
            if L <= self.crop_len:
                pad = torch.zeros(2, self.crop_len - L)
                iq_crop = torch.cat([iq, pad], dim=1)
            else:
                s = (L - self.crop_len) // 2
                iq_crop = iq[:, s:s + self.crop_len]
        else:
            t_start = row['t_start']
            t_end   = row['t_end']
            
            start_idx = max(0, int((t_start / 1000.0) * self.fs))
            end_idx   = min(L, int((t_end / 1000.0) * self.fs))
            
            if start_idx >= end_idx:
                end_idx = min(L, start_idx + 1024)
                
            iq_crop = iq[:, start_idx:end_idx]
            
        # RMS Normalization
        power = iq_crop.pow(2).mean().clamp(min=1e-12).sqrt()
        iq_crop = iq_crop / power
        # --- FIX CRÍTICO DE PADDING ---
        # Rellenar con ceros hasta self.crop_len (131072 muestras = 9.4 ms) o truncar.
        # Esto estandariza el padding para que AdaptiveAvgPool1d reciba siempre la misma 
        # escala de señal/ceros, igualando las condiciones de entrenamiento y de inferencia.
        TARGET_LEN = self.crop_len
        c_dim, l_dim = iq_crop.shape
        if l_dim < TARGET_LEN:
            pad = torch.zeros(c_dim, TARGET_LEN - l_dim, dtype=iq_crop.dtype)
            iq_crop = torch.cat([iq_crop, pad], dim=1)
        else:
            iq_crop = iq_crop[:, :TARGET_LEN]
            
        if self.augment:
            theta = torch.rand(1).item() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
            iq_crop = torch.matmul(rot, iq_crop)
            
            noise_power = 0.02
            noise = torch.randn_like(iq_crop) * noise_power
            iq_crop = iq_crop + noise
            
        feats = self.phys_features[idx]
        label = torch.tensor([row['label']], dtype=torch.float32)
        
        return {
            'iq': iq_crop.clone(),
            'feats': feats,
            'label': label,
            'filename': file_path,
            'burst_id': row['burst_id'],
            'snr': row['snr'],
            'target_mc': row['target_multiclass']
        }

def pad_seq_collate(batch):
    # Ya no necesitamos padding dinámico porque __getitem__ garantiza 
    # que todos los tensores tienen longitud TARGET_LEN (131072).
    iqs = [item['iq'] for item in batch]
    feats = [item['feats'] for item in batch]
    labels = [item['label'] for item in batch]
        
    return {
        'iq': torch.stack(iqs, dim=0),
        'feats': torch.stack(feats, dim=0),
        'label': torch.stack(labels, dim=0)
    }
