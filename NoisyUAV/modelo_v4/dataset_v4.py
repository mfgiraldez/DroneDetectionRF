import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class DualDatasetV4(Dataset):
    def __init__(self, csv_path, data_dir, split='train', target_len=131072, fs=14e6, augmentation=True):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.data_dir = data_dir
        self.target_len = target_len
        self.fs = fs
        self.augmentation = augmentation and (split == 'train')

    def add_awgn(self, iq, snr_target_db):
        """ Inyecta ruido gaussiano complejo para alcanzar la SNR objetivo """
        # iq: [2, L]
        p_sig = torch.mean(iq**2)
        if p_sig < 1e-10: return iq
        
        snr_linear = 10**(snr_target_db / 10.0)
        p_noise = p_sig / snr_linear
        
        noise = torch.randn_like(iq) * torch.sqrt(p_noise / 2.0)
        return iq + noise

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fpath = os.path.join(self.data_dir, row['filename'])
        
        # Carga perezosa del IQ
        d = torch.load(fpath, map_location='cpu', weights_only=False)
        iq_full = d['x_iq'].float() # [2, N]
        
        # Extraer ventana de 9.4ms centrada en t_center (ms)
        t_samples = int((row['t_center'] / 1000.0) * self.fs)
        
        # Jitter temporal en entrenamiento (+- 1ms)
        if self.augmentation:
            t_samples += random.randint(-int(0.001 * self.fs), int(0.001 * self.fs))
            
        half_len = self.target_len // 2
        start = max(0, t_samples - half_len)
        end = start + self.target_len
        
        if end > iq_full.shape[1]:
            end = iq_full.shape[1]
            start = max(0, end - self.target_len)
            
        iq_crop = iq_full[:, start:end].clone()
        
        # Pad si es necesario
        if iq_crop.shape[1] < self.target_len:
            pad = torch.zeros(2, self.target_len - iq_crop.shape[1])
            iq_crop = torch.cat([iq_crop, pad], dim=1)
            
        # Aumentación: Bajar SNR de drones reales a niveles hostiles
        z_peak = row['z_peak']
        n_bins = row['n_bins']
        
        if self.augmentation and row['type'] == 'drone_real' and random.random() < 0.6:
            # Rango de -20 a -4 dB en pasos de 2 dB
            target_snr = random.choice(list(range(-20, -3, 2)))
            iq_crop = self.add_awgn(iq_crop, target_snr)
            # Al enterrarlo en ruido, las features CFAR originales ya no valen
            # Las bajamos proporcionalmente (z_peak es escala lineal de potencia aprox)
            z_peak = z_peak * (10**(target_snr/20.0)) 
            n_bins = 0 # En ruido extremo el ancho de banda es indistinguible
            
        # Normalización RMS OBLIGATORIA
        rms = torch.sqrt(torch.mean(iq_crop**2) + 1e-12)
        iq_crop = iq_crop / rms
        
        # Features Físicas: [global_nf, global_H_mean, z_peak, n_bins]
        phys = torch.tensor([
            row['global_nf'] / 10.0,      # Norm básica
            row['global_H_mean'] / 10.0,
            z_peak / 50.0,              # Norm básica z-score
            n_bins / 2048.0             # Norm frecuencia
        ], dtype=torch.float32)
        
        label = torch.tensor([row['label']], dtype=torch.float32)
        
        return iq_crop, phys, label
