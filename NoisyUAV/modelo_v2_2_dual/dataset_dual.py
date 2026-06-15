"""
Dataset Dual-Stream V2 — Nivel de Instancia + AWGN Augmentation
================================================================
Cada fila del CSV = 1 muestra de entrenamiento.
En modo train, las muestras de dron a SNR >= 10 dB tienen probabilidad
de ser degradadas con AWGN sintético a SNRs hostiles (-10 a -20 dB).
Esto genera muestras de baja SNR con etiquetas 100% correctas.
"""
import os, random, math, sys
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

sys.path.append(r"c:\repos\DroneDetectionRF")
from NoisyUAV.funciones.dsp_rf.detector_entropia import detectar_bursts

class DualDataset(Dataset):
    def __init__(self, csv_file, data_dir, split='train', fs=14e6, window_len=131072,
                 augment_prob=0.4, augment_snr_range=(-20, -8)):
        df = pd.read_csv(csv_file)
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.data_dir = data_dir
        self.split = split
        self.fs = fs
        self.window_len = window_len
        self.half_win = window_len // 2
        self.augment_prob = augment_prob
        self.augment_snr_range = augment_snr_range
        
        # Índices de muestras de dron con SNR alta (candidatas a augmentación)
        if split == 'train':
            self.high_snr_drone_idx = self.df[
                (self.df['label'] == 1) & (self.df['snr'] >= 10) & (self.df['is_fallback'] == False)
            ].index.tolist()
            print(f"  [AUG] {len(self.high_snr_drone_idx)} muestras de dron SNR>=10 disponibles para AWGN augmentation")

    def __len__(self):
        return len(self.df)
    
    def _add_awgn(self, iq_window, target_snr_db):
        """
        Añade AWGN complejo a una ventana IQ ya normalizada por RMS.
        Tras la normalización RMS, la potencia de la señal es ~1.0.
        target_snr_db controla cuánto ruido se inyecta.
        """
        signal_power = iq_window.pow(2).mean()
        noise_power = signal_power / (10 ** (target_snr_db / 10.0))
        noise_std = math.sqrt(noise_power.item())
        noise = torch.randn_like(iq_window) * noise_std
        return iq_window + noise

    def __getitem__(self, idx):
        # Decidir si hacemos augmentación AWGN (solo en train)
        do_augment = (
            self.split == 'train' 
            and random.random() < self.augment_prob 
            and len(self.high_snr_drone_idx) > 0
        )
        
        if do_augment:
            # Sustituimos esta muestra por una de dron a SNR alta degradada
            aug_idx = random.choice(self.high_snr_drone_idx)
            row = self.df.iloc[aug_idx]
        else:
            row = self.df.iloc[idx]
            
        file_path = os.path.join(self.data_dir, row['filename'])
        label = float(row['label'])
        
        # Physical features
        global_nf = float(row['global_nf'])
        global_H = float(row['global_H_mean'])
        z_peak = float(row['z_peak'])
        
        t_center = float(row['t_center'])
        
        # Load Raw IQ
        try:
            d = torch.load(file_path, map_location='cpu', weights_only=False)
            iq_full = d['x_iq'].float()
        except Exception:
            iq_full = torch.zeros(2, 1050000)
            
        max_idx = iq_full.shape[1]
        
        # Jitter temporal en entrenamiento (±1 ms)
        if self.split == 'train':
            jitter_ms = random.uniform(-1.0, 1.0)
            t_center = max(0.0, t_center + jitter_ms)
            
        # Recortar ventana fija de 9.4 ms alrededor del centro
        c_idx = int((t_center / 1000.0) * self.fs)
        start = c_idx - self.half_win
        end = c_idx + self.half_win
        
        if start < 0:
            win = iq_full[:, 0:self.window_len]
        elif end > max_idx:
            win = iq_full[:, max_idx - self.window_len:max_idx]
        else:
            win = iq_full[:, start:end]
            
        # [CRÍTICO] Normalización RMS
        power = win.pow(2).mean().clamp(min=1e-12).sqrt()
        win = win / power
        
        # AWGN Augmentation: degradar a SNR hostil
        if do_augment:
            target_snr = random.uniform(*self.augment_snr_range)
            win = self._add_awgn(win, target_snr)
            
            # [NUEVO V2.2] Recalcular CFAR analíticamente sobre el tensor degradado
            # win es [2, 131072] (9.4 ms) en CPU
            # adaptive_window_ms=0 porque en 9.4 ms solo tiene sentido el umbral global
            _, _, H_smooth, _, nf_v, ns, _, bursts = detectar_bursts(
                win, fs=self.fs, adaptive_window_ms=0
            )
            
            global_nf = float(np.median(nf_v))
            global_H = float(np.mean(H_smooth))
            
            if bursts:
                z_peak = float(max([b['z_peak'] for b in bursts]))
            else:
                # Fallback analítico si no detecta ráfaga que supere el umbral
                z_peak = float((np.min(H_smooth) - global_nf) / (ns + 1e-10))
        
        phys = torch.tensor([global_nf, global_H, z_peak], dtype=torch.float32)
        
        return win, phys, torch.tensor(label, dtype=torch.float32)
