"""
Dataset Dual-Stream V3b — Nivel de Instancia + n_bins_peak + Index
========================================================================
Idéntico a V3a, con una adición crítica para DivideMix:
Devuelve también el `idx` original de la muestra. Esto permite al bucle
de entrenamiento rastrear la "loss" de cada muestra específica para
ajustar el GMM.
"""
import os, random, math
import torch
from torch.utils.data import Dataset
import pandas as pd

NPERSEG = 2048

class DualDataset(Dataset):
    def __init__(self, csv_file, data_dir, split='train', fs=14e6,
                 window_len=131072, augment_prob=0.4,
                 augment_snr_range=(-20, -8), eval_mode=False):
        df = pd.read_csv(csv_file)
        self.df       = df[df['split'] == split].reset_index(drop=True)
        self.data_dir = data_dir
        self.split    = split
        self.fs       = fs
        self.window_len = window_len
        self.half_win   = window_len // 2
        self.augment_prob      = augment_prob
        self.augment_snr_range = augment_snr_range
        self.eval_mode = eval_mode  # Si es True, no aplica augmentación (para evaluar losses)

        if 'n_bins_peak' not in self.df.columns:
            raise ValueError("El CSV no contiene 'n_bins_peak'.")

        if split == 'train':
            self.high_snr_drone_idx = self.df[
                (self.df['label'] == 1) &
                (self.df['snr'] >= 10) &
                (self.df['is_fallback'] == False)
            ].index.tolist()

    def __len__(self):
        return len(self.df)

    def _add_awgn(self, iq_window, target_snr_db):
        signal_power = iq_window.pow(2).mean()
        noise_power  = signal_power / (10 ** (target_snr_db / 10.0))
        noise_std    = math.sqrt(noise_power.item())
        return iq_window + torch.randn_like(iq_window) * noise_std

    def __getitem__(self, idx):
        do_augment = (
            self.split == 'train'
            and not self.eval_mode
            and random.random() < self.augment_prob
            and len(self.high_snr_drone_idx) > 0
        )

        if do_augment:
            aug_idx = random.choice(self.high_snr_drone_idx)
            row = self.df.iloc[aug_idx]
        else:
            row = self.df.iloc[idx]

        file_path = os.path.join(self.data_dir, row['filename'])
        label     = float(row['label'])

        global_nf   = float(row['global_nf'])
        global_H    = float(row['global_H_mean'])
        z_peak      = float(row['z_peak'])
        n_bins_norm = float(row['n_bins_peak']) / NPERSEG
        t_center    = float(row['t_center'])

        try:
            d       = torch.load(file_path, map_location='cpu', weights_only=False)
            iq_full = d['x_iq'].float()
        except Exception:
            iq_full = torch.zeros(2, 1050000)

        max_idx = iq_full.shape[1]

        if self.split == 'train' and not self.eval_mode:
            jitter_ms = random.uniform(-1.0, 1.0)
            t_center  = max(0.0, t_center + jitter_ms)

        c_idx = int((t_center / 1000.0) * self.fs)
        start = c_idx - self.half_win
        end   = c_idx + self.half_win

        if start < 0:
            win = iq_full[:, 0:self.window_len]
        elif end > max_idx:
            win = iq_full[:, max_idx - self.window_len:max_idx]
        else:
            win = iq_full[:, start:end]

        power = win.pow(2).mean().clamp(min=1e-12).sqrt()
        win   = win / power

        if do_augment:
            target_snr = random.uniform(*self.augment_snr_range)
            win        = self._add_awgn(win, target_snr)
            z_peak     = 0.0
            n_bins_norm = 0.0

        phys = torch.tensor([global_nf, global_H, z_peak, n_bins_norm], dtype=torch.float32)

        # Retornamos el idx para poder trazar la loss por muestra en DivideMix
        return win, phys, torch.tensor(label, dtype=torch.float32), idx
