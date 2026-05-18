"""
dataset_ht.py — Dataset Dual-Stream para Hard Test
====================================================
Idéntico al dataset_dual.py del V2.1 con una diferencia clave:
- La pool de AWGN augmentation también excluye Target=5 (TARGET_HELD_OUT).
  Esto garantiza que el dron Taranis NO entra por ninguna vía,
  ni directamente ni como señal maestra para degradación sintética.
"""
import os, random, math
import torch
from torch.utils.data import Dataset
import pandas as pd

TARGET_HELD_OUT = 5  # Taranis — excluido de train/val y de AWGN augmentation

class HardTestDataset(Dataset):
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

        if split == 'train':
            # CRÍTICO: excluir Target=5 también de la pool de augmentación
            self.high_snr_drone_idx = self.df[
                (self.df['label'] == 1) &
                (self.df['snr'] >= 10) &
                (self.df['is_fallback'] == False) &
                (self.df['target_multiclass'] != TARGET_HELD_OUT)   # <-- exclusión explícita
            ].index.tolist()
            print(f"  [HT-AUG] {len(self.high_snr_drone_idx)} muestras dron SNR>=10 para AWGN "
                  f"(Target={TARGET_HELD_OUT} excluido)")

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
        global_nf = float(row['global_nf'])
        global_H  = float(row['global_H_mean'])
        z_peak    = float(row['z_peak'])
        t_center  = float(row['t_center'])

        try:
            d = torch.load(file_path, map_location='cpu', weights_only=False)
            iq_full = d['x_iq'].float()
        except Exception:
            iq_full = torch.zeros(2, 1050000)

        max_idx = iq_full.shape[1]

        if self.split == 'train':
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

        # Normalización RMS
        power = win.pow(2).mean().clamp(min=1e-12).sqrt()
        win   = win / power

        if do_augment:
            target_snr = random.uniform(*self.augment_snr_range)
            win    = self._add_awgn(win, target_snr)
            z_peak = 0.0

        phys = torch.tensor([global_nf, global_H, z_peak], dtype=torch.float32)
        return win, phys, torch.tensor(label, dtype=torch.float32)
