import os
import random
import logging
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import sys
sys.path.insert(0, r'c:\repos\DroneDetectionRF')

from NoisyUAV.scripts.SNR_estimation import add_awgn_noise

FS = 14_000_000
TARGET_LEN = 131_072
MAX_WINDOWS = 16

PHYS_COLS = [
    "dur_ms", "z_peak", "drop_b", "n_act_burst",
    "global_nf", "global_ns", "global_H_mean", "global_p75_act",
]

def batched_burst_iq_to_xin_tensor(
    iq_crop: torch.Tensor, # [N, 2, L]
    nfft: int = 1024,
    hop_length: int = 512,
    spec_h: int = 256,
    spec_w: int = 256,
    db_clip: float = 60.0,
) -> torch.Tensor:
    """Versión vectorizada para calcular N ventanas a la vez (10x más rápido)."""
    if iq_crop.dim() == 2:
        iq_crop = iq_crop.unsqueeze(0)
        
    N = iq_crop.shape[0]
    rms = iq_crop.pow(2).mean(dim=[1, 2], keepdim=True).clamp(min=1e-12).sqrt()
    iq = iq_crop / rms
    
    sig_complex = torch.complex(iq[:, 0, :], iq[:, 1, :]) # [N, L]
    window = torch.hann_window(nfft, device=iq.device)
    
    stft = torch.stft(
        sig_complex, n_fft=nfft, hop_length=hop_length,
        win_length=nfft, window=window,
        center=False, return_complex=True, onesided=False,
    ) # [N, F, T]
    
    psd_db = 10.0 * torch.log10(stft.abs().pow(2) + 1e-12)
    
    psd_max = psd_db.amax(dim=(1, 2), keepdim=True)
    psd_db = torch.max(psd_db, psd_max - db_clip)
    
    psd_min = psd_db.amin(dim=(1, 2), keepdim=True)
    log_psd = (psd_db - psd_min) / (psd_max - psd_min).clamp(min=1e-8) # [N, F, T]
    
    log_psd_r = F.interpolate(
        log_psd.unsqueeze(1).float(),
        size=(spec_h, spec_w), mode="bilinear", align_corners=False,
    ) # [N, 1, H, W]
    
    img_pad = F.pad(log_psd_r, (1, 1, 1, 1), mode="reflect")
    Kx = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=iq.device).view(1, 1, 3, 3)
    Ky = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], device=iq.device).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(img_pad, Kx)
    grad_y = F.conv2d(img_pad, Ky)
    sobel = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8) # [N, 1, H, W]
    
    s_min = sobel.amin(dim=(1, 2, 3), keepdim=True)
    s_max = sobel.amax(dim=(1, 2, 3), keepdim=True)
    sobel_norm = (sobel - s_min) / (s_max - s_min + 1e-8)
    
    return torch.cat([log_psd_r, sobel_norm], dim=1) # [N, 2, H, W]

class MILAlumnXinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, phys_mean=None, phys_std=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        
        # Filtro inicial: todas las muestras
        self.active_indices = list(range(len(self.df)))
        self.phase = 3
        self.augment = False
        self.use_mil = True
        
        # Features Físicas
        phys_raw = torch.tensor(self.df[PHYS_COLS].values, dtype=torch.float32)
        if phys_mean is not None and phys_std is not None:
            self.phys_mean, self.phys_std = phys_mean, phys_std
        else:
            self.phys_mean = phys_raw.mean(dim=0)
            self.phys_std  = phys_raw.std(dim=0).clamp(min=1e-8)
        self.phys_feats = ((phys_raw - self.phys_mean) / self.phys_std).clamp(-5., 5.)

    def set_phase(self, phase: int):
        if not self.is_train: return
        self.phase = phase
        if phase == 1:
            self.active_indices = self.df[self.df["snr"] >= 10.0].index.tolist()
            self.augment = False
        elif phase == 2:
            self.active_indices = self.df[self.df["snr"] >= 0.0].index.tolist()
            self.augment = True
        else:
            self.active_indices = list(range(len(self.df)))
            self.augment = True
        logging.info(f"Dataset Fase {phase}: {len(self.active_indices)} muestras activas.")

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.active_indices[idx]
        row = self.df.iloc[real_idx]
        
        pl = float(row["pseudo_label"])
        pl = 0.0 if pl < 0.0 else (1.0 if pl > 1.0 else pl)
        label = torch.tensor([pl], dtype=torch.float32)
        prob_ia = torch.tensor([float(row.get("prob_ia", 1.0))], dtype=torch.float32).clamp(0.0, 1.0)
        
        feats = self.phys_feats[real_idx]
        full_path = self.data_dir / row["file_path"]
        
        d = torch.load(full_path, map_location="cpu", weights_only=False)
        iq = d["x_iq"].float()
        L = iq.shape[1]
        
        is_dummy = bool(row.get("fallback", False))
        
        if is_dummy and self.use_mil:
            windows_iq = []
            step = TARGET_LEN // 2
            for start in range(0, L - TARGET_LEN + 1, step):
                w = iq[:, start:start+TARGET_LEN]
                windows_iq.append(w)
            
            if not windows_iq:
                pad_len = max(0, TARGET_LEN - L)
                w = torch.cat([iq, torch.zeros(2, pad_len)], dim=1)[:, :TARGET_LEN]
                windows_iq.append(w)
                
            if len(windows_iq) > MAX_WINDOWS:
                windows_iq = windows_iq[:MAX_WINDOWS]
                
            # Batch STFT for incredible speedup
            w_tensor = torch.stack(windows_iq, dim=0) # [N, 2, L]
            spec = batched_burst_iq_to_xin_tensor(w_tensor) # [N, 2, 256, 256]
            feats = feats.unsqueeze(0).repeat(spec.shape[0], 1) # [N, 8]
        else:
            start_idx = max(0, int(float(row["t_start"]) / 1000.0 * FS))
            end_idx   = min(L, int(float(row["t_end"])   / 1000.0 * FS))
            if start_idx >= end_idx:
                end_idx = min(L, start_idx + 1024)

            iq_crop = iq[:, start_idx:end_idx]
            
            if self.augment and float(row["snr"]) >= 20.0 and random.random() < 0.5:
                target_snr = -5 if self.phase == 2 else random.choice([-10, -15, -20])
                iq_crop = add_awgn_noise(iq_crop, target_snr)

            cl = iq_crop.shape[1]
            if cl < TARGET_LEN:
                iq_crop = torch.cat([iq_crop, torch.zeros(2, TARGET_LEN - cl)], dim=1)
            else:
                iq_crop = iq_crop[:, :TARGET_LEN]

            spec = batched_burst_iq_to_xin_tensor(iq_crop.unsqueeze(0)) # [1, 2, 256, 256]
            feats = feats.unsqueeze(0) # [1, 8]
            
        return {
            "spec": spec,
            "feats": feats,
            "label": label,
            "prob_ia": prob_ia,
            "snr": float(row["snr"]),
            "fallback": is_dummy
        }
