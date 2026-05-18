"""
model.py — DualStreamCVCNN V2.1 (Hard Test)
Idéntico al original v2_1. No se modifica la arquitectura.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, iq_emb, psd_emb):
        concat = torch.cat([iq_emb, psd_emb], dim=1)
        weights = self.attn(concat)
        w_iq  = weights[:, 0].unsqueeze(1)
        w_psd = weights[:, 1].unsqueeze(1)
        fused = w_iq * iq_emb + w_psd * psd_emb
        return fused, weights


class DualStreamCVCNN(nn.Module):
    def __init__(self, target_len=131072, n_fft_bins=2048, dropout=0.3):
        super().__init__()
        self.target_len = target_len
        self.pool_size  = target_len // n_fft_bins

        # Stream 1 — Dominio Temporal (IQ)
        self.iq_cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=128, stride=4, padding=64),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )

        # Stream 2 — Dominio Frecuencial (PSD)
        self.psd_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )

        self.iq_proj   = nn.Linear(256, 128)
        self.fusion    = AttentionFusion(embed_dim=128)
        self.phys_mlp  = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(128 + 16, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.register_buffer("hann_window", torch.hann_window(target_len))

    def compute_psd(self, iq):
        B, C, L = iq.shape
        if L < self.target_len:
            pad = torch.zeros(B, C, self.target_len - L, device=iq.device)
            iq_pad = torch.cat([iq, pad], dim=2)
        else:
            iq_pad = iq[:, :, :self.target_len]
        iq_complex = torch.complex(iq_pad[:, 0, :], iq_pad[:, 1, :])
        windowed   = iq_complex * self.hann_window
        fft_out    = torch.fft.fftshift(torch.fft.fft(windowed), dim=1)
        psd        = torch.abs(fft_out) ** 2
        psd_pooled = F.avg_pool1d(psd.unsqueeze(1), kernel_size=self.pool_size,
                                  stride=self.pool_size)
        return 10 * torch.log10(psd_pooled + 1e-12)

    def forward(self, iq, phys_feats):
        B = iq.size(0)
        psd      = self.compute_psd(iq)
        iq_feat  = self.iq_cnn(iq).view(B, -1)
        iq_emb   = self.iq_proj(iq_feat)
        psd_emb  = self.psd_cnn(psd).view(B, -1)
        fused, attn_weights = self.fusion(iq_emb, psd_emb)
        phys_emb = self.phys_mlp(phys_feats)
        concat   = torch.cat([fused, phys_emb], dim=1)
        logits   = self.classifier(concat)
        return logits, attn_weights
