"""
alumn_xin_cvcnn.py — Arquitectura Hibrida AlumnXin (CV-CNN 2D + Features Fisicas)
===================================================================================
Combina lo mejor de dos modelos:

  RAMA A — Backbone 2D de Xin et al. (2026):
      Entrada: espectrograma complejo [B, 2, 256, 256]
        Canal 0: log-PSD normalizado (STFT, 60dB clip)
        Canal 1: mapa de bordes Sobel (estructura temporal FHSS)
      4 bloques ComplexConv2d con CReLU y ComplexBatchNorm2d
      -> embedding visual [B, 9216]

  RAMA B — Features Fisicas de Burst (Alumno V1):
      8 features por muestra (dur_ms, z_peak, drop_b, n_act_burst,
                               global_nf, global_ns, global_H_mean, global_p75_act)
      BatchNorm1d + clamp [-5, 5]
      -> embedding fisico [B, 8]

  FUSION:
      concat [B, 9216 + 8] = [B, 9224]
      MLP: 9224 -> 512 -> ReLU -> Dropout(0.4) -> 256 -> ReLU -> 1 (logit)

La combinacion es sinergica:
  - La rama 2D ve la forma espectral global (cajitas FHSS visibles en el
    espectrograma), robusta a SNR alto y media.
  - La rama fisica aporta el contexto cuantitativo del burst (duracion,
    z_peak, ancho de banda) que discrimina FHSS vs BT/WiFi a bajo SNR
    donde el espectrograma se degrada en ruido plano.

Parametros totales: ~20.6M (backbone 2D) + ~5K (MLP fusion) ~ 20.6M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------- #
# Capas complejas (reutilizadas del modulo Xin)                                #
# ---------------------------------------------------------------------------- #

class ComplexConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=True):
        super().__init__()
        kw = dict(kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.w_rr = nn.Conv2d(in_ch, out_ch, **kw)
        self.w_ri = nn.Conv2d(in_ch, out_ch, **kw)
        self.w_ir = nn.Conv2d(in_ch, out_ch, **kw)
        self.w_ii = nn.Conv2d(in_ch, out_ch, **kw)
        if bias:
            self.b_r = nn.Parameter(torch.zeros(out_ch))
            self.b_i = nn.Parameter(torch.zeros(out_ch))
        else:
            self.register_parameter("b_r", None)
            self.register_parameter("b_i", None)

    def forward(self, x_r, x_i):
        out_r = self.w_rr(x_r) - self.w_ii(x_i)
        out_i = self.w_ri(x_r) + self.w_ir(x_i)
        if self.b_r is not None:
            out_r = out_r + self.b_r.view(1, -1, 1, 1)
            out_i = out_i + self.b_i.view(1, -1, 1, 1)
        return out_r, out_i


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, x_r, x_i):
        return self.bn_r(x_r), self.bn_i(x_i)


class CReLU(nn.Module):
    def forward(self, x_r, x_i):
        mag = torch.sqrt(x_r.pow(2) + x_i.pow(2) + 1e-8)
        scale = F.relu(mag) / mag
        return x_r * scale, x_i * scale


class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool_r = nn.MaxPool2d(kernel_size, stride)
        self.pool_i = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x_r, x_i):
        return self.pool_r(x_r), self.pool_i(x_i)


class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size=(6, 6)):
        super().__init__()
        self.pool_r = nn.AdaptiveAvgPool2d(output_size)
        self.pool_i = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x_r, x_i):
        return self.pool_r(x_r), self.pool_i(x_i)


class ComplexDropout2d(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x_r, x_i):
        if not self.training or self.p == 0.0:
            return x_r, x_i
        mask = torch.empty_like(x_r).bernoulli_(1 - self.p) / (1 - self.p)
        return x_r * mask, x_i * mask


class CVCNNBlock2D(nn.Module):
    """Bloque convolucional complejo 2D (identico a Xin et al.)."""
    def __init__(self, in_ch, out_ch, kernel_size=5, use_maxpool=True, dropout_rate=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv    = ComplexConv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.cbn     = ComplexBatchNorm2d(out_ch)
        self.crelu   = CReLU()
        self.pool    = ComplexMaxPool2d(2, 2) if use_maxpool else ComplexAdaptiveAvgPool2d((6, 6))
        self.dropout = ComplexDropout2d(p=dropout_rate)

    def forward(self, x_r, x_i):
        x_r, x_i = self.conv(x_r, x_i)
        x_r, x_i = self.cbn(x_r, x_i)
        x_r, x_i = self.crelu(x_r, x_i)
        x_r, x_i = self.pool(x_r, x_i)
        x_r, x_i = self.dropout(x_r, x_i)
        return x_r, x_i


# ---------------------------------------------------------------------------- #
# Modelo hibrido principal                                                      #
# ---------------------------------------------------------------------------- #

PHYS_DIM  = 8     # dur_ms, z_peak, drop_b, n_act_burst, global_nf, global_ns, global_H_mean, global_p75_act
CNN_FLAT  = 9216  # 256 canales x 6 x 6

class AlumnXinCVCNN(nn.Module):
    """
    Modelo hibrido que fusiona el backbone CV-CNN 2D de Xin et al. con las
    features fisicas de burst del pipeline Alumno V1.

    Entradas
    --------
    spec  : Tensor [B, 2, 256, 256] — espectrograma complejo (log-PSD + Sobel)
    feats : Tensor [B, 8]           — features fisicas ya normalizadas (z-score)

    Salida
    ------
    Tensor [B, 1] — logit binario (sin sigmoid; usar BCEWithLogitsLoss)
    """

    _DROPOUT_2D = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.0}

    def __init__(self, phys_dim: int = PHYS_DIM, kernel_size: int = 5,
                 fusion_hidden: int = 512, dropout_fusion: float = 0.4):
        super().__init__()

        # ── RAMA A: Backbone 2D (identico a XinCVCNN) ────────────────────────
        self.block1 = CVCNNBlock2D(  1,  64, kernel_size, use_maxpool=True,  dropout_rate=self._DROPOUT_2D[1])
        self.block2 = CVCNNBlock2D( 64, 128, kernel_size, use_maxpool=True,  dropout_rate=self._DROPOUT_2D[2])
        self.block3 = CVCNNBlock2D(128, 256, kernel_size, use_maxpool=True,  dropout_rate=self._DROPOUT_2D[3])
        self.block4 = CVCNNBlock2D(256, 256, kernel_size, use_maxpool=False, dropout_rate=self._DROPOUT_2D[4])

        # ── RAMA B: Features fisicas ──────────────────────────────────────────
        self.phys_bn = nn.BatchNorm1d(phys_dim)

        # ── FUSION: MLP conjunto ──────────────────────────────────────────────
        fusion_in = CNN_FLAT + phys_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fusion),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.BatchNorm1d(fusion_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fusion),
            nn.Linear(fusion_hidden // 2, 1),
        )

    def forward(self, spec: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # Rama A: espectrograma complejo -> embedding visual
        x_r = spec[:, 0:1, :, :]
        x_i = spec[:, 1:2, :, :]
        x_r, x_i = self.block1(x_r, x_i)
        x_r, x_i = self.block2(x_r, x_i)
        x_r, x_i = self.block3(x_r, x_i)
        x_r, x_i = self.block4(x_r, x_i)
        mag  = torch.sqrt(x_r.pow(2) + x_i.pow(2) + 1e-8)   # [B, 256, 6, 6]
        flat = mag.view(mag.size(0), -1)                       # [B, 9216]

        # Rama B: features fisicas normalizadas
        feats_norm = self.phys_bn(feats)                       # [B, 8]

        # Fusion
        fused = torch.cat([flat, feats_norm], dim=1)           # [B, 9224]
        return self.fusion_mlp(fused)                          # [B, 1]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
