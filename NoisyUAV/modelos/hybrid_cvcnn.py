"""
hybrid_cvcnn.py
===============
HybridCVCNN — Arquitectura híbrida que fusiona:
  1. Backbone CV-CNN (Complex-Valued 1D CNN) sobre crops IQ largos
  2. Features físicas del detector de entropía Shannon (12 dimensiones)

La fusión produce representaciones que combinan la información temporal
de la waveform IQ (patrones de fase FHSS) con estadísticos globales de
la transmisión (duración del burst, bins activos, entropía espectral).

Diagrama de flujo:
    IQ crop [B, 2, N]
        │
    CV-CNN backbone                 Physical Features [B, 12]
    4×ComplexConvBlock                    │
    + modulus + AvgPool              BatchNorm1d(12)
    + Linear(8192→256)                    │
        │  [B, 256]                       │ [B, 12]
        └──────────────── cat ────────────┘
                               │
                           [B, 268]
                               │
                   MLP: 268 → 256 → 128 → 1
                               │
                           logit [B, 1]
                    (BCEWithLogitsLoss en train)

Pesos de referencia:
    CV-CNN backbone   : ~8.5 M parámetros (compatible con checkpoints anteriores)
    Fusion MLP        : ~0.1 M parámetros
    TOTAL             : ~8.6 M parámetros
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from NoisyUAV.modelos.cvcnn import (
    ComplexConvBlock,
    ComplexConv1d,
    CReLU,
    modulus,
    ComplexBatchNorm1d,
)

PHYS_DIM   = 12    # dimensión del vector de features físicas
CNN_EMBED  = 256   # dimensión del embedding del backbone
FUSE_DIM   = CNN_EMBED + PHYS_DIM  # 268


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE CV-CNN (sin clasificador final)
# ─────────────────────────────────────────────────────────────────────────────

class CVCNNBackbone(nn.Module):
    """
    Extractor de features del modelo Complex-Valued CNN.
    Igual que ComplexConv1DNet pero devuelve un vector de embedding
    de dimensión `embed_dim` en lugar de logits de clase.

    Arquitectura idéntica al baseline hasta el AdaptiveAvgPool1d,
    luego un Linear(8192, embed_dim) en lugar del clasificador completo.
    """

    def __init__(self, embed_dim: int = CNN_EMBED, pool_size: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.pool_size = pool_size

        # 4 bloques convolucionales complejos (idénticos al baseline)
        self.features = nn.Sequential(
            ComplexConvBlock(  1,  32, kernel_size=31, stride=2, padding=15),
            ComplexConvBlock( 32,  64, kernel_size=15, stride=2, padding=7),
            ComplexConvBlock( 64, 128, kernel_size= 7, stride=2, padding=3),
            ComplexConvBlock(128, 128, kernel_size= 3, stride=1, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(pool_size)

        # Cabeza de embedding
        fc_in = 128 * pool_size   # por ejemplo: 128 * 32 = 4096
        self.embed_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, embed_dim),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2, N]   → embedding: [B, embed_dim]
        N puede ser variable (AdaptiveAvgPool normaliza la longitud).
        """
        z = self.features(x)       # [B, 256, N']  (2·128 canales complejos)
        z = modulus(z)             # [B, 128, N']  (módulo → dominio real)
        z = self.adaptive_pool(z)  # [B, 128, pool_size]
        return self.embed_head(z)  # [B, embed_dim]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# MODELO PRINCIPAL: HybridCVCNN
# ─────────────────────────────────────────────────────────────────────────────

class HybridCVCNN(nn.Module):
    """
    Detector híbrido: CV-CNN backbone + Physical Feature Fusion.

    Entradas
    --------
    iq_crop   : Tensor [B, 2, N]   — crop IQ (N recomendado: 131072 ≈ 9.4ms)
    phys_feat : Tensor [B, 12]     — features físicas del detector de entropía
                                     (ya PRE-NORMALIZADAS exteriormente: z-score)

    Salida
    ------
    logit : Tensor [B, 1]
        Sin sigmoid. Usar:
          - BCEWithLogitsLoss durante entrenamiento
          - torch.sigmoid(logit) > threshold para inferencia

    Parámetros
    ----------
    phys_dim    : dimensión del vector de features físicas    (default: 12)
    cnn_embed   : dimensión del embedding del backbone        (default: 256)
    pool_size   : output del AdaptiveAvgPool dentro del CNN   (default: 32)
    hidden_dim  : capa oculta del MLP de fusión              (default: 256)
    dropout_cnn : dropout del backbone CV-CNN                 (default: 0.3)
    dropout_fuse: dropout del MLP de fusión                  (default: 0.4)
    """

    def __init__(
        self,
        phys_dim:     int   = PHYS_DIM,
        cnn_embed:    int   = CNN_EMBED,
        pool_size:    int   = 32,
        hidden_dim:   int   = 256,
        dropout_cnn:  float = 0.3,
        dropout_fuse: float = 0.4,
    ):
        super().__init__()
        self.phys_dim   = phys_dim
        self.cnn_embed  = cnn_embed

        # ── Rama 1: CV-CNN backbone ─────────────────────────────────────────
        self.backbone = CVCNNBackbone(
            embed_dim=cnn_embed,
            pool_size=pool_size,
            dropout=dropout_cnn,
        )

        # ── Rama 2: Normalización de features físicas ───────────────────────
        # BatchNorm sobre las 12 features (learn scale+shift per feature)
        self.phys_bn = nn.BatchNorm1d(phys_dim)

        # ── Fusión y clasificación ──────────────────────────────────────────
        fuse_in = cnn_embed + phys_dim  # 256 + 12 = 268
        self.classifier = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_fuse),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_fuse / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_classifier()

    def forward(
        self,
        iq_crop:   torch.Tensor,
        phys_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        iq_crop   : [B, 2, N]  →  CNN embedding  [B, cnn_embed]
        phys_feat : [B, 12]    →  norm features  [B, 12]
                                →  concat        [B, fuse_in]
                                →  MLP           [B, 1]
        """
        # Embeddings del backbone CV-CNN
        cnn_emb = self.backbone(iq_crop)          # [B, cnn_embed]

        # Normalización de features físicas
        phys_n  = self.phys_bn(phys_feat)         # [B, 12]

        # Fusión y clasificación
        fused = torch.cat([cnn_emb, phys_n], dim=1)  # [B, fuse_in]
        return self.classifier(fused)                  # [B, 1]

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        backbone_p  = self.backbone.count_parameters()
        total_p     = self.count_parameters()
        fusion_p    = total_p - backbone_p
        print("=" * 60)
        print("  HybridCVCNN — Physical-Feature-Fused Drone Detector")
        print("-" * 60)
        print(f"  CV-CNN Backbone    : {backbone_p:>12,}")
        print(f"  Physical BN + MLP  : {fusion_p:>12,}")
        print(f"> TOTAL              : {total_p:>12,}")
        print(f"  IQ crop input      : [B, 2, N]  (N variable, reco. 131072)")
        print(f"  Physical feat input: [B, {self.phys_dim}]")
        print(f"  CNN embed dim      : {self.cnn_embed}")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class HybridDataset(torch.utils.data.Dataset):
    """
    Dataset para HybridCVCNN.

    Devuelve por cada muestra:
        (iq_crop [2, crop_len], phys_feat [12], label int)

    Estrategia de crop: aleatorio durante entrenamiento, centrado en validación.

    Las features físicas se cargan desde la caché pre-computada (no se recomputan
    en tiempo real para no duplicar la STFT dentro del DataLoader).
    """

    def __init__(
        self,
        df,
        cache: dict,
        crop_len: int = 131072,
        augment:  bool = False,
        phys_mean: "Optional[torch.Tensor]" = None,
        phys_std:  "Optional[torch.Tensor]" = None,
    ):
        """
        Parámetros
        ----------
        df        : DataFrame con columnas ['filepath', 'label', 'snr', 'grupo']
        cache     : dict {filename: np.array [12]}  — features pre-computadas
        crop_len  : longitud del crop IQ para la rama CNN  (≈9.4ms @ 14MHz)
        augment   : si True aplica rotación de fase aleatoria y ruido aditivo leve
        phys_mean : [12] media para normalizar features físicas (de train set)
        phys_std  : [12] std  para normalizar features físicas (de train set)
        """
        self.df       = df.reset_index(drop=True)
        self.cache    = cache
        self.crop_len = crop_len
        self.augment  = augment

        # Normalización de features físicas (z-score)
        if phys_mean is not None:
            self.phys_mean = torch.as_tensor(phys_mean, dtype=torch.float32)
            self.phys_std  = torch.as_tensor(phys_std,  dtype=torch.float32)
        else:
            self.phys_mean = torch.zeros(PHYS_DIM)
            self.phys_std  = torch.ones(PHYS_DIM)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ── Cargar IQ ────────────────────────────────────────────────────────
        d  = torch.load(row['filepath'], map_location='cpu', weights_only=False)
        iq = d['x_iq'].float()   # [2, 1048576]

        # ── Crop IQ ──────────────────────────────────────────────────────────
        L = iq.shape[1]
        if L <= self.crop_len:
            pad = torch.zeros(2, self.crop_len - L)
            iq_crop = torch.cat([iq, pad], dim=1)
        else:
            if self.augment:
                start = torch.randint(0, L - self.crop_len, (1,)).item()
            else:
                start = (L - self.crop_len) // 2   # crop centrado
            iq_crop = iq[:, start:start + self.crop_len]

        # ── Normalización RMS (OBLIGATORIA) ───────────────────────────────────
        # Elimina la dependencia de amplitud absoluta (que varía >10^5x entre SNRs).
        # Preserva todo lo demás: fase, estructura temporal, patrón FHSS.
        power = iq_crop.pow(2).mean().clamp(min=1e-12).sqrt()
        iq_crop = iq_crop / power

        # ── Augmentación ─────────────────────────────────────────────────────
        if self.augment:
            # Rotación de fase aleatoria: z → z·e^(jθ)
            theta = torch.rand(1) * 2 * 3.141592653589793
            c, s  = theta.cos(), theta.sin()
            re, im = iq_crop[0], iq_crop[1]
            iq_crop = torch.stack([re * c - im * s, re * s + im * c])

            # Ruido AWGN adaptativo (std = 1-3% de la amplitud de la señal)
            amp   = iq_crop.norm(dim=0).mean().clamp(min=1e-6)
            noise = torch.randn_like(iq_crop) * amp * torch.empty(1).uniform_(0.01, 0.03)
            iq_crop = iq_crop + noise

        # ── Features físicas ─────────────────────────────────────────────────
        key = Path(str(row['filepath'])).name

        raw  = self.cache.get(key, None)
        if raw is None:
            phys = torch.zeros(PHYS_DIM)
        else:
            phys = torch.from_numpy(raw.copy())

        # Z-score normalización
        phys = (phys - self.phys_mean) / (self.phys_std + 1e-8)
        phys = torch.clamp(phys, -5.0, 5.0)   # clamp extremos

        label = int(row['label'])
        return iq_crop, phys, label


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch_hybrid(
    model:     HybridCVCNN,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler=None,
    mixup_alpha: float = 0.3,
    mixup_prob:  float = 0.5,
    grad_clip:   float = 2.0,
) -> dict:
    """
    Entrena un epoch del HybridCVCNN.
    Soporta AMP (scaler) y MixUp sobre los dos tipos de entrada.

    Retorna
    -------
    dict con 'loss' y 'acc' del epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for iq_crop, phys, labels in loader:
        iq_crop = iq_crop.to(device)
        phys    = phys.to(device)
        labels  = labels.to(device).float()

        # MixUp
        do_mixup = (torch.rand(1).item() < mixup_prob) and (mixup_alpha > 0)
        if do_mixup:
            lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
            idx = torch.randperm(iq_crop.size(0), device=device)
            iq_crop = lam * iq_crop + (1 - lam) * iq_crop[idx]
            phys    = lam * phys    + (1 - lam) * phys[idx]
            labels_b = labels[idx]
        else:
            lam, labels_b = 1.0, labels

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logit = model(iq_crop, phys)              # [B, 1]
                if do_mixup:
                    loss = lam * criterion(logit.squeeze(1), labels) + \
                           (1 - lam) * criterion(logit.squeeze(1), labels_b)
                else:
                    loss = criterion(logit.squeeze(1), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logit = model(iq_crop, phys)
            if do_mixup:
                loss = lam * criterion(logit.squeeze(1), labels) + \
                       (1 - lam) * criterion(logit.squeeze(1), labels_b)
            else:
                loss = criterion(logit.squeeze(1), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logit.squeeze(1)) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total   += labels.size(0)

    return {
        'loss': total_loss / max(len(loader), 1),
        'acc':  correct / max(total, 1),
    }


@torch.no_grad()
def evaluate_hybrid(
    model:     HybridCVCNN,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Evalúa el HybridCVCNN en un DataLoader.

    Retorna
    -------
    dict con: loss, acc, precision, recall, f1, specificity, probs, labels, preds
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    model.eval()
    all_probs, all_labels, all_preds = [], [], []
    total_loss = 0.0

    for iq_crop, phys, labels in loader:
        iq_crop = iq_crop.to(device)
        phys    = phys.to(device)
        labels_d = labels.to(device).float()

        logit = model(iq_crop, phys)
        loss  = criterion(logit.squeeze(1), labels_d)
        total_loss += loss.item()

        probs = torch.sigmoid(logit.squeeze(1))
        preds = (probs > threshold).long()

        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())

    import numpy as np
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc  = (y_true == y_pred).mean()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn   = ((y_true == 0) & (y_pred == 0)).sum()
    fp   = ((y_true == 0) & (y_pred == 1)).sum()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'loss':        total_loss / max(len(loader), 1),
        'acc':         float(acc),
        'precision':   float(prec),
        'recall':      float(rec),
        'f1':          float(f1),
        'specificity': float(spec),
        'probs':       y_prob,
        'labels':      y_true,
        'preds':       y_pred,
    }
