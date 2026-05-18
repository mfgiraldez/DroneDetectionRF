import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, r'c:\repos\DroneDetectionRF')

from NoisyUAV.modelo_alumn_xin.alumn_xin_cvcnn import CVCNNBlock2D, CNN_FLAT, PHYS_DIM

class MILAlumnXinCVCNN(nn.Module):
    """
    Arquitectura Híbrida MIL-Alumn-Xin
    Integra Multiple Instance Learning (Max-Logit Pooling).
    """
    _DROPOUT_2D = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.0}

    def __init__(self, phys_dim: int = PHYS_DIM, kernel_size: int = 5,
                 fusion_hidden: int = 512, dropout_fusion: float = 0.4):
        super().__init__()
        
        # RAMA A: Backbone 2D (Reutilizado de Xin et al.)
        self.block1 = CVCNNBlock2D(  1,  64, kernel_size, use_maxpool=True,  dropout_rate=self._DROPOUT_2D[1])
        self.block2 = CVCNNBlock2D( 64, 128, kernel_size, use_maxpool=True,  dropout_rate=self._DROPOUT_2D[2])
        self.block3 = CVCNNBlock2D(128, 256, kernel_size, use_maxpool=True,  dropout_rate=self._DROPOUT_2D[3])
        self.block4 = CVCNNBlock2D(256, 256, kernel_size, use_maxpool=False, dropout_rate=self._DROPOUT_2D[4])

        # RAMA B: Features físicas
        self.phys_bn = nn.BatchNorm1d(phys_dim)

        # FUSIÓN: MLP Clasificador
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

    def forward_instance(self, spec: torch.Tensor, feats: torch.Tensor):
        # Rama A: Espectrograma
        x_r, x_i = spec[:, 0:1, :, :], spec[:, 1:2, :, :]
        x_r, x_i = self.block1(x_r, x_i)
        x_r, x_i = self.block2(x_r, x_i)
        x_r, x_i = self.block3(x_r, x_i)
        x_r, x_i = self.block4(x_r, x_i)
        mag  = torch.sqrt(x_r.pow(2) + x_i.pow(2) + 1e-8)
        flat = mag.view(mag.size(0), -1)

        # Rama B: Features físicas
        feats_norm = self.phys_bn(feats)

        # Fusión
        fused = torch.cat([flat, feats_norm], dim=1)
        
        # Logit Dron (Clasificación Principal)
        logit_dron = self.fusion_mlp(fused)
        
        return logit_dron

    def forward(self, spec: torch.Tensor, feats: torch.Tensor):
        """
        Soporta Single-Instance (modo estándar) y Multiple-Instance (modo MIL).
        - spec dim 4: [B, 2, H, W] -> Modo estándar
        - spec dim 5: [B, N, 2, H, W] -> Modo MIL (Max-Logit Pooling)
        """
        if spec.dim() == 5:
            B, N, C, H, W = spec.shape
            spec_flat = spec.view(B * N, C, H, W)
            feats_flat = feats.view(B * N, -1)
            
            logit_dron_flat = self.forward_instance(spec_flat, feats_flat)
            
            logit_dron = logit_dron_flat.view(B, N)
            
            # Max-Logit Pooling (MIL)
            logit_dron_pool, _ = torch.max(logit_dron, dim=1, keepdim=True)
            
            return logit_dron_pool
        else:
            return self.forward_instance(spec, feats)
