"""
Módulo de modelos de Deep Learning para detección de drones.

Modelos disponibles
-------------------
ComplexConv1DNet  : CV-CNN 1D de valores complejos (baseline, ~75% acc).
MaRNetFusion      : Bi-Mamba + ResNet + Deep Statistical Fusion (SotA, target >90% acc).
"""
from .cvcnn import ComplexConv1DNet, ComplexConv1d, CReLU, modulus
from .marnet_fusion import (
    MaRNetFusion,
    RFDroneDataset,
    SoftThresholdingBlock,
    ResNetBranch,
    MambaBranch,
    MLPBranch,
    PAM_Fusion,
    train_one_epoch,
    evaluate,
    mixup_batch,
    mixup_criterion,
)

__all__ = [
    # CV-CNN (baseline)
    "ComplexConv1DNet", "ComplexConv1d", "CReLU", "modulus",
    # MaRNet-Fusion (SotA multi-branch)
    "MaRNetFusion", "RFDroneDataset",
    "SoftThresholdingBlock", "ResNetBranch", "MambaBranch", "MLPBranch", "PAM_Fusion",
    "train_one_epoch", "evaluate", "mixup_batch", "mixup_criterion",
]
