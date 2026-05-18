"""
xin_cvcnn.py — Arquitectura CV-CNN 2D (Xin et al., 2026)
==========================================================
Implementa una Red Neuronal Convolucional de Valores Complejos bidimensional
para la clasificación binaria de señales RF de UAV.

Arquitectura (adaptada para clasificación binaria Dron vs. Ruido):
    ┌─────────────────────────────────────────────────────┐
    │ Entrada: [B, 2, H, W]                               │
    │   Canal 0 (Real):      log-PSD normalizado          │
    │   Canal 1 (Imag):      mapa de bordes Sobel         │
    ├─────────────────────────────────────────────────────┤
    │ Bloque 1: ComplexConv2d( 1→64, k=5) + CBN + CReLU  │
    │           + MaxPool2d(2×2) + Dropout(0.3)           │
    │ Bloque 2: ComplexConv2d(64→128,k=5) + CBN + CReLU  │
    │           + MaxPool2d(2×2) + Dropout(0.4)           │
    │ Bloque 3: ComplexConv2d(128→256,k=5)+ CBN + CReLU  │
    │           + MaxPool2d(2×2) + Dropout(0.5)           │
    │ Bloque 4: ComplexConv2d(256→256,k=5)+ CBN + CReLU  │
    │           + AdaptiveAvgPool2d(6×6) + Dropout(0.0)   │
    ├─────────────────────────────────────────────────────┤
    │ Módulo |z|: magnitud del número complejo            │
    │ Flatten: [B, 256×6×6] = [B, 9216]                  │
    ├─────────────────────────────────────────────────────┤
    │ FC: 9216 → 1024 → ReLU → Dropout(0.5)              │
    │      1024 →  512 → ReLU                             │
    │       512 →    1 (logit binario)                    │
    └─────────────────────────────────────────────────────┘

Capas complejas:
    - ComplexConv2d: Multiplicación compleja mediante cuatro ramas reales
      (cálculo de Wirtinger, nativo en PyTorch para backprop).
    - ComplexBatchNorm2d: Normalización independiente de parte real e
      imaginaria, preservando la distribución conjunta.
    - CReLU: ReLU aplicado a la magnitud |z|, preservando la fase.
      z_out = z · ReLU(|z|) / |z|

Dropouts reducidos ("Reduced Dropout"):
    El artículo identifica que aplicar Dropout tras la cuarta capa produce
    colapso del rendimiento. Esta implementación lo omite estrictamente en
    el bloque final (dropout_rate=0.0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Capas de Valores Complejos
# ─────────────────────────────────────────────────────────────────────────────

class ComplexConv2d(nn.Module):
    """
    Convolución 2D sobre números complejos mediante el cálculo de Wirtinger.

    La multiplicación compleja (a+jb)(c+jd) = (ac-bd) + j(ad+bc) se descompone
    en cuatro convoluciones reales separadas:
        Re_out = Conv(Re_in, W_rr) - Conv(Im_in, W_ii) + b_r
        Im_out = Conv(Re_in, W_ri) + Conv(Im_in, W_ir) + b_i

    Parámetros
    ----------
    in_channels  : Canales de entrada (en la parte real O imaginaria).
    out_channels : Canales de salida.
    kernel_size  : Tamaño del kernel convolucional.
    stride       : Paso de la convolución.
    padding      : Relleno simétrico.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        kw = dict(kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.w_rr = nn.Conv2d(in_channels, out_channels, **kw)
        self.w_ri = nn.Conv2d(in_channels, out_channels, **kw)
        self.w_ir = nn.Conv2d(in_channels, out_channels, **kw)
        self.w_ii = nn.Conv2d(in_channels, out_channels, **kw)

        if bias:
            self.b_r = nn.Parameter(torch.zeros(out_channels))
            self.b_i = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("b_r", None)
            self.register_parameter("b_i", None)

    def forward(
        self, x_r: torch.Tensor, x_i: torch.Tensor
    ):
        out_r = self.w_rr(x_r) - self.w_ii(x_i)
        out_i = self.w_ri(x_r) + self.w_ir(x_i)
        if self.b_r is not None:
            out_r = out_r + self.b_r.view(1, -1, 1, 1)
            out_i = out_i + self.b_i.view(1, -1, 1, 1)
        return out_r, out_i


class ComplexBatchNorm2d(nn.Module):
    """
    Normalización de lote aplicada de forma independiente a la parte real
    e imaginaria, preservando la consistencia de las distribuciones.

    Nota: La normalización de lote compleja covariante (Trabelsi et al., 2018)
    requiere la estimación de la matriz de covarianza [2×2]; aquí se adopta
    la aproximación simplificada de Xin et al. (BN independiente por parte),
    computacionalmente equivalente cuando las partes real e imaginaria son
    estadísticamente independientes.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor):
        return self.bn_r(x_r), self.bn_i(x_i)


class CReLU(nn.Module):
    """
    Complex ReLU: activa en función de la magnitud, preservando la fase.

        |z| = sqrt(Re² + Im² + ε)
        z_out = z · max(0, |z|) / |z|
              = z · ReLU(|z|) / |z|

    Esto es equivalente a proyectar el número complejo al semiplano |z| ≥ 0,
    preservando la dirección de fase intacta cuando la magnitud es positiva.
    """

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor):
        mag = torch.sqrt(x_r.pow(2) + x_i.pow(2) + 1e-8)
        scale = F.relu(mag) / mag
        return x_r * scale, x_i * scale


class ComplexMaxPool2d(nn.Module):
    """MaxPooling 2D independiente en parte real e imaginaria."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool_r = nn.MaxPool2d(kernel_size, stride)
        self.pool_i = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor):
        return self.pool_r(x_r), self.pool_i(x_i)


class ComplexAdaptiveAvgPool2d(nn.Module):
    """Average Pooling Adaptativo 2D independiente en parte real e imaginaria."""

    def __init__(self, output_size=(6, 6)):
        super().__init__()
        self.pool_r = nn.AdaptiveAvgPool2d(output_size)
        self.pool_i = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor):
        return self.pool_r(x_r), self.pool_i(x_i)


class ComplexDropout2d(nn.Module):
    """
    Dropout 2D aplicado de forma sincronizada a parte real e imaginaria.
    Usar la misma máscara preserva la coherencia espacial del número complejo.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor):
        if not self.training or self.p == 0.0:
            return x_r, x_i
        # Máscara binaria compartida
        mask = torch.empty_like(x_r).bernoulli_(1 - self.p) / (1 - self.p)
        return x_r * mask, x_i * mask


# ─────────────────────────────────────────────────────────────────────────────
# Bloque convolucional complejo
# ─────────────────────────────────────────────────────────────────────────────

class CVCNNBlock2D(nn.Module):
    """
    Bloque convolucional complejo 2D estándar del artículo.

    Composición: ComplexConv2d → CBN → CReLU → Pool → Dropout

    Parámetros
    ----------
    in_channels   : Canales de entrada.
    out_channels  : Canales de salida.
    kernel_size   : Tamaño del kernel (Xin et al. óptimo: k=5).
    use_maxpool   : True = MaxPool(2×2); False = AdaptiveAvgPool(6×6).
    dropout_rate  : Tasa de Dropout. 0.0 = sin dropout (bloque final).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        use_maxpool: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.cbn  = ComplexBatchNorm2d(out_channels)
        self.crelu = CReLU()

        if use_maxpool:
            self.pool = ComplexMaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = ComplexAdaptiveAvgPool2d(output_size=(6, 6))

        self.dropout = ComplexDropout2d(p=dropout_rate)

    def forward(self, x_r: torch.Tensor, x_i: torch.Tensor):
        x_r, x_i = self.conv(x_r, x_i)
        x_r, x_i = self.cbn(x_r, x_i)
        x_r, x_i = self.crelu(x_r, x_i)
        x_r, x_i = self.pool(x_r, x_i)
        x_r, x_i = self.dropout(x_r, x_i)
        return x_r, x_i


# ─────────────────────────────────────────────────────────────────────────────
# Arquitectura principal
# ─────────────────────────────────────────────────────────────────────────────

class XinCVCNN(nn.Module):
    """
    CV-CNN 2D para clasificación binaria Dron (1) vs. Ruido (0).

    Adaptación directa de Xin et al. (2026) al dominio binario de NoisyUAV.
    La entrada tiene 1 canal complejo (en lugar de los múltiples canales del
    espectrograma RGB del paper original), lo que mantiene la comparabilidad
    arquitectónica manteniendo la fidelidad metodológica.

    Entrada:  [B, 2, spec_h, spec_w]
    Salida:   [B, 1] — logit (sin sigmoid, compatible con BCEWithLogitsLoss)

    Parámetros
    ----------
    num_classes : Dimensión de la salida. Default: 1 (clasificación binaria).
    kernel_size : Tamaño de kernel. Óptimo según ablación del paper: k=5.
    """

    # Dropout reducido: {bloque: tasa} — Bloque 4 sin dropout (crítico)
    _DROPOUT = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.0}

    def __init__(self, num_classes: int = 1, kernel_size: int = 5):
        super().__init__()

        # Extractor de características (4 bloques complejos)
        # in_channels=1: cada bloque recibe 1 canal complejo (real + imag separados)
        self.block1 = CVCNNBlock2D(  1,  64, kernel_size, use_maxpool=True,  dropout_rate=self._DROPOUT[1])
        self.block2 = CVCNNBlock2D( 64, 128, kernel_size, use_maxpool=True,  dropout_rate=self._DROPOUT[2])
        self.block3 = CVCNNBlock2D(128, 256, kernel_size, use_maxpool=True,  dropout_rate=self._DROPOUT[3])
        self.block4 = CVCNNBlock2D(256, 256, kernel_size, use_maxpool=False, dropout_rate=self._DROPOUT[4])

        # Clasificador Fully Connected
        # 256 canales × 6 × 6 = 9216 (igual que el paper original)
        self.fc = nn.Sequential(
            nn.Linear(9216, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x : Tensor [B, 2, H, W]
              Canal 0 → parte Real (log-PSD)
              Canal 1 → parte Imaginaria (Sobel)

        Retorna
        -------
        Tensor [B, 1] con logits (sin sigmoid).
        """
        x_r = x[:, 0:1, :, :]   # [B, 1, H, W]
        x_i = x[:, 1:2, :, :]   # [B, 1, H, W]

        x_r, x_i = self.block1(x_r, x_i)
        x_r, x_i = self.block2(x_r, x_i)
        x_r, x_i = self.block3(x_r, x_i)
        x_r, x_i = self.block4(x_r, x_i)

        # Módulo |z|: colapsamos la representación compleja al dominio real
        # para la clasificación. Esto es equivalente a la "conversión de
        # amplitud" descrita en Xin et al. antes de la capa FC.
        mag = torch.sqrt(x_r.pow(2) + x_i.pow(2) + 1e-8)   # [B, 256, 6, 6]
        flat = mag.view(mag.size(0), -1)                      # [B, 9216]

        return self.fc(flat)   # [B, 1]

    def count_parameters(self) -> int:
        """Retorna el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
