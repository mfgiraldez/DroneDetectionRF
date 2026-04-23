"""
cvcnn.py — Complex-Valued 1D CNN para detección de ráfagas de drones en señales I/Q
=====================================================================================

Implementa una red neuronal convolucional de valores complejos (CV-CNN) basada en
la regla de multiplicación de Wirtinger para señales I/Q en banda base.

Referencia principal
--------------------
    Bassey, Qian & Li (2021). "A Survey of Complex-Valued Neural Networks".
    arXiv:2101.12249 — §3.1 (ComplexConv), §4.2 (CReLU), §4.3 (BN complejo)

Motivación física
-----------------
Un salto FHSS introduce una rotación de fase en el plano complejo I/Q. Una CNN real
operando sólo sobre el módulo |IQ| pierde esta información. Un filtro complejo
W ∈ C^k aplicado sobre z = I + jQ aprende a detectar esas rotaciones vía:

    Re(W·z) = Re(W)·I − Im(W)·Q
    Im(W·z) = Re(W)·Q + Im(W)·I

La red aprende Re(W) e Im(W) de forma independiente, pero el acoplamiento cruzado
(la resta y la suma) está hardcodeado en la arquitectura, forzando la coherencia
matemática de la multiplicación compleja.

Arquitectura (ComplexConv1DNet)
-------------------------------
    Input [B, 2, N]   →  z = I + jQ   (N = longitud variable del crop)
    │
    ├── ComplexConv1d( 1→ 32, k=31, stride=2) + CReLU
    ├── ComplexConv1d(32→ 64, k=15, stride=2) + CReLU
    ├── ComplexConv1d(64→128, k= 7, stride=2) + CReLU
    ├── ComplexConv1d(128→128,k= 3, stride=1) + CReLU
    │
    ├── Módulo: |z| = sqrt(Re² + Im²)  →  dominio real  [B, 128, N']
    ├── AdaptiveAvgPool1d(64)            →  [B, 128, 64]  (longitud fija)
    │
    ├── Flatten → [B, 8192]
    ├── Linear(8192, 512) + ReLU + Dropout(0.4)
    ├── Linear(512,   64) + ReLU
    └── Linear( 64,    2) → logits [Clase0: Background, Clase1: Drone]

Justificación de hiperparámetros
---------------------------------
    - Kernels grandes (k=31, k=15) al principio: necesitamos ver varias longitudes 
      de onda de la portadora para capturar la fase coherentemente.
    - stride=2 en las 3 primeras capas: subsampleo progresivo sin perder info de
      fase (equivalente a decimación de la señal compleja).
    - AdaptiveAvgPool1d(64): elimina la necesidad de padding al manejar crops de
      longitud variable. Cada posición en el output representa un "resumen" de un
      segmento temporal del crop.
    - Dropout(0.4): regularización agresiva justificada por el Label Noise de la
      Opción A (BT camuflado en archivos de drones).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# BLOQUES PRIMITIVOS COMPLEJOS
# ═══════════════════════════════════════════════════════════════════════════════

class ComplexConv1d(nn.Module):
    """
    Convolución 1D de valores complejos implementada via regla de Wirtinger.

    Representación interna: el tensor de entrada x tiene shape [B, 2·C_in, N],
    donde los primeros C_in canales son la parte real y los siguientes C_in son
    la parte imaginaria.

    La operación matemática para un filtro W = W_re + j·W_im es:
        Re(W * x) = W_re(x_re) − W_im(x_im)
        Im(W * x) = W_re(x_im) + W_im(x_re)

    Parámetros
    ----------
    in_channels  : número de canales complejos de entrada  (C_in)
    out_channels : número de canales complejos de salida   (C_out)
    kernel_size  : tamaño del filtro 1D
    stride, padding, dilation, groups, bias : igual que nn.Conv1d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        # Dos convoluciones reales que representan Re(W) e Im(W)
        self.conv_re = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
        )
        self.conv_im = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 2·C_in,  N]
        -> [B, 2·C_out, N']
        """
        c = x.shape[1] // 2          # C_in
        x_re, x_im = x[:, :c, :], x[:, c:, :]

        out_re = self.conv_re(x_re) - self.conv_im(x_im)   
        out_im = self.conv_re(x_im) + self.conv_im(x_re)   

        return torch.cat([out_re, out_im], dim=1)           # [B, 2·C_out, N']


class ComplexBatchNorm1d(nn.Module):
    """
    Batch Normalization para señales complejas.

    Normaliza parte real e imaginaria por separado compartiendo un único
    vector de parámetros (gamma, beta) por cada mitad. Esto es suficiente
    para estabilizar el entrenamiento con señales RF sin la complejidad
    completa del BN covariante complejo (Trabelsi et al., 2018).
    """

    def __init__(self, num_complex_channels: int, **kwargs):
        super().__init__()
        # BN independiente para Re e Im
        self.bn_re = nn.BatchNorm1d(num_complex_channels, **kwargs)
        self.bn_im = nn.BatchNorm1d(num_complex_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1] // 2
        out_re = self.bn_re(x[:, :c, :])
        out_im = self.bn_im(x[:, c:, :])
        return torch.cat([out_re, out_im], dim=1)


class CReLU(nn.Module):
    """
    Complex ReLU (CReLU).

    Aplica ReLU por separado a parte real e imaginaria:
        CReLU(z) = ReLU(Re(z)) + j·ReLU(Im(z))

    Referencia: Bassey et al. (2021) §4.2 — la activación más estable
    para CVNNs en aplicaciones de señal.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)   # ReLU element-wise sobre todo el tensor [Re|Im]


def modulus(x: torch.Tensor) -> torch.Tensor:
    """
    Calcula el módulo |z| = sqrt(Re² + Im²) para regresar al dominio real.

    Entrada : [B, 2·C, N]
    Salida  : [B,   C, N]
    """
    c = x.shape[1] // 2
    re, im = x[:, :c, :], x[:, c:, :]
    return torch.sqrt(re ** 2 + im ** 2 + 1e-8)   # eps para estabilidad numérica


# ═══════════════════════════════════════════════════════════════════════════════
# BLOQUE CONVOLUCIONAL COMPLEJO (Conv + BN + CReLU)
# ═══════════════════════════════════════════════════════════════════════════════

class ComplexConvBlock(nn.Module):
    """
    Bloque básico: ComplexConv1d → ComplexBatchNorm1d → CReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = ComplexConv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn = ComplexBatchNorm1d(out_channels)
        self.act = CReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class ComplexConv1DNet(nn.Module):
    """
    CV-CNN para clasificación binaria de ráfagas I/Q: Dron vs Fondo.

    Entrada
    -------
    x : Tensor [B, 2, N]
        Canal 0 = I (parte real de la señal compleja)
        Canal 1 = Q (parte imaginaria de la señal compleja)
        N puede variar entre muestras; AdaptiveAvgPool1d lo normaliza.

    Salida
    ------
    logits : Tensor [B, num_classes]
        Sin softmax (usar CrossEntropyLoss directamente).

    Parámetros
    ----------
    num_classes     : número de clases de salida (default: 2)
    pool_output_size: tamaño del output de AdaptiveAvgPool1d (default: 64)
    dropout         : tasa de dropout en el clasificador (default: 0.4)

    Ejemplo de uso
    --------------
    >>> model = ComplexConv1DNet()
    >>> x = torch.randn(8, 2, 14000)   # batch de 8 crops de ~1ms a 14MHz
    >>> logits = model(x)              # → [8, 2]
    """

    def __init__(
        self,
        num_classes: int = 2,
        pool_output_size: int = 64,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pool_output_size = pool_output_size

        # ── Extractor de características complejas ────────────────────────────
        # Kernels grandes primero: capturamos ciclos completos de portadora.
        # Cada stride=2 divide la longitud temporal a la mitad.
        self.features = nn.Sequential(
            ComplexConvBlock(  1,  32, kernel_size=31, stride=2, padding=15),
            ComplexConvBlock( 32,  64, kernel_size=15, stride=2, padding=7),
            ComplexConvBlock( 64, 128, kernel_size= 7, stride=2, padding=3),
            ComplexConvBlock(128, 128, kernel_size= 3, stride=1, padding=1),
        )

        # ── Proyección al dominio real + pooling adaptativo ───────────────────
        # modulus() colapsa los 2·128 = 256 canales complejos a 128 reales.
        # AdaptiveAvgPool1d fija la longitud temporal a pool_output_size,
        # permitiendo que el clasificador opere independientemente del
        # tamaño del crop de entrada.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(pool_output_size)

        # ── Clasificador ─────────────────────────────────────────────────────
        fc_in = 128 * pool_output_size   # 128 * 64 = 8192
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        # Inicialización de pesos
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2, N]  →  logits: [B, num_classes]
        """
        # Preparar señal como tensor complejo: [B, 2, N] → [B, 2·1, N]
        # (los 2 canales de entrada representan 1 canal complejo: Re=canal 0, Im=canal 1)
        z = x   # shape: [B, 2, N]  →  ComplexConvBlock espera [B, 2·C_in, N] con C_in=1

        # Extracción de features en el plano complejo
        z = self.features(z)            # [B, 256, N']  (2·128 canales complejos)

        # Paso al dominio real via módulo
        z = modulus(z)                  # [B, 128, N']

        # Pooling adaptativo → longitud temporal fija
        z = self.adaptive_pool(z)       # [B, 128, 64]

        # Clasificación
        return self.classifier(z)       # [B, 2]

    def _init_weights(self):
        """
        Inicialización de Kaiming para convoluciones y Xavier para lineal.
        Recomendado por el survey (§5.1) para training estable de CVNNs.
        """
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
        """Devuelve el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Imprime un resumen rápido del modelo."""
        total = self.count_parameters()
        print("=" * 55)
        print(f"  ComplexConv1DNet — Binary Burst Classifier")
        print(f"  Clases             : {self.num_classes}")
        print(f"  Pool output size   : {self.pool_output_size}")
        print(f"  Parámetros total   : {total:,}")
        print(f"  Aprox. VRAM (bs=64): ~{total * 4 * 64 / 1e9:.2f} GB (estimate)")
        print("=" * 55)


# ═══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import torch

    model = ComplexConv1DNet(num_classes=2, pool_output_size=64, dropout=0.4)
    model.summary()

    # Test con crops de longitud variable (simulando batches reales)
    for N, desc in [(7_000, "~0.5ms"), (14_000, "~1ms"), (70_000, "~5ms"), (200_000, "~14ms")]:
        x = torch.randn(4, 2, N)      # batch de 4 muestras
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=1)
        print(f"  Input [4, 2, {N:>7,}] ({desc}) → logits {tuple(logits.shape)}"
              f"  |  probs ~{probs.mean(0).tolist()}")

    print("\n✅ Smoke test PASSED — El modelo acepta longitudes variables sin padding.")
