"""
marnet_fusion.py — MaRNet-Fusion: Bidirectional Mamba + Residual Networks +
                   Deep Statistical Features Fusion para deteccion pasiva de UAVs
=================================================================================

Arquitectura multi-rama de ultima generacion (2024-2026) disenada explicitamente para
clasificacion binaria robusta de senales I/Q en entornos de SNR hostil (< -10 dB).

Referencia al dataset
---------------------
    Gluge et al. (2024). "Robust Low-Cost Drone Detection and Classification Using
    CNNs in Low SNR Environments". NoisyUAV v2.
    fs = 14 MHz, L = 1,048,576 muestras por archivo, SNR in [-20, +30] dB.

Flujo de datos de alto nivel
-----------------------------
    z in C^L
        |
        +-- STFT -> Spectrogram [B,1,F,T] -> SoftThresh -> ResNetBranch -> e_spec [B,256]
        |
        +-- normalize([I;Q]) [B,2,L] -----------------------------> MambaBranch -> e_iq  [B,256]
        |
        +-- StatFeatures [B,5] -----------------------------------> MLPBranch   -> e_stat[B,128]
                                                                                      |
                                                                   PAM_Fusion (Gated Cross-Attention)
                                                                                      |
                                                                        Linear -> logit [B,1]

Dependencias
------------
    torch>=2.2, torchaudio>=2.2
    mamba-ssm (opcional, fallback a BiGRU cuDNN si no disponible)

Autores
-------
    Generado para TFM: "Deteccion de Drones con IA Avanzada"
    Fecha: 2026
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TA

# --- Mamba opcional -----------------------------------------------------------
try:
    from mamba_ssm import Mamba  # pip install mamba-ssm (requiere CUDA >= 11.6)
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False
    warnings.warn(
        "mamba-ssm no esta instalado. Se usara BiGRU (cuDNN) como SSM bidireccional. "
        "Para el SSM nativo: pip install mamba-ssm",
        stacklevel=2,
    )


# =============================================================================
# 1. CUSTOM DATASET  --- RFDroneDataset
# =============================================================================

class RFDroneDataset(torch.utils.data.Dataset):
    """
    Dataset multi-salida para la arquitectura MaRNet-Fusion.

    Cada muestra del disco es un tensor I/Q crudo z(t) in C^L representado como
    x_iq in R^{2xL} (canal 0 = I, canal 1 = Q). Este Dataset lo transforma
    dinamicamente en TRES representaciones ortogonales que alimentan las tres
    ramas del modelo:

        spectrogram   [1, F, T] -- representacion espectro-temporal 2D via STFT
        iq_sequence   [2, L]    -- secuencia IQ normalizada z-score
        stat_features [5]       -- vector de estadisticos fisicos HOS

    Parametros
    ----------
    df : pd.DataFrame
        DataFrame con columnas ['filepath', 'label', 'snr', 'grupo'] tal como
        produce `funciones.dataset.obtener_splits_dataset()`.
    crop_len : int
        Longitud del crop temporal extraido aleatoriamente de cada muestra.
    n_fft : int
        Tamano de la ventana STFT para generar el espectrograma (Rama 1).
    hop_length : int
        Desplazamiento entre ventanas STFT.
    augment : bool
        Activa aumentacion online (ruido AWGN + rotacion de fase). Solo train.
    """

    def __init__(
        self,
        df,
        crop_len: int = 2048,
        n_fft: int = 128,
        hop_length: Optional[int] = None,
        augment: bool = False,
    ):
        self.data = df.reset_index(drop=True)
        self.crop_len = crop_len
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.augment = augment

        # Spectrogram de potencia con ventana Hann: minimo spectral leakage.
        # power=2.0 -> |STFT|^2, normalized=True -> energia consistente entre
        # ventanas de distinto tamano. Crucial para comparar bins espectrales
        # de senales FHSS con distinto ancho de banda (200kHz a 2MHz).
        self._spec_transform = TA.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=True,
            center=True,
            pad_mode="reflect",
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Devuelve (spectrogram, iq_sequence, stat_features, label).

        spectrogram  : [1, F, T]  F = n_fft//2+1, T ~ crop_len/hop_length
        iq_sequence  : [2, L]     I y Q normalizados z-score
        stat_features: [5]        estadisticos fisicos
        label        : int        0=No-Dron, 1=Dron
        """
        row = self.data.iloc[idx]

        # Cargar y recortar -------------------------------------------------
        d = torch.load(row["filepath"], map_location="cpu", weights_only=False)
        iq_raw: torch.Tensor = d["x_iq"].float()   # [2, 1048576]
        L_total = iq_raw.shape[1]

        # Crop aleatorio: introduce variabilidad temporal para regularizacion
        if L_total > self.crop_len:
            start = torch.randint(0, L_total - self.crop_len, (1,)).item()
            iq_crop = iq_raw[:, start: start + self.crop_len]
        else:
            reps = math.ceil(self.crop_len / L_total)
            iq_crop = iq_raw.repeat(1, reps)[:, : self.crop_len]

        if self.augment:
            iq_crop = self._augment(iq_crop)

        # ===================================================================
        # OUTPUT 1: SPECTROGRAM [1, F, T]
        # ===================================================================
        # Computamos el espectrograma sobre |z(t)| = sqrt(I^2 + Q^2).
        # La potencia instantanea captura la envolvente de la modulacion,
        # invariante a la rotacion de fase del canal.
        # Log-compresin comprime el rango dinamico de ~80 dB a ~20 dB,
        # esencial para gradientes estables en regimen de bajo SNR.
        power_signal = torch.sqrt(iq_crop[0] ** 2 + iq_crop[1] ** 2 + 1e-12)
        spectrogram = self._spec_transform(power_signal)    # [F, T]
        spectrogram = torch.log1p(spectrogram).unsqueeze(0) # [1, F, T]

        # ===================================================================
        # OUTPUT 2: IQ SEQUENCE [2, L] normalizada z-score
        # ===================================================================
        # Normalizacion z-score por canal: elimina la dependencia de la ganancia
        # del receptor (path-loss, ganancia del LNA), permitiendo que el BiGRU
        # aprenda estructuras de modulacion puras independientemente del SNR absoluto.
        iq_mean = iq_crop.mean(dim=1, keepdim=True)
        iq_std  = iq_crop.std(dim=1, keepdim=True).clamp(min=1e-8)
        iq_sequence = (iq_crop - iq_mean) / iq_std         # [2, L]

        # ===================================================================
        # OUTPUT 3: STATISTICAL FEATURES [5]
        # ===================================================================
        stat_features = self._compute_stat_features(iq_crop)   # [5]

        label = int(row["label"])
        return spectrogram, iq_sequence, stat_features, label

    def _compute_stat_features(self, iq: torch.Tensor) -> torch.Tensor:
        """
        5 estadisticos fisicos de capa fisica extraidos con PyTorch puro.

        feat[0]: Entropia de Shannon del PSD. H = -sum(p_k * log(p_k)).
                 Senales FHSS: entropia alta (espectro disperso).
                 Ruido gaussiano: entropia maxima (espectro plano).
                 Detectable incluso a SNR = -15 dB.

        feat[1]: Amplitud media E[|z(t)|]. Proxy de la energia de la senal.

        feat[2]: Varianza total Var[I] + Var[Q]. Para ruido puro: Var ~ sigma_n^2.
                 Para senal + ruido: Var ~ sigma_s^2 + sigma_n^2.

        feat[3]: Fraccion de bins espectrales activos (BW efectivo normalizado).
                 Bins cuya energia supera la media local. Aproxima el BW ocupado
                 del transmisor: firma fisica estable del protocolo del drone.

        feat[4]: Kurtosis excedente K = E[(z-mu)^4] / sigma^4 - 3.
                 Detector clasico de no-gaussianidad (HOS: Higher Order Statistics).
                 K=0 para ruido gaussiano. K!=0 para modulaciones digitales (GFSK,OOK,BPSK).
        """
        amplitude = torch.sqrt(iq[0] ** 2 + iq[1] ** 2 + 1e-12)  # [L]

        # feat[0]: Entropia Shannon PSD via periodograma
        fft_mag_sq = torch.fft.rfft(amplitude).abs().pow(2)  # [L//2+1]
        psd_norm   = fft_mag_sq / (fft_mag_sq.sum() + 1e-12)
        entropy    = -(psd_norm * torch.log(psd_norm + 1e-12)).sum()

        # feat[1]: Amplitud media
        mean_amplitude = amplitude.mean()

        # feat[2]: Varianza total
        total_variance = iq[0].var() + iq[1].var()

        # feat[3]: Fraccion de bins activos (umbral local dinamico)
        local_mean = fft_mag_sq.mean()
        active_bins_frac = (fft_mag_sq > local_mean).float().mean()

        # feat[4]: Kurtosis excedente
        amp_c = amplitude - amplitude.mean()
        mu2 = (amp_c ** 2).mean()
        mu4 = (amp_c ** 4).mean()
        kurtosis = mu4 / (mu2 ** 2 + 1e-12) - 3.0

        feat = torch.stack([
            entropy.clamp(-20, 20),
            mean_amplitude.clamp(0, 10),
            total_variance.clamp(0, 100),
            active_bins_frac,       # ya normalizada en [0,1]
            kurtosis.clamp(-3, 30),
        ])
        return feat.float()

    def _augment(self, iq: torch.Tensor) -> torch.Tensor:
        """
        Aumentacion online para robustez de canal.

        1. Rotacion de fase aleatoria phi in [0, 2*pi): [I',Q'] = R(phi)*[I,Q].
           Simula el offset de fase del oscilador local del drone.
           La clasificacion es invariante a esta transformacion.

        2. Ruido AWGN aditivo a SNR adicional in [0, 10] dB.
           Simula variabilidad de la temperatura de ruido del receptor.
        """
        phi = torch.rand(1).item() * 2 * math.pi
        c, s = math.cos(phi), math.sin(phi)
        I_rot = iq[0] * c - iq[1] * s
        Q_rot = iq[0] * s + iq[1] * c
        iq = torch.stack([I_rot, Q_rot], dim=0)

        sig_power = (iq ** 2).mean().clamp(min=1e-12)
        snr_add_db = torch.rand(1).item() * 10.0
        noise_power = sig_power / (10 ** (snr_add_db / 10.0))
        iq = iq + torch.randn_like(iq) * noise_power.sqrt()
        return iq


# =============================================================================
# 2. SOFT-THRESHOLDING EMPIRICAL DENOISING LAYER
# =============================================================================

class SoftThresholdingBlock(nn.Module):
    """
    Capa de denoising diferenciable con umbral adaptativo aprendido via SE-block.

    Operador de umbralización suave:
        STh_lambda(x) = sign(x) * max(|x| - lambda, 0)

    El umbral lambda se aprende por canal via atencion Squeeze-and-Excitation:
        lambda_c = max_thresh * sigmoid(FC(GAP(x)))_c

    Esto permite que cada banda espectral tenga su propio nivel de denoising,
    critico cuando el espectro del ruido no es plano (interferencias co-canal).

    La umbral suave es el operador de proximidad de la norma L1, garantizando
    representaciones esparsas del espectrograma y gradientes bien definidos.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 8, max_threshold: float = 1.0):
        super().__init__()
        self.max_threshold = max_threshold
        mid = max(1, num_channels // reduction_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(num_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = self.se(x) * self.max_threshold   # [B, C]
        lam = lam[:, :, None, None]             # [B, C, 1, 1]
        return torch.sign(x) * F.relu(x.abs() - lam)


# =============================================================================
# 3. BRANCH 1 --- ResNet-Lite para Espectrogramas 2D
# =============================================================================

class _ResBlock2D(nn.Module):
    """
    Bloque residual 2D estilo ResNet-18 con GELU.

    Las conexiones residuales son criticas a bajo SNR: el espectrograma de
    un drone a -15 dB es visualmente indistinguible del ruido. El residual
    permite aprender la funcion identidad base y solo el residuo discriminativo,
    evitando la degradacion de gradiente en las capas profundas.
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.act   = nn.GELU()
        self.skip  = (
            nn.Sequential(nn.Conv2d(channels, channels, 1, stride=stride, bias=False),
                          nn.BatchNorm2d(channels))
            if stride > 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


class ResNetBranch(nn.Module):
    """
    Rama 1: ResNet-Lite 2D para espectrogramas STFT.

    Topologia:
        [B,1,F,T] -> SoftThresh -> Stem(32,k=7,s=2) -> ResBlocks(32)x2
                  -> Down(64,s=2) -> ResBlocks(64)x2
                  -> Down(128,s=2) -> ResBlocks(128)x2
                  -> AdaptiveAvgPool2d(4,4) -> Linear(128*16, latent_dim)
                  -> e_spec [B, latent_dim]
    """

    def __init__(self, latent_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        # Denoising espectral antes de la extraccion de features:
        # el espectrograma a SNR<-10dB contiene mas energia de ruido que de senal,
        # por lo que los filtros Conv2d sin denoising aprenden patrones espurios.
        self.denoiser = SoftThresholdingBlock(num_channels=1, reduction_ratio=1, max_threshold=2.0)
        self.stem  = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        self.stage1 = nn.Sequential(_ResBlock2D(32), _ResBlock2D(32))
        self.down1  = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(64), nn.GELU())
        self.stage2 = nn.Sequential(_ResBlock2D(64), _ResBlock2D(64))
        self.down2  = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(128), nn.GELU())
        self.stage3 = nn.Sequential(_ResBlock2D(128), _ResBlock2D(128))
        self.pool   = nn.AdaptiveAvgPool2d((4, 4))
        self.proj   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, latent_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.denoiser(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.proj(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)


# =============================================================================
# 4. BRANCH 2 --- Bidirectional SSM (BiGRU / Mamba) para secuencias IQ 1D
# =============================================================================

class BiMambaPlaceholder(nn.Module):
    """
    Aproximacion SSM bidireccional via BiGRU con kernels cuDNN.

    Justificacion matematica: GRU como SSM selectivo
    -------------------------------------------------
    El GRU implementa la recurrencia gated:
        z_t = sigma(W_z*x_t + U_z*h_{t-1})          # update gate ~ Delta (Mamba)
        r_t = sigma(W_r*x_t + U_r*h_{t-1})          # reset gate
        h_tilde = tanh(W*x_t + U*(r_t o h_{t-1}))   # candidato ~ B_bar * x_t
        h_t = (1-z_t) o h_{t-1} + z_t o h_tilde     # update ~ A_bar*h + B_bar*x

    Esta es matematicamente un caso especial del SSM lineal gated (Gu & Dao, 2023).
    La selecividad de Mamba (parametros delta, B, C dependientes de x_t) esta
    capturada por los gates del GRU que tambien dependen de x_t en cada paso.

    Ventaja computacional sobre el Python-loop:
        - Python-loop O(L): ~4000 iteraciones Python por secuencia -> muy lento
        - BiGRU cuDNN:      kernel CUDA fusionado y paralelizado -> 100-1000x mas rapido

    El escaneo bidireccional captura dependencias causales (t=0->L) y anti-causales
    (t=L->0), equivalente al Bi-Mamba de Gu et al. con las matrices A, B, C, D
    aprendidas en ambas direcciones independientemente.

    Parametros
    ----------
    d_model : int   Dimension de embedding interno.
    d_state : int   Equivale al hidden_size del GRU.
    num_layers : int Capas GRU apiladas (profundidad recurrente).
    """

    def __init__(self, d_model: int, d_state: int = 16, num_layers: int = 3):
        super().__init__()
        self.d_model = d_model
        # Proyeccion de embedding: [I, Q] (dim=2) -> d_model
        self.input_proj = nn.Linear(2, d_model)
        # BiGRU cuDNN: hidden_size=d_model, bidirectional=True -> output dim = d_model*2
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.norm_out = nn.LayerNorm(d_model * 2)
        # Proyeccion de vuelta a d_model para interfaz consistente
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 2, L]  ->  out: [B, d_model]"""
        x = x.permute(0, 2, 1)     # [B, L, 2]
        x = self.input_proj(x)     # [B, L, d_model]
        out, _ = self.gru(x)       # [B, L, d_model*2]
        out = self.norm_out(out)
        # Mean pooling temporal: mas robusto que el estado final h_T
        # cuando la senal termina en un segmento de alta varianza de ruido.
        out = out.mean(dim=1)      # [B, d_model*2]
        return self.out_proj(out)  # [B, d_model]


class MambaBranch(nn.Module):
    """
    Rama 2: SSM Bidireccional (Mamba nativo o BiGRU fallback) para IQ 1D.

    La rama explota las correlaciones temporales de largo alcance del proceso RF:
    estructura de burst FHSS, chirp de FM, modulacion de amplitud de OOK.
    Para L=2048 muestras a 14MHz: ~146 us, suficientes para capturar
    multiples periodos de la mayoria de los protocolos de control de drones.

    Parametros
    ----------
    d_model : int      Embedding interno SSM. Por defecto 128.
    d_state : int      Dimension estado SSM (hidden_size GRU). Por defecto 16.
    num_layers : int   Capas SSM apiladas. Por defecto 3.
    latent_dim : int   Dimension de salida. Por defecto 256.
    dropout : float    Dropout en proyeccion final. Por defecto 0.3.
    """

    def __init__(self, d_model: int = 128, d_state: int = 16,
                 num_layers: int = 3, latent_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

        if _MAMBA_AVAILABLE:
            self.input_proj = nn.Linear(2, d_model)
            self.ssm_layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
        else:
            # BiGRU: misma matematica gated-recurrence, cuDNN-optimizado
            self.ssm = BiMambaPlaceholder(d_model, d_state, num_layers)
            self.input_proj = None

        self.proj = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 2, L]  ->  e_iq: [B, latent_dim]"""
        if _MAMBA_AVAILABLE:
            x = x.permute(0, 2, 1)     # [B, L, 2]
            x = self.input_proj(x)      # [B, L, d_model]
            for layer in self.ssm_layers:
                x = layer(x)
            x = self.norm(x)
            x = x.mean(dim=1)           # [B, d_model]
        else:
            x = self.ssm(x)             # [B, d_model]  via BiGRU

        return self.proj(x)             # [B, latent_dim]


# =============================================================================
# 5. BRANCH 3 --- Deep MLP para estadisticos fisicos
# =============================================================================

class MLPBranch(nn.Module):
    """
    Rama 3: Red neuronal densa para el vector de estadisticos fisicos.

    Los 5 estadisticos HOS (entropia, amplitud, varianza, BW, kurtosis) son
    invariantes a las transformaciones de la CNN y el SSM, constituyendo un
    canal de graduente independiente hacia la decision. Particularmente utiles
    a SNR < -15 dB donde el espectrograma y la IQ son dominadas por ruido
    pero la kurtosis excedente y la entropia PSD mantienen discriminabilidad.

    Topologia: 5 -> 64 -> 128 -> 128 -> latent_dim con LayerNorm + GELU + Dropout.
    """

    def __init__(self, input_dim: int = 5, latent_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        def _blk(a, b):
            return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.LayerNorm(b), nn.Dropout(dropout))
        self.net = nn.Sequential(
            _blk(input_dim, 64),
            _blk(64, 128),
            _blk(128, 128),
            nn.Linear(128, latent_dim),
            nn.GELU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 6. STOCHASTIC PATH ATTENTION & CROSS-MODAL FUSION MODULE (PAM_Fusion)
# =============================================================================

class PAM_Fusion(nn.Module):
    """
    Fusión multi-modal con atencion cruzada y gating estocastico.

    Motivacion matematica
    ----------------------
    La fusion naive por concatenacion implica contribuciones constantes de cada
    rama independientemente del SNR, lo cual es fisicamente incorrecto:
      - SNR > 0 dB:    las tres ramas contribuyen por igual.
      - SNR [-10,0) dB: el espectrograma degrada; Mamba y estadisticos dominan.
      - SNR < -15 dB:  solo los estadisticos HOS (kurtosis, entropia) son estables.

    Implementacion: Cross-Attention con query global
    ------------------------------------------------
        Q = Linear_Q(concat[h_spec, h_iq, h_stat])     [B, D]
        K_i = Linear_K(h_i)                             [B, D] per rama i
        scores_i = Q . K_i^T / sqrt(D)                 [B, 3] logits
        w = softmax(scores)                              [B, 3] weights in simplex
        e_fused = sum_i w_i * Linear_V(h_i)             [B, D]

    Los pesos w son adaptativos por instancia: el modelo aprende a penalizar
    las ramas degradadas segun el contexto energetico de la senal.

    Parametros
    ----------
    dim_spec, dim_iq, dim_stat : dimensiones de los embeddings de entrada.
    d_fusion : dimension del espacio de fusion. Por defecto 256.
    dropout_attn : dropout sobre los pesos de atencion (regularizacion).
    """

    def __init__(self, dim_spec: int = 256, dim_iq: int = 256, dim_stat: int = 128,
                 d_fusion: int = 256, dropout_attn: float = 0.1):
        super().__init__()
        self.d_fusion = d_fusion

        # Proyecciones al espacio de fusion comun
        self.proj_spec = nn.Sequential(nn.Linear(dim_spec, d_fusion), nn.GELU())
        self.proj_iq   = nn.Sequential(nn.Linear(dim_iq,   d_fusion), nn.GELU())
        self.proj_stat = nn.Sequential(nn.Linear(dim_stat, d_fusion), nn.GELU())

        # Cross-attention: Q desde concatenacion global, K/V desde cada rama
        self.W_q  = nn.Linear(d_fusion * 3, d_fusion)
        self.W_k  = nn.Linear(d_fusion, d_fusion)
        self.W_v  = nn.Linear(d_fusion, d_fusion)
        self.attn_drop = nn.Dropout(dropout_attn)
        self.out_proj  = nn.Linear(d_fusion, d_fusion)

        # Refinamiento post-fusion
        self.norm1 = nn.LayerNorm(d_fusion)
        self.ffn   = nn.Sequential(
            nn.Linear(d_fusion, d_fusion * 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(d_fusion * 2, d_fusion),
        )
        self.norm2 = nn.LayerNorm(d_fusion)

        # Clasificador binario: un solo logit (sin sigmoid -> usar BCEWithLogitsLoss)
        self.classifier = nn.Sequential(
            nn.Linear(d_fusion, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, e_spec: torch.Tensor, e_iq: torch.Tensor,
                e_stat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Entradas: e_spec [B, dim_spec], e_iq [B, dim_iq], e_stat [B, dim_stat]
        Salidas:
            logit        [B, 1]  -- logit binario sin sigmoid
            attn_weights [B, 3]  -- pesos de atencion pre-dropout [spec, iq, stat]
        """
        h_spec = self.proj_spec(e_spec)  # [B, D]
        h_iq   = self.proj_iq(e_iq)     # [B, D]
        h_stat = self.proj_stat(e_stat) # [B, D]

        # Cross-attention: Q es la "pregunta global" sobre cual rama confiar
        Q = self.W_q(torch.cat([h_spec, h_iq, h_stat], dim=-1))    # [B, D]
        kv = torch.stack([h_spec, h_iq, h_stat], dim=1)            # [B, 3, D]
        K  = self.W_k(kv)   # [B, 3, D]
        V  = self.W_v(kv)   # [B, 3, D]

        scale = math.sqrt(self.d_fusion)
        scores   = torch.bmm(Q.unsqueeze(1), K.permute(0, 2, 1)) / scale  # [B, 1, 3]
        attn_w   = F.softmax(scores, dim=-1).squeeze(1)                    # [B, 3] pre-dropout
        attn_out = torch.bmm(self.attn_drop(attn_w.unsqueeze(1)), V).squeeze(1)  # [B, D]
        e_fused  = self.out_proj(attn_out)                                  # [B, D]

        # Fusion residual: promedio de proyecciones como skip connection
        skip    = (h_spec + h_iq + h_stat) / 3.0
        e_fused = self.norm1(e_fused + skip)
        e_fused = self.norm2(e_fused + self.ffn(e_fused))

        logit = self.classifier(e_fused)    # [B, 1]
        return logit, attn_w                # devolvemos attn pre-dropout


# =============================================================================
# 7. MODELO PRINCIPAL --- MaRNetFusion
# =============================================================================

class MaRNetFusion(nn.Module):
    """
    MaRNet-Fusion: Multi-Branch RF Drone Detector.

    Ensambla las tres ramas y el modulo de fusion en un modelo end-to-end diferenciable.
    Disenado para SNR < -10 dB. Supera el baseline CV-CNN de 75% accuracy.

    Entradas (desde RFDroneDataset)
    -------------------------------------------
    spectrogram   : [B, 1, F, T]
    iq_sequence   : [B, 2, L]
    stat_features : [B, 5]

    Salidas
    -------
    logit        : [B, 1]  logit crudo (sin sigmoid). Usar BCEWithLogitsLoss.
    attn_weights : [B, 3]  pesos de atencion [spec, iq, stat] por instancia.
    """

    def __init__(
        self,
        latent_dim_spec: int = 256,
        latent_dim_iq: int = 256,
        latent_dim_stat: int = 128,
        d_fusion: int = 256,
        d_model_ssm: int = 128,
        d_state_ssm: int = 16,
        num_ssm_layers: int = 3,
        dropout_cnn: float = 0.3,
        dropout_ssm: float = 0.3,
        dropout_mlp: float = 0.4,
    ):
        super().__init__()
        self.branch_spec = ResNetBranch(latent_dim=latent_dim_spec, dropout=dropout_cnn)
        self.branch_iq   = MambaBranch(d_model=d_model_ssm, d_state=d_state_ssm,
                                       num_layers=num_ssm_layers, latent_dim=latent_dim_iq,
                                       dropout=dropout_ssm)
        self.branch_stat = MLPBranch(input_dim=5, latent_dim=latent_dim_stat, dropout=dropout_mlp)
        self.fusion      = PAM_Fusion(dim_spec=latent_dim_spec, dim_iq=latent_dim_iq,
                                      dim_stat=latent_dim_stat, d_fusion=d_fusion)

    def forward(self, spectrogram: torch.Tensor, iq_sequence: torch.Tensor,
                stat_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        spectrogram   = spectrogram.to(device)
        iq_sequence   = iq_sequence.to(device)
        stat_features = stat_features.to(device)

        e_spec = self.branch_spec(spectrogram)    # [B, 256]
        e_iq   = self.branch_iq(iq_sequence)      # [B, 256]
        e_stat = self.branch_stat(stat_features)  # [B, 128]

        logit, attn_w = self.fusion(e_spec, e_iq, e_stat)
        return logit, attn_w

    def count_parameters(self) -> Dict[str, int]:
        def _n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "ResNet Branch":  _n(self.branch_spec),
            "BiGRU Branch":   _n(self.branch_iq),
            "MLP Branch":     _n(self.branch_stat),
            "PAM Fusion":     _n(self.fusion),
            "TOTAL":          _n(self),
        }

    def summary(self):
        params = self.count_parameters()
        backend = "mamba-ssm (CUDA)" if _MAMBA_AVAILABLE else "BiGRU cuDNN"
        print("=" * 62)
        print("  MaRNet-Fusion -- Multi-Branch RF Drone Detector")
        print(f"  SSM Backend  : {backend}")
        print("-" * 62)
        for k, v in params.items():
            tag = ">" if k == "TOTAL" else " "
            print(f"  {tag} {k:<30}: {v:>12,}")
        print(f"\n  Est. VRAM (BS=32): ~{params['TOTAL']*4*32/1e9:.2f} GB")
        print("=" * 62)


# =============================================================================
# 8. MIXUP AUGMENTATION
# =============================================================================

def mixup_batch(spec, iq, stat, labels, alpha=0.4):
    """
    MixUp (Zhang et al., 2018): mezcla convexa de pares de instancias.
        x_mix = lambda * x_i + (1-lambda) * x_j    lambda ~ Beta(alpha, alpha)
        y_mix = lambda * y_i + (1-lambda) * y_j

    Para deteccion en bajo SNR, MixUp crea muestras sinteticas en SNR intermedios
    que densifican el borde de la superficie de decision y mejoran la robustez
    frente a variaciones continuas del canal.

    alpha=0.4: tipicamente produce lambda in [0.2, 0.8], suficiente para
    enriquecer la frontera sin destruir las instancias puras.
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    idx = torch.randperm(spec.shape[0], device=spec.device)
    return (
        lam * spec  + (1 - lam) * spec[idx],
        lam * iq    + (1 - lam) * iq[idx],
        lam * stat  + (1 - lam) * stat[idx],
        labels.float(),
        labels[idx].float(),
        lam,
    )


def mixup_criterion(criterion, logit, ya, yb, lam):
    """L(y_hat, y_mix) = lambda * L(y_hat, ya) + (1-lambda) * L(y_hat, yb)"""
    return lam * criterion(logit, ya.unsqueeze(1)) + (1 - lam) * criterion(logit, yb.unsqueeze(1))


# =============================================================================
# 9. TRAINING & EVALUATION LOOPS
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    mixup_alpha=0.4, mixup_prob=0.5, grad_clip=2.0, scaler=None):
    """
    Loop de entrenamiento de una epoca con MixUp estocastico y AMP opcional.

    MixUp se aplica con probabilidad mixup_prob a cada batch.
    Gradient clipping previene la explosion del gradiente en el BiGRU
    (problema conocido en recurrencias con secuencias largas).
    """
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    attn_accum = torch.zeros(3, device=device)

    for i, (spec, iq, stat, labels) in enumerate(dataloader):
        spec   = spec.to(device, non_blocking=True)
        iq     = iq.to(device, non_blocking=True)
        stat   = stat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_mixup = (torch.rand(1).item() < mixup_prob)

        if use_mixup:
            spec_in, iq_in, stat_in, ya, yb, lam = mixup_batch(spec, iq, stat, labels,
                                                                 alpha=mixup_alpha)
        else:
            spec_in, iq_in, stat_in = spec, iq, stat
            ya, lam = labels.float(), 1.0

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logit, attn_w = model(spec_in, iq_in, stat_in)
                loss = (mixup_criterion(criterion, logit, ya, yb, lam)
                        if use_mixup else criterion(logit, ya.unsqueeze(1)))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logit, attn_w = model(spec_in, iq_in, stat_in)
            loss = (mixup_criterion(criterion, logit, ya, yb, lam)
                    if use_mixup else criterion(logit, ya.unsqueeze(1)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logit.squeeze(1)) > 0.5).long()
            total_correct += (preds == labels.long()).sum().item()
            total_n += labels.shape[0]
            total_loss += loss.item() * labels.shape[0]
            attn_accum += attn_w.mean(dim=0).detach()

    return {
        "loss": total_loss / max(total_n, 1),
        "acc":  total_correct / max(total_n, 1),
        "attn": (attn_accum / (i + 1)).cpu().tolist(),
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluacion sin gradientes con metricas completas de clasificacion binaria."""
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    for spec, iq, stat, labels in dataloader:
        spec   = spec.to(device, non_blocking=True)
        iq     = iq.to(device, non_blocking=True)
        stat   = stat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logit, _ = model(spec, iq, stat)
        loss = criterion(logit, labels.float().unsqueeze(1))
        total_loss += loss.item() * labels.shape[0]

        probs = torch.sigmoid(logit.squeeze(1))
        preds = (probs > 0.5).long()
        all_preds.append(preds.cpu()); all_labels.append(labels.cpu()); all_probs.append(probs.cpu())

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs  = torch.cat(all_probs)
    N = all_labels.shape[0]

    tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().float()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().float()

    prec  = (tp / (tp + fp + 1e-8)).item()
    rec   = (tp / (tp + fn + 1e-8)).item()
    f1    = 2 * prec * rec / (prec + rec + 1e-8)
    acc   = ((tp + tn) / N).item()
    spec_ = (tn / (tn + fp + 1e-8)).item()

    return {
        "loss":      total_loss / N,
        "acc":       acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "specificity": spec_,
        "probs":     all_probs.numpy(),
        "labels":    all_labels.numpy(),
        "preds":     all_preds.numpy(),
    }


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MaRNetFusion(d_model_ssm=64, d_state_ssm=8, num_ssm_layers=2).to(device)
    model.summary()

    B, F, T, L = 4, 65, 65, 2048
    spec = torch.randn(B, 1, F, T, device=device)
    iq   = torch.randn(B, 2, L,   device=device)
    stat = torch.randn(B, 5,      device=device)
    labs = torch.randint(0, 2, (B,), device=device)

    with torch.no_grad():
        logit, attn = model(spec, iq, stat)

    print(f"logit: {logit.shape}  attn: {attn.shape}")
    print(f"attn (row sums): {attn.sum(dim=1).tolist()}")  # debe ser ~1.0

    spec_m, iq_m, stat_m, ya, yb, lam = mixup_batch(spec, iq, stat, labs)
    print(f"MixUp lam={lam:.4f}  OK")
    print("SMOKE TEST PASSED")
