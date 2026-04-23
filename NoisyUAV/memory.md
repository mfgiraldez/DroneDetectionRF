# memory.md — Memoria Técnica del Proyecto
> Actualizado: 2026-04-23 (HybridCVCNN Run #1 + lección crítica de normalización) | TFM: Detección de Drones UAV con IA Avanzada

---

## 1. Dataset NoisyUAV v2

### Estructura de ficheros
```
C:\TFM_data\NoisyUAV\drone_RF_data\
└── IQdata_sample{N}_target{T}_snr{S}.pt
    N = índice de muestra
    T = clase (1=drone, 2=AWGN, 3=WiFi, 4=Bluetooth)
    S = SNR en dB (entero, puede ser negativo)
```

### Constantes clave
- **Fs = 14 MHz** (frecuencia de muestreo)
- **N_samples = 1,048,576** (~75 ms por muestra)
- **target_noise = 4** (cualquier T ≠ 1 → clase Ruido=0; T=1 → Drone=1)
- Clases de drone: DJI, FutabaT14, FutabaT7, Graupner, Taranis, Turnigy

### Grupos SNR
| Grupo | Rango | Descripción |
|---|---|---|
| A | SNR >= 10 dB | Fácil, señal muy visible |
| B | -6 <= SNR < 10 dB | Moderado |
| C | SNR < -6 dB | Hostil, objetivo principal del TFM |

### Estructura del tensor en los .pt
```python
d = torch.load(filepath, weights_only=False)
d['x_iq']   # Tensor [2, 1048576] — canal 0=I, canal 1=Q
d['label']  # int (1=Drone, 2=AWGN, 3=WiFi, 4=BT)
d['snr']    # float/int en dB
```

### Splits del dataset
- Train: ~70% | Val: ~15% | Test: ~15%
- Estratificado por: clase binaria + SNR (combinación)
- Función: `NoisyUAV.funciones.dataset.obtener_splits_dataset(data_dir=...)`

---

## 2. Entorno de Desarrollo

- **OS:** Windows 10/11
- **Python:** 3.x en conda env `IAIAVv3`
- **Ruta Python:** `C:\Users\Manuel\anaconda3\envs\IAIAVv3\python.exe`
- **REGLA CRÍTICA:** Siempre `cmd /c "python ..."` para subprocesos (EOF correcto en Windows)
- **num_workers = 0** en DataLoader (obligatorio en Windows, evita deadlocks)
- **matplotlib.use("Agg")** en scripts de fondo (sin display)
- **PYTHONIOENCODING = "utf-8"** al inicio de scripts para evitar UnicodeEncodeError en cp1252

---

## 3. Modelos Implementados

### 3.1 Baseline: CV-CNN (ComplexConv1DNet)
- **Archivo:** `NoisyUAV/modelos/cvcnn.py`
- **Accuracy:** ~75% (referencia principal)
- **Arquitectura:** 4 × ComplexConvBlock (convoluciones complejas Wirtinger)
  - Canales: 1→32→64→128→128 (complejos)
  - Stride: 2,2,2,1
  - Módulo |z| → dominio real
  - AdaptiveAvgPool1d → Linear → Clasificador
- **Input:** [B, 2, N] — crop de longitud variable
- **Output:** [B, 2] — logits binarios

### 3.2 MaRNet-Fusion (FRACASÓ — Run #1)
- **Archivo:** `NoisyUAV/modelos/marnet_fusion.py`
- **Accuracy:** 65.66% | F1=0.7270
- **Problemas:**
  - crop_len=2048 (≈146 µs) — demasiado corto para estructura FHSS (necesitan ≥4.7 ms)
  - NaN en TrLoss épocas 29-37 (colapso BiGRU en curriculum learning fase 2)
  - PAM_Fusion colapsó a bias constante → modelo decía "Drone" siempre
  - Recall=91.5%, Specificity=39.9% — sesgo masivo
  - n_fft=128 → espectrograma 65×65 — insuficiente para FHSS vs OFDM
- **Informe:** `NoisyUAV/resultados_marnet/`

### 3.3 HybridCVCNN (COMPLETADO — Run #1 exitoso)
- **Archivos:**
  - `NoisyUAV/modelos/hybrid_cvcnn.py` — modelo + dataset + utilidades
  - `NoisyUAV/funciones/physical_features.py` — extracción + caché de features
  - `NoisyUAV/run_hybrid_experiment.py` — pipeline autónomo completo
  - `NoisyUAV/regenerate_from_checkpoint.py` — regenerar resultados desde checkpoint
- **Arquitectura:**
  ```
  IQ crop [B, 2, 131072] (~9.4 ms @ 14 MHz)
       │
  CV-CNN Backbone (4×ComplexConvBlock + modulus + AdaptivePool)
       → embedding [B, 256]
       │
  Physical Features (detector entropía Shannon):
       noise_floor, noise_sigma, mean_n_active, p75_n_active,
       H_min, H_mean, n_bursts, dur_ms, z_peak, drop_b, n_act, dur_total
       → [B, 12] (z-score normalizado)
       │
  concat [B, 268] → MLP(268→256→128→1) → logit
  ```
- **crop_len = 131072** (≈9.4 ms — cubre un período FHSS completo)
- **Params reales: 1,429,849** (backbone: 1,327,168 + MLP: 102,681)
- **Input 2:** features físicas pre-computadas desde caché .npz

#### Resultados del Entrenamiento (2026-04-23)
- **Best epoch:** 13 | Early stopping en época 25 (patience=12)
- **Tiempo total:** 3.35 horas (RTX 4060, crop_len=131072, batch_size=32)
- **Tiempo por época:** ~8.5 min (CPU DataLoader, 12420 muestras)

| Métrica | **HybridCVCNN** | Baseline CV-CNN | MaRNet-Fusion |
|---|---|---|---|
| **Accuracy** | **80.92%** | ~75% | 65.66% |
| **F1-Score** | **0.8112** | ~0.750 | 0.7270 |
| **AUC-ROC** | **0.9078** | -- | -- |
| Precision | 80.22% | -- | -- |
| Recall | 82.03% | ~91.5% (sesgado) | 91.5% |
| **Specificity** | **79.80%** | -- | 39.9% (sesgado) |
| Mejora vs baseline | **+5.92 p.p.** | -- | -9.34 p.p. |

#### Resultados por Grupo SNR (Test Set)
| Grupo | Rango | Acc media | Destacable |
|---|---|---|---|
| A (fácil) | SNR >= 10 dB | ~90% | SNR=+4 dB: **100%** perfecto |
| B (medio) | -6..+10 dB | ~82% | Sólido y equilibrado |
| C (hostil) | SNR < -6 dB | ~55% | Mejor que MaRNet (50%) |

#### Comportamiento por SNR destacable
- SNR=-20 dB: Acc=43.7% (límite físico — señal inapreciable)
- SNR=-10 dB: Acc=79.4% — ya supera el baseline
- SNR=-8 dB:  Acc=81.4%
- SNR=+4 dB:  Acc=**100.0%** — clasificación perfecta
- SNR=+8 dB:  Acc=95.1%
- Recall=100% en SNR >= +8 dB (cero falsos negativos en señal limpia)

#### Observaciones y próximos pasos
- El modelo ya NO está sesgado (Specificity=79.8% vs 39.9% de MaRNet)
- El grupo C (SNR<-6) sigue siendo el punto débil — Aproximación 2 puede mejorar esto
- La alta Recall en SNR alto pero baja Specificity en SNR muy bajo sugiere que el modelo
  aprende correctamente las características físicas del drone pero le cuesta en señal inapreciable
- **Checkpoint guardado:** `resultados_hybrid/checkpoints/best_model.pt`

---

## 4. Detector de Entropía Shannon

- **Archivo:** `NoisyUAV/funciones/detector_entropia.py`
- **Función principal:** `detectar_bursts(iq_tensor, fs, nperseg, ...)`
- **Output:** `(t, H, H_smooth, umbral_v, nf_v, ns, n_active, bursts)`
- **`bursts`:** lista de dicts con claves: `z_peak`, `n_act`, `dur_ms`, `drop_b`, `t_start`, `t_end`
- **Método:** STFT + entropía whitened + CFAR adaptativo + morfología

### 12 Features Físicas (HybridCVCNN)
| Idx | Nombre | Descripción | Discrimina |
|---|---|---|---|
| 0 | noise_floor | Piso de ruido CFAR (bits) | SNR implícito |
| 1 | noise_sigma | Dispersión MAD del piso | Estabilidad espectral |
| 2 | mean_n_active | Bins activos medios | Ancho de banda |
| 3 | p75_n_active | P75 bins activos | WiFi OFDM >> 512 bins |
| 4 | H_min | Mínimo de entropía | Concentración espectral |
| 5 | H_mean | Entropía media | Proxy del piso de ruido |
| 6 | n_bursts | Número de bursts detectados | FHSS = múltiples hops |
| 7 | dur_ms_main | Duración burst principal (ms) | FHSS: 0.5-5 ms; WiFi: >10 ms |
| 8 | z_peak_main | Significancia estadística burst | Confianza de la detección |
| 9 | drop_b_main | Caída de entropía (bits) | Pureza espectral |
| 10 | n_act_main | Bins activos en el burst | FHSS: 58-200; WiFi: >512 |
| 11 | dur_total | Duración total de actividad (ms) | Patrón temporal |

---

## 5. Resultados del Diagnóstico Exploratorio (2026-04-23)

Clasificador lineal (Logistic Regression) sobre las 12 features físicas, 416 muestras:

| Grupo | Acc | AUC | Top features |
|---|---|---|---|
| Global | 0.745 | 0.825 | z_peak_main, H_mean, noise_sigma |
| A (SNR>=10) | **0.949** | **0.982** | z_peak_main, H_min, drop_b_main |
| B (-6..10) | 0.727 | 0.856 | z_peak_main, noise_sigma, H_min |
| C (SNR<-6) | 0.723 | 0.779 | noise_sigma, noise_floor, mean_n_active |

**Conclusión:** Las features del detector discriminan incluso a bajo SNR (Grupo C: 72.3%, mejor que MaRNet-Fusion 50.84%).
Hipótesis confirmadas: H1 ✓, H2 ✓.

---

## 6. Roadmap de Investigación (Plan de 4 Aproximaciones)

| # | Nombre | Riesgo | Estado | Acc |
|---|---|---|---|---|
| 1 | HybridCVCNN (CV-CNN + physical features) | Bajo | **COMPLETADO** | **80.92%** |
| 2 | Prototypical/Contrastive Network | Medio | PENDIENTE | -- |
| 3 | High-SNR Teacher -> Low-SNR Student | Medio-alto | PENDIENTE | -- |
| 4 | S4D/Mamba PyTorch puro | Alto | PENDIENTE | -- |

---

## 7. Estructura de Resultados

```
NoisyUAV/
├── resultados_marnet/           Resultados de MaRNet-Fusion (fracasado, Acc=65.66%)
│   └── diagnostic/              Figuras del diagnostico exploratorio
├── resultados_hybrid/           Resultados de HybridCVCNN (COMPLETADO, Acc=80.92%)
│   ├── checkpoints/
│   │   ├── best_model.pt        Mejor checkpoint epoch=13 (Val F1=0.8161)
│   │   └── last_model.pt        Ultimo checkpoint (epoch=25)
│   ├── features_cache.npz       Cache de 12 physical features (17744 entradas, ~35MB)
│   ├── figures/                 6 figuras PNG (confusion, ROC, per-SNR, score dist...)
│   ├── informe_hybrid_cvcnn.md  Informe completo en Markdown
│   └── metricas_completas.json  Metricas en JSON para analisis posterior
```

---

## 8. ERROR CRITICO COMETIDO: Falta de Normalización IQ (Run #1)

> LECCIÓN APRENDIDA — No repetir jamás

En el primer entrenamiento del HybridCVCNN (Run #1, 2026-04-23) **se entrenó SIN normalizar la amplitud de la señal IQ**. Esto es un error fundamental en cualquier pipeline de ML con señales RF.

**Síntoma visible:** Val Loss oscilando bruscamente (0.37 a 0.60) sin converger suavemente.

**Causa:** Las señales a distintos SNRs tienen amplitudes radicalmente diferentes:
- SNR=+30 dB: potencia ~10^5 veces mayor que SNR=-20 dB
- Sin normalización, el BatchNorm interno ve distribuciones muy distintas por batch
- Batches mezclados con SNRs muy distantes causan inestabilidad en los gradientes

**Corrección obligatoria** — añadir SIEMPRE en `Dataset.__getitem__` tras el crop:
```python
# Normalización por potencia RMS (elimina escala, preserva fase y patrón temporal)
power = iq_crop.pow(2).mean().clamp(min=1e-12).sqrt()
iq_crop = iq_crop / power
```

**Impacto estimado de corregirlo:**
- Val Loss convergería suave (sin oscilaciones)
- Mejor generalización en Grupo C (SNR bajo)
- Convergencia en menos épocas
- El modelo ya llegó al 80.92% SIN normalizar — con normalización debería superar el 83-85%

**Estado:** Pendiente de re-entrenamiento (Run #2) con normalización correcta.

---

## 9. Bugs y Gotchas Conocidos

- **[CRITICO] IQ sin normalizar:** Siempre normalizar por potencia RMS antes de entrar al modelo (ver sección 8)
- **UnicodeEncodeError cp1252:** Añadir `os.environ["PYTHONIOENCODING"] = "utf-8"` al inicio de todos los scripts. Nunca usar caracteres (→, ≤, ≥) en print statements
- **DataLoader Windows:** Siempre `num_workers=0`
- **torch.load:** Pasar `weights_only=False` para cargar dicts que incluyen tensores arbitrarios
- **AMP en CPU:** `use_amp=True` solo activa con CUDA; si CPU, scaler=None automatico
- **Cache .npz:** Usar `allow_pickle=False`; las claves son nombres de fichero (no rutas completas)
- **Output buffering:** Usar `python -u` para ver logs en tiempo real al lanzar con cmd /c
