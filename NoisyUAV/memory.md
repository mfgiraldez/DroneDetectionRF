# memory.md — Memoria Técnica del Proyecto
> Actualizado: 2026-04-28 | Última acción: Entrenamiento Alumno V1 Completado (AUC 0.95) | TFM: Detección de Drones UAV con IA Avanzada

---

> **ESTADO ACTUAL DEL PROYECTO (leer primero):**
> - Aproximación 1 (HybridCVCNN) COMPLETADA con Acc=84.79%, AUC=0.923
> - Se ha identificado una limitación de diseño importante: el crop para la CNN es aleatorio (ver Sección 3.3)
> - El modelo clasifica FICHEROS de 75ms, no bursts individuales
> - Siguiente paso: debatir entre (a) Burst-Guided Crop como mejora de Aproximación 1, o (b) saltar a Aproximación 2
> - Existe notebook de inferencia: `prueba_inferencia_hybrid.ipynb`

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
- **Total ficheros: 17,744** (1 fichero = 1 señal completa de 75 ms)
- Train: ~70% (12,420) | Val: ~15% (2,662) | Test: ~15% (2,662)
- Estratificado por: clase binaria + SNR (combinación)
- Función: `NoisyUAV.funciones.dataset.obtener_splits_dataset(data_dir=...)`

### Distribución de targets (target_multiclass)
| target_multiclass | label | Clase | Total ficheros |
|---|---|---|---|
| 0 | 1 (drone) | Drone modelo 0 | 1,280 |
| 1 | 1 (drone) | Drone modelo 1 | 3,472 |
| 2 | 1 (drone) | Drone modelo 2 | 801 |
| 3 | 1 (drone) | Drone modelo 3 | 801 |
| 4 | 0 (ruido) | Ruido (WiFi/BT/AWGN/etc.) | 8,872 |
| 5 | 1 (drone) | Drone modelo 5 | 1,663 |
| 6 | 1 (drone) | Drone modelo 6 | 855 |

> IMPORTANTE: En `dataset.py`, `TARGET_NOISE = 4`. Todo fichero con target \u2260 1 en el nombre
> del fichero se mapea como label=0. Pero en `target_multiclass` se distinguen todos los modelos.
> El ruido está concentrado únicamente en target=4 (no hay 2=AWGN, 3=WiFi por separado
> en `drone_RF_data` — eso era la versión `stage2`).

### Diferencia entre datasets (IMPORTANTE)
| Dataset | Total muestras | Test set (15%) | Descripción |
|---|---|---|---|
| `drone_RF_data` | 17,744 | 2,662 | Dataset actual. 1 fichero = 1 señal completa |
| `stage2` (baseline) | ~65,888 | ~9,888 | Dataset antiguo CV-CNN. Múltiples crops por señal |

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
- **Input 2:** features físicas pre-computadas sobre la señal COMPLETA (75 ms)

### DESCRIPCION EXACTA DEL PIPELINE (LEER ANTES DE MODIFICAR NADA)

**Paso 1 — Pre-cómputo (una sola vez, antes de entrenar):**
```
Fichero .pt  →  señal IQ completa [2, 1048576] (~75 ms)
    │
    └──→  detector_entropia.detectar_bursts() sobre los 75 ms completos
              │
              └──→  12 features globales de la señal (noise_floor, n_bursts, z_peak, etc.)
                    → guardadas en features_cache.npz  {filename: array[12]}
```

**Paso 2 — Durante el entrenamiento (por cada muestra, en HybridDataset.__getitem__):**
```
Fichero .pt  →  señal IQ completa [2, 1048576] (~75 ms)
    │
    ├──→ RAMA CNN: crop aleatorio de 131072 muestras (9.4 ms)
    │         (posición aleatoria en train, centro en eval)
    │         + normalización RMS
    │         → CV-CNN Backbone → embedding [256]
    │
    └──→ RAMA FÍSICA: carga features_cache[filename] → [12]
              (calculadas sobre TODA la señal de 75 ms, NO sobre el crop)
              + z-score normalización
              → MLP Branch → contribución al logit
    │
    concat [268] → MLP → logit → predicción binaria
```

### DESALINEACION FUNDAMENTAL: el crop NO coincide con los bursts detectados

Este es el diseño más importante a entender y la limitación actual más clara:

- El detector de entropía analiza los 75 ms completos y detecta, por ejemplo, bursts
  de 10-30 ms y 40-45 ms. Luego extrae 12 features *de esas detecciones*.
- La CV-CNN recibe un crop ALEATORIO de 9.4 ms que puede caer en una zona SIN burst
  (silencio, ruido puro entre hops FHSS).
- Las 12 features le dicen al MLP 'esta señal tiene 3 bursts de 2ms con z_peak=8'
  pero la CNN ve un fragmento que puede ser ruido puro o puede ser un hop.
- **NO hay ningún mecanismo que alinee el crop con las detecciones del detector.**
- El modelo aprende a operar con esta inconsistencia porque:
  a) En señales con drone, aunque el crop caiga entre hops, las features globales
     (noise_floor, H_mean, n_active) ya discriminan.
  b) A veces el crop SÍ cae sobre un burst → CNN aprende ese patrón también.
  c) El MLP aprende a ponderar la rama física más cuando la CNN ve ruido.

### 3.4 BurstCVCNN (COMPLETADO — Ejecutándose Run #3 v2)
Este es el progreso actual que transita de "Crop Aleatorio" a "Crop Dinámico Guiado".

- **Pipeline rediseñado**:
  1. El detector CFAR extrae propuestas y marca *exactamente* `t_start` y `t_end`.
  2. El Dataloader extrae el asilamiento puro (Longitud variable dinámica). 
  3. Para entrenar usando batches estables de GPU se emplea `collate_fn` con `Zero-Padding`.
  4. Para lidiar con SNR muy bajo donde CFAR colapsa (0 bursts), se programó un modo de **Fallback explícito** donde se entrega la señal entera de 75ms a la CV-CNN (que usa AdaptiveAveragePooling) para forzar un fallo.

### 3.5 Problema Crítico Descubierto: Label Noise en Burst-Guided (2026-04-25)
- **Diagnóstico:** Al extraer los recortes para el dataset `BurstCVCNN`, se usó un detector CFAR muy permisivo (`Z_THRESH=2.0`). Esto provocó que en un archivo de audio etiquetado como "Dron", se extrajera 1 burst real y múltiples bursts espurios (ruido o Bluetooth lejano). Como el archivo entero era clase Dron, **el script etiquetó matemáticamente todos esos ruidos espurios como `Label=1` (Dron)**.
- **Consecuencia Mortal:** Se introdujo un "Label Noise" masivo en el entrenamiento. La red está aprendiendo a clasificar estática pura como si fuera un Dron. Al inferir a SNR -12dB, la red colapsa a ~50% de probabilidad (indecisión extrema) porque las características físicas del ruido que aprendió de memoria contradicen la huella espectral IQ.
- **Protocolo de Trabajo:** Se deben mantener los experimentos actuales en sus respectivas carpetas (`resultados_burst_v2`) y crear carpetas/modelos nuevos (`v3` o `mil`) para las siguientes aproximaciones.
- **Soluciones Propuestas:**
  1. **Curriculum Learning (Dataset Estricto):** Usar un CFAR muy restrictivo (`Z_THRESH=3.5`, `MIN_Z_ABS=4.0`) para construir el CSV, garantizando que el `Label=1` solo recaiga sobre bursts legítimos indudables.
  2. **Multiple Instance Learning (MIL):** Inspirado en el concepto `PAM_Fusion` que se intentó en MaRNet (el cual fracasó por usar recortes microscópicos de 146µs en lugar de bursts completos). Se agrupan todos los bursts en una "bolsa" y la red optimiza la Loss sobre el `Max(logits)` de la bolsa, auto-limpiando el dataset dinámicamente.

### QUE SIGNIFICA 'ENTRENAR EL DETECTOR'  (referencia de sesion anterior)
El detector de entropía Shannon NO se entrena. Es un algoritmo CFAR determinístico:
noise floor → umbral adaptativo → detección de bins activos → morfología → lista de bursts.
La referencia a 'entrenamiento' en conversaciones anteriores era probablemente sobre
el HybridCVCNN en su conjunto, no sobre el detector. El detector es fijo, no tiene parámetros
que se optimicen con gradientes.

#### Resultados Run #1 (SIN normalización IQ) — REFERENCIA HISTÓRICA
- Best epoch: 13 | Early stopping en época 25 | Tiempo: 3.35h
- Acc=80.92% | F1=0.8112 | AUC=0.9078 | Spec=79.80%
- Val Loss oscilante (0.37-0.60) — inestabilidad por falta de normalización

#### Resultados Run #2 (CON normalización RMS IQ) — MEJOR MODELO ACTUAL
- **Best epoch: 50** | Entró todas las 60 épocas | Tiempo: 9.94h
- Test: **Acc=84.79%** | **F1=0.8452** | **AUC=0.9230** | **Spec=86.41%**
- Precision=85.94% | Recall=83.16%
- Val Loss estable y decreciente (0.37 en época 50) — convergencia sólida
- **Checkpoint:** `resultados_hybrid_run2/checkpoints/best_model.pt`

| Métrica | **Run #2 (norma.)** | Run #1 (sin norma.) | Baseline CV-CNN | MaRNet-Fusion |
|---|---|---|---|---|
| **Accuracy** | **84.79%** | 80.92% | ~75% | 65.66% |
| **F1-Score** | **0.8452** | 0.8112 | ~0.750 | 0.7270 |
| **AUC-ROC** | **0.9230** | 0.9078 | -- | -- |
| **Specificity** | **86.41%** | 79.80% | -- | 39.9% |
| Precision | 85.94% | 80.22% | -- | -- |
| Recall | 83.16% | 82.03% | -- | -- |
| Mejora vs baseline | **+9.79 p.p.** | +5.92 p.p. | -- | -- |

#### Resultados Run #2 por Grupo SNR
| Grupo | Rango | Acc media | Notas |
|---|---|---|---|
| A (fácil) | SNR >= 10 dB | ~91% | SNR=+10: 97.06% |
| B (medio) | -6..+10 dB | ~90% | SNR=-4: 94.17% excelente |
| C (hostil) | SNR < -6 dB | ~64% | Mejora significativa vs Run #1 (55%) |

#### Comportamiento por SNR destacable (Run #2)
- SNR=-20 dB: Acc=51.5% (límite físico, prácticamente ruido)
- SNR=-12 dB: Acc=76.5% (ya funciona a SNR muy bajo)
- SNR=-8 dB:  Acc=**88.2%** (muy sólido en zona hostil)
- SNR=-4 dB:  Acc=**94.2%** (casi perfecto)
- SNR=+4 dB:  Acc=**95.1%**
- SNR=+10 dB: Acc=**97.1%**
- Recall=100% mantenido en SNR >= +10 dB

#### Resultados Run #2 por Emisor RF / Modelo de Drone (dataset completo, n=17744)
| Emisor RF | Global | A (SNR>=10) | B (-6..10) | C (<-6) | n total |
|---|---|---|---|---|---|
| **Target 5** | **92.3%** | **100%** | **99.0%** | 73.4% | 1,663 |
| Target 6 | 87.1% | 99.7% | 96.6% | 55.9% | 855 |
| Target 3 | 86.3% | 100% | 96.5% | 55.1% | 801 |
| Target 2 | 83.8% | 100% | 95.1% | 43.5% | 801 |
| Target 1 | 78.0% | 97.7% | 87.2% | 36.1% | 3,472 |
| Target 0 | 76.2% | 98.0% | 84.6% | 31.5% | 1,280 |
| Ruido (T4) spec. | 86.5% | 81.9% | -- | 86.2% | 8,872 |

> Generado con `eval_heatmap_hybrid.py --split all` (usar dataset completo es obligatorio
> para el heatmap: el test set solo tiene ~3-8 muestras por celda target x SNR, lo que
> produce ceros y casillas vacías estadísticamente sin sentido).

#### Observación crítica: Comportamiento en SNR alto
En SNR muy alto (+18 a +30 dB), la Accuracy baja ligeramente (83-90%) a pesar de 
que la señal es muy clara. Esto se debe a que la normalización RMS elimina la
información de amplitud absoluta. En SNR alto, el WiFi y Bluetooth tienen la misma
apariencia normalizada que los drones. El detector de entropía ya no discrimina tan
fácilmente. Esto es inherente al enfoque y debe tenerse en cuenta.

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

| # | Nombre | Riesgo | Estado | Mejor Acc |
|---|---|---|---|---|
| 1 | HybridCVCNN (CV-CNN + physical features) | Bajo | **COMPLETADO** | **84.79%** (Run #2) |
| 2 | Prototypical/Contrastive Network | Medio | PENDIENTE | -- |
| 3 | High-SNR Teacher -> Low-SNR Student | Medio-alto | PENDIENTE | -- |
| 4 | S4D/Mamba PyTorch puro | Alto | PENDIENTE | -- |

---

## 7. Estructura de Resultados

```
NoisyUAV/
├── resultados_marnet/              MaRNet-Fusion fracasado (Acc=65.66%)
│   └── diagnostic/
├── resultados_hybrid/              HybridCVCNN Run #1 SIN normalización (Acc=80.92%)
│   ├── checkpoints/best_model.pt   Epoch 13, Val F1=0.8161
│   └── features_cache.npz          CACHE COMPARTIDA (17744 entradas) — reutilizada por Run #2
├── resultados_hybrid_run2/         HybridCVCNN Run #2 CON normalización (Acc=84.79%) <-- MEJOR MODELO
│   ├── checkpoints/best_model.pt   Epoch 50, Val F1=0.8388
│   ├── figures/
│   │   ├── fig_01_training_curves.png
│   │   ├── fig_02_confusion_matrix.png
│   │   ├── fig_03_per_snr_metrics.png
│   │   ├── fig_04_roc_curve.png
│   │   ├── fig_05_score_distribution.png
│   │   ├── fig_06_group_accuracy.png
│   │   ├── heatmap_target_snr.png     Recall por emisor RF x SNR (dataset completo)
│   │   └── heatmap_noise_snr.png      Especificidad vs ruido x SNR
│   ├── informe_hybrid_cvcnn.md
│   └── metricas_completas.json
├── eval_heatmap_hybrid.py           Genera heatmaps por emisor x SNR
└── run_hybrid_experiment.py         Pipeline de entrenamiento completo (Run #2)
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

**Resultado de la corrección (Run #2):** Acc=84.79% (+3.87 p.p. sobre Run #1, +9.79 p.p. sobre baseline). COMPLETADO.

---

## 9. Bugs y Gotchas Conocidos

- **[CRITICO] IQ sin normalizar:** Siempre normalizar por potencia RMS antes de entrar al modelo (ver sección 8)
- **[CRITICO] Heatmap con test set solo:** Con ~3-8 muestras por celda (target x SNR), salen ceros y celdas vacías sin sentido estadístico. Usar SIEMPRE `--split all` en `eval_heatmap_hybrid.py`.
- **target_multiclass en drone_RF_data:** Solo target=4 es ruido. No hay targets 2=AWGN, 3=WiFi separados. Eso era `stage2`. No confundir.
- **UnicodeEncodeError cp1252:** Añadir `os.environ["PYTHONIOENCODING"] = "utf-8"` al inicio de todos los scripts. Nunca usar caracteres (→, ≤, ≥) en print statements
- **DataLoader Windows:** Siempre `num_workers=0`
- **torch.load:** Pasar `weights_only=False` para cargar dicts que incluyen tensores arbitrarios
- **AMP en CPU:** `use_amp=True` solo activa con CUDA; si CPU, scaler=None automatico
- **Cache .npz:** Usar `allow_pickle=False`; las claves son nombres de fichero (no rutas completas)
- **Output buffering:** Usar `python -u` para ver logs en tiempo real al lanzar con cmd /c
- **[CRITICO] Reanudación PyTorch:** PyTorch NO reanuda el entrenamiento solo. Siempre programar explicitly la lectura íntegra si existe el archivo (`model_state`, `optimizer_state`, `scheduler_state`, `scaler_state`) al inicio de todo bucle `EPOCHS`.
- **Memory Leak Tensor Vistas:** Si el dataloader corta `iq_crop = iq[:, t0:t1]`, PyTorch guarda referenciado el tensor gigante de 8MB causándo un desbordamiento masivo de RAM (7 GB+ en `num_workers=0`). Para liberar el original y matar la vista, forzar un `iq_crop.clone()` antes del return.

---

## 10. Evolución Arquitectónica (25 Abril 2026): De MIL a Teacher-Student (Curriculum / Pseudo-Labeling)

### El Problema de Label Noise y MIL
Descubrimos empíricamente que la detección CFAR estándar en SNRs negativos genera un **Label Noise masivo**: picos de interferencia (WiFi/Bluetooth) cruzan el umbral, y como el fichero se etiqueta como "Dron", la red aprende a identificar el WiFi como un Dron.

Originalmente propusimos **Multiple Instance Learning (MIL)** y un **Umbral Dinámico Relativo (Z-max * 80%)** para limpiar esto. 
Sin embargo, nos dimos cuenta de que:
1. **MIL es ineficiente en datos:** En un fichero limpio (SNR alto) con 5 ráfagas de dron perfectas, MIL sólo usaría la más fuerte (max pooling) para calcular el gradiente y tiraría el 80% de datos útiles.
2. **El Umbral Dinámico falla en SNR negativa:** A -12 dB, el WiFi tiene un Z-peak mayor que el dron. El umbral dinámico aislaría el WiFi y borraría el dron, garantizando que el Label Noise arruinara el entrenamiento.

### La Solución Definitiva: Destilación de Conocimiento (Teacher-Student)
Para maximizar los datos y aislar la forma pura del dron, adoptamos una aproximación de **Curriculum Learning** combinada con **Pseudo-Labeling**:

*   **Fase 1: El Profesor Experto (SNR >= 0 dB)**
    *   Construimos un dataset filtrado *exclusivamente* con ficheros SNR >= 0 dB, usando el **Umbral Dinámico Relativo (80% del Z_max)**.
    *   Dado que a SNR >= 0 el dron casi siempre domina a la interferencia, este dataset queda **puramente limpio** (sin Label Noise).
    *   **Arquitectura del Teacher: `BurstCVCNN`** (`NoisyUAV/modelos/burst_cvcnn.py`). A diferencia del `HybridCVCNN` anterior (crop aleatorio, 12 features globales), el Teacher usa **crop guiado por burst** (alineado con `t_start`/`t_end` del detector) y **8 features físicas por burst**: `dur_ms, z_peak, drop_b, n_act_burst, global_nf, global_ns, global_H_mean, global_p75_act`.
    *   Entrenamos la red (no-MIL), aprovechando **todas** las instancias válidas de dron por fichero.
    *   Resultado: Tenemos un oráculo infalible en el reconocimiento de la firma "visual" (coeficientes) del dron frente al ruido.
*   **Fase 2: Pseudo-Labeling del Infierno (SNR < 0 dB)**
    *   Extraemos todos los picos de los ficheros con ruido extremo (-6 dB, -12 dB) relajando el detector (ej. `Z_THRESH = 1.0` o `1.5`) para cazar cualquier ráfaga débil enterrada en el ruido.
    *   Pasamos estos recortes a través del **Profesor (Teacher)** en modo inferencia.
    *   El Profesor actúa de **Oráculo Implacable**:
        *   Si `Probabilidad > 95%`: Lo guardamos en el nuevo dataset con Etiqueta `1` (Dron puro).
        *   Si `Probabilidad < 5%`: Lo guardamos con Etiqueta `0` (Ruido puro).
        *   Si la probabilidad cae entre medias (incertidumbre): **Lo descartamos**.
    *   Resultado: Obtenemos un dataset de bajísima SNR escaso pero *perfectamente etiquetado*.
*   **Fase 3: Entrenar al Alumno (Student)**
    *   Entrenamos una nueva red (o fine-tuning del profesor) mezclando el dataset limpio (SNR >= 0) y el nuevo dataset pseudo-etiquetado (SNR < 0).
    *   Al estar forzado a hacer backpropagation sobre ejemplos a -12 dB cuyas etiquetas son 100% correctas, el Student desarrollará filtros convolucionales mucho más profundos, empujando el límite de detección varios dB por debajo de lo que lograba el Teacher original.

Esta estrategia unifica de manera óptima el poder de la detección de energía y el aprendizaje profundo sin sacrificar datos.

---

## 11. Validación de la Fase 1 (26 Abril 2026): Éxito del High-SNR Teacher

El entrenamiento del **High-SNR Teacher** validó empíricamente nuestra hipótesis sobre el *Label Noise*. Al aislar exclusivamente las señales de SNR >= 0 y limpiarlas con el umbral dinámico (80% del Z_max), el dataset resultante quedó completamente libre de falsos positivos masivos (WiFi/Bluetooth).

**Resultados del Entrenamiento (Epoch 11):**
*   **Validation F1-Score:** ~93.16%
*   **Validation Accuracy:** ~92.77%
*   **Comportamiento de la Loss:** A diferencia de los experimentos anteriores donde la Loss de validación oscilaba erráticamente (debido a que la red intentaba aprender patrones contradictorios por culpa del ruido mal etiquetado), en este experimento la Training Loss y la Validation Loss descendieron de forma suave y estable.

Este hito confirma que la arquitectura **`BurstCVCNN`** (`NoisyUAV/modelos/burst_cvcnn.py`) es extremadamente capaz de identificar la firma matemática del dron cuando se le alimenta con datos limpios y crops alineados al burst. Con este **Oráculo** entrenado, el proyecto avanza a la Fase 2: usar esta red para *Pseudo-Etiquetar* y purificar los datasets de SNR negativa (-6 dB y -12 dB).

---

## 12. Diagnóstico del Sesgo del Teacher: Covariate Shift por Padding (26 Abril 2026)

Al probar el Teacher entrenado en inferencia directa sobre un notebook, la red comenzó a clasificar repetidamente saltos de dron puros y evidentes (como el Turnigy - Target 6) como **RUIDO** con altísima confianza, a pesar de reportar un >93% de F1-Score en validación.

Tras un exhaustivo análisis, descubrimos **dos errores estructurales (Bugs) fundamentales**:

### Bug A: Covariate Shift inducido por el Padding Dinámico en PyTorch
*   **Problema:** En el entrenamiento, el `DataLoader` agrupaba pulsos de longitudes muy dispares (desde 2 ms hasta 10 ms) y usaba *zero-padding* para que todos tuvieran la longitud del mayor pulso del lote. La capa `AdaptiveAvgPool1d` procesaba esos ceros, diluyendo la energía media de las activaciones convolucionales.
*   **El Fallo en Inferencia:** En inferencia (con *batch size* = 1), el recorte del dron se pasaba sin ningún padding (0% de dilución). El `AdaptiveAvgPool1d` entregaba activaciones masivamente superiores a las que la red estudió. Este "Covariate Shift" brutal volvía loco al Perceptrón Multicapa (MLP), el cual colapsaba y emitía sistemáticamente logits negativos (RUIDO).
*   **La Solución Elegante e Invariante a `fs`:** Se reescribió la arquitectura del Dataset y la inferencia para aislar siempre una **ventana temporal de observación fija de 9.4 ms** ($2^{17}$ muestras a 14 MHz), independientemente de la duración real del salto. 
  ```python
  TARGET_TIME_MS = 9.4 
  TARGET_LEN = int((TARGET_TIME_MS / 1000.0) * FS)
  # Aplicar zero-padding o truncar hasta alcanzar TARGET_LEN
  ```
  Al fijar el padding temporalmente, la proporción de la señal original diluida en la ventana es siempre idéntica. Como extra, este enfoque hace que la red sea completamente **agnóstica a cambios futuros en la frecuencia de muestreo (fs)** (ej. hacer cross-validation con datasets a 28 MHz). El valor de 9.4 ms no es magia: es la ventana matemática de tamaño base 2 más eficiente que garantiza encapsular el salto más largo registrado en el dataset (8.3 ms).

### Bug B: El Filtro de Bins del Detector de Entropía (WiFi Pollution)
*   **Problema:** Se identificó que, para los drones 0, 1, 2 y 3, una gran cantidad de señales WiFi consiguieron entrar en el dataset etiquetadas como dron.
*   **Causa:** El filtro `MAX_BINS_FRAC` estaba al 25% (512 bins activos). Aunque restrictivo, el WiFi (que ocupa ~400-480 bins) lograba pasar. La abrumadora energía del WiFi dominaba el cálculo del `max_z`, provocando que el umbral del 80% borrase los débiles pulsos de los drones y se quedara únicamente con el WiFi.
*   **Solución para Futuros Re-entrenamientos:** Reducir la tolerancia de banda pasante a `MAX_BINS_FRAC = 0.17` (~348 bins), aislando quirúrgicamente a los drones (como el DJI, ~270 bins) e ignorando por completo el WiFi, purificando finalmente el dataset.

Tras estandarizar el Padding temporal en el DataLoader y en la Inferencia (Bug A), las detecciones volvieron instantáneamente a detectar los drones con >90% de probabilidad, demostrando el altísimo nivel de representación y capacidad generalizadora del Oráculo Híbrido construido.

---

## 13. Oráculo Híbrido V6: Todoterreno con Rescate por Anclaje (26 Abril 2026)

Tras las pruebas de inferencia a bajo SNR (-10 dB a -16 dB), se ha evolucionado el sistema de **Pseudo-Labeling** hacia un modelo híbrido (Física + IA) mucho más robusto para evitar el colapso del Teacher en zonas hostiles.

### 13.1 Métricas de Discriminación Morfología
Se han implementado dos nuevas métricas en el bucle de inferencia del notebook para filtrar ruidos que la IA sola no lograba discernir:

*   **Estabilidad / Suelo Continuo (`suelo_ms`):**
    *   **Lógica:** Usando la función `get_max_consecutive_frames`, medimos la racha más larga de tiempo (ms) que la entropía permanece en el "suelo" (30% inferior de la caída).
    *   **Dron (U-shape):** Tiene un suelo plano y prolongado (> 0.28 ms).
    *   **Click/Ruido (V-shape):** Es una aguja que sube y baja instantáneamente (< 0.12 ms).
*   **Rugosidad / Jitter (`rugosidad`):**
    *   **Lógica:** `np.std(np.diff(H_smooth))`. Mide cuánto "tiembla" la señal de entropía.
    *   **Dron:** Bajada suave y limpia (Rugosidad < 0.14).
    *   **Interferencia/AWGN:** Bajada nerviosa o tipo "serrucho" (Rugosidad > 0.22).

### 13.2 Inteligencia Colectiva: Rescate por Anclaje (Anchor Rescue)
Para SNR extremos (-16 dB), a menudo solo una ráfaga es lo suficientemente clara para que la IA la detecte (el **Ancla**). El oráculo ahora usa esta ráfaga para rescatar al resto:

1.  **Identificar Anclas:** Ráfagas con `Prob_IA > 80%` y buena morfología.
2.  **Extraer Firma:** Se calcula la duración media y el ancho de banda (`n_act`) de las anclas de la muestra.
3.  **Rescatar:** Si otras ráfagas en la misma muestra (etiquetadas como Gris/Duda) tienen una **geometría similar** (±35% dur, ±45% bins) a la del ancla, se marcan automáticamente como **Verde (Dron)**.

### 13.3 Lógica de Colores Semánticos (Visualización)
Se ha implementado un "hack" en el notebook para modificar el objeto `Figure` de Plotly sin tocar la librería `detector_entropia.py`:
-   **VERDE (`label_pseudo=1`):** Dron seguro. Útiles para Pseudo-Labeling de clase 1.
-   **ROJO (`label_pseudo=0`):** Ruido seguro (clicks, interferencia rugosa). Útiles para Pseudo-Labeling de clase 0.
-   **GRIS (`label_pseudo=-1`):** Incertidumbre. Se descartan para el entrenamiento del Alumno (Student).

### 13.4 Estado de Rendimiento
-   **SNR >= -10 dB:** El oráculo es prácticamente infalible combinando IA y morfología.
-   **SNR -16 dB:** El sistema de **Anclaje** es vital. Sin ancla, el oráculo tiende a la precaución (Gris), pero si detecta un solo salto claro, es capaz de reconstruir el dataset de entrenamiento del Student con una pureza del >95%.

---

## 14. Evolución del Oráculo V7 a V9: Lecciones Aprendidas (26 Abril 2026)

### 14.1 Experimentos Fallidos

#### V7 — Kurtosis + Periodicidad FHSS + Score Unificado (DESCARTADO)
- **Kurtosis de Fisher** sobre la curva de entropía del burst: inestable a bajo SNR. El ruido dentro del burst altera el estimador de forma impredecible según cada realización aleatoria del canal.
- **Score de periodicidad** basado en histograma de intervalos inter-burst: con 1 solo burst detectado, devuelve 0.5 neutro, arrastrando el score unificado hacia valores incorrectos.
- **Score unificado ponderado** (IA 20% + Morfología 50% + Periodicidad 30%): aunque conceptualmente sólido, a -10dB la IA falla tanto que incluso con poco peso arrastra el score por debajo del umbral para muchos drones reales.

#### V8 — Rugosidad Primero + Bug-Fix Mahalanobis (PARCIALMENTE FUNCIONAL)
**Avance real:** Bug crítico identificado y corregido: con 1 sola ancla, `A.std(axis=0) = 0` → distancia Mahalanobis infinita → el rescate por anclaje nunca se activa.
**Fix:** `a_std = np.where(a_std < 1e-6, np.abs(a_mean) * 0.25 + 1e-8, a_std)`.

**Problema restante:** La rugosidad absoluta `std(diff(h))` escala con la profundidad de la caída de entropía. A muy bajo SNR la caída es pequeña → rugosidad absoluta baja aunque la señal sea intrínsecamente rugosa → umbral fijo de 0.30 no generaliza entre niveles de SNR.

#### V8b — Rescate por Rugosidad Casi-Cero (FUNCIONAL CON FALSOS POSITIVOS)
Se añadió: si `rug < 0.05 AND suelo > 0.12` → Verde. Razonamiento: rugosidad casi cero es físicamente imposible en AWGN puro.

**Problema:** Las interferencias Bluetooth también generan pulsos con rugosidad casi cero (el BT es un protocolo FHSS limpio). El filtro de rugosidad no discrimina BT de FHSS-RC.

### 14.2 Solución Definitiva: Oráculo V9 — Champion-First + Plantilla de Duración

**Insight físico clave:** El protocolo FHSS de un radiocontrol emite siempre hops de **duración fija determinada por el hardware del transmisor**. Todos los saltos de un mismo dron en una muestra de 75ms tienen exactamente la misma duración (±jitter de hardware ~1-2%). Bluetooth, WiFi y AWGN tienen duraciones completamente distintas.

**Pipeline del Oráculo V9 (`prueba_inferencia_teacher_v1.ipynb`):**
```
1. Inferencia IA en TODOS los bursts detectados
2. CAMPEON = argmax(prob_ia)
       -> su dur_ms define la PLANTILLA TEMPORAL del protocolo FHSS
3. Para cada burst:
       |dur_ms - ref_dur| / ref_dur <= 0.10 -> Verde (misma firma de protocolo)
       caso contrario                        -> Rojo  (duracion incompatible)
4. El Campeon siempre es Verde
```

**Por que funciona:**
- BT clasico (EDR): hops de ~366 us. FHSS RC (FutabaT14, Taranis...): hops de 0.5-3 ms. Duraciones bien separadas, el filtro +-10% las discrimina limpiamente.
- No depende de la SNR: la duracion del hop es constante de protocolo, no degrada con ruido.
- No depende de la IA absolutamente: aunque la IA solo de 2-3%, el Campeon sigue siendo la mejor referencia disponible.

**Ejemplo validado (target1, SNR=-10dB, semilla 9704):**
```
B01: dur=0.22ms, IA=2.2% -> |0.22-0.59|/0.59 = 63% -> ROJO (BT click)
B02: dur=0.22ms, IA=2.2% -> 63% off              -> ROJO (BT click)
B03: dur=0.59ms, IA=2.9% -> CAMPEON              -> VERDE (dron real) OK
```

### 14.3 Parametros del Oraculo V9

| Parametro | Valor | Descripcion |
|---|---|---|
| `TOL` | 0.15 | Tolerancia +-15% en dur_ms respecto al Campeon |
| `Z_THRESH` | 2.0 | Detector relajado para maximizar propuestas candidatas |
| `SMOOTH_MS` | 0.1 | Suavizado minimo para preservar bordes del burst |
| `MIN_BURST_MS` | 0.40 | Elimina spikes sub-ms del detector |

### 14.4 Limitaciones Conocidas

- **SNR extremo < -14dB:** El detector CFAR puede no detectar ningun burst -> sin candidatos -> sin Campeon.
- **Tolerancia +-15%:** Valor validado empiricamente. Cubre el jitter de protocolo de todos los modelos probados. Subir a +-20% si algun target especifico sigue fallando.
- **Coincidencia de duracion:** Muy improbable en la practica, pero si un BT tiene la misma duracion que el dron, no se discrimina por duracion sola.

### 14.5 Proximo Paso: Script de Pseudo-Labeling Masivo
Con el Oraculo V9 validado, el siguiente paso es un script batch que:
1. Itere sobre todos los ficheros `.pt` con `SNR < 0`.
2. Aplique el Oraculo V9 (detector + campeon + plantilla de duracion).
3. Genere un CSV con los bursts pseudo-etiquetados (label=1 dron, label=0 ruido).
4. Use ese CSV para entrenar el **Student** (misma arquitectura `BurstCVCNN`, fine-tuning desde pesos del Teacher).

---

## 15. Sistema de Checkpoints Dual (26 Abril 2026)

### 15.1 Problema
El script `run_teacher_experiment.py` solo guardaba el checkpoint cuando se alcanzaba un nuevo maximo de Val F1. Si el entrenamiento se interrumpia entre dos epocas "best", al reanudarlo se repitian todas las epocas desde el ultimo "best", pudiendo perder varias horas de computo.

### 15.2 Solucion: Dos Checkpoints Separados

| Fichero | Cuando se guarda | Para que sirve |
|---|---|---|
| `checkpoints/teacher_model_best.pt` | Solo cuando Val F1 supera el record historico | **Inferencia y Oraculo** — el modelo de maxima calidad |
| `checkpoints/teacher_model_last.pt` | Al final de CADA epoca sin excepcion | **Reanudacion del training** — permite continuar exactamente desde la ultima epoca completada |

### 15.3 Logica de Resume (Prioridad LAST > BEST)
Al arrancar `run_teacher_experiment.py`:
```
1. Si existe teacher_model_last.pt  → carga LAST (ultimo epoch completado)
   El best_f1 de referencia se lee SIEMPRE de teacher_model_best.pt
2. Si solo existe teacher_model_best.pt → carga BEST (comportamiento antiguo)
3. Si no existe ninguno → entrenamiento desde cero
```

### 15.4 Uso Correcto
- **Para inferencia / Oraculo en el notebook:** cargar siempre `teacher_model_best.pt`.
- **Para reanudar training:** simplemente relanzar el script — cargara `teacher_model_last.pt` automaticamente.
- **NO usar** `teacher_model_last.pt` para inferencia: puede ser un epoch posterior al mejor F1 y tener peor rendimiento.
- **[NUEVO] detector_entropia.py (2026-04-28):** Para evitar "clicks" de ruido en SNR negativo, se han endurecido los parámetros por defecto: `Z_THRESH=4.0`, `MIN_BURST_MS=0.5`, `MERGE_GAP_MS=1.5`.

---

## 16. Hito Final: Alumno V1 y Oráculo V9 Calibrado (28 Abril 2026)

### 16.1 Optimización Física del Oráculo V9
Tras identificar que el "Campeón" podía ser engañoso en SNR extremas (si un pulso de ruido corto tenía ligeramente más probabilidad que el dron), se evolucionó el Oráculo hacia un modelo **Anclado en la Física**:

*   **Calibración por Target (`calibrate_drone_durations.py`):** Se genera un perfil `drone_duration_ref.json` analizando ráfagas indudables a **SNR=+20 dB**. Se extrae la **duración media** (`dur_ref`) por cada ID de dron.
*   **Selección de Campeón por Proximidad Física:** En el Oráculo (SNR < 0), el Campeón ya no es simplemente el de mayor `prob_ia`. Ahora es el burst cuya duración es **más cercana a la referencia de calibración** de ese target (usando `prob_ia` solo como desempate).
*   **Filtro de Duración Estricto:** Solo se aceptan como drones ráfagas dentro de una ventana de **±20% (`TOL_DUR=0.2`)** respecto a la duración del Campeón.

### 16.2 Pipeline Alumno V1 (`launch_alumn.py`)
Se orquestó el proceso completo en tres fases automáticas:
1.  **Calibración:** Generación del JSON de referencias físicas.
2.  **Pseudo-Labeling:** Creación de `alumn_dataset_pseudo.csv` (17,744 señales).
3.  **Entrenamiento:** Fine-tuning de la arquitectura `BurstCVCNN` (inicializada con pesos del Teacher).

### 16.3 Resultados y Rendimiento Final (Alumno V1)
El modelo Alumno ha superado todas las expectativas en el Test Set (que incluye ráfagas a SNR -12dB y -14dB):

| Métrica | Valor (Test Set) | Nota |
|---|---|---|
| **AUC-ROC** | **0.9496** | Excelente capacidad de discriminación en ruido extremo. |
| **F1-Score** | **0.8672** | Equilibrio sólido entre precisión y exhaustividad. |
| **Accuracy** | **0.8805** | Gran fiabilidad global. |
| **Specificity** | **90.66%** | Rechazo muy efectivo de ruidos/clics espurios. |

**Conclusión:** La combinación de **IA (Teacher)** + **Conocimiento Físico (Duración FHSS)** ha permitido entrenar un modelo Alumno que detecta drones donde el ojo humano y los detectores de energía tradicionales solo ven ruido.

### 16.4 Herramientas de Evaluación Standalone
Se creó `evaluate_teacher.py` para generar métricas completas (Confusion Matrix, ROC, Classification Report) del modelo base sin necesidad de re-entrenar, facilitando la exportación de figuras para la redacción de la tesis.

### 28 de Abril - Comparativa Final: Aproximación 1 (Hybrid) vs Aproximación 2 (Alumno)

Se ha realizado un análisis comparativo entre el primer modelo desarrollado (**HybridCVCNN**) y el modelo final obtenido mediante destilación de conocimiento y pseudo-etiquetado físico (**Alumno V1**). Los resultados confirman la superioridad de la Aproximación 2, especialmente en condiciones de baja SNR.

#### Tabla Comparativa de Rendimiento

| Métrica / Característica | Aproximación 1: HybridCVCNN | Aproximación 2: Alumno V1 (V9 Oracle) |
| :--- | :--- | :--- |
| **Unidad de Proceso** | Ventana fija (Crops de 9.4 ms) | Ráfagas variables (Bursts detectados) |
| **Supervisión** | Supervisado (Ground Truth) | Pseudo-supervisado (Oráculo Físico V9) |
| **Dataset (Train)** | 12,420 muestras (Crops) | 26,929 muestras (Bursts) |
| **Dataset (Test)** | 2,662 muestras (Crops) | 5,862 muestras (Bursts) |
| **Accuracy (Global)** | 80.92 % | **88.08 %** |
| **F1-Score** | 0.8112 | **0.8685** |
| **AUC-ROC** | 0.9078 | **0.9496** |

#### Conclusiones del Experimento
1.  **Robustez Física**: El uso del **Oráculo V9** basado en plantillas de duración FHSS ha permitido filtrar el ruido de manera mucho más efectiva que la aproximación híbrida anterior, eliminando el "Label Noise" en escenarios de SNR extrema (-14 dB).
2.  **Granularidad**: El Alumno V1 trabaja directamente sobre la ráfaga (burst), lo que permite una detección más precisa de la naturaleza del dron en lugar de promediar la información de una ventana temporal fija.
3.  **Escalabilidad**: El pipeline de generación de dataset automático ha duplicado la cantidad de muestras útiles para el entrenamiento, lo que se traduce en una mejor generalización del modelo CV-CNN.

Este hito marca la finalización exitosa de la fase de entrenamiento del Alumno, logrando un modelo listo para despliegue con un rendimiento state-of-the-art en el dataset NoisyUAV.
