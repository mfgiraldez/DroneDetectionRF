# Informe de Evaluacion: HybridCVCNN

> **Generado automaticamente** | 2026-04-24 05:59:23
> TFM: *Deteccion de Drones con IA Avanzada* | Dataset: NoisyUAV v2
> Arquitectura: **HybridCVCNN (CV-CNN + Physical Feature Fusion)**

---

## 1. Resumen Ejecutivo

HybridCVCNN combina el backbone Complex-Valued CNN con 12 features fisicas
extraidas del detector de entropia Shannon (duracion de burst, bins activos,
z-score estadistico, caida de entropia, etc.).

| Metrica | Valor |
|---|---|
| **Accuracy (Test)** | **0.8479** (84.79%) |
| **F1-Score (Test)** | **0.8452** |
| **AUC-ROC** | **0.9230** |
| Precision | 0.8594 (85.94%) |
| Recall | 0.8316 (83.16%) |
| Especificidad | 0.8641 (86.41%) |
| Baseline CV-CNN | ~75.00% |
| Mejora absoluta | **+9.79 p.p.** |

---

## 2. Arquitectura del Modelo

```
IQ crop [B, 2, 131072]  (~9.4ms a 14MHz)
|
+-- CV-CNN Backbone (4 ComplexConvBlock complex + modulus + AdaptivePool)
|   +-- ComplexConv1d(1->32,  k=31, s=2) + CReLU
|   +-- ComplexConv1d(32->64, k=15, s=2) + CReLU
|   +-- ComplexConv1d(64->128,k=7,  s=2) + CReLU
|   +-- ComplexConv1d(128->128,k=3, s=1) + CReLU
|   +-- modulus |z| -> [128, N'] -> AdaptivePool(32) -> Linear -> [256]
|   e_cnn [B, 256]
|
+-- Physical Features (detector de entropia Shannon)
    noise_floor, noise_sigma, mean_n_active, p75_n_active,
    H_min, H_mean, n_bursts, dur_ms, z_peak, drop_b, n_act, dur_total
    e_phys [B, 12]  (pre-computado, z-score normalizado)
    |
PAM Fusion: concat([e_cnn, e_phys]) -> MLP(268->256->128->1)
-> Logit [B, 1] (BCEWithLogitsLoss)
```

| Subsistema | Parametros |
|---|---|
| CV-CNN Backbone | 1,327,168 |
| Physical BN + MLP | 102,681 |
| **TOTAL** | **1,429,849** |

> IQ Crop Length: 131,072 samples (~9.4 ms @ 14 MHz)
> Tiempo total experimento: 9.94 horas

---

## 3. Configuracion del Experimento

```python
CROP_LEN    = 131072   # ~9.4 ms @ 14 MHz
BATCH_SIZE  = 32
EPOCHS      = 60 (early stopping en 50)
LR          = 0.0003
WEIGHT_DECAY= 0.0001
MIXUP_ALPHA = 0.3
MIXUP_PROB  = 0.5
GRAD_CLIP   = 2.0
```

---

## 4. Resultados del Entrenamiento

El modelo convergio en la **epoca 50** con Val F1 = 0.8388.

![Curvas de Entrenamiento](figures/fig_01_training_curves.png)

| Metrica | Train | Validacion |
|---|---|---|
| Loss (BCE) | 0.4493 | 0.3754 |
| Accuracy | 74.24% | 84.22% |
| F1-Score | -- | 0.8388 |

---

## 5. Resultados en el Conjunto de Test

### 5.1 Metricas Globales

![Matriz de Confusion](figures/fig_02_confusion_matrix.png)
![ROC Curve](figures/fig_04_roc_curve.png)
![Score Distribution](figures/fig_05_score_distribution.png)

### 5.2 Analisis por Nivel de SNR

![Metricas por SNR](figures/fig_03_per_snr_metrics.png)
![Accuracy por Grupo](figures/fig_06_group_accuracy.png)

| Grupo SNR | Rango | Acc Media | F1 Media |
|---|---|---|---|
| Grupo A (facil) | SNR >= 10 dB | 0.9023 | 0.9114 |
| Grupo B (medio) | -6..10 dB | 0.9182 | 0.9154 |
| Grupo C (dificil) | SNR < -6 dB | 0.6820 | 0.5910 |

#### Tabla completa por SNR

| SNR (dB) | Acc | Precision | Recall | F1 | n |
|---|---|---|---|---|---|
|  -20 | 0.5146 | 0.5455 | 0.2308 | 0.3243 | 103 |
|  -18 | 0.6275 | 0.8421 | 0.3137 | 0.4571 | 102 |
|  -16 | 0.5728 | 0.6333 | 0.3654 | 0.4634 | 103 |
|  -14 | 0.6373 | 0.7333 | 0.4314 | 0.5432 | 102 |
|  -12 | 0.7647 | 0.8857 | 0.6078 | 0.7209 | 102 |
|  -10 | 0.7745 | 0.8333 | 0.6863 | 0.7527 | 102 |
|   -8 | 0.8824 | 0.9333 | 0.8235 | 0.8750 | 102 |
|   -6 | 0.8137 | 0.8810 | 0.7255 | 0.7957 | 102 |
|   -4 | 0.9417 | 0.9412 | 0.9412 | 0.9412 | 103 |
|   -2 | 0.8922 | 0.9000 | 0.8824 | 0.8911 | 102 |
|   +0 | 0.8922 | 0.9167 | 0.8627 | 0.8889 | 102 |
|   +2 | 0.9320 | 0.9400 | 0.9216 | 0.9307 | 103 |
|   +4 | 0.9510 | 0.9423 | 0.9608 | 0.9515 | 102 |
|   +6 | 0.9519 | 0.9273 | 0.9808 | 0.9533 | 104 |
|   +8 | 0.9706 | 0.9615 | 0.9804 | 0.9709 | 102 |
|  +10 | 0.9706 | 0.9444 | 1.0000 | 0.9714 | 102 |
|  +12 | 0.9216 | 0.8644 | 1.0000 | 0.9273 | 102 |
|  +14 | 0.9327 | 0.8947 | 0.9808 | 0.9358 | 104 |
|  +16 | 0.9412 | 0.9245 | 0.9608 | 0.9423 | 102 |
|  +18 | 0.9020 | 0.8361 | 1.0000 | 0.9107 | 102 |
|  +20 | 0.9216 | 0.8644 | 1.0000 | 0.9273 | 102 |
|  +22 | 0.8824 | 0.8095 | 1.0000 | 0.8947 | 102 |
|  +24 | 0.8738 | 0.8065 | 0.9804 | 0.8850 | 103 |
|  +26 | 0.8725 | 0.7969 | 1.0000 | 0.8870 | 102 |
|  +28 | 0.8738 | 0.7969 | 1.0000 | 0.8870 | 103 |
|  +30 | 0.8333 | 0.7500 | 1.0000 | 0.8571 | 102 |

---

## 6. Comparacion con el Baseline

| Modelo | Acc Global | Grupo C (SNR<-6) | Parametros |
|---|---|---|---|
| **HybridCVCNN** | **84.79%** | **68.20%** | 1,429,849 |
| CV-CNN (baseline) | ~75.00% | ~55-65% | ~8,500,000 |
| MaRNet-Fusion     | 65.66%  | ~50.84% | 3,134,179 |

---

## 7. Conclusiones

1. **La fusion de features fisicas mejora** el baseline (+9.79 p.p.).
2. **El detector de entropia actua como preprocesado discriminativo**: las features de burst
   (dur_ms, n_act, z_peak) capturan informacion que la CNN por si sola no extrae.
3. **Recall y especificidad**: un mejor equilibrio entre ambos indica que el modelo no
   esta sesgado a predecir siempre la misma clase (problema de MaRNet-Fusion).

---

## Referencias

- Gluge et al. (2024). *Robust Low-Cost Drone Detection*. NoisyUAV v2.
- Bassey et al. (2021). *A Survey of Complex-Valued Neural Networks*. arXiv:2101.12249.
- Zhang et al. (2018). *MixUp: Beyond Empirical Risk Minimization*. ICLR 2018.

---
*Informe generado automaticamente por `run_hybrid_experiment.py`*
*Duracion total del experimento: 9.94 horas*