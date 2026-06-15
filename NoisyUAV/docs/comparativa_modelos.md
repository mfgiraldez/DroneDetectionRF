# Evolución y Comparativa de Arquitecturas de Detección de Drones por RF

Este documento detalla el desarrollo e iteración de los modelos y conjuntos de datos diseñados para la detección y clasificación de drones por radiofrecuencia (RF) en entornos ruidosos (NoisyUAV). El objetivo es documentar de forma rigurosa y académica la evolución de las arquitecturas en su orden cronológico de desarrollo (V1 $\to$ V2 $\to$ V3 $\to$ V4 $\to$ V2.1 Golden $\to$ V5) para la memoria del Trabajo de Fin de Máster (TFM).

---

## 1. Tabla Comparativa de Arquitecturas y Datasets (Orden Cronológico)

La siguiente tabla resume los aspectos fundamentales de cada variante en la secuencia temporal real en que fueron concebidas y validadas.

| Variante / Paso | Estructura de Red | Dataset y Filtros de Train | Vector de Entrada Física | Aumentación AWGN y Tratamiento del Z-score | Resultados en Test / Golden Set |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Teacher V1** | `BurstCVCNN` (Canal único Complex-Valued Conv1D + MLP física). Canales máx: 128. | Datos sintéticos puros, alta SNR, sin interferencias ni desvanecimiento. | 8 variables locales y globales (duración, Z-peak, drop, etc.). | Sin aumentación de ruido térmico artificial. | Oráculo espectro-temporal utilizado exclusivamente para pseudo-etiquetado. |
| **1. Alumno V1** | `BurstCVCNN` (Inicializado con pesos de Teacher V1 + Fine-tuning). | `alumn_dataset_pseudo_v3.csv`. Excluye fallbacks de baja SNR ($\le -14\text{ dB}$). | 8 variables normalizadas con las medias estadísticas del Teacher. | Rotación de fase IQ aleatoria $\theta \in [0, 2\pi]$ y ruido de base estático de potencia 0.02. | Detección aceptable a SNR alta/media; colapso de precisión en baja SNR. |
| **2. Model V2** | `DualStreamCVCNN` (Streams paralelos IQ y Log-PSD Welch + `AttentionFusion`). | Balanceo por submuestreo de ráfagas. Detección por bursts discretos. | 3 variables de contexto (`global_nf`, `global_H_mean`, `z_peak`). | Aumentación AWGN adaptativa en train (SNR uniforme de $-20$ a $-8\text{ dB}$). | Gran mejora espectro-temporal; sensible a falsas alarmas espectrales WiFi. |
| **3. Model V3a** | `DualStreamCVCNN` (MLP física de 4 entradas). Streams idénticos a V2. | `dataset_v6_pointers.csv` (Enriquecido con `n_bins_peak` vía acoplamiento temporal). | 4 variables (añade `n_bins_norm` = bins activos / 2048). | Aumentación AWGN clásica con re-estimación espectral. | Mayor capacidad discriminatoria WiFi, pero vulnerable a ruido térmico puro. |
| **3. Model V3b** | `DualStreamCVCNN` (MLP de 4 entradas + Entrenamiento robusto GMM). | `dataset_v6_pointers.csv`. Warmup de 5 épocas, luego ponderación GMM. | 4 variables de entrada. | Aumentación clásica combinada con blindaje de gradiente. | Alta robustez ante falsas alarmas y ráfagas mal etiquetadas en entrenamiento. |
| **4. Model V4** | `DualStreamCVCNN_V4` (Streams duplicados: IQ a 512, PSD a 256. Fusión multicapa). | `dataset_v4_train_val.csv` (Hard Negative Mining; exclusión estricta de Golden Set). | 4 variables normalizadas estáticamente por divisores constantes. | **Atenuación analítica:** $z\_peak = z\_peak \times 10^{\frac{\text{SNR}_{\text{target}}}{20.0}}$ (atenuación lineal). | **Fracaso empírico.** Caída masiva de Recall por colapso del CFAR teórico en SNR extrema. |
| **5. Model V2.1 (Golden)** | `DualStreamCVCNN` (Identidad convolucional a V2; Inferencia sliding window). | `dataset_v2_1_clean_pointers.csv` (Mismo train de V2). Inferencia con Golden Set (3,744 ficheros). | 3 variables de contexto, inyectadas de forma global a nivel de fichero (75 ms). | **AWGN Z-Score Reset:** `z_peak` se fuerza a `0.0` si la muestra se degrada con ruido. | **90.05% F1-score** y **96.29% Especificidad** en Golden Set. Modelo óptimo de producción. |
| **6. Model V5** | `GatedAttentionMIL` (Bag-Level Multiple Instance Learning con codificador dual). | Bolsas de ráfagas. Pre-filtro LightGBM con umbral de confianza $\ge 0.6$. | 11 variables físicas extraídas localmente por ráfaga. | Aumentación de ruido al 30% a nivel de instancia dentro de la bolsa. | Limitado a SNR media; colapso en SNR extrema por criba excesiva de LightGBM. |
| **7. Model V2.2 (Dynamic)**| `DualStreamCVCNN` (Identidad convolucional a V2). | `dataset_v2_1_clean_pointers.csv`. Data augmentation sobre ventana de 9.4 ms. | 3 variables de contexto (`global_nf`, `global_H_mean`, `z_peak`). | **Recálculo CFAR Dinámico:** Se inyecta AWGN y se vuelve a calcular matemáticamente el $Z\_score$ y nivel de ruido en tiempo real. | En desarrollo (Paradigma Teórico Puro). |

---

## 2. Análisis Técnico Detallado de las Arquitecturas y Datasets

### 2.1. Paso 1 — Teacher V1 y Alumno V1 (La Aproximación Monocanal Compleja)

#### Diseño Arquitectónico (CVCNN monocanal)
La arquitectura V1 se fundamenta en el procesamiento directo del dominio del tiempo mediante la preservación de la relación de fase IQ utilizando operaciones de convolución complejas. La red implementa el backbone `CVCNNBackbone`, compuesto por cuatro bloques de convolución compleja (`ComplexConvBlock`) con las siguientes dimensiones y parámetros:
1.  **Complejo Conv 1:** Entrada de 1 canal complejo (magnitud real e imaginaria mapeadas por separado en la operación convolucional), 32 canales de salida, tamaño de filtro de 11, zancada (stride) de 2 y relleno (padding) de 5.
2.  **Complejo Conv 2:** Entrada de 32 canales complejos, 64 canales de salida, kernel de 11, stride de 2, padding de 5.
3.  **Complejo Conv 3:** Entrada de 64 canales complejos, 128 canales de salida, kernel de 11, stride de 2, padding de 5.
4.  **Complejo Conv 4:** Entrada de 128 canales complejos, 128 canales de salida, kernel de 11, stride de 1, padding de 5.

Tras la extracción convolucional, se extrae el módulo de la señal compleja resultante y se reduce la dimensionalidad temporal a través de un promedio global adaptativo (`nn.AdaptiveAvgPool1d(32)`). La representación aplanada resultante alimenta una capa lineal que genera un embedding temporal de 256 dimensiones (`cnn_embed`). 

Paralelamente, un MLP de variables físicas proyecta un vector de 8 variables normalizadas (`dur_ms`, `z_peak`, `drop_b`, `n_act_burst`, `global_nf`, `global_ns`, `global_H_mean`, `global_p75_act`) mediante normalización por lote (`BatchNorm1d`). El embedding temporal y el físico se concatenan, dando paso a un clasificador MLP final (`Linear(264, 256) -> BatchNorm -> ReLU -> Dropout(0.4) -> Linear(256, 128) -> BatchNorm -> ReLU -> Dropout(0.4) -> Linear(128, 1)`).

#### Estructura y Limpieza del Dataset (`alumn_dataset_pseudo_v3.csv`)
Para el Alumno V1, el conjunto de datos de ráfagas ruidosas reales requería un preprocesamiento físico de segmentación sumamente riguroso debido a la inestabilidad de la señal a baja SNR:
*   **Segmentación Espectro-Temporal Adaptativa:** Para extraer los recortes temporales del dataset masivo en bruto, se implementaron umbrales adaptativos en la función de detección de entropía espectral. La sensibilidad del Z-score límite del CFAR se calibró en función de la SNR del archivo:
    *   $\text{SNR} \ge 0\text{ dB}$: $Z\_score = 4.0$ (máxima pureza espacial del salto FHSS).
    *   $-10\text{ dB} < \text{SNR} < 0\text{ dB}$: $Z\_score = 2.5$.
    *   $\text{SNR} \le -10\text{ dB}$: $Z\_score = 1.8$ (permite capturar variaciones leves de entropía en ráfagas sumergidas en ruido).
*   **Gestión de Fallbacks:** En SNR hostiles, la ceguera del detector analítico provocaba que archivos completos de dron quedasen sin representación (0 bursts detectados). Se programó una estrategia de fallback que incluía la macroventana completa de 75 ms ($1,050,000$ muestras a $14\text{ MHz}$) con etiqueta real de dron (`fallback = True`), asegurando representatividad en el conjunto de entrenamiento.
*   **Filtrado por Oráculo (Teacher V1):** Para purificar las ráfagas a baja SNR, se cargaron los tiempos de salto FHSS teóricos en `drone_duration_ref.json`. Se aisló el burst "campeón" de cada archivo (el que mejor encajaba en duración y mayor verosimilitud daba en el Teacher). Las ráfagas restantes del mismo fichero que diferían más de un $\pm 20\%$ en duración frente al campeón fueron reclasificadas a `0` (ruido térmico/interferencia espectral), reduciendo el ruido de etiquetas en train.
*   **Protección del Alumno:** En el entrenamiento final del Alumno, se excluyeron sistemáticamente los fallbacks de dron a SNR muy baja ($\le -14\text{ dB}$) para evitar que el clasificador asociara el ruido blanco puro del receptor con la clase dron.

---

### 2.2. Paso 2 — Model V2 (La Fusión Dinámica Espectro-Temporal)

#### Diseño Arquitectónico (Dual-Stream CVCNN)
Para mitigar la dispersión espectral introducida por el ruido térmico en el dominio del tiempo, se descartó el diseño monocanal y se implementó un modelo de doble flujo paralelo (Dual-Stream):
1.  **Vía Temporal (Raw IQ) - El Kernel Masivo de Integración:** Diseñada para detectar patrones de amplitud y fase a nivel de muestra cruda. Su componente más crítico es la **primera capa convolucional con un filtro gigante de integración temporal (`kernel_size=128`, `stride=4`, `padding=64`)**. 
    *   *Fundamento Físico:* A diferencia de las redes de visión que usan kernels pequeños ($3 \times 3$) para buscar texturas finas desde el inicio, en RF el ruido térmico blanco provoca fluctuaciones masivas de alta frecuencia a nivel de muestra. Este "kernel masivo" actúa como un promediador de paso bajo: toma 128 muestras de golpe e integra su energía, "suavizando" la nieve electromagnética y permitiendo extraer la envolvente de potencia general de la señal subyacente.
    *   *Capas Profundas:* Una vez filtrado el ruido base, las capas sucesivas pueden buscar patrones más finos utilizando kernels progresivamente más pequeños: `Conv1d(k=31, stride=2)`, `Conv1d(k=7)` y `Conv1d(k=3)`, todas con `BatchNorm1d` y `MaxPool1d(4)`. La dimensión temporal se colapsa finalmente mediante un pooling global promedio (`AdaptiveAvgPool1d(1)`) que genera un embedding temporal de 256 dimensiones, proyectado linealmente a 128 (`iq_proj`).
2.  **Vía Espectral (GPU Log-PSD):** Diseñada para buscar la firma espectral estática de los saltos FHSS. Calcula la densidad espectral de potencia (Welch Log-PSD) en tiempo real en la GPU a partir de las muestras IQ aplicando una ventana de Hann de 131,072 muestras, ejecutando una FFT de alta resolución, aplicando un centrado espectral (`fftshift`), un agrupamiento por promedio (`avg_pool1d` con zancada y ventana de 64 muestras para reducir la dimensionalidad de 131,072 a 2048) y conversión a escala logarítmica ($10 \log_{10}(P + 10^{-12})$). La salida resultante ($1 \times 2048$ bins espectrales) es procesada por una CNN 1D independiente con reducción progresiva: `Conv1d(1, 32, k=15)` $\to$ `MaxPool1d(2)` $\to$ `Conv1d(32, 64, k=7)` $\to$ `MaxPool1d(2)` $\to$ `Conv1d(64, 128, k=3)`. Un pooling adaptativo final produce un embedding espectral de 128 dimensiones.

#### Fusión por Pesos de Atención (`AttentionFusion`)
Los embeddings temporal y espectral se fusionan a través del módulo de atención `AttentionFusion`. El vector concatenado ($256$ dimensiones) pasa por una capa lineal intermedia (`Linear(256, 128) -> ReLU -> Linear(128, 2) -> Softmax`). Los dos coeficientes de salida ($w_{\text{IQ}}$ y $w_{\text{PSD}}$) ponderan dinámicamente cada embedding mediante una suma ponderada:
$$\mathbf{e}_{\text{fusion}} = w_{\text{IQ}} \cdot \mathbf{e}_{\text{IQ}} + w_{\text{PSD}} \cdot \mathbf{e}_{\text{PSD}}$$
Este vector fusionado de 128 dimensiones se concatena con la salida del MLP físico (un clasificador lineal de 3 entradas proyectado a 16 dimensiones que procesa `global_nf`, `global_H_mean` y `z_peak`) y alimenta al clasificador MLP de salida.

#### Composición del Dataset y Aumentación
El dataset de entrenamiento de la V2 se estructuró a nivel de instancias temporales de 9.4 ms balanceadas por submuestreo.
*   **Aumentación AWGN Dinámica (Train):** Durante el bucle de entrenamiento, se inyecta ruido gaussiano complejo (AWGN) a SNRs continuas y uniformes de $-20$ a $-8\text{ dB}$ para robustecer la extracción espectro-temporal de ráfagas individuales.

---

### 2.3. Paso 3 — Model V3a y Model V3b (La Incorporación del Ancho de Banda y Robustez GMM)

#### Diseño Arquitectónico
Ambos modelos mantienen exactamente el backbone Dual-Stream (IQ + PSD) de la V2. La única modificación en el diseño de la red radica en la expansión de la primera capa del MLP físico, que pasa de recibir 3 variables a recibir **4 variables físicas** (`Linear(4, 16)`):
$$\mathbf{f}_{\text{phys}} = [\text{global\_nf}, \text{global\_H\_mean}, \text{z\_peak}, \text{n\_bins\_norm}]$$
Donde `n_bins_norm` representa el ancho de banda ocupado por la ráfaga normalizado respecto al tamaño de la FFT de Welch ($\text{n\_bins\_peak} / 2048 \in [0, 1]$).

#### Dataset de Entrenamiento (`dataset_v6_pointers.csv`)
Para entrenar estas variantes, se enriquecieron las ráfagas del dataset original mediante una correspondencia temporal rigurosa:
*   Se ejecutó `detectar_bursts` sobre los archivos con un $Z\_score = 4.0$ estricto para extraer las variaciones locales del espectro.
*   Para cada ráfaga indexada en el CSV, se buscó su `t_center` y se emparejó con las ráfagas reales detectadas por el CFAR dentro de un umbral de tolerancia temporal estricto de $\pm 2\text{ ms}$. 
*   Al producirse el emparejamiento, se extrajo el vector espectral `n_active` y se asignó el valor máximo de bins de frecuencia activos detectados en ese intervalo a la columna `n_bins_peak`. En ráfagas fallback o sin emparejamiento, se asignó `0.0`.

#### Bucle de Entrenamiento Robusto V3b (GMM DivideMix)
La variante V3b implementa un bucle de entrenamiento robusto contra el ruido de etiquetas (Label Noise) común en entornos RF reales (ej. ráfagas de WiFi transitorias etiquetadas accidentalmente como ruido o viceversa):
*   **Warmup Espectral:** Durante las primeras 5 épocas, la red se entrena de forma convencional con pérdidas BCE crudas para que los filtros convolucionales aprendan las firmas espectrales básicas.
*   **Ajuste del Modelo de Mezcla Gaussiana (GMM):** A partir de la época 6, al final de cada iteración, se calculan las pérdidas individuales de cada muestra del lote. Se entrena un GMM de 2 componentes (una gaussiana de pérdidas bajas para muestras limpias y otra de pérdidas altas para ráfagas corruptas) sobre este vector de pérdidas.
*   **Ponderación del Gradiente:** Para cada muestra $i$, el GMM devuelve la probabilidad a posteriori de pertenecer a la componente de pérdidas bajas ($w_i$). La pérdida final que propaga el gradiente se escala dinámicamente por este peso:
    $$\mathcal{L}_{\text{final}} = \frac{1}{B} \sum_{i=1}^{B} w_i \cdot \text{BCE}(\hat{y}_i, y_i)$$
    De esta forma, las ráfagas con un etiquetado erróneo o con colisiones espectrales severas WiFi/Dron no impactan negativamente en la actualización de los pesos del modelo.

---

### 2.4. Paso 4 — Model V4 (La Trampa de la Atenuación Teórica Sobredimensionada)

#### Diseño Arquitectónico (DualStreamCVCNN_V4)
El modelo V4 intentó solucionar las limitaciones de detección aumentando masivamente la capacidad de representación convolucional de la red, duplicando el número de canales en ambos streams:
*   **Stream IQ (Tiempo):** Se expandió el número de canales de los bloques convolucionales secuencialmente (`Conv1d(2, 64, k=128)` -> `Conv1d(64, 128, k=31)` -> `Conv1d(128, 256, k=7)` -> `Conv1d(256, 512, k=3)`). La salida se reduce a un embedding temporal de 512 canales, el cual se proyecta linealmente a 256.
*   **Stream PSD (Frecuencia):** Se incrementaron los canales a `Conv1d(1, 64, k=15)` -> `Conv1d(64, 128, k=7)` -> `Conv1d(128, 256, k=3)`, generando un embedding espectral de 256 canales.
*   **AttentionFusion Multicapa:** Se potenció el módulo de atención añadiendo normalización por lote y abandono regulado (`BatchNorm1d` y `Dropout(0.2)`) en el MLP de atención interna (`Linear(512, 256) -> BatchNorm -> ReLU -> Dropout -> Linear(256, 64) -> ReLU -> Linear(64, 2)`).
*   **Clasificador Final Profundo:** La salida fusionada de 256 canales se concatena con un MLP físico expandido (`Linear(4, 32) -> ReLU -> BatchNorm -> Linear(32, 16) -> ReLU`) y se clasificó mediante un clasificador final de tres capas (`Linear(272, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 64) -> ReLU -> Linear(64, 1)`).

#### Dataset de Entrenamiento (`dataset_v4_train_val.csv` - Hard Negative Mining)
Se diseñó un conjunto de datos específico enfocado en forzar a la red a discriminar las fronteras más complejas de la señal mediante minería de negativos difíciles:
*   **Silencios Intradron (Negativos Difíciles):** Para ficheros reales de dron a alta SNR ($\ge 8\text{ dB}$), se aislaron los intervalos temporales vacíos de al menos 9.4 ms donde no había transmisión FHSS activa. Se extrajo aleatoriamente **1 ventana de silencio por fichero** etiquetada como `0` (`type: 'silence_in_drone'`), forzando al clasificador a no activarse por el simple hecho de que el archivo contuviera la firma de ruido estática del transmisor del dron.
*   **Interferencias Activas:** Para archivos de ruido, se identificaron mediante CFAR las ráfagas reales de WiFi o Bluetooth transitorias, recortándolas y etiquetándolas como negativos activos (`type: 'interference'`).

#### La Aumentación y el Colapso Físico del Z-score Atenuado (Causa del Fracaso)
Durante la fase de aumentación AWGN en entrenamiento (probabilidad del **`60%`** sobre muestras `drone_real`), la red V4 no reseteaba `z_peak = 0.0`. Inyectaba ruido a SNRs discretas de $-20$ a $-4\text{ dB}$ (pasos de 2 dB), aplicando una **fórmula de atenuación analítica lineal de potencia**:
$$z\_peak_{\text{augmented}} = z\_peak_{\text{original}} \times 10^{\frac{\text{SNR}_{\text{target}}}{20.0}}$$
*   **Por qué fracasó en el mundo real:** En entrenamiento, al degradar sintéticamente un dron a $-15\text{ dB}$, el Z-score de entrada resultante de aplicar la fórmula teórica era un valor atenuado pero mayor que cero (ej. $z\_peak \approx 1.8$). 
*   Sin embargo, en inferencia sobre el Golden Set real de test a baja SNR, la estimación real del detector CFAR devolvía un $z\_peak$ de **`0.0` absoluto** (debido al enmascaramiento total de la entropía espectral por el ruido térmico). 
*   Como la red neuronal sobredimensionada V4 se había vuelto adicta a encontrar un valor de $z\_peak$ proporcionalmente atenuado pero positivo para la clase dron, al recibir un `0.0` real lo clasificó sistemáticamente como "silencio" o "ruido térmico puro", provocando un desplome masivo del Recall y un fallo de generalización catastrófico. Además, la excesiva capacidad convolucional de la red (filtro de 512 canales) provocó que la red memorizara patrones de ruido estáticos del lote de entrenamiento.

---

### 2.5. Paso 5 — Model V2.1 Golden (La Redención Espectro-Temporal)

Tras el fracaso del sobredimensionamiento analítico del modelo V4, se regresó a la arquitectura moderada y resiliente de la V2 (heredando su **kernel temporal masivo de 128** que tan bien filtraba el ruido de fondo), refinando por completo el bucle de entrenamiento (AWGN) y el pipeline de inferencia temporal continua sobre el test set para fundar el modelo definitivo del proyecto.

#### Composición del Dataset y Aumentación Inteligente (`dataset_v2_1_clean_pointers.csv`)
El conjunto de datos se re-estructuró a nivel de recortes de ventana temporal fija de 9.4 ms ($131,072$ muestras) centrados rigurosamente en el burst.
*   **AWGN Z-Score Reset (Train):** Durante el bucle de entrenamiento, se seleccionan los bursts de dron a alta SNR ($\ge 10\text{ dB}$). Con una probabilidad del **`40%`**, la muestra del batch es sustituida por una de estas ráfagas puras degradada artificialmente con AWGN complejo en un rango uniforme y continuo de **$-20$ a $-8\text{ dB}$**. Tras inyectar el ruido térmico, **se fuerza el valor físico de `z_peak` a `0.0`**. Esto educa a las capas convolucionales de los streams temporal y espectral a buscar la firma física real del dron, desvinculando la clasificación final de la presencia de un pico de energía analítico visible por el CFAR.
*   **Jittering Temporal (Train):** Se introduce un desplazamiento aleatorio uniforme de $\pm 1\text{ ms}$ en el centro del burst (`t_center`) para robustecer a la red frente a desajustes de fase.

#### Lógica de Inferencia Sliding Window (Test Golden Set)
Para evaluar de forma real los ficheros completos de 75 ms (Golden Set de 3,744 ficheros):
1.  **Estimación Física Global Única:** El detector de entropía (CFAR) se ejecuta **globalmente una sola vez sobre los 75 ms completos** del archivo para garantizar la estabilidad de las estimaciones físicas globales (`global_nf`, `global_H_mean`, `z_peak`).
2.  **Inferencia Sliding Window:** Se desliza una cuadrícula regular de **16 sub-ventanas uniformes de 9.4 ms** (con un solapamiento del 50%) inyectando el mismo tensor físico global en todas ellas en el forward pass de la red.
3.  **Filtro de Persistencia Temporal:** Para evitar falsas alarmas espectrales causadas por rebotes rápidos de WiFi, se exige que al menos **2 sub-ventanas consecutivas** superen el umbral óptimo de probabilidad de dron ($\tau = 0.75$) para disparar la alerta global del archivo.
*   **Rendimiento:** Este modelo consiguió el rendimiento óptimo definitivo del proyecto: **$90.05\%$ de F1-Score** y **$96.29\%$ de Especificidad**.

---

### 2.6. Paso 6 — Model V5 (Multiple Instance Learning Espectro-Temporal)

#### Diseño Arquitectónico (Gated ABMIL)
El modelo V5 aborda la detección de drones a nivel de archivo completo (75 ms), conceptualizando el problema bajo el paradigma de **Aprendizaje por Instancias Múltiples (MIL)**. El archivo de 75 ms representa una "Bolsa" (Bag) que contiene un conjunto dinámico de hasta 32 ráfagas o "Instancias" individuales.
*   **Segmentación Sensible y Caché:** Cada fichero se procesa con un segmentador de entropía muy sensible ($Z\_score = 1.5$, duración mínima de $0.4\text{ ms}$) para asegurar la captura de cualquier ráfaga débil de dron. Las instancias resultantes se almacenan en una caché rápida de disco (`burst_cache/`).
*   **Codificador Dual (`BurstEncoder`):** Cada ráfaga superviviente pasa por un codificador convolucional de dos ramas independientes:
    *   *Rama IQ (Tiempo):* Procesa ventanas de $1024$ muestras mediante `Conv1d(2, 32)` -> `Conv1d(32, 64)` -> `Conv1d(64, 128)` -> `AdaptiveAvgPool1d(16)` -> `Linear` -> Embedding de 64.
    *   *Rama PSD (Espectro):* Procesa espectros Welch de 512 bins mediante una secuencia homóloga de convoluciones y pooling, generando otro embedding de 64.
     Ambos embeddings se concatenan en un vector de representación de instancia de 128 dimensiones.
*   **Agregación por Atención Puerta (`GatedAttentionMIL`):** Para clasificar la bolsa completa, se calculan pesos de atención dinámicos para cada instancia $j$ mediante una estructura bi-rama atencional con compuerta (gated):
    $$a_j = \frac{\exp\left(\mathbf{w}^\top \left(\tanh(\mathbf{V} \mathbf{h}_j^\top) \odot \sigma(\mathbf{U} \mathbf{h}_j^\top)\right)\right)}{\sum_{k=1}^{K} \exp\left(\mathbf{w}^\top \left(\tanh(\mathbf{V} \mathbf{h}_k^\top) \odot \sigma(\mathbf{U} \mathbf{h}_k^\top)\right)\right)}$$
    Donde $\mathbf{V}$ y $\mathbf{U}$ son matrices de proyección lineal de dimensión $64 \times 128$, $\mathbf{w}$ es un vector de proyección de $64 \times 1$, $\odot$ representa el producto de Hadamard (elemento a elemento) y $\sigma$ es la función sigmoide que actúa como puerta de paso de información. El embedding fusionado de la bolsa es la suma ponderada $\mathbf{z} = \sum_{j} a_j \mathbf{h}_j$.

#### El Filtro Guardián LightGBM y el Colapso de Detección a Baja SNR
Para evitar saturar la red de atención MIL con ráfagas ruidosas, se implementó un modelo LightGBM en CPU que actuó como un pre-filtro de ráfagas interferentes.
*   **Mecánica del Filtro:** El LightGBM fue entrenado con un vector de 11 características físicas locales utilizando ráfagas limpias ($\text{SNR} \ge 18\text{ dB}$).
*   **Fallo de Acoplamiento Espectral:** A baja SNR, las variables físicas que alimentan al LightGBM sufren una distorsión no lineal severa debida al ruido térmico de fondo (el ancho de banda instantáneo estimado se ensancha artificialmente, la flatitud espectral tiende a $1.0$ y la curtosis decae drásticamente).
*   **El Efecto de Criba Catastrófica:** Al evaluar una ráfaga real de dron a baja SNR (ej. $-12\text{ dB}$), el clasificador LightGBM, incapaz de correlacionar estas características ruidosas con la firma limpia aprendida en entrenamiento, les asigna una probabilidad de dron `p_drone` inferior a $0.1$. 
*   Como el umbral de supervivencia de bolsa está fijado en `FILTER_CONF_THRESHOLD = 0.6`, **el filtro descarta y borra la ráfaga de dron de la bolsa antes de que esta pueda ser procesada por la red profunda**. Si alguna ráfaga ruidosa logra superar el filtro, la red de atención ABMIL diluye su peso en el promedio de la bolsa al competir con el resto de instancias que contienen ruido de fondo plano.

---

### 2.7. Paso 7 — Model V2.2 (Recálculo Dinámico CFAR - El Paradigma Teórico Puro)

A raíz de una revisión teórica del pipeline de aumentación (AWGN), se plantea esta variante como evolución conceptual directa de la V2.1 para explorar el límite de precisión analítica.

#### Fundamento Teórico y Eliminación de Heurísticas
En la V2.1, para evitar que la red sufriera de *Data Leakage* al inyectarle ruido a una muestra limpia cuyo $Z\_score$ precalculado seguía siendo alto, se aplicó la heurística de forzar `z_peak = 0.0`. 
La variante V2.2 adopta un enfoque de **purismo analítico**: al inyectar ruido térmico (AWGN) en el tensor IQ de 9.4 ms, no se fuerza el valor físico a cero, sino que **se ejecuta de nuevo el detector de entropía espectral completo (CFAR) en tiempo real sobre la nueva señal degradada**.
*   **Lo que aprende la red:** En lugar de asumir ceguera total desde un principio, la red recibe el impacto matemático real que el ruido ha causado en el umbral de fondo (`global_nf`) y en el valle de entropía (`z_peak`). La red observa y aprende la degradación continua y natural del algoritmo CFAR a medida que baja la SNR. Al ser la ventana de sólo 9.4 ms, se evalúa cómo el CFAR estima el ruido sin contexto amplio, representando la dificultad algorítmica real.

#### Desafío de Ingeniería
Este paradigma exige ejecutar la STFT, la Welch Log-PSD, el umbral adaptativo y la morfología matemática dinámica dentro del método `__getitem__` del DataLoader de PyTorch para el 40% de las muestras en cada época. Para contrarrestar este inmenso cuello de botella de procesamiento (puramente en CPU), se requiere un entrenamiento prolongado en equipos dedicados, permitiendo equilibrar el peso computacional a cambio de la máxima pureza teórica del modelo.

---

## 3. Conclusiones y Recomendaciones de Diseño Espectro-Temporal

La comparativa histórica de este proyecto aporta tres conclusiones fundamentales para el marco de tu defensa de TFM:

1.  **Superioridad del Procesamiento Desacoplado (Sliding Window):** Intentar segmentar ráfagas de forma analítica en entornos ruidosos (como en V1, V4 y V5) introduce un "efecto dominó" donde el fallo estadístico del detector analítico (CFAR) en baja SNR ciega al clasificador de Deep Learning posterior. Burlar esta limitación requiere un diseño desacoplado como el de la V2.1 Golden: **detección temporal ciega por ventanas deslizantes y una inyección física global y estable**.
2.  **La Importancia del "AWGN Z-Score Reset":** Los modelos basados en características mixtas (físicas y espectro-temporales) sufren sesgos de calibración muy severos. Entrenar simulando atenuaciones teóricas de potencia en variables físicas (como el escalado lineal del Z-score en V4) genera un desajuste posicional insalvable frente a la realidad del receptor de RF. Forzar el Z-score a `0.0` en train educa a la red a ser verdaderamente autónoma ante el ruido térmico.
3.  **Regularización Espectral frente a Capacidad:** Aumentar la profundidad y el número de canales de la red (V4) no compensa la degradación física de la señal RF. La red V2.1, con una capacidad convolucional moderada (filtro inicial de 32 canales y kernel masivo de 128), demostró una capacidad de generalización muy superior frente al sobreajuste al ruido electromagnético de la variante V4.
