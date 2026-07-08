# Análisis de Anotaciones del Tutor (Borrador 2)

Este documento recopila y analiza exhaustivamente las anotaciones (texto manuscrito, subrayados, círculos, marcas) realizadas por el tutor en el documento `tfm_manfergir_borrador2jjmf.pdf`. Están ordenadas cronológicamente según aparecen en el documento.

---

## 1. Introducción
**Pág. Doc: 3 (Pág. PDF: 17) | Sección: 1.3 Limitaciones del Estado del Arte...**
*   **Anotación:** "puede aportar una ventaja competitiva en la clasificación. *¿Quién dice esto? referencia?*"
*   **Interpretación:** El tutor pide añadir una cita bibliográfica que respalde la afirmación de que conservar la fase geométrica determinista aporta ventajas competitivas en clasificación a baja SNR.

**Pág. Doc: 4 (Pág. PDF: 18) | Sección: 1.4 Hipótesis y Objetivos**
*   **Anotación:** Marcas de inserción/separación en el Objetivo 1, al final de "lógicas FHSS.".
*   **Interpretación:** Posiblemente sugiere revisar la puntuación o separar ese objetivo en dos ideas. 

**Pág. Doc: 5 (Pág. PDF: 19) | Figura 1.1**
*   **Anotación:** Garabato en rojo (una onda y un círculo).
*   **Interpretación:** Parece una marca incidental sin un mensaje textual claro, quizás probando el lápiz táctil o marcando el diagrama de flujo.

**Pág. Doc: 6 (Pág. PDF: 20) | Sección: 1.5 Alcance y Limitaciones**
*   **Anotación:** En el margen izquierdo abarcando los primeros párrafos pone **"REESCRIBIR"**. Sobre la palabra "documento" (en el primer párrafo) propone cambiarla por **"proyecto"**. En el tercer párrafo, rodea "arreglos" y escribe **"matrices?"** (sugiriendo cambiar "arreglos lógicos programables" por "matrices lógicas programables" para traducir *arrays*).
*   **Interpretación:** El tutor considera que la redacción de los límites del experimento debe ser más formal o clara (sustituir documento por proyecto, arreglos por matrices FPGA).

**Pág. Doc: 6 (Pág. PDF: 20) | Sección: 1.7 Estructura del Documento**
*   **Anotación:** Al final del punto sobre el Capítulo 2 escribe: *"¿Y? Supongo que se entenderá más adelante"*.
*   **Interpretación:** La descripción de lo que hace el Capítulo 2 se le queda algo coja o inconclusa en este punto del texto.

---

## 2. Estado del Arte
**Pág. Doc: 9 (Pág. PDF: 23) | Tabla 2.1**
*   **Anotación:** Rodea el rango "> 5 km" en Radiofrecuencia y escribe: *"Esperaría un < x y no > x. pero ok!"*.
*   **Interpretación:** El tutor señala que habitualmente los rangos de alcance se definen con un límite superior ("menor que X km"), pero acepta tu forma de expresarlo ("más de 5 km").

**Pág. Doc: 10 (Pág. PDF: 24) | Sección: 2.1.2**
*   **Anotación:** En la cabecera de la página: *"Ser uniforme en la escritura. He visto (Radar Cross Section (RCS)), (Radar Cross Sector, RCS), (radar Cross Section, RCS)..."*.
*   **Interpretación:** Falta de homogeneidad al referirte a las siglas RCS en secciones anteriores. Revisa mayúsculas/minúsculas y la palabra "Section" vs "Sector".
*   **Anotación:** Sobre los modelos de Machine Learning escribe: *"no se de donde viene aprendizaje automático -> NOSOTROS 'aprendizaje máquina'"*.
*   **Interpretación:** Sugerencia terminológica estricta: prefiere que traduzcas ML como "aprendizaje máquina" en lugar de "aprendizaje automático".

**Pág. Doc: 11 (Pág. PDF: 25) | Sección 2.1.2 (continuación)**
*   **Anotación:** En la definición de PSD y STFT: *"si defines todo de nuevo, ..."*.
*   **Interpretación:** Parece indicar que estás sobre-explicando o repitiendo acrónimos que ya se han definido o que son demasiado básicos.
*   **Anotación:** En la definición de FHSS, rodea EDR (~366 us) y escribe: *"-> 0.36 ms ~= 0.4 ??"*. Y sobre "de bins" anota *"bins ?? SC?"*.
*   **Interpretación:** Pide homogeneizar las unidades de tiempo (usar ms en lugar de microsegundos para comparar más fácil) y aclarar el término "bins" (quizás usar subportadoras o *subcarriers* SC).

**Pág. Doc: 12 (Pág. PDF: 26) | Sección 2.1.4**
*   **Anotación:** En DroneSentry subraya "es" y en AUDS rodea "8 km ,".
*   **Interpretación:** Pequeños fallos de redacción o formato (coma separada del texto).

**Pág. Doc: 13 (Pág. PDF: 27) | Sección 2.2.1**
*   **Anotación:** Al pie de página: *"No se si al ppio o en algún sitio podrían hablar de RPAS, Drone, UAV y por qué se usa aquí UAV."*
*   **Interpretación:** Sugiere añadir un pequeño párrafo aclaratorio en la Introducción sobre la terminología de aeronaves no tripuladas y por qué te decantas por las siglas UAV.

**Pág. Doc: 16 (Pág. PDF: 30) | Sección 2.4.1**
*   **Anotación:** Rodea "ruidoso" en el título y añade *"? Drone, UAV, ..."*.
*   **Interpretación:** Pide que especifiques mejor a qué "señales RF" te refieres (de drones).
*   **Anotación:** Subraya GMM. Rodea la palabra "alta" y anota *"baja?"*. Y comenta: *"Entiendo que se aprende a ver que muestran son claramente drone"*.
*   **Interpretación:** Tienes una errata lógica: pusiste "incertidumbre alta" cuando quizás te referías a "incertidumbre baja" (es decir, muestras de las que el modelo está muy seguro para usarlas en aprendizaje semi-supervisado).
*   **Anotación:** En 2.4.2 anota *"Define pseudo etiqueta"* y en 2.4.3 *"¿se etiqueta la secuencia completa?"* al lado de noisy-OR. Antes de 2.4.3 añade: *"A ver : se usa FHSS para mejorar detección? El FHSS no depende del modelo?"*.

**Pág. Doc: 17 y 18 (Págs. PDF: 31 y 32) | Sección 2.5 y 2.6**
*   **Anotación:** Multitud de "ticks" de validación ("check" en rojo) en las listas de requisitos. Un acierto esta parte.
*   **Anotación:** En 2.6, rodea "Capítulo ??" y pone flechas.
*   **Interpretación:** Te dejaste una referencia cruzada rota (`\ref` roto) en el texto final del capítulo 2.

---

## 3. Marco Teórico y Metodología
**Pág. Doc: 19 (Pág. PDF: 33) | Sección 3.1**
*   **Anotación:** Subraya la explicación de señal, interferencia y ruido. Sobre $x(t)$ pregunta: *"¿Qué modulación usan?"*.
*   **Interpretación:** Quiere que concretes las modulaciones subyacentes típicas (ej: GFSK, LoRa, etc.) que van montadas sobre el FHSS.

**Pág. Doc: 20 y 21 (Págs. PDF: 34 y 35) | Sección 3.2 Dataset**
*   **Anotación:** Pide que se enumeren más claramente *"se resumen principales características"*.
*   **Anotación:** *"tras este procesamiento es, por tanto, Fs = 14 MHz -> ¿Xq?"*.
*   **Interpretación:** Pide justificar matemáticamente por qué el decimation pasa exactamente a 14 MHz (¿por el ancho de banda del canal Wi-Fi o del ISM?).
*   **Anotación:** En la clase Target 0, anota *"Me he liado un poco. ¿ES 4?"*. En el Grupo C pone *"¿Consistencia (aprendizaje automático?)"* y un gran *"? Creía que nosotros usábamos DL"*.
*   **Interpretación:** Confusión terminológica. Si tu título habla de Deep Learning y Redes Neuronales, usar de repente "Machine Learning" genérico le confunde. Debes ser coherente.
*   **Anotación:** En 3.2.1 subraya *"el propósito fundamental... no radica en la identificación... sino en la detección robusta"* y anota **"Dejar claro objetivo Intro"**.
*   **Interpretación:** Este es un punto fundamental de tu TFM. Sugiere que lleves esta frase potente a la Introducción/Objetivos.

**Pág. Doc: 22 y 23 (Págs. PDF: 36 y 37) | Figuras y Tablas**
*   **Anotación:** En la Figura 3.1 pregunta por la leyenda: *"? 's colores? DJI, Futaba, ..."*.
*   **Anotación:** En la Tabla 3.1 sobre el Ruido: *"No se toma el ruido en esta banda?"*.

**Pág. Doc: 27 (Pág. PDF: 41) | Sección 3.4 Dispersión Temporal**
*   **Anotación:** Sobre "convoluciones unidimensionales" anota: *"No entiendo esto. Supongo que lo harán en algún trabajo, citar ?"*.
*   **Interpretación:** Pide cita para afirmar que las CNN 1D reducen la complejidad O(N*K).
*   **Anotación:** Sobre "agrava la ceguera del modelo", anota *"por qué, explicar mejor por qué se ha elegido este valor"*.
*   **Interpretación:** Requiere mayor profundidad física/matemática para explicar por qué una ventana de 75 ms "ciega" a la red (es por el ratio de ocupación temporal del salto vs el ruido).

**Pág. Doc: 28 (Pág. PDF: 42) | Degradación SNR y ERM**
*   **Anotación:** Arriba indica *"Esto es potencia, no PSD. ver (3.3)"*. 
*   **Anotación:** Pregunta por los rangos *"? en R, Z?"* y rodea notación matemática ($\tilde{y} \in Y$, $P_x$, $\mathbf{x} \in \mathbb{C}^N$).
*   **Anotación:** En Ecuación 3.8: *"¿qué es M?"* (No has definido $M$, que supongo es el tamaño del *batch* o dataset).
*   **Anotación:** Tacha "Orthogonal frequency-division multiplexing" y marca un 0 en *shortcut learning*.
*   **Interpretación:** La carga matemática de esta página le parece algo densa, mal referenciada o con variables sin definir ($M$).

**Pág. Doc: 29 a 32 (Págs. PDF: 43 a 46) | Sección 3.5 Entropía y CFAR**
*   **Anotación:** Arriba: *"No se puede estar definiendo todo recursivamente..."*. En 3.5.1 sobre la $P_{k,m}$ pregunta *"Gaussian Kernel? distribución de pk?"*. 
*   **Anotación:** Sobre el umbral (pág 30 del doc): *"El solapamiento es importante, y también asegurar que el barrido/ancho ventana/solapamiento permite detectar una ráfaga de drone en el peor escenario"*.
*   **Anotación:** En Pág 31 del doc: *"? No sería válido para drones con CDMA?"* (Buena pregunta táctica).
*   **Anotación:** Pág 32 del doc: *"Creo que no se ha hablado de umbral CFAR"*. En la leyenda de la Fig 3.4: *"? P_s se ha definido?"*.
*   **Interpretación:** Falta contexto previo para presentar el CFAR. Lo usas antes de haberlo definido rigurosamente y las variables de tus gráficas no concuerdan con el texto.

**Pág. Doc: 33 (Pág. PDF: 47) | Sección 3.6 CV-CNN**
*   **Anotación:** Pide una **"[cita]"** antes de 3.6.1.
*   **Anotación:** En la Ecuación de Wirtinger 3.14 anota: *"No se bien para qué se pone", "dC/dW*?", "i?"*.
*   **Interpretación:** No le queda claro por qué introduces la derivación manual de Wirtinger si luego vas a usar un framework que lo hace automático, o no entiende la nomenclatura usada para la derivada compleja.
