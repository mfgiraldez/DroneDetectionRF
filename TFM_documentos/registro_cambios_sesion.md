# Registro de Cambios - Revisión del Estado del Arte (Capítulo 2)

A continuación se detalla exhaustivamente el registro de todas las modificaciones aplicadas al archivo `20_EstadoDelArte_2.tex` durante la sesión de trabajo, orientadas a elevar el rigor científico, mejorar la estructura argumental y cumplir con las anotaciones del tutor.

## 1. Refinamiento de las características del protocolo FHSS
Se reescribieron por completo los tres ítems descriptivos de las señales FHSS para evitar adelantar datos metodológicos y centrarse en el marco teórico:
- **Duración del salto**: Se matizó que la extracción de la duración falla tanto en alta SNR (por la enorme similitud temporal con los paquetes Bluetooth) como en baja SNR (debido a la corrupción de los transitorios por ruido térmico en la señal IQ). Se justificó así por qué el sistema debe apoyarse en la Densidad Espectral de Potencia (PSD).
- **Número de saltos y tasa de repetición**: Se eliminaron las referencias prematuras a los parámetros específicos del dataset del proyecto (75 ms, 14 MHz), generalizando la explicación desde el punto de vista del estándar de telecomunicaciones.
- **Ancho de banda instantáneo**: Se corrigió el uso del término coloquial "bins", sustituyéndolo por terminología matemáticamente rigurosa ("subportadoras ortogonales" en el caso de la comparativa con Wi-Fi/OFDM), aclarando así las dudas del tutor.
- Se aplicó una revisión gramatical global sobre estos tres puntos para eliminar erratas (como "0,5,ms" y "dificultades a en función").

## 2. Introducción a la sección de Aprendizaje Máquina (Machine Learning)
Se llevó a cabo una restructuración del inicio de la sección 2.2 para dotarla de mayor peso académico:
- **Actualización de Nomenclatura**: Se sustituyó el término "aprendizaje automático" por "aprendizaje máquina" en el título de la sección y en la redacción, tal y como se acordó.
- **Clarificación Terminológica (UAV vs. Dron)**: Para cumplir la directriz estricta del tutor, se eliminó el acrónimo `\glspl{uav}` en la oración de apertura, reemplazándolo directamente por "drones". 
- **Inclusión de Bibliografía Core**: Se añadieron citas directas del archivo `.bib` para justificar el uso de algoritmos clásicos (Medaiyese, Ezuma) frente a los enfoques modernos de Deep Learning (Allahham, Zheng).
- **Justificación del Deep Learning**: Se reescribió la explicación sobre las debilidades del ML clásico. Se argumentó científicamente que, ante la falta de información *a priori*, el diseño manual de descriptores matemáticos aboca a usar estadísticos genéricos (medias, varianzas) que son insuficientes para capturar la complejidad no lineal del problema, justificando así el uso de redes de extremo a extremo.

## 3. Revisión de la subsección CNN-2D (Espectrogramas)
Se realizó una limpieza integral del texto para purgarlo de florituras literarias ("barrera infranqueable", "corpus inmenso", "destrucción deliberada") y convertirlo en un texto puramente analítico y objetivo:
- **Formalización de resultados**: Se reescribieron los aportes de Glüge *et al.* y Alam *et al.*, corrigiendo además la sintaxis LaTeX de las citas (`\textit{et al.}`).
- **Inclusión de Visión Artificial (YOLO/ResNet)**: A petición del usuario, se rastreó la bibliografía para añadir un párrafo completamente nuevo que conecta el uso de espectrogramas con los modelos de detección de objetos. Se incluyeron citas a trabajos del proyecto (Zhao usando YOLO, Olesiński con CNNs de Región de Interés, y Cai con redes ligeras) para demostrar que se aplica *Computer Vision* de forma directa sobre la radiofrecuencia.
- **Inclusión del hándicap computacional**: Se amplió el último párrafo de la sección para incluir que, además de la pérdida de fase, las CNN-2D sufren de una inmensa carga computacional (decenas de millones de parámetros entrenables) que dificulta severamente su despliegue en hardware embebido o plataformas tácticas C-UAS.

## 4. Revisión de la introducción a CNN-1D (Señal IQ nativa)
Se evitó la redundancia conceptual con el bloque anterior:
- **Doble Ventaja Estratégica**: Se reescribió el primer párrafo de las redes 1D para no repetir el concepto de la "pérdida de fase". En su lugar, se argumentó como una doble ventaja:
  - *Ventaja Analítica*: Preservación intrínseca de la fase para discriminar colisiones de banda estrecha frente al ruido.
  - *Ventaja de Ingeniería*: Eliminación de las costosas etapas de preprocesamiento matemático (cálculo constante de FFT/STFT), reduciendo radicalmente los tiempos de inferencia en sistemas de recursos limitados.

---
*Fin del registro de sesión.*
