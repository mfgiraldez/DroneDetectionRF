# -*- coding: utf-8 -*-
import os

mem_path = r"c:\repos\DroneDetectionRF\NoisyUAV\memory.md"
content = """
### 10. Expansión y Reescritura del Capítulo 1 (Introducción)
Se realizó una expansión masiva y reescritura aséptica del Capítulo 1 tras considerar que la versión anterior era demasiado breve (4 páginas) y adolecía de lenguaje rimbombante ("AI-isms"):

- **Contexto técnico ampliado:** Se profundizó en las Radios Definidas por Software (SDR) y en las vulnerabilidades de los sistemas colaborativos (Remote ID).
- **Problema de la Fase (Justificación de CV-CNN):** Se incorporó una justificación teórica clave sobre cómo los espectrogramas 2D del estado del arte descartan la fase de la señal I/Q, destruyendo la frontera matemática contra el ruido aleatorio en baja SNR.
- **Sección de Alcance:** Se acotó formalmente el proyecto (banda ISM 2.4 GHz, interferencias OFDM/FHSS, simulación offline por GPU, inyección AWGN hasta -20dB) para asentar expectativas realistas y alinear el capítulo 1 con el 3 y 4.
- **Contribuciones formales:** Se desglosaron las tres aportaciones principales: el detector de entropía contra OFDM, la topología DualStream-SlidingWindow, y el sostenimiento de la Exhaustividad a -14dB.
"""

with open(mem_path, "a", encoding="utf-8") as f:
    f.write(content)

print("memory.md actualizado exitosamente con los cambios del Capítulo 1.")
