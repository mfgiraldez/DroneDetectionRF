# -*- coding: utf-8 -*-
import os

mem_path = r"c:\repos\DroneDetectionRF\NoisyUAV\memory.md"
content = """
### 9. Revisión Integral de los Tutores (Corrección del TFM)
Durante esta sesión intensiva se ha purgado y refinado el documento (Capítulos 3, 4 y 5) basándonos en los comentarios de los tutores, aplicando un tono académico estricto y eliminando rastros de "AI-isms". Los cambios críticos son:

- **Estandarización de Nomenclatura Matemática (Cap. 3 y 4):** Se corrigió la colisión estructural de variables. $N$ queda bloqueado para el número de muestras temporales ($\sim 10^6$). $N_w$ representa la ventana de Welch/STFT y el número de *bins*. $C_{\text{in}}$ y $C_{\text{out}}$ representan los mapas de las CNN.
- **Detector CFAR y Masking Effect (Cap. 3):** Se analizó la literatura clásica (Anastassopoulos, 1992) para justificar la ineficacia del CFAR ante señales sumergidas en el ruido debido al *masking effect*. Se demostró analíticamente que ignorar el Wi-Fi es una característica altamente deseable del método de entropía para evitar saturar el clasificador en la banda ISM.
- **Rigor Físico y Estadístico (Cap. 4):** 
    - Se definió el ruido del Data Augmentation rigurosamente como un **proceso gaussiano complejo propio (CSCG)**.
    - Se arregló la integración narrativa de la ecuación del estadístico $z_{\text{peak}}$, justificando el signo negativo de su numerador (caída de entropía) y enlazándolo correctamente con el filtro probabilístico posterior.
    - Se clarificó la ecuación del optimizador **AdamW**, definiendo correctamente las estimaciones corregidas por sesgo y el desacoplamiento del *weight decay*.
    - Se reescribió la justificación de los tamaños del *kernel* (campo receptivo temporal) eliminando vaguedades.
- **Adaptación de Métricas (Cap. 5):** Se validó el uso del término "Exhaustividad" como traducción técnica de *Recall* en la introducción del capítulo, salvando así la coherencia de todas las gráficas generadas (Precision-Recall) sin necesidad de recompilar figuras.
- **Nueva Regla Operativa (Agent Rule):** Se ha creado el archivo `.agents/AGENTS.md` con la regla mandatoria de **LEER EL CONTEXTO ANTES DE REDACTAR** para asegurar modificaciones orgánicas y no redundantes en LaTeX.
"""

with open(mem_path, "a", encoding="utf-8") as f:
    f.write(content)

print("memory.md actualizado exitosamente con la sesión de corrección del TFM.")
