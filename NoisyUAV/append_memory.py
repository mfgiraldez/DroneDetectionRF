# -*- coding: utf-8 -*-
import os

mem_path = r"c:\repos\DroneDetectionRF\NoisyUAV\memory.md"
content = """
### 8. Descarte de MIL y Promoción a Arquitectura Definitiva (DualStream-SlidingWindow V2.2)
Durante la sesión actual se tomaron decisiones arquitectónicas críticas que alteran la estructura final de la memoria del TFM:

- **Cancelación Definitiva del Modelo MIL**: Tras lanzar la evaluación de `MultipleInstanceLearning-GatedAttention`, el coste computacional inasumible (aprox. 500s por época evaluando bolsas superpuestas masivas) y el desvío del enfoque de tiempo real obligaron a eliminar este modelo por completo del proyecto. Las referencias y figuras de MIL quedan descartadas.
- **`DualStream-SlidingWindow` (V2.2) como Modelo de Producción Final**: Se decidió promover el modelo V2.2 a "Arquitectura Definitiva" con su propia `\\section` en `40_Arquitectura.tex`.
- **Refinamiento de la Skill `human-academic-writer`**: Se aplicó una actualización estricta a la *skill* de redacción académica, eliminando por completo las estructuras de relleno clásicas de IA ("Es importante destacar...", transiciones redundantes). El foco debe ser la densidad técnica absoluta combinada con una altísima claridad didáctica, explicando el "porqué" físico antes de la formulación puramente matemática.
- **Rigidez Estructural entre Capítulos 4 y 5**: Se reafirma categóricamente la prohibición de incluir métricas empíricas (como *recall*, exactitud o matrices) en el Capítulo 4. El Capítulo 4 está estrictamente blindado para la topología, el diseño estructural y las motivaciones físicas. Todas las métricas gráficas (que ya están exitosamente generadas en las carpetas `DualStream-Deep` y `modelo_v2_2_dual/figuras_golden`) quedan postergadas en bloque para el inminente Capítulo 5.
- **Redacción de Arquitectura Definitiva Completada**: Se han entregado al usuario las subsecciones de *Arquitectura Interna y Filtrado Frontal* (explicando el núcleo pasabajos masivo de 128 muestras y la Transformada de Fourier interna) y la *Estrategia de Optimización y Entrenamiento* (BCE directa sobre conjunto purgado de *label noise*, AdamW, truncado de gradientes a 1.0 y escalado dinámico de LR).
"""

with open(mem_path, "a", encoding="utf-8") as f:
    f.write(content)

print("memory.md actualizado correctamente.")
