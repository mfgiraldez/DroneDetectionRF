# Session Handoff - TFM Drone Detection RF

**Fecha/Hora:** 2026-07-02
**Estado Global del Proyecto:** Capítulos 1 al 5 completamente revisados y formateados. Queda pendiente el Capítulo 6 y la inserción de imágenes.

## 1. Contexto de Trabajo (LaTeX)
- **Directorio Principal:** `c:\repos\DroneDetectionRF\TFM_documentos\contenidos`
- **Archivo Raíz (Compilación):** `c:\repos\DroneDetectionRF\TFM_documentos\documento (3).tex` (Usar este nombre al compilar, ignorar otros).
- **Archivos de Capítulos Completados:** 
  - `00_frontmatter.tex` (Nota: El usuario prefiere usar la versión renombrada, pero actualmente se han mantenido los backups)
  - `10_Introduccion.tex`
  - `20_EstadoDelArte_2.tex`
  - `30_MarcoTeorico.tex`
  - `40_ArquitecturaImplementacion.tex`
  - `50_ResultadosDiscusion.tex`

## 2. Reglas Estrictas de Estilo (¡CRÍTICO para el Agente Entrante!)
Durante esta sesión se han establecido y validado las siguientes reglas de oro por orden expresa del usuario:
1. **Nada de cursiva ni negrita en textos normales ni itemizes:** `\textbf{}` y `\textit{}` SOLO se permiten para:
   - Términos extranjeros (ej. \textit{dataset}, \textit{offline}).
   - Títulos de figuras (`\caption`).
   - Cabeceras de tablas.
   - NUNCA usarlas para resaltar palabras en párrafos ni en el inicio de un `\item`.
2. **Cero Lenguaje de IA:** Están terminantemente prohibidas expresiones como "cabe destacar", "en conclusión", "es crucial", "categórico". 
3. **Uso de "físico" y "analítico":** Solo usar estas palabras en contextos estricta y puramente científico/matemáticos. No usarlas de forma metafórica (ej. MAL: "oráculo físico", "severidad analítica". BIEN: "rigor matemático", "umbral de ruido").
4. **Coherencia SNR:** El umbral de SNR crítica para el dataset es **$-12$ dB** (Grupo D). NO usar $-14$ dB ni mezclar cifras.
5. **Espectrograma:** Nunca decir que el espectrograma "fracasa". Usar descripciones técnicas precisas (e.g. "la magnitud queda fuertemente enmascarada por el ruido estocástico").

## 3. Trabajo Inmediato Pendiente
1. **Diseñar y redactar el Capítulo 6 (Conclusiones y Trabajo Futuro):**
   - Resumir los hallazgos técnicos (DualStream V2.1 superó en métricas gracias a su persistencia temporal vs Sliding Window simple).
   - Detallar el trabajo futuro (Implementación del Dynamic CFAR para entrenar redes ajustadas al nivel de ruido local exacto de la ráfaga de 9.4 ms, mejorando la especificidad).
2. **Gestión de Figuras:** Revisar los `TODO` de imágenes (si los hay) e insertar/generar/arreglar la maquetación de gráficos en los capítulos.

> **NOTA AL AGENTE ENTRANTE:** Revisa a fondo el archivo `c:\repos\DroneDetectionRF\NoisyUAV\memory.md` para conocer la historia detallada de todas las arquitecturas de redes neuronales (Dual-Stream V2.1 y Dual-Stream V2.2) y su comportamiento a distintas SNRs. El usuario demanda un tono académico aséptico y directo, sin rodeos, siguiendo la skill `thesis-writing`.
