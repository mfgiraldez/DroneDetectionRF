import os

filepath = r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\10_Introduccion.tex"

with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()

replacements = {
    # Objectives
    r'\item \textit{Diseño de un bloque discriminador basado en entropía de Shannon:}': r'\item Diseño de un bloque discriminador basado en entropía de Shannon:',
    r'\item \textit{Implementación de convolución compleja 1D (CV-CNN):}': r'\item Implementación de convolución compleja 1D (CV-CNN):',
    r'\item \textit{Arquitectura DualStream con fusión temporal y espectral:}': r'\item Arquitectura DualStream con fusión temporal y espectral:',
    r'\item \textit{Caracterización del umbral de degradación frente a ruido:}': r'\item Caracterización del umbral de degradación frente a ruido:',
    
    # Contributions
    r'\item \textit{Validación de la detección de transmisiones basada en entropía de Shannon:}': r'\item Validación de la detección de transmisiones basada en entropía de Shannon:',
    r'\item \textit{Topología DualStream-SlidingWindow:}': r'\item Topología DualStream-SlidingWindow:',
    r'\item \textit{Caracterización del umbral de degradación térmica:}': r'\item Caracterización del umbral de degradación térmica:',
    
    # Structure
    r'\item \textit{Capítulo~\ref{cap:EstadoDelArte} (Estado del Arte):}': r'\item Capítulo~\ref{cap:EstadoDelArte} (Estado del Arte):',
    r'\item \textit{Capítulo~\ref{cap:MarcoTeorico} (Marco Teórico y Fundamentos):}': r'\item Capítulo~\ref{cap:MarcoTeorico} (Marco Teórico y Fundamentos):',
    r'\item \textit{Capítulo~\ref{cap:arquitectura} (Arquitectura e Implementación):}': r'\item Capítulo~\ref{cap:arquitectura} (Arquitectura e Implementación):',
    r'\item \textit{Capítulo~\ref{cap:resultados_discusion} (Resultados y Discusión):}': r'\item Capítulo~\ref{cap:resultados_discusion} (Resultados y Discusión):',
    r'\item \textit{Capítulo~\ref{cap:Conclusiones} (Conclusiones y Trabajo Futuro):}': r'\item Capítulo~\ref{cap:Conclusiones} (Conclusiones y Trabajo Futuro):'
}

for old, new in replacements.items():
    text = text.replace(old, new)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(text)

print("Removed italics from itemize lists.")
