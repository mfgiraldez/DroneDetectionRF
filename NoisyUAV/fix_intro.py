import re

filepath = r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\10_Introduccion.tex"

with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Expand the Scope section into fluid paragraphs (no list)
# We will use simple substring extraction to replace it
start_marker = r"% -----------------------------------------------------------------------------" + "\n" + r"\section{Alcance y Limitaciones del Entorno Experimental}"
end_marker = r"% -----------------------------------------------------------------------------" + "\n" + r"\section{Contribuciones Principales}"

if start_marker in text and end_marker in text:
    before = text.split(start_marker)[0]
    after = text.split(end_marker)[1]
    
    scope_replacement = start_marker + """
\label{sec:alcance}
% -----------------------------------------------------------------------------

El rigor metodológico inherente a la investigación en telecomunicaciones exige acotar con precisión las fronteras operativas del entorno de experimentación. Esto garantiza la reproducibilidad íntegra del estudio al establecer las condiciones de contorno bajo las cuales el modelo resulta matemáticamente válido.

En primer lugar, desde la perspectiva espectral, la inferencia algorítmica se circunscribe exclusivamente a las firmas capturadas en la banda ISM de 2,4~GHz. Si bien un porcentaje significativo de la telemetría y el vídeo \\textit{FPV} de plataformas recientes opera en 5,8~GHz, dicho espectro queda excluido del perímetro de este documento. Esta restricción tiene el objetivo deliberado de forzar al modelo a resolver la coexistencia con el protocolo IEEE~802.11b/g/n, el cual representa la fuente de interferencia \\textit{OFDM} masiva y predominante en las capas bajas de los entornos urbanos densos.

En segundo lugar, el flujo de datos se asienta íntegramente sobre la explotación del recurso abierto NoisyUAV. El modelo de desvanecimiento asumido para emular la lejanía táctica del dron respecto al nodo sensor es puramente térmico (\\textit{AWGN}). A través de la inserción controlada de varianza estocástica, se somete a las señales empíricas a barridos de degradación que abarcan desde una región cuasi-ideal ($+30$~dB) hasta la práctica extinción analítica de la ráfaga subyacente ($-20$~dB). No obstante, es imperativo señalar que la investigación no aborda perfiles probabilísticos de ensanchamiento por retardo, como los canales con desvanecimientos de Rayleigh o Nakagami, ni incorpora técnicas de interferencia direccional maliciosa por parte de actores hostiles (\\textit{jamming}).

Finalmente, toda la orquestación del código —desde el acondicionamiento primario de los tensores hasta la validación de los grafos neuronales— se diseña para una ejecución asíncrona (\\textit{offline}) centralizada, apoyándose en aceleración de coma flotante sobre arquitecturas GPU comerciales (NVIDIA \\textit{CUDA}). Por consiguiente, la síntesis de los pesos a lenguajes descriptores de bloques orientada a su incrustación sobre arreglos lógicos programables (FPGA) o cualquier plataforma embebida de tiempo real, se declara formalmente fuera del propósito fundacional de esta investigación.

"""
    text = before + scope_replacement + end_marker + after

# 2. Refactor Contributions to remove bold and make it look elegant.
text = text.replace(r'\item \textbf{Validación de la detección de transmisiones basada en entropía de Shannon:}', r'\item \textit{Validación de la detección de transmisiones basada en entropía de Shannon:}')
text = text.replace(r'\item \textbf{Topología DualStream-SlidingWindow:}', r'\item \textit{Topología DualStream-SlidingWindow:}')
text = text.replace(r'\item \textbf{Caracterización del umbral de degradación térmica:}', r'\item \textit{Caracterización del umbral de degradación térmica:}')

# 3. Refactor Structure to remove bold
text = text.replace(r'\item \textbf{Capítulo~\ref{cap:EstadoDelArte} (Estado del Arte):}', r'\item \textit{Capítulo~\ref{cap:EstadoDelArte} (Estado del Arte):}')
text = text.replace(r'\item \textbf{Capítulo~\ref{cap:MarcoTeorico} (Marco Teórico y Fundamentos):}', r'\item \textit{Capítulo~\ref{cap:MarcoTeorico} (Marco Teórico y Fundamentos):}')
text = text.replace(r'\item \textbf{Capítulo~\ref{cap:arquitectura} (Arquitectura e Implementación):}', r'\item \textit{Capítulo~\ref{cap:arquitectura} (Arquitectura e Implementación):}')
text = text.replace(r'\item \textbf{Capítulo~\ref{cap:resultados_discusion} (Resultados y Discusión):}', r'\item \textit{Capítulo~\ref{cap:resultados_discusion} (Resultados y Discusión):}')
text = text.replace(r'\item \textbf{Capítulo~\ref{cap:Conclusiones} (Conclusiones y Trabajo Futuro):}', r'\item \textit{Capítulo~\ref{cap:Conclusiones} (Conclusiones y Trabajo Futuro):}')

# 4. Refactor Objectives list to remove bold
text = text.replace(r'\item \textbf{Diseño de un bloque discriminador basado en entropía de Shannon:}', r'\item \textit{Diseño de un bloque discriminador basado en entropía de Shannon:}')
text = text.replace(r'\item \textbf{Implementación de convolución compleja 1D (CV-CNN):}', r'\item \textit{Implementación de convolución compleja 1D (CV-CNN):}')
text = text.replace(r'\item \textbf{Arquitectura DualStream con fusión temporal y espectral:}', r'\item \textit{Arquitectura DualStream con fusión temporal y espectral:}')
text = text.replace(r'\item \textbf{Caracterización del umbral de degradación frente a ruido:}', r'\item \textit{Caracterización del umbral de degradación frente a ruido:}')

# 5. Make sure English terms are italicized
def italicize(word, text):
    pattern = re.compile(rf'(?<!\{{)\b({word})\b(?!\}})', re.IGNORECASE)
    return pattern.sub(r'\\textit{\1}', text)

for word in ["firmware", "hardware", "offline", "Recall"]:
    text = italicize(word, text)

# Also fix the Objectives intro to not use bold:
text = text.replace(r'persigue un \textbf{objetivo general} que consiste', r'persigue un \textit{objetivo general} que consiste')
text = text.replace(r'los siguientes \textbf{objetivos específicos}:', r'los siguientes \textit{objetivos específicos}:')

with open(filepath, "w", encoding="utf-8") as f:
    f.write(text)

print("Formatting applied.")
