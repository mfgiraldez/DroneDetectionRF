import re

# 1. READ INTRO
with open('contenidos/10_Introduccion.tex', 'r', encoding='utf-8') as f:
    intro = f.read()

# 2. EXTRACT SECTION 3
sec3_pattern = re.compile(r'(% ─+\n\\section\{Principales sistemas de defensa anti-dron}.*?)(?=% ─+\n\\section\{Limitaciones fundamentales)', re.DOTALL)
match_sec3 = sec3_pattern.search(intro)
if match_sec3:
    sistemas_comerciales_text = match_sec3.group(1)
    intro = intro.replace(sistemas_comerciales_text, '')
else:
    print("Section 3 not found in intro")

# 3. ADD CITATIONS IN INTRO & REPLACE ENGLISH EMDASHES
intro = intro.replace('—denominados\n\gls{uav} en la literatura anglosajona—', ', denominados\n\gls{uav} en la literatura anglosajona,')
intro = intro.replace('—fotografía aérea, cartografía, agricultura de precisión, reparto de mercancías—', ', como fotografía aérea, cartografía, agricultura de precisión o reparto de mercancías,')
intro = intro.replace('—como el cierre del Aeropuerto de Gatwick en\ndiciembre de 2018, que afectó a más de 140\,000 pasajeros—', ', como el cierre del Aeropuerto de Gatwick en\ndiciembre de 2018 que afectó a más de 140\,000 pasajeros \\cite{ezuma_detection_2020},')
intro = intro.replace('contemporáneas.', 'contemporáneas \\cite{lacy_machine_2024}.')
intro = intro.replace('—conocidos en la literatura como sistemas\n\\textit{Counter-UAS} (C-UAS) o anti-dron—', ', conocidos en la literatura como sistemas\n\\textit{Counter-UAS} (C-UAS) o anti-dron,')
intro = intro.replace('—radar activo, sensores electroópticos o detección acústica—', ', como radar activo, sensores electroópticos o detección acústica,')
intro = intro.replace('—incluyendo modelos kamikazes\nde bajo coste denominados \\textit{First Person View} (FPV)—', ', incluyendo modelos kamikazes\nde bajo coste denominados \\textit{First Person View} (FPV),')
intro = intro.replace('lustros.', 'lustros \\cite{dod_cuas_2021}.')
intro = intro.replace('todo \gls{uav} en vuelo.', 'todo \gls{uav} en vuelo \\cite{easa_remoteid_2021}.')

# Add transition text where section 3 used to be
transicion = """Para hacer frente a estas amenazas emergentes, han surgido aproximaciones tecnológicas heterogéneas basadas en radares, sensores acústicos, cámaras electroópticas y receptores de radiofrecuencia. El análisis detallado de estas tecnologías y de las principales plataformas comerciales y militares actuales se aborda en el Capítulo~\\ref{cap:EstadoDelArte}. No obstante, la mayoría de estos sistemas se enfrentan a severas limitaciones operativas en entornos electromagnéticamente adversos que motivan el presente trabajo.

"""
intro = intro.replace('colaboración del operador.\n\n% ─', 'colaboración del operador.\n\n' + transicion + '% ─')

# 4. REMOVE AUTONOMOUS UAV PARAGRAPH
uav_autonomos_pattern = re.compile(r'\\paragraph\{Amenaza de los \\gls\{uav\} autónomos sin enlace RF activo\.\}.*?(?=\\paragraph)', re.DOTALL)
intro = re.sub(uav_autonomos_pattern, '', intro)

# Rename the section Title
intro = intro.replace('\\section{Limitaciones fundamentales de los sistemas actuales}', '\\section{Limitaciones fundamentales de los enfoques basados en RF}')
intro = intro.replace('los sistemas C-UAS de última generación presentan', 'los sistemas C-UAS basados en radiofrecuencia presentan')

# Fix other em-dashes left
intro = intro.replace('—que distribuyen', ', que distribuyen')
intro = intro.replace('singular— representan', 'singular, representan')
intro = intro.replace('—como una aeronave comercial—', ', como una aeronave comercial,')
intro = intro.replace('—en lugar de depender de firmas preprogramadas—', ', en lugar de depender de firmas preprogramadas,')
intro = intro.replace('—denominadas', ', denominadas')
intro = intro.replace('—que operan', ', que operan')

# 5. ADD SIMPLIFIED DIAGRAM TO OBJECTIVES
tikz_diagram = r"""diversidad de condiciones electromagnéticas adversas disponible públicamente.

\end{enumerate}

Para facilitar la comprensión global de la solución propuesta, la Figura~\ref{fig:esquema_general} ilustra de manera simplificada la arquitectura de detección diseñada en este trabajo, mostrando el flujo de la señal desde la captura RF hasta la decisión del clasificador.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\linewidth]{../figuras_arquitectura/singlestream_architecture.pdf}
    \caption{Esquema general simplificado de la arquitectura de detección de \gls{uav} propuesta basada en redes neuronales de valor complejo y extracción de características físicas.}
    \label{fig:esquema_general}
\end{figure}
"""
intro = intro.replace('diversidad de condiciones electromagnéticas adversas disponible públicamente.\n\n\end{enumerate}', tikz_diagram)

with open('contenidos/10_Introduccion.tex', 'w', encoding='utf-8') as f:
    f.write(intro)

# 6. READ ESTADO DEL ARTE
with open('contenidos/20_EstadoDelArte.tex', 'r', encoding='utf-8') as f:
    estado = f.read()

# Fix em-dashes in EstadoDelArte
estado = estado.replace('—radiofrecuencia, radar, electroóptica-infrarroja y acústica—', ', a saber, radiofrecuencia, radar, electroóptica-infrarroja y acústica,')
estado = estado.replace('—especialmente los de estructura plástica con rotores de pequeño diámetro—', ', especialmente los de estructura plástica con rotores de pequeño diámetro,')
estado = estado.replace('—transición crepuscular en la que la\ntemperatura del \gls{uav} es similar a la del fondo—', ', la transición crepuscular en la que la\ntemperatura del \gls{uav} es similar a la del fondo,')
estado = estado.replace('—sin emisión de energía que delate al sistema de\ndetección—', ', sin emisión de energía que delate al sistema de\ndetección,')
estado = estado.replace('—y, por extensión, sobre el modelo de \gls{uav} y\nla identidad del piloto mediante triangulación—', ', y, por extensión, sobre el modelo de \gls{uav} y\nla identidad del piloto mediante triangulación,')
estado = estado.replace('—típicamente entre 58 y 200 bins\na 14\,kHz de resolución frecuencial—', ', típicamente entre 58 y 200 bins\na 14\,kHz de resolución frecuencial,')
estado = estado.replace('—incluyendo los\nsistemas Futaba FASST, FrSky ACCESS/D16, Graupner HoTT y TBS Crossfire—', ', incluyendo los\nsistemas Futaba FASST, FrSky ACCESS/D16, Graupner HoTT y TBS Crossfire,')
estado = estado.replace('—con un jitter de fabricación inferior al 2\,\%—', ', con un jitter de fabricación inferior al 2\,\%,')
estado = estado.replace('—hasta $-20$\,dB—', ', hasta $-20$\,dB,')
estado = estado.replace('—combinando ventanas de análisis cortas para capturar\ntransitorios y largas para capturar patrones de periodicidad—', ', combinando ventanas de análisis cortas para capturar\ntransitorios y largas para capturar patrones de periodicidad,')
estado = estado.replace('—por ejemplo, un salto FHSS de un \gls{uav}\nde radiocontrol y una ráfaga Bluetooth de corta duración—', ', por ejemplo, un salto FHSS de un \gls{uav}\nde radiocontrol y una ráfaga Bluetooth de corta duración,')
estado = estado.replace('—distinguiendo unidades\ndel mismo modelo en función de las imperfecciones hardware del transmisor—', ', distinguiendo unidades\ndel mismo modelo en función de las imperfecciones hardware del transmisor,')
estado = estado.replace('—en particular las redes de memoria de larga y\ncorta duración (LSTM) y sus variantes bidireccionales (BiLSTM)—', ', en particular las redes de memoria de larga y\ncorta duración (LSTM) y sus variantes bidireccionales (BiLSTM),')
estado = estado.replace('—como CBAM (\\textit{Convolutional Block\nAttention Module}) \cite{woo_cbam_2018}—', ', como CBAM (\\textit{Convolutional Block\nAttention Module}) \cite{woo_cbam_2018},')
estado = estado.replace('—que fusionan información procedente de múltiples\nmodalidades o representaciones de la señal—', ', que fusionan información procedente de múltiples\nmodalidades o representaciones de la señal,')
estado = estado.replace('—como la entropía cruzada binaria—', ', como la entropía cruzada binaria,')
estado = estado.replace('—\gls{crelu}: $\\text{CReLU}(z) = \\text{ReLU}(\\text{Re}(z)) + j\\cdot\\text{ReLU}(\\text{Im}(z))$—', ', \gls{crelu} ($\\text{CReLU}(z) = \\text{ReLU}(\\text{Re}(z)) + j\\cdot\\text{ReLU}(\\text{Im}(z))$),')
estado = estado.replace('—que codifican la fase\nde la portadora—', ', que codifican la fase\nde la portadora,')
estado = estado.replace('—donde la distinción visual entre señal\nde interés e interferencia es ambigua incluso para un experto humano—', ', donde la distinción visual entre señal\nde interés e interferencia es ambigua incluso para un experto humano,')
estado = estado.replace('—interferencias Bluetooth o Wi-Fi que el detector confunde con saltos FHSS—', ', interferencias Bluetooth o Wi-Fi que el detector confunde con saltos FHSS,')
estado = estado.replace('—en el que un modelo «profesor» entrenado en\ncondiciones favorables genera pseudo-etiquetas para muestras de condiciones\nadversas, que se emplean para entrenar un modelo «alumno» más robusto—', ', en el que un modelo «profesor» entrenado en\ncondiciones favorables genera pseudo-etiquetas para muestras de condiciones\nadversas y que se emplean para entrenar un modelo «alumno» más robusto,')
estado = estado.replace('—ráfagas individuales—', ', ráfagas individuales,')
estado = estado.replace('—operador \\textit{noisy-OR}—', ', operador \\textit{noisy-OR},')

# 7. ADD SECTION 3 TEXT INTO ESTADO DEL ARTE
sec_target = "% ══════════════════════════════════════════════════════════════════════════════\n\\section{Aprendizaje automático para la detección de señales RF de UAV}"

sistemas_comerciales_text = sistemas_comerciales_text.replace(r'\section{Principales sistemas de defensa anti-dron}', r'\subsection{Sistemas C-UAS Comerciales y Militares}')
sistemas_comerciales_text = sistemas_comerciales_text.replace(r'\label{sec:sistemas_antidron}', r'\label{subsec:sistemas_comerciales}')
# Replace market citation need
sistemas_comerciales_text = sistemas_comerciales_text.replace('según estimaciones de diversos\ninformes sectoriales.', 'según estimaciones de diversos\ninformes sectoriales \\cite{dod_cuas_2021}.')

estado = estado.replace(sec_target, sistemas_comerciales_text + '\n\n' + sec_target)

with open('contenidos/20_EstadoDelArte.tex', 'w', encoding='utf-8') as f:
    f.write(estado)

print("Refactor completed successfully!")
