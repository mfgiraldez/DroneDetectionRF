import re
with open('contenidos/20_EstadoDelArte.tex', 'r', encoding='utf-8') as f:
    estado = f.read()

sistemas_comerciales_text = r"""
% -------------------------------------------------------------------------------
\subsection{Sistemas C-UAS Comerciales y Militares}
\label{subsec:sistemas_comerciales}
% -------------------------------------------------------------------------------

El mercado de los sistemas C-UAS ha experimentado un crecimiento exponencial desde
2020. Valorado en aproximadamente 2\,700 millones de dólares en 2024, se proyecta
que alcance los 11\,000 millones antes de 2030, según estimaciones de diversos
informes sectoriales \cite{dod_cuas_2021}. A continuación se describen los sistemas de referencia más
representativos, con énfasis en su modalidad de detección y sus capacidades
operativas documentadas.

\paragraph{DroneShield DroneSentry y RfPatrol.}\mbox{}\\[0.5em]
DroneShield es una empresa australiana especializada en C-UAS cuya plataforma
\textit{DroneSentry} implementa una detección multicapa basada en la fusión de
sensores RF, radar de onda continua y cámaras electroópticas. El módulo RF analiza
el espectro entre 70\,MHz y 6\,GHz, siendo capaz de identificar protocolos de
control DJI, Futaba y otros fabricantes comerciales mediante análisis espectral en
tiempo real. Su variante portátil \textit{RfPatrol} está concebida para protección
de personal en movimiento y ha sido desplegada por unidades militares de varios
países aliados de la OTAN. Ambas plataformas han sido objeto de contratos
multimillonarios con organismos de defensa de Estados Unidos y la Unión Europea
durante el período 2024-2025.

\paragraph{Dedrone RF-300.}\mbox{}\\[0.5em]
Dedrone, empresa de origen alemán actualmente con sede en Estados Unidos, ofrece
la plataforma \textit{DedroneRF-300}, un sistema pasivo de detección RF que opera
mediante el análisis de las bandas \gls{ism} de 2,4\,GHz y 5,8\,GHz. El sistema
incorpora una base de datos de firmas espectrales de más de 500 modelos de \gls{uav}
comerciales y militares, permitiendo la identificación por protocolos y la
geolocalización del piloto mediante triangulación de la señal de control. Ha sido
desplegado en eventos de alta seguridad, incluyendo el Foro Económico Mundial de
Davos y diversas instalaciones de la Agencia Central de Inteligencia (CIA).

\paragraph{Thales RAPIDFire y sistemas de energía dirigida.}\mbox{}\\[0.5em]
Thales ha desarrollado el sistema \textit{RAPIDFire}, originalmente concebido como
cañón antiaéreo, que ha sido adaptado para la neutralización de \gls{uav} mediante
proyectiles de alta cadencia de tiro. En paralelo, el creciente interés en los
sistemas de energía dirigida , en particular el láser de alta energía y los emisores
de microondas de alta potencia (HPM), refleja la búsqueda de soluciones de
neutralización con menor coste por disparo y menor riesgo de daños colaterales en
entornos urbanos. Sin embargo, estos sistemas requieren que la detección y el
seguimiento del objetivo estén garantizados con anterioridad a la neutralización,
lo que sitúa a los módulos de detección RF como el primer eslabón imprescindible
de la cadena C-UAS.

\paragraph{AUDS (\textit{Anti-UAV Defence System}).}\mbox{}\\[0.5em]
El sistema AUDS, desarrollado conjuntamente por Chess Dynamics, Enterprise Control
Systems y Blighter Surveillance Systems, combina radar de escaneo electrónico, cámara
electroóptica de largo alcance e inhibidor de radiofrecuencia (\textit{jammer}).
Su arquitectura permite la detección a distancias de hasta 8\,km , dependiendo de las
condiciones atmosféricas y el tamaño del \gls{uav}, y la neutralización mediante la
inhibición selectiva de las frecuencias de telemetría y control. Ha sido evaluado
por el Ministerio de Defensa del Reino Unido y desplegado en operaciones en el
teatro de operaciones de Oriente Próximo.

\paragraph{Rafael Drone Dome.}\mbox{}\\[0.5em]
El sistema israelí \textit{Drone Dome}, desarrollado por Rafael Advanced Defense
Systems, representa la arquitectura C-UAS de referencia en escenarios de alta
intensidad. Integra radares 3D de banda X, sensores electroópticos/infrarrojos y un
sistema de guerra electrónica capaz de bloquear los canales de telemetría GPS y RF.
Adicionalmente, incorpora un módulo de neutralización mediante láser de estado sólido
de alta potencia. El sistema fue desplegado durante los Juegos Olímpicos de Tokio
2020 y se ha empleado operativamente en la protección del espacio aéreo israelí
frente a incursiones de \gls{uav} procedentes del Líbano y Gaza.
"""

target = "% ------------------------------------------------------------------------------\n\\section{Aprendizaje automático para la detección de señales RF de UAV}"

if target in estado:
    estado = estado.replace(target, sistemas_comerciales_text + "\n\n" + target)
    with open('contenidos/20_EstadoDelArte.tex', 'w', encoding='utf-8') as f:
        f.write(estado)
    print("Success")
else:
    print("Not found")

