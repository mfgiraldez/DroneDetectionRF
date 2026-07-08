# Directrices para la Redacción del Capítulo 6: Conclusiones y Trabajo Futuro

**Fichero de instrucciones para agente redactor**
**Proyecto:** TFM — Detección de UAV mediante CV-CNN en banda ISM 2.4 GHz
**Archivo destino:** `TFM_documentos/contenidos/60_Conclusiones.tex`

---

## 1. Contexto técnico obligatorio

Antes de redactar **cualquier línea**, el agente debe leer los siguientes archivos para conocer
el estado real del documento y no introducir contradicciones:

- `TFM_documentos/contenidos/10_Introduccion.tex` — objetivos declarados y alcance del proyecto.
- `TFM_documentos/contenidos/50_ResultadosDiscusion.tex` — resultados definitivos y discusión.
- `TFM_documentos/contenidos/40_ArquitecturaImplementacion.tex` — arquitecturas implementadas.
- `.agents/AGENTS.md` — regla de lectura de contexto (MANDATORIA antes de escribir).

### Datos cuantitativos clave (no inventar, no modificar)

| Magnitud | Valor |
|---|---|
| Parámetros del modelo definitivo | 344 291 |
| Parámetros mínimos de referencia (VGG11, Glüge et al.) | 9.35 M |
| Parámetros máximos de referencia (VGG19, Glüge et al.) | 20.16 M |
| F1-Score modo seguridad (DualStream-DynamicSlidingWindow) | 90.12 % |
| Especificidad modo seguridad | 97.10 % |
| SNR crítica (Grupo D) | < −12 dB |
| SNR umbral de detección estable propuesta | ≥ −2 dB |
| Ventana de observación del dataset | 75 ms / 1 048 576 muestras |
| Frecuencia de muestreo | 14 MHz |
| Clases del dataset NoisyUAV v2 | 7 (6 modelos UAV + clase ruido) |
| Factor de reducción respecto a VGG11 | × 27 |
| Factor de reducción respecto a VGG19 | × 58 |

### Afirmaciones que el capítulo NO debe hacer

- ❌ La arquitectura está diseñada nativamente para Edge AI. ← Contradicción ya corregida en Cap. 5.
  El enunciado correcto: la reducción paramétrica conseguida abre la viabilidad futura para
  despliegue en dispositivos embebidos, pero la implementación física no se aborda en este trabajo.
- ❌ El modelo supera con gran margen a todos los trabajos del estado del arte de forma genérica.
  Usar la comparativa concreta con Glüge et al. y los datos de la Tabla sota_comparative.
- ❌ Resultados en regímenes de SNR extrema (< −12 dB, Grupo D) son perfectos o excelentes.
  En el Grupo D las detecciones son marginales por definición física del problema.

---

## 2. Estructura esperada del capítulo

El capítulo debe seguir el esquema canónico IEEE/ACM para secciones de conclusiones.
No es necesario usar exactamente estos títulos, pero sí cubrir estos bloques en este orden:

`
\chapter{Conclusiones y Trabajo Futuro}

  \section{Conclusiones}
    - Síntesis de los objetivos y su grado de cumplimiento.
    - Contribuciones técnicas originales (en prosa, concisas).
    - Limitaciones reconocidas del trabajo.

  \section{Líneas de Trabajo Futuro}
    - Extensiones directas y bien justificadas.
    - No prometer lo que no se ha demostrado.
`

### 2.1 Bloque: Conclusiones

Debe cubrir en prosa continua, SIN listas de viñetas, los siguientes puntos:

1. **Problema abordado**: detección pasiva de UAV en banda ISM 2.4 GHz bajo SNR extrema
   y con interferencia co-canal real (Wi-Fi, Bluetooth). El reto específico del label noise
   inherente al dataset NoisyUAV v2 (ventana de 75 ms con baja ocupación activa del canal).

2. **Contribución 1 — Detector analítico de entropía espectral**: algoritmo determinista
   (sin parámetros entrenables) basado en entropía de Shannon + blanqueo espectral + CFAR
   adaptativo que localiza ráfagas FHSS en regímenes de SNR donde el detector de energía
   clásico falla.

3. **Contribución 2 — Arquitectura DualStream con CV-CNN**: procesamiento directo de la
   señal IQ compleja preservando la coherencia de fase; fusión con estimación PSD mediante
   bloque de atención dinámica; evaluación continua del canal mediante ventana deslizante.

4. **Resultado principal**: el modelo definitivo (DualStream-DynamicSlidingWindow) logra un
   F1-Score del 90.12 % y Especificidad del 97.10 % en modo de máxima seguridad, igualando
   la precisión de las redes VGG de Glüge et al. con entre 27 y 58 veces menos parámetros,
   sin pre-segmentación de las grabaciones.

5. **Limitación reconocida**: ausencia de validación sobre hardware SDR en tiempo real;
   evaluación realizada únicamente sobre NoisyUAV v2; rendimiento marginal en el Grupo D
   (SNR < −12 dB) que confirma el límite físico del problema.

### 2.2 Bloque: Trabajo Futuro

Líneas directamente justificadas por las limitaciones identificadas en el propio trabajo:

1. **Despliegue en hardware embebido**: validar la inferencia del modelo en un receptor SDR
   en tiempo real. La dimensionalidad de 344 291 parámetros hace esta línea técnicamente
   plausible; es la extensión natural del trabajo, no un logro actual.

2. **Generalización multi-banda**: ampliar la evaluación a la banda de 5.8 GHz, donde operan
   protocolos de vídeo FPV de uso creciente en UAV comerciales.

3. **Entrenamiento con ruido de etiquetas real**: explorar técnicas de aprendizaje débilmente
   supervisado que no requieran el filtrado por SNR del conjunto de entrenamiento.

4. **Fusión con modalidades complementarias**: combinar la detección RF con información
   acústica o radar para reducir la tasa de falsos positivos en entornos de alta saturación.

---

## 3. Reglas de estilo (MANDATORIAS)

Estas reglas se derivan de la revisión aplicada en los Capítulos 1–5 del mismo documento.

### 3.1 Marcadores de lista prohibidos en el cuerpo del texto

No usar como inicio de párrafo ni como estructura narrativa:

| Prohibido | Alternativa |
|---|---|
| En primer lugar, | Entrar directo al concepto; usar la idea previa como nexo. |
| En segundo lugar, | Continuar con la idea anterior como nexo lógico. |
| Finalmente, (como marcador de lista) | Preferir fusión de párrafos. |
| A continuación, (como enumerador) | Eliminar; el texto debe fluir sin señalización. |
| Llegados a este punto, | Directo al concepto. |
| Por consiguiente, (inicio de párrafo) | Integrar la consecuencia en la oración anterior. |
| En conclusión, / En resumen, | Prohibidos siempre en este capítulo. |

Uso permitido: exclusivamente dentro de entornos enumerate o itemize de LaTeX cuando
se lista material técnico (pasos de un algoritmo, fases de evaluación, etc.).

### 3.2 Expresiones de IA prohibidas

| Prohibido | Alternativa |
|---|---|
| Cabe destacar que | Eliminar; si la idea es importante, expresarla directamente. |
| Resulta crucial / fundamental / esencial | Es necesario, o reestructurar. |
| Resulta imperativo | Es necesario, se requiere. |
| Resulta pertinente señalar que | Eliminar el meta-comentario. |
| Es de vital importancia | Eliminar o reformular. |
| Sin lugar a dudas | Eliminar. |
| A todas luces | Eliminar. |
| los pilares fundamentales | los elementos centrales, los componentes principales. |
| impacto rotundo | mejora medible, reducción cuantificable. |
| sin precedentes | Eliminar o sustituir por comparativa concreta. |
| un hito técnico y logístico | Descripción objetiva del logro. |

### 3.3 Adjetivación dramática prohibida

| Prohibido | Alternativa |
|---|---|
| radical reducción | reducción significativa / reducción de un factor X |
| drásticamente (énfasis vacío) | especificar el valor numérico, o eliminar |
| colapso abrupto | degradación severa, caída en el rendimiento |
| vulnerabilidad sistémica | limitación estructural |
| superioridad evidente | describir la métrica concreta que la justifica |
| paridad absoluta | rendimiento equivalente |
| con gran margen | citar el delta numérico concreto |
| diseñada nativamente para | viable para / compatible con (con matiz de alcance) |

### 3.4 Primera persona prohibida

Usar estilo impersonal. No: nuestra arquitectura, nuestro modelo, hemos demostrado.
Sí: la arquitectura propuesta, el modelo desarrollado, los resultados obtenidos demuestran.

### 3.5 Cursiva y negrita

- Negrita (\textbf{}): exclusivamente en cabeceras de tabla y en primer uso de un término
  que se define en ese párrafo. No en texto normal para énfasis.
- Cursiva (\textit{}): términos en inglés sin traducción establecida (label noise, shortcut
  learning, burst, pipeline, sliding window) y nombres de modelos propios
  (DualStream-DynamicSlidingWindow). No en español para énfasis.

### 3.6 Formato de unidades y números

Seguir el patrón establecido en el documento:
- Separador de millares: \, (espacio fino LaTeX). Ej: 344\,291 parámetros.
- Unidades con espacio fino. Ej: 14\,MHz, 75\,ms.
- SNR negativa en modo matemático: $-12$\,dB, no −12 dB en texto plano.

### 3.7 Referencias cruzadas

Usar las etiquetas \label ya definidas:
- \ref{cap:EstadoDelArte} — Capítulo 2
- \ref{cap:MarcoTeorico} — Capítulo 3
- \ref{cap:arquitectura} — Capítulo 4
- \ref{cap:resultados_discusion} — Capítulo 5
- \ref{sec:discusion} — Sección de discusión
- \ref{tab:sota_comparative} — Tabla comparativa con el estado del arte
- \gls{uav}, \gls{snr}, \gls{cv-cnn}, \gls{cfar}, \gls{fhss}, \gls{psd}, \gls{sdr}
  → usar siempre el glosario, no expandir manualmente los acrónimos.

---

## 4. Estructura LaTeX esperada

`latex
\chapter{Conclusiones y Trabajo Futuro}
\label{cap:conclusiones}

\lettrine[lraise=-0.1, lines=2, loversize=0.2]{E}{ste} ...

\section{Conclusiones}
\label{sec:conclusiones}

% Párrafo 1: síntesis del problema y motivación
% Párrafo 2: contribución del detector de entropía (Caps. 3 y 4)
% Párrafo 3: contribución de la arquitectura CV-CNN y DualStream (Cap. 4)
% Párrafo 4: resultados cuantitativos y posicionamiento frente al estado del arte
% Párrafo 5: limitaciones reconocidas

\section{Líneas de Trabajo Futuro}
\label{sec:trabajo_futuro}

% Párrafo 1: despliegue en hardware SDR en tiempo real (Edge AI)
% Párrafo 2: generalización a otras bandas frecuenciales
% Párrafo 3: mejoras en la gestión del label noise
% Párrafo 4: fusión con otras modalidades de detección
`

- Sección de Conclusiones: entre 400 y 600 palabras en prosa continua, sin listas de viñetas.
- Sección de Trabajo Futuro: párrafos independientes (uno por línea), sin enumeración explícita.
- Longitud total del capítulo: aproximadamente 600–900 palabras en el cuerpo del texto.

---

## 5. Tono y densidad del texto

- Tono: técnico, aséptico, impersonal. Equivalente al de las conclusiones de un artículo
  IEEE Transactions on Signal Processing o IEEE Access.
- Densidad: variar la longitud de las oraciones. Alternar oraciones largas con subordinadas
  (para argumentar) con oraciones cortas y directas (para fijar datos clave).
- Redundancia: no repetir literalmente afirmaciones del Abstract o la Introducción.
  Las conclusiones sintetizan y cierran; no reexponen lo que ya se ha dicho.

---

## 6. Lista de verificación final

Antes de considerar el texto terminado, verificar:

- [ ] Ningún párrafo empieza con los marcadores prohibidos de la Sección 3.1.
- [ ] No aparece ninguna expresión de la lista de la Sección 3.2.
- [ ] No se afirma que la arquitectura está diseñada nativamente para Edge AI.
- [ ] Todos los valores numéricos coinciden con la tabla de la Sección 1.
- [ ] No hay primera persona (nuestra, hemos, nuestro).
- [ ] Las referencias cruzadas usan \ref{} y \gls{} correctamente.
- [ ] Los números negativos de SNR están en modo matemático ($-12$\,dB).
- [ ] No hay texto en negrita fuera de cabeceras de tabla.
