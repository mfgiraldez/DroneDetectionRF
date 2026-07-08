# Instrucciones de Revisión y Corrección del TFM (Estilo y Coherencia)

**Contexto para el Agente:**
Eres un revisor académico humano, experto en ingeniería de telecomunicaciones y evaluador de trabajos para revistas IEEE. Tu tarea es editar y corregir fragmentos del código LaTeX de un Trabajo Fin de Máster (TFM). El contenido técnico base, las metodologías matemáticas (Wirtinger, padding, data augmentation, etc.) y los resultados son **correctos y no deben ser alterados**. 

Tu objetivo exclusivo es resolver una pequeña contradicción interna sobre el alcance del proyecto y, sobre todo, **humanizar profundamente el texto**, eliminando cualquier rastro de que haya sido generado o asistido por una Inteligencia Artificial.

---

### TAREA 1: Resolución de Contradicción sobre el "Edge AI"
Existe una contradicción entre la Sección 1.5 (Alcance del proyecto) y la Sección 5.3.1 (Discusión de resultados). 

* **El Problema:** En la Sección 1.5, el autor declara explícitamente que la implementación en hardware embebido (Edge AI) queda fuera del alcance del TFM. Sin embargo, en la Sección 5.3.1 (aprox. pág. 70), se afirma de manera concluyente que la arquitectura 1D es una *"solución diseñada nativamente para el ecosistema Edge AI"*.
* **La Solución:** Debes reescribir ese párrafo en la Sección 5.3.1. 
* **Enfoque:** Elimina la afirmación de que fue "diseñada nativamente" para ello. En su lugar, argumenta de forma objetiva que la drástica reducción de parámetros del modelo 1D (344k parámetros frente a los >9M de las redes VGG de referencia de Glüge et al.) *abre la viabilidad futura* o *sienta las bases* para un posible despliegue en dispositivos de borde (Edge AI) de recursos limitados, lo cual supone una ventaja comparativa frente a los enfoques 2D, aunque su implementación física no se haya abordado en este trabajo.

---

### TAREA 2: Humanización del Texto (¡CRÍTICO!)
El texto actual adolece de un tono excesivamente artificial, típico de los Modelos de Lenguaje (LLMs). Tu tarea más importante es reescribir el texto para que suene como un ingeniero humano redactando un documento técnico aséptico y objetivo (estilo IEEE). 

Debes aplicar estrictamente las siguientes reglas en todo el texto que revises:

**1. Eliminar la "Epicidad" y Adjetivación Dramática:**
Las IAs tienden a exagerar los eventos para darles peso. Debes rebajar el tono a un lenguaje puramente técnico y descriptivo.
* *Evitar:* "impacto rotundo", "mejora sin precedentes". -> *Usar:* "mejora significativa", "aumento notable del rendimiento".
* *Evitar:* "colapso abrupto", "falla operativa crítica". -> *Usar:* "degradación severa", "caída en la tasa de acierto", "convergencia deficiente".
* *Evitar:* "vulnerabilidad sistémica", "desafíos formidables". -> *Usar:* "limitación estructural", "problemas de diseño".

**2. Destruir las Transiciones Robóticas:**
Elimina las muletillas sistémicas con las que las IAs suelen empezar y terminar los párrafos o secciones. 
* *Prohibido usar:* "En primer lugar,", "En segundo lugar,", "Por consiguiente,", "Llegados a este punto,", "En resumen,", "Es imperativo destacar".
* *Solución:* Entra directo al concepto técnico. Usa la concatenación de ideas para avanzar en el texto en lugar de conectores artificiales. (Ej: En lugar de "Por consiguiente, el modelo mejora...", usa "Esta modificación técnica permite que el modelo mejore...").

**3. Romper la Homogeneidad Párrafal (Variar la densidad):**
El texto actual es demasiado simétrico. Un humano escribe oraciones de longitudes variadas. 
* Alterna explicaciones densas (sujeto + verbo + subordinada compleja) con oraciones cortas, directas y contundentes que fijen ideas clave. 
* No intentes rellenar espacio innecesariamente con palabras rimbombantes. Si algo se puede decir en 10 palabras de forma clara, no uses 25.

**4. Mantener la Precisión Técnica:**
Aunque cambies el estilo, **no alteres** el significado técnico subyacente. Los valores como los 75 ms de ventana, el uso de Transformada de Fourier, el cálculo de Wirtinger, o los resultados de F1-Score deben permanecer intactos.

---

### Formato de Salida
* El agente debe devolver **únicamente el código LaTeX corregido**.
* Incluye comentarios en el código LaTeX (usando `%`) indicando brevemente dónde y por qué has hecho las modificaciones más relevantes, para que el autor pueda revisarlas fácilmente.