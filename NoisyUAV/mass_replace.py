import os

def replace_in_file(filepath, replacements):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    for old, new in replacements:
        text = text.replace(old, new)
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

# 1. Update Chapter 1 (10_Introduccion.tex)
replace_in_file(
    r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\10_Introduccion.tex",
    [
        ("el espectrograma de módulo queda dominado por ruido estocástico, mientras que la componente de fase conserva una estructura geométrica determinista vinculada a la portadora que puede resultar discriminante.", 
         "la magnitud del espectrograma comienza a quedar fuertemente enmascarada por el ruido estocástico, mientras que la componente de fase conserva una estructura geométrica determinista vinculada a la portadora que puede aportar una ventaja competitiva en la clasificación.")
    ]
)

# 2. Update Chapter 3 (30_MarcoTeorico.tex)
replace_in_file(
    r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\30_MarcoTeorico.tex",
    [
        ("-14\\,dB", "-12\\,dB"),
        ("fracasa por completo.", "ve severamente mermada su capacidad de detección.")
    ]
)

# 3. Update Chapter 4 (40_ArquitecturaImplementacion.tex)
replace_in_file(
    r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\40_ArquitecturaImplementacion.tex",
    [
        ("vector analítico I/Q", "vector fasorial I/Q"),
        ("limitación de carácter físico", "limitación de carácter estructural"),
        ("cuarta característica física", "cuarta métrica"),
        ("detalló analíticamente", "detalló"),
        ("mecanismo analítico intrínseco", "mecanismo algorítmico intrínseco"),
        ("parámetros físicos escalares", "parámetros escalares"),
        ("Motivación Analítica", "Motivación Teórica"),
        ("garantiza de manera analítica", "garantiza por diseño"),
        ("duración física", "duración temporal"),
        ("metadatos físicos", "metadatos contextuales")
    ]
)

# 4. Update Chapter 5 (50_ResultadosDiscusion.tex)
replace_in_file(
    r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\50_ResultadosDiscusion.tex",
    [
        ("degradación física", "degradación estocástica"),
        ("umbral físico local", "umbral de ruido local"),
        ("capacidad analítica", "capacidad de inferencia"),
        ("umbral físico exacto", "umbral de ruido exacto")
    ]
)

print("Mass replacement complete.")
