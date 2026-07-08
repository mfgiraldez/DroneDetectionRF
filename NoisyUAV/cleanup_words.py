import os

def replace_in_file(filepath, replacements):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    for old, new in replacements:
        text = text.replace(old, new)
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

# Fix -14 dB
replace_in_file(
    r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\30_MarcoTeorico.tex",
    [
        (r"$-14$\,dB", r"$-12$\,dB"),
        (r"$-14$~dB", r"$-12$~dB"),
        (r"-14\,dB", r"-12\,dB"),
        (r"cabe destacar", r"es relevante señalar"),
        (r"presencia física de regímenes", r"presencia inherente de regímenes"),
        (r"inspección física de las", r"inspección directa de las"),
        (r"motivación analítica", r"motivación teórica"),
        (r"escenario analítico", r"escenario de prueba")
    ]
)

replace_in_file(
    r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\40_ArquitecturaImplementacion.tex",
    [
        (r"cabe destacar", r"conviene señalar"),
        (r"segmentar físicamente cada", r"aislar temporalmente cada"),
        (r"severidad analítica", r"rigor matemático")
    ]
)

replace_in_file(
    r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\20_EstadoDelArte_2.tex",
    [
        (r"oráculo físico", r"referencia fundamental"),
        (r"físicamente con SNR", r"explícitamente con SNR")
    ]
)

replace_in_file(
    r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\10_Introduccion.tex",
    [
        (r"extinción analítica de la ráfaga", r"extinción matemática de la ráfaga")
    ]
)

print("Final cleanup complete.")
