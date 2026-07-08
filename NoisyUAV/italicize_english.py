import os
import glob
import re

path = r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos"
active_files = [
    "10_Introduccion.tex",
    "20_EstadoDelArte_2.tex",
    "30_MarcoTeorico.tex",
    "40_ArquitecturaImplementacion.tex",
    "50_ResultadosDiscusion.tex"
]

words_to_italicize = [
    r"offline", r"spoofing", r"jamming", r"clutter", r"multipath",
    r"hardware", r"firmware", r"dataset", r"datasets", r"burst", r"bursts",
    r"shortcut learning", r"label noise", r"phased-array", r"clutter",
    r"z-score", r"padding", r"stride", r"max-pooling", r"batch normalization",
    r"dropout", r"pipeline"
]

def italicize_word(word_pattern, text):
    # Match the word but only if it's NOT immediately preceded by { (which usually means \textit{word})
    # or preceded by \gls{ (which means it's a glossary term).
    # We'll use a negative lookbehind for {
    pattern = re.compile(rf'(?<!\{{)\b({word_pattern})\b(?!\}})', re.IGNORECASE)
    return pattern.sub(r'\\textit{\1}', text)

for filename in active_files:
    filepath = os.path.join(path, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    for word in words_to_italicize:
        text = italicize_word(word, text)
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

print("English terms italicized.")
