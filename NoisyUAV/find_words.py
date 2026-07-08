import os
import glob
import re

path = r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos"
files = glob.glob(os.path.join(path, "*.tex"))

words_to_find = [
    r"analític[oa]s?",
    r"físic[oa]s?",
    r"categóric[oa]s?",
    r"es crucial",
    r"cabe destacar",
    r"en conclusión",
    r"-14\s*(?:\\,|~)?dB"
]

for file in files:
    if "(1)" in file or "intro.tex" in file or "(2)" in file: continue # Skip obsolete
    print(f"--- {os.path.basename(file)} ---")
    with open(file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            for word in words_to_find:
                if re.search(word, line, re.IGNORECASE):
                    print(f"L{i+1} [{word}]: {line.strip()}")
