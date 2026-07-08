import re

path = r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\10_Introduccion.tex"

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Fix doubled section blocks
# Find pattern:
# % -----\n\section{...}\n\label{...}\n% -----\n\n% -----\n\section{...}\n\label{...}\n% -----
pattern = re.compile(r'(% -+\n\\section\{[^\}]+\}\n\\label\{[^\}]+\}\n% -+\n)\n\1')
text = pattern.sub(r'\1', text)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Duplicates fixed.")
