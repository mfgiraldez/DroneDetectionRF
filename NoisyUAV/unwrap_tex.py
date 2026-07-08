import os
import re

in_path = r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\10_Introduccion (1).tex"
out_path = r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos\10_Introduccion.tex"

with open(in_path, "r", encoding="utf-8") as f:
    text = f.read()

# We want to remove single newlines but preserve double newlines (paragraph breaks).
# Also preserve newlines that start with a LaTeX command like \section, \begin, \end, \item, %, etc.
# A simple way to unwrap: split by double newlines to get paragraphs.
paragraphs = re.split(r'\n\s*\n', text)

unwrapped_paragraphs = []
for p in paragraphs:
    # If the paragraph is a comment block, don't touch it much.
    if p.startswith('%'):
        unwrapped_paragraphs.append(p)
    elif p.startswith('\\begin') or p.startswith('\\end'):
        # Just keep it as is, but maybe unwrap inside? It's safer to not aggressively unwrap environments unless needed.
        # Actually, for standard text, we can just replace single newlines with a space.
        pass
        
    # A robust unwrap: replace single newlines with space, EXCEPT if the next line is a comment or a list item.
    # Let's do it line by line.
    lines = p.split('\n')
    new_p = ""
    for i, line in enumerate(lines):
        if i == 0:
            new_p += line
        else:
            # If the current line starts with \item, \\, %, \section, etc., keep the newline.
            if re.match(r'^\s*(\\|%|})', line) or line.strip() == "":
                new_p += "\n" + line
            else:
                # If the previous line ended with a comment, we need a newline.
                if new_p.strip().endswith('%'):
                    new_p += "\n" + line
                else:
                    # Otherwise, join with space
                    new_p += " " + line.strip()
    
    # Fix multiple spaces
    new_p = re.sub(r' +', ' ', new_p)
    unwrapped_paragraphs.append(new_p)

final_text = "\n\n".join(unwrapped_paragraphs)

with open(out_path, "w", encoding="utf-8") as f:
    f.write(final_text)

print("Done unwrapping.")
