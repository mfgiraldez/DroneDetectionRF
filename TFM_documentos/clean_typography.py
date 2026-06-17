import re

def clean_text(text):
    # 1. Replace em-dash with comma + space, but be careful with existing spaces.
    # We will replace all em-dashes with commas, then clean up the spaces.
    text = text.replace('—', ',')
    
    # 2. Fix spaces around commas
    text = text.replace(' ,', ',')
    text = text.replace(',,', ',')
    
    # 3. Fix cases where comma is at the start of a line. 
    # Example: "palabra\n, como" -> "palabra,\ncomo"
    text = re.sub(r'([^\s])\n,\s*', r'\1,\n', text)
    
    # 4. Ensure space after comma if followed by a letter
    text = re.sub(r',([a-zA-Z])', r', \1', text)
    
    # Clean up double spaces just in case
    text = text.replace('  ', ' ')
    
    # Keep math mode untouched if possible, but our replacements above are mostly safe.
    # Exception: $80\,\%$ might have been affected?
    # No, \, is backslash comma, not a space comma.
    
    return text

for filename in ['contenidos/10_Introduccion.tex', 'contenidos/20_EstadoDelArte.tex']:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # For lines like % Capítulo 1 — Introducción
    content = content.replace('%  Capítulo 1 — Introducción', '%  Capítulo 1 - Introducción')
    content = content.replace('%  Capítulo 2 — Estado del Arte', '%  Capítulo 2 - Introducción')

    new_content = clean_text(content)
    
    # Fix specific math/latex things that might have been broken by space after comma
    new_content = new_content.replace('\\, \\%', '\\,\\%')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)

print("Done")
