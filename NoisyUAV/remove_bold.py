import os
import glob

# Path to the LaTeX files
path = r"c:\repos\DroneDetectionRF\TFM_documentos\contenidos"
tex_files = glob.glob(os.path.join(path, "*.tex"))

def unwrap_textbf(line):
    # If it's a table row or caption, leave it alone
    if "&" in line or "\\caption" in line or "\\toprule" in line or "\\midrule" in line:
        return line
        
    # We want to replace \textbf{X} with X, even with nested braces inside X.
    # We will do this by finding \textbf{ and then finding the matching closing brace.
    while "\\textbf{" in line:
        start_idx = line.find("\\textbf{")
        # Find matching closing brace
        brace_count = 0
        end_idx = -1
        for i in range(start_idx + 8, len(line)):
            if line[i] == '{':
                brace_count += 1
            elif line[i] == '}':
                if brace_count == 0:
                    end_idx = i
                    break
                else:
                    brace_count -= 1
        
        if end_idx != -1:
            # Extract content
            inner_content = line[start_idx + 8 : end_idx]
            # Replace in line
            line = line[:start_idx] + inner_content + line[end_idx+1:]
        else:
            # Malformed or multiline \textbf, break to avoid infinite loop
            break
            
    return line

for filepath in tex_files:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        new_lines.append(unwrap_textbf(line))
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

print("Bold text removed successfully from all files (except tables/captions).")
