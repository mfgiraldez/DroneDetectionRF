import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

CSV_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\modelo_v2_2_dual\figuras_golden_filtrado_per_window\golden_results_v2_1.csv"
IMG_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\resultados_gluge.png"
OUT_PATH = r"c:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\resultados_gluge_overlay.pdf"
OUT_PNG = r"c:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\resultados_gluge_overlay.png"

def get_accuracy_at_threshold(df, thresh=0.50):
    df_copy = df.copy()
    # 'pred_bin' was precalculated at a specific threshold (e.g. 0.75), we need to use 'max_prob' to recalculate for thresh=0.50
    df_copy['predicted'] = (df_copy['prob_max'] >= thresh).astype(int)
    # Target 4 is noise (label 0), others are drone (label 1)
    df_copy['correct'] = (df_copy['label'] == df_copy['predicted']).astype(int)
    
    snrs = sorted(df_copy['snr'].unique())
    accs = []
    for s in snrs:
        accs.append(df_copy[df_copy['snr'] == s]['correct'].mean())
    return snrs, accs

def main():
    df = pd.read_csv(CSV_PATH)
    
    # In Gluge et al., they use Balanced Accuracy, but we plot Accuracy (or we can calculate Balanced Accuracy just in case).
    # "Accuracy vs SNR" is what the user asked for. 
    # Let's use thresh=0.50 to compare fairly, since they don't use CFAR.
    snrs, accs = get_accuracy_at_threshold(df, 0.50)
    
    img = Image.open(IMG_PATH)
    width, height = img.size
    
    # The axes of Glüge plot go from X=-22 to X=32 roughly, Y=0.48 to Y=1.02.
    # We need to manually calibrate the extent.
    # Let's create a figure that matches the aspect ratio of the image.
    fig, ax = plt.subplots(figsize=(10, 10 * (height / width)))
    
    # To calibrate `extent`, we will plot it, maybe adjust slightly
    # Looking at a typical matplotlib plot saved as image:
    # X=-20 is at some pixel, X=30 is at another.
    # It's much easier to just plot our line without axes, over the image, but we need the exact extent.
    
    # We will assume extent=[xmin, xmax, ymin, ymax].
    # By visual inspection of Gluge's plot (typical matplotlib):
    # Left margin ~ 12% of width, Right margin ~ 90%
    # Bottom margin ~ 11% of height, Top margin ~ 92%
    # x-axis: -20 to +30.
    # y-axis: 0.5 to 1.0.
    
    # A better approach: plot the image using `imshow` with `extent` that maps the *pixels* to the *data coordinates*.
    # Actually, we map the data coordinates to pixels or map the image to data coordinates.
    # Let's map the image to the axes.
    # If x-axis left spine is at X=-22 and right spine is at X=32, 
    # and y-axis bottom spine is at Y=0.48 and top spine is at Y=1.02,
    # we can set extent=[-25, 35, 0.45, 1.05] approximately and tune it.
    
    # Let's write a script that generates a few test overlays with different extents so the user/we can pick the best.
    
    # Calibrated extent by trial and error typically:
    # the image covers X from -24.5 to 33.5, and Y from 0.44 to 1.03
    img_extent = [-25.8, 32.5, 0.44, 1.025] # Estimated.
    
    ax.imshow(img, extent=img_extent, aspect='auto')
    
    # Plot our data
    ax.plot(snrs, accs, color='red', linewidth=3, marker='D', markersize=6, label='DualStream-DynamicSlidingWindow (Propuesto)')
    
    ax.set_xlim(-20, 30)
    ax.set_ylim(0.5, 1.0)
    ax.axis('off') # Hide our axes, rely on the image's axes
    
    # Save a clean version without our own axes covering it
    plt.savefig(OUT_PNG, bbox_inches='tight', pad_inches=0, dpi=300)
    
    # Generate an HTML file to visually tune the alignment
    html_content = f"""
    <html>
    <body>
    <h2>Overlay Test</h2>
    <img src="resultados_gluge_overlay.png" style="max-width:800px; border: 1px solid black;" />
    </body>
    </html>
    """
    with open(r"c:\repos\DroneDetectionRF\NoisyUAV\figuras_TFM_reducidas\tune_overlay.html", "w") as f:
        f.write(html_content)
        
if __name__ == "__main__":
    main()
