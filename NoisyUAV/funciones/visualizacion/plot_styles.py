import matplotlib.pyplot as plt

def apply_ieee_style():
    """
    Sets the matplotlib style for IEEE/Nature publication quality.
    Features:
    - White background
    - Crisp vector-like line rendering
    - Large and legible sans-serif font (Arial, DejaVu Sans...)
    - Colorblind friendly distinct palette
    - Light gray grid for professional look
    """
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.color': '#DDDDDD',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'grid.alpha': 1.0,
        
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#CCCCCC',
        
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'Nimbus Roman', 'Georgia', 'serif'],
        'text.color': '#222222',
        'axes.labelcolor': '#222222',
        'xtick.color': '#222222',
        'ytick.color': '#222222',
        
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'png'
    })

def get_color_palette():
    """Returns a highly legible, colorblind-friendly palette."""
    return {
        'blue':   '#0072B2',  # Distinct blue
        'orange': '#D55E00',  # Distinct orange (vermilion)
        'green':  '#009E73',  # Sea green
        'yellow': '#F0E442',  # Bright yellow
        'purple': '#CC79A7',  # Red-purple
        'cyan':   '#56B4E9',  # Sky blue
        'gray':   '#7F7F7F',
        'red':    '#E6194B' 
    }

def clean_spines(ax):
    """Removes top and right spines strictly for clean plotting."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
