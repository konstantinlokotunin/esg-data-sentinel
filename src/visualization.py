from matplotlib import pyplot as plt
import seaborn as sns

# Globale Definition der zentralen Styling-Schnittstelle für ein einheitliches Design
def style_ax(ax, title, xlabel, ylabel, show_grid=False):
    sns.despine(ax=ax, left=False, bottom=False)
    ax.set_facecolor("#F7FBFF") # Konsistenter, leicht bläulicher Hintergrund
    if show_grid:
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#ccc")
    ax.set_title(title, fontsize=14, weight="bold", pad=15, color="#222222")
    ax.set_xlabel(xlabel, fontsize=12, labelpad=8, color="#222222")
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8, color="#222222")

# Globales Seaborn-Theme initialisieren, um Darstellungsfehler zu vermeiden
sns.set_theme(style="whitegrid", font="sans-serif")

# --- PLOT 1: Klassenverteilung (Globale Class Imbalance) ---
fig1, ax1 = plt.subplots(figsize=(6, 5))