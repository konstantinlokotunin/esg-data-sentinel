"""
visualization.py
Kapselt die Generierung von Kontrollplots für die Bonus-Anforderungen.
Trennt die reine Plot-Erstellung strikt von den Datei-Schreiboperationen (I/O).
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from pathlib import Path

file_path = r"C:\Users\konst\Documents\Python Projects\ESG Data Sentinel\esg-data-sentinel\outputs\figures"

OUTPUT_DIR = Path(file_path)

def create_individual_fig(figsize=(14, 7)):
    """
    Creates a standard figure and axis.
    """

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

# Globale Definition der zentralen Styling-Schnittstelle für ein einheitliches Design
def style_ax(ax, title, xlabel=None, ylabel=None, show_grid=False):
    sns.set_theme(style="white", font="sans-serif")
    sns.despine(ax=ax, left=False, bottom=False)
    for spine in ["left", "bottom", "right", "top"]:
        ax.spines[spine].set_linewidth(1.1)
        ax.spines[spine].set_color("#9ca3af")
    ax.set_facecolor("#F7FBFF") # Konsistenter, leicht bläulicher Hintergrund
    if show_grid:
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#ccc")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13, labelpad=8, color="#222222")
    else:
        ax.set_xlabel(None)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, labelpad=8, color="#222222")
    else:
        ax.set_ylabel(None)
    ax.set_title(title, fontsize=15, weight="bold", pad=15, color="#222222")

def plot_anomaly_score_distribution(df):
    """
    Plots the distribution of anomaly scores.
    Lower Anomaly_Score values mean more unusual records.
    """

    fig, ax = create_individual_fig(figsize=(14, 7))
    style_ax(ax, title="Distribution of Anomaly Scores", xlabel="Anomaly Score", ylabel="Number of Records")

    sns.histplot(
        data=df,
        x="Anomaly_Score",
        bins=40,
        color="#1a73e8",
        edgecolor="white",
        alpha=0.85,
        ax=ax
    )

    ax.axvline(
        x=0,
        color="#DB3A34",
        linestyle="--",
        linewidth=2,
        label="Decision threshold"
    )
    ax.legend(loc="best", frameon=False, fontsize=11)

    output_path = OUTPUT_DIR / "anomaly_score_distribution.png"
    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300
    )

    print("\nSaved figure 1:", output_path)
    return fig


def plot_anomalies_by_sector(df):
    """
    Plots the top sectors by number of detected anomalies.
    """

    sector_counts = (
        df[df["Is_Anomaly"] == 1]
        .groupby("Sector")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="Number of Anomalies")
    )

    fig, ax = create_individual_fig(figsize=(14, 7))

    sns.barplot(
        data=sector_counts,
        x="Sector",
        y="Number of Anomalies",
        color="#1a73e8",
        alpha=0.85,
        ax=ax
    )
    style_ax(ax, title="Sectors by Anomaly Count", ylabel="Number of Anomalies")
    ax.tick_params(axis='x', rotation=20)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)

    total_anomalies = len(df[df["Is_Anomaly"] == 1])
    for bar in ax.patches:
        anomaly_count = int(bar.get_height())
        percentage = (anomaly_count / total_anomalies) * 100
        label_text = f"{anomaly_count}\n({percentage:.1f}%)"
    
        ax.text(
            x=bar.get_x() + bar.get_width() / 2,
            y=anomaly_count + 0.25,
            s=label_text,
            ha="center",
            va="bottom",
            fontsize=10,
            weight="semibold",
            color="#222222"
    )

    output_path = OUTPUT_DIR / "sectors_by_anomaly_count.png"
    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300
    )

    print("Saved figure 2:", output_path)
    return fig

def plot_anomalies_by_pollutant_group(df):
    """
    Plots the top pollutant groups by number of detected anomalies.
    """

    group_counts = (
        df[df["Is_Anomaly"] == 1]
        .groupby("Pollutant_Group")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="Number of Anomalies")
    )

    fig, ax = create_individual_fig(figsize=(14, 7))

    sns.barplot(
        data=group_counts,
        x="Pollutant_Group",
        y="Number of Anomalies",
        color="#1a73e8",
        alpha=0.85,
        ax=ax
    )
    style_ax(ax, title="Anomalies by Pollutant Group", ylabel="Number of Anomalies")
    ax.tick_params(axis='x', rotation=20)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)

    total_anomalies = len(df[df["Is_Anomaly"] == 1])
    for bar in ax.patches:
        anomaly_count = int(bar.get_height())
        percentage = (anomaly_count / total_anomalies) * 100
        label_text = f"{anomaly_count}\n({percentage:.1f}%)"
    
        ax.text(
            x=bar.get_x() + bar.get_width() / 2,
            y=anomaly_count + 0.25,
            s=label_text,
            ha="center",
            va="bottom",
            fontsize=10,
            weight="semibold",
            color="#222222"
    )

    output_path = OUTPUT_DIR / "anomalies_by_pollutant_group.png"
    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300
    )

    print("Saved figure 3:", output_path)
    return fig

def plot_anomalies_by_sector_and_pollutant_group(df):

    anomalies = df[df["Is_Anomaly"] == 1].copy()
    anomalies["Short_Label"] = anomalies["Pollutant_Group"].str.split("-").str[0]

    heatmap_data = pd.crosstab(
        anomalies["Sector"],
        anomalies["Short_Label"]
    )

    fig, ax = create_individual_fig(figsize=(14, 7))

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=1.2,
        linecolor="#D2D3DB",
        cbar=True,
        square=True,
        annot_kws={"size": 12, "weight": "semibold"},
        ax=ax
    )
    style_ax(ax, title="Anomalies by Sector and Pollutant Group")
    ax.tick_params(axis='both', which='major', labelsize=11, length=0)
    ax.tick_params(axis='x', rotation=20)

    output_path = OUTPUT_DIR / "anomalies_by_sector_and_pollutant_group.png"
    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300
    )

    print("Saved figure 4:", output_path)
    return fig