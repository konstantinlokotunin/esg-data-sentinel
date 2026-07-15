"""
visualization.py
Kapselt die Generierung von Kontrollplots für die Bonus-Anforderungen.
Trennt die reine Plot-Erstellung strikt von den Datei-Schreiboperationen (I/O).
"""

from pathlib import Path
import matplotlib.figure as figure
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class DashboardRenderer():
    """Komponente zur hochdimensionalen Feature-Generierung"""
    def __init__(self, df: pd.DataFrame):
        self.df_results = df

    def create_individual_fig(self, figsize: tuple = (14, 7)):
        """
        Erstellt ein standardisiertes Matplotlib Figure- und Axis-Objekt.
        """

        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    def style_ax(self, ax, title: str, xlabel, ylabel, show_grid: bool = False) -> None:
        """
        Styling-Schnittstelle für ein konsistentes, professionelles Layout.
        """

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

    def plot_anomaly_score_distribution(self) -> figure.Figure:
        """
        Erstellt den Plot für die Verteilung der Anomaly Scores.
        Niedrigere Werte für den Anomaly Score deuten auf ungewöhnlichere Datensätze hin.
        """

        fig, ax = self.create_individual_fig(figsize=(14, 7))
        self.style_ax(ax, title="Distribution of Anomaly Scores", xlabel="Anomaly Score", ylabel="Number of Records")

        sns.histplot(
            data=self.df_results,
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
        return fig

    def plot_anomalies_by_sector(self) -> figure.Figure:
        """
        Erstellt das Balkendiagramm der Anomalien nach Sektor.
        """

        sector_counts = (
            self.df_results[self.df_results["Is_Anomaly"] == 1]
            .groupby("Sector")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="Number of Anomalies")
        )

        fig, ax = self.create_individual_fig(figsize=(14, 7))

        sns.barplot(
            data=sector_counts,
            x="Sector",
            y="Number of Anomalies",
            color="#1a73e8",
            alpha=0.85,
            ax=ax
        )
        self.style_ax(ax, title="Sectors by Anomaly Count", xlabel=None, ylabel="Number of Anomalies")
        ax.tick_params(axis='x', rotation=20)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.05)

        total_anomalies = len(self.df_results[self.df_results["Is_Anomaly"] == 1])
        for bar in ax.patches:
            anomaly_count = int(bar.get_height())
            percentage = (anomaly_count / total_anomalies) * 100 if total_anomalies > 0 else 0
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
        return fig

    def plot_anomalies_by_pollutant_group(self) -> figure.Figure:
        """
        Erstellt das Balkendiagramm nach Schadstoffgruppe.
        """

        group_counts = (
            self.df_results[self.df_results["Is_Anomaly"] == 1]
            .groupby("Pollutant_Group")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="Number of Anomalies")
        )

        fig, ax = self.create_individual_fig(figsize=(14, 7))

        sns.barplot(
            data=group_counts,
            x="Pollutant_Group",
            y="Number of Anomalies",
            color="#1a73e8",
            alpha=0.85,
            ax=ax
        )
        self.style_ax(ax, title="Anomalies by Pollutant Group", xlabel=None, ylabel="Number of Anomalies")
        ax.tick_params(axis='x', rotation=20)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.05)

        total_anomalies = len(self.df_results[self.df_results["Is_Anomaly"] == 1])
        for bar in ax.patches:
            anomaly_count = int(bar.get_height())
            percentage = (anomaly_count / total_anomalies) * 100 if total_anomalies > 0 else 0
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
        return fig

    def plot_anomalies_by_sector_and_pollutant_group(self) -> figure.Figure:
        """
        Erstellt eine Heatmap für Sektoren und Schadstoffgruppen 
        """

        anomalies = self.df_results[self.df_results["Is_Anomaly"] == 1].copy()
        anomalies["Short_Label"] = anomalies["Pollutant_Group"].str.split("-").str[0]

        heatmap_data = pd.crosstab(
            anomalies["Sector"],
            anomalies["Short_Label"]
        )

        fig, ax = self.create_individual_fig(figsize=(14, 7))

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
        self.style_ax(ax, title="Anomalies by Sector and Pollutant Group", xlabel=None, ylabel=None)
        ax.tick_params(axis='both', which='major', labelsize=11, length=0)
        ax.tick_params(axis='x', rotation=20)
        return fig

    def generate_and_save_plots(self, output_dir: Path) -> None:
        """
        Zentrale I/O-Steuerungsfunktion für Grafiken. 
        Wird von main.py aufgerufen. Verhindert Speicherlecks (Memory Leaks).
        """
        # Einmalig das globale Theme für diesen Run setzen
        sns.set_theme(style="white", font="sans-serif")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Zuordnungen von Dateinamen zu reinen Plot-Funktionen
        plot_jobs = {
            "anomaly_score_distribution.png": self.plot_anomaly_score_distribution,
            "sectors_by_anomaly_count.png": self.plot_anomalies_by_sector,
            "anomalies_by_pollutant_group.png": self.plot_anomalies_by_pollutant_group,
            "anomalies_by_sector_and_pollutant_group.png": self.plot_anomalies_by_sector_and_pollutant_group
        }

        # 2. Schleife führt Berechnungen aus und isoliert den I/O-Schreibprozess
        for filename, plot in plot_jobs.items():
            fig = plot()
            target_path = output_dir / filename
            fig.savefig(target_path, bbox_inches="tight", dpi=300)
        
            # Wichtig: Schließen des Plots verhindert, dass das Notebook im RAM abstürzt
            plt.close(fig)

    def __repr__(self) -> str:
        return f"DashboardRenderer(DataShape={self.df_results.shape})"