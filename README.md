# ESG Data Sentinel Austria

Ein Machine-Learning-Projekt zur Erkennung potenzieller Datenqualitätsrisiken in österreichischen industriellen Emissionsdaten.

---

## 📌 Übersicht & Zielsetzung

Das Projekt identifiziert auffällige Datenpunkte (Red Flags) in öffentlich berichteten Emissionsdaten österreichischer Industrieanlagen. Es dient als Vorstufe zur ESG-Datenvalidierung, beispielsweise im Rahmen von *Limited Assurance Engagements*.

### Erkannte Anomalietypen
* **Numerische Fehler:** Negative Werte oder extreme Ausreißer.
* **Metrische Fehler:** Mögliche Skalierungs- und Einheitenfehler.
* **Temporale Fehler:** Extreme Year-over-Year-Veränderungen (Vorjahresvergleich).
* **Kontextuelle Fehler:** Sektor- oder schadstoffspezifische Abweichungen.

---

## 🤖 Machine Learning Ansatz

Da verlässliche Fehlerlabels fehlen, nutzt das Projekt ein **unüberwachtes Verfahren (Unsupervised Learning)**.

* **Algorithmus:** `IsolationForest`
* **Vorteil:** Benötigt keine gelabelten Trainingsdaten; isoliert Anomalien basierend auf numerischen Mustern.
* **Klassifikation:**
  * `1` (Normal) ➔ Unauffälliger Datenpunkt.
  * `-1` (Anomalie) ➔ Potenziell prüfungswürdiger Datenpunkt / Red Flag.

> 💡 **Methodischer Hinweis:** Fehlende Werte (*Missing Values*) werden vorab deterministisch via Pandas in der Transformationsphase bereinigt. Das ML-Modell fokussiert sich rein auf komplexe, nicht-regelbasierte Musterabweichungen.

---

## 📊 Datenbasis & Scope

Die Daten stammen aus dem **European Industrial Emissions Portal** mit folgendem Fokus:

* **Geografie:** Österreich (AT)
* **Datentyp:** Industrielle Emissionsdaten (tabellarisch)
* **Scope:** Fokus auf Datenintegrität und Qualitätssicherung

---

## 📂 Projektstruktur

```text
esg-data-sentinel-austria/
├── data/
│   ├── raw/                 # Unveränderte Originaldaten
│   ├── interim/             # Bereinigte Zwischenstände
│   └── processed/           # Modellbereite Feature-Matrizen
├── notebooks/               # Explorative Analysen (EDA)
├── outputs/
│   ├── figures/             # Evaluierungs-Plots
│   └── models/              # Serialisierte IsolationForest-Modelle
├── src/
│   ├── extract.py           # Daten-Ingestion
│   ├── transform.py         # Vorverarbeitung & Missing-Value-Checks
│   ├── inject_errors.py     # Synthetische Fehlereinspeisung zu Testzwecken
│   ├── train_model.py       # IsolationForest Training
│   ├── evaluate_model.py    # Performance-Metrizen
│   └── visualization.py     # Plotting-Funktionen
├── app.py                   # UI / Dashboard (optional)
├── run_pipeline.py          # Zentrales Orchestrierungs-Skript
├── requirements.txt         # Python-Abhängigkeiten
└── README.md                # Dokumentation
```

---

## ⚡ Installation & Ausführung

### 1. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 2. Pipeline starten
```bash
python run_pipeline.py
```