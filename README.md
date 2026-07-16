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

Das Projekt ist strikt modular nach Schichten und Paketen organisiert, um Zirkelbezüge zu vermeiden:

```text
esg-data-sentinel/
│
├── data/                            # Reiner Datenordner
│   └── raw/                         # Unveränderte Original-CSV-Datei
│
├── outputs/                         # Speicherort für generierte Artefakte
│   ├── figures/                     # Generierte Analyse-Plots
│   └── anomaly_reports.xlsx         # Konsolidierter Multi-Sheet Excel-Bericht
│
├── models/                          # Speicherort für serialisierte ML-Objekte
│   ├── model.joblib                 # Das gespeicherte, trainierte ML-Modell
│   └── scaler.joblib                # Der gespeicherte Daten-Scaler
│
├── src/                             # Haupt-Code-Ordner
│   │
│   ├── pipeline/                    # PAKET 1: Datenbeschaffung & Bereinigung
│   │   ├── __init__.py              # Paket-Initialisierung
│   │   ├── errors.py                # Enthält: InvalidFileFormat, DataValidationError, EmptyDatasetError
│   │   ├── extract.py               # Lazy-Evaluation Chunk-Generator
│   │   └── cleaning.py              # Bereinigung & Validierung
│   │
│   └── ml/                          # PAKET 2: Daten-Analyse & Modellierung
│      ├── __init__.py               # Paket-Initialisierung
│      ├── transform.py              # Schadstoff-Klassifizierung & Feature Engineering
│      ├── train_model.py            # Kernfunktion für das Modell-Fitting
│      ├── evaluate_model.py         # Metriken-Aggregation & Excel-I/O
│      └── visualization.py          # Reine Plotting-Funktionen
│
├── tests/
│   └── test_transform.py          # Automatisierte Unittests
├── main.py                        # Haupt-Pipeline: ETL bis Excel-Report
├── train.py                       # Separates Skript: Modell trainieren & speichern
├── predict.py                     # Separates Skript: Modell laden & anwenden
├── requirements.txt               # Paket-Abhängigkeiten
└── README.md                      # Projektdokumentation
```

---

## ⚡ Installation & Ausführung

### 1. Abhängigkeiten installieren
Stellen Sie sicher, dass Ihre virtuelle Umgebung (`venv`) aktiv ist.
```bash
pip install -r requirements.txt
```

### 2. ETL-Pipeline & Reporting starten
Generiert den Excel-Report im Ordner `outputs/` und die Kontrollgrafiken.
```bash
python main.py
```

### 3. Machine-Learning-Modell trainieren und speichern
Teilt die Daten, trainiert den Isolation Forest und exportiert das fertige Modell nach `models/`.
```bash
python train.py
```

### 4. Vorhersagen auf neuen Daten berechnen (Inferenz)
Lädt das fertige Modell aus der Datei und berechnet Vorhersagen für neue Datensätze, ohne neu zu trainieren.
```bash
python predict.py
```