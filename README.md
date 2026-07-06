# ESG Data Sentinel Austria

Ein Machine-Learning-Projekt zur Erkennung potenzieller Datenqualitätsrisiken in österreichischen industriellen Emissionsdaten.

Das Projekt ist als einfache modulare Datenpipeline aufgebaut:

Datenimport → Transformation → Feature Engineering → Anomaly Detection → Evaluation & Interpretation

---

## Projektidee

Ziel des Projekts ist es, potenzielle Datenqualitätsrisiken in berichteten Emissionsdaten österreichischer Industrieanlagen zu erkennen.

Dafür werden reale österreichische Emissionsdaten verwendet und ein **unsupervised Anomaly-Detection-Modell mit IsolationForest** trainiert.

Das Modell soll auffällige Datenpunkte identifizieren, zum Beispiel:

- negative Emissionswerte
- ungewöhnlich große oder kleine Werte
- mögliche Skalierungs- bzw. Einheitenfehler
- extreme Year-over-Year-Veränderungen
- sektor- oder schadstoffspezifische Ausreißer

Das Modell klassifiziert die Daten nicht auf Basis echter Fehlerlabels, sondern markiert ungewöhnliche Datenpunkte als potenzielle Red Flags.

---

## Machine-Learning-Ansatz

Da öffentlich verfügbare Energie- und Emissionsdatensätze jedoch in der Regel keine verlässlichen Labels für fehlerhafte Datenpunkte enthalten, wird ein **unüberwachtes Anomalieerkennungsverfahren** verwendet.

Der gewählte Algorithmus ist:

**IsolationForest**

IsolationForest eignet sich für dieses Projekt, weil das Modell keine gelabelten Trainingsdaten benötigt. Es lernt aus den vorhandenen numerischen Datenmustern, welche Beobachtungen ungewöhnlich erscheinen.

Die Ergebnisse werden als Review-Hinweise interpretiert:

- `normal = unauffälliger Datenpunkt`
- `anomaly = potenziell prüfungswürdiger Datenpunkt / Red Fla`

Wichtig: Das Modell erkennt keine rechtlich oder fachlich endgültig bestätigten Fehler. Es identifiziert Datenpunkte, die im Rahmen einer ESG-Datenvalidierung z.B. im Rahmen von Limited Assurance Engagements näher geprüft werden sollten.

---

## Datenquelle

Das Projekt verwendet industrielle Emissionsdaten aus dem European Industrial Emissions Portal.

Der Projektumfang ist bewusst eingeschränkt auf:

- Land: Österreich
- Datenart: industrielle Emissionsdaten
- Datenformat: tabellarische Daten
- Analysefokus: potenzielle Datenqualitätsrisiken
- ML-Aufgabe: unsupervised anomaly detection

---

## Methodischer Hinweis

Fehlende Werte werden nicht mit dem Machine-Learning-Modell erkannt. Sie werden bereits im Transformationsschritt mit Pandas geprüft, zum Beispiel über Missing-Value-Checks.

Das IsolationForest-Modell wird stattdessen für Auffälligkeiten verwendet, die nicht immer durch einfache Regeln erkennbar sind, zum Beispiel:

- ungewöhnlich hohe oder niedrige Emissionswerte
- mögliche Skalierungsfehler
- extreme Veränderungen gegenüber dem Vorjahr
- Ausreißer innerhalb bestimmter Sektoren oder Schadstoffgruppen

---

## Projektstruktur

```text
esg-data-sentinel-austria/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── src/
│   ├── extract.py
│   ├── transform.py
│   ├── inject_errors.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── visualization.py
│
├── app.py
├── run_pipeline.py
├── requirements.txt
└── README.md

---

## Installation

```bash
pip install -r requirements.txt

---

## Ausführung

```bash
python run_pipeline.py