"""
app.py
Interaktives Streamlit-Dashboard für den ESG Data Sentinel.
Ermöglicht den Drag-and-Drop-Upload von ESG-Portfoliodaten.
Analyse erfolgt komplett im Hintergrund ohne Code-Sichtbarkeit für den Anwender.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from src.pipeline.cleaning import DataCleaner
from src.ml.transform import DataTransformer
from src.ml.train_model import ModelPipeline
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration für ein professionelles Erscheinungsbild
st.set_page_config(
    page_title="ESG Data Sentinel — Audit Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konsistentes Farb-Theme definieren (Deep Navy, ESG Green, Red-Flag Orange)
COLOR_NAVY = "#0B1B2B"
COLOR_GREEN = "#00A86B"
COLOR_RED = "#FF5A36"

# CSS-Injektion für professionelles Styling
st.markdown(f"""
    <style>
    .main-title {{ color: {COLOR_NAVY}; font-size: 38px; font-weight: bold; margin-bottom: 5px; }}
    .sub-title {{ color: "#5a6b7c"; font-size: 16px; margin-bottom: 25px; }}
    .metric-card {{ background-color: #F7FBFF; border: 1px solid #D2D3DB; border-radius: 8px; padding: 20px; text-align: center; }}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR (Metadaten und Scope) ---
with st.sidebar:
    st.markdown("### 🛡️ Systemkonfiguration")
    st.info("Prüfmodul: Vorprüfungshandlungen im Risk Assessment (NaBeG / CSRD Portfolio-Schnittstelle)")
    st.markdown("---")
    st.markdown("**User:** ...")
    st.markdown("**Verband:** ...")
    st.markdown("**Scope:** Scope 3 — Financed Emissions Validierung")

# --- HAUPTSEITE (Titelzeile) ---
st.markdown("<div class='main-title'>ESG Data Sentinel Austria</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Automatisierte Validierung und analytische Prüfungshandlungen für Nachhaltigkeitsdaten</div>", unsafe_allow_html=True)

st.markdown("---")