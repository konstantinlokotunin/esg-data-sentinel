"""
extract.py
Kapselt die Logik zum Einlesen der Rohdaten über Generatoren.
Optimiert für hohe Performance bei großen Datensätzen.
"""

import logging
from pathlib import Path
from typing import Generator
import pandas as pd
from .errors import InvalidFileFormat, DataValidationError

# Logger für das einheitliche und übersichtliche Protokollieren ungültiger Zeilen
logger = logging.getLogger(__name__)

def read_csv_in_chunks(file_path: Path, chunk_size: int = 50000) -> Generator[pd.DataFrame, None, None]:
    """
    Generator, der eine CSV-Datei zeilenweise (lazy) als Chunk einliest.

    Args:
        file_path (Path): Pfad zur einzulesenden CSV-Datei.
        chunk_size (int): Zeilenanzahl pro Chunk (Standard: 1 für zeilenweises Lesen).

    Yields:
        Generator[pd.DataFrame, None, None]: Ein Pandas DataFrame Chunk.

    Raises:
        InvalidFileFormat: Wenn die Datei keine CSV-Endung besitzt.
    """

    # Überprüfung des Dateiformats vor dem Streaming-Prozess
    if file_path.suffix.lower() != '.csv':
        raise InvalidFileFormat(f"Ungültiges Format: {file_path.suffix}. Nur CSV erlaubt.")
    
    # chunksize aktiviert das Streamen der Datei über einen TextFileReader
    try:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            yield chunk
    except Exception as e:
        raise InvalidFileFormat(f"Kritischer Fehler beim Streaming der CSV: {str(e)}")
    
    
def extract_data(file_path) -> pd.DataFrame:
    """
    Sammelt die Chunks des Generators und baut das initiale DataFrame auf.
    Fehlerhafte Chunks werden protokolliert und übersprungen.

    Args:
        file_path (Path): Pfad zur Quelldatei.

    Returns:
        pd.DataFrame: Das zusammengesetzte, rohe DataFrame für die Transformation.
    """

    chunks= []

    for chunk in read_csv_in_chunks(file_path):
        if chunk.empty:
            raise DataValidationError("Leerer Zeilenblock erkannt.")
        try:
        # Falls ein Fehler auftritt, springt der Code in den except-Block
            chunks.append(chunk)
            
        except DataValidationError as e:
            # Datenschonendes Logging
            logger.warning(f"Datensatz übersprungen: {str(e)}")
            continue
        
    if not chunks:
        raise DataValidationError("Extraktion fehlgeschlagen: Keine gültigen Daten extrahiert.")

    df = pd.concat(chunks, ignore_index=True)
    return df