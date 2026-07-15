"""
extract.py
Kapselt die Rohdatenbeschaffung über Generatoren in spezialisierten Komponenten-Klassen.
Optimiert für hohe Performance bei großen Datensätzen.
"""

from pathlib import Path
from typing import Generator
import pandas as pd
from .errors import InvalidFileFormat, DataValidationError

class DataLoader:
    """Komponente zum ressourcenschonenden Streamen von Rohdaten"""
    def __init__(self, file_path: Path, chunk_size: int = 50000):
        self.file_path = file_path
        self.chunk_size = chunk_size

    def read_csv_in_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """Interner Lazy-Evaluation Generator, der eine CSV-Datei zeilenweise als Chunks einliest."""

        # Überprüfung des Dateiformats vor dem Streaming-Prozess
        if self.file_path.suffix.lower() != '.csv':
            raise InvalidFileFormat(f"Ungültiges Format: {self.file_path.suffix}. Nur CSV erlaubt.")
    
        # chunksize aktiviert das Streamen der Datei über einen TextFileReader
        try:
            for chunk in pd.read_csv(self.file_path, chunksize=self.chunk_size, low_memory=False):
                yield chunk
        except Exception as e:
            raise InvalidFileFormat(f"Kritischer Fehler beim Streaming: {str(e)}")
    
    def extract_data(self) -> pd.DataFrame:
        """
        Sammelt die Chunks des Generators und baut das initiale DataFrame auf.
        Fehlerhafte Chunks werden protokolliert und übersprungen.
        """
        chunks= []
        for chunk in self.read_csv_in_chunks():
            if chunk.empty:
                raise DataValidationError("Leerer Zeilenblock erkannt.")
            try:
                chunks.append(chunk)      
            except DataValidationError as e:
                continue
        if not chunks:
            raise DataValidationError("Extraktion fehlgeschlagen: Keine Daten extrahiert.")

        df_raw = pd.concat(chunks, ignore_index=True)
        return df_raw
    
    def __repr__(self) -> str:
        return f"Class DataLoader (Target='{self.file_path.name}', Chunk Size={self.chunk_size})"