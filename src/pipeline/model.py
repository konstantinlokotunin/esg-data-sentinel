"""
model.py
Definiert das objektorientierte Datenmodell für die Pipeline.
"""

class EmissionRecord:
    """Repräsentiert einen einzelnen, bereinigten Luftschadstoffdatensatz."""
    
    def __init__(self, year: int, sector: str, facility: str, pollutant: str, amount: float):
        self.year = int(year)
        self.sector = sector
        self.facility = facility
        self.pollutant = pollutant
        self.amount = float(amount)
