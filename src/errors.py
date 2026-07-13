class InvalidFileFormat(Exception):
    """
    Wird ausgelöst, wenn das Dateiformat nicht unterstützt wird (z. B. kein CSV).
    """
    pass

class DataValidationError(Exception):
    """
    Wird ausgelöst, wenn eine einzelne Zeile kritische Validierungsfehler aufweist.
    """
    pass

class EmptyDatasetError(Exception):
    """
    Wird ausgelöst, wenn das Dataset nach der Bereinigung leer ist.
    """
    pass