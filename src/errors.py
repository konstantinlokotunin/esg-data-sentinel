class InvalidFileFormat(Exception):
    """
    Raised when the input file is not a supported CSV or Excel file.
    """
    pass

class EmptyDatasetError(Exception):
    """
    Raised when the dataset becomes empty after cleaning or filtering.
    """
    pass