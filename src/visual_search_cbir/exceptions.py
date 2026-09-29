class VisualSearchError(Exception):
    """Base exception for the visual-search application."""


class DatasetError(VisualSearchError):
    """Raised when the dataset layout is invalid or contains no supported images."""


class ArtifactError(VisualSearchError):
    """Raised when an index artifact cannot be saved or loaded."""


class FeatureExtractionError(VisualSearchError):
    """Raised when image features cannot be extracted."""
