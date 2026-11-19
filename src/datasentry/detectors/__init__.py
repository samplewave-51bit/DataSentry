"""
DataSentry Detectors Package - Pluggable Modality Detectors
"""

from datasentry.detectors.base import BaseDetector
from datasentry.detectors.text import TextDriftDetector, ingest_document_with_docling
from datasentry.detectors.timeseries import TimeSeriesDetector

__all__ = [
    "BaseDetector",
    "TimeSeriesDetector",
    "TextDriftDetector",
    "ingest_document_with_docling",
]
