"""
DataSentry Detectors - Base Plugin Interface

Defines the pluggable BaseDetector abstract class ensuring uniform
lifecycle, modality tagging, and report outputs across modalities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from datasentry.schemas import AxisType


class BaseDetector(ABC):
    """Abstract base class for all DataSentry pluggable detectors."""

    name: str = "base_detector"
    modality: AxisType = AxisType.QUALITY

    def __init__(self, **kwargs: Any):
        self.config: Dict[str, Any] = kwargs

    @abstractmethod
    def detect(
        self,
        current_data: Any,
        reference_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Executes diagnostic detection on input payload.
        Returns a validated Pydantic report model.
        """
        raise NotImplementedError
