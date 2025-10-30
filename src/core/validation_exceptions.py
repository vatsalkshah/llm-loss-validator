"""Domain-specific exceptions used by the validation runner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class ValidationException(Exception):
    """Base class for all validation related failures.

    Parameters
    ----------
    message:
        Human readable description.
    assignment_id:
        Optional assignment identifier used for logging and telemetry.
    """

    def __init__(self, message: str, assignment_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.assignment_id = assignment_id


@dataclass(slots=True)
class ValidationFailureContext:
    """Rich context passed alongside exceptions for telemetry/debugging."""

    assignment_id: Optional[str]
    model_name: Optional[str]
    task_id: Optional[str]


class ModelParamsExceededException(ValidationException):
    """Raised when the submitted model exceeds the advertised parameter cap."""

    def __init__(self, actual_params: int, max_params: int, assignment_id: Optional[str] = None) -> None:
        message = (
            "Model parameter count exceeded the permitted limit: "
            f"actual={actual_params}, limit={max_params}"
        )
        super().__init__(message, assignment_id=assignment_id)
        self.actual_params = actual_params
        self.max_params = max_params


class InvalidModelException(ValidationException):
    """Raised when a repository cannot be loaded or fails sanity checks."""


class InvalidDatasetException(ValidationException):
    """Raised when the evaluation dataset is malformed or inaccessible."""


class InvalidLoraConfigException(ValidationException):
    """Raised when LoRA adapters are missing required metadata."""


class TransientValidationException(ValidationException):
    """Raised for transient, retryable problems (network hiccups, rate limits)."""
