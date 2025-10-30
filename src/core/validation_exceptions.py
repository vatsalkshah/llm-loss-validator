"""
Structured exceptions for validation workflow.

This module defines a hierarchy of exceptions that allow callers
to differentiate between parameter-limit failures, model loading issues,
transient errors, and other validation problems.
"""


class ValidationException(Exception):
    """Base exception for all validation-related errors."""

    def __init__(self, message: str, assignment_id: str = None):
        self.message = message
        self.assignment_id = assignment_id
        super().__init__(message)


class ModelParamsExceededException(ValidationException):
    """
    Raised when the model's parameter count exceeds the allowed limit.
    This is a validation failure condition that should result in a high loss submission.
    """

    def __init__(self, actual_params: int, max_params: int, assignment_id: str = None):
        self.actual_params = actual_params
        self.max_params = max_params
        message = (
            f"Model params ({actual_params}) exceed limit ({max_params})"
        )
        super().__init__(message, assignment_id)


class InvalidModelException(ValidationException):
    """
    Raised when the model cannot be loaded or is invalid.
    Examples: repo not accessible, lora_only flag mismatch, missing base model.
    """

    pass


class InvalidDatasetException(ValidationException):
    """
    Raised when the dataset cannot be loaded or is invalid.
    Examples: file not found, empty dataset, parsing errors.
    """

    pass


class InvalidLoraConfigException(ValidationException):
    """
    Raised when LoRA adapter configuration is invalid or incompatible.
    Examples: missing base_model_name_or_path, unsupported base model.
    """

    pass


class TransientValidationException(ValidationException):
    """
    Raised for transient errors that may be retryable.
    Examples: temporary network issues, temporary resource unavailability.
    """

    pass
