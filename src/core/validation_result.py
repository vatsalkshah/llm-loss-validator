"""
Structured result objects for validation workflow.

This module defines data classes to return validation results,
allowing downstream callers to inspect outcomes without side effects.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ValidationResult:
    """
    Represents the outcome of a validation run.

    Attributes:
        success: Whether validation completed successfully
        eval_loss: Raw evaluation loss in nats (NaN if not computed)
        bpc: Bits Per Character metric (inf if invalid)
        bppl: bits Per Character Perplexity (inf if invalid)
        eval_loss_to_submit: The loss value to submit to the API
        total_bytes: Total bytes in evaluation dataset
        total_target_tokens: Total target tokens in evaluation dataset
        token_byte_ratio: Token/Byte ratio
        vocab_size: Tokenizer vocabulary size
        model_params_m: Model parameter count in millions
        assignment_id: Assignment ID for this validation
        error_message: Error message if validation failed
        metadata: Additional metadata
    """

    success: bool
    eval_loss: float = float("nan")
    bpc: float = float("inf")
    bppl: float = float("inf")
    eval_loss_to_submit: float = float("inf")
    total_bytes: int = 0
    total_target_tokens: int = 0
    token_byte_ratio: float = float("inf")
    vocab_size: int = 0
    model_params_m: float = float("nan")
    assignment_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_params_exceeded(self) -> bool:
        """Check if this result represents a parameter limit failure."""
        return not self.success and "exceed" in (self.error_message or "").lower()

    def is_bpc_valid(self) -> bool:
        """Check if BPC metric is valid (not inf)."""
        import math
        return not math.isinf(self.bpc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary."""
        return {
            "success": self.success,
            "eval_loss": self.eval_loss,
            "bpc": self.bpc,
            "bppl": self.bppl,
            "eval_loss_to_submit": self.eval_loss_to_submit,
            "total_bytes": self.total_bytes,
            "total_target_tokens": self.total_target_tokens,
            "token_byte_ratio": self.token_byte_ratio,
            "vocab_size": self.vocab_size,
            "model_params_m": self.model_params_m,
            "assignment_id": self.assignment_id,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }
