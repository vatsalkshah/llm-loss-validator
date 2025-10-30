"""Utilities for representing validation outcomes in a structured way."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(slots=True)
class ValidationResult:
    """Container describing the outcome of a validation assignment."""

    success: bool
    eval_loss: float = float("nan")
    eval_loss_to_submit: float = float("inf")
    bpc: float = float("inf")
    bppl: float = float("inf")
    total_bytes: int = 0
    total_target_tokens: int = 0
    token_byte_ratio: float = float("inf")
    vocab_size: int = 0
    model_params_m: float = float("nan")
    assignment_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_params_exceeded(self) -> bool:
        """Return True when the failure is due to parameter limits."""

        if not self.error_message:
            return False
        return "exceed" in self.error_message.lower()

    def is_loss_valid(self) -> bool:
        """Check that the evaluation loss is a finite floating point value."""

        try:
            return float(self.eval_loss) == self.eval_loss and self.eval_loss not in {float("inf"), float("-inf")}
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the result to a primitive dictionary for logging/telemetry."""

        return {
            "success": self.success,
            "eval_loss": self.eval_loss,
            "eval_loss_to_submit": self.eval_loss_to_submit,
            "bpc": self.bpc,
            "bppl": self.bppl,
            "total_bytes": self.total_bytes,
            "total_target_tokens": self.total_target_tokens,
            "token_byte_ratio": self.token_byte_ratio,
            "vocab_size": self.vocab_size,
            "model_params_m": self.model_params_m,
            "assignment_id": self.assignment_id,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }
