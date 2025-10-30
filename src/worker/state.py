"""Shared worker state primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class WorkerMode(str, Enum):
    VALIDATION = "validation"
    INFERENCE = "inference"
    DUAL = "dual"


@dataclass(slots=True)
class WorkerState:
    mode: WorkerMode
    validation_active: bool = False
    inference_active: bool = False
    last_assignment_id: Optional[str] = None
    extra: dict = field(default_factory=dict)


__all__ = ["WorkerMode", "WorkerState"]
