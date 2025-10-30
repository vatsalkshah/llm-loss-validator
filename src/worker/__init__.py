"""
Worker module for dual-mode orchestration.

This package exposes the dual-mode worker manager that coordinates
validation assignments and inference serving within a single process.
"""

from worker.manager import DualModeWorker
from worker.inference_guard import InferenceActivityTracker

__all__ = ["DualModeWorker", "InferenceActivityTracker"]
