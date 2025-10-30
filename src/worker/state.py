"""
Worker State Management Module.

Defines enums, dataclasses, and state tracking for the dual-mode worker,
including priority states and active request tracking.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class WorkerMode(Enum):
    """Current operational mode of the worker."""
    IDLE = "idle"
    INFERENCE_ACTIVE = "inference_active"
    VALIDATION_PENDING = "validation_pending"
    VALIDATION_ACTIVE = "validation_active"
    SHUTDOWN = "shutdown"


class ValidationJobStatus(Enum):
    """Status of a validation job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ValidationJob:
    """Represents a validation assignment to be processed."""
    assignment_id: str
    model_name_or_path: str
    base_model: str
    eval_file: str
    context_length: int
    max_params: int
    revision: str = "main"
    status: ValidationJobStatus = ValidationJobStatus.PENDING


class WorkerState:
    """
    Thread-safe state manager for the dual-mode worker.
    
    Coordinates between inference serving and validation jobs,
    tracking active inference requests and managing priority.
    """
    
    def __init__(self):
        """Initialize worker state."""
        self._mode = WorkerMode.IDLE
        self._active_inference_count = 0
        self._pending_validation: Optional[ValidationJob] = None
        self._lock = asyncio.Lock()
        self._inference_clear = asyncio.Event()
        self._inference_clear.set()  # Initially no blocking
        self._validation_complete = asyncio.Event()
        self._validation_complete.set()  # Initially no validation running
    
    async def get_mode(self) -> WorkerMode:
        """Get current worker mode."""
        async with self._lock:
            return self._mode
    
    async def set_mode(self, mode: WorkerMode):
        """Set worker mode."""
        async with self._lock:
            self._mode = mode
    
    async def start_inference_request(self) -> bool:
        """
        Attempt to start an inference request.
        
        Returns:
            True if request can proceed, False if blocked by validation
        """
        async with self._lock:
            # Block new requests if validation is pending or active
            if self._mode in (WorkerMode.VALIDATION_PENDING, WorkerMode.VALIDATION_ACTIVE):
                return False
            
            self._active_inference_count += 1
            if self._active_inference_count > 0:
                self._inference_clear.clear()
                self._mode = WorkerMode.INFERENCE_ACTIVE
            
            return True
    
    async def end_inference_request(self):
        """Mark an inference request as complete."""
        async with self._lock:
            self._active_inference_count = max(0, self._active_inference_count - 1)
            
            if self._active_inference_count == 0:
                self._inference_clear.set()
                # Return to idle if no validation pending
                if self._mode == WorkerMode.INFERENCE_ACTIVE:
                    self._mode = WorkerMode.IDLE
    
    async def get_active_inference_count(self) -> int:
        """Get count of active inference requests."""
        async with self._lock:
            return self._active_inference_count
    
    async def queue_validation(self, job: ValidationJob):
        """
        Queue a validation job for execution.
        
        Args:
            job: ValidationJob to queue
        """
        async with self._lock:
            self._pending_validation = job
            self._mode = WorkerMode.VALIDATION_PENDING
            # Block new inference requests
            # Existing requests will complete naturally
    
    async def get_pending_validation(self) -> Optional[ValidationJob]:
        """Get the pending validation job if any."""
        async with self._lock:
            return self._pending_validation
    
    async def start_validation(self):
        """Mark validation as started."""
        async with self._lock:
            if self._pending_validation:
                self._pending_validation.status = ValidationJobStatus.RUNNING
            self._mode = WorkerMode.VALIDATION_ACTIVE
            self._validation_complete.clear()
    
    async def complete_validation(self, success: bool = True):
        """
        Mark validation as complete.
        
        Args:
            success: Whether validation completed successfully
        """
        async with self._lock:
            if self._pending_validation:
                self._pending_validation.status = (
                    ValidationJobStatus.COMPLETED if success else ValidationJobStatus.FAILED
                )
            self._pending_validation = None
            self._mode = WorkerMode.IDLE
            self._validation_complete.set()
    
    async def wait_for_inference_clear(self):
        """Wait for all active inference requests to complete."""
        await self._inference_clear.wait()
    
    async def wait_for_validation_complete(self):
        """Wait for validation to complete."""
        await self._validation_complete.wait()
    
    async def is_shutdown(self) -> bool:
        """Check if worker is shutting down."""
        async with self._lock:
            return self._mode == WorkerMode.SHUTDOWN
    
    async def initiate_shutdown(self):
        """Initiate worker shutdown."""
        async with self._lock:
            self._mode = WorkerMode.SHUTDOWN
