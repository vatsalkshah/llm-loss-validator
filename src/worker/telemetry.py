"""
Telemetry and monitoring for dual-mode worker.

Provides periodic status reporting and metrics collection.
"""

import asyncio
from typing import Optional

from loguru import logger

from worker.state import WorkerState, WorkerMode


class TelemetryReporter:
    """
    Periodic telemetry reporter for worker health and status.
    
    Logs worker state, active requests, and validation status at
    configurable intervals.
    """
    
    def __init__(
        self,
        state: WorkerState,
        interval: int = 60,
    ):
        """
        Initialize telemetry reporter.
        
        Args:
            state: WorkerState to monitor
            interval: Reporting interval in seconds
        """
        self._state = state
        self._interval = interval
        self._task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    async def start(self):
        """Start the telemetry reporting loop."""
        logger.info(f"Starting telemetry reporter (interval: {self._interval}s)")
        self._shutdown = False
        self._task = asyncio.create_task(self._reporting_loop())
    
    async def stop(self):
        """Stop the telemetry reporter."""
        logger.info("Stopping telemetry reporter")
        self._shutdown = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _reporting_loop(self):
        """Main telemetry reporting loop."""
        try:
            while not self._shutdown:
                await self._report_status()
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            logger.info("Telemetry reporter cancelled")
            raise
        except Exception as e:
            logger.error(f"Telemetry reporter error: {e}")
    
    async def _report_status(self):
        """Report current worker status."""
        try:
            mode = await self._state.get_mode()
            active_count = await self._state.get_active_inference_count()
            pending_validation = await self._state.get_pending_validation()
            
            status_msg = [
                "Worker Status:",
                f"  Mode: {mode.value}",
                f"  Active Inference Requests: {active_count}",
            ]
            
            if pending_validation:
                status_msg.append(
                    f"  Pending Validation: {pending_validation.assignment_id} "
                    f"(status: {pending_validation.status.value})"
                )
            
            logger.info("\n".join(status_msg))
            
        except Exception as e:
            logger.error(f"Error reporting status: {e}")
