"""
Dual-mode worker manager.

Coordinates validation assignments and inference serving in a single process,
ensuring validation jobs get priority while allowing in-flight inference
requests to complete before pre-emption.
"""

import asyncio
import os
import signal
from pathlib import Path
from typing import Optional

import uvicorn
from loguru import logger
from transformers import file_utils

from client.fed_ledger import FedLedger
from core.model_cache import ModelCacheManager
from core.validation_runner import ValidationRunner
from server.config import config as server_config
from worker.inference_guard import InferenceActivityTracker
from worker.state import WorkerState
from worker.telemetry import TelemetryReporter
from worker.validation_poller import ValidationPoller


class DualModeWorker:
    """
    Worker orchestrator for dual-mode operation.
    
    Manages:
    - Inference server (FastAPI/uvicorn) in background
    - Validation assignment polling (FedLedger)
    - Model cache hydration
    - Priority scheduling (validation preempts inference)
    - Clean startup/shutdown
    """
    
    def __init__(
        self,
        flock_api_key: str,
        hf_token: str,
        validation_args_file: str,
        task_id: str,
        cache_dir: Optional[str] = None,
        inference_host: str = "0.0.0.0",
        inference_port: int = 8000,
        polling_interval: int = 180,
        telemetry_interval: int = 60,
        telemetry_webhook: Optional[str] = None,
        telemetry_location: Optional[str] = None,
        telemetry_worker_id: Optional[str] = None,
        telemetry_enabled: Optional[bool] = None,
        lora_only: bool = True,
    ):
        """
        Initialize the dual-mode worker.
        
        Args:
            flock_api_key: Flock API key for FedLedger
            hf_token: HuggingFace token
            validation_args_file: Path to validation config JSON
            task_id: Comma-separated task IDs to validate
            cache_dir: Model cache directory
            inference_host: Host for inference server
            inference_port: Port for inference server
            polling_interval: Validation polling interval in seconds
            telemetry_interval: Telemetry reporting interval in seconds
            lora_only: Only validate LoRA models
        """
        self.flock_api_key = flock_api_key
        self.hf_token = hf_token
        self.validation_args_file = validation_args_file
        self.task_id = task_id
        self.inference_host = inference_host
        self.inference_port = inference_port
        self.polling_interval = polling_interval
        self.telemetry_interval = telemetry_interval
        self.lora_only = lora_only
        
        # Initialize components
        self._state = WorkerState()
        self._activity_tracker = InferenceActivityTracker(self._state)
        
        # Cache manager
        self._cache_manager = ModelCacheManager(
            cache_dir=cache_dir or file_utils.default_cache_path,
            hf_token=hf_token,
        )
        
        # FedLedger client
        self._fed_ledger = FedLedger(flock_api_key)
        
        # Validation runner
        self._validation_runner = ValidationRunner(hf_token=hf_token)
        
        # Validation poller
        self._validation_poller = ValidationPoller(
            state=self._state,
            fed_ledger=self._fed_ledger,
            validation_runner=self._validation_runner,
            cache_manager=self._cache_manager,
            validation_args_file=validation_args_file,
            task_id=task_id,
            polling_interval=polling_interval,
            lora_only=lora_only,
        )
        
        # Telemetry reporter
        self._telemetry = TelemetryReporter(
            state=self._state,
            cache_manager=self._cache_manager,
            fed_ledger=self._fed_ledger,
            interval=telemetry_interval,
            webhook_url=telemetry_webhook,
            location=telemetry_location,
            worker_id=telemetry_worker_id,
            enabled=telemetry_enabled,
        )
        
        # Uvicorn server
        self._server: Optional[uvicorn.Server] = None
        self._server_task: Optional[asyncio.Task] = None
        
        # Shutdown flag
        self._shutdown_event = asyncio.Event()
        
        logger.info("DualModeWorker initialized")
        logger.info(f"  Task IDs: {task_id}")
        logger.info(f"  Inference: {inference_host}:{inference_port}")
        logger.info(f"  Polling interval: {polling_interval}s")
        logger.info(f"  Telemetry interval: {telemetry_interval}s")
    
    async def start(self):
        """Start the dual-mode worker."""
        logger.info("Starting dual-mode worker")
        
        # Register signal handlers
        self._register_signal_handlers()
        
        # Start telemetry
        await self._telemetry.start()
        
        # Start validation poller
        await self._validation_poller.start()
        
        # Start inference server
        await self._start_inference_server()
        
        logger.info("Dual-mode worker started successfully")
        
        # Wait for shutdown signal
        await self._shutdown_event.wait()
        
        # Cleanup
        await self.stop()
    
    async def stop(self):
        """Stop the dual-mode worker."""
        logger.info("Stopping dual-mode worker")
        
        # Signal shutdown to state
        await self._state.initiate_shutdown()
        
        # Stop validation poller
        await self._validation_poller.stop()
        
        # Stop telemetry
        await self._telemetry.stop()
        
        # Stop inference server
        await self._stop_inference_server()
        
        logger.info("Dual-mode worker stopped")
    
    async def _start_inference_server(self):
        """Start the inference server in the background."""
        logger.info("Starting inference server")
        
        # Import the app factory
        from server.main import create_app_with_tracker
        
        # Create FastAPI app with activity tracker
        app = create_app_with_tracker(self._activity_tracker)
        
        # Configure uvicorn
        uvicorn_config = uvicorn.Config(
            app=app,
            host=self.inference_host,
            port=self.inference_port,
            log_level="info",
            access_log=False,  # Reduce noise
        )
        
        # Add TLS if configured
        if server_config.tls_enabled and server_config.validate_tls_config():
            uvicorn_config.ssl_certfile = server_config.tls_cert_path
            uvicorn_config.ssl_keyfile = server_config.tls_key_path
            logger.info("TLS enabled for inference server")
        
        self._server = uvicorn.Server(uvicorn_config)
        
        # Run server in background task
        self._server_task = asyncio.create_task(self._server.serve())
        
        # Give server time to start
        await asyncio.sleep(1)
        
        logger.info(f"Inference server started on {self.inference_host}:{self.inference_port}")
    
    async def _stop_inference_server(self):
        """Stop the inference server."""
        if self._server:
            logger.info("Stopping inference server")
            self._server.should_exit = True
            
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("Inference server did not stop gracefully, cancelling")
                    self._server_task.cancel()
                    try:
                        await self._server_task
                    except asyncio.CancelledError:
                        pass
            
            logger.info("Inference server stopped")
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        
        def handle_shutdown(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            loop.create_task(self._trigger_shutdown())
        
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
    
    async def _trigger_shutdown(self):
        """Trigger graceful shutdown."""
        self._shutdown_event.set()
    
    def get_activity_tracker(self) -> InferenceActivityTracker:
        """Get the inference activity tracker."""
        return self._activity_tracker
    
    def get_state(self) -> WorkerState:
        """Get the worker state."""
        return self._state


async def run_worker(
    flock_api_key: str,
    hf_token: str,
    validation_args_file: str,
    task_id: str,
    cache_dir: Optional[str] = None,
    inference_host: str = "0.0.0.0",
    inference_port: int = 8000,
    polling_interval: int = 180,
    telemetry_interval: int = 60,
    telemetry_webhook: Optional[str] = None,
    telemetry_location: Optional[str] = None,
    telemetry_worker_id: Optional[str] = None,
    telemetry_enabled: Optional[bool] = None,
    lora_only: bool = True,
):
    """
    Run the dual-mode worker.
    
    This is the main entry point for starting the orchestrator.
    """
    worker = DualModeWorker(
        flock_api_key=flock_api_key,
        hf_token=hf_token,
        validation_args_file=validation_args_file,
        task_id=task_id,
        cache_dir=cache_dir,
        inference_host=inference_host,
        inference_port=inference_port,
        polling_interval=polling_interval,
        telemetry_interval=telemetry_interval,
        telemetry_webhook=telemetry_webhook,
        telemetry_location=telemetry_location,
        telemetry_worker_id=telemetry_worker_id,
        telemetry_enabled=telemetry_enabled,
        lora_only=lora_only,
    )
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in worker: {e}", exc_info=True)
        raise
    finally:
        await worker.stop()
