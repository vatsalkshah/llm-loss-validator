"""
Validation assignment poller for dual-mode worker.

Polls FedLedger for validation assignments at regular intervals,
hydrates models via the cache manager, and dispatches to ValidationRunner.
"""

import asyncio
import os
import tempfile
import time
from typing import Optional

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import HfArgumentParser, TrainingArguments

from client.fed_ledger import FedLedger
from core.gpu_utils import get_gpu_type
from core.model_cache import ModelCacheManager
from core.validation_runner import ValidationRunner, LOSS_FOR_MODEL_PARAMS_EXCEED
from core.validation_exceptions import (
    InvalidDatasetException,
    InvalidLoraConfigException,
    InvalidModelException,
    ModelParamsExceededException,
)
from worker.state import WorkerState, ValidationJob


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def download_validation_file(url: str) -> str:
    """
    Download a validation dataset from a signed URL.
    
    Args:
        url: Signed URL to download from
        
    Returns:
        Path to the downloaded temporary file
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            temp_file.flush()
            temp_file.seek(0)
            file_path = temp_file.name
            logger.info(f"Downloaded validation file to {file_path}")
            return file_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading validation file: {e}")
        raise


class ValidationPoller:
    """
    Polls FedLedger for validation assignments and executes them.
    
    Respects the existing 3-minute polling contract with FedLedger,
    integrates with the cache manager for model hydration, and
    coordinates with WorkerState for priority scheduling.
    """
    
    def __init__(
        self,
        state: WorkerState,
        fed_ledger: FedLedger,
        validation_runner: ValidationRunner,
        cache_manager: ModelCacheManager,
        validation_args_file: str,
        task_id: str,
        polling_interval: int = 180,  # 3 minutes
        lora_only: bool = True,
    ):
        """
        Initialize the validation poller.
        
        Args:
            state: WorkerState for coordination
            fed_ledger: FedLedger client
            validation_runner: ValidationRunner for execution
            cache_manager: ModelCacheManager for model hydration
            validation_args_file: Path to validation config JSON
            task_id: Comma-separated task IDs to poll
            polling_interval: Polling interval in seconds (default: 180)
            lora_only: Only validate LoRA models
        """
        self._state = state
        self._fed_ledger = fed_ledger
        self._runner = validation_runner
        self._cache_manager = cache_manager
        self._validation_args_file = validation_args_file
        self._task_ids = task_id.split(",")
        self._polling_interval = polling_interval
        self._lora_only = lora_only
        self._shutdown = False
        self._task: Optional[asyncio.Task] = None
        self._last_successful_request_time = [time.time()] * len(self._task_ids)
        
        # Load validation args
        parser = HfArgumentParser(TrainingArguments)
        self._val_args = parser.parse_json_file(json_file=validation_args_file)[0]
        self._gpu_type = get_gpu_type()
        
        logger.info(f"ValidationPoller initialized for tasks: {self._task_ids}")
    
    async def start(self):
        """Start the validation polling loop."""
        logger.info("Starting validation poller")
        self._shutdown = False
        self._task = asyncio.create_task(self._polling_loop())
    
    async def stop(self):
        """Stop the validation poller."""
        logger.info("Stopping validation poller")
        self._shutdown = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _polling_loop(self):
        """Main polling loop."""
        try:
            while not self._shutdown:
                try:
                    await self._poll_and_execute()
                except Exception as e:
                    logger.error(f"Error in polling loop: {e}", exc_info=True)
                
                # Sleep for polling interval
                await asyncio.sleep(self._polling_interval)
                
        except asyncio.CancelledError:
            logger.info("Validation poller cancelled")
            raise
    
    async def _poll_and_execute(self):
        """Poll for an assignment and execute if available."""
        # Try each task ID in round-robin fashion
        assignment_response = None
        
        for index, task_id in enumerate(self._task_ids):
            # Run blocking request in executor
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, self._fed_ledger.request_validation_assignment, task_id
            )
            
            if resp.status_code == 200:
                self._last_successful_request_time[index] = time.time()
                assignment_response = resp
                break
            else:
                # Handle rate limiting
                if resp.json() == {
                    "detail": "Rate limit reached for validation assignment lookup: 1 per 3 minutes"
                }:
                    time_since_last = time.time() - self._last_successful_request_time[index]
                    if time_since_last < self._polling_interval:
                        remaining = self._polling_interval - time_since_last
                        logger.info(
                            f"Rate limited for task {task_id}, "
                            f"waiting {int(remaining)}s until next attempt"
                        )
                elif resp.json() == {"detail": "No task submissions available to validate"}:
                    logger.debug(f"No assignments available for task {task_id}")
                else:
                    logger.warning(f"Failed to request assignment for task {task_id}: {resp.content}")
        
        if not assignment_response or assignment_response.status_code != 200:
            logger.debug("No assignments available across all tasks")
            return
        
        # Parse assignment
        assignment = assignment_response.json()
        assignment_id = assignment["id"]
        
        logger.info(f"Received validation assignment: {assignment_id}")
        
        # Download evaluation file
        eval_file = None
        try:
            loop = asyncio.get_event_loop()
            eval_file = await loop.run_in_executor(
                None, download_validation_file, assignment["data"]["validation_set_url"]
            )
            
            # Create validation job
            job = ValidationJob(
                assignment_id=assignment_id,
                model_name_or_path=assignment["task_submission"]["data"]["hg_repo_id"],
                base_model=assignment["data"]["base_model"],
                eval_file=eval_file,
                context_length=assignment["data"]["context_length"],
                max_params=assignment["data"]["max_params"],
                revision=assignment["task_submission"]["data"].get("revision", "main"),
            )
            
            # Queue validation job (blocks new inference)
            await self._state.queue_validation(job)
            
            # Wait for in-flight inference to complete
            logger.info("Waiting for active inference requests to complete...")
            await self._state.wait_for_inference_clear()
            
            # Mark validation as started
            await self._state.start_validation()
            
            # Optionally hydrate model via cache manager
            await self._maybe_hydrate_model(job)
            
            # Execute validation in executor (blocking operation)
            logger.info(f"Executing validation for assignment {assignment_id}")
            success = await self._execute_validation(job)
            
            # Mark validation complete
            await self._state.complete_validation(success=success)
            
            logger.info(f"Validation completed for assignment {assignment_id}")
            
        except Exception as e:
            logger.error(f"Error executing validation assignment {assignment_id}: {e}", exc_info=True)
            await self._state.complete_validation(success=False)
            # Mark assignment as failed
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._fed_ledger.mark_assignment_as_failed, assignment_id
            )
        finally:
            # Clean up eval file
            if eval_file and os.path.exists(eval_file):
                try:
                    os.remove(eval_file)
                except Exception as e:
                    logger.warning(f"Failed to remove eval file: {e}")
    
    async def _maybe_hydrate_model(self, job: ValidationJob):
        """
        Optionally pre-download/hydrate the model via cache manager.
        
        Args:
            job: ValidationJob containing model details
        """
        try:
            # Check if model is cached
            if not self._cache_manager.is_model_cached(job.model_name_or_path, job.revision):
                logger.info(f"Pre-downloading model {job.model_name_or_path} via cache manager")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._cache_manager.download_model,
                    job.model_name_or_path,
                    "lora" if self._lora_only else "full",
                    job.revision,
                )
                logger.info(f"Model {job.model_name_or_path} cached successfully")
            else:
                logger.info(f"Model {job.model_name_or_path} already cached")
        except Exception as e:
            logger.warning(f"Failed to hydrate model via cache manager: {e}")
            # Continue anyway - ValidationRunner will download if needed
    
    async def _execute_validation(self, job: ValidationJob) -> bool:
        """
        Execute the validation job.
        
        Args:
            job: ValidationJob to execute
            
        Returns:
            True if validation completed successfully, False otherwise
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Run validation in executor
            result = await loop.run_in_executor(
                None,
                self._runner.evaluate_model,
                job.model_name_or_path,
                job.base_model,
                job.eval_file,
                job.context_length,
                job.max_params,
                self._val_args,
                job.assignment_id,
                self._lora_only,
                job.revision,
            )
            
            # Submit result
            resp = await loop.run_in_executor(
                None,
                self._fed_ledger.submit_validation_result,
                job.assignment_id,
                result.eval_loss_to_submit,
                self._gpu_type,
            )
            
            if resp.status_code != 200:
                logger.error(f"Failed to submit validation result: {resp.content}")
                if resp.json() == {"detail": "Validation assignment is not in validating status"}:
                    logger.info("Assignment no longer in validating status, marking as failed")
                    await loop.run_in_executor(
                        None, self._fed_ledger.mark_assignment_as_failed, job.assignment_id
                    )
                return False
            
            logger.info(
                f"Successfully submitted validation result (BPC: {result.eval_loss_to_submit}) "
                f"for assignment {job.assignment_id}"
            )
            return True
            
        except ModelParamsExceededException as e:
            logger.error(f"Model params exceeded: {e}")
            # Submit high loss
            resp = await loop.run_in_executor(
                None,
                self._fed_ledger.submit_validation_result,
                job.assignment_id,
                LOSS_FOR_MODEL_PARAMS_EXCEED,
                self._gpu_type,
            )
            if resp.status_code != 200:
                logger.error(f"Failed to submit validation result: {resp.content}")
            return False
            
        except (InvalidModelException, InvalidLoraConfigException, InvalidDatasetException) as e:
            logger.error(f"Invalid model/dataset: {e}")
            await loop.run_in_executor(
                None, self._fed_ledger.mark_assignment_as_failed, job.assignment_id
            )
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}", exc_info=True)
            await loop.run_in_executor(
                None, self._fed_ledger.mark_assignment_as_failed, job.assignment_id
            )
            return False
