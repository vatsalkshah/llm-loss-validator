import os
import time
import shutil

import click
import requests
import tempfile
from loguru import logger
from transformers import HfArgumentParser, TrainingArguments, file_utils

from dotenv import load_dotenv
from pathlib import Path
from core.gpu_utils import get_gpu_type
from core.constant import SUPPORTED_BASE_MODELS
from core.exception import (
    handle_os_error,
    handle_runtime_error,
    handle_value_error,
)
from core.log_utils import _log_summary_table
from core.validation_runner import ValidationRunner, LOSS_FOR_MODEL_PARAMS_EXCEED
from core.validation_exceptions import (
    ModelParamsExceededException,
    InvalidModelException,
    InvalidLoraConfigException,
    InvalidDatasetException,
)
from tenacity import retry, stop_after_attempt, wait_exponential
from client.fed_ledger import FedLedger
import sys


load_dotenv()
TIME_SLEEP = int(os.getenv("TIME_SLEEP", 60 * 3))
ASSIGNMENT_LOOKUP_INTERVAL = 60 * 3  # 3 minutes
FLOCK_API_KEY = os.getenv("FLOCK_API_KEY")
if FLOCK_API_KEY is None:
    raise ValueError("FLOCK_API_KEY is not set")
HF_TOKEN = os.getenv("HF_TOKEN")
IS_DOCKER_CONTAINER = os.getenv("IS_DOCKER_CONTAINER", False)

if not IS_DOCKER_CONTAINER:
    import git  # only import git in non-docker container environment because it is not installed in docker image

if HF_TOKEN is None:
    raise ValueError(
        "You need to set HF_TOKEN to download some gated model from HuggingFace"
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def download_file(url):
    try:
        # Send a GET request to the signed URL
        response = requests.get(url, stream=True)
        # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()

        # Create a temporary file to save the content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write the content to the temp file in binary mode
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)

            # move the file pointer to the beginning of the file
            temp_file.flush()
            temp_file.seek(0)

            # get the file path
            file_path = temp_file.name
            logger.info(f"Downloaded the file to {file_path}")

            return file_path

    except requests.exceptions.RequestException as e:
        # Handle any exception that can be raised by the requests library
        logger.error(f"An error occurred while downloading the file: {e}")
        raise e


def is_latest_version(repo_path: str):
    """
    Check if the current branch is up-to-date with the remote main branch.
    Parameters:
    - repo_path (str or Path): The path to the git repository.
    """
    try:
        repo = git.Repo(repo_path)
        origin = repo.remotes.origin
        origin.fetch()

        local_commit = repo.commit("main")
        remote_commit = repo.commit("origin/main")

        if local_commit.hexsha != remote_commit.hexsha:
            logger.error(
                "The local code is not up to date with the main branch.Pls update your version"
            )
            raise
    except git.exc.InvalidGitRepositoryError:
        logger.error("This is not a git repository.")
        raise
    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise


def clean_model_cache(
    auto_clean_cache: bool, cache_path: str = file_utils.default_cache_path
):
    """
    Cleans up the local model cache directory by removing directories that are not
    listed in SUPPORTED_BASE_MODELS.

    Parameters:
    - auto_clean_cache (bool): A flag to determine whether to clean the cache.
    - cache_path (str): The path to the cache directory. Defaults to file_utils.default_cache_path.
    """
    if not auto_clean_cache:
        return

    try:
        cache_path = Path(cache_path)
        for item in cache_path.iterdir():
            if item.is_dir() and item.name.startswith("models"):
                if item.name not in {
                    f"models--{BASE_MODEL.replace('/', '--')}"
                    for BASE_MODEL in SUPPORTED_BASE_MODELS
                }:
                    shutil.rmtree(item)
                    logger.info(f"Removed directory: {item}")
        logger.info("Successfully cleaned up the local model cache")
    except (OSError, shutil.Error) as e:
        logger.error(f"Failed to clean up the local model cache: {e}")


@click.group()
def cli():
    pass


@click.command()
@click.option("--model_name_or_path", required=True, type=str, help="")
@click.option("--base_model", required=True, type=str, help="")
@click.option("--eval_file", default="./data/dummy_data.jsonl", type=str, help="")
@click.option("--context_length", required=True, type=int)
@click.option("--max_params", required=True, type=int)
@click.option(
    "--validation_args_file",
    type=str,
    default="validation_config.json.example",
    help="",
)
@click.option(
    "--assignment_id",
    type=str,
    help="The id of the validation assignment",
)
@click.option(
    "--local_test",
    is_flag=True,
    help="Run the script in local test mode to avoid submitting to the server",
)
@click.option(
    "--lora_only", type=bool, default=True, help="Only validate repo with lora weight"
)
def validate(
    model_name_or_path: str,
    base_model: str,
    eval_file: str,
    context_length: int,
    max_params: int,
    validation_args_file: str,
    assignment_id: str = None,
    local_test: bool = False,
    lora_only: bool = True,
    revision: str = "main",
):
    if not local_test and assignment_id is None:
        raise ValueError(
            "assignment_id is required for submitting validation result to the server"
        )

    try:
        fed_ledger = FedLedger(FLOCK_API_KEY)
        parser = HfArgumentParser(TrainingArguments)
        val_args = parser.parse_json_file(json_file=validation_args_file)[0]
        gpu_type = get_gpu_type()

        runner = ValidationRunner(hf_token=HF_TOKEN)
        result = runner.evaluate_model(
            model_name_or_path=model_name_or_path,
            base_model=base_model,
            eval_file=eval_file,
            context_length=context_length,
            max_params=max_params,
            val_args=val_args,
            assignment_id=assignment_id,
            lora_only=lora_only,
            revision=revision,
        )

        _log_summary_table(
            model_name_or_path=model_name_or_path,
            eval_loss=result.eval_loss,
            bpc_metrics={
                "bpc": result.bpc,
                "bppl": result.bppl,
                "nll_token_nats_total": result.metadata.get("nll_token_nats_total", float("nan")),
                "nll_token_bits_total": result.metadata.get("nll_token_bits_total", float("nan")),
            },
            token_byte_ratio=result.token_byte_ratio,
            total_target_tokens=result.total_target_tokens,
            total_bytes=result.total_bytes,
            vocab_size=result.vocab_size,
            model_params_m=result.model_params_m,
        )

        if local_test:
            logger.info(
                "The model can be correctly validated by validators (raw loss)."
            )
            if not result.is_bpc_valid():
                logger.warning(
                    "Could not calculate BPC/bPPL for local test due to zero bytes or invalid loss."
                )
            return

        resp = fed_ledger.submit_validation_result(
            assignment_id=assignment_id,
            loss=result.eval_loss_to_submit,
            gpu_type=gpu_type,
        )
        if resp.status_code != 200:
            logger.error(f"Failed to submit validation result: {resp.content}")
            if resp.json() == {
                "detail": "Validation assignment is not in validating status"
            }:
                logger.info(
                    "Validation assignment is not in validating status anymore, marking it as failed"
                )
                fed_ledger.mark_assignment_as_failed(assignment_id)
            return
        logger.info(
            f"Successfully submitted validation result (BPC: {result.eval_loss_to_submit}) for assignment {assignment_id}"
        )

    except ModelParamsExceededException as e:
        logger.error(str(e))
        if not local_test:
            resp = fed_ledger.submit_validation_result(
                assignment_id=assignment_id,
                loss=LOSS_FOR_MODEL_PARAMS_EXCEED,
                gpu_type=gpu_type,
            )
            if resp.status_code != 200:
                logger.error(f"Failed to submit validation result: {resp.content}")
        return
    except (InvalidModelException, InvalidLoraConfigException, InvalidDatasetException) as e:
        logger.error(str(e))
        if not local_test:
            fed_ledger.mark_assignment_as_failed(assignment_id)
        return
    except Exception as e:
        raise e


@click.command()
@click.option(
    "--validation_args_file",
    type=str,
    default="validation_config.json.example",
    help="",
)
@click.option(
    "--task_id",
    type=str,
    help="The id of the task",
)
@click.option(
    "--auto_clean_cache",
    type=bool,
    default=True,
    help="Auto clean the model cache except for the base model",
)
@click.option(
    "--lora_only", type=bool, default=True, help="Only validate repo with lora weight"
)
def loop(
    validation_args_file: str,
    task_id: str = None,
    auto_clean_cache: bool = True,
    lora_only: bool = True,
):
    if task_id is None:
        raise ValueError("task_id is required for asking assignment_id")
    if auto_clean_cache:
        logger.info("Auto clean the model cache except for the base model")
    else:
        logger.info("Skip auto clean the model cache")

    repo_path = Path(__file__).resolve().parent.parent

    if not IS_DOCKER_CONTAINER:
        is_latest_version(repo_path)
    else:
        logger.info("Skip checking the latest version in docker container")
        logger.info(
            "Please make sure you are using the latest version of the docker image."
        )

    fed_ledger = FedLedger(FLOCK_API_KEY)
    task_id_list = task_id.split(",")
    logger.info(f"Validating task_id: {task_id_list}")
    last_successful_request_time = [time.time()] * len(task_id_list)
    while True:
        clean_model_cache(auto_clean_cache)

        for index, task_id_num in enumerate(task_id_list):
            resp = fed_ledger.request_validation_assignment(task_id_num)
            if resp.status_code == 200:
                last_successful_request_time[index] = time.time()
                break
            else:
                if resp.json() == {
                    "detail": "No task submissions available to validate"
                }:
                    logger.info(
                        "Failed to ask assignment_id: No task submissions available to validate"
                    )
                else:
                    logger.error(f"Failed to ask assignment_id: {resp.content}")
                if resp.json() == {
                    "detail": "Rate limit reached for validation assignment lookup: 1 per 3 minutes"
                }:
                    time_since_last_success = (
                        time.time() - last_successful_request_time[index]
                    )
                    if time_since_last_success < ASSIGNMENT_LOOKUP_INTERVAL:
                        time_to_sleep = (
                            ASSIGNMENT_LOOKUP_INTERVAL - time_since_last_success
                        )
                        logger.info(f"Sleeping for {int(time_to_sleep)} seconds")
                        time.sleep(time_to_sleep)
                    continue
                else:
                    logger.info(f"Sleeping for {int(TIME_SLEEP)} seconds")
                    time.sleep(TIME_SLEEP)
                    continue

        if resp is None or resp.status_code != 200:
            continue
        resp = resp.json()
        eval_file = download_file(resp["data"]["validation_set_url"])
        revision = resp["task_submission"]["data"].get("revision", "main")
        assignment_id = resp["id"]

        for attempt in range(3):
            try:
                ctx = click.Context(validate)
                ctx.invoke(
                    validate,
                    model_name_or_path=resp["task_submission"]["data"]["hg_repo_id"],
                    base_model=resp["data"]["base_model"],
                    eval_file=eval_file,
                    context_length=resp["data"]["context_length"],
                    max_params=resp["data"]["max_params"],
                    validation_args_file=validation_args_file,
                    assignment_id=resp["id"],
                    local_test=False,
                    lora_only=lora_only,
                    revision=revision,
                )
                break  # Break the loop if no exception
            except KeyboardInterrupt:
                # directly terminate the process if keyboard interrupt
                sys.exit(1)
            except OSError as e:
                handle_os_error(e, assignment_id, fed_ledger)
            except RuntimeError as e:
                handle_runtime_error(e, assignment_id, fed_ledger)
            except ValueError as e:
                handle_value_error(e, assignment_id, fed_ledger)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    logger.error(
                        f"Marking assignment {assignment_id} as failed after 3 attempts"
                    )
                    fed_ledger.mark_assignment_as_failed(assignment_id)

        os.remove(eval_file)


@click.command()
@click.option(
    "--validation_args_file",
    type=str,
    default="validation_config.json.example",
    help="Path to validation config JSON",
)
@click.option(
    "--task_id",
    type=str,
    required=True,
    help="Comma-separated task IDs to validate",
)
@click.option(
    "--inference_host",
    type=str,
    default="0.0.0.0",
    help="Host for inference server",
)
@click.option(
    "--inference_port",
    type=int,
    default=8000,
    help="Port for inference server",
)
@click.option(
    "--polling_interval",
    type=int,
    default=180,
    help="Validation polling interval in seconds (default: 180)",
)
@click.option(
    "--telemetry_interval",
    type=int,
    default=60,
    help="Telemetry reporting interval in seconds (default: 60)",
)
@click.option(
    "--lora_only",
    type=bool,
    default=True,
    help="Only validate LoRA models",
)
def worker(
    validation_args_file: str,
    task_id: str,
    inference_host: str,
    inference_port: int,
    polling_interval: int,
    telemetry_interval: int,
    lora_only: bool,
):
    """
    Run the dual-mode worker.
    
    Coordinates validation assignments and inference serving in a single process,
    with validation jobs getting priority while allowing in-flight inference
    requests to complete before pre-emption.
    """
    import asyncio
    from worker.manager import run_worker
    
    logger.info("Starting dual-mode worker")
    logger.info(f"Task IDs: {task_id}")
    logger.info(f"Inference endpoint: {inference_host}:{inference_port}")
    logger.info(f"Polling interval: {polling_interval}s")
    
    try:
        asyncio.run(
            run_worker(
                flock_api_key=FLOCK_API_KEY,
                hf_token=HF_TOKEN,
                validation_args_file=validation_args_file,
                task_id=task_id,
                inference_host=inference_host,
                inference_port=inference_port,
                polling_interval=polling_interval,
                telemetry_interval=telemetry_interval,
                lora_only=lora_only,
            )
        )
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in worker: {e}")
        raise


cli.add_command(validate)
cli.add_command(loop)
cli.add_command(worker)

if __name__ == "__main__":
    cli()
