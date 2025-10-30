import requests
from typing import Dict, List, Optional
from loguru import logger


class FedLedger:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://fed-ledger-prod.flock.io/api"
        self.api_version = "v1"
        self.url = f"{self.base_url}/{self.api_version}"
        self.headers = {
            "flock-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def request_validation_assignment(self, task_id: str):
        url = f"{self.url}/tasks/request-validation-assignment/{task_id}"
        response = requests.post(url, headers=self.headers)
        return response

    def submit_validation_result(self, assignment_id: str, loss: float, gpu_type: str):
        url = f"{self.url}/tasks/update-validation-assignment/{assignment_id}"
        response = requests.post(
            url,
            headers=self.headers,
            json={
                "status": "completed",
                "data": {
                    "loss": loss,
                    "gpu_type": gpu_type,
                },
            },
        )
        return response

    def mark_assignment_as_failed(self, assignment_id: str):
        url = f"{self.url}/tasks/update-validation-assignment/{assignment_id}"
        response = requests.post(
            url,
            headers=self.headers,
            json={
                "status": "failed",
            },
        )
        return response

    def get_cache_commands(self, worker_id: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Pull cache management commands from the Fed Ledger backend.

        This is a stub method that retrieves pending cache operations (download, evict, list)
        from the backend. Commands are expected to have the following structure:
        {
            "command": "download" | "evict" | "list",
            "model_id": str,
            "revision": str (optional),
            "model_type": str (optional),
            ...
        }

        Args:
            worker_id: Optional worker identifier to fetch commands for specific worker

        Returns:
            List of command dictionaries if successful, None if endpoint unavailable or error

        Gracefully handles:
            - 404: Endpoint not yet implemented
            - 503: Service temporarily unavailable
            - Network errors
        """
        try:
            url = f"{self.url}/cache/commands"
            if worker_id:
                url += f"?worker_id={worker_id}"

            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                commands = data.get("commands", [])
                logger.info(f"Retrieved {len(commands)} cache commands")
                return commands

            elif response.status_code == 404:
                logger.debug("Cache commands endpoint not implemented (404)")
                return None

            elif response.status_code == 503:
                logger.warning("Cache commands service temporarily unavailable (503)")
                return None

            else:
                logger.warning(
                    f"Unexpected response from cache commands endpoint: {response.status_code}"
                )
                return None

        except requests.exceptions.Timeout:
            logger.warning("Cache commands request timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning("Cannot connect to cache commands endpoint")
            return None
        except Exception as e:
            logger.warning(f"Error fetching cache commands: {e}")
            return None

    def report_cache_state(self, cache_stats: Dict, worker_id: Optional[str] = None) -> bool:
        """
        Report current cache state to the Fed Ledger backend.

        This method sends cache statistics to the backend for monitoring and
        coordination purposes. The backend can use this information to make
        decisions about model distribution across workers.

        Args:
            cache_stats: Dictionary with cache statistics, typically from
                        ModelCacheManager.get_cache_stats()
            worker_id: Optional worker identifier

        Returns:
            True if report was successfully submitted, False otherwise

        Gracefully handles:
            - 404: Endpoint not yet implemented
            - 503: Service temporarily unavailable
            - Network errors
        """
        try:
            url = f"{self.url}/cache/report"

            payload = {
                "cache_stats": cache_stats,
                "timestamp": cache_stats.get("timestamp", None),
            }

            if worker_id:
                payload["worker_id"] = worker_id

            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.debug("Cache state reported successfully")
                return True

            elif response.status_code == 404:
                logger.debug("Cache report endpoint not implemented (404)")
                return False

            elif response.status_code == 503:
                logger.warning("Cache report service temporarily unavailable (503)")
                return False

            else:
                logger.warning(
                    f"Unexpected response from cache report endpoint: {response.status_code}"
                )
                return False

        except requests.exceptions.Timeout:
            logger.warning("Cache report request timed out")
            return False
        except requests.exceptions.ConnectionError:
            logger.warning("Cannot connect to cache report endpoint")
            return False
        except Exception as e:
            logger.warning(f"Error reporting cache state: {e}")
            return False

    def execute_cache_command(self, command_id: str, status: str, result: Optional[Dict] = None) -> bool:
        """
        Report the execution result of a cache command back to the backend.

        Args:
            command_id: Identifier of the executed command
            status: Execution status - "completed", "failed", "skipped"
            result: Optional result data (e.g., cache stats after eviction)

        Returns:
            True if report was successful, False otherwise
        """
        try:
            url = f"{self.url}/cache/commands/{command_id}/result"

            payload = {
                "status": status,
            }

            if result:
                payload["result"] = result

            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=10
            )

            if response.status_code in (200, 404):
                return True
            else:
                logger.warning(
                    f"Failed to report command execution: {response.status_code}"
                )
                return False

        except Exception as e:
            logger.warning(f"Error reporting command execution: {e}")
            return False

    def report_worker_heartbeat(self, payload: Dict) -> bool:
        """
        Publish a worker heartbeat payload to the Fed Ledger backend.

        This is a stub integration point that allows the backend to ingest
        worker telemetry (status, cache contents, GPU availability, etc.).

        Args:
            payload: Telemetry payload to submit

        Returns:
            True if the heartbeat was accepted, False otherwise
        """
        try:
            url = f"{self.url}/telemetry/heartbeat"
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=10,
            )

            if response.status_code in (200, 202):
                logger.debug("FedLedger heartbeat accepted")
                return True
            if response.status_code == 404:
                logger.debug("Telemetry heartbeat endpoint not available (404)")
                return False
            if response.status_code == 503:
                logger.warning("Telemetry heartbeat service unavailable (503)")
                return False

            logger.warning(
                "Unexpected response from telemetry heartbeat endpoint",
                status_code=response.status_code,
                body=response.text,
            )
            return False

        except requests.exceptions.Timeout:
            logger.warning("Telemetry heartbeat request timed out")
            return False
        except requests.exceptions.ConnectionError:
            logger.warning("Cannot reach telemetry heartbeat endpoint")
            return False
        except Exception as e:
            logger.warning(f"Error reporting telemetry heartbeat: {e}")
            return False
