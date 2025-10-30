import requests


class FedLedger:
    def __init__(self, api_key: str, base_url: str = "https://fed-ledger-prod.flock.io/api", api_version: str = "v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version.strip("/")
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

    def submit_heartbeat(self, worker_id: str, payload: dict):
        url = f"{self.url}/workers/{worker_id}/heartbeat"
        return requests.post(url, headers=self.headers, json=payload)

    def submit_inference_metrics(self, worker_id: str, payload: dict):
        url = f"{self.url}/workers/{worker_id}/inference-metrics"
        return requests.post(url, headers=self.headers, json=payload)
