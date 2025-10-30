from unittest.mock import patch

from src.client.fed_ledger import FedLedger


@patch("src.client.fed_ledger.requests.post")
def test_submit_heartbeat(mock_post):
    ledger = FedLedger("key", base_url="https://example.com/api")
    ledger.submit_heartbeat("worker-1", {"status": "ok"})
    args, kwargs = mock_post.call_args
    assert args[0] == "https://example.com/api/v1/workers/worker-1/heartbeat"
    assert kwargs["json"]["status"] == "ok"


@patch("src.client.fed_ledger.requests.post")
def test_request_validation_assignment(mock_post):
    ledger = FedLedger("key", base_url="https://example.com/api")
    ledger.request_validation_assignment("7")
    args, _ = mock_post.call_args
    assert args[0].endswith("/tasks/request-validation-assignment/7")
