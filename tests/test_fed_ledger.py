import unittest
from unittest.mock import MagicMock, patch
import requests

from src.client.fed_ledger import FedLedger


class TestFedLedgerCacheCommands(unittest.TestCase):
    def setUp(self):
        self.api_key = "test-api-key"
        self.client = FedLedger(self.api_key)

    @patch("src.client.fed_ledger.requests.get")
    def test_get_cache_commands_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "commands": [
                {
                    "command": "download",
                    "model_id": "test/model",
                    "revision": "main",
                    "model_type": "base",
                },
                {
                    "command": "evict",
                    "model_id": "old/model",
                    "revision": "main",
                },
            ]
        }
        mock_get.return_value = mock_response

        result = self.client.get_cache_commands()

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["command"], "download")
        self.assertEqual(result[1]["command"], "evict")

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn("cache/commands", call_args[0][0])

    @patch("src.client.fed_ledger.requests.get")
    def test_get_cache_commands_with_worker_id(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"commands": []}
        mock_get.return_value = mock_response

        result = self.client.get_cache_commands(worker_id="worker-123")

        self.assertEqual(result, [])
        call_args = mock_get.call_args
        self.assertIn("worker_id=worker-123", call_args[0][0])

    @patch("src.client.fed_ledger.requests.get")
    def test_get_cache_commands_endpoint_not_implemented(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = self.client.get_cache_commands()

        self.assertIsNone(result)

    @patch("src.client.fed_ledger.requests.get")
    def test_get_cache_commands_service_unavailable(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response

        result = self.client.get_cache_commands()

        self.assertIsNone(result)

    @patch("src.client.fed_ledger.requests.get")
    def test_get_cache_commands_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout()

        result = self.client.get_cache_commands()

        self.assertIsNone(result)

    @patch("src.client.fed_ledger.requests.get")
    def test_get_cache_commands_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = self.client.get_cache_commands()

        self.assertIsNone(result)

    @patch("src.client.fed_ledger.requests.get")
    def test_get_cache_commands_unexpected_error(self, mock_get):
        mock_get.side_effect = Exception("Unexpected error")

        result = self.client.get_cache_commands()

        self.assertIsNone(result)


class TestFedLedgerCacheReport(unittest.TestCase):
    def setUp(self):
        self.api_key = "test-api-key"
        self.client = FedLedger(self.api_key)

    @patch("src.client.fed_ledger.requests.post")
    def test_report_cache_state_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        cache_stats = {
            "total_size_bytes": 1024000,
            "model_count": 5,
            "active_count": 2,
            "timestamp": 1234567890,
        }

        result = self.client.report_cache_state(cache_stats)

        self.assertTrue(result)
        mock_post.assert_called_once()
        
        call_args = mock_post.call_args
        self.assertIn("cache/report", call_args[0][0])
        
        payload = call_args[1]["json"]
        self.assertEqual(payload["cache_stats"], cache_stats)

    @patch("src.client.fed_ledger.requests.post")
    def test_report_cache_state_with_worker_id(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        cache_stats = {"total_size_bytes": 1024}

        result = self.client.report_cache_state(cache_stats, worker_id="worker-456")

        self.assertTrue(result)
        
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["worker_id"], "worker-456")

    @patch("src.client.fed_ledger.requests.post")
    def test_report_cache_state_endpoint_not_implemented(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        result = self.client.report_cache_state({})

        self.assertFalse(result)

    @patch("src.client.fed_ledger.requests.post")
    def test_report_cache_state_service_unavailable(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_post.return_value = mock_response

        result = self.client.report_cache_state({})

        self.assertFalse(result)

    @patch("src.client.fed_ledger.requests.post")
    def test_report_cache_state_timeout(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout()

        result = self.client.report_cache_state({})

        self.assertFalse(result)

    @patch("src.client.fed_ledger.requests.post")
    def test_report_cache_state_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError()

        result = self.client.report_cache_state({})

        self.assertFalse(result)


class TestFedLedgerExecuteCacheCommand(unittest.TestCase):
    def setUp(self):
        self.api_key = "test-api-key"
        self.client = FedLedger(self.api_key)

    @patch("src.client.fed_ledger.requests.post")
    def test_execute_cache_command_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = self.client.execute_cache_command(
            "cmd-123",
            "completed",
            {"evicted_count": 2}
        )

        self.assertTrue(result)
        
        call_args = mock_post.call_args
        self.assertIn("cache/commands/cmd-123/result", call_args[0][0])
        
        payload = call_args[1]["json"]
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["result"]["evicted_count"], 2)

    @patch("src.client.fed_ledger.requests.post")
    def test_execute_cache_command_without_result(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = self.client.execute_cache_command("cmd-123", "failed")

        self.assertTrue(result)
        
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn("result", payload)

    @patch("src.client.fed_ledger.requests.post")
    def test_execute_cache_command_not_implemented(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        result = self.client.execute_cache_command("cmd-123", "completed")

        self.assertTrue(result)

    @patch("src.client.fed_ledger.requests.post")
    def test_execute_cache_command_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = self.client.execute_cache_command("cmd-123", "completed")

        self.assertFalse(result)

    @patch("src.client.fed_ledger.requests.post")
    def test_execute_cache_command_exception(self, mock_post):
        mock_post.side_effect = Exception("Network error")

        result = self.client.execute_cache_command("cmd-123", "completed")

        self.assertFalse(result)


class TestFedLedgerExistingMethods(unittest.TestCase):
    def setUp(self):
        self.api_key = "test-api-key"
        self.client = FedLedger(self.api_key)

    def test_initialization(self):
        self.assertEqual(self.client.api_key, self.api_key)
        self.assertEqual(
            self.client.url,
            "https://fed-ledger-prod.flock.io/api/v1"
        )
        self.assertEqual(
            self.client.headers["flock-api-key"],
            self.api_key
        )

    @patch("src.client.fed_ledger.requests.post")
    def test_request_validation_assignment(self, mock_post):
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        result = self.client.request_validation_assignment("task-123")

        self.assertEqual(result, mock_response)
        call_args = mock_post.call_args
        self.assertIn("request-validation-assignment/task-123", call_args[0][0])

    @patch("src.client.fed_ledger.requests.post")
    def test_submit_validation_result(self, mock_post):
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        result = self.client.submit_validation_result(
            "assign-123",
            1.5,
            "A100"
        )

        self.assertEqual(result, mock_response)
        
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["data"]["loss"], 1.5)
        self.assertEqual(payload["data"]["gpu_type"], "A100")

    @patch("src.client.fed_ledger.requests.post")
    def test_mark_assignment_as_failed(self, mock_post):
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        result = self.client.mark_assignment_as_failed("assign-123")

        self.assertEqual(result, mock_response)
        
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["status"], "failed")


if __name__ == "__main__":
    unittest.main()
