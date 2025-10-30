import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    from src.core.validation_runner import ValidationRunner, LOSS_FOR_MODEL_PARAMS_EXCEED
    TORCH_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        TORCH_AVAILABLE = False
        ValidationRunner = None  # type: ignore[assignment]
        LOSS_FOR_MODEL_PARAMS_EXCEED = None  # type: ignore[assignment]
    else:
        raise

from src.core.validation_result import ValidationResult
from src.core.validation_exceptions import (
    InvalidLoraConfigException,
    ModelParamsExceededException,
)


class DummyParam:
    def __init__(self, count):
        self._count = count

    def numel(self):
        return self._count


class DummyModel:
    def __init__(self, param_count):
        self.config = SimpleNamespace(to_dict=lambda: {})
        self._param_count = param_count

    def get_memory_footprint(self):
        return 1024

    def parameters(self):
        return [DummyParam(self._param_count)]

    def cpu(self):
        pass


@unittest.skipIf(not TORCH_AVAILABLE, "torch not available")
class TestValidationRunnerEvaluate(unittest.TestCase):
    def setUp(self):
        self.runner = ValidationRunner(hf_token="test-token")
        self.val_args = SimpleNamespace(use_cpu=True, fp16=False)

    @patch("src.core.validation_runner.os.path.exists", return_value=False)
    @patch("src.core.validation_runner.torch.cuda.empty_cache")
    @patch("src.core.validation_runner.SFTDataCollator")
    @patch("src.core.validation_runner.Trainer")
    @patch("src.core.validation_runner.calculate_bpc_bppl_metrics")
    @patch("src.core.validation_runner.calculate_bytes_and_tokens")
    def test_evaluate_model_success(
        self,
        mock_calc_bytes,
        mock_calc_bpc,
        mock_trainer_cls,
        mock_collator,
        mock_empty_cache,
        mock_path_exists,
    ):
        mock_calc_bytes.return_value = (400, 200)
        mock_calc_bpc.return_value = {
            "bpc": 0.5,
            "bppl": 1.41,
            "nll_token_nats_total": 123.0,
            "nll_token_bits_total": 456.0,
        }

        trainer_instance = MagicMock()
        trainer_instance.evaluate.return_value = {"eval_loss": 1.25}
        mock_trainer_cls.return_value = trainer_instance

        tokenizer_mock = MagicMock()
        tokenizer_mock.vocab_size = 32000

        with patch.object(self.runner, "determine_tokenizer_path", return_value=("tokenizer", False)), \
            patch.object(self.runner, "load_tokenizer", return_value=tokenizer_mock), \
            patch.object(self.runner, "load_sft_dataset", return_value=MagicMock()), \
            patch.object(self.runner, "load_model", return_value=DummyModel(param_count=2_000_000)):

            result = self.runner.evaluate_model(
                model_name_or_path="meta/model",
                base_model="llama",
                eval_file="/tmp/eval.jsonl",
                context_length=2048,
                max_params=5_000_000,
                val_args=self.val_args,
                assignment_id="assign-1",
                lora_only=False,
                revision="main",
            )

        self.assertTrue(result.success)
        self.assertEqual(result.eval_loss, 1.25)
        self.assertAlmostEqual(result.eval_loss_to_submit, 0.5)
        self.assertAlmostEqual(result.token_byte_ratio, 0.5)
        self.assertEqual(result.total_bytes, 400)
        self.assertEqual(result.total_target_tokens, 200)
        self.assertEqual(result.vocab_size, 32000)
        self.assertAlmostEqual(result.model_params_m, 2_000_000 / 1e6)
        self.assertEqual(result.metadata["nll_token_nats_total"], 123.0)
        self.assertEqual(result.metadata["nll_token_bits_total"], 456.0)

    @patch("src.core.validation_runner.os.path.exists", return_value=False)
    @patch("src.core.validation_runner.torch.cuda.empty_cache")
    @patch("src.core.validation_runner.calculate_bytes_and_tokens", return_value=(400, 200))
    def test_evaluate_model_param_limit(
        self,
        mock_calc_bytes,
        mock_empty_cache,
        mock_path_exists,
    ):
        tokenizer_mock = MagicMock()
        tokenizer_mock.vocab_size = 32000

        with patch.object(self.runner, "determine_tokenizer_path", return_value=("tokenizer", False)), \
            patch.object(self.runner, "load_tokenizer", return_value=tokenizer_mock), \
            patch.object(self.runner, "load_sft_dataset", return_value=MagicMock()), \
            patch.object(self.runner, "load_model", return_value=DummyModel(param_count=50_000_000)):

            with self.assertRaises(ModelParamsExceededException) as ctx:
                self.runner.evaluate_model(
                    model_name_or_path="meta/model",
                    base_model="llama",
                    eval_file="/tmp/eval.jsonl",
                    context_length=2048,
                    max_params=1_000_000,
                    val_args=self.val_args,
                    assignment_id="assign-2",
                    lora_only=False,
                    revision="main",
                )

        self.assertIn("exceeds the limit", str(ctx.exception))

    @patch("src.core.validation_runner.os.path.exists", return_value=False)
    @patch("src.core.validation_runner.torch.cuda.empty_cache")
    def test_evaluate_model_lora_config_error(
        self,
        mock_empty_cache,
        mock_path_exists,
    ):
        with patch.object(
            self.runner,
            "determine_tokenizer_path",
            side_effect=InvalidLoraConfigException("adapter config missing"),
        ):
            with self.assertRaises(InvalidLoraConfigException):
                self.runner.evaluate_model(
                    model_name_or_path="meta/model",
                    base_model="llama",
                    eval_file="/tmp/eval.jsonl",
                    context_length=2048,
                    max_params=5_000_000,
                    val_args=self.val_args,
                    assignment_id="assign-3",
                    lora_only=False,
                    revision="main",
                )


class TestValidationResultHelpers(unittest.TestCase):
    def test_validation_result_to_dict(self):
        result = ValidationResult(
            success=True,
            eval_loss=1.23,
            bpc=0.5,
            bppl=1.4,
            eval_loss_to_submit=0.5,
            total_bytes=400,
            total_target_tokens=200,
            token_byte_ratio=0.5,
            vocab_size=32000,
            model_params_m=2.0,
            assignment_id="assign-1",
            metadata={"foo": "bar"},
        )

        result_dict = result.to_dict()
        self.assertEqual(result_dict["success"], True)
        self.assertEqual(result_dict["eval_loss"], 1.23)
        self.assertEqual(result_dict["bpc"], 0.5)
        self.assertEqual(result_dict["metadata"], {"foo": "bar"})

    def test_validation_result_is_bpc_valid(self):
        valid = ValidationResult(success=True, bpc=0.5)
        invalid = ValidationResult(success=True, bpc=float("inf"))

        self.assertTrue(valid.is_bpc_valid())
        self.assertFalse(invalid.is_bpc_valid())

    def test_validation_result_is_params_exceeded(self):
        exceeded = ValidationResult(success=False, error_message="Params exceed limit")
        other = ValidationResult(success=False, error_message="Other error")

        self.assertTrue(exceeded.is_params_exceeded())
        self.assertFalse(other.is_params_exceeded())


if __name__ == "__main__":
    unittest.main()
