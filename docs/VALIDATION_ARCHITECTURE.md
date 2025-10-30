# Validation Architecture

This document describes the modular architecture of the validation system after the refactoring to support programmatic access by schedulers, inference services, and downstream integrations.

## Module Overview

### Core Service Layer (`src/core/validation_runner.py`)

The `ValidationRunner` class provides a reusable service for model validation with the following key methods:

- **`__init__(hf_token)`**: Initialize the runner with HuggingFace authentication token
- **`load_tokenizer(model_name_or_path)`**: Load and configure a tokenizer from HuggingFace
- **`load_model(model_name_or_path, lora_only, revision, val_args, cached_lora)`**: Load a model, optionally merging LoRA adapters
- **`load_sft_dataset(eval_file, max_seq_length, template_name, tokenizer)`**: Load a supervised fine-tuning dataset
- **`determine_tokenizer_path(model_name_or_path, revision)`**: Determine the correct tokenizer path, handling LoRA models
- **`evaluate_model(...)`**: Execute a complete model evaluation workflow and return structured results

#### Key Features

1. **Stateless Design**: Each evaluation is self-contained with explicit parameters
2. **Resource Management**: Automatic cleanup of models and datasets via context managers
3. **Error Propagation**: Raises structured exceptions for different failure modes
4. **Result Objects**: Returns `ValidationResult` dataclass instead of side effects

### Result Objects (`src/core/validation_result.py`)

The `ValidationResult` dataclass encapsulates evaluation outcomes:

```python
@dataclass
class ValidationResult:
    success: bool
    eval_loss: float
    bpc: float  # Bits Per Character
    bppl: float  # bits Per Character Perplexity
    eval_loss_to_submit: float
    total_bytes: int
    total_target_tokens: int
    token_byte_ratio: float
    vocab_size: int
    model_params_m: float
    assignment_id: Optional[str]
    error_message: Optional[str]
    metadata: Dict[str, Any]
```

#### Helper Methods

- **`is_params_exceeded()`**: Check if result represents a parameter limit failure
- **`is_bpc_valid()`**: Check if BPC metric is valid (not inf)
- **`to_dict()`**: Convert result to a dictionary for serialization

### Structured Exceptions (`src/core/validation_exceptions.py`)

A hierarchy of exceptions allows callers to differentiate failure modes:

- **`ValidationException`**: Base exception for all validation-related errors
- **`ModelParamsExceededException`**: Model parameter count exceeds allowed limit
- **`InvalidModelException`**: Model cannot be loaded or is invalid
- **`InvalidLoraConfigException`**: LoRA adapter configuration is invalid or incompatible
- **`InvalidDatasetException`**: Dataset cannot be loaded or is invalid
- **`TransientValidationException`**: Transient errors that may be retryable

Each exception includes:
- `message`: Human-readable error description
- `assignment_id`: Optional assignment identifier for tracking

### CLI Layer (`src/validate.py`)

The existing CLI commands have been refactored to delegate to the service layer while maintaining backward compatibility:

#### `validate` Command

- Parses arguments and configuration
- Creates `ValidationRunner` instance
- Calls `evaluate_model()` to get `ValidationResult`
- Logs summary table
- Submits results to Fed Ledger API
- Handles structured exceptions appropriately

#### `loop` Command

- Polls for validation assignments
- Delegates to `validate` command for each assignment
- Maintains existing retry and error handling logic

## Usage Examples

### Programmatic Usage

```python
from transformers import HfArgumentParser, TrainingArguments
from core.validation_runner import ValidationRunner
from core.validation_exceptions import ModelParamsExceededException

# Initialize runner
runner = ValidationRunner(hf_token="your_hf_token")

# Parse validation arguments
parser = HfArgumentParser(TrainingArguments)
val_args = parser.parse_json_file(json_file="validation_config.json")[0]

# Run evaluation
try:
    result = runner.evaluate_model(
        model_name_or_path="meta-llama/Llama-3.1-8B",
        base_model="llama3",
        eval_file="./data/eval.jsonl",
        context_length=4096,
        max_params=9000000000,
        val_args=val_args,
        assignment_id="assignment_123",
        lora_only=False,
        revision="main",
    )
    
    if result.success:
        print(f"Evaluation successful: BPC={result.bpc}, Loss={result.eval_loss}")
    else:
        print(f"Evaluation failed: {result.error_message}")
        
except ModelParamsExceededException as e:
    print(f"Model too large: {e.actual_params} > {e.max_params}")
    
except InvalidLoraConfigException as e:
    print(f"Invalid LoRA config: {e.message}")
```

### CLI Usage (Unchanged)

```bash
# Single validation
python src/validate.py validate \
    --model_name_or_path "meta-llama/Llama-3.1-8B" \
    --base_model "llama3" \
    --eval_file "./data/eval.jsonl" \
    --context_length 4096 \
    --max_params 9000000000 \
    --validation_args_file "validation_config.json" \
    --assignment_id "assignment_123"

# Continuous loop
python src/validate.py loop \
    --validation_args_file "validation_config.json" \
    --task_id "1,2,3" \
    --auto_clean_cache True
```

## Testing

Tests are provided in `tests/core/test_validation_runner.py`:

- **Unit tests** for individual methods (tokenizer loading, model loading, etc.)
- **Mocked dependencies** to avoid heavy model downloads in CI
- **Exception handling** tests to verify error propagation
- **Result object** tests to verify structured data

Run tests:

```bash
python -m pytest tests/core/test_validation_runner.py -v
```

## Migration Guide

### For Existing Automation Scripts

No changes required. The CLI interface remains backward compatible.

### For New Integrations

1. Import `ValidationRunner` from `src.core.validation_runner`
2. Initialize with HuggingFace token
3. Call `evaluate_model()` with appropriate parameters
4. Handle `ValidationResult` and structured exceptions
5. Implement custom logic for result submission/logging

### Error Handling Best Practices

```python
from core.validation_exceptions import (
    ModelParamsExceededException,
    InvalidModelException,
    InvalidLoraConfigException,
    TransientValidationException,
)

try:
    result = runner.evaluate_model(...)
except ModelParamsExceededException as e:
    # Submit high loss or reject submission
    handle_params_exceeded(e.actual_params, e.max_params)
except (InvalidModelException, InvalidLoraConfigException) as e:
    # Mark assignment as failed
    mark_failed(e.assignment_id, e.message)
except TransientValidationException as e:
    # Retry with backoff
    retry_with_backoff(e)
except Exception as e:
    # Unknown error - log and escalate
    log_error(e)
    raise
```

## Design Decisions

1. **Service Layer**: Extracted core logic from CLI to enable programmatic access
2. **Dataclasses**: Used for result objects to provide type safety and easy serialization
3. **Structured Exceptions**: Hierarchy allows callers to handle different failure modes appropriately
4. **Resource Cleanup**: Automatic cleanup in `finally` block to prevent memory leaks
5. **Backward Compatibility**: Existing CLI commands work unchanged to avoid breaking automation

## Future Enhancements

1. **Async Support**: Add async variants of evaluation methods for concurrent processing
2. **Streaming Results**: Support streaming partial results for long-running evaluations
3. **Checkpoint Resume**: Support resuming interrupted evaluations from checkpoints
4. **Metrics Callbacks**: Allow custom callbacks during evaluation for real-time monitoring
5. **Batch Evaluation**: Support evaluating multiple models in a single call

## Contributing

When modifying the validation core:

1. Update tests in `tests/core/test_validation_runner.py`
2. Maintain backward compatibility in CLI layer
3. Document new exceptions in this file
4. Update result object schema if adding new metrics
5. Add usage examples for new features
