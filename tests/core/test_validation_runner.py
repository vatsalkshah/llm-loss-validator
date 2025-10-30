from dataclasses import dataclass

from src.core.validation_runner import AssignmentContext, ValidationRunner
from src.core.validation_result import ValidationResult


@dataclass
class DummyLedger:
    submitted: list
    failed: list

    def submit_validation_result(self, assignment_id, loss, gpu_type):
        self.submitted.append((assignment_id, loss, gpu_type))

    def mark_assignment_as_failed(self, assignment_id):
        self.failed.append(assignment_id)


def test_runner_success():
    ledger = DummyLedger([], [])

    def executor(payload):
        return ValidationResult(success=True, eval_loss=1.0, eval_loss_to_submit=1.0, assignment_id=payload["id"], metadata={"gpu_type": "A100"})

    runner = ValidationRunner(executor, fed_ledger=ledger)
    ctx = AssignmentContext(task_id="42", assignment_id="a1", payload={"id": "a1", "model": "m"})
    result = runner.run(ctx)

    assert result.success is True
    assert ledger.submitted[0][0] == "a1"


def test_runner_failure_marks_assignment():
    ledger = DummyLedger([], [])

    def executor(payload):
        return ValidationResult(success=False, assignment_id=payload["id"], error_message="boom")

    runner = ValidationRunner(executor, fed_ledger=ledger)
    ctx = AssignmentContext(task_id="42", assignment_id="a1", payload={"id": "a1"})
    result = runner.run(ctx)

    assert result.success is False
    assert ledger.failed == ["a1"]
