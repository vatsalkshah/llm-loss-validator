"""High-level helper for executing validation assignments."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .model_cache import ModelCacheManager
from .telemetry import TelemetryClient
from .validation_exceptions import ValidationException
from .validation_result import ValidationResult

logger = logging.getLogger(__name__)

ExecutorFn = Callable[[Dict[str, object]], ValidationResult]


@dataclass(slots=True)
class AssignmentContext:
    """Lightweight wrapper around the FedLedger assignment payload."""

    task_id: str
    assignment_id: str
    payload: Dict[str, object]

    def model_id(self) -> Optional[str]:
        return self.payload.get("model") or self.payload.get("model_name_or_path")


class ValidationRunner:
    """Coordinates evaluation, cache updates, and FedLedger reporting."""

    def __init__(
        self,
        executor: ExecutorFn,
        *,
        cache: Optional[ModelCacheManager] = None,
        telemetry: Optional[TelemetryClient] = None,
        fed_ledger=None,
    ) -> None:
        self._executor = executor
        self._cache = cache
        self._telemetry = telemetry
        self._fed_ledger = fed_ledger

    def run(self, ctx: AssignmentContext) -> ValidationResult:
        logger.info("Starting validation assignment %s", ctx.assignment_id)
        try:
            result = self._executor(ctx.payload)
        except ValidationException as exc:
            logger.warning("Validation error for %s: %s", ctx.assignment_id, exc)
            result = ValidationResult(
                success=False,
                assignment_id=ctx.assignment_id,
                error_message=str(exc),
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Unexpected validation failure")
            result = ValidationResult(
                success=False,
                assignment_id=ctx.assignment_id,
                error_message=str(exc),
            )

        self._post_process(ctx, result)
        return result

    # ------------------------------------------------------------------
    def _post_process(self, ctx: AssignmentContext, result: ValidationResult) -> None:
        model_id = ctx.model_id()
        if result.success:
            logger.info("Validation of %s succeeded (loss=%s)", ctx.assignment_id, result.eval_loss_to_submit)
            if self._fed_ledger is not None:
                self._fed_ledger.submit_validation_result(
                    ctx.assignment_id,
                    result.eval_loss_to_submit,
                    result.metadata.get("gpu_type", "unknown"),
                )
            if self._cache is not None and model_id:
                self._cache.touch(str(model_id))
        else:
            logger.error("Validation of %s failed: %s", ctx.assignment_id, result.error_message)
            if self._fed_ledger is not None:
                self._fed_ledger.mark_assignment_as_failed(ctx.assignment_id)

        if self._telemetry is not None:
            payload = result.to_dict()
            payload.update({
                "task_id": ctx.task_id,
                "model_id": model_id,
            })
            try:
                self._telemetry.emit("validation", payload)
            except Exception:  # pragma: no cover - telemetry best effort
                logger.debug("Telemetry emission failed", exc_info=True)


__all__ = ["ValidationRunner", "AssignmentContext"]
