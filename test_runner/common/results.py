import os
import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from common.config import CI_ENV_VARS
from common.specs import CheckSpec


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Status(StrEnum):
    PASSED = "passed"
    FAILED = "failed"        # critical check did not meet its threshold
    WARNING = "warning"      # non-critical check did not meet its threshold
    ERROR = "error"          # the check could not be executed (adapter crash, missing tool/metric, ...)


@dataclass
class CheckResult:
    spec: CheckSpec
    status: Status
    metric: str | None = None
    actual_value: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_s: float = 0.0
    finished_at: datetime = field(default_factory=utcnow)
    # Original exception, kept so the test can re-raise it with its traceback; never persisted
    exception: BaseException | None = field(default=None, repr=False, compare=False)

    @property
    def message(self) -> str:
        s = self.spec
        return f"[{s.test_name}] Actual: {self.actual_value} (Required {s.operator} {s.threshold_value})"

    def to_document(self, run: "RunContext") -> dict[str, Any]:
        s = self.spec
        return {
            "run_id": run.run_id,
            "definition_name": s.definition_name,
            "model_name": s.model_name,
            "quantization": s.quantization,
            "test_name": s.test_name,
            "tool": s.tool,
            "suite": s.suite,
            "metric": self.metric,
            "operator": s.operator,
            "threshold_value": s.threshold_value,
            "is_critical": s.is_critical,
            "actual_value": self.actual_value,
            "metrics": self.metrics,
            "status": self.status.value,
            "passed": self.status is Status.PASSED,
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
            "started_at": run.started_at,
            "finished_at": self.finished_at,
            "host": run.host,
            "ci": run.ci,
        }


@dataclass
class RunContext:
    """Metadata shared by all results of a single runner invocation."""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: datetime = field(default_factory=utcnow)
    host: str = field(default_factory=platform.node)
    ci: dict[str, str] = field(
        default_factory=lambda: {key: os.environ[var] for key, var in CI_ENV_VARS.items() if var in os.environ}
    )


@dataclass
class ResultRecorder:
    """Collects the results of a run so they can be persisted once at the end."""
    run: RunContext = field(default_factory=RunContext)
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.results.append(result)

    def documents(self) -> list[dict[str, Any]]:
        return [r.to_document(self.run) for r in self.results]
