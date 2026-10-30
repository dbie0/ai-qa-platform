"""Framework-independent execution of one check (no pytest imports here)."""
import logging
import time
from collections.abc import Mapping
from typing import Any

from common.adapters import find_adapter_class
from common.errors import MetricNotFoundError
from common.evaluator import evaluate_threshold
from common.results import CheckResult, Status
from common.specs import CheckSpec

log = logging.getLogger(__name__)


def extract_metric(spec: CheckSpec, adapter: Any, output: Any) -> tuple[str, float, dict[str, Any]]:
    """Pick the value to compare from adapter output (a single number or a dict of metrics)."""
    if isinstance(output, Mapping):
        metric = spec.suite or getattr(adapter, "default_metric", None) or spec.test_name
        if metric not in output:
            raise MetricNotFoundError(f"Metric '{metric}' was not returned by the adapter!")
        return metric, output[metric], dict(output)
    return spec.test_name, output, {spec.test_name: output}


def run_check(spec: CheckSpec, model_path: str | None = None) -> CheckResult:
    """Run the adapter for `spec` on `model_path` and compare the outcome with the threshold.

    `model_path` is the path produced by the model processor; it defaults to the model name from the spec.

    Never raises: failures to execute are reported as Status.ERROR with the exception attached.
    """
    started = time.perf_counter()
    try:
        adapter = find_adapter_class(spec.tool)()
        output = adapter.run(model_path=model_path or spec.model_name, suite=spec.suite)
        metric, value, metrics = extract_metric(spec, adapter, output)
        passed = evaluate_threshold(value, spec.operator, spec.threshold_value)
    except Exception as exc:
        log.exception("Check %s could not be executed", spec.id)
        return CheckResult(spec, Status.ERROR, error=f"{type(exc).__name__}: {exc}",
                           duration_s=time.perf_counter() - started, exception=exc)

    if passed:
        status = Status.PASSED
    else:
        status = Status.FAILED if spec.is_critical else Status.WARNING
    return CheckResult(spec, status, metric=metric, actual_value=value, metrics=metrics,
                       duration_s=time.perf_counter() - started)
