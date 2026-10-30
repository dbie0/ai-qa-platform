import sys

import pytest

from common.results import CheckResult, Status
from common.runner import run_check
from common.specs import CheckSpec


def test_model_against_criteria(test_spec: CheckSpec, model_path: str, result_recorder):
    result = run_check(test_spec, model_path)
    result_recorder.add(result)
    _report(result)


def _report(result: CheckResult) -> None:
    """Translate a CheckResult into a pytest outcome."""
    match result.status:
        case Status.PASSED:
            return
        case Status.ERROR:
            raise result.exception
        case Status.FAILED:
            pytest.fail("CRITICAL FAILURE: " + result.message)
        case Status.WARNING:
            pytest.xfail("Non-critical metric failed: " + result.message)


def main(argv=None) -> int:
    """Convenience wrapper: `python run_tests.py --db-test-name X --tool-filter=Y [pytest args]`.

    Equivalent to `pytest run_tests.py --db-test-name X --tool-filter=Y [pytest args]`;
    the model processor named by the `ml_toolkit` field of the DB definition (default: optimum_ov when absent)
    prepares the model path the tests run against; the options are defined in conftest.py.
    """
    args = sys.argv[1:] if argv is None else argv
    return int(pytest.main([__file__, *args]))


if __name__ == "__main__":
    sys.exit(main())
