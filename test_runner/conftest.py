import logging
import os
from dataclasses import dataclass

import pytest

from common.config import DEFAULT_MODELS_PATH
from common.db import fetch_definition, open_database, save_results
from common.errors import RunnerError
from common.preparation import ModelPreparer
from common.results import ResultRecorder
from common.specs import CheckSpec, build_specs, parse_tool_filter

log = logging.getLogger(__name__)


@dataclass
class _Loaded:
    definition: dict
    specs: list[CheckSpec]


_loaded_key = pytest.StashKey[_Loaded]()
_recorder_key = pytest.StashKey[ResultRecorder]()


def pytest_addoption(parser):
    group = parser.getgroup("test-definition")
    group.addoption(
        "--db-test-name",
        default=os.environ.get("DB_TEST_NAME"),
        help="test_name of the definition document in validation.test_definitions "
             "(env: TEST_NAME), e.g. Qwen2.5-1.5B-Instruct_Basic",
    )
    group.addoption(
        "--tool-filter",
        default=os.environ.get("TOOL_FILTER"),
        help="Only run tests of these tools (comma separated, env: TOOL_FILTER), e.g. lm-eval",
    )
    group.addoption(
        "--models-path",
        default=DEFAULT_MODELS_PATH,
        help="Directory where the model processor stores prepared models (env: MODELS_PATH)",
    )
    group.addoption(
        "--no-save-results",
        action="store_true",
        default=False,
        help="Do not store the results in the database (by default they go to validation.test_results)",
    )


def pytest_configure(config):
    config.stash[_recorder_key] = ResultRecorder()


def _load(config) -> _Loaded:
    """Fetch the definition once per session and build the (filtered) list of checks."""
    if _loaded_key not in config.stash:
        test_name = config.getoption("--db-test-name")
        if not test_name:
            raise pytest.UsageError("--db-test-name (or TEST_NAME env var) is required")
        try:
            with open_database() as db:
                definition = fetch_definition(db, test_name)
        except RunnerError as exc:
            raise pytest.UsageError(str(exc)) from None
        tool_filter = parse_tool_filter(config.getoption("--tool-filter"))
        config.stash[_loaded_key] = _Loaded(definition, build_specs(definition, tool_filter))
    return config.stash[_loaded_key]


def pytest_generate_tests(metafunc):
    if "test_spec" not in metafunc.fixturenames:
        return
    specs = _load(metafunc.config).specs
    # An empty list makes pytest report the test as skipped, so the junit xml is still produced
    metafunc.parametrize("test_spec", specs, ids=[s.id for s in specs])


@pytest.fixture(scope="session")
def result_recorder(request) -> ResultRecorder:
    return request.config.stash[_recorder_key]


@pytest.fixture(scope="session")
def model_preparer(request) -> ModelPreparer:
    config = request.config
    return ModelPreparer(config.getoption("--models-path"))


@pytest.fixture
def model_path(test_spec: CheckSpec, model_preparer: ModelPreparer) -> str:
    """Preparation stage: path of the model produced by the processor named in the definition's `ml_toolkit`."""
    return model_preparer.prepare(test_spec)


def pytest_sessionfinish(session):
    """Persist the collected results. A storage problem is reported but never changes the test outcome."""
    config = session.config
    recorder = config.stash.get(_recorder_key, None)
    loaded = config.stash.get(_loaded_key, None)
    if recorder is None or not recorder.results or config.getoption("--no-save-results"):
        return

    reporter = config.pluginmanager.get_plugin("terminalreporter")
    try:
        with open_database() as db:
            count = save_results(db, recorder, loaded.definition if loaded else None)
        message = f"Saved {count} result(s) to the database (run_id={recorder.run.run_id})"
    except Exception as exc:
        log.exception("Could not save results")
        message = f"WARNING: could not save results to the database: {exc}"
    if reporter:
        reporter.write_line(message)
    else:
        print(message)
