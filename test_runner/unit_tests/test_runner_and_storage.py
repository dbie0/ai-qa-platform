from common import db as db_module
from common.errors import MetricNotFoundError, ToolNotRegisteredError
from common.results import ResultRecorder, Status
from common.runner import run_check
from common.specs import CheckSpec, build_specs, parse_test_name, parse_tool_filter

DEFINITION = {
    "test_name": "X", "model_name": "M", "revision": 3,
    "configurations": [{"quantization": "fp16", "tests": [
        {"test_name": "lm-eval.hellaswag", "operator": ">=", "threshold_value": 68.5, "is_critical": True},
        {"test_name": "py_benchmark", "operator": "<=", "threshold_value": 45},
    ]}],
}


def spec(**kw) -> CheckSpec:
    base = dict(definition_name="X", model_name="M", quantization="fp16", test_name="lm-eval.hellaswag",
                tool="lm-eval", suite="hellaswag", operator=">=", threshold_value=68.5, is_critical=True)
    return CheckSpec(**{**base, **kw})


def test_parse_helpers():
    assert parse_test_name("lm-eval.hellaswag") == ("lm-eval", "hellaswag")
    assert parse_test_name("py_benchmark") == ("py_benchmark", None)
    assert parse_tool_filter("lm_eval, PY_benchmark") == {"lm-eval", "py-benchmark"}
    assert parse_tool_filter(None) == set()


def test_build_specs_respects_filter():
    assert len(build_specs(DEFINITION)) == 2
    only = build_specs(DEFINITION, parse_tool_filter("lm_eval"))
    assert [s.test_name for s in only] == ["lm-eval.hellaswag"]
    assert build_specs(DEFINITION, {"nope"}) == []


def test_run_check_statuses(monkeypatch):
    class Adapter:
        default_metric = "lat"
        def __init__(self): pass
        def run(self, **kw): return {"lat": 40.0, "other": 1}

    monkeypatch.setattr("common.runner.find_adapter_class", lambda tool: Adapter)
    ok = run_check(spec(suite=None, tool="b", test_name="b", operator="<=", threshold_value=45))
    assert (ok.status, ok.metric, ok.actual_value) == (Status.PASSED, "lat", 40.0)
    assert ok.metrics == {"lat": 40.0, "other": 1}

    assert run_check(spec(suite=None, operator=">=", threshold_value=45)).status is Status.FAILED
    assert run_check(spec(suite=None, operator=">=", threshold_value=45, is_critical=False)).status is Status.WARNING

    err = run_check(spec(suite="missing"))
    assert err.status is Status.ERROR and isinstance(err.exception, MetricNotFoundError)


def test_run_check_scalar_and_unknown_tool(monkeypatch):
    class Scalar:
        def run(self, **kw): return 71.2

    monkeypatch.setattr("common.runner.find_adapter_class", lambda tool: Scalar)
    assert run_check(spec()).status is Status.PASSED

    def missing(tool): raise ToolNotRegisteredError(tool)
    monkeypatch.setattr("common.runner.find_adapter_class", missing)
    assert run_check(spec()).status is Status.ERROR


class FakeCollection:
    def __init__(self): self.inserted, self.updates = [], []
    def insert_many(self, docs): self.inserted += docs
    def update_one(self, flt, update): self.updates.append((flt, update))


def test_save_results(monkeypatch):
    class Adapter:
        def run(self, **kw): return 10.0
    monkeypatch.setattr("common.runner.find_adapter_class", lambda tool: Adapter)

    recorder = ResultRecorder()
    recorder.add(run_check(spec()))
    db = {"test_results": FakeCollection(), "test_definitions": FakeCollection()}
    assert db_module.save_results(db, recorder, DEFINITION) == 1

    doc = db["test_results"].inserted[0]
    assert doc["run_id"] == recorder.run.run_id and doc["status"] == "failed" and not doc["passed"]
    assert doc["definition_revision"] == 3 and doc["actual_value"] == 10.0
    assert db["test_definitions"].updates[0][0] == {"test_name": "X"}
    assert db_module.save_results(db, ResultRecorder(), DEFINITION) == 0


def test_model_preparer_caches_and_passes_path(monkeypatch):
    from common.preparation import ModelPreparer

    calls = []

    class Processor:
        def download_and_quantize(self, model_name, quantization, base_path):
            calls.append((model_name, quantization, base_path))
            return f"{base_path}/{model_name}-{quantization}"

    requested = []

    def find(name):
        requested.append(name)
        return Processor

    monkeypatch.setattr("common.preparation.find_model_processor_class", find)
    preparer = ModelPreparer("/models")
    assert preparer.prepare(spec()) == "/models/M-fp16"
    assert preparer.prepare(spec(test_name="py_benchmark")) == "/models/M-fp16"
    assert calls == [("M", "fp16", "/models")]
    assert requested == ["optimum_ov"]  # no ml_toolkit -> default processor

    assert preparer.prepare(spec(ml_toolkit="dummy")) == "/models/M-fp16"
    assert requested == ["optimum_ov", "dummy"]  # processor name comes from the definition
    assert len(calls) == 2

    seen = {}

    class Adapter:
        def run(self, model_path, **kw):
            seen["path"] = model_path
            return 70.0

    monkeypatch.setattr("common.runner.find_adapter_class", lambda tool: Adapter)
    assert run_check(spec(), "/models/M-fp16").status is Status.PASSED
    assert seen["path"] == "/models/M-fp16"


def test_unknown_model_processor():
    import pytest
    from common.adapters import find_model_processor_class
    from common.errors import ModelProcessorNotRegisteredError

    with pytest.raises(ModelProcessorNotRegisteredError):
        find_model_processor_class("nope")
    assert find_model_processor_class.__name__
