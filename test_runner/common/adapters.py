import importlib.metadata
import tomllib
from pathlib import Path

from common.config import ENTRY_POINT_GROUP, MODEL_PROCESSOR_ENTRY_POINT_GROUP
from common.errors import ModelProcessorNotRegisteredError, ToolNotRegisteredError
from common.specs import normalize

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
    """Entry points of `group` declared in pyproject.toml, used when the package isn't pip-installed."""
    if not PYPROJECT.exists():
        return []
    with PYPROJECT.open("rb") as f:
        declared = tomllib.load(f).get("project", {}).get("entry-points", {}).get(group, {})
    return [importlib.metadata.EntryPoint(name, value, group) for name, value in declared.items()]


def _find_entry_point(group: str, name: str) -> importlib.metadata.EntryPoint | None:
    """Installed entry points win; the ones declared in pyproject.toml are the fallback."""
    installed = list(importlib.metadata.entry_points(group=group))
    for ep in [*installed, *_pyproject_entry_points(group)]:
        if normalize(ep.name.rsplit(".", 1)[-1]) == normalize(name):
            return ep
    return None


def find_adapter_class(tool: str):
    """Return the adapter class registered for `tool`, or raise ToolNotRegisteredError."""
    ep = _find_entry_point(ENTRY_POINT_GROUP, tool)
    if ep is None:
        raise ToolNotRegisteredError(f"Tool '{tool}' not registered in pyproject.toml!")
    return ep.load()


def find_model_processor_class(name: str):
    """Return the model processor class registered as `name`, or raise ModelProcessorNotRegisteredError."""
    ep = _find_entry_point(MODEL_PROCESSOR_ENTRY_POINT_GROUP, name)
    if ep is None:
        raise ModelProcessorNotRegisteredError(f"Model processor '{name}' not registered in pyproject.toml!")
    return ep.load()
