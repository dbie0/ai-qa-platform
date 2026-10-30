from dataclasses import dataclass
from typing import Any, Mapping


def normalize(name: str) -> str:
    """Tool names are compared case-insensitively and '_' is treated like '-'."""
    return name.strip().lower().replace("_", "-")


def parse_test_name(full_name: str) -> tuple[str, str | None]:
    """'lm-eval.hellaswag' -> ('lm-eval', 'hellaswag'); 'lm-eval' -> ('lm-eval', None)"""
    tool, _, suite = full_name.partition(".")
    return tool, (suite or None)


def parse_tool_filter(raw: str | None) -> set[str]:
    """'lm-eval, py_benchmark' -> {'lm-eval', 'py-benchmark'}; empty means 'no filter'."""
    if not raw:
        return set()
    return {normalize(t) for t in raw.split(",") if t.strip()}


@dataclass(frozen=True)
class CheckSpec:
    """One test entry of a definition document, flattened together with its configuration."""
    definition_name: str
    model_name: str
    quantization: str
    test_name: str
    tool: str
    suite: str | None
    operator: str
    threshold_value: float
    is_critical: bool = False
    ml_toolkit: str | None = None

    @property
    def id(self) -> str:
        return f"{self.quantization}-{self.test_name}"


def build_specs(definition: Mapping[str, Any], tool_filter: set[str] | None = None) -> list[CheckSpec]:
    """Flatten a DB document into one spec per test entry, keeping only tools allowed by the filter."""
    specs = []
    for config in definition.get("configurations", []):
        for test in config.get("tests", []):
            tool, suite = parse_test_name(test["test_name"])
            if tool_filter and normalize(tool) not in tool_filter:
                continue
            specs.append(CheckSpec(
                definition_name=definition["test_name"],
                model_name=definition["model_name"],
                quantization=config["quantization"],
                test_name=test["test_name"],
                tool=tool,
                suite=suite,
                operator=test["operator"],
                threshold_value=test["threshold_value"],
                is_critical=test.get("is_critical", False),
                ml_toolkit=definition.get("ml_toolkit"),
            ))
    return specs
