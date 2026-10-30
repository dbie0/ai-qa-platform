"""Preparation stage: turn a model name into a ready-to-use local model path (no pytest imports here)."""
import logging

from common.adapters import find_model_processor_class
from common.config import DEFAULT_MODEL_PROCESSOR
from common.specs import CheckSpec

log = logging.getLogger(__name__)


class ModelPreparer:
    """Runs the model processor named by the definition's `ml_toolkit` once per (processor, model, quantization)
    and caches the resulting path."""

    def __init__(self, base_path: str, default_processor: str = DEFAULT_MODEL_PROCESSOR):
        self.base_path = base_path
        self.default_processor = default_processor
        self._processors: dict[str, object] = {}
        self._paths: dict[tuple[str, str, str], str] = {}
        self._errors: dict[tuple[str, str, str], Exception] = {}

    def processor_name_for(self, spec: CheckSpec) -> str:
        """The processor is the definition's `ml_toolkit`; the default is used only when the field is missing."""
        return (spec.ml_toolkit or "").strip() or self.default_processor

    def prepare(self, spec: CheckSpec) -> str:
        """Return the model path for `spec`; a failed preparation is not retried for the other checks."""
        name = self.processor_name_for(spec)
        key = (name, spec.model_name, spec.quantization)
        if key in self._errors:
            raise self._errors[key]
        if key not in self._paths:
            try:
                if name not in self._processors:
                    self._processors[name] = find_model_processor_class(name)()
                log.info("Preparing %s (%s) with '%s'", spec.model_name, spec.quantization, name)
                self._paths[key] = self._processors[name].download_and_quantize(
                    model_name=spec.model_name, quantization=spec.quantization, base_path=self.base_path)
            except Exception as exc:
                self._errors[key] = exc
                raise
        return self._paths[key]
