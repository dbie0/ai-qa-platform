import os

from model_processor.model_processor_adapter_interface import BaseModelProcessorAdapter


class DummyAdapter(BaseModelProcessorAdapter):
    def download_and_quantize(self, model_name: str, quantization: str, base_path: str) -> str:
        output_path = os.path.join(base_path, f"{model_name.split('/')[-1]}-{quantization}")
        return output_path