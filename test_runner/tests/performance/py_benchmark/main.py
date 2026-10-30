from tests.test_adapter_interface import BaseTestAdapter
from typing import Dict


class PyBenchAdapter(BaseTestAdapter):
    # Metric compared against the threshold when the test name has no parameter (plain "py_benchmark")
    default_metric = "latency_ms_per_token"

    def run(self, model_path: str, quantization: str, suite: str | None) -> Dict[str, float]:
        print(f"Running full benchmark for {model_path}...")

        # In reality, this runs your LLM inference script once
        return {
            "latency_ms_per_token": 42.1,
            "tokens_per_sec": 23.7,
            "vram_usage_gb": 15.2
        }