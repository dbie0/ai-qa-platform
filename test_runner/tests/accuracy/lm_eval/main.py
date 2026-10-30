import json
import logging
import os

import lm_eval
from lm_eval.utils import handle_non_serializable

from tests.test_adapter_interface import BaseTestAdapter

log = logging.getLogger(__name__)
os.environ["OPENVINO_DISABLE_TELEMETRY"] = "1"

class LmEvalAdapter(BaseTestAdapter):
    def run(self, model_path: str, suite: str | None) -> float:
        log.info(f"{suite=}, model_path={model_path}")

        results = lm_eval.simple_evaluate(
            model="openvino",
            model_args=f"pretrained={model_path}",
            tasks=[suite],
            num_fewshot=0,
            limit=20,
            device="cpu"
        )
        with open('/tmp/results.json', "w", encoding="utf-8") as f:
            # Use handle_non_serializable to avoid JSON serialization errors
            # that can sometimes happen with raw eval harness outputs
            json.dump(results, f, default=handle_non_serializable, indent=2)
        return results["results"][suite]["acc,none"]