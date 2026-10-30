from abc import ABC, abstractmethod
from typing import Dict

class BaseTestAdapter(ABC):
    @abstractmethod
    def run(self, model_path: str, suite: str | None) -> Dict[str, float]:
        """
        Executes the tool and returns all gathered metrics.
        Returns: {'latency_ms_per_token': 42.1, 'tokens_per_sec': 23.5, ...}
        """
        pass