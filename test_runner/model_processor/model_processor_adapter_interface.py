from abc import ABC, abstractmethod

class BaseModelProcessorAdapter(ABC):
    @abstractmethod
    def download_and_quantize(self, model_name: str, quantization: str, base_path: str) -> str:
        """
        Executes the tool to download and quantize model and returns the path.
        Returns: /home/ubuntu/models/ov_model_int8/
        """
        pass