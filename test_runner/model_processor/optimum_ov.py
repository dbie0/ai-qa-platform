import logging

from model_processor.model_processor_adapter_interface import BaseModelProcessorAdapter
import os
import sys
import subprocess

log = logging.getLogger(__name__)


class OptimumAdapter(BaseModelProcessorAdapter):
    @staticmethod
    def _get_dir_size(path: str) -> int:
        """Return total size in bytes of all files under the given path."""
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for root, _, files in os.walk(path):
            for name in files:
                file_path = os.path.join(root, name)
                if not os.path.islink(file_path):
                    total += os.path.getsize(file_path)
        return total

    def download_and_quantize(self, model_name: str, quantization: str, base_path: str) -> str:
        log.info(f"Starting eventual download and quantization process of {model_name} to {quantization}.")

        output_path = os.path.join(base_path, f"{model_name.split('/')[-1]}-{quantization}")

        venv_bin_dir = os.path.dirname(sys.executable)
        optimum_cli_path = os.path.join(venv_bin_dir, "optimum-cli")

        command = [
            optimum_cli_path,  # Use the absolute path to the venv binary!
            "export", "openvino",
            "--model", model_name,
            output_path,
            "--weight-format", quantization
        ]

        log.debug(f"Running command: {' '.join(command)}")

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print("Export completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Standard Error:\n{e.stderr}")
            raise RuntimeError(f"Failed to quantize model {model_name}.") from e

        model_size = self._get_dir_size(output_path)
        if model_size == 0:
            raise RuntimeError(f"Quantized model {model_name} is empty, output path: {output_path}")
        log.info(f"Quantized model size: {model_size / (1024 ** 3):.2f} GiB ({model_size} bytes). Located in {output_path}")
        return output_path
