"""Central place for names, environment variables and defaults used by the test runner."""
import os

# MongoDB
ENV_MONGO_URI = "MONGO_URI"
DB_NAME = "validation"
DEFINITIONS_COLLECTION = "test_definitions"
RESULTS_COLLECTION = os.environ.get("RESULTS_COLLECTION", "test_results")
MONGO_TIMEOUT_MS = 10_000

# Adapter discovery (see [project.entry-points."tests"] in pyproject.toml)
ENTRY_POINT_GROUP = "tests"

# Model processor discovery (see [project.entry-points."model_processors"] in pyproject.toml)
MODEL_PROCESSOR_ENTRY_POINT_GROUP = "model_processors"
DEFAULT_MODEL_PROCESSOR = "optimum_ov"
DEFAULT_MODELS_PATH = os.environ.get("MODELS_PATH", "models")

# Optional CI metadata attached to every saved result (Jenkins exports these)
CI_ENV_VARS = {
    "job_name": "JOB_NAME",
    "build_number": "BUILD_NUMBER",
    "build_url": "BUILD_URL",
    "node_name": "NODE_NAME",
}
