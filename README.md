# AI QA Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-pytest-orange.svg)](https://pytest.org/)
[![Database](https://img.shields.io/badge/storage-MongoDB-green.svg)](https://www.mongodb.com/)
[![CI/CD](https://img.shields.io/badge/pipeline-Jenkins-red.svg)](https://www.jenkins.io/)
[![Hardware Acceleration](https://img.shields.io/badge/acceleration-Intel%20OpenVINO-blueviolet.svg)](https://github.com/openvinotoolkit/openvino)

An automated, extensible Quality Assurance and Validation platform for Large Language Models (LLMs) and ML models. The platform enables database-driven test configuration, automated model preparation and quantization, parallelized accuracy and performance benchmarking against strict SLA thresholds, and continuous integration via Jenkins.

---

## Table of Contents

- [Overview](#overview)
- [Architecture & Workflow](#architecture--workflow)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [1. Infrastructure Setup](#1-infrastructure-setup)
  - [2. Environment & Installation](#2-environment--installation)
- [Configuration & Data Formats](#configuration--data-formats)
  - [Test Definition Document](#test-definition-document)
  - [Environment Variables](#environment-variables)
- [Running Tests](#running-tests)
  - [Command-Line Interface](#command-line-interface)
  - [Filtering Tests by Tool](#filtering-tests-by-tool)
  - [Running Unit Tests](#running-unit-tests)
- [Extending the Platform](#extending-the-platform)
  - [Adding a Test Adapter](#adding-a-test-adapter)
  - [Adding a Model Processor](#adding-a-model-processor)
- [CI/CD Pipeline](#cicd-pipeline)
  - [Dynamic Discovery & Parallel Execution](#dynamic-discovery--parallel-execution)
  - [Agent Hardware Configuration](#agent-hardware-configuration)

---

## Overview

Modern LLM deployment requires verifying both **accuracy** (e.g., benchmark score retention after quantization) and **performance** (e.g., token latency, throughput, memory footprint) across multiple quantization targets (`int4`, `int8`, `fp16`).

**AI QA Platform** solves this by:
1. Fetching model validation specifications dynamically from MongoDB.
2. Automatically downloading and quantizing models (e.g., with Intel OpenVINO / Optimum Intel) on demand with local caching.
3. Dynamically parametrizing and executing benchmarks (`lm-eval`, `py_benchmark`, etc.) through Pytest.
4. Comparing real-world benchmark metrics against configured threshold criteria and tagging failures as critical vs non-critical warnings.
5. Persisting complete test telemetry and run metadata back to MongoDB and generating standard JUnit XML reports for CI.

---

## Architecture & Workflow

```mermaid
flowchart TD
    subgraph Storage ["MongoDB (`validation`)"]
        Definitions[("'test_definitions'<br>Test specs & thresholds")];
        Results[("'test_results'<br>Metrics, runs & hardware info")];
    end

    subgraph CI ["Jenkins CI Pipeline"]
        DiscoveryStage["Stage 1: Tool Discovery<br>(`discover_types.py`)"];
        ParallelStage["Stage 2: Parallel Benchmarks<br>(Dynamic worker fan-out)"];
    end

    subgraph Runner ["Test Runner (`test_runner/`)"]
        Pytest["Pytest Runner (`run_tests.py`)"];
        ModelPrep["Model Preparer (`ModelPreparer`)<br>Download & Quantize (Optimum / OpenVINO)"];
        TestAdapters["Test Adapters<br>`lm-eval` | `py_benchmark`"];
        Evaluator["Threshold Evaluator<br>`>=`, `<=`, `==`, `>`, `<`"];
    end

    Definitions -->|Fetch test plan| DiscoveryStage
    DiscoveryStage -->|"Tool matrix: [lm-eval, py_benchmark]"| ParallelStage
    ParallelStage -->|Execute on agent node| Pytest
    Pytest --> ModelPrep
    ModelPrep --> TestAdapters
    TestAdapters --> Evaluator
    Evaluator -->|Record check results| Results
    Pytest -->|JUnit XML report| ParallelStage
```

---

## Project Structure

```text
ai-qa-platform/
├── docs/
│   ├── Connect_Jenkins_Agent.md   # Setup guide for Linux / Intel OpenVINO Jenkins worker nodes
│   └── testsuite_example.json     # Sample test definition JSON for MongoDB
├── infra/
│   └── docker-compose.yaml        # MongoDB 9.0 and Jenkins 2.587 container services
├── pipelines/
│   └── Jenkinsfile                # Dynamic multi-stage declarative Jenkins pipeline
├── skills/
│   └── system/run-workflow/       # Agent workflow automation skills
└── test_runner/
    ├── common/                    # Core library and runner logic
    │   ├── adapters.py            # Entry point discovery for tools and model processors
    │   ├── config.py              # Environment variables and system defaults
    │   ├── db.py                  # MongoDB connection, query, and persistence handlers
    │   ├── errors.py              # Custom error taxonomy
    │   ├── evaluator.py           # Threshold comparison engine
    │   ├── preparation.py         # Model download/quantization cache manager
    │   ├── results.py             # Result data models and run telemetry collection
    │   ├── runner.py              # Single check execution and metric extraction
    │   └── specs.py               # Test specification dataclasses and parsers
    ├── model_processor/           # Model conversion and quantization adapters
    │   ├── model_processor_adapter_interface.py  # BaseModelProcessorAdapter interface
    │   ├── optimum_ov.py          # Hugging Face Optimum Intel / OpenVINO exporter
    │   └── dummy.py               # Mock processor for fast testing
    ├── tests/                     # Evaluation tool adapters
    │   ├── test_adapter_interface.py             # BaseTestAdapter interface
    │   ├── accuracy/
    │   │   └── lm_eval/           # Language Model Evaluation Harness (EleutherAI lm-eval)
    │   └── performance/
    │       └── py_benchmark/      # Latency, throughput, and memory benchmarking
    ├── unit_tests/
    │   └── test_runner_and_storage.py  # Unit tests covering runner and storage routines
    ├── conftest.py                # Pytest hooks for dynamic test generation & reporting
    ├── discover_types.py          # CLI script to inspect required tools from MongoDB
    ├── pyproject.toml             # Package definition, entry points, and optional extras
    └── run_tests.py               # Pytest entry-point script
```

---

## Key Features

- **Database-Driven Specifications**: Test suites, model variants, quantization targets, thresholds, and criticality are stored in MongoDB, allowing test plans to update without changing runner code.
- **Pluggable Entry Points**: Model processors and test tools register as Python entry points (`[project.entry-points."tests"]` and `[project.entry-points."model_processors"]`), making it trivial to add new frameworks.
- **Smart Model Preparation & Caching**: Models are downloaded and quantized on demand once per `(processor, model, quantization)` tuple per session, preventing redundant conversion overhead.
- **Fine-Grained SLA Evaluation**:
  - Operators: `>=`, `<=`, `==`, `>`, `<`.
  - Distinguishes **Critical Failures** (stops build / marks failure) from **Non-Critical Warnings** (surfaced as Pytest `xfail`).
- **Rich Telemetry**: Saves model name, revision, quantization, exact metrics dict, execution duration, system hostname, and Jenkins CI build details (`JOB_NAME`, `BUILD_NUMBER`, `BUILD_URL`, `NODE_NAME`) to MongoDB.
- **Dynamic Jenkins CI Matrix**: Stage 1 inspects the MongoDB plan and identifies the exact toolchain required; Stage 2 dynamically provisions parallel workers with hardware acceleration (e.g. Intel GPU/CPU) and installs only the necessary dependencies.

---

## Prerequisites

- **Python**: 3.11 or later
- **Docker & Docker Compose**: For local infrastructure (MongoDB, Jenkins)
- **Hardware Acceleration (Optional)**: Intel OpenCL / OpenVINO drivers for quantized model inference

---

## Getting Started

### 1. Infrastructure Setup

Launch the bundled MongoDB and Jenkins services:

```bash
cd infra
export MONGO_USER=admin
export MONGO_PASS=secretpassword
docker compose up -d
```

- **MongoDB**: Available on `localhost:27017`
- **Jenkins**: Available on `http://localhost:8080` (agent communication on `50000`)

To seed the initial test definition, insert a plan into MongoDB (`validation.test_definitions`):

```bash
mongosh "mongodb://admin:secretpassword@localhost:27017/validation?authSource=admin" \
  --eval 'db.test_definitions.insertOne(cat("../docs/testsuite_example.json"))'
```

### 2. Environment & Installation

Navigate to `test_runner/` and set up a virtual environment:

```bash
cd test_runner
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install the test suite package in editable mode. You can install only the adapters needed for your environment:

```bash
# Base installation (pytest, pymongo)
pip install -e .

# With OpenVINO / Optimum Intel processor support
pip install -e '.[optimum_ov]'

# With lm-eval accuracy tool adapter
pip install -e '.[lm-eval]'

# With py-benchmark performance adapter
pip install -e '.[py-benchmark]'

# Or install all extras together:
pip install -e '.[optimum_ov,lm-eval,py-benchmark]'
```

---

## Configuration & Data Formats

### Test Definition Document

Stored in `validation.test_definitions`:

```json
{
  "test_name": "Qwen2.5-1.5B-Instruct_Basic",
  "ml_toolkit": "optimum-ov",
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "configurations": [
    {
      "quantization": "int4",
      "tests": [
        {
          "test_name": "lm-eval.hellaswag",
          "operator": ">=",
          "threshold_value": 68.5,
          "is_critical": true
        },
        {
          "test_name": "py_benchmark",
          "operator": "<=",
          "threshold_value": 45.0,
          "is_critical": false
        }
      ]
    },
    {
      "quantization": "int8",
      "tests": [
        {
          "test_name": "lm-eval.hellaswag",
          "operator": ">=",
          "threshold_value": 68.5,
          "is_critical": true
        }
      ]
    }
  ],
  "revision": 1
}
```

- `test_name`: Name of the test suite document.
- `ml_toolkit`: Identifier of the model processor registered in `pyproject.toml` (e.g. `optimum_ov`, `dummy`).
- `model_name`: Hugging Face model identifier or path.
- `configurations`: List of quantization targets (`int4`, `int8`, `fp16`, etc.) and their test checks.
- `test_name` in test: Format `<tool>.<metric_or_suite>` (e.g. `lm-eval.hellaswag` or `py_benchmark`).
- `operator`: Comparison operator (`>=`, `<=`, `==`, `>`, `<`).
- `threshold_value`: Threshold metric required to pass.
- `is_critical`: If `true`, a failure fails the Pytest session (`pytest.fail`). If `false`, it marks the test as expected failure (`pytest.xfail`).

### Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MONGO_URI` | MongoDB connection URI string (e.g. `mongodb://user:pass@host:27017/validation?authSource=admin`) | *Required* |
| `DB_TEST_NAME` | Name of the test definition in `validation.test_definitions` | *None* |
| `TOOL_FILTER` | Comma-separated list of tools to run (e.g. `lm-eval,py_benchmark`) | *None (runs all)* |
| `MODELS_PATH` | Local directory for storing downloaded and quantized models | `models` |
| `RESULTS_COLLECTION` | Target collection name for saving test outcomes | `test_results` |

---

## Running Tests

### Command-Line Interface

Ensure `MONGO_URI` is set:

```bash
export MONGO_URI="mongodb://admin:secretpassword@localhost:27017/validation?authSource=admin"
```

Run tests using `run_tests.py` or directly via `pytest`:

```bash
# Using run_tests.py
python run_tests.py --db-test-name=Qwen2.5-1.5B-Instruct_Basic -v -s

# Or directly through pytest
pytest run_tests.py --db-test-name=Qwen2.5-1.5B-Instruct_Basic -v -s
```

### Filtering Tests by Tool

To run only specific evaluation tools (e.g. only accuracy evaluation):

```bash
python run_tests.py --db-test-name=Qwen2.5-1.5B-Instruct_Basic --tool-filter=lm-eval -v -s
```

### Disabling Database Writes

To run tests without saving results to MongoDB:

```bash
python run_tests.py --db-test-name=Qwen2.5-1.5B-Instruct_Basic --no-save-results
```

### Running Unit Tests

To run the runner logic and storage unit tests (which use mocks and don't require external services):

```bash
pytest unit_tests/
```

---

## Extending the Platform

### Adding a Test Adapter

1. Subclass `BaseTestAdapter` in `tests/`:

```python
# test_runner/tests/performance/custom_bench/main.py
from tests.test_adapter_interface import BaseTestAdapter
from typing import Dict

class CustomBenchAdapter(BaseTestAdapter):
    default_metric = "latency_ms"

    def run(self, model_path: str, suite: str | None) -> Dict[str, float]:
        # Run benchmark logic against model_path
        return {
            "latency_ms": 38.4,
            "throughput": 45.2
        }
```

2. Register your adapter in `pyproject.toml`:

```toml
[project.entry-points."tests"]
"custom_bench" = "tests.performance.custom_bench.main:CustomBenchAdapter"
```

### Adding a Model Processor

1. Subclass `BaseModelProcessorAdapter` in `model_processor/`:

```python
# test_runner/model_processor/vllm_exporter.py
import os
from model_processor.model_processor_adapter_interface import BaseModelProcessorAdapter

class VllmAdapter(BaseModelProcessorAdapter):
    def download_and_quantize(self, model_name: str, quantization: str, base_path: str) -> str:
        output_path = os.path.join(base_path, f"{model_name.replace('/', '_')}-{quantization}")
        # Execute model quantization and save to output_path
        return output_path
```

2. Register the processor in `pyproject.toml`:

```toml
[project.entry-points."model_processors"]
"vllm" = "model_processor.vllm_exporter:VllmAdapter"
```

---

## CI/CD Pipeline

The platform includes a production-grade Jenkins pipeline (`pipelines/Jenkinsfile`):

```text
               [ Jenkins Pipeline ]
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
     [Discover Tools]       (Polls MongoDB for
    (Runs on 'python')       required test tools)
            │
            ▼
    [Run Benchmarks]        (Installs exact tool extras
   (Parallel Matrix)         and runs pytest per tool)
   ┌────────┴────────┐
   ▼                 ▼
[Bench-lm-eval]  [Bench-py_benchmark]
(Runs on 'intel'  (Runs on 'intel' agent)
 hardware node)
   └────────┬────────┘
            ▼
    [Publish Results]       (Unstashes XMLs & generates
    (JUnit XML Test)         Jenkins trend graphs)
```

### Dynamic Discovery & Parallel Execution

1. **Discover Stage**:
   Runs `discover_types.py` against MongoDB to query the required `ml_toolkit` and `unique_tools` for the configured `DB_TEST_NAME`.
2. **Parallel Benchmarks Stage**:
   Dynamically constructs a parallel execution branch for each unique tool. Each worker node (labeled `intel`) isolates dependencies by installing `.[${toolName}]` and `.[${ML_TOOLKIT}]`.
3. **Post Stage**:
   Gathers all JUnit XML files (`results_${toolName}.xml`) and publishes test trends directly within Jenkins.

### Agent Hardware Configuration

For instructions on configuring Ubuntu/Debian worker nodes with Intel OpenCL runtimes, drivers, and the Jenkins JNLP agent service, see [Connect_Jenkins_Agent.md](docs/Connect_Jenkins_Agent.md).
