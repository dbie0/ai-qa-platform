class RunnerError(Exception):
    """Base class for all errors raised by the test runner."""


class ConfigError(RunnerError):
    """Runner is misconfigured (e.g. MONGO_URI is not set)."""


class DefinitionNotFoundError(RunnerError):
    """Requested test definition does not exist in the database."""


class ToolNotRegisteredError(RunnerError):
    """No adapter is registered for the tool."""


class ModelProcessorNotRegisteredError(ToolNotRegisteredError):
    """No model processor is registered under the requested name."""


class MetricNotFoundError(RunnerError):
    """Adapter did not return the metric the test asked for."""
