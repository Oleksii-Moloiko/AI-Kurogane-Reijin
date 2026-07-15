"""User-facing LLM errors."""


class LLMError(RuntimeError):
    """Base provider error."""


class ProviderUnavailableError(LLMError):
    """Provider service cannot be reached."""


class ModelUnavailableError(LLMError):
    """Requested model is not available."""
