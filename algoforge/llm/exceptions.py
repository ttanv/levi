"""Custom exceptions for LLM operations."""


class LLMError(Exception):
    """Base exception for LLM errors."""

    pass


class LLMConnectionError(LLMError):
    """Error connecting to LLM provider."""

    pass


class LLMTimeoutError(LLMError):
    """LLM request timed out."""

    pass


class LLMResponseError(LLMError):
    """Invalid or unexpected response from LLM."""

    pass


class LLMConfigurationError(LLMError):
    """Invalid local/global LLM configuration."""

    pass


class LLMAuthenticationError(LLMError):
    """Authentication or authorization failure with an LLM provider."""

    pass


class LLMRateLimitError(LLMError):
    """Rate-limited request to an LLM provider."""

    pass


class LLMRetryExhaustedError(LLMError):
    """All retry attempts exhausted."""

    def __init__(self, message: str, last_error: Exception):
        super().__init__(message)
        self.last_error = last_error
