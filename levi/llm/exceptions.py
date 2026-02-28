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


class LLMAuthenticationError(LLMError):
    """Authentication or authorization failure with an LLM provider."""

    pass


class LLMRateLimitError(LLMError):
    """Rate-limited request to an LLM provider."""

    pass
