class CRSError(Exception):
    """Base exception class for CRS system errors."""


class ActionExecutionError(CRSError):
    """Raised when an action fails to execute properly."""


class StateError(CRSError):
    """Raised when the conversation state is invalid."""


class LLMError(CRSError):
    """Raised when an LLM response is invalid."""
