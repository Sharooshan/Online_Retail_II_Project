
class RetailPipelineError(Exception):
    """Base exception for errors in the retail data pipeline."""
    pass


class RetailDataLoadError(RetailPipelineError):
    """Raised when retail data cannot be loaded."""
    pass
