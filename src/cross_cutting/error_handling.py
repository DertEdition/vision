"""
Error Handling

Centralized error handling utilities.
"""

from typing import Callable, TypeVar, Optional, Any
from functools import wraps
import logging
import traceback

from ..domain.exceptions import DomainException


logger = logging.getLogger(__name__)

T = TypeVar("T")


def handle_exception(
    default_return: T,
    log_level: int = logging.ERROR,
    reraise: bool = False
) -> Callable:
    """
    Decorator for handling exceptions with consistent logging.
    
    Args:
        default_return: Value to return on exception
        log_level: Logging level for caught exceptions
        reraise: Whether to reraise the exception after logging
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except DomainException as e:
                logger.log(log_level, f"{func.__name__} failed: {e}")
                if e.details:
                    logger.log(log_level, f"Details: {e.details}")
                if reraise:
                    raise
                return default_return
            except Exception as e:
                logger.log(log_level, f"{func.__name__} unexpected error: {e}")
                logger.debug(traceback.format_exc())
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


class ErrorHandler:
    """
    Context manager for error handling.
    
    Usage:
        with ErrorHandler(logger) as handler:
            # do something risky
            if handler.has_error:
                # handle error
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        context: str = "",
        suppress: bool = False
    ):
        """
        Initialize error handler.
        
        Args:
            logger: Logger for error messages
            context: Context string for error messages
            suppress: Whether to suppress exceptions
        """
        self.logger = logger
        self.context = context
        self.suppress = suppress
        self.error: Optional[Exception] = None
        self.error_message: Optional[str] = None
    
    def __enter__(self) -> "ErrorHandler":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val is not None:
            self.error = exc_val
            self.error_message = str(exc_val)
            
            if self.context:
                self.logger.error(f"[{self.context}] {exc_val}")
            else:
                self.logger.error(str(exc_val))
            
            if isinstance(exc_val, DomainException):
                self.logger.debug(f"Details: {exc_val.details}")
            else:
                self.logger.debug(traceback.format_exc())
        
        return self.suppress
    
    @property
    def has_error(self) -> bool:
        """Check if an error occurred."""
        return self.error is not None
    
    @property
    def is_recoverable(self) -> bool:
        """Check if the error is recoverable."""
        if self.error is None:
            return True
        if isinstance(self.error, DomainException):
            return self.error.is_recoverable
        return False


def safe_call(
    func: Callable,
    *args,
    default: Any = None,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Any:
    """
    Safely call a function with error handling.
    
    Args:
        func: Function to call
        *args: Positional arguments
        default: Default value on error
        logger: Optional logger
        **kwargs: Keyword arguments
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.warning(f"safe_call({func.__name__}) failed: {e}")
        return default
