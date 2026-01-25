"""
Logging Configuration

Structured logging for the drug image pipeline.
"""

import logging
import sys
from typing import Optional
from datetime import datetime


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        format_string: Custom format string
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger for our package
    root_logger = logging.getLogger("drug_image_pipeline")
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"drug_image_pipeline.{name}")


class PipelineLogger:
    """
    Specialized logger for pipeline execution.
    
    Provides structured logging for pipeline stages and metrics.
    """
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.logger = logging.getLogger(f"drug_image_pipeline.pipeline.{request_id[:8]}")
        self._stage_start_times = {}
    
    def stage_start(self, stage_name: str) -> None:
        """Log stage start."""
        self._stage_start_times[stage_name] = datetime.now()
        self.logger.info(f"Stage '{stage_name}' started")
    
    def stage_end(self, stage_name: str, success: bool = True) -> None:
        """Log stage completion."""
        duration = 0
        if stage_name in self._stage_start_times:
            delta = datetime.now() - self._stage_start_times[stage_name]
            duration = delta.total_seconds() * 1000
        
        status = "completed" if success else "failed"
        self.logger.info(f"Stage '{stage_name}' {status} in {duration:.2f}ms")
    
    def stage_error(self, stage_name: str, error: Exception) -> None:
        """Log stage error."""
        self.logger.error(f"Stage '{stage_name}' error: {error}")
    
    def metric(self, name: str, value: float, unit: str = "") -> None:
        """Log a metric."""
        self.logger.info(f"Metric [{name}]: {value}{unit}")
