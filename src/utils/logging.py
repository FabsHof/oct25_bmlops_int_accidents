"""
Centralized logging configuration for the accidents project.

This module provides a consistent logging setup across all modules,
with both console and file output.
"""

import logging
import os
from pathlib import Path
from typing import Optional


# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return the application logger.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: logs/app.log)
        log_format: Custom log format string
        
    Returns:
        Configured logger instance
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # Create logger
    _logger = logging.getLogger('accidents')
    _logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times
    if _logger.handlers:
        return _logger
    
    # Set default format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # Console handler - explicitly use sys.stdout to avoid stderr marking as ERROR
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)
    
    # File handler
    try:
        if log_file is None:
            # Try to get log file path from environment variable
            log_file = os.getenv('APP_LOG_FILE')
            if log_file is None:
                # Check if we're in Airflow environment
                airflow_home = os.getenv('AIRFLOW_HOME', '/opt/airflow')
                if os.path.exists(airflow_home):
                    log_file = os.path.join(airflow_home, 'logs', 'app.log')
                else:
                    # Default to logs/app.log in project root
                    project_root = Path(__file__).parent.parent.parent
                    log_file = os.path.join(project_root, 'logs', 'app.log')
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)
    except (PermissionError, OSError) as e:
        # If we can't create the log file (e.g., in Docker without permissions),
        # just log to console
        _logger.warning(f"Could not create log file: {e}. Logging to console only.")
    
    return _logger


def get_logger() -> logging.Logger:
    """
    Get the application logger instance.
    
    If the logger hasn't been set up yet, it will be initialized
    with default settings.
    
    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


def debug(msg: str, *args, **kwargs) -> None:
    """Log a debug message."""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log an info message."""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log a warning message."""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log an error message."""
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log a critical message."""
    get_logger().critical(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs) -> None:
    """Log an exception with traceback."""
    get_logger().exception(msg, *args, **kwargs)
