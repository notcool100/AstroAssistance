"""
Logging configuration for AstroAssistance.
"""
import os
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime

from src.core.config import config_manager


def setup_logger():
    """Configure the logger for the application."""
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"astro_assistance_{timestamp}.log")
    
    # Configure logger
    logger.remove()  # Remove default handler
    
    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    
    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="1 month",
    )
    
    return logger


# Create a singleton logger instance
app_logger = setup_logger()