"""
Common utilities for neural network operations.
"""
import logging
import os
import torch
from colorama import init, Fore, Style


def setup_logging(module_name='neural_network'):
    """Set up logging configuration with color support.

    Args:
        module_name (str, optional): Name of the module. Defaults to 'neural_network'

    Returns:
        logging.Logger: Configured logger instance
    """
    init(autoreset=True)  # Initialize colorama

    class ColorFormatter(logging.Formatter):
        """Formatter that adds colors to log levels."""

        COLORS = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.MAGENTA
        }

        def format(self, record):
            levelname = record.levelname
            color = self.COLORS.get(levelname, "")
            record.levelname = f"{color}{levelname}{Style.RESET_ALL}"
            message = super().format(record)
            record.levelname = levelname
            return message

    # Setup log directory and file
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(log_dir, f'{module_name}.log')

    # Configure handlers
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter('%(asctime)s [%(levelname)s] %(message)s'))

    # Configure logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers = []

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(module_name='neural_network'):
    """Get or create a logger instance.

    Args:
        module_name (str, optional): Name of the module. Defaults to 'neural_network'

    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(module_name)
    if not logger.handlers:
        logger = setup_logging(module_name)
    return logger


def get_device():
    """Get the appropriate device for PyTorch operations.

    Returns:
        str: 'cuda' if GPU is available, otherwise 'cpu'
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'
