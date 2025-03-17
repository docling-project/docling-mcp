import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(log_file='docling_mcp.log'):
    """Setup and return a logger for the entire project."""
    # Create logger
    logger = logging.getLogger('project_name')
    logger.setLevel(logging.INFO)

    # Create a handler and set its level to INFO
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)

    """
    # Prevent adding handlers multiple times
    if not logger.handlers:
        # Create handlers
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10485760, backupCount=5)  # 10MB per file, keep 5 backups
        console_handler = logging.StreamHandler(sys.stdout)

        # Set levels
        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    """
    
    return logger


    
