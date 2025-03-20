import logging


def setup_logger(log_file="docling_mcp.log"):
    """Setup and return a logger for the entire project."""
    # Create logger
    logger = logging.getLogger("project_name")
    logger.setLevel(logging.INFO)

    # Create a handler and set its level to INFO
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger
