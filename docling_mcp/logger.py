"""Utility module for logging."""

import logging


def setup_logger() -> logging.Logger:
    """Setup and return a logger for the entire project."""
    # Create logger
    logger = logging.getLogger("docling_mcp")
    logger.setLevel(logging.INFO)

    # Repeated calls must not stack handlers, or every record is emitted
    # once per module that called setup_logger(). Only this module's own
    # handler counts; host-installed handlers do not suppress setup.
    if any(h.name == "docling_mcp" for h in logger.handlers):
        return logger

    # Create a handler and set its level to INFO
    handler = logging.StreamHandler()
    handler.name = "docling_mcp"
    handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger; records are fully handled here, so do
    # not also propagate them to the root logger's handlers
    logger.addHandler(handler)
    logger.propagate = False

    return logger
