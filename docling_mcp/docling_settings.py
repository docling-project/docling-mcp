"""This module manages the settings for Docling."""

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.settings import settings

from docling_mcp.logger import setup_logger

# Create a default project logger
logger = setup_logger()


# Configure accelerator settings
def configure_accelerator(doc_batch_size: int = 1) -> bool:
    """Configure the accelerator device for Docling."""
    try:
        # Check if the accelerator_device attribute exists
        if hasattr(settings.perf, "accelerator_device"):
            # Try to use MPS (Metal Performance Shaders) on macOS
            settings.perf.accelerator_device = AcceleratorDevice.MPS
            logger.info(
                f"Configured accelerator device: {settings.perf.accelerator_device}"
            )
        else:
            logger.info(
                "Accelerator device configuration not supported in this version of Docling"
            )

        # Optimize batch processing
        settings.perf.doc_batch_size = doc_batch_size  # Process one document at a time
        logger.info(f"Configured batch size: {settings.perf.doc_batch_size}")

        return True
    except Exception as e:
        logger.warning(f"Failed to configure accelerator: {e}")
        return False
